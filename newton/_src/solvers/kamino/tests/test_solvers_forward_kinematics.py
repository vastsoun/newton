# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KAMINO: UNIT TESTS: SOLVERS: FORWARD KINEMATICS
"""

import hashlib
import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.joints import JointActuationType, JointDoFType
from newton._src.solvers.kamino.core.model import Model
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.solvers.fk import ForwardKinematicsSolver, ForwardKinematicsSolverSettings
from newton._src.solvers.kamino.tests.utils.diff_check import diff_check, run_test_single_joint_examples
from newton._src.solvers.kamino.utils.io.usd import USDImporter

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Tests
###


class JacobianCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.has_cuda = self.default_device.is_cuda

    def tearDown(self):
        self.default_device = None

    def test_Jacobian_check(self):
        # Initialize RNG
        test_name = "Forward Kinematics Jacobian check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        def test_function(model: Model):
            assert model.size.num_worlds == 1  # For simplicity we assume a single world

            # Generate (random) body poses
            bodies_q_np = rng.uniform(-1.0, 1.0, 7 * model.size.sum_of_num_bodies).astype("float32")
            bodies_q = wp.from_numpy(bodies_q_np, dtype=wp.transformf, device=model.device)

            # Generate (random) base pose
            base_q_np = rng.uniform(-1.0, 1.0, 7).astype("float32")
            base_q = wp.from_numpy(base_q_np, dtype=wp.transformf, device=model.device)

            # Generate (random) actuated coordinates
            actuators_q_np = rng.uniform(-1.0, 1.0, model.size.sum_of_num_actuated_joint_coords).astype("float32")
            actuators_q = wp.from_numpy(actuators_q_np, dtype=wp.float32, device=model.device)

            # Evaluate analytic Jacobian
            solver = ForwardKinematicsSolver(model=model)
            pos_control_transforms = solver.eval_position_control_transformations(base_q, actuators_q)
            jacobian = solver.eval_kinematic_constraints_jacobian(bodies_q, pos_control_transforms)

            # Check against finite differences Jacobian
            def eval_constraints(bodies_q_stepped_np):
                bodies_q.assign(bodies_q_stepped_np)
                constraints = solver.eval_kinematic_constraints(bodies_q, pos_control_transforms)
                bodies_q.assign(bodies_q_np)  # Reset state
                return constraints.numpy()[0]

            return diff_check(
                eval_constraints,
                jacobian.numpy()[0],
                bodies_q_np,
                epsilon=1e-4,
                tolerance_abs=5e-3,
                tolerance_rel=5e-3,
            )

        success = run_test_single_joint_examples(test_function, test_name, device=self.default_device)
        self.assertTrue(success)


def get_actuators_q_quaternion_first_ids(model: Model):
    """Lists the first index of every unit quaternion 4-segment in the model's actuated coordinates."""
    act_types = model.joints.act_type.numpy()
    dof_types = model.joints.dof_type.numpy()
    num_coords = model.joints.num_coords.numpy()
    coord_id = 0
    quat_ids = []
    for jt_id in range(model.size.sum_of_num_joints):
        if act_types[jt_id] == JointActuationType.PASSIVE:
            continue
        if dof_types[jt_id] == JointDoFType.SPHERICAL:
            quat_ids.append(coord_id)
        elif dof_types[jt_id] == JointDoFType.FREE:
            quat_ids.append(coord_id + 3)
        coord_id += num_coords[jt_id]
    return quat_ids


def simulate_random_poses(
    model: Model,
    num_poses: int,
    min_base_q: np.ndarray,
    max_base_q: np.ndarray,
    min_actuators_q: np.ndarray,
    max_actuators_q: np.ndarray,
    rng: np.random._generator.Generator,
    use_graph: bool = False,
    verbose: bool = False,
):
    # Check dimensions
    base_q_size = 7 * model.size.num_worlds
    actuators_q_size = model.size.sum_of_num_actuated_joint_dofs
    assert len(min_base_q) == base_q_size
    assert len(max_base_q) == base_q_size
    assert len(min_actuators_q) == actuators_q_size
    assert len(max_actuators_q) == actuators_q_size

    # Generate (random) base_q, actuators_q
    base_q_np = np.zeros((num_poses, base_q_size))
    for i in range(base_q_size):
        base_q_np[:, i] = rng.uniform(min_base_q[i], max_base_q[i], num_poses)
    actuators_q_np = np.zeros((num_poses, actuators_q_size))
    for i in range(actuators_q_size):
        actuators_q_np[:, i] = rng.uniform(min_actuators_q[i], max_actuators_q[i], num_poses)

    # Normalize quaternions in base_q, actuators_q
    for i in range(model.size.num_worlds):
        base_q_np[:, 7 * i + 3 : 7 * i + 7] /= np.linalg.norm(base_q_np[:, 7 * i + 3 : 7 * i + 7], axis=1)[:, None]
    quat_ids = get_actuators_q_quaternion_first_ids(model)
    for i in quat_ids:
        actuators_q_np[:, i : i + 4] /= np.linalg.norm(actuators_q_np[:, i : i + 4], axis=1)[:, None]

    # Run forward kinematics on all random poses
    settings = ForwardKinematicsSolverSettings()
    settings.reset_state = True
    solver = ForwardKinematicsSolver(model, settings)
    success_flags = []
    bodies_q = wp.array(shape=(model.size.sum_of_num_bodies), dtype=wp.transformf, device=model.device)
    base_q = wp.array(shape=(model.size.num_worlds), dtype=wp.transformf, device=model.device)
    actuators_q = wp.array(shape=(actuators_q_size), dtype=wp.float32, device=model.device)
    for i in range(num_poses):
        base_q.assign(base_q_np[i])
        actuators_q.assign(actuators_q_np[i])
        status = solver.solve_fk(
            base_q, actuators_q, bodies_q, use_graph=use_graph, verbose=verbose, return_status=True
        )
        success_flags.append(status.success.min() == 1)

    success = np.sum(success_flags) == num_poses
    if not success:
        print(f"Random poses simulation failed, {np.sum(success_flags)}/{num_poses} poses successful")
    return success


class DRTestMechanismRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.has_cuda = self.default_device.is_cuda
        self.verbose = False

    def tearDown(self):
        self.default_device = None

    def test_mechanism_FK_random_poses(self):
        # Initialize RNG
        test_name = "Test mechanism FK random poses check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load model
        examples_path = get_examples_usd_assets_path()
        if not examples_path:
            self.skipTest("Examples USD assets path not found. Skipping test.")
        model_path = os.path.join(examples_path, "dr_testmech/usd/dr_testmech.usda")
        builder = USDImporter().import_from(model_path)
        builder.set_base_joint(joint_name="base")
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max = np.radians(100.0)
        base_q_min = np.array(3 * [-0.2] + 4 * [-1.0])
        base_q_max = np.array(3 * [0.2] + 4 * [1.0])
        success = simulate_random_poses(
            model,
            num_poses,
            base_q_min,
            base_q_max,
            np.array([-theta_max]),
            np.array([theta_max]),
            rng,
            self.has_cuda,
            self.verbose,
        )
        self.assertTrue(success)


class DRLegsRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.has_cuda = self.default_device.is_cuda
        self.verbose = False

    def tearDown(self):
        self.default_device = None

    def test_dr_legs_FK_random_poses(self):
        # Initialize RNG
        test_name = "FK random poses check for dr_legs model"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load model and set base body to pelvis
        examples_path = get_examples_usd_assets_path()
        if not examples_path:
            self.skipTest("Examples USD assets path not found. Skipping test.")
        model_path = os.path.join(examples_path, "dr_legs/usd/dr_legs_with_boxes.usda")
        builder = USDImporter().import_from(model_path)
        builder.set_base_body(body_name="pelvis")
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max = np.radians(10.0)  # Angles too far from the initial pose lead to singularities
        base_q_min = np.array(3 * [-0.2] + 4 * [-1.0])
        base_q_max = np.array(3 * [0.2] + 4 * [1.0])
        num_actuator_coords = model.size.sum_of_num_actuated_joint_coords
        success = simulate_random_poses(
            model,
            num_poses,
            base_q_min,
            base_q_max,
            np.array(num_actuator_coords * [-theta_max]),
            np.array(num_actuator_coords * [theta_max]),
            rng,
            self.has_cuda,
            self.verbose,
        )
        self.assertTrue(success)


class HeterogenousModelRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.has_cuda = self.default_device.is_cuda
        self.verbose = False

    def tearDown(self):
        self.default_device = None

    def test_heterogenous_model_FK_random_poses(self):
        # Initialize RNG
        test_name = "Heterogenous model (test mechanism + dr_legs) FK random poses check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load models
        examples_path = get_examples_usd_assets_path()
        if not examples_path:
            self.skipTest("Examples USD assets path not found. Skipping test.")
        model_path = os.path.join(examples_path, "dr_testmech/usd/dr_testmech.usda")
        builder = USDImporter().import_from(model_path)
        builder.set_base_joint(joint_name="base")
        model_path1 = os.path.join(examples_path, "dr_legs/usd/dr_legs_with_boxes.usda")
        builder1 = USDImporter().import_from(model_path1)
        builder1.set_base_body(body_name="pelvis")
        builder.add_builder(builder1)
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max_test_mech = np.radians(100.0)
        theta_max_dr_legs = np.radians(10.0)
        base_q_min = np.array(3 * [-0.2] + 4 * [-1.0] + 3 * [-0.2] + 4 * [-1.0])
        base_q_max = np.array(3 * [0.2] + 4 * [1.0] + 3 * [0.2] + 4 * [1.0])
        max_controls = np.array([theta_max_test_mech] + builder1.num_actuated_joint_coords * [theta_max_dr_legs])
        success = simulate_random_poses(
            model, num_poses, base_q_min, base_q_max, -max_controls, max_controls, rng, self.has_cuda, self.verbose
        )
        self.assertTrue(success)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(
        linewidth=999999, edgeitems=999999, threshold=999999, precision=10, suppress=True
    )  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
