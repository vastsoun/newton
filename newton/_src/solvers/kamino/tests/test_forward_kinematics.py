###########################################################################
# KAMINO: UNIT TESTS: KINEMATICS: FORWARD KINEMATICS
###########################################################################

import hashlib
import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.joints import JointActuationType, JointDoFType
from newton._src.solvers.kamino.core.types import mat33f, vec3f
from newton._src.solvers.kamino.kinematics.forward_kinematics import ForwardKinematicsSolver
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.tests.test_utils_diff_check import diff_check, run_test_single_joint_examples
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

    def tearDown(self):
        self.default_device = None

    def test_Jacobian_check(self):
        # Initialize RNG
        test_name = "Forward Kinematics Jacobian check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        def test_function(model):
            assert model.size.num_worlds == 1  # For simplicity we assume a single world

            # Generate (random) state
            state = model.data(device=self.default_device)
            num_bodies_q_i = state.bodies.q_i.shape[0] * 7
            random_state = rng.uniform(-1.0, 1.0, num_bodies_q_i).astype("float32")
            state.bodies.q_i.assign(random_state)

            # Generate (random) controls
            num_joints_q_j = state.joints.q_j.shape[0]
            state.joints.q_j.assign(rng.uniform(-1.0, 1.0, num_joints_q_j).astype("float32"))

            # Evaluate analytic Jacobian
            solver = ForwardKinematicsSolver(model)
            pos_control_transforms = solver.eval_position_control_transformations(state)
            jacobian = solver.eval_kinematic_constraints_jacobian(state, pos_control_transforms)

            # Check against finite differences Jacobian
            rb_state_init_np = state.bodies.q_i.numpy().flatten()  # Save current state of bodies

            def eval_constraints(rb_state_np):
                state.bodies.q_i.assign(rb_state_np)
                constraints = solver.eval_kinematic_constraints(state, pos_control_transforms)
                state.bodies.q_i.assign(rb_state_init_np)  # Reset state
                return constraints.numpy()[0]

            return diff_check(
                eval_constraints,
                jacobian.numpy()[0],
                rb_state_init_np,
                epsilon=1e-4,
                tolerance_abs=5e-3,
                tolerance_rel=5e-3,
            )

        success = run_test_single_joint_examples(test_function, test_name, device=self.default_device)
        self.assertTrue(success)


class WalkerRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_walker_FK_random_poses(self):
        # Initialize RNG
        test_name = "Walker FK random poses check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load model and add fixed joint on pelvis
        model_path = os.path.join(get_examples_usd_assets_path(), "walker/walker_floating_with_boxes.usda")
        builder = USDImporter().import_from(model_path)
        builder.add_joint(
            JointActuationType.PASSIVE,
            JointDoFType.FIXED,
            -1,
            0,
            builder.bodies[0].q_i_0[:3],
            vec3f(0.0, 0.0, 0.0),
            mat33f(np.identity(3)),
            name="Fixed pelvis",
        )
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Generate (random) controls
        num_poses = 50
        num_controls = model.info.num_actuated_joint_dofs.numpy()[0]
        theta_max = np.radians(10.0)  # Angles too far from the initial pose lead to singularities
        p_control_random = rng.uniform(-theta_max, theta_max, (num_poses, num_controls))

        # Convert to generalized positions (adding zeros for the passive joint dofs)
        num_gen_pos = model.info.num_joint_dofs.numpy()[0]
        gen_pos_random = np.zeros((num_poses, num_gen_pos))
        actuated_dofs_offet = model.joints.actuated_dofs_offset.numpy()
        for id_gen_pos in range(num_gen_pos):
            id_control = actuated_dofs_offet[id_gen_pos]
            if id_control >= 0:
                gen_pos_random[:, id_gen_pos] = p_control_random[:, id_control]

        # Run forward kinematics on all random poses
        model_data = model.data(device=self.default_device)
        solver = ForwardKinematicsSolver(model)
        success_flags = []
        for i in range(num_poses):
            model_data.joints.q_j.assign(gen_pos_random[i, :])
            solver.solve_fk(model_data, reset_state=True)
            success_flags.append(solver.newton_success.numpy()[0])

        success = np.sum(success_flags) == num_poses
        if not success:
            print(f"{test_name} failed, {np.sum(success_flags)}/{num_poses} poses successful")
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
    wp.config.verbose = True
    # wp.clear_kernel_cache()
    # wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
