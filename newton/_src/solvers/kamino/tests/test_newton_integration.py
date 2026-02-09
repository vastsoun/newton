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

"""TODO"""

import copy
import math
import os
import unittest

import warp as wp

import newton
import newton._src.solvers.kamino.tests.utils.checks as test_util_checks
from newton._src.core import Axis
from newton._src.sim import (
    ActuatorMode,
    Control,
    Model,
    ModelBuilder,
    State,
)
from newton._src.solvers.kamino.core import inertia
from newton._src.solvers.kamino.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino.core.control import ControlKamino
from newton._src.solvers.kamino.core.joints import JOINT_QMAX, JOINT_QMIN
from newton._src.solvers.kamino.core.model import ModelKamino
from newton._src.solvers.kamino.core.state import StateKamino
from newton._src.solvers.kamino.models import basics, get_basics_usd_assets_path, get_examples_usd_assets_path
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.io.usd import USDImporter

###
# Utilities
###


def build_boxes_fourbar_newton(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    fixedbase: bool = False,
    floatingbase: bool = True,
    limits: bool = True,
    ground: bool = True,
    verbose: bool = False,
    new_world: bool = True,
    actuator_ids: list[int] | None = None,
) -> ModelBuilder:
    """
    Constructs a basic model of a four-bar linkage.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to create a new world in the builder for this model.\n
            If `True`, a new world is created and added to the builder.

    Returns:
        ModelBuilder: A model builder containing the four-bar linkage.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world()

    # Set default actuator IDs if none are provided
    if actuator_ids is None:
        actuator_ids = [1, 3]
    elif not isinstance(actuator_ids, list):
        raise TypeError("actuator_ids, if specified, must be provided as a list of integers.")

    ###
    # Base Parameters
    ###

    # Constant to set an initial z offset for the bodies
    # NOTE: for testing purposes, recommend values are {0.0, -0.001}
    z_0 = z_offset

    # Box dimensions
    d = 0.01
    w = 0.01
    h = 0.1

    # Margins
    mj = 0.001
    dj = 0.5 * d + mj

    ###
    # Body parameters
    ###

    # Box dimensions
    d_1 = h
    w_1 = w
    h_1 = d
    d_2 = d
    w_2 = w
    h_2 = h
    d_3 = h
    w_3 = w
    h_3 = d
    d_4 = d
    w_4 = w
    h_4 = h

    # Inertial properties
    m_i = 1.0
    i_I_i_1 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_1, w_1, h_1)
    i_I_i_2 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_2, w_2, h_2)
    i_I_i_3 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_3, w_3, h_3)
    i_I_i_4 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_4, w_4, h_4)
    if verbose:
        print(f"i_I_i_1:\n{i_I_i_1}")
        print(f"i_I_i_2:\n{i_I_i_2}")
        print(f"i_I_i_3:\n{i_I_i_3}")
        print(f"i_I_i_4:\n{i_I_i_4}")

    # Initial body positions
    r_0 = wp.vec3f(0.0, 0.0, z_0)
    dr_b1 = wp.vec3f(0.0, 0.0, 0.5 * d)
    dr_b2 = wp.vec3f(0.5 * h + dj, 0.0, 0.5 * h + dj)
    dr_b3 = wp.vec3f(0.0, 0.0, 0.5 * d + h + dj + mj)
    dr_b4 = wp.vec3f(-0.5 * h - dj, 0.0, 0.5 * h + dj)

    # Initial positions of the bodies
    r_b1 = r_0 + dr_b1
    r_b2 = r_b1 + dr_b2
    r_b3 = r_b1 + dr_b3
    r_b4 = r_b1 + dr_b4
    if verbose:
        print(f"r_b1: {r_b1}")
        print(f"r_b2: {r_b2}")
        print(f"r_b3: {r_b3}")
        print(f"r_b4: {r_b4}")

    # Initial body poses
    q_i_1 = wp.transformf(r_b1, wp.quat_identity(dtype=wp.float32))
    q_i_2 = wp.transformf(r_b2, wp.quat_identity(dtype=wp.float32))
    q_i_3 = wp.transformf(r_b3, wp.quat_identity(dtype=wp.float32))
    q_i_4 = wp.transformf(r_b4, wp.quat_identity(dtype=wp.float32))

    # Initial joint positions
    r_j1 = wp.vec3f(r_b2.x, 0.0, r_b1.z)
    r_j2 = wp.vec3f(r_b2.x, 0.0, r_b3.z)
    r_j3 = wp.vec3f(r_b4.x, 0.0, r_b3.z)
    r_j4 = wp.vec3f(r_b4.x, 0.0, r_b1.z)
    if verbose:
        print(f"r_j1: {r_j1}")
        print(f"r_j2: {r_j2}")
        print(f"r_j3: {r_j3}")
        print(f"r_j4: {r_j4}")

    ###
    # Bodies
    ###

    bid1 = _builder.add_link(
        key="link_1",
        mass=m_i,
        I_m=i_I_i_1,
        xform=q_i_1,
        lock_inertia=True,
    )

    bid2 = _builder.add_link(
        key="link_2",
        mass=m_i,
        I_m=i_I_i_2,
        xform=q_i_2,
        lock_inertia=True,
    )

    bid3 = _builder.add_link(
        key="link_3",
        mass=m_i,
        I_m=i_I_i_3,
        xform=q_i_3,
        lock_inertia=True,
    )

    bid4 = _builder.add_link(
        key="link_4",
        mass=m_i,
        I_m=i_I_i_4,
        xform=q_i_4,
        lock_inertia=True,
    )

    ###
    # Geometries
    ###

    _builder.add_shape_box(
        key="box_1",
        body=bid1,
        hx=0.5 * d_1,
        hy=0.5 * w_1,
        hz=0.5 * h_1,
        cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
    )
    _builder.add_shape_box(
        key="box_2",
        body=bid2,
        hx=0.5 * d_2,
        hy=0.5 * w_2,
        hz=0.5 * h_2,
        cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
    )

    _builder.add_shape_box(
        key="box_3",
        body=bid3,
        hx=0.5 * d_3,
        hy=0.5 * w_3,
        hz=0.5 * h_3,
        cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
    )

    _builder.add_shape_box(
        key="box_4",
        body=bid4,
        hx=0.5 * d_4,
        hy=0.5 * w_4,
        hz=0.5 * h_4,
        cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_shape_box(
            key="ground",
            body=-1,
            hx=10.0,
            hy=10.0,
            hz=0.5,
            xform=wp.transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
        )

    ###
    # Joints
    ###

    if limits:
        qmin = -0.25 * math.pi
        qmax = 0.25 * math.pi
    else:
        qmin = float(JOINT_QMIN)
        qmax = float(JOINT_QMAX)

    if fixedbase:
        _builder.add_joint_fixed(
            key="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transformf(-r_b1, wp.quat_identity(dtype=wp.float32)),
        )

    if floatingbase:
        _builder.add_joint_free(
            key="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

    passive_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=ActuatorMode.NONE,
        limit_lower=qmin,
        limit_upper=qmax,
    )
    actuated_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=ActuatorMode.EFFORT,
        limit_lower=qmin,
        limit_upper=qmax,
    )

    _builder.add_joint_revolute(
        key="link1_to_link2",
        parent=bid1,
        child=bid2,
        axis=actuated_joint_dof_config if 1 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j1 - r_b1, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j1 - r_b2, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        key="link2_to_link3",
        parent=bid2,
        child=bid3,
        axis=actuated_joint_dof_config if 2 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j2 - r_b2, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j2 - r_b3, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        key="link3_to_link4",
        parent=bid3,
        child=bid4,
        axis=actuated_joint_dof_config if 3 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j3 - r_b3, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j3 - r_b4, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        key="link4_to_link1",
        parent=bid4,
        child=bid1,
        axis=actuated_joint_dof_config if 4 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j4 - r_b4, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j4 - r_b1, wp.quat_identity(dtype=wp.float32)),
    )

    # Signal the end of setting-up the new world
    if new_world or builder is None:
        _builder.end_world()

    # Return the lists of element indices
    return _builder


###
# Tests
###


class TestKaminoContainers(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        # self.verbose = test_context.verbose  # Set to True to enable verbose output
        self.verbose = True  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_model_conversions_fourbar_from_builder(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on a simple fourbar model created explicitly using the builder.
        """
        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = build_boxes_fourbar_newton(
            builder=builder_0,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[1, 3],
        )

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # Create a fourbar using Kamino's ModelBuilderKamino
        builder_1: ModelBuilderKamino = basics.build_boxes_fourbar(
            builder=None,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[1, 3],
        )

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_2, model_1)

    def test_01_model_conversions_fourbar_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on a simple fourbar model loaded from USD.
        """
        # TODO
        USD_MODEL_PATH = os.path.join(get_basics_usd_assets_path(), "boxes_fourbar.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.begin_world()
        builder_0.add_usd(source=USD_MODEL_PATH, joint_ordering=None)
        builder_0.end_world()

        # TODO
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=True)

        # TODO
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_2, model_1)

    def test_02_model_conversions_dr_testmech_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the DR testmechanism model loaded from USD.
        """
        # TODO
        USD_MODEL_PATH = os.path.join(get_examples_usd_assets_path(), "dr_testmech/usd/dr_testmech.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.begin_world()
        builder_0.add_usd(source=USD_MODEL_PATH, joint_ordering=None)
        builder_0.end_world()

        # TODO
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=True)

        # TODO
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_2, model_1)

    def test_03_model_conversions_dr_legs_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the DR legs model loaded from USD.
        """
        # TODO
        USD_MODEL_PATH = os.path.join(get_examples_usd_assets_path(), "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.begin_world()
        builder_0.add_usd(source=USD_MODEL_PATH, joint_ordering=None)
        builder_0.end_world()

        # TODO
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=True)

        # TODO
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)

        # np.set_printoptions(precision=23, suppress=True)
        # model_0_i_I_i_np = model_0.body_inertia.numpy()
        # model_1_i_I_i_np = model_1.bodies.i_I_i.numpy()
        # model_2_i_I_i_np = model_2.bodies.i_I_i.numpy()
        # for i in range(model_0.body_count):
        #     msg.warning("body %s inertia comparison:\nmodel_0:\n%s\nmodel_1:\n%s\nmodel_2:\n%s\n", i, model_0_i_I_i_np[i], model_1_i_I_i_np[i], model_2_i_I_i_np[i])
        #     np.testing.assert_allclose(
        #         actual=model_0_i_I_i_np[i],
        #         desired=model_1_i_I_i_np[i],
        #         err_msg=f"Body {i} inertia does not match between model_0 and model_1.",
        #         atol=1e-7,
        #         rtol=1e-7,
        #     )
        #     np.testing.assert_allclose(
        #         actual=model_0_i_I_i_np[i],
        #         desired=model_2_i_I_i_np[i],
        #         err_msg=f"Body {i} inertia does not match between model_0 and model_2.",
        #     )

        # np.set_printoptions(precision=23, suppress=True)
        # model_0_inv_i_I_i_np = model_0.body_inv_inertia.numpy()
        # model_1_inv_i_I_i_np = model_1.bodies.inv_i_I_i.numpy()
        # model_2_inv_i_I_i_np = model_2.bodies.inv_i_I_i.numpy()
        # for i in range(model_0.body_count):
        #     msg.warning("body %s inertia comparison:\nmodel_0:\n%s\nmodel_1:\n%s\nmodel_2:\n%s\n", i, model_0_inv_i_I_i_np[i], model_1_inv_i_I_i_np[i], model_2_inv_i_I_i_np[i])
        #     np.testing.assert_allclose(
        #         actual=model_0_inv_i_I_i_np[i],
        #         desired=model_1_inv_i_I_i_np[i],
        #         err_msg=f"Body {i} inertia does not match between model_0 and model_1.",
        #         atol=1e-6,
        #         rtol=1e-6,
        #     )
        #     np.testing.assert_allclose(
        #         actual=model_0_inv_i_I_i_np[i],
        #         desired=model_2_inv_i_I_i_np[i],
        #         err_msg=f"Body {i} inertia does not match between model_0 and model_2.",
        #         atol=1e-7,
        #         rtol=1e-7,
        #     )

        # NOTE: We don't check mesh geometry pointers since they have been loaded separately
        # TODO: Check mesh geometry data explicitly: vertices, triangle, normals etc
        test_util_checks.assert_model_equal(
            self, model_2, model_1, check_geom_source_ptr=False, check_geom_group_and_collides=False
        )

    def test_04_model_conversions_anymal_d_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the Anymal D model loaded from USD.
        """
        # TODO
        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.begin_world()
        builder_0.add_usd(
            source=asset_file,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        builder_0.end_world()

        # TODO
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(source=asset_file, load_static_geometry=True)

        # TODO
        model_0: Model = builder_0.finalize()
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        msg.critical("model_0.joint_X_p:\n%s", model_0.joint_X_p)
        msg.critical("model_0.joint_X_c:\n%s", model_0.joint_X_c)
        msg.error("model_1.joints.B_r_Bj:\n%s", model_1.joints.B_r_Bj)
        msg.error("model_2.joints.B_r_Bj:\n%s\n", model_2.joints.B_r_Bj)
        msg.error("model_1.joints.F_r_Fj:\n%s", model_1.joints.F_r_Fj)
        msg.error("model_2.joints.F_r_Fj:\n%s\n", model_2.joints.F_r_Fj)
        # NOTE: We don't check mesh geometry pointers since they have been loaded separately
        # TODO: Check mesh geometry data explicitly: vertices, triangle, normals etc
        test_util_checks.assert_model_equal(
            self, model_2, model_1, check_geom_source_ptr=False, check_geom_group_and_collides=False
        )

    def test_10_state_conversions(self):
        """
        Test the conversion operations between newton.State and kamino.StateKamino.
        """
        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = build_boxes_fourbar_newton(
            builder=builder_0,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[2, 4],
        )

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # Create a fourbar using Kamino's ModelBuilderKamino
        builder_1: ModelBuilderKamino = basics.build_boxes_fourbar(
            builder=None,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[2, 4],
        )

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_1, model_2)

        # Create a Newton state container
        state_0: State = model_0.state()
        self.assertIsInstance(state_0.body_q, wp.array)
        self.assertEqual(state_0.body_q.size, model_0.body_count)
        self.assertIsNotNone(state_0.joint_q_prev)
        self.assertEqual(state_0.joint_q_prev.size, model_0.joint_coord_count)
        self.assertIsNotNone(state_0.joint_lambdas)
        self.assertEqual(state_0.joint_lambdas.size, model_0.joint_constraint_count)

        # Create a Kamino state container
        state_1: StateKamino = model_1.state()
        self.assertIsInstance(state_1.q_i, wp.array)
        self.assertEqual(state_1.q_i.size, model_1.size.sum_of_num_bodies)

        state_2: StateKamino = StateKamino.from_newton(model_0, state_0, True, False)
        self.assertIsInstance(state_2.q_i, wp.array)
        self.assertEqual(state_2.q_i.size, model_1.size.sum_of_num_bodies)
        self.assertIs(state_2.q_i, state_0.body_q)
        self.assertIs(state_2.u_i.ptr, state_0.body_qd.ptr)  # NOTE: Check ptr due to conversion from wp.spatial_vectorf
        self.assertIs(state_2.w_i.ptr, state_0.body_f.ptr)  # NOTE: Check ptr due to conversion from wp.spatial_vectorf
        self.assertIs(state_2.q_j, state_0.joint_q)
        self.assertIs(state_2.dq_j, state_0.joint_qd)
        self.assertIs(state_2.q_j_p, state_0.joint_q_prev)
        test_util_checks.assert_state_equal(self, state_2, state_1)

        state_3: State = StateKamino.to_newton(model_0, state_2)
        self.assertIsInstance(state_3.body_q, wp.array)
        self.assertEqual(state_3.body_q.size, model_0.body_count)
        self.assertIs(state_3.body_q, state_2.q_i)
        self.assertIs(state_3.body_qd.ptr, state_2.u_i.ptr)  # NOTE: Check ptr due to conversion from vec6f
        self.assertIs(state_3.body_f.ptr, state_2.w_i.ptr)  # NOTE: Check ptr due to conversion from vec6f
        self.assertIs(state_3.joint_q, state_2.q_j)
        self.assertIs(state_3.joint_qd, state_2.dq_j)
        self.assertIs(state_3.joint_q_prev, state_2.q_j_p)

    def test_20_control_conversions(self):
        """
        Test the conversions between newton.Control and kamino.ControlKamino.
        """
        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = build_boxes_fourbar_newton(
            builder=builder_0,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[1, 2, 3, 4],
        )

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # Create a fourbar using Kamino's ModelBuilderKamino
        builder_1: ModelBuilderKamino = basics.build_boxes_fourbar(
            builder=None,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[1, 2, 3, 4],
        )

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_1, model_2)

        # Create a Newton control container
        control_0: Control = model_0.control()
        self.assertIsInstance(control_0.joint_f, wp.array)
        self.assertEqual(control_0.joint_f.size, model_0.joint_dof_count)

        # Create a Kamino control container
        control_1: ControlKamino = model_1.control()
        self.assertIsInstance(control_1.tau_j, wp.array)
        self.assertEqual(control_1.tau_j.size, model_1.size.sum_of_num_joint_dofs)

        # Create a Kamino control container
        control_2: ControlKamino = ControlKamino.from_newton(control_0)
        self.assertIsInstance(control_2.tau_j, wp.array)
        self.assertIs(control_2.tau_j, control_0.joint_f)
        self.assertEqual(control_2.tau_j.size, model_0.joint_dof_count)
        test_util_checks.assert_control_equal(self, control_2, control_1)

        # Convert back to a Newton control container
        control_3: Control = ControlKamino.to_newton(control_2)
        self.assertIsInstance(control_3.joint_f, wp.array)
        self.assertIs(control_3.joint_f, control_2.tau_j)
        self.assertEqual(control_3.joint_f.size, model_0.joint_dof_count)


# class TestKaminoNewtonIntegration(unittest.TestCase):
#     def setUp(self):
#         if not test_context.setup_done:
#             setup_tests(clear_cache=False)
#         self.default_device = wp.get_device(test_context.device)
#         # self.verbose = test_context.verbose  # Set to True to enable verbose output
#         self.verbose = True  # Set to True to enable verbose output

#         # Set debug-level logging to print verbose test output to console
#         if self.verbose:
#             print("\n")  # Add newline before test output for better readability
#             msg.set_log_level(msg.LogLevel.INFO)
#         else:
#             msg.reset_log_level()

#     def tearDown(self):
#         self.default_device = None
#         if self.verbose:
#             msg.reset_log_level()

#     def test_usd_boxes_fourbar_newton(self):
#         """
#         TODO
#         """
#         builder = ModelBuilder()

#         # TODO
#         USD_MODEL_PATH = os.path.join(get_basics_usd_assets_path(), "boxes_fourbar.usda")
#         builder.begin_world()
#         builder.add_usd(source=USD_MODEL_PATH, joint_ordering=None)
#         builder.end_world()

#         # TODO
#         msg.info("builder.particle_count: %s", builder.particle_count)
#         msg.info("builder.body_count: %s", builder.body_count)
#         msg.info("builder.body_world: %s", builder.body_world)
#         msg.info("builder.body_key: %s", builder.body_key)
#         msg.info("builder.shape_count: %s", builder.shape_count)
#         msg.info("builder.shape_world: %s", builder.shape_world)
#         msg.info("builder.shape_key: %s", builder.shape_key)
#         msg.info("builder.joint_count: %s", builder.joint_count)
#         msg.info("builder.joint_coord_count: %s", builder.joint_coord_count)
#         msg.info("builder.joint_dof_count: %s", builder.joint_dof_count)
#         msg.info("builder.joint_key: %s", builder.joint_key)
#         msg.info("builder.joint_world: %s", builder.joint_world)
#         msg.info("builder.joint_parent: %s", builder.joint_parent)
#         msg.info("builder.joint_child: %s", builder.joint_child)
#         msg.info("builder.joint_q_start: %s", builder.joint_q_start)
#         msg.info("builder.joint_qd_start: %s", builder.joint_qd_start)
#         msg.info("builder.articulation_count: %s", builder.articulation_count)
#         msg.info("builder.articulation_start: %s", builder.articulation_start)
#         msg.info("builder.articulation_world: %s", builder.articulation_world)

#     def test_boxes_fourbar_newton(self):
#         """
#         TODO
#         """
#         ###
#         # Builders
#         ###

#         # Create a fourbar using Newton's ModelBuilder
#         builder_0: ModelBuilder = build_boxes_fourbar_newton(
#             builder=None,
#             z_offset=0.0,
#             fixedbase=False,
#             floatingbase=True,
#             limits=True,
#             ground=True,
#             new_world=True,
#             actuator_ids=[0, 1, 2, 3, 4],
#         )

#         # Duplicate the world to test multi-world handling
#         builder_0.begin_world()
#         builder_0.add_builder(copy.deepcopy(builder_0))
#         builder_0.end_world()

#         # Create a fourbar using Kamino's ModelBuilderKamino
#         builder_1: ModelBuilderKamino = basics.build_boxes_fourbar(
#             builder=None,
#             z_offset=0.0,
#             fixedbase=False,
#             floatingbase=True,
#             limits=True,
#             ground=True,
#             new_world=True,
#             actuator_ids=[0, 1, 2, 3, 4],
#         )

#         # Duplicate the world to test multi-world handling
#         builder_1.add_builder(copy.deepcopy(builder_1))

#         ###
#         # Models
#         ###

#         # Create models from the builders and conversion operations, and check for consistency
#         model_0_nwt: Model = builder_0.finalize(skip_validation_joints=True)
#         model_0: ModelKamino = ModelKamino.from_newton(model_0_nwt)
#         model_1: ModelKamino = builder_1.finalize()
#         assert_model_equal(self, model_0, model_1)

#         ###
#         # State
#         ###

#         # Create states from the models and conversion operations, and check for consistency
#         state_0_nwt: State = model_0_nwt.state()
#         state_0: StateKamino = model_0.state()
#         state_1: StateKamino = model_1.state()
#         state_2 = StateKamino.from_newton(model_0_nwt, state_0_nwt)
#         assert_state_equal(self, state_0, state_1)
#         assert_state_equal(self, state_1, state_2)

#         ###
#         # Control
#         ###

#         # Create controls from the models and conversion operations, and check for consistency
#         control_0_nwt: Control = model_0_nwt.control()
#         control_0: ControlKamino = model_0.control()
#         control_1: ControlKamino = model_1.control()
#         control_2 = ControlKamino.from_newton(control_0_nwt)
#         assert_control_equal(self, control_0, control_1)
#         assert_control_equal(self, control_1, control_2)

#         ###
#         # Data
#         ###

#         # TODO: Add checks
#         # data_0: DataKamino = model_0.data()
#         # data_1: DataKamino = model_1.data()
#         # assert_data_equal(self, data_0, data_1)

#         ###
#         # Contacts
#         ###

#         # TODO: Add checks
#         model_max_contacts_1, world_max_contacts_1 = builder_1.compute_required_contact_capacity(
#             max_contacts_per_pair=DEFAULT_GEOM_PAIR_MAX_CONTACTS,
#             max_contacts_per_world=None,  # Let the builder compute this value from the number of geoms/shapes
#         )

#         # TODO
#         msg.info("model_0_nwt.shape_collision_filter_pairs: %s", model_0_nwt.shape_collision_filter_pairs)
#         msg.info("model_0_nwt.shape_contact_pair_count: %s", model_0_nwt.shape_contact_pair_count)
#         msg.info("model_0_nwt.shape_contact_pairs:\n%s\n\n", model_0_nwt.shape_contact_pairs)

#         # TODO
#         msg.info("model_0_nwt.rigid_contact_max: %s", model_0_nwt.rigid_contact_max)
#         msg.info("builder_1.model_max_contacts_1: %s", model_max_contacts_1)
#         msg.info("builder_1.world_max_contacts_1: %s\n\n", world_max_contacts_1)

#         # TODO
#         contacts_0_nwt: Contacts = Contacts(
#             rigid_contact_max=model_0_nwt.rigid_contact_max,
#             soft_contact_max=0,
#             requires_grad=model_0_nwt.requires_grad,
#             device=model_0_nwt.device,
#             per_contact_shape_properties=False,
#             clear_buffers=True,
#         )
#         contacts_1: ContactsKamino = ContactsKamino(capacity=world_max_contacts_1, device=model_1.device)

#         # TODO
#         msg.info("contacts_0_nwt.rigid_contact_max: %s", contacts_0_nwt.rigid_contact_max)
#         msg.info("contacts_1.model_max_contacts_host: %s", contacts_1.model_max_contacts_host)
#         msg.info("contacts_1.world_max_contacts_host: %s\n\n", contacts_1.world_max_contacts_host)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
