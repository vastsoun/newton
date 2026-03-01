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
Unit tests for the :class:`ModelKamino` class and related functionality.
"""

import copy
import math
import os
import unittest

import warp as wp

# Newton imports
import newton

# Kamino imports
import newton._src.solvers.kamino.tests.utils.checks as test_util_checks
from newton._src.core import Axis
from newton._src.sim import (
    # JointType,
    Control,
    JointTargetMode,
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
from newton._src.solvers.kamino.models.builders import utils as model_utils
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils import print as print_utils
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
    dynamic_joints: bool = False,
    implicit_pd: bool = False,
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
        label="link_1",
        mass=m_i,
        inertia=i_I_i_1,
        xform=q_i_1,
        lock_inertia=True,
    )

    bid2 = _builder.add_link(
        label="link_2",
        mass=m_i,
        inertia=i_I_i_2,
        xform=q_i_2,
        lock_inertia=True,
    )

    bid3 = _builder.add_link(
        label="link_3",
        mass=m_i,
        inertia=i_I_i_3,
        xform=q_i_3,
        lock_inertia=True,
    )

    bid4 = _builder.add_link(
        label="link_4",
        mass=m_i,
        inertia=i_I_i_4,
        xform=q_i_4,
        lock_inertia=True,
    )

    ###
    # Geometries
    ###

    _builder.add_shape_box(
        label="box_1",
        body=bid1,
        hx=0.5 * d_1,
        hy=0.5 * w_1,
        hz=0.5 * h_1,
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )
    _builder.add_shape_box(
        label="box_2",
        body=bid2,
        hx=0.5 * d_2,
        hy=0.5 * w_2,
        hz=0.5 * h_2,
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )

    _builder.add_shape_box(
        label="box_3",
        body=bid3,
        hx=0.5 * d_3,
        hy=0.5 * w_3,
        hz=0.5 * h_3,
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )

    _builder.add_shape_box(
        label="box_4",
        body=bid4,
        hx=0.5 * d_4,
        hy=0.5 * w_4,
        hz=0.5 * h_4,
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_shape_box(
            label="ground",
            body=-1,
            hx=10.0,
            hy=10.0,
            hz=0.5,
            xform=wp.transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
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
            label="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transformf(-r_b1, wp.quat_identity(dtype=wp.float32)),
        )

    if floatingbase:
        _builder.add_joint_free(
            label="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

    passive_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.NONE,
        limit_lower=qmin,
        limit_upper=qmax,
    )
    effort_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.EFFORT,
        limit_lower=qmin,
        limit_upper=qmax,
        armature=0.1 if dynamic_joints else 0.0,
        friction=0.001 if dynamic_joints else 0.0,
    )
    pd_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.POSITION_VELOCITY,
        armature=0.1 if dynamic_joints else 0.0,
        friction=0.001 if dynamic_joints else 0.0,
        target_ke=1000.0,
        target_kd=20.0,
        limit_lower=qmin,
        limit_upper=qmax,
    )

    joint_1_config_if_implicit_pd = pd_joint_dof_config if implicit_pd else effort_joint_dof_config
    joint_1_config_if_actuated = joint_1_config_if_implicit_pd if 1 in actuator_ids else passive_joint_dof_config
    _builder.add_joint_revolute(
        label="link1_to_link2",
        parent=bid1,
        child=bid2,
        axis=joint_1_config_if_actuated if 1 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j1 - r_b1, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j1 - r_b2, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        label="link2_to_link3",
        parent=bid2,
        child=bid3,
        axis=effort_joint_dof_config if 2 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j2 - r_b2, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j2 - r_b3, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        label="link3_to_link4",
        parent=bid3,
        child=bid4,
        axis=effort_joint_dof_config if 3 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j3 - r_b3, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j3 - r_b4, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        label="link4_to_link1",
        parent=bid4,
        child=bid1,
        axis=effort_joint_dof_config if 4 in actuator_ids else passive_joint_dof_config,
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


class TestModel(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_single_model(self):
        # Create a model builder
        builder = basics.build_boxes_hinged()

        # Finalize the model
        model: ModelKamino = builder.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(state)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, builder.num_bodies)
        self.assertEqual(model.size.sum_of_num_joints, builder.num_joints)
        self.assertEqual(model.size.sum_of_num_geoms, builder.num_geoms)
        self.assertEqual(model.device, self.default_device)

    def test_02_double_model(self):
        # Create a model builder
        builder1 = basics.build_boxes_hinged()
        builder2 = basics.build_boxes_nunchaku()

        # Compute the total number of elements from the two builders
        total_nb = builder1.num_bodies + builder2.num_bodies
        total_nj = builder1.num_joints + builder2.num_joints
        total_ng = builder1.num_geoms + builder2.num_geoms

        # Add the second builder to the first one
        builder1.add_builder(builder2)

        # Finalize the model
        model: ModelKamino = builder1.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)

        # Create a model state
        data = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(data)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, total_nb)
        self.assertEqual(model.size.sum_of_num_joints, total_nj)
        self.assertEqual(model.size.sum_of_num_geoms, total_ng)

    def test_03_homogeneous_model(self):
        # Constants
        num_worlds = 4

        # Create a model builder
        builder = model_utils.make_homogeneous_builder(num_worlds=num_worlds, build_fn=basics.build_boxes_hinged)

        # Finalize the model
        model: ModelKamino = builder.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(state)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, num_worlds * 2)
        self.assertEqual(model.size.sum_of_num_joints, num_worlds * 1)
        self.assertEqual(model.size.sum_of_num_geoms, num_worlds * 3)
        self.assertEqual(model.device, self.default_device)

    def test_04_hetereogeneous_model(self):
        # Create a model builder
        builder = basics.make_basics_heterogeneous_builder()
        num_worlds = builder.num_worlds

        # Finalize the model
        model: ModelKamino = builder.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)
            print("")  # Add a newline for better readability
            print_utils.print_model_bodies(model)
            print("")  # Add a newline for better readability
            print_utils.print_model_joints(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(state)

        # Check the model info entries
        self.assertEqual(model.info.num_worlds, num_worlds)


class TestModelConversions(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        # self.verbose = test_context.verbose  # Set to True to enable verbose output
        self.verbose = True  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)  # TODO @nvtw: set this to DEBUG when investigating noted issues
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
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = build_boxes_fourbar_newton(
            builder=builder_0,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            dynamic_joints=False,
            implicit_pd=False,
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
            dynamic_joints=False,
            implicit_pd=False,
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
        # Define the path to the USD file for the fourbar model
        USD_MODEL_PATH = os.path.join(get_basics_usd_assets_path(), "boxes_fourbar.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=USD_MODEL_PATH,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder_0.end_world()

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=USD_MODEL_PATH,
            load_drive_dynamics=True,
            load_static_geometry=True,
            force_show_colliders=True,
            use_prim_path_names=True,
        )

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_2, model_1)

        # TODO: IMPLEMENT THIS CHECK: We wanna see if the both generate
        # the same data containers and unilateral constraint info
        # data_1: DataKamino = model_1.data()
        # data_2: DataKamino = model_2.data()
        # make_unilateral_constraints_info(model=model_1, data=data_1)
        # make_unilateral_constraints_info(model=model_2, data=data_2)
        # test_util_checks.assert_model_equal(self, model_2, model_1)
        # test_util_checks.assert_data_equal(self, data_2, data_1)

    def test_02_model_conversions_dr_testmech_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the DR testmechanism model loaded from USD.
        """
        # Define the path to the USD file for the DR testmechanism model
        USD_MODEL_PATH = os.path.join(get_examples_usd_assets_path(), "dr_testmech/usd/dr_testmech.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=USD_MODEL_PATH,
            joint_ordering=None,
            force_show_colliders=True,
        )
        builder_0.end_world()

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=USD_MODEL_PATH,
            load_static_geometry=True,
            retain_joint_ordering=False,
            meshes_are_collidable=True,
            force_show_colliders=True,
            use_prim_path_names=True,
        )

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        # NOTE: We don't check:
        # - mesh geometry pointers since they have been loaded separately
        test_util_checks.assert_model_equal(self, model_2, model_1, excluded=["ptr"])

    def test_03_model_conversions_dr_legs_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the DR legs model loaded from USD.
        """
        # Define the path to the USD file for the DR legs model
        USD_MODEL_PATH = os.path.join(get_examples_usd_assets_path(), "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=USD_MODEL_PATH,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder_0.end_world()

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=USD_MODEL_PATH,
            load_drive_dynamics=True,
            force_show_colliders=True,
            use_prim_path_names=True,
        )

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        # NOTE: We don't check:
        # - mesh geometry pointers since they have been loaded separately
        # - the shape contact group (TODO @nvtw: investigate why) because newton.ModelBuilder
        #   sets it to `1` even for non-collidable visual shapes
        # - shape gap since newton.ModelBuilder sets it to `0.001` for all shapes even if
        #   the default shape config has gap=0.0
        # - excluded/filtered collision pairs since newton.ModelBuilder preemptively adds
        #   geom-pairs of joint neighbours to `shape_collision_filter_pairs` regardless of
        #   whether they are actually collidable or not, which leads to differences in the
        #   number of excluded pairs and their contents
        excluded = ["ptr", "group", "gap", "num_excluded_pairs", "excluded_pairs"]
        test_util_checks.assert_model_equal(self, model_2, model_1, excluded=excluded)

    def test_04_model_conversions_anymal_d_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the Anymal D model loaded from USD.
        """
        # Define the path to the USD file for the Anymal D model
        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=asset_file,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        builder_0.end_world()

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=asset_file,
            load_static_geometry=True,
            retain_geom_ordering=False,
            use_articulation_root_name=False,
            force_show_colliders=True,
            use_prim_path_names=True,
        )

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize()
        # TODO @nvtw: Why are shape_collision_group[i] values for
        # visual shapes set to `=1` since they are not collidable?
        msg.error(f"model_0.shape_collision_group:\n{model_0.shape_collision_group}\n")
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        # NOTE: We don't check mesh geometry pointers since they have been loaded separately
        excluded = [
            "i_r_com_i",  # TODO: Investigate if the difference is expected or not
            "i_I_i",  # TODO: Investigate if the difference is expected or not
            "inv_i_I_i",  # TODO: Investigate if the difference is expected or not
            "q_i_0",  # TODO: Investigate if the difference is expected or not
            "B_r_Bj",  # TODO: Investigate if the difference is expected or not
            "F_r_Fj",  # TODO: Investigate if the difference is expected or not
            "X_j",  # TODO: Investigate if the difference is expected or not
            "q_j_0",  # TODO: Investigate if the difference is expected or not
            "num_collidable_pairs",  # TODO: newton.ModelBuilder preemptively adding geom-pairs to shape_collision_filter_pairs
            "num_excluded_pairs",  # TODO: newton.ModelBuilder preemptively adding geom-pairs to shape_collision_filter_pairs
            "model_minimum_contacts",  # TODO: Investigate
            "world_minimum_contacts",  # TODO: Investigate
            "offset",  # TODO: Investigate if the difference is expected or not
            "group",  # TODO: newton.ModelBuilder setting shape_collision_group=1 for all shapes even non-collidable ones
            "gap",  # TODO: newton.ModelBuilder setting shape gap to 0.001 for all shapes even if default shape config has gap=0.0
            "ptr",  # Exclude geometry pointers since they have been loaded separately
            "collidable_pairs",  # TODO @nvtw: not sure why these are different
            "excluded_pairs",  # TODO: newton.ModelBuilder preemptively adding geom-pairs to shape_collision_filter_pairs
        ]
        test_util_checks.assert_model_equal(self, model_2, model_1, excluded=excluded)

    def test_10_state_conversions(self):
        """
        Test the conversion operations between newton.State and kamino.StateKamino.
        """
        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

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
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = build_boxes_fourbar_newton(
            builder=builder_0,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            # dynamic_joints=True,
            # implicit_pd=True,
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
            # dynamic_joints=True,
            # implicit_pd=True,
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
        test_util_checks.assert_model_equal(self, model_2, model_1)

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


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
