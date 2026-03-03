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
import os
import unittest

import warp as wp

import newton
import newton._src.solvers.kamino.tests.utils.checks as test_util_checks
from newton._src.sim import Control, Model, ModelBuilder, State
from newton._src.solvers.kamino.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino.core.control import ControlKamino
from newton._src.solvers.kamino.core.model import ModelKamino
from newton._src.solvers.kamino.core.state import StateKamino
from newton._src.solvers.kamino.models import basics as basics_kamino
from newton._src.solvers.kamino.models import basics_newton, get_basics_usd_assets_path, get_examples_usd_assets_path
from newton._src.solvers.kamino.models.builders import utils as model_utils
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils import print as print_utils
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.io.usd import USDImporter

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
        builder = basics_kamino.build_boxes_hinged()

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
        builder1 = basics_kamino.build_boxes_hinged()
        builder2 = basics_kamino.build_boxes_nunchaku()

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
        builder = model_utils.make_homogeneous_builder(num_worlds=num_worlds, build_fn=basics_kamino.build_boxes_hinged)

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
        builder = basics_kamino.make_basics_heterogeneous_builder()
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
        builder_0: ModelBuilder = basics_newton.build_boxes_fourbar(
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
        builder_1: ModelBuilderKamino = basics_kamino.build_boxes_fourbar(
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
        builder_0: ModelBuilder = basics_newton.build_boxes_fourbar(
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
        builder_1: ModelBuilderKamino = basics_kamino.build_boxes_fourbar(
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

        state_2: StateKamino = StateKamino.from_newton(model_2.size, model_0, state_0, True, False)
        self.assertIsInstance(state_2.q_i, wp.array)
        self.assertEqual(state_2.q_i.size, model_1.size.sum_of_num_bodies)
        # NOTE: Check ptr due to conversion from wp.spatial_vectorf
        self.assertIs(state_2.u_i.ptr, state_0.body_qd.ptr)
        self.assertIs(state_2.w_i.ptr, state_0.body_f_total.ptr)
        self.assertIs(state_2.w_i_e.ptr, state_0.body_f.ptr)
        # NOTE: Check that the same arrays because these should be pure references
        self.assertIs(state_2.q_i, state_0.body_q)
        self.assertIs(state_2.q_j, state_0.joint_q)
        self.assertIs(state_2.dq_j, state_0.joint_qd)
        self.assertIs(state_2.q_j_p, state_0.joint_q_prev)
        self.assertIs(state_2.lambda_j, state_0.joint_lambdas)
        test_util_checks.assert_state_equal(self, state_2, state_1)

        state_3: State = StateKamino.to_newton(model_0, state_2)
        self.assertIsInstance(state_3.body_q, wp.array)
        self.assertEqual(state_3.body_q.size, model_0.body_count)
        # NOTE: Check ptr due to conversion from vec6f
        self.assertIs(state_3.body_qd.ptr, state_2.u_i.ptr)
        self.assertIs(state_3.body_f_total.ptr, state_2.w_i.ptr)
        self.assertIs(state_3.body_f.ptr, state_2.w_i_e.ptr)
        # NOTE: Check that the same arrays because these should be pure references
        self.assertIs(state_3.body_q, state_2.q_i)
        self.assertIs(state_3.joint_q, state_2.q_j)
        self.assertIs(state_3.joint_qd, state_2.dq_j)
        self.assertIs(state_3.joint_q_prev, state_2.q_j_p)
        self.assertIs(state_3.joint_lambdas, state_2.lambda_j)

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
        builder_0: ModelBuilder = basics_newton.build_boxes_fourbar(
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
        builder_1: ModelBuilderKamino = basics_kamino.build_boxes_fourbar(
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
