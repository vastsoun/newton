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
import os
import unittest

# import numpy as np
import warp as wp

from newton._src.sim import (
    Model,
    ModelAttributeAssignment,
    ModelAttributeFrequency,
    ModelBuilder,
    State,
)
from newton._src.solvers.kamino.core.new_model import KaminoModel
from newton._src.solvers.kamino.core.new_data import KaminoData
from newton._src.solvers.kamino.models import get_basics_usd_assets_path
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils import logger as msg

###
# Utilities
###


def register_solver_attributes(builder: ModelBuilder) -> None:
    """
    TODO
    """
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="num_body_dofs",
            frequency=ModelAttributeFrequency.WORLD,
            assignment=ModelAttributeAssignment.MODEL,
            dtype=wp.int32,
            default=0,
            namespace="info",
        )
    )


###
# Tests
###


class TestKaminoNewtonIntegration(unittest.TestCase):
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

    def test_custom_fourbar(self):
        """
        TODO
        """
        builder = ModelBuilder()

        # TODO
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_index",
                frequency=ModelAttributeFrequency.BODY,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_index",
                frequency=ModelAttributeFrequency.JOINT,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="shape_index",
                frequency=ModelAttributeFrequency.SHAPE,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
            )
        )

        # TODO
        builder.begin_world()
        builder.add_link(key="body_0", mass=1.0, custom_attributes={"body_index": 0})
        builder.add_link(key="body_1", mass=1.0, custom_attributes={"body_index": 1})
        builder.add_link(key="body_2", mass=1.0, custom_attributes={"body_index": 2})
        builder.add_link(key="body_3", mass=1.0, custom_attributes={"body_index": 3})
        builder.add_joint_free(
            key="joint_0",
            child=0,
            parent_xform=wp.transformf(),
            child_xform=wp.transformf(),
            custom_attributes={"joint_index": 0},
        )
        builder.add_joint_revolute(
            key="joint_1",
            parent=0,
            child=1,
            axis=(0, 0, 1),
            parent_xform=wp.transformf(),
            child_xform=wp.transformf(),
            custom_attributes={"joint_index": 1},
        )
        builder.add_joint_revolute(
            key="joint_2",
            parent=0,
            child=1,
            axis=(0, 0, 1),
            parent_xform=wp.transformf(),
            child_xform=wp.transformf(),
            custom_attributes={"joint_index": 2},
        )
        builder.add_joint_revolute(
            key="joint_3",
            parent=1,
            child=2,
            axis=(0, 0, 1),
            parent_xform=wp.transformf(),
            child_xform=wp.transformf(),
            custom_attributes={"joint_index": 3},
        )
        builder.add_joint_revolute(
            key="joint_4",
            parent=0,
            child=1,
            axis=(0, 0, 1),
            parent_xform=wp.transformf(),
            child_xform=wp.transformf(),
            custom_attributes={"joint_index": 4},
        )
        builder.add_shape_box(key="geom_0", body=0, hx=0.25, hy=0.05, hz=0.05, custom_attributes={"shape_index": 0})
        builder.add_shape_box(key="geom_1", body=1, hx=0.05, hy=0.05, hz=0.25, custom_attributes={"shape_index": 1})
        builder.add_shape_box(key="geom_2", body=2, hx=0.25, hy=0.05, hz=0.05, custom_attributes={"shape_index": 2})
        builder.add_shape_box(key="geom_3", body=3, hx=0.05, hy=0.05, hz=0.25, custom_attributes={"shape_index": 3})
        builder.end_world()

        # TODO
        builder.begin_world()
        builder.add_builder(copy.deepcopy(builder))
        builder.end_world()

        # TODO
        msg.info("builder.num_worlds: %s", builder.num_worlds)
        msg.info("builder.body_count: %s", builder.body_count)
        msg.info("builder.body_world: %s", builder.body_world)
        msg.info("builder.body_key: %s", builder.body_key)
        msg.info("builder.body_mass: %s", builder.body_mass)
        msg.info("builder.body_com: %s", builder.body_com)
        msg.info("builder.body_q: %s", builder.body_q)
        msg.info("builder.body_qd: %s", builder.body_qd)
        msg.info("builder.body_inertia: %s", builder.body_inertia)
        msg.info("builder.joint_count: %s", builder.joint_count)
        msg.info("builder.joint_coord_count: %s", builder.joint_coord_count)
        msg.info("builder.joint_dof_count: %s", builder.joint_dof_count)
        msg.info("builder.joint_world: %s", builder.joint_world)
        msg.info("builder.joint_key: %s", builder.joint_key)
        msg.info("builder.joint_parent: %s", builder.joint_parent)
        msg.info("builder.joint_child: %s", builder.joint_child)
        msg.info("builder.shape_count: %s", builder.shape_count)
        msg.info("builder.shape_world: %s", builder.shape_world)
        msg.info("builder.shape_key: %s", builder.shape_key)
        msg.info("builder.shape_type: %s", builder.shape_type)
        msg.info("builder.articulation_count: %s\n", builder.articulation_count)

        # TODO
        model: Model = builder.finalize()
        msg.info("model.body_key (type: %s): %s", type(model.body_key), model.body_key)
        msg.info("model.body_mass (type: %s): %s", type(model.body_mass), model.body_mass)
        msg.info("model.body_com (type: %s):\n%s", type(model.body_com), model.body_com)
        msg.info("model.body_inertia (type: %s):\n%s", type(model.body_inertia), model.body_inertia)
        msg.info("model.body_index (type: %s): %s", type(model.body_index), model.body_index)
        msg.info("model.joint_key (type: %s): %s", type(model.joint_key), model.joint_key)
        msg.info("model.joint_type (type: %s): %s", type(model.joint_type), model.joint_type)
        msg.info("model.joint_parent (type: %s): %s", type(model.joint_parent), model.joint_parent)
        msg.info("model.joint_child (type: %s): %s", type(model.joint_child), model.joint_child)
        msg.info("model.joint_q_start (type: %s): %s", type(model.joint_q_start), model.joint_q_start)
        msg.info("model.joint_qd_start (type: %s): %s", type(model.joint_qd_start), model.joint_qd_start)
        msg.info("model.joint_q (type: %s): %s", type(model.joint_q), model.joint_q)
        msg.info("model.joint_qd (type: %s): %s", type(model.joint_qd), model.joint_qd)
        msg.info("model.joint_dof_dim (type: %s):\n%s", type(model.joint_dof_dim), model.joint_dof_dim)
        msg.info("model.joint_world (type: %s): %s", type(model.joint_world), model.joint_world)
        msg.info("model.joint_index (type: %s): %s", type(model.joint_index), model.joint_index)
        msg.info("model.shape_index (type: %s): %s\n", type(model.shape_index), model.shape_index)
        # TODO
        msg.info("model.articulation_count (type: %s): %s", type(model.articulation_count), model.articulation_count)
        msg.info("model.articulation_key (type: %s): %s", type(model.articulation_key), model.articulation_key)
        msg.info("model.articulation_start (type: %s): %s", type(model.articulation_start), model.articulation_start)
        msg.info("model.articulation_world (type: %s): %s\n", type(model.articulation_world), model.articulation_world)

        # # TODO
        # self.assertTrue(np.allclose(model.body_index.numpy(), [0, 1, 2, 3]))
        # self.assertTrue(np.allclose(model.joint_index.numpy(), [0, 1, 2, 3, 4]))
        # self.assertTrue(np.allclose(model.shape_index.numpy(), [0, 1, 2, 3]))

        # TODO
        state: State = model.state()
        msg.info("state.body_count (type: %s): %s", type(state.body_count), state.body_count)
        msg.info("state.particle_count (type: %s): %s", type(state.particle_count), state.particle_count)
        msg.info("state.joint_coord_count (type: %s): %s", type(state.joint_coord_count), state.joint_coord_count)
        msg.info("state.joint_dof_count (type: %s): %s", type(state.joint_dof_count), state.joint_dof_count)
        msg.info("state.particle_q (type: %s):\n%s", type(state.particle_q), state.particle_q)
        msg.info("state.particle_qd (type: %s):\n%s", type(state.particle_qd), state.particle_qd)
        msg.info("state.particle_f (type: %s):\n%s", type(state.particle_f), state.particle_f)
        msg.info("state.body_q (type: %s):\n%s", type(state.body_q), state.body_q)
        msg.info("state.body_qd (type: %s):\n%s", type(state.body_qd), state.body_qd)
        msg.info("state.body_q_prev (type: %s):\n%s", type(state.body_q_prev), state.body_q_prev)
        msg.info("state.body_qdd (type: %s):\n%s", type(state.body_qdd), state.body_qdd)
        msg.info("state.body_f (type: %s):\n%s", type(state.body_f), state.body_f)
        msg.info("state.body_parent_f (type: %s):\n%s", type(state.body_parent_f), state.body_parent_f)
        msg.info("state.joint_q (type: %s):\n%s", type(state.joint_q), state.joint_q)
        msg.info("state.joint_qd (type: %s):\n%s", type(state.joint_qd), state.joint_qd)

        # TODO
        kmn_model: KaminoModel = KaminoModel(model)
        # kmn_data: KaminoData = kmn_model.data()

    def test_usd_boxes_fourbar(self):
        """
        TODO
        """
        builder = ModelBuilder()

        # TODO
        USD_MODEL_PATH = os.path.join(get_basics_usd_assets_path(), "boxes_fourbar.usda")
        builder.begin_world()
        builder.add_usd(source=USD_MODEL_PATH, joint_ordering=None)
        builder.end_world()

        # TODO
        msg.info("builder.body_count: %s", builder.body_count)
        msg.info("builder.joint_count: %s", builder.joint_count)
        msg.info("builder.joint_coord_count: %s", builder.joint_coord_count)
        msg.info("builder.joint_dof_count: %s", builder.joint_dof_count)
        msg.info("builder.joint_key: %s", builder.joint_key)
        msg.info("builder.joint_world: %s", builder.joint_world)
        msg.info("builder.joint_parent: %s", builder.joint_parent)
        msg.info("builder.joint_child: %s", builder.joint_child)
        msg.info("builder.articulation_count: %s", builder.articulation_count)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
