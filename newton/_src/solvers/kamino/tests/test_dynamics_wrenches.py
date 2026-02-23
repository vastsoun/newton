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
Unit tests for `dynamics/wrenches.py`.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core import ModelBuilder
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.types import float32, int32, mat33f, transformf, vec3f, vec6f
from newton._src.solvers.kamino.dynamics.wrenches import (
    compute_constraint_body_wrenches_sparse,
    compute_constraint_body_wrenches_dense,
)
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.detector import CollisionDetector, CollisionDetectorSettings
from newton._src.solvers.kamino.kinematics.constraints import make_unilateral_constraints_info, update_constraints_info
from newton._src.solvers.kamino.kinematics.jacobians import (
    DenseSystemJacobians,
    SparseSystemJacobians,
)
from newton._src.solvers.kamino.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.models.builders.basics import (
    build_box_on_plane,
    build_boxes_fourbar,
    build_boxes_hinged,
    build_boxes_nunchaku,
    build_cartpole,
    make_basics_heterogeneous_builder,
)
from newton._src.solvers.kamino.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.print import (
    print_model_constraint_info,
    print_model_data_info,
)
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.tests.test_kinematics_jacobians import (
    extract_cts_jacobians,
    extract_dofs_jacobians,
    set_fourbar_body_states,
)

###
# Tests
###


class TestDynamicsWrenches(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = True  # Set to True for verbose output

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

    def test_compute_constraint_body_wrenches_box_on_plane(self):
        pass  # Placeholder for actual test implementation


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
