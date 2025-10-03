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

"""Unit tests for the LLT solvers in linalg/linear.py"""

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino.linalg.linear import LLTSequentialSolver
from newton._src.solvers.kamino.tests.utils.random import RandomProblemLLT

###
# Tests
###


class TestLinAlgLLTSequentialSolver(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = True  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default_solver(self):
        """
        Test the default constructor of the LLTSequentialSolver class.
        """
        llt = LLTSequentialSolver(device=self.default_device)
        self.assertIsNone(llt._operator)
        self.assertEqual(llt.dtype, float32)
        self.assertEqual(llt.device, self.default_device)

    def test_01_single_problem_dims_all_active(self):
        """
        Test the sequential LLT solver on a single small problem.
        """
        # Constants
        N = 12

        # Create a single-instance problem
        problem = RandomProblemLLT(
            dims=N,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=float32,
            device=self.default_device,
            verbose=self.verbose,
        )
        msg.debug("Problem:\n%s\n", problem)
        msg.debug("b_np:\n%s\n", problem.b_np[0])
        msg.debug("A_np:\n%s\n", problem.A_np[0])
        msg.debug("X_np:\n%s\n", problem.X_np[0])
        msg.debug("y_np:\n%s\n", problem.y_np[0])
        msg.debug("x_np:\n%s\n", problem.x_np[0])
        msg.info("A_wp:\n%s\n", problem.A_wp.numpy().reshape((problem.dims[0], problem.dims[0])))
        msg.info("b_wp:\n%s\n", problem.b_wp.numpy().reshape((problem.dims[0],)))

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.allocate(dimensions=problem.dims, dtype=problem.wp_dtype, device=self.default_device)
        msg.info("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        msg.info("operator.info:\n%s\n", operator.info)
        msg.info("operator.mat:\n%s\n", operator.mat.numpy().reshape((problem.dims[0], problem.dims[0])))

        # Create a SequentialCholeskyFactorizer instance
        llt = LLTSequentialSolver(operator=operator, device=self.default_device)
        self.assertIsNotNone(llt._operator)
        self.assertEqual(llt.dtype, problem.wp_dtype)
        self.assertEqual(llt.device, self.default_device)
        self.assertEqual(llt._L.size, problem.A_wp.size)
        self.assertEqual(llt._y.size, problem.b_wp.size)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=10000, precision=10, suppress=True)  # Suppress scientific notation

    # Initialize Warp
    wp.init()

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
