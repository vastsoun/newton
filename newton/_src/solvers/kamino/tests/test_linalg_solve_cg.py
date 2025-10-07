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

"""Unit tests for the LLTSequentialSolver from linalg/linear.py"""

import unittest

import numpy as np
import warp as wp

from ..core.types import float32
from ..linalg.conjugate import ConjugateGradientSolver
from ..linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from .utils.extract import get_vector_block
from .utils.print import print_error_stats
from .utils.random import RandomProblemLLT


class TestLinalgCG(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.device = wp.get_device()
        self.verbose = True

    def tearDown(self):
        pass

    def test_sm_linear_operator(self):
        pass

    def test_solve(self):
        problems = {
            "small_full": {"maxdims": [4, 7], "dims": [4, 7]},
            "small_partial": {"maxdims": [20, 23], "dims": [14, 11]},
            "large_partial": {"maxdims": [16, 64, 128, 512, 1024], "dims": [11, 51, 101, 376, 999]},
        }
        for problem_name, problem_params in problems.items():
            with self.subTest(problem=problem_name):
                problem = RandomProblemLLT(
                    **problem_params,
                    seed=self.seed,
                    np_dtype=np.float32,
                    wp_dtype=float32,
                    device=self.device,
                )

                # Create the linear operator meta-data
                opinfo = DenseSquareMultiLinearInfo()
                opinfo.allocate(dimensions=problem.maxdims, dtype=problem.wp_dtype, device=self.device)
                opinfo.dim = wp.array(problem_params["dims"], dtype=opinfo.dim.dtype)
                if self.verbose:
                    print("opinfo:\n")
                    print(opinfo)
                b_np = problem.b_wp.numpy()
                b_np[problem.dims[0] :] = 0
                x_wp = wp.zeros_like(problem.b_wp, device=self.device)

                operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
                cg_solver = ConjugateGradientSolver(operator, device=self.device)
                cg_solver.compute(problem.A_wp)
                cg_solver.solve(problem.b_wp, x_wp, zero_x=True)

                x_wp_np = x_wp.numpy()

                for block_idx, block_act in enumerate(problem.dims):
                    x_found = get_vector_block(block_idx, x_wp_np, problem.dims, problem.maxdims)[:block_act]
                    is_x_close = np.allclose(x_found, problem.x_np[block_idx][:block_act], rtol=1e-3, atol=1e-4)
                    if self.verbose and sum(problem_params["maxdims"]) < 20:
                        print("x:")
                        print(x_found)
                        print("x_goal:")
                        print(problem.x_np[block_idx])
                        print_error_stats("x", x_found, problem.x_np[block_idx], problem.dims[block_idx])
                    self.assertTrue(is_x_close)


if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, precision=10, threshold=20000, suppress=True)  # Suppress scientific notation

    # Initialize Warp
    wp.init()

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False

    # Clear caches
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
