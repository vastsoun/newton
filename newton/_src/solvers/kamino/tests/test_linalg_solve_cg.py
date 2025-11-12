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

"""Unit tests for the CGSolver class from linalg/conjugate.py"""

import itertools
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.linalg.conjugate import (
    CGSolver,
    make_dense_square_matrix_operator,
)
from newton._src.solvers.kamino.tests.utils.extract import get_vector_block
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.tests.utils.random import RandomProblemLLT


class TestLinalgConjugate(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.device = wp.get_device()
        # self.device = "cpu"
        self.verbose = True

    def tearDown(self):
        pass

    def test_solve(self):
        problems = {
            "small_full": {"maxdims": 7, "dims": [4, 7]},
            "small_partial": {"maxdims": 23, "dims": [14, 11]},
            "large_partial": {"maxdims": 1024, "dims": [11, 51, 101, 376, 999]},
        }
        solvers = {
            "CG": CGSolver,
            # "CR": CRSolver,
        }
        for (problem_name, problem_params), (solver_name, solver) in itertools.product(
            problems.items(), solvers.items()
        ):
            with self.subTest(problem=problem_name, solver=solver_name):
                problem = RandomProblemLLT(
                    **problem_params,
                    seed=self.seed,
                    np_dtype=np.float32,
                    wp_dtype=float32,
                    device=self.device,
                )

                n_envs = problem.num_blocks
                maxdim = int(problem.maxdims[0])

                A_2d = problem.A_wp.reshape((n_envs, maxdim * maxdim))
                b_2d = problem.b_wp.reshape((n_envs, maxdim))
                x_wp = wp.zeros_like(b_2d, device=self.device)

                env_active = wp.full(n_envs, True, dtype=wp.bool, device=self.device)
                operator = make_dense_square_matrix_operator(
                    A=A_2d,
                    active_dims=problem.dim_wp,
                    max_dims=maxdim,
                    matrix_stride=maxdim,
                )

                atol_sq = wp.full(n_envs, 1.0e-16, dtype=problem.wp_dtype, device=self.device)
                cg_solver = solver(
                    A=operator,
                    active_dims=problem.dim_wp,
                    env_active=env_active,
                    atol_sq=atol_sq,
                    maxiter=None,
                    M=None,
                    callback=None,
                    check_every=0,
                    use_cuda_graph=False,
                )
                cg_solver.solve(b_2d, x_wp)

                x_wp_np = x_wp.numpy().reshape(-1)

                if self.verbose:
                    pass
                for block_idx, block_act in enumerate(problem.dims):
                    x_found = get_vector_block(block_idx, x_wp_np, problem.dims, problem.maxdims)[:block_act]
                    is_x_close = np.allclose(x_found, problem.x_np[block_idx][:block_act], rtol=1e-3, atol=1e-4)
                    if self.verbose:
                        if sum(problem.dims) < 20:
                            print("x:")
                            print(x_found)
                            print("x_goal:")
                            print(problem.x_np[block_idx])
                        print_error_stats("x", x_found, problem.x_np[block_idx], problem.dims[block_idx])
                    self.assertTrue(is_x_close)


if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, precision=10, threshold=20000, suppress=True)  # Suppress scientific notation

    wp.init()

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False

    # Clear caches
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
