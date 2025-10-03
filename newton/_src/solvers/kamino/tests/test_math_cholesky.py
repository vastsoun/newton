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

"""Unit tests for LLT (i.e. Cholesky) factorization and solve operations."""

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.linalg.cholesky import (
    cholesky_blocked_factorize,
    make_cholesky_blocked_factorize_kernel,
)
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.tests.utils.random import RandomProblemLLT

###
# Tests
###


class TestMathCholesky(unittest.TestCase):
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

    # def test_01_single_cholesky_sequential_factorize_kernel(self):
    #     """
    #     Test the sequential Cholesky factorization kernel on a single problem.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=1000,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_sequential_factorize(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         A=problem.A_wp,
    #         L=L_wp,
    #     )

    #     # Check results for the single block
    #     L_wp_np = L_wp.numpy().reshape((problem.dims[0], problem.dims[0]))
    #     if self.verbose:
    #         print(f"L_np ({problem.X_np[0].shape}):\n{problem.X_np[0]}")
    #         print(f"L_wp ({L_wp_np.shape}):\n{L_wp_np}")

    #     # Check matrix factorization against numpy
    #     is_L_close = np.allclose(L_wp_np, problem.X_np[0], rtol=1e-4, atol=1e-6)
    #     if not is_L_close or self.verbose:
    #         print_error_stats("L", L_wp_np, problem.X_np[0], problem.dims[0])
    #     self.assertTrue(is_L_close)

    #     # Reconstruct the original matrix A from the factorization
    #     A_wp_np = L_wp_np @ L_wp_np.T
    #     if self.verbose:
    #         print(f"A_np ({problem.A_np[0].shape}):\n{problem.A_np[0]}")
    #         print(f"A_wp ({A_wp_np.shape}):\n{A_wp_np}")

    #     # Check matrix reconstruction against original matrix
    #     is_A_close = np.allclose(A_wp_np, problem.A_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_A_close or self.verbose:
    #         print_error_stats("A", A_wp_np, problem.A_np[0], problem.dims[0])
    #     self.assertTrue(is_A_close)

    # def test_02_multiple_cholesky_sequential_factorize_kernel(self):
    #     """
    #     Test the sequential Cholesky factorization kernel on multiple problems.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=[16, 64, 128, 512, 1024],
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_sequential_factorize(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         A=problem.A_wp,
    #         L=L_wp,
    #     )

    #     # Convert the warp array to numpy for verification
    #     L_wp_np = L_wp.numpy()

    #     # Check results for each block
    #     for i in range(problem.num_blocks):
    #         start_A = problem.mio_wp.numpy()[i]
    #         dim_A = problem.dim_wp.numpy()[i]
    #         size_A = dim_A * dim_A
    #         end_A = start_A + size_A
    #         L_wp_np_i = L_wp_np[start_A:end_A].reshape((problem.dims[i], problem.dims[i]))
    #         if self.verbose:
    #             print(f"[{i}]: L_np ({problem.X_np[i].shape}):\n{problem.X_np[i]}")
    #             print(f"[{i}]: L_wp ({L_wp_np_i.shape}):\n{L_wp_np_i}")

    #         # Check matrix factorization against numpy
    #         is_L_close = np.allclose(L_wp_np_i, problem.X_np[i], rtol=1e-4, atol=1e-6)
    #         if not is_L_close or self.verbose:
    #             print_error_stats(f"L[{i}]", L_wp_np_i, problem.X_np[i], problem.dims[i])
    #         self.assertTrue(is_L_close)

    #         # Reconstruct the original matrix A from the factorization
    #         A_wp_np_i = L_wp_np_i @ L_wp_np_i.T
    #         if self.verbose:
    #             print(f"[{i}]: A_np ({problem.A_np[i].shape}):\n{problem.A_np[i]}")
    #             print(f"[{i}]: A_wp ({A_wp_np_i.shape}):\n{A_wp_np_i}")

    #         # Check matrix reconstruction against original matrix
    #         is_A_close = np.allclose(A_wp_np_i, problem.A_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_A_close or self.verbose:
    #             print_error_stats(f"A[{i}]", A_wp_np_i, problem.A_np[i], problem.dims[i])
    #         self.assertTrue(is_A_close)

    # def test_03_single_cholesky_sequential_solve_forward_backward_kernels(self):
    #     """
    #     Test the sequential Cholesky forward-backward solve kernels on a single problem.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=1000,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
    #     y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
    #     x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_sequential_factorize(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         A=problem.A_wp,
    #         L=L_wp,
    #     )

    #     # Solve the system using the warp-based Cholesky solve
    #     cholesky_sequential_solve_forward(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         vio=problem.vio_wp,
    #         L=L_wp,
    #         b=problem.b_wp,
    #         y=y_wp,
    #     )

    #     # Solve the backward system using the warp-based Cholesky solve
    #     cholesky_sequential_solve_backward(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         vio=problem.vio_wp,
    #         L=L_wp,
    #         y=y_wp,
    #         x=x_wp,
    #     )

    #     # Convert the warp arrays to numpy for verification
    #     y_wp_np = y_wp.numpy()
    #     x_wp_np = x_wp.numpy()

    #     # Check results for the single block
    #     if self.verbose:
    #         print(f"y_np ({problem.y_np[0].shape}):\n{problem.y_np[0]}")
    #         print(f"y_wp ({y_wp_np.shape}):\n{y_wp_np}")

    #     # Assert the result is as expected
    #     is_y_close = np.allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_y_close and self.verbose:
    #         print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
    #     self.assertTrue(is_y_close)

    #     # Check results for the single block
    #     if self.verbose:
    #         print(f"x_np ({problem.x_np[0].shape}):\n{problem.x_np[0]}")
    #         print(f"x_wp ({x_wp_np.shape}):\n{x_wp_np}")

    #     # Assert the result is as expected
    #     is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_x_close and self.verbose:
    #         print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
    #     self.assertTrue(is_x_close)

    # def test_04_multiple_cholesky_sequential_solve_forward_backward_kernels(self):
    #     """
    #     Test the sequential Cholesky forward-backward solve kernels on multiple problems.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=[16, 64, 128, 512, 1024],
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
    #     y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
    #     x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_sequential_factorize(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         A=problem.A_wp,
    #         L=L_wp,
    #     )

    #     # Solve the forward system using the warp-based Cholesky solve
    #     cholesky_sequential_solve_forward(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         vio=problem.vio_wp,
    #         L=L_wp,
    #         b=problem.b_wp,
    #         y=y_wp,
    #     )

    #     # Solve the backward system using the warp-based Cholesky solve
    #     cholesky_sequential_solve_backward(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         vio=problem.vio_wp,
    #         L=L_wp,
    #         y=y_wp,
    #         x=x_wp,
    #     )

    #     # Convert the warp arrays to numpy for verification
    #     y_wp_np = y_wp.numpy()
    #     x_wp_np = x_wp.numpy()

    #     # Check results for each block
    #     for i in range(problem.num_blocks):
    #         dim_i = problem.dim_wp.numpy()[i]
    #         vio_i = problem.vio_wp.numpy()[i]
    #         y_wp_np_i = y_wp_np[vio_i : vio_i + dim_i]
    #         x_wp_np_i = x_wp_np[vio_i : vio_i + dim_i]

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: y_np ({problem.y_np[i].shape}):\n{problem.y_np[i]}")
    #             print(f"[{i}]: y_wp ({y_wp_np_i.shape}):\n{y_wp_np_i}")

    #         # Assert the result is as expected
    #         is_y_close = np.allclose(y_wp_np_i, problem.y_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_y_close and self.verbose:
    #             print_error_stats(f"y[{i}]", y_wp_np_i, problem.y_np[i], problem.dims[i])
    #         self.assertTrue(is_y_close)

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: x_np ({problem.x_np[i].shape}):\n{problem.x_np[i]}")
    #             print(f"[{i}]: x_wp ({x_wp_np_i.shape}):\n{x_wp_np_i}")

    #         # Assert the result is as expected
    #         is_x_close = np.allclose(x_wp_np_i, problem.x_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_x_close and self.verbose:
    #             print_error_stats(f"x[{i}]", x_wp_np_i, problem.x_np[i], problem.dims[i])
    #         self.assertTrue(is_x_close)

    # def test_05_single_cholesky_sequential_solve_kernel(self):
    #     """
    #     Test the sequential Cholesky forward-backward solve kernels on a single problem.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=1000,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
    #     y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
    #     x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_sequential_factorize(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         A=problem.A_wp,
    #         L=L_wp,
    #     )

    #     # Solve the system using the warp-based Cholesky solve
    #     cholesky_sequential_solve(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         vio=problem.vio_wp,
    #         L=L_wp,
    #         b=problem.b_wp,
    #         y=y_wp,
    #         x=x_wp,
    #     )

    #     # Convert the warp arrays to numpy for verification
    #     y_wp_np = y_wp.numpy()
    #     x_wp_np = x_wp.numpy()

    #     # Check results for the single block
    #     if self.verbose:
    #         print(f"y_np ({problem.y_np[0].shape}):\n{problem.y_np[0]}")
    #         print(f"y_wp ({y_wp_np.shape}):\n{y_wp_np}")

    #     # Assert the result is as expected
    #     is_y_close = np.allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_y_close and self.verbose:
    #         print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
    #     self.assertTrue(is_y_close)

    #     # Check results for the single block
    #     if self.verbose:
    #         print(f"x_np ({problem.x_np[0].shape}):\n{problem.x_np[0]}")
    #         print(f"x_wp ({x_wp_np.shape}):\n{x_wp_np}")

    #     # Assert the result is as expected
    #     is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_x_close and self.verbose:
    #         print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
    #     self.assertTrue(is_x_close)

    # def test_06_multiple_cholesky_sequential_solve_kernel(self):
    #     """
    #     Test the sequential Cholesky forward-backward solve kernels on multiple problems.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=[16, 64, 128, 512, 1024],
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
    #     y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
    #     x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_sequential_factorize(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         A=problem.A_wp,
    #         L=L_wp,
    #     )

    #     # Solve the forward system using the warp-based Cholesky solve
    #     cholesky_sequential_solve(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         vio=problem.vio_wp,
    #         L=L_wp,
    #         b=problem.b_wp,
    #         y=y_wp,
    #         x=x_wp,
    #     )

    #     # Convert the warp arrays to numpy for verification
    #     y_wp_np = y_wp.numpy()
    #     x_wp_np = x_wp.numpy()

    #     # Check results for each block
    #     for i in range(problem.num_blocks):
    #         dim_i = problem.dim_wp.numpy()[i]
    #         vio_i = problem.vio_wp.numpy()[i]
    #         y_wp_np_i = y_wp_np[vio_i : vio_i + dim_i]
    #         x_wp_np_i = x_wp_np[vio_i : vio_i + dim_i]

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: y_np ({problem.y_np[i].shape}):\n{problem.y_np[i]}")
    #             print(f"[{i}]: y_wp ({y_wp_np_i.shape}):\n{y_wp_np_i}")

    #         # Assert the result is as expected
    #         is_y_close = np.allclose(y_wp_np_i, problem.y_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_y_close and self.verbose:
    #             print_error_stats(f"y[{i}]", y_wp_np_i, problem.y_np[i], problem.dims[i])
    #         self.assertTrue(is_y_close)

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: x_np ({problem.x_np[i].shape}):\n{problem.x_np[i]}")
    #             print(f"[{i}]: x_wp ({x_wp_np_i.shape}):\n{x_wp_np_i}")

    #         # Assert the result is as expected
    #         is_x_close = np.allclose(x_wp_np_i, problem.x_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_x_close and self.verbose:
    #             print_error_stats(f"x[{i}]", x_wp_np_i, problem.x_np[i], problem.dims[i])
    #         self.assertTrue(is_x_close)

    # def test_07_single_cholesky_sequential_solve_inplace_kernel(self):
    #     """
    #     Test the sequential Cholesky forward-backward solve kernels on a single problem.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=1000,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
    #     x_wp = wp.empty_like(problem.b_wp, device=self.default_device)
    #     wp.copy(x_wp, problem.b_wp)  # Copy b into x for inplace solve

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_sequential_factorize(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         A=problem.A_wp,
    #         L=L_wp,
    #     )

    #     # Solve the system using the warp-based Cholesky solve
    #     cholesky_sequential_solve_inplace(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         vio=problem.vio_wp,
    #         L=L_wp,
    #         x=x_wp,
    #     )

    #     # Check results for the single block
    #     x_wp_np = x_wp.numpy()
    #     if self.verbose:
    #         print(f"x_np ({problem.x_np[0].shape}):\n{problem.x_np[0]}")
    #         print(f"x_wp ({x_wp_np.shape}):\n{x_wp_np}")

    #     # Assert the result is as expected
    #     is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_x_close and self.verbose:
    #         print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
    #     self.assertTrue(is_x_close)

    # def test_08_multiple_cholesky_sequential_solve_inplace_kernel(self):
    #     """
    #     Test the sequential Cholesky forward-backward solve kernels on multiple problems.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=[16, 64, 128, 512, 1024],
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
    #     x_wp = wp.empty_like(problem.b_wp, device=self.default_device)
    #     wp.copy(x_wp, problem.b_wp)  # Copy b into x for inplace solve

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_sequential_factorize(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         A=problem.A_wp,
    #         L=L_wp,
    #     )

    #     # Solve the forward system using the warp-based Cholesky solve
    #     cholesky_sequential_solve_inplace(
    #         num_blocks=problem.num_blocks,
    #         maxdim=problem.maxdim_wp,
    #         dim=problem.dim_wp,
    #         mio=problem.mio_wp,
    #         vio=problem.vio_wp,
    #         L=L_wp,
    #         x=x_wp,
    #     )

    #     # Convert the warp arrays to numpy for verification
    #     x_wp_np = x_wp.numpy()

    #     # Check results for each block
    #     for i in range(problem.num_blocks):
    #         dim_i = problem.dim_wp.numpy()[i]
    #         vio_i = problem.vio_wp.numpy()[i]
    #         x_wp_np_i = x_wp_np[vio_i : vio_i + dim_i]

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: x_np ({problem.x_np[i].shape}):\n{problem.x_np[i]}")
    #             print(f"[{i}]: x_wp ({x_wp_np_i.shape}):\n{x_wp_np_i}")

    #         # Assert the result is as expected
    #         is_x_close = np.allclose(x_wp_np_i, problem.x_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_x_close and self.verbose:
    #             print_error_stats(f"x[{i}]", x_wp_np_i, problem.x_np[i], problem.dims[i])
    #         self.assertTrue(is_x_close)

    # def test_09_single_cholesky_sequential_factorizer(self):
    #     """
    #     Test the sequential Cholesky forward-backward solve kernels on a single problem.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=1000,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays for the solution vector
    #     x_wp = wp.empty_like(problem.b_wp, device=self.default_device)

    #     # Create a SequentialCholeskyFactorizer instance
    #     factorizer = SequentialCholeskyFactorizer(dims=problem.dims, device=self.default_device)

    #     # Factorize the matrix A
    #     factorizer.factorize(A=problem.A_wp)

    #     # Solve the system using the factorized matrix
    #     factorizer.solve(b=problem.b_wp, x=x_wp)

    #     # Check results for the single block
    #     x_wp_np = x_wp.numpy()
    #     if self.verbose:
    #         print(f"x_np ({problem.x_np[0].shape}):\n{problem.x_np[0]}")
    #         print(f"x_wp ({x_wp_np.shape}):\n{x_wp_np}")

    #     # Assert the result is as expected
    #     is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_x_close and self.verbose:
    #         print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
    #     self.assertTrue(is_x_close)

    #     # Clear the solution vector and initialize it with b
    #     x_wp.zero_()
    #     wp.copy(x_wp, problem.b_wp)

    #     # Solve the system in-place using the factorized matrix
    #     factorizer.solve_inplace(x=x_wp)

    #     # Check results for the single block
    #     x_wp_np = x_wp.numpy()
    #     if self.verbose:
    #         print(f"x_np ({problem.x_np[0].shape}):\n{problem.x_np[0]}")
    #         print(f"x_wp ({x_wp_np.shape}):\n{x_wp_np}")

    #     # Assert the result is as expected
    #     is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_x_close and self.verbose:
    #         print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
    #     self.assertTrue(is_x_close)

    # def test_10_multiple_cholesky_sequential_factorizer(self):
    #     """
    #     Test the sequential Cholesky forward-backward solve kernels on multiple problems.
    #     """
    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=[16, 64, 128, 512, 1024],
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays for the solution vector
    #     x_wp = wp.empty_like(problem.b_wp, device=self.default_device)

    #     # Create a SequentialCholeskyFactorizer instance
    #     factorizer = SequentialCholeskyFactorizer(dims=problem.dims, device=self.default_device)

    #     # Factorize the matrix A
    #     factorizer.factorize(A=problem.A_wp)

    #     # Solve the system using the factorized matrix
    #     factorizer.solve(b=problem.b_wp, x=x_wp)

    #     # Convert the warp arrays to numpy for verification
    #     x_wp_np = x_wp.numpy()

    #     # Check results for each block
    #     for i in range(problem.num_blocks):
    #         dim_i = problem.dim_wp.numpy()[i]
    #         vio_i = problem.vio_wp.numpy()[i]
    #         x_wp_np_i = x_wp_np[vio_i : vio_i + dim_i]

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: x_np ({problem.x_np[i].shape}):\n{problem.x_np[i]}")
    #             print(f"[{i}]: x_wp ({x_wp_np_i.shape}):\n{x_wp_np_i}")

    #         # Assert the result is as expected
    #         is_x_close = np.allclose(x_wp_np_i, problem.x_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_x_close and self.verbose:
    #             print_error_stats(f"x[{i}]", x_wp_np_i, problem.x_np[i], problem.dims[i])
    #         self.assertTrue(is_x_close)

    #     # Clear the solution vector and initialize it with b
    #     x_wp.zero_()
    #     wp.copy(x_wp, problem.b_wp)

    #     # Solve the system in-place using the factorized matrix
    #     factorizer.solve_inplace(x=x_wp)

    #     # Check results for each block
    #     for i in range(problem.num_blocks):
    #         dim_i = problem.dim_wp.numpy()[i]
    #         vio_i = problem.vio_wp.numpy()[i]
    #         x_wp_np_i = x_wp_np[vio_i : vio_i + dim_i]

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: x_np ({problem.x_np[i].shape}):\n{problem.x_np[i]}")
    #             print(f"[{i}]: x_wp ({x_wp_np_i.shape}):\n{x_wp_np_i}")

    #         # Assert the result is as expected
    #         is_x_close = np.allclose(x_wp_np_i, problem.x_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_x_close and self.verbose:
    #             print_error_stats(f"x[{i}]", x_wp_np_i, problem.x_np[i], problem.dims[i])
    #         self.assertTrue(is_x_close)

    def test_11_single_cholesky_blocked_factorize_kernel(self):
        """
        Tests the blocked Cholesky factorization kernel on a single problem.
        """
        # Constants
        N = 12
        block_size = 4  # Block size for the blocked factorization
        block_dim = 128

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

        # Construct warp arrays
        L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)

        # Create the blocked factorization kernel
        factorize_kernel = make_cholesky_blocked_factorize_kernel(block_size)

        # Determine the necessary reshaping for the matrix A
        mat_shape = (problem.dims[0], problem.dims[0])

        # Compute the warp-based sequential Cholesky decomposition
        cholesky_blocked_factorize(
            num_blocks=problem.num_blocks,
            dim=problem.dim_wp,
            rio=problem.vio_wp,
            A=problem.A_wp.reshape(mat_shape),
            L=L_wp.reshape(mat_shape),
            kernel=factorize_kernel,
            block_dim=block_dim,
        )

        # Convert the warp arrays to numpy for verification
        L_wp_np = L_wp.numpy().reshape(mat_shape)

        # Check results for the single block
        if self.verbose:
            msg.info("L_np (%s):\n%s\n", problem.X_np[0].shape, problem.X_np[0])
            msg.info("L_wp (%s):\n%s\n", L_wp_np.shape, L_wp_np)

        # Check matrix factorization against numpy
        is_L_close = np.allclose(L_wp_np, problem.X_np[0], rtol=1e-4, atol=1e-6)
        if not is_L_close and self.verbose:
            print_error_stats("L", L_wp_np, problem.X_np[0], problem.dims[0])
        self.assertTrue(is_L_close)

        # Reconstruct the original matrix A from the factorization
        A_rec_wp_np = L_wp_np @ L_wp_np.T
        if self.verbose:
            msg.info("A_np (%s):\n%s\n", problem.A_np[0].shape, problem.A_np[0])
            msg.info("A_rec_wp_np (%s):\n%s\n", A_rec_wp_np.shape, A_rec_wp_np)

        # Check matrix reconstruction against original matrix
        is_A_close = np.allclose(A_rec_wp_np, problem.A_np[0], rtol=1e-3, atol=1e-4)
        if not is_A_close and self.verbose:
            print_error_stats("A", A_rec_wp_np, problem.A_np[0], problem.dims[0])
        self.assertTrue(is_A_close)

    # def test_12_multiple_cholesky_blocked_factorize_kernel(self):
    #     """
    #     Test the blocked Cholesky factorization kernel on multiple problems.
    #     """
    #     # Constants
    #     mbdim = 20
    #     dims = [mbdim] * 5  # Dimensions for each matrix block
    #     block_size = 16  # Block size for the blocked factorization
    #     block_dim = 128

    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=dims,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)

    #     # Create the blocked factorization kernel
    #     factorize_kernel = make_cholesky_blocked_factorize_kernel(block_size)

    #     # Determine the necessary reshaping for the matrix A
    #     mbsize = problem.A_wp.size
    #     mbrows = mbsize // mbdim
    #     mat_shape = (mbrows, mbdim)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_blocked_factorize(
    #         num_blocks=problem.num_blocks,
    #         dim=problem.dim_wp,
    #         rio=problem.vio_wp,
    #         A=problem.A_wp.reshape(mat_shape),
    #         L=L_wp.reshape(mat_shape),
    #         kernel=factorize_kernel,
    #         block_dim=block_dim,
    #     )

    #     # Convert the warp arrays to numpy for verification
    #     L_wp_np = L_wp.numpy()

    #     # Check results for the single block
    #     for i in range(problem.num_blocks):
    #         # Extract the individual factorization block
    #         start_A = problem.mio_wp.numpy()[i]
    #         dim_A = problem.dim_wp.numpy()[i]
    #         size_A = dim_A * dim_A
    #         end_A = start_A + size_A
    #         L_wp_np_i = L_wp_np[start_A:end_A].reshape((problem.dims[i], problem.dims[i]))
    #         if self.verbose:
    #             print(f"[{i}]: L_np ({problem.X_np[i].shape}):\n{problem.X_np[i]}")
    #             print(f"[{i}]: L_wp ({L_wp_np_i.shape}):\n{L_wp_np_i}")

    #         # Check matrix factorization against numpy
    #         is_L_close = np.allclose(L_wp_np_i, problem.X_np[i], rtol=1e-4, atol=1e-6)
    #         if not is_L_close and self.verbose:
    #             print_error_stats(f"L[{i}]", L_wp_np_i, problem.X_np[i], problem.dims[i])
    #         self.assertTrue(is_L_close)

    #         # Reconstruct the original matrix A from the factorization
    #         A_wp_np = L_wp_np_i @ L_wp_np_i.T
    #         if self.verbose:
    #             print(f"[{i}]: A_np ({problem.A_np[i].shape}):\n{problem.A_np[i]}")
    #             print(f"[{i}]: A_wp ({A_wp_np.shape}):\n{A_wp_np}")

    #         # Check matrix reconstruction against original matrix
    #         is_A_close = np.allclose(A_wp_np, problem.A_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_A_close and self.verbose:
    #             print_error_stats(f"A[{i}]", A_wp_np, problem.A_np[i], problem.dims[i])
    #         self.assertTrue(is_A_close)

    # def test_13_single_cholesky_blocked_solve_kernel(self):
    #     """
    #     Tests the blocked Cholesky factorization kernel on a single problem.
    #     """
    #     # Constants
    #     block_size = 16  # Block size for the blocked factorization
    #     block_dim = 128

    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=10,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
    #     y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
    #     x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

    #     # Create the blocked factorization and solve kernels
    #     factorize_kernel = make_cholesky_blocked_factorize_kernel(block_size)
    #     solve_kernel = make_cholesky_blocked_solve_kernel(block_size)

    #     # Determine the necessary reshaping for the matrix A
    #     mat_shape = (problem.dims[0], problem.dims[0])
    #     vec_shape = (problem.dims[0], 1)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_blocked_factorize(
    #         num_blocks=problem.num_blocks,
    #         dim=problem.dim_wp,
    #         rio=problem.vio_wp,
    #         A=problem.A_wp.reshape(mat_shape),
    #         L=L_wp.reshape(mat_shape),
    #         kernel=factorize_kernel,
    #         block_dim=block_dim,
    #     )

    #     # Solve the system using the warp-based Cholesky solve
    #     cholesky_blocked_solve(
    #         num_blocks=problem.num_blocks,
    #         dim=problem.dim_wp,
    #         rio=problem.vio_wp,
    #         L=L_wp.reshape(mat_shape),
    #         b=problem.b_wp.reshape(vec_shape),
    #         y=y_wp.reshape(vec_shape),
    #         x=x_wp.reshape(vec_shape),
    #         kernel=solve_kernel,
    #         block_dim=block_dim,
    #     )

    #     # Convert the warp arrays to numpy for verification
    #     y_wp_np = y_wp.numpy()
    #     x_wp_np = x_wp.numpy()

    #     # Check results for the single block
    #     if self.verbose:
    #         print(f"y_np ({problem.y_np[0].shape}):\n{problem.y_np[0]}")
    #         print(f"y_wp ({y_wp_np.shape}):\n{y_wp_np}")

    #     # Assert the result is as expected
    #     is_y_close = np.allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_y_close and self.verbose:
    #         print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
    #     self.assertTrue(is_y_close)

    #     # Check results for the single block
    #     if self.verbose:
    #         print(f"x_np ({problem.x_np[0].shape}):\n{problem.x_np[0]}")
    #         print(f"x_wp ({x_wp_np.shape}):\n{x_wp_np}")

    #     # Assert the result is as expected
    #     is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_x_close and self.verbose:
    #         print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
    #     self.assertTrue(is_x_close)

    # def test_14_multiple_cholesky_blocked_solve_kernel(self):
    #     """
    #     Test the blocked Cholesky factorization kernel on multiple problems.
    #     """
    #     # Constants
    #     mbdim = 20
    #     dims = [mbdim] * 5  # Dimensions for each matrix block
    #     block_size = 16  # Block size for the blocked factorization
    #     block_dim = 128

    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=dims,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
    #     y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
    #     x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

    #     # Create the blocked factorization and solve kernels
    #     factorize_kernel = make_cholesky_blocked_factorize_kernel(block_size)
    #     solve_kernel = make_cholesky_blocked_solve_kernel(block_size)

    #     # Determine the necessary reshaping for the matrix A
    #     mbsize = problem.A_wp.size
    #     mbrows = mbsize // mbdim
    #     mat_shape = (mbrows, mbdim)
    #     vbsize = problem.b_wp.size
    #     vec_shape = (vbsize, 1)

    #     # Compute the warp-based sequential Cholesky decomposition
    #     cholesky_blocked_factorize(
    #         num_blocks=problem.num_blocks,
    #         dim=problem.dim_wp,
    #         rio=problem.vio_wp,
    #         A=problem.A_wp.reshape(mat_shape),
    #         L=L_wp.reshape(mat_shape),
    #         kernel=factorize_kernel,
    #         block_dim=block_dim,
    #     )

    #     # Solve the system using the warp-based Cholesky solve
    #     cholesky_blocked_solve(
    #         num_blocks=problem.num_blocks,
    #         dim=problem.dim_wp,
    #         rio=problem.vio_wp,
    #         L=L_wp.reshape(mat_shape),
    #         b=problem.b_wp.reshape(vec_shape),
    #         y=y_wp.reshape(vec_shape),
    #         x=x_wp.reshape(vec_shape),
    #         kernel=solve_kernel,
    #         block_dim=block_dim,
    #     )

    #     # Convert the warp arrays to numpy for verification
    #     y_wp_np = y_wp.numpy()
    #     x_wp_np = x_wp.numpy()

    #     # Check results for the single block
    #     for i in range(problem.num_blocks):
    #         dim_i = problem.dim_wp.numpy()[i]
    #         vio_i = problem.vio_wp.numpy()[i]
    #         y_wp_np_i = y_wp_np[vio_i : vio_i + dim_i]
    #         x_wp_np_i = x_wp_np[vio_i : vio_i + dim_i]

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: y_np ({problem.y_np[i].shape}):\n{problem.y_np[i]}")
    #             print(f"[{i}]: y_wp ({y_wp_np_i.shape}):\n{y_wp_np_i}")

    #         # Assert the result is as expected
    #         is_y_close = np.allclose(y_wp_np_i, problem.y_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_y_close and self.verbose:
    #             print_error_stats(f"y[{i}]", y_wp_np_i, problem.y_np[i], problem.dims[i])
    #         self.assertTrue(is_y_close)

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: x_np ({problem.x_np[i].shape}):\n{problem.x_np[i]}")
    #             print(f"[{i}]: x_wp ({x_wp_np_i.shape}):\n{x_wp_np_i}")

    #         # Assert the result is as expected
    #         is_x_close = np.allclose(x_wp_np_i, problem.x_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_x_close and self.verbose:
    #             print_error_stats(f"x[{i}]", x_wp_np_i, problem.x_np[i], problem.dims[i])
    #         self.assertTrue(is_x_close)

    # def test_15_single_cholesky_blocked_factorizer(self):
    #     """
    #     Tests the blocked Cholesky factorizer on a single problem.
    #     """
    #     # Constants
    #     block_size = 16  # Block size for the blocked factorization
    #     solve_block_dim = 64
    #     factortize_block_dim = 128

    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=10,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

    #     # Create the blocked cholesky factorizer
    #     factorizer = BlockedCholeskyFactorizer(
    #         dims=problem.dims,
    #         device=self.default_device,
    #         block_size=block_size,
    #         solve_block_dim=solve_block_dim,
    #         factortize_block_dim=factortize_block_dim,
    #     )

    #     # Factorize the matrix A and solve the linear system
    #     factorizer.factorize(A=problem.A_wp)
    #     factorizer.solve(b=problem.b_wp, x=x_wp)

    #     # Convert the warp arrays to numpy for verification
    #     x_wp_np = x_wp.numpy()

    #     # Check results for the single block
    #     if self.verbose:
    #         print(f"x_np ({problem.x_np[0].shape}):\n{problem.x_np[0]}")
    #         print(f"x_wp ({x_wp_np.shape}):\n{x_wp_np}")

    #     # Assert the result is as expected
    #     is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
    #     if not is_x_close and self.verbose:
    #         print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
    #     self.assertTrue(is_x_close)

    #     # # Reset the solution vector and solve in-place
    #     # x_wp.zero_()
    #     # wp.copy(x_wp, problem.b_wp)

    #     # # Solve the system in-place using the factorized matrix
    #     # factorizer.solve_inplace(x=x_wp)

    #     # # Convert the warp arrays to numpy for verification
    #     # x_wp_np = x_wp.numpy()

    #     # # Check results for the single block
    #     # if self.verbose:
    #     #     print(f"x_np ({problem.x_np[0].shape}):\n{problem.x_np[0]}")
    #     #     print(f"x_wp ({x_wp_np.shape}):\n{x_wp_np}")

    #     # # Assert the result is as expected
    #     # is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
    #     # if not is_x_close and self.verbose:
    #     #     print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
    #     # self.assertTrue(is_x_close)

    # def test_16_multiple_cholesky_blocked_factorizer(self):
    #     """
    #     Test the blocked Cholesky factorizer on multiple problems.
    #     """
    #     # Constants
    #     mbdim = 10
    #     dims = [mbdim] * 3  # Dimensions for each matrix block
    #     block_size = 16  # Block size for the blocked factorization
    #     solve_block_dim = 64
    #     factortize_block_dim = 128

    #     # Create a single-instance problem
    #     problem = RandomProblemCholesky(
    #         dims=dims,
    #         seed=self.seed,
    #         np_dtype=np.float32,
    #         wp_dtype=float32,
    #         device=self.default_device,
    #         verbose=self.verbose,
    #     )

    #     # Construct warp arrays
    #     x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

    #     # Create the blocked cholesky factorizer
    #     factorizer = BlockedCholeskyFactorizer(
    #         dims=problem.dims,
    #         device=self.default_device,
    #         block_size=block_size,
    #         solve_block_dim=solve_block_dim,
    #         factortize_block_dim=factortize_block_dim,
    #     )

    #     # Factorize the matrix A and solve the linear system
    #     factorizer.factorize(A=problem.A_wp)
    #     factorizer.solve(b=problem.b_wp, x=x_wp)

    #     # Convert the warp arrays to numpy for verification
    #     x_wp_np = x_wp.numpy()

    #     # Check results for the single block
    #     for i in range(problem.num_blocks):
    #         dim_i = problem.dim_wp.numpy()[i]
    #         vio_i = problem.vio_wp.numpy()[i]
    #         x_wp_np_i = x_wp_np[vio_i : vio_i + dim_i]

    #         # Check results for the single block
    #         if self.verbose:
    #             print(f"[{i}]: x_np ({problem.x_np[i].shape}):\n{problem.x_np[i]}")
    #             print(f"[{i}]: x_wp ({x_wp_np_i.shape}):\n{x_wp_np_i}")

    #         # Assert the result is as expected
    #         is_x_close = np.allclose(x_wp_np_i, problem.x_np[i], rtol=1e-3, atol=1e-4)
    #         if not is_x_close and self.verbose:
    #             print_error_stats(f"x[{i}]", x_wp_np_i, problem.x_np[i], problem.dims[i])
    #         self.assertTrue(is_x_close)

    #     # # Reset the solution vector and solve in-place
    #     # x_wp.zero_()
    #     # wp.copy(x_wp, problem.b_wp)

    #     # # Solve the system in-place using the factorized matrix
    #     # factorizer.solve_inplace(x=x_wp)

    #     # # Convert the warp arrays to numpy for verification
    #     # x_wp_np = x_wp.numpy()
    #     # print(f"x_wp_np ({x_wp_np.shape}):\n{x_wp_np}")
    #     # L_wp_np = factorizer.L.numpy()

    #     # # Check results for the single block
    #     # for i in range(problem.num_blocks):
    #     #     # Extract the individual factorization block
    #     #     start_A = problem.mio_wp.numpy()[i]
    #     #     dim_A = problem.dim_wp.numpy()[i]
    #     #     size_A = dim_A * dim_A
    #     #     end_A = start_A + size_A
    #     #     L_wp_np_i = L_wp_np[start_A:end_A].reshape((problem.dims[i], problem.dims[i]))
    #     #     if self.verbose:
    #     #         print(f"[{i}]: L_np ({problem.X_np[i].shape}):\n{problem.X_np[i]}")
    #     #         print(f"[{i}]: L_wp ({L_wp_np_i.shape}):\n{L_wp_np_i}")

    #     #     # Extract the individual solution block
    #     #     dim_i = problem.dim_wp.numpy()[i]
    #     #     vio_i = problem.vio_wp.numpy()[i]
    #     #     x_wp_np_i = x_wp_np[vio_i:vio_i + dim_i]

    #     #     # Check results for the single block
    #     #     if self.verbose:
    #     #         print(f"[{i}]: x_np ({problem.x_np[i].shape}):\n{problem.x_np[i]}")
    #     #         print(f"[{i}]: x_wp ({x_wp_np_i.shape}):\n{x_wp_np_i}")

    #     #     # Assert the result is as expected
    #     #     is_x_close = np.allclose(x_wp_np_i, problem.x_np[i], rtol=1e-3, atol=1e-4)
    #     #     if not is_x_close and self.verbose:
    #     #         print_error_stats(f"x[{i}]", x_wp_np_i, problem.x_np[i], problem.dims[i])
    #     #     self.assertTrue(is_x_close)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
