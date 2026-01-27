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

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.linalg.conjugate import BatchedLinearOperator, CGSolver, CRSolver
from newton._src.solvers.kamino.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino.linalg.sparse import BlockDType, BlockSparseMatrices
from newton._src.solvers.kamino.linalg.utils.rand import random_spd_matrix
from newton._src.solvers.kamino.tests.utils.extract import get_vector_block
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.tests.utils.rand import RandomProblemLLT


class TestLinalgConjugate(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.verbose = True

    def tearDown(self):
        pass

    def _test_solve(self, solver_cls, problem_params, device):
        problem = RandomProblemLLT(
            **problem_params,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=float32,
            device=device,
        )

        n_worlds = problem.num_blocks
        maxdim = int(problem.maxdims[0])

        b_2d = problem.b_wp.reshape((n_worlds, maxdim))
        x_wp = wp.zeros_like(b_2d, device=device)

        world_active = wp.full(n_worlds, True, dtype=wp.bool, device=device)

        # Create operator - use maxdim for allocation, then set actual dims
        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[maxdim] * n_worlds, dtype=float32, device=device)
        info.dim = problem.dim_wp  # Override with actual active dimensions
        operator = DenseLinearOperatorData(info=info, mat=problem.A_wp)
        A = BatchedLinearOperator.from_dense(operator)

        atol = wp.full(n_worlds, 1.0e-8, dtype=problem.wp_dtype, device=device)
        rtol = wp.full(n_worlds, 1.0e-8, dtype=problem.wp_dtype, device=device)
        solver = solver_cls(
            A=A,
            world_active=world_active,
            atol=atol,
            rtol=rtol,
            maxiter=None,
            Mi=None,
            callback=None,
            use_cuda_graph=False,
        )
        cur_iter, r_norm_sq, atol_sq = solver.solve(b_2d, x_wp)

        x_wp_np = x_wp.numpy().reshape(-1)

        if self.verbose:
            pass
        for block_idx, block_act in enumerate(problem.dims):
            x_found = get_vector_block(block_idx, x_wp_np, problem.dims, problem.maxdims)[:block_act]
            is_x_close = np.allclose(x_found, problem.x_np[block_idx][:block_act], rtol=1e-3, atol=1e-4)
            if self.verbose:
                print(f"Cur iter: {cur_iter}")
                print(f"R norm sq {r_norm_sq}")
                print(f"Atol sq: {atol_sq}")
                if sum(problem.dims) < 20:
                    print("x:")
                    print(x_found)
                    print("x_goal:")
                    print(problem.x_np[block_idx])
                print_error_stats("x", x_found, problem.x_np[block_idx], problem.dims[block_idx])
            self.assertTrue(is_x_close)

    @classmethod
    def _problem_params(cls):
        problems = {
            "small_full": {"maxdims": 7, "dims": [4, 7]},
            "small_partial": {"maxdims": 23, "dims": [14, 11]},
            "large_partial": {"maxdims": 1024, "dims": [11, 51, 101, 376, 999]},
        }
        return problems

    def test_solve_cg_cpu(self):
        device = "cpu"
        self.skipTest("No CPU tests")
        solver_cls = CGSolver
        for problem_name, problem_params in self._problem_params().items():
            with self.subTest(problem=problem_name, solver=solver_cls.__name__):
                self._test_solve(solver_cls, problem_params, device)

    def test_solve_cr_cpu(self):
        device = "cpu"
        self.skipTest("No CPU tests")
        solver_cls = CRSolver
        for problem_name, problem_params in self._problem_params().items():
            with self.subTest(problem=problem_name, solver=solver_cls.__name__):
                self._test_solve(solver_cls, problem_params, device)

    def test_solve_cg_cuda(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()
        solver_cls = CGSolver
        for problem_name, problem_params in self._problem_params().items():
            with self.subTest(problem=problem_name, solver=solver_cls.__name__):
                self._test_solve(solver_cls, problem_params, device)

    def test_solve_cr_cuda(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()
        solver_cls = CRSolver
        for problem_name, problem_params in self._problem_params().items():
            with self.subTest(problem=problem_name, solver=solver_cls.__name__):
                self._test_solve(solver_cls, problem_params, device)

    def _test_sparse_solve(self, solver_cls, n_worlds, dim, block_size, device):
        """Test CG/CR with sparse matrices built from random SPD matrices."""
        rng = np.random.default_rng(self.seed)

        # Pad to block-aligned size
        n_blocks_per_dim = (dim + block_size - 1) // block_size
        padded_dim = n_blocks_per_dim * block_size
        total_blocks = n_blocks_per_dim * n_blocks_per_dim

        # Generate random SPD matrices and RHS vectors
        A_list, A_padded_list, b_list, x_ref_list = [], [], [], []
        for i in range(n_worlds):
            A = random_spd_matrix(dim=dim, seed=self.seed + i, dtype=np.float32)
            A_padded = np.zeros((padded_dim, padded_dim), dtype=np.float32)
            A_padded[:dim, :dim] = A
            b = rng.standard_normal(dim).astype(np.float32)
            A_list.append(A)
            A_padded_list.append(A_padded)
            b_list.append(b)
            x_ref_list.append(np.linalg.solve(A, b))

        # Block coordinates (all blocks, row-major) - same for all worlds
        coords = [(bi * block_size, bj * block_size) for bi in range(n_blocks_per_dim) for bj in range(n_blocks_per_dim)]
        all_coords = np.array(coords * n_worlds, dtype=np.int32)

        # Build BlockSparseMatrices
        bsm = BlockSparseMatrices()
        bsm.finalize(
            max_dims=[(padded_dim, padded_dim)] * n_worlds,
            capacities=[total_blocks] * n_worlds,
            nzb_dtype=BlockDType(float32, (block_size, block_size)),
            device=device,
        )
        bsm.dims.assign(np.array([[padded_dim, padded_dim]] * n_worlds, dtype=np.int32))
        bsm.num_nzb.assign(np.array([total_blocks] * n_worlds, dtype=np.int32))
        bsm.nzb_coords.assign(all_coords)
        bsm.assign(A_padded_list)

        # Build dense operator for comparison
        A_dense = np.array([A.flatten() for A in A_padded_list], dtype=np.float32)
        A_wp = wp.array(A_dense, dtype=float32, device=device)
        active_dims = wp.array([dim] * n_worlds, dtype=wp.int32, device=device)

        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[padded_dim] * n_worlds, dtype=float32, device=device)
        info.dim = active_dims
        dense_op = BatchedLinearOperator.from_dense(DenseLinearOperatorData(info=info, mat=A_wp))
        sparse_op = BatchedLinearOperator.from_block_sparse(bsm, active_dims)

        # Prepare RHS
        b_2d = np.zeros((n_worlds, padded_dim), dtype=np.float32)
        for m, b in enumerate(b_list):
            b_2d[m, :dim] = b
        b_wp = wp.array(b_2d, dtype=float32, device=device)

        world_active = wp.full(n_worlds, True, dtype=wp.bool, device=device)
        atol = wp.full(n_worlds, 1.0e-6, dtype=float32, device=device)
        rtol = wp.full(n_worlds, 1.0e-6, dtype=float32, device=device)

        # Solve with dense operator
        x_dense = wp.zeros((n_worlds, padded_dim), dtype=float32, device=device)
        solver_dense = solver_cls(
            A=dense_op, world_active=world_active, atol=atol, rtol=rtol,
            maxiter=None, Mi=None, callback=None, use_cuda_graph=False,
        )
        solver_dense.solve(b_wp, x_dense)

        # Solve with sparse operator
        x_sparse = wp.zeros((n_worlds, padded_dim), dtype=float32, device=device)
        solver_sparse = solver_cls(
            A=sparse_op, world_active=world_active, atol=atol, rtol=rtol,
            maxiter=None, Mi=None, callback=None, use_cuda_graph=False,
        )
        solver_sparse.solve(b_wp, x_sparse)

        # Compare results
        x_dense_np = x_dense.numpy()
        x_sparse_np = x_sparse.numpy()
        for m in range(n_worlds):
            x_d = x_dense_np[m, :dim]
            x_s = x_sparse_np[m, :dim]
            x_ref = x_ref_list[m]

            if self.verbose:
                print(f"World {m}:")
                print_error_stats("x_dense vs ref", x_d, x_ref, dim)
                print_error_stats("x_sparse vs ref", x_s, x_ref, dim)
                print_error_stats("x_dense vs x_sparse", x_d, x_s, dim)

            self.assertTrue(np.allclose(x_d, x_ref, rtol=1e-3, atol=1e-4), "Dense solution differs from reference")
            self.assertTrue(np.allclose(x_s, x_ref, rtol=1e-3, atol=1e-4), "Sparse solution differs from reference")
            self.assertTrue(np.allclose(x_d, x_s, rtol=1e-5, atol=1e-6), "Dense and sparse solutions differ")

    @classmethod
    def _sparse_problem_params(cls):
        return {
            "small_4x4_blocks": {"n_worlds": 2, "dim": 16, "block_size": 4},
            "medium_6x6_blocks": {"n_worlds": 3, "dim": 48, "block_size": 6},
        }

    def test_sparse_solve_cg_cuda(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()
        for problem_name, params in self._sparse_problem_params().items():
            with self.subTest(problem=problem_name, solver="CGSolver"):
                self._test_sparse_solve(CGSolver, device=device, **params)

    def test_sparse_solve_cr_cuda(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()
        for problem_name, params in self._sparse_problem_params().items():
            with self.subTest(problem=problem_name, solver="CRSolver"):
                self._test_sparse_solve(CRSolver, device=device, **params)


    def _build_sparse_operator(self, A: np.ndarray, block_size: int, device):
        """Helper to build a sparse operator from a dense matrix."""
        dim = A.shape[0]
        n_blocks = dim // block_size
        total_blocks = n_blocks * n_blocks

        # Set up block coordinates (all blocks, row-major order)
        coords = [(bi * block_size, bj * block_size) for bi in range(n_blocks) for bj in range(n_blocks)]

        bsm = BlockSparseMatrices()
        bsm.finalize(
            max_dims=[(dim, dim)],
            capacities=[total_blocks],
            nzb_dtype=BlockDType(float32, (block_size, block_size)),
            device=device,
        )
        bsm.dims.assign(np.array([[dim, dim]], dtype=np.int32))
        bsm.num_nzb.assign(np.array([total_blocks], dtype=np.int32))
        bsm.nzb_coords.assign(np.array(coords, dtype=np.int32))
        bsm.assign([A])

        active_dims = wp.array([dim], dtype=wp.int32, device=device)
        return BatchedLinearOperator.from_block_sparse(bsm, active_dims)

    def test_sparse_cg_solve_simple(self):
        """Test CG solve with sparse operator on a 16x16 system with 4x4 blocks."""
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()

        dim, block_size = 16, 4
        A = random_spd_matrix(dim=dim, seed=self.seed, dtype=np.float32)
        b = np.random.default_rng(self.seed).standard_normal(dim).astype(np.float32)
        x_ref = np.linalg.solve(A, b)

        sparse_op = self._build_sparse_operator(A, block_size, device)

        b_wp = wp.array(b.reshape(1, -1), dtype=float32, device=device)
        x_wp = wp.zeros((1, dim), dtype=float32, device=device)
        world_active = wp.full(1, True, dtype=wp.bool, device=device)
        atol = wp.full(1, 1e-6, dtype=float32, device=device)
        rtol = wp.full(1, 1e-6, dtype=float32, device=device)

        solver = CGSolver(
            A=sparse_op,
            world_active=world_active,
            atol=atol,
            rtol=rtol,
            maxiter=None,
            Mi=None,
            use_cuda_graph=False,
        )
        solver.solve(b_wp, x_wp)

        x_result = x_wp.numpy().flatten()
        self.assertTrue(
            np.allclose(x_result, x_ref, rtol=1e-3, atol=1e-4),
            f"CG solve failed: {x_result} vs {x_ref}, error={np.abs(x_result - x_ref).max():.2e}",
        )


if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, precision=10, threshold=20000, suppress=True)  # Suppress scientific notation

    wp.init()

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False

    # Clear caches
    # wp.clear_kernel_cache()
    # wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
