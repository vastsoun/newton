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

"""Unit tests for linear algebra utilities"""

import unittest
from typing import Any

import numpy as np
import scipy.linalg

import newton._src.solvers.kamino.utils.linalg as linalg
import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.core.types import override

###
# Test utilities
###


class TestLinearSolver(linalg.LinearSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
    ):
        """Provides an example of a custom linear solver which adds additional internal data."""
        # Declare implementation-specific members
        self._norm_A: float | None = None
        # Initialize base class
        super().__init__(A=A, atol=atol, rtol=rtol, dtype=dtype)

    @override
    def _compute_impl(self, A: np.ndarray):
        """Provides an example of a pre-computation operation where info about the matrix is computed and stored."""
        self._norm_A = np.linalg.norm(A)

    @override
    def _solve_inplace_impl(self, b: np.ndarray):
        """Provides an example of a solve operation where the solution is computed and stored in-place."""
        x = np.linalg.solve(self._matrix, b)
        b[:] = x


class TestIndirectSolver(linalg.IndirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
        record_errors: bool = False,
        record_residuals: bool = False,
    ):
        """Provides an example of a custom indirect solver which adds additional internal data."""
        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            record_errors=record_errors,
            record_residuals=record_residuals,
        )

    @override
    def _solve_iterative_impl(self, b: np.ndarray) -> linalg.LinearSolution:
        """Provides an example of a solve operation where the solution is accompanied by convergence info."""
        x = np.linalg.solve(self._matrix, b)
        solution = linalg.LinearSolution()
        solution.converged = True
        solution.iterations = 1
        solution.x = x
        if self._record_errors:
            solution.e = np.array([0.0], dtype=self._dtype)
        if self._record_residuals:
            solution.r = np.array([0.0], dtype=self._dtype)
        return solution


class TestDirectSolver(linalg.DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        """Provides an example of a custom direct solver which adds additional internal data."""

        # Declare implementation-specific members
        self._L: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._L = np.linalg.cholesky(A, upper=False)

    @override
    def _unpack_impl(self) -> None:
        pass

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._L @ self._L.T

    @override
    def _solve_inplace_impl(self, x: np.ndarray):
        scipy.linalg.cho_solve((self._L, True), x, overwrite_b=True)


###
# Tests
###


class TestUtilsLinAlgLinearSolver(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)

        # Define test parameters
        self.dtype = np.float32
        self.epsilon = np.finfo(self.dtype).eps
        self.atol = np.finfo(self.dtype).eps
        self.rtol = np.finfo(self.dtype).eps
        self.max_iterations: int = 1000

        # Print test configuration
        msg.debug(
            "\nTest Setup:"
            f"\n  dtype: {self.dtype}"
            f"\n  epsilon: {self.epsilon}"
            f"\n  atol: {self.atol}"
            f"\n  rtol: {self.rtol}"
            f"\n  max_iterations: {self.max_iterations}\n"
        )

        # Define a simple symmetric positive-definite matrix and right-hand side
        self.A = np.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype)  # symmetric positive-definite matrix
        self.b = np.array([1.0, 2.0], dtype=self.dtype)  # right-hand side in range-space of A
        msg.debug(
            "\nLinear system:\n"
            f"\nA {self.A.shape}[{self.A.dtype}]:\n{self.A}\n"
            f"\nb {self.b.shape}[{self.b.dtype}]:\n{self.b}\n"
        )

        # Compute matrix properties
        self.norm_A = np.linalg.norm(self.A)
        self.rank_A = np.linalg.matrix_rank(self.A)
        self.cond_A = np.linalg.cond(self.A)
        self.lambda_A = np.linalg.eigvals(self.A)
        msg.debug(
            "\nmatrix properties:"
            f"\n  norm(A): {self.norm_A}"
            f"\n  rank(A): {self.rank_A}"
            f"\n  cond(A): {self.cond_A}"
            f"\n  lambda(A): {self.lambda_A}\n"
        )

    def tearDown(self):
        if self.verbose:
            msg.reset_log_level()

    def test_01_defaulted_linear_solver(self):
        solver = TestLinearSolver()

        # Check default-initialized attributes
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix: {solver._matrix}")
        msg.debug(f"solver._norm_A: {solver._norm_A}\n")
        eps64 = np.finfo(np.float64).eps
        self.assertEqual(solver._dtype, np.float64)
        self.assertEqual(solver._atol, eps64)
        self.assertEqual(solver._rtol, eps64)
        self.assertIsNone(solver._matrix)
        self.assertIsNone(solver._norm_A)

        # Test compute method
        solver.compute(self.A)
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix:\n{solver._matrix}")
        msg.debug(f"solver._norm_A: {solver._norm_A}\n")
        self.assertEqual(solver._dtype, self.dtype)
        self.assertEqual(solver._atol, self.atol)
        self.assertEqual(solver._rtol, self.rtol)
        self.assertTrue(np.array_equal(solver._matrix, self.A))
        self.assertEqual(solver._norm_A, self.norm_A)

        # Compute reference solution with numpy
        x_np = np.linalg.solve(self.A, self.b)
        msg.debug(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

        # Test solve method w/o error computation
        x_0 = solver.solve(self.b, compute_error=False)
        msg.debug(f"\nx_0 {x_0.shape}[{x_0.dtype}]:\n{x_0}\n")
        self.assertTrue(np.allclose(x_0, x_np, atol=self.atol, rtol=self.rtol))

        # Test solve method with error computation
        x_1 = solver.solve(self.b, compute_error=True)
        msg.debug(f"\nx_1 {x_1.shape}[{x_1.dtype}]:\n{x_1}\n")
        self.assertTrue(np.allclose(x_1, x_np, atol=self.atol, rtol=self.rtol))

        # Test solve_inplace method w/o error computation
        x_2 = self.b.copy()
        solver.solve_inplace(x_2, compute_error=False)
        msg.debug(f"\nx_2 {x_2.shape}[{x_2.dtype}]:\n{x_2}\n")
        self.assertTrue(np.allclose(x_2, x_np, atol=self.atol, rtol=self.rtol))

        # Test solve_inplace method with error computation
        x_3 = self.b.copy()
        solver.solve_inplace(x_3, compute_error=True)
        msg.debug(f"\nx_3 {x_3.shape}[{x_3.dtype}]:\n{x_3}\n")
        self.assertTrue(np.allclose(x_3, x_np, atol=self.atol, rtol=self.rtol))

    def test_02_initialized_linear_solver(self):
        solver = TestLinearSolver(self.A, atol=self.atol, rtol=self.rtol)

        # Check initialized attributes
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix:\n{solver._matrix}")
        msg.debug(f"solver._norm_A: {solver._norm_A}\n")
        self.assertEqual(solver._dtype, self.dtype)
        self.assertEqual(solver._atol, self.atol)
        self.assertEqual(solver._rtol, self.rtol)
        self.assertTrue(np.array_equal(solver._matrix, self.A))
        self.assertEqual(solver._norm_A, self.norm_A)

        # Compute reference solution with numpy
        x_np = np.linalg.solve(self.A, self.b)
        msg.debug(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

        # Test solve method w/o error computation
        x_0 = solver.solve(self.b, compute_error=False)
        msg.debug(f"\nx_0 {x_0.shape}[{x_0.dtype}]:\n{x_0}\n")
        self.assertTrue(np.allclose(x_0, x_np, atol=self.atol, rtol=self.rtol))

        # Test solve method with error computation
        x_1 = solver.solve(self.b, compute_error=True)
        msg.debug(f"\nx_1 {x_1.shape}[{x_1.dtype}]:\n{x_1}\n")
        self.assertTrue(np.allclose(x_1, x_np, atol=self.atol, rtol=self.rtol))

        # Test solve_inplace method w/o error computation
        x_2 = self.b.copy()
        solver.solve_inplace(x_2, compute_error=False)
        msg.debug(f"\nx_2 {x_2.shape}[{x_2.dtype}]:\n{x_2}\n")
        self.assertTrue(np.allclose(x_2, x_np, atol=self.atol, rtol=self.rtol))

        # Test solve_inplace method with error computation
        x_3 = self.b.copy()
        solver.solve_inplace(x_3, compute_error=True)
        msg.debug(f"\nx_3 {x_3.shape}[{x_3.dtype}]:\n{x_3}\n")
        self.assertTrue(np.allclose(x_3, x_np, atol=self.atol, rtol=self.rtol))

    def test_03_defaulted_indirect_solver(self):
        solver = TestIndirectSolver()

        # Check defaulted attributes
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix: {solver._matrix}")
        msg.debug(f"solver._record_errors: {solver._record_errors}")
        msg.debug(f"solver._record_residuals: {solver._record_residuals}\n")
        eps64 = np.finfo(np.float64).eps
        self.assertEqual(solver._dtype, np.float64)
        self.assertEqual(solver._atol, eps64)
        self.assertEqual(solver._rtol, eps64)
        self.assertIsNone(solver._matrix)
        self.assertFalse(solver._record_errors)
        self.assertFalse(solver._record_residuals)

        # Configure solver for problem
        solver.compute(self.A)
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix:\n{solver._matrix}\n")
        self.assertEqual(solver._dtype, self.dtype)
        self.assertEqual(solver._atol, self.atol)
        self.assertEqual(solver._rtol, self.rtol)
        self.assertTrue(np.array_equal(solver._matrix, self.A))

        # Compute reference solution with numpy
        x_np = np.linalg.solve(self.A, self.b)
        msg.debug(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

        # Test solve method w/o error computation
        x_0 = solver.solve(self.b, compute_error=False)
        msg.debug(f"\nx_0 {x_0.shape}[{x_0.dtype}]:\n{x_0}\n")
        self.assertTrue(np.allclose(x_0, x_np, atol=self.atol, rtol=self.rtol))
        self.assertTrue(solver.solution.converged)
        self.assertEqual(solver.solution.iterations, 1)
        self.assertIsNone(solver.solution.e)
        self.assertIsNone(solver.solution.r)

        # Test solve method with error computation
        x_1 = solver.solve(self.b, compute_error=True)
        msg.debug(f"\nx_1 {x_1.shape}[{x_1.dtype}]:\n{x_1}\n")
        self.assertTrue(np.allclose(x_1, x_np, atol=self.atol, rtol=self.rtol))
        self.assertTrue(solver.solution.converged)
        self.assertEqual(solver.solution.iterations, 1)
        self.assertIsNone(solver.solution.e)
        self.assertIsNone(solver.solution.r)

        # Test solve_inplace method w/o error computation
        x_2 = self.b.copy()
        solver.solve_inplace(x_2, compute_error=False)
        msg.debug(f"\nx_2 {x_2.shape}[{x_2.dtype}]:\n{x_2}\n")
        self.assertTrue(np.allclose(x_2, x_np, atol=self.atol, rtol=self.rtol))
        self.assertTrue(solver.solution.converged)
        self.assertEqual(solver.solution.iterations, 1)
        self.assertIsNone(solver.solution.e)
        self.assertIsNone(solver.solution.r)

        # Test solve_inplace method with error computation
        x_3 = self.b.copy()
        solver.solve_inplace(x_3, compute_error=True)
        msg.debug(f"\nx_3 {x_3.shape}[{x_3.dtype}]:\n{x_3}\n")
        self.assertTrue(np.allclose(x_3, x_np, atol=self.atol, rtol=self.rtol))
        self.assertTrue(solver.solution.converged)
        self.assertEqual(solver.solution.iterations, 1)
        self.assertIsNone(solver.solution.e)
        self.assertIsNone(solver.solution.r)

    def test_04_initialized_indirect_solver(self):
        solver = TestIndirectSolver(A=self.A, atol=self.atol, rtol=self.rtol, record_errors=True, record_residuals=True)

        # Check initialized attributes
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix:\n{solver._matrix}")
        msg.debug(f"solver._record_errors: {solver._record_errors}")
        msg.debug(f"solver._record_residuals: {solver._record_residuals}\n")
        self.assertEqual(solver._dtype, self.dtype)
        self.assertEqual(solver._atol, self.atol)
        self.assertEqual(solver._rtol, self.rtol)
        self.assertTrue(np.array_equal(solver._matrix, self.A))
        self.assertTrue(solver._record_errors)
        self.assertTrue(solver._record_residuals)

        # Compute reference solution with numpy
        x_np = np.linalg.solve(self.A, self.b)
        msg.debug(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

        # Test solve method w/o error computation
        x_0 = solver.solve(self.b, compute_error=False)
        msg.debug(f"\nx_0 {x_0.shape}[{x_0.dtype}]:\n{x_0}\n")
        self.assertTrue(np.allclose(x_0, x_np, atol=self.atol, rtol=self.rtol))
        self.assertTrue(solver.solution.converged)
        self.assertEqual(solver.solution.iterations, 1)
        self.assertTrue(np.array_equal(solver.solution.e, np.array([0.0], dtype=self.dtype)))
        self.assertTrue(np.array_equal(solver.solution.r, np.array([0.0], dtype=self.dtype)))

        # Test solve method with error computation
        x_1 = solver.solve(self.b, compute_error=True)
        msg.debug(f"\nx_1 {x_1.shape}[{x_1.dtype}]:\n{x_1}\n")
        self.assertTrue(np.allclose(x_1, x_np, atol=self.atol, rtol=self.rtol))
        self.assertTrue(solver.solution.converged)
        self.assertEqual(solver.solution.iterations, 1)
        self.assertTrue(np.array_equal(solver.solution.e, np.array([0.0], dtype=self.dtype)))
        self.assertTrue(np.array_equal(solver.solution.r, np.array([0.0], dtype=self.dtype)))

        # Test solve_inplace method w/o error computation
        x_2 = self.b.copy()
        solver.solve_inplace(x_2, compute_error=False)
        msg.debug(f"\nx_2 {x_2.shape}[{x_2.dtype}]:\n{x_2}\n")
        self.assertTrue(np.allclose(x_2, x_np, atol=self.atol, rtol=self.rtol))
        self.assertTrue(solver.solution.converged)
        self.assertEqual(solver.solution.iterations, 1)
        self.assertTrue(np.array_equal(solver.solution.e, np.array([0.0], dtype=self.dtype)))
        self.assertTrue(np.array_equal(solver.solution.r, np.array([0.0], dtype=self.dtype)))

        # Test solve_inplace method with error computation
        x_3 = self.b.copy()
        solver.solve_inplace(x_3, compute_error=True)
        msg.debug(f"\nx_3 {x_3.shape}[{x_3.dtype}]:\n{x_3}\n")
        self.assertTrue(np.allclose(x_3, x_np, atol=self.atol, rtol=self.rtol))
        self.assertTrue(solver.solution.converged)
        self.assertEqual(solver.solution.iterations, 1)
        self.assertTrue(np.array_equal(solver.solution.e, np.array([0.0], dtype=self.dtype)))
        self.assertTrue(np.array_equal(solver.solution.r, np.array([0.0], dtype=self.dtype)))

    def test_05_defaulted_direct_solver(self):
        solver = TestDirectSolver()

        # Check defaulted attributes
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix: {solver._matrix}")
        msg.debug(f"solver._ftol: {solver._ftol}")
        msg.debug(f"solver._has_factors: {solver._has_factors}")
        msg.debug(f"solver._has_unpacked: {solver._has_unpacked}")
        msg.debug(f"solver._success: {solver._success}")
        msg.debug(f"solver.compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"solver.compute_error_rel: {solver.compute_error_rel}\n")
        eps64 = np.finfo(np.float64).eps
        self.assertEqual(solver._dtype, np.float64)
        self.assertEqual(solver._atol, eps64)
        self.assertEqual(solver._rtol, eps64)
        self.assertIsNone(solver._matrix)
        self.assertEqual(solver._ftol, eps64)
        self.assertFalse(solver._has_factors)
        self.assertFalse(solver._has_unpacked)
        self.assertFalse(solver._success)
        self.assertEqual(solver.compute_error_abs, np.inf)
        self.assertEqual(solver.compute_error_rel, np.inf)

        # Configure solver for problem
        solver.compute(self.A, compute_error=True, check_error=True, check_symmetry=True)
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix:\n{solver._matrix}\n")
        self.assertEqual(solver._dtype, self.dtype)
        self.assertEqual(solver._atol, self.atol)
        self.assertEqual(solver._rtol, self.rtol)
        self.assertTrue(np.array_equal(solver._matrix, self.A))
        self.assertTrue(solver._has_factors)
        self.assertFalse(solver._has_unpacked)
        self.assertTrue(solver._success)
        self.assertIsNotNone(solver.compute_error_abs)
        self.assertIsNotNone(solver.compute_error_rel)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)

        # Compute reference solution with numpy
        x_np = np.linalg.solve(self.A, self.b)
        msg.debug(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

        # Test solve method w/o error computation
        x_0 = solver.solve(self.b, compute_error=False)
        msg.debug(f"\nx_0 {x_0.shape}[{x_0.dtype}]:\n{x_0}\n")
        msg.debug(f"solver.solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"solver.solve_error_rel: {solver.solve_error_rel}\n")
        self.assertTrue(np.allclose(x_0, x_np, atol=self.atol, rtol=self.rtol))
        self.assertEqual(solver.solve_error_abs, np.inf)
        self.assertEqual(solver.solve_error_rel, np.inf)

        # Test solve method with error computation
        x_1 = solver.solve(self.b, compute_error=True)
        msg.debug(f"\nx_1 {x_1.shape}[{x_1.dtype}]:\n{x_1}\n")
        msg.debug(f"solver.solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"solver.solve_error_rel: {solver.solve_error_rel}\n")
        self.assertTrue(np.allclose(x_1, x_np, atol=self.atol, rtol=self.rtol))
        self.assertIsNotNone(solver.solve_error_abs)
        self.assertIsNotNone(solver.solve_error_rel)
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)

        # Test solve_inplace method w/o error computation
        x_2 = self.b.copy()
        solver.solve_inplace(x_2, compute_error=False)
        msg.debug(f"\nx_2 {x_2.shape}[{x_2.dtype}]:\n{x_2}\n")
        msg.debug(f"solver.solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"solver.solve_error_rel: {solver.solve_error_rel}\n")
        self.assertTrue(np.allclose(x_2, x_np, atol=self.atol, rtol=self.rtol))
        self.assertEqual(solver.solve_error_abs, np.inf)
        self.assertEqual(solver.solve_error_rel, np.inf)

        # Test solve_inplace method with error computation
        x_3 = self.b.copy()
        solver.solve_inplace(x_3, compute_error=True)
        msg.debug(f"\nx_3 {x_3.shape}[{x_3.dtype}]:\n{x_3}\n")
        msg.debug(f"solver.solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"solver.solve_error_rel: {solver.solve_error_rel}\n")
        self.assertTrue(np.allclose(x_3, x_np, atol=self.atol, rtol=self.rtol))
        self.assertIsNotNone(solver.solve_error_abs)
        self.assertIsNotNone(solver.solve_error_rel)
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)

    def test_06_initialized_direct_solver(self):
        solver = TestDirectSolver(
            A=self.A, atol=self.atol, rtol=self.rtol, compute_error=True, check_error=True, check_symmetry=True
        )

        # Check initialized attributes
        msg.debug(f"solver._dtype: {solver._dtype}")
        msg.debug(f"solver._atol: {solver._atol}")
        msg.debug(f"solver._rtol: {solver._rtol}")
        msg.debug(f"solver._matrix:\n{solver._matrix}\n")
        self.assertEqual(solver._dtype, self.dtype)
        self.assertEqual(solver._atol, self.atol)
        self.assertEqual(solver._rtol, self.rtol)
        self.assertTrue(np.array_equal(solver._matrix, self.A))
        self.assertTrue(solver._has_factors)
        self.assertFalse(solver._has_unpacked)
        self.assertTrue(solver._success)
        self.assertIsNotNone(solver.compute_error_abs)
        self.assertIsNotNone(solver.compute_error_rel)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)

        # Compute reference solution with numpy
        x_np = np.linalg.solve(self.A, self.b)
        msg.debug(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

        # Test solve method w/o error computation
        x_0 = solver.solve(self.b, compute_error=False)
        msg.debug(f"\nx_0 {x_0.shape}[{x_0.dtype}]:\n{x_0}\n")
        msg.debug(f"solver.solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"solver.solve_error_rel: {solver.solve_error_rel}\n")
        self.assertTrue(np.allclose(x_0, x_np, atol=self.atol, rtol=self.rtol))
        self.assertEqual(solver.solve_error_abs, np.inf)
        self.assertEqual(solver.solve_error_rel, np.inf)

        # Test solve method with error computation
        x_1 = solver.solve(self.b, compute_error=True)
        msg.debug(f"\nx_1 {x_1.shape}[{x_1.dtype}]:\n{x_1}\n")
        msg.debug(f"solver.solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"solver.solve_error_rel: {solver.solve_error_rel}\n")
        self.assertTrue(np.allclose(x_1, x_np, atol=self.atol, rtol=self.rtol))
        self.assertIsNotNone(solver.solve_error_abs)
        self.assertIsNotNone(solver.solve_error_rel)
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)

        # Test solve_inplace method w/o error computation
        x_2 = self.b.copy()
        solver.solve_inplace(x_2, compute_error=False)
        msg.debug(f"\nx_2 {x_2.shape}[{x_2.dtype}]:\n{x_2}\n")
        msg.debug(f"solver.solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"solver.solve_error_rel: {solver.solve_error_rel}\n")
        self.assertTrue(np.allclose(x_2, x_np, atol=self.atol, rtol=self.rtol))
        self.assertEqual(solver.solve_error_abs, np.inf)
        self.assertEqual(solver.solve_error_rel, np.inf)

        # Test solve_inplace method with error computation
        x_3 = self.b.copy()
        solver.solve_inplace(x_3, compute_error=True)
        msg.debug(f"\nx_3 {x_3.shape}[{x_3.dtype}]:\n{x_3}\n")
        msg.debug(f"solver.solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"solver.solve_error_rel: {solver.solve_error_rel}\n")
        self.assertTrue(np.allclose(x_3, x_np, atol=self.atol, rtol=self.rtol))
        self.assertIsNotNone(solver.solve_error_abs)
        self.assertIsNotNone(solver.solve_error_rel)
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)

    def test_07_iterative_solver_methods(self):
        # Print test configuration
        msg.debug("\n----------------------------")
        msg.debug(f"dtype: {self.dtype}")
        msg.debug("\n----------------------------")
        msg.debug(f"epsilon: {self.epsilon}")
        msg.debug(f"atol: {self.atol}")
        msg.debug(f"rtol: {self.rtol}")
        msg.debug(f"max_iterations: {self.max_iterations}")

        # Define a simple symmetric positive-definite matrix and right-hand side
        A = np.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype)  # symmetric positive-definite matrix
        b = np.array([1.0, 2.0], dtype=self.dtype)  # right-hand side in range-space of A
        msg.debug("\n----------------------------")
        msg.debug(f"A {A.shape}[{A.dtype}]:\n{A}\n")
        msg.debug(f"b {b.shape}[{b.dtype}]:\n{b}\n")

        # Analyze matrix
        norm_A = np.linalg.norm(A)
        rank_A = np.linalg.matrix_rank(A)
        cond_A = np.linalg.cond(A)
        lambda_A = np.linalg.eigvals(A)
        msg.debug("\n----------------------------")
        msg.debug(f"norm(A): {norm_A}")
        msg.debug(f"rank(A): {rank_A}")
        msg.debug(f"cond(A): {cond_A}")
        msg.debug(f"lambda(A): {lambda_A}")

        # Compute reference solution with numpy
        x_np = np.linalg.solve(A, b)
        # Solve with our Jacobi implementation
        sol_jac = linalg.jacobi(A=A, b=b, atol=self.atol, rtol=self.rtol, max_iterations=self.max_iterations)
        # Solve with our Gauss-Seidel implementation
        sol_gs = linalg.gauss_seidel(A=A, b=b, atol=self.atol, rtol=self.rtol, max_iterations=self.max_iterations)
        # Solve with our Successive Over-Relaxation implementation
        sol_sor = linalg.successive_over_relaxation(
            A=A, b=b, omega=1.25, atol=self.atol, rtol=self.rtol, max_iterations=self.max_iterations
        )
        # Solve with our Conjugate Gradient implementation
        sol_cg = linalg.conjugate_gradient(A=A, b=b, atol=self.atol, rtol=self.rtol, max_iterations=self.max_iterations)
        # Solve with our Minimum Residual implementation
        sol_minres = linalg.minimum_residual(
            A=A, b=b, atol=self.atol, rtol=self.rtol, max_iterations=self.max_iterations
        )

        # Check convergence
        self.assertTrue(sol_jac.converged)
        self.assertTrue(sol_gs.converged)
        self.assertTrue(sol_sor.converged)
        self.assertTrue(sol_cg.converged)
        self.assertTrue(sol_minres.converged)

        # Assert solutions are close to reference
        self.assertTrue(np.allclose(x_np, sol_jac.x, atol=self.atol, rtol=self.rtol))
        self.assertTrue(np.allclose(x_np, sol_gs.x, atol=self.atol, rtol=self.rtol))
        self.assertTrue(np.allclose(x_np, sol_sor.x, atol=self.atol, rtol=self.rtol))
        self.assertTrue(np.allclose(x_np, sol_cg.x, atol=self.atol, rtol=self.rtol))
        self.assertTrue(np.allclose(x_np, sol_minres.x, atol=self.atol, rtol=self.rtol))

        # Compute residuals
        r_np_inf = linalg.linear.linsys_error_inf(A, b, x_np)
        r_jac_inf = linalg.linear.linsys_error_inf(A, b, sol_jac.x)
        r_gs_inf = linalg.linear.linsys_error_inf(A, b, sol_gs.x)
        r_sor_inf = linalg.linear.linsys_error_inf(A, b, sol_sor.x)
        r_cg_inf = linalg.linear.linsys_error_inf(A, b, sol_cg.x)
        r_minres_inf = linalg.linear.linsys_error_inf(A, b, sol_minres.x)

        # Print results
        msg.debug("\n----------------------------")
        msg.debug(f"x_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")
        msg.debug(f"x_jac {sol_jac.x.shape}[{sol_jac.x.dtype}]:\n{sol_jac.x}\n")
        msg.debug(f"x_gs {sol_gs.x.shape}[{sol_gs.x.dtype}]:\n{sol_gs.x}\n")
        msg.debug(f"x_sor {sol_sor.x.shape}[{sol_sor.x.dtype}]:\n{sol_sor.x}\n")
        msg.debug(f"x_cg {sol_cg.x.shape}[{sol_cg.x.dtype}]:\n{sol_cg.x}\n")
        msg.debug(f"x_minres {sol_minres.x.shape}[{sol_minres.x.dtype}]:\n{sol_minres.x}\n")
        msg.debug("\n----------------------------")
        msg.debug(f"r_np_inf     : {r_np_inf}")
        msg.debug(f"r_jac_inf    : {r_jac_inf}")
        msg.debug(f"r_gs_inf     : {r_gs_inf}")
        msg.debug(f"r_sor_inf    : {r_sor_inf}")
        msg.debug(f"r_cg_inf     : {r_cg_inf}")
        msg.debug(f"r_minres_inf : {r_minres_inf}\n")


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=2000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Run all tests
    unittest.main(verbosity=2)
