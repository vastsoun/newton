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

import numpy as np

import newton._src.solvers.kamino.utils.linalg as linalg
import newton._src.solvers.kamino.utils.logger as msg

###
# Tests
###


class TestUtilsLinAlgLinearSolvers(unittest.TestCase):
    def setUp(self):
        self.verbose = True  # Set to True for verbose output
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)

        # Define test parameters
        self.dtype = np.float32
        self.itype = np.int32
        self.epsilon = np.finfo(self.dtype).eps
        self.atol = np.finfo(self.dtype).eps
        self.rtol = np.finfo(self.dtype).eps
        self.ftol = np.finfo(self.dtype).eps
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

        # Compute reference solution using NumPy
        self.x_ref = np.linalg.solve(self.A, self.b)
        msg.debug(f"x_ref {self.x_ref.shape}, {self.x_ref.dtype}:\n{self.x_ref}\n")

    def test_01_defaulted_numpy_solver(self):
        solver = linalg.NumPySolver()
        solver.compute(self.A)
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"NumPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"NumPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"NumPySolver: solve_error_rel: {solver.solve_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_02_initialized_numpy_solver(self):
        solver = linalg.NumPySolver(A=self.A, atol=self.atol, rtol=self.rtol)
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"NumPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"NumPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"NumPySolver: solve_error_rel: {solver.solve_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_03_defaulted_scipy_solver(self):
        solver = linalg.SciPySolver()
        solver.compute(self.A)
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"SciPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"SciPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"SciPySolver: solve_error_rel: {solver.solve_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_04_initialized_scipy_solver(self):
        solver = linalg.SciPySolver(A=self.A, atol=self.atol, rtol=self.rtol)
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"SciPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"SciPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"SciPySolver: solve_error_rel: {solver.solve_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_05_defaulted_llt_numpy_solver(self):
        solver = linalg.LLTNumPySolver()
        solver.compute(self.A, compute_error=True, check_error=True, check_symmetry=True)
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LLTNumPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LLTNumPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LLTNumPySolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LLTNumPySolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LLTNumPySolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_06_initialized_llt_numpy_solver(self):
        solver = linalg.LLTNumPySolver(
            A=self.A, atol=self.atol, rtol=self.rtol, compute_error=True, check_error=True, check_symmetry=True
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LLTNumPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LLTNumPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LLTNumPySolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LLTNumPySolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LLTNumPySolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_07_defaulted_llt_scipy_solver(self):
        solver = linalg.LLTSciPySolver()
        solver.compute(self.A, compute_error=True, check_error=True, check_symmetry=True)
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LLTSciPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LLTSciPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LLTSciPySolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LLTSciPySolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LLTSciPySolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_08_initialized_llt_scipy_solver(self):
        solver = linalg.LLTSciPySolver(
            A=self.A, atol=self.atol, rtol=self.rtol, compute_error=True, check_error=True, check_symmetry=True
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LLTSciPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LLTSciPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LLTSciPySolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LLTSciPySolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LLTSciPySolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_09_defaulted_ldlt_scipy_solver(self):
        solver = linalg.LDLTSciPySolver()
        solver.compute(self.A, compute_error=True, check_error=True, check_symmetry=True)
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LDLTSciPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LDLTSciPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LDLTSciPySolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LDLTSciPySolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LDLTSciPySolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_10_initialized_ldlt_scipy_solver(self):
        solver = linalg.LDLTSciPySolver(
            A=self.A, atol=self.atol, rtol=self.rtol, compute_error=True, check_error=True, check_symmetry=True
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LDLTSciPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LDLTSciPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LDLTSciPySolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LDLTSciPySolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LDLTSciPySolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_11_defaulted_lu_scipy_solver(self):
        solver = linalg.LUSciPySolver()
        solver.compute(self.A, compute_error=True, check_error=True)
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LUSciPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LUSciPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LUSciPySolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LUSciPySolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LUSciPySolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_12_initialized_lu_scipy_solver(self):
        solver = linalg.LUSciPySolver(
            A=self.A, atol=self.atol, rtol=self.rtol, compute_error=True, check_error=True, check_symmetry=True
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LUSciPySolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LUSciPySolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LUSciPySolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LUSciPySolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LUSciPySolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_13_jacobi_solver(self):
        solver = linalg.JacobiSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            max_iterations=self.max_iterations,
            record_errors=True,
            record_residuals=True,
        )
        x_0 = np.zeros_like(self.b)
        x = solver.solve(self.b, x_0=x_0, compute_error=True)
        msg.debug(f"JacobiSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"JacobiSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"JacobiSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"JacobiSolver: converged: {solver.solution.converged}")
        msg.debug(f"JacobiSolver: iterations: {solver.solution.iterations}")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(solver.solution.converged)
        self.assertTrue(solver.solution.iterations <= self.max_iterations)
        self.assertTrue(np.array_equal(x, solver.solution.x))
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_14_gauss_seidel_solver(self):
        solver = linalg.GaussSeidelSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            max_iterations=self.max_iterations,
            record_errors=True,
            record_residuals=True,
        )
        x_0 = np.zeros_like(self.b)
        x = solver.solve(self.b, x_0=x_0, compute_error=True)
        msg.debug(f"GaussSeidelSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"GaussSeidelSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"GaussSeidelSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"GaussSeidelSolver: converged: {solver.solution.converged}")
        msg.debug(f"GaussSeidelSolver: iterations: {solver.solution.iterations}")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(solver.solution.converged)
        self.assertTrue(solver.solution.iterations <= self.max_iterations)
        self.assertTrue(np.array_equal(x, solver.solution.x))
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_15_successive_over_relaxation_solver(self):
        solver = linalg.SORSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            record_errors=True,
            record_residuals=True,
            max_iterations=self.max_iterations,
            omega=1.05,
        )
        x_0 = np.zeros_like(self.b)
        x = solver.solve(self.b, x_0=x_0, compute_error=True)
        msg.debug(f"SORSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"SORSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"SORSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"SORSolver: converged: {solver.solution.converged}")
        msg.debug(f"SORSolver: iterations: {solver.solution.iterations}")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(solver.solution.converged)
        self.assertTrue(solver.solution.iterations <= self.max_iterations)
        self.assertTrue(np.array_equal(x, solver.solution.x))
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_16_conjugate_gradient_solver(self):
        solver = linalg.ConjugateGradientSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            record_errors=True,
            record_residuals=True,
            max_iterations=self.max_iterations,
        )
        x_0 = np.zeros_like(self.b)
        x = solver.solve(self.b, x_0=x_0, compute_error=True)
        msg.debug(f"ConjugateGradientSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"ConjugateGradientSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"ConjugateGradientSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"ConjugateGradientSolver: converged: {solver.solution.converged}")
        msg.debug(f"ConjugateGradientSolver: iterations: {solver.solution.iterations}")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(solver.solution.converged)
        self.assertTrue(solver.solution.iterations <= self.max_iterations)
        self.assertTrue(np.array_equal(x, solver.solution.x))
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_17_minimum_residual_solver(self):
        solver = linalg.MinimumResidualSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            record_errors=True,
            record_residuals=True,
            max_iterations=self.max_iterations,
        )
        x_0 = np.zeros_like(self.b)
        x = solver.solve(self.b, x_0=x_0, compute_error=True)
        msg.debug(f"MinimumResidualSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"MinimumResidualSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"MinimumResidualSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"MinimumResidualSolver: converged: {solver.solution.converged}")
        msg.debug(f"MinimumResidualSolver: iterations: {solver.solution.iterations}")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertTrue(solver.solution.converged)
        self.assertTrue(solver.solution.iterations <= self.max_iterations)
        self.assertTrue(np.array_equal(x, solver.solution.x))
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_18_llt_std_solver(self):
        solver = linalg.LLTStdSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            ftol=self.ftol,
            compute_error=True,
            check_error=True,
            check_symmetry=True,
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LLTStdSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LLTStdSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LLTStdSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LLTStdSolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LLTStdSolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_19_ldlt_nopivot_solver(self):
        solver = linalg.LDLTNoPivotSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            ftol=self.ftol,
            compute_error=True,
            check_error=True,
            check_symmetry=True,
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LDLTNoPivotSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LDLTNoPivotSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LDLTNoPivotSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LDLTNoPivotSolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LDLTNoPivotSolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_20_ldlt_blocked_solver(self):
        solver = linalg.LDLTBlockedSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            ftol=self.ftol,
            compute_error=True,
            check_error=True,
            check_symmetry=True,
            blocksize=1,
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LDLTBlockedSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LDLTBlockedSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LDLTBlockedSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LDLTBlockedSolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LDLTBlockedSolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    # def test_21_ldlt_bk_solver(self):
    #     solver = linalg.LDLTBunchKaufmanSolver(
    #         A=self.A,
    #         atol=self.atol,
    #         rtol=self.rtol,
    #         ftol=self.ftol,
    #         itype=self.itype,
    #         compute_error=True,
    #         check_error=True,
    #         check_symmetry=True,
    #     )
    #     x = solver.solve(self.b, compute_error=True)
    #     msg.debug(f"LDLTBunchKaufmanSolver: x {x.shape}, {x.dtype}:\n{x}\n")
    #     msg.debug(f"LDLTBunchKaufmanSolver: solve_error_abs: {solver.solve_error_abs}")
    #     msg.debug(f"LDLTBunchKaufmanSolver: solve_error_rel: {solver.solve_error_rel}")
    #     msg.debug(f"LDLTBunchKaufmanSolver: compute_error_abs: {solver.compute_error_abs}")
    #     msg.debug(f"LDLTBunchKaufmanSolver: compute_error_rel: {solver.compute_error_rel}\n")
    #     self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
    #     self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
    #     self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
    #     self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
    #     self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_22_ldlt_eigen3_solver(self):
        solver = linalg.LDLTEigen3Solver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            ftol=self.ftol,
            itype=self.itype,
            compute_error=True,
            check_error=True,
            check_symmetry=True,
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LDLTEigen3Solver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LDLTEigen3Solver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LDLTEigen3Solver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LDLTEigen3Solver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LDLTEigen3Solver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))

    def test_23_lu_nopivot_solver(self):
        solver = linalg.LUNoPivotSolver(
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            ftol=self.ftol,
            compute_error=True,
            check_error=True,
            check_symmetry=True,
        )
        x = solver.solve(self.b, compute_error=True)
        msg.debug(f"LUNoPivotSolver: x {x.shape}, {x.dtype}:\n{x}\n")
        msg.debug(f"LUNoPivotSolver: solve_error_abs: {solver.solve_error_abs}")
        msg.debug(f"LUNoPivotSolver: solve_error_rel: {solver.solve_error_rel}")
        msg.debug(f"LUNoPivotSolver: compute_error_abs: {solver.compute_error_abs}")
        msg.debug(f"LUNoPivotSolver: compute_error_rel: {solver.compute_error_rel}\n")
        self.assertAlmostEqual(solver.solve_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.solve_error_rel, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_abs, 0.0, places=6)
        self.assertAlmostEqual(solver.compute_error_rel, 0.0, places=6)
        self.assertTrue(np.allclose(x, self.x_ref, atol=self.atol, rtol=self.rtol))


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=2000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Run all tests
    unittest.main(verbosity=2)
