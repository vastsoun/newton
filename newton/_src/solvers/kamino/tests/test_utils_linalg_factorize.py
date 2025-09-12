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
from dataclasses import dataclass

import numpy as np

import newton._src.solvers.kamino.tests.utils.random as rand

# Module to be tested
import newton._src.solvers.kamino.utils.linalg as linalg

###
# Helpers
###


@dataclass
class FactorizerProblem:
    A: np.ndarray
    b: np.ndarray
    A_norm_l2: float
    A_norm_inf: float
    b_norm_l2: float
    b_norm_inf: float


@dataclass
class FactorizerPerformance:
    abs_factorization_error_norm_l2: float
    abs_factorization_error_norm_inf: float
    rel_factorization_error_norm_l2: float
    rel_factorization_error_norm_inf: float
    abs_solve_error_norm_l2: float
    abs_solve_error_norm_inf: float
    rel_solve_error_norm_l2: float
    rel_solve_error_norm_inf: float


def run_factorizer_performance_test(
    factorizer: linalg.FactorizerType, problem: FactorizerProblem
) -> FactorizerPerformance:
    # TODO
    factorizer.factorize(problem.A)
    A_rec = factorizer.reconstructed()
    A_err = problem.A - A_rec
    abs_factorization_error_norm_l2 = np.linalg.norm(A_err)
    abs_factorization_error_norm_inf = np.max(np.abs(A_err))
    rel_factorization_error_norm_l2 = (
        abs_factorization_error_norm_l2 / problem.A_norm_l2 if problem.A_norm_l2 > 0 else np.nan
    )
    rel_factorization_error_norm_inf = (
        abs_factorization_error_norm_inf / problem.A_norm_inf if problem.A_norm_inf > 0 else np.nan
    )

    # TODO
    x = factorizer.solve(problem.b)
    solve_err = problem.A @ x - problem.b
    abs_solve_error_norm_l2 = np.linalg.norm(solve_err)
    abs_solve_error_norm_inf = np.max(np.abs(solve_err))
    rel_solve_error_norm_l2 = abs_solve_error_norm_l2 / problem.b_norm_l2 if problem.b_norm_l2 > 0 else np.nan
    rel_solve_error_norm_inf = abs_solve_error_norm_inf / problem.b_norm_inf if problem.b_norm_inf > 0 else np.nan

    # TODO
    return FactorizerPerformance(
        abs_factorization_error_norm_l2=abs_factorization_error_norm_l2,
        abs_factorization_error_norm_inf=abs_factorization_error_norm_inf,
        rel_factorization_error_norm_l2=rel_factorization_error_norm_l2,
        rel_factorization_error_norm_inf=rel_factorization_error_norm_inf,
        abs_solve_error_norm_l2=abs_solve_error_norm_l2,
        abs_solve_error_norm_inf=abs_solve_error_norm_inf,
        rel_solve_error_norm_l2=rel_solve_error_norm_l2,
        rel_solve_error_norm_inf=rel_solve_error_norm_inf,
    )


###
# Tests
###


class TestFactorizations(unittest.TestCase):
    def setUp(self):
        self.verbose = True  # Set to True for verbose output

    def test_cholesky_lower_on_small(self):
        print("\n")  # Add a newline for better readability in output

        # Define data parameters
        dtype = np.float32
        dim = 200
        scale = 1.0  # 1e+4
        seed = 42

        # Generate a set of eigenvalues
        eigenvalues = np.linspace(10.0, 1.0, num=dim, dtype=dtype)
        # eigenvalues = np.linspace(10.0, -10.0, num=dim, dtype=dtype)
        # eigenvalues = [10, 9, 8, -7, 6, 5, -1e-7, -0.17, 1.0, 0]
        # eigenvalues = rand.eigenvalues_from_distribution(size=dim, dtype=dtype)
        print(f"eigenvalues ({len(eigenvalues)})[{eigenvalues.dtype}]:\n{eigenvalues}\n")

        # Generate a random symmetric matrix
        A = rand.random_symmetric_matrix(dim, eigenvalues=eigenvalues, scale=scale, dtype=dtype, seed=seed)
        print(f"A ({A.shape})[{A.dtype}]:\n{A}\n")
        A_props = linalg.SquareSymmetricMatrixProperties(A)
        print(f"A properties:\n{A_props}\n")

        # Generate a random right-hand side vector to complete the linear system
        b = rand.random_rhs_for_matrix(A, scale=scale)
        b_norm_l2 = np.linalg.norm(b)
        b_norm_inf = np.max(np.abs(b))
        print(f"b ({b.shape})[{b.dtype}]:\n{b}\n")
        print(f"b norm (L2): {b_norm_l2}")
        print(f"b norm (L∞): {b_norm_inf}\n")

        factorizer = linalg.Cholesky(A=A, itype=np.int32, check_symmetry=True, compute_error=True)
        # factorizer = linalg.LDLTNoPivot(A=A, itype=np.int32, check_symmetry=True, compute_error=True)
        # factorizer = linalg.LDLTBlocked(A=A, itype=np.int32, check_symmetry=True, compute_error=True)
        # factorizer = linalg.LDLTBunchKaufman(A=A, itype=np.int32, check_symmetry=True, compute_error=True)
        # factorizer = linalg.LDLTEigen3(A=A, itype=np.int32, check_symmetry=True, compute_error=True)
        X = factorizer.matrix
        print(f"X ({X.shape})[{X.dtype}]:\n{X}\n")

        A_rec = factorizer.reconstructed()
        print(f"A_rec ({A_rec.shape})[{A_rec.dtype}]:\n{A_rec}\n")

        x = factorizer.solve(b)
        print(f"x ({x.shape})[{x.dtype}]:\n{x}\n")

        A_rec_props = linalg.SquareSymmetricMatrixProperties(A_rec)
        print(f"A_rec properties:\n{A_rec_props}\n")

        A_err = A - A_rec
        A_err_norm_l2 = np.linalg.norm(A_err)
        A_err_norm_inf = np.max(np.abs(A_err))
        print(f"factorization: absolute error (L2): {A_err_norm_l2}")
        print(f"factorization: absolute error (L∞): {A_err_norm_inf}")
        print(
            f"factorization: relative error (L2): {A_err_norm_l2 / A_props.norm_l2 if A_props.norm_l2 > 0 else np.nan}"
        )
        print(
            f"factorization: relative error (L∞): {A_err_norm_inf / A_props.norm_inf if A_props.norm_inf > 0 else np.nan}\n"
        )

        solve_err = A @ x - b
        solver_error_norm_l2 = np.linalg.norm(solve_err)
        solver_error_norm_inf = np.max(np.abs(solve_err))
        print(f"solution: absolute error (L2): {solver_error_norm_l2}")
        print(f"solution: absolute error (L∞): {solver_error_norm_inf}")
        print(f"solution: relative error (L2): {solver_error_norm_l2 / b_norm_l2 if b_norm_l2 > 0 else np.nan}")
        print(f"solution: relative error (L∞): {solver_error_norm_inf / b_norm_inf if b_norm_inf > 0 else np.nan}\n")


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=10000, precision=10, suppress=True)  # Suppress scientific notation

    # Run all tests
    unittest.main(verbosity=2)
