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

"""KAMINO: Utilities: Linear Algebra: Fixed-point iteration solvers"""

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from newton._src.solvers.kamino.utils.linalg.linear import LinearSolver

if sys.version_info >= (3, 12):
    from typing import override
else:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback no-op decorator if typing_extensions is not available
        def override(func):
            return func

###
# Types
###


@dataclass
class FixedPointSolution:
    x: np.ndarray | None = None
    error: float = np.inf
    iterations: int = 0
    converged: bool = False
    res_history: list[float] = field(default_factory=list)


###
# Utilities
###


def _check_system_compatibility(A: np.ndarray, b: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix A must be square (n x n) but has shape {A.shape}.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError(f"Vector b ({b.shape}) must have compatible dimensions with A ({A.shape}).")
    if b.dtype != A.dtype:
        raise ValueError(f"Vector b ({b.dtype}) must have the same data type as matrix A ({A.dtype}).")


def _check_initial_guess(A: np.ndarray, x_0: np.ndarray | None) -> np.ndarray:
    if x_0 is None:
        return np.zeros(A.shape[1], dtype=A.dtype)
    if x_0.ndim != 1 or x_0.shape[0] != A.shape[1]:
        raise ValueError(f"Initial guess x_0 ({x_0.shape}) must have compatible dimensions with A ({A.shape}).")
    if x_0.dtype != A.dtype:
        raise ValueError(f"Initial guess x_0 ({x_0.dtype}) must have the same data type as matrix A ({A.dtype}).")
    return x_0


###
# Functions
###


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    tolerance = A.dtype.type(tolerance)

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_p[k]
            x_n[j] = (b[j] - sum) / A[j, j]

        solution.error = np.max(np.abs(x_n - x_p))
        if compute_residuals:
            solution.res_history.append(A @ x_n - b)
        if solution.error < tolerance:
            solution.converged = True
            break

        x_p[:] = x_n

    solution.x = x_n
    return solution


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    tolerance = A.dtype.type(tolerance)

    n = A.shape[0]
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_n[k]
            x_n[j] = (b[j] - sum) / A[j, j]

        solution.error = np.max(np.abs(x_n - x_0))
        if compute_residuals:
            solution.res_history.append(A @ x_n - b)
        if solution.error < tolerance:
            solution.converged = True
            break

        x_0[:] = x_n

    solution.x = x_n
    return solution


def successive_over_relaxation(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    omega: float = 1.0,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    if not (0.0 < omega < 2.0):
        raise ValueError(f"Relaxation factor omega must be in the range (0, 2) but is {omega}.")
    omega = A.dtype.type(omega)
    tolerance = A.dtype.type(tolerance)

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_n[k]
            x_n[j] = x_p[j] + omega * ((b[j] - sum) / A[j, j] - x_p[j])

        solution.error = np.max(np.abs(x_n - x_p))
        if compute_residuals:
            solution.res_history.append(A @ x_n - b)
        if solution.error < tolerance:
            solution.converged = True
            break

        x_p[:] = x_n

    solution.x = x_n
    return solution


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    epsilon: float = 1e-12,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    epsilon = A.dtype.type(epsilon)
    tolerance = A.dtype.type(tolerance)

    r = b - A @ x_0
    g = r.copy()
    Ag = np.zeros_like(g)
    x = x_0.copy()
    rsold = np.dot(r, r)
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        Ag[:] = A @ g
        gAg = np.dot(g, Ag)
        alpha = rsold / max(gAg, epsilon)  # cap denom to avoid div-by-zero
        x += alpha * g
        r -= alpha * Ag
        rsnew = np.dot(r, r)

        solution.error = np.max(np.abs(r))
        if compute_residuals:
            solution.res_history.append(A @ x - b)
        if solution.error < tolerance:
            solution.converged = True
            break

        beta = rsnew / max(rsold, epsilon)  # cap denom to avoid div-by-zero
        g = r + beta * g
        rsold = rsnew

    solution.x = x
    return solution


def minimum_residual(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    epsilon: float = 1e-12,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    epsilon = A.dtype.type(epsilon)
    tolerance = A.dtype.type(tolerance)

    n = A.shape[0]
    r = b - A @ x_0
    p0 = r.copy()
    s0 = A @ p0
    p1 = p0.copy()
    s1 = s0.copy()
    x = x_0.copy()
    p2 = np.zeros(n)
    s2 = np.zeros(n)
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        p2[:] = p1
        p1[:] = p0
        s2[:] = s1
        s1[:] = s0
        s1_prod = np.dot(s1, s1)
        s2_prod = np.dot(s2, s2)

        alpha = np.dot(r, s1) / max(s1_prod, epsilon)  # cap denom to avoid div-by-zero
        x += alpha * p1
        r -= alpha * s1

        solution.error = np.max(np.abs(r))
        if solution.error < tolerance:
            solution.converged = True
            break

        p0[:] = s1
        s0[:] = A @ s1

        beta = np.dot(s0, s1) / max(s1_prod, epsilon)  # cap denom to avoid div-by-zero
        p0 -= beta * p1
        s0 -= beta * s1

        if i > 0:
            gamma = np.dot(s0, s2) / max(s2_prod, epsilon)  # cap denom to avoid div-by-zero
            p0 -= gamma * p2
            s0 -= gamma * s2

    solution.x = x
    return solution


###
# Classes
###


class JacobiSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, compute_errors, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations

    @override
    def _compute_impl(self, A: np.ndarray):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        result = jacobi(
            self._A,
            b,
            x_0,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class GaussSeidelSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations

    @override
    def _compute_impl(self, A: np.ndarray):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        result = gauss_seidel(
            self._A,
            b,
            x_0,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class SORSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        omega: float = 1.0,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations
        self._omega: float = omega

    @override
    def _compute_impl(self, A: np.ndarray):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        result = successive_over_relaxation(
            self._A,
            b,
            x_0,
            omega=self._omega,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class ConjugateGradientSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        epsilon: float = 1e-12,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations
        self._epsilon: float = epsilon

    @override
    def _compute_impl(self, A: np.ndarray):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        epsilon = self._epsilon
        result = conjugate_gradient(
            self._A,
            b,
            x_0,
            epsilon=epsilon,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class MinimumResidualSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        epsilon: float = 1e-12,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations
        self._epsilon: float = epsilon

    @override
    def _compute_impl(self, A: np.ndarray):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        epsilon = self._epsilon
        result = minimum_residual(
            self._A,
            b,
            x_0,
            epsilon=epsilon,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


# ---------------------------
# Example usage / sanity checks
# ---------------------------
if __name__ == "__main__":
    # ----------------------------
    np.set_printoptions(linewidth=2000, precision=10, threshold=10000, suppress=False)  # Suppress scientific notation

    # ----------------------------
    # dtype = np.float64
    dtype = np.float32
    print("----------------------------")
    print(f"dtype: {dtype}")

    # ----------------------------
    epsilon = np.finfo(dtype).eps
    tolerance = np.finfo(dtype).eps
    max_iterations = 1000
    print("----------------------------")
    print(f"epsilon: {epsilon}")
    print(f"tolerance: {tolerance}")
    print(f"max_iterations: {max_iterations}")

    # ----------------------------
    A = np.array([[2.0, 1.0], [1.0, 4.0]], dtype=dtype)  # symmetric positive-definite matrix
    b = np.array([1.0, 2.0], dtype=dtype)  # right-hand side in range-space of A
    print("----------------------------")
    print(f"\nA {A.shape}[{A.dtype}]:\n{A}\n")
    print(f"\nb {b.shape}[{b.dtype}]:\n{b}\n")

    # ----------------------------
    lambdas_A = np.linalg.eigvals(A)
    rank_A = np.linalg.matrix_rank(A)
    cond_A = np.linalg.cond(A)
    print("----------------------------")
    print(f"lambda(A): {lambdas_A}")
    print(f"rank(A): {rank_A}")
    print(f"cond(A): {cond_A}")

    # ---------------------------- Reference
    x_np = np.linalg.solve(A, b)
    r_np = A @ x_np - b
    print("----------------------------")
    print(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

    # ---------------------------- Jacobi
    # jac = jacobi(A=A, b=b, x_0=None, tolerance=tolerance, max_iterations=max_iterations)
    jac_solver = JacobiSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations)
    jac_solver.compute(A)
    jac = jac_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_jac = A @ jac.x - b
    print("----------------------------")
    print(f"Jacobi:  converged: {jac.converged}")
    print(f"Jacobi: iterations: {jac.iterations}")
    print(f"Jacobi:      error: {jac.error}")
    print("----------------------------")
    print(f"\nx_jac {jac.x.shape}[{jac.x.dtype}]:\n{jac.x}\n")

    # ---------------------------- Gauss-Seidel
    # gs = gauss_seidel(A=A, b=b, x_0=None, tolerance=tolerance, max_iterations=max_iterations)
    gs_solver = GaussSeidelSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations)
    gs_solver.compute(A)
    gs = gs_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_gs = A @ gs.x - b
    print("----------------------------")
    print(f"Gauss-Seidel:  converged: {gs.converged}")
    print(f"Gauss-Seidel: iterations: {gs.iterations}")
    print(f"Gauss-Seidel:      error: {gs.error}")
    print("----------------------------")
    print(f"\nx_gs {gs.x.shape}[{gs.x.dtype}]:\n{gs.x}\n")

    # ---------------------------- Successive Over-Relaxation
    # sor = successive_over_relaxation(A=A, b=b, x_0=None, omega=1.25, tolerance=tolerance, max_iterations=max_iterations)
    sor_solver = SORSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations, omega=1.25)
    sor_solver.compute(A)
    sor = sor_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_sor = A @ sor.x - b
    print("----------------------------")
    print(f"Successive Over-Relaxation:  converged: {sor.converged}")
    print(f"Successive Over-Relaxation: iterations: {sor.iterations}")
    print(f"Successive Over-Relaxation:      error: {sor.error}")
    print("----------------------------")
    print(f"\nx_sor {sor.x.shape}[{sor.x.dtype}]:\n{sor.x}\n")

    # ---------------------------- Conjugate Gradient
    # cg = conjugate_gradient(A=A, b=b, x_0=None, epsilon=epsilon, tolerance=tolerance, max_iterations=max_iterations)
    cg_solver = ConjugateGradientSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations, epsilon=epsilon)
    cg_solver.compute(A)
    cg = cg_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_cg = A @ cg.x - b
    print("----------------------------")
    print(f"Conjugate Gradient:  converged: {cg.converged}")
    print(f"Conjugate Gradient: iterations: {cg.iterations}")
    print(f"Conjugate Gradient:      error: {cg.error}")
    print("----------------------------")
    print(f"\nx_cg {cg.x.shape}[{cg.x.dtype}]:\n{cg.x}\n")

    # ---------------------------- Minimum Residual
    # minres = minimum_residual(A=A, b=b, x_0=None, epsilon=epsilon, tolerance=tolerance, max_iterations=max_iterations)
    minres_solver = MinimumResidualSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations, epsilon=epsilon)
    minres_solver.compute(A)
    minres = minres_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_minres = A @ minres.x - b
    print("----------------------------")
    print(f"Minimum Residual:  converged: {minres.converged}")
    print(f"Minimum Residual: iterations: {minres.iterations}")
    print(f"Minimum Residual:      error: {minres.error}")
    print("----------------------------")
    print(f"\nx_minres {minres.x.shape}[{minres.x.dtype}]:\n{minres.x}\n")

    # ----------------------------
    r_np_l2 = np.linalg.norm(r_np, ord=2)
    r_jac_l2 = np.linalg.norm(r_jac, ord=2)
    r_gs_l2 = np.linalg.norm(r_gs, ord=2)
    r_sor_l2 = np.linalg.norm(r_sor, ord=2)
    r_cg_l2 = np.linalg.norm(r_cg, ord=2)
    r_minres_l2 = np.linalg.norm(r_minres, ord=2)

    # ----------------------------
    r_np_infnorm = np.linalg.norm(r_np, ord=np.inf)
    r_jac_infnorm = np.linalg.norm(r_jac, ord=np.inf)
    r_gs_infnorm = np.linalg.norm(r_gs, ord=np.inf)
    r_sor_infnorm = np.linalg.norm(r_sor, ord=np.inf)
    r_cg_infnorm = np.linalg.norm(r_cg, ord=np.inf)
    r_minres_infnorm = np.linalg.norm(r_minres, ord=np.inf)

    # ----------------------------
    print("----------------------------")
    print(f"r_np_l2: {r_np_l2}")
    print(f"r_jac_l2: {r_jac_l2}")
    print(f"r_gs_l2: {r_gs_l2}")
    print(f"r_sor_l2: {r_sor_l2}")
    print(f"r_cg_l2: {r_cg_l2}")
    print(f"r_minres_l2: {r_minres_l2}")
    print("----------------------------")
    print(f"r_np_infnorm: {r_np_infnorm}")
    print(f"r_jac_infnorm: {r_jac_infnorm}")
    print(f"r_gs_infnorm: {r_gs_infnorm}")
    print(f"r_sor_infnorm: {r_sor_infnorm}")
    print(f"r_cg_infnorm: {r_cg_infnorm}")
    print(f"r_minres_infnorm: {r_minres_infnorm}")
