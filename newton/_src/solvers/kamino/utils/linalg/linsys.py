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

from dataclasses import dataclass

import numpy as np

###
# Types
###


@dataclass
class FixedPointSolution:
    x: np.ndarray | None = None
    error: float = np.inf
    iterations: int = 0
    converged: bool = False


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
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                if k != j:
                    sum_ += A[j, k] * x_p[k]
            x_n[j] = (b[j] - sum_) / A[j, j]

        solution.error = np.max(np.abs(x_n - x_p))

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
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)

    n = A.shape[0]
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                if k != j:
                    sum_ += A[j, k] * x_n[k]
            x_n[j] = (b[j] - sum_) / A[j, j]

        solution.error = np.max(np.abs(x_n - x_0))

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
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    if not (0.0 < omega < 2.0):
        raise ValueError(f"Relaxation factor omega must be in the range (0, 2) but is {omega}.")

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                if k != j:
                    sum_ += A[j, k] * x_n[k]
            x_n[j] = x_p[j] + omega * ((b[j] - sum_) / A[j, j] - x_p[j])

        solution.error = np.max(np.abs(x_n - x_p))

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
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)

    n = A.shape[0]
    r = b - A @ x_0
    x = x_0.copy()
    g = r.copy()
    r_old = np.zeros(n)
    Ag = np.zeros(n)
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        r_old[:] = r
        Ag[:] = A @ g
        alpha = np.dot(r, r) / max(np.dot(g, Ag), epsilon)
        x += alpha * g
        r -= alpha * Ag

        solution.error = np.max(np.abs(r))

        if solution.error < tolerance:
            solution.converged = True
            break

        beta = np.dot(r, r) / max(np.dot(r_old, r_old), epsilon)
        g = r + beta * g

    solution.x = x
    return solution


def minimum_residual(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    epsilon: float = 1e-12,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)

    n = A.shape[0]
    r = b - A @ x_0
    x = x_0.copy()
    p0 = r.copy()
    s0 = A @ p0
    p1 = p0.copy()
    s1 = s0.copy()
    p2 = np.zeros(n)
    s2 = np.zeros(n)
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        p2[:] = p1
        p1[:] = p0
        s2[:] = s1
        s1[:] = s0

        alpha = np.dot(r, s1) / max(np.dot(s1, s1), epsilon)
        x += alpha * p1
        r -= alpha * s1

        solution.error = np.max(np.abs(r))

        if solution.error < tolerance:
            solution.converged = True
            break

        p0[:] = s1
        s0[:] = A @ s1

        beta = np.dot(s0, s1) / max(np.dot(s1, s1), epsilon)
        p0 -= beta * p1
        s0 -= beta * s1

        if i > 0:
            gamma = np.dot(s0, s2) / max(np.dot(s2, s2), epsilon)
            p0 -= gamma * p2
            s0 -= gamma * s2

    solution.x = x
    return solution


###
# Classes
###

# ---------------------------
# Example usage / sanity checks
# ---------------------------
if __name__ == "__main__":
    # ----------------------------
    dtype = np.float64
    # dtype = np.float32
    print("----------------------------")
    print(f"dtype: {dtype}")

    # ----------------------------
    A = np.array([[2.0, 1.0], [3.0, 4.0]], dtype=dtype)
    b = np.array([5.0, 11.0], dtype=dtype)
    print("----------------------------")
    print(f"\nA {A.shape}[{A.dtype}]:\n{A}\n")
    print(f"\nb {b.shape}[{b.dtype}]:\n{b}\n")

    # ---------------------------- Reference
    x_np = np.linalg.solve(A, b)
    r_np = A @ x_np - b
    print("----------------------------")
    print(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

    # ---------------------------- Jacobi
    jac = jacobi(A=A, b=b, x_0=None, tolerance=1e-15, max_iterations=1000)
    r_jac = A @ jac.x - b
    print("----------------------------")
    print(f"Jacobi:  converged: {jac.converged}")
    print(f"Jacobi: iterations: {jac.iterations}")
    print(f"Jacobi:      error: {jac.error}")
    print("----------------------------")
    print(f"\nx_jac {jac.x.shape}[{jac.x.dtype}]:\n{jac.x}\n")

    # ---------------------------- Gauss-Seidel
    gs = gauss_seidel(A=A, b=b, x_0=None, tolerance=1e-15, max_iterations=1000)
    r_gs = A @ gs.x - b
    print("----------------------------")
    print(f"Gauss-Seidel:  converged: {gs.converged}")
    print(f"Gauss-Seidel: iterations: {gs.iterations}")
    print(f"Gauss-Seidel:      error: {gs.error}")
    print("----------------------------")
    print(f"\nx_gs {gs.x.shape}[{gs.x.dtype}]:\n{gs.x}\n")

    # ---------------------------- Successive Over-Relaxation
    sor = successive_over_relaxation(A=A, b=b, x_0=None, omega=1.25, tolerance=1e-15, max_iterations=1000)
    r_sor = A @ sor.x - b
    print("----------------------------")
    print(f"Successive Over-Relaxation:  converged: {sor.converged}")
    print(f"Successive Over-Relaxation: iterations: {sor.iterations}")
    print(f"Successive Over-Relaxation:      error: {sor.error}")
    print("----------------------------")
    print(f"\nx_sor {sor.x.shape}[{sor.x.dtype}]:\n{sor.x}\n")

    # ---------------------------- Conjugate Gradient
    cg = conjugate_gradient(A=A, b=b, x_0=None, epsilon=1e-15, tolerance=1e-15, max_iterations=1000)
    r_cg = A @ cg.x - b
    print("----------------------------")
    print(f"Conjugate Gradient:  converged: {cg.converged}")
    print(f"Conjugate Gradient: iterations: {cg.iterations}")
    print(f"Conjugate Gradient:      error: {cg.error}")
    print("----------------------------")
    print(f"\nx_cg {cg.x.shape}[{cg.x.dtype}]:\n{cg.x}\n")

    # ---------------------------- Minimum Residual
    minres = minimum_residual(A=A, b=b, x_0=None, epsilon=1e-15, tolerance=1e-15, max_iterations=1000)
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
