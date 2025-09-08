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


def _check_compatibility(A: np.ndarray, b: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix A must be square (n x n) but has shape {A.shape}.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError(f"Vector b ({b.shape}) must have compatible dimensions with A ({A.shape}).")
    if b.dtype != A.dtype:
        raise ValueError(f"Vector b ({b.dtype}) must have the same data type as matrix A ({A.dtype}).")


def _check_initial_guess(b: np.ndarray, x_0: np.ndarray | None) -> np.ndarray:
    if x_0 is None:
        return np.zeros(b.shape[0], dtype=b.dtype)
    if x_0.ndim != 1 or x_0.shape[0] != b.shape[0]:
        raise ValueError(f"Initial guess x_0 ({x_0.shape}) must have compatible dimensions with b ({b.shape}).")
    if x_0.dtype != b.dtype:
        raise ValueError(f"Initial guess x_0 ({x_0.dtype}) must have the same data type as vector b ({b.dtype}).")
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
    verbose: bool = False,
) -> FixedPointSolution:
    # Check if inputs are compatible
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix A must be square (n x n) but has shape {A.shape}.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError(f"Vector b ({b.shape}) must have compatible dimensions with A ({A.shape}).")
    if b.dtype != A.dtype:
        raise ValueError(f"Vector b ({b.dtype}) must have the same data type as matrix A ({A.dtype}).")

    # Set initial guess to zeros if not provided
    if x_0 is None:
        x_0 = np.zeros_like(b)

    # TODO
    if verbose:
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        S = -np.linalg.inv(D) @ (L + U)
        print("[Jacobi]: Iteration Matrix: S:\n", S)

    # Initialize internals
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


def gauss_seidel(A, b, x0, tolerance, max_iterations, verbose=False):
    solution = FixedPointSolution()

    if verbose:
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        S = -np.linalg.inv(D + L) @ U
        print("[GaussSeidel]: Iteration Matrix: S:\n", S)

    x_n = x0.copy()
    n = A.shape[0]

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                if k != j:
                    sum_ += A[j, k] * x_n[k]
            x_n[j] = (b[j] - sum_) / A[j, j]

        solution.error = np.max(np.abs(x_n - x0))

        if solution.error < tolerance:
            solution.converged = True
            break

        x0[:] = x_n

    solution.x = x_n
    return solution


def successive_over_relaxation(A, b, x0, omega, tolerance, max_iterations, verbose=False):
    solution = FixedPointSolution()

    if verbose:
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        S = -np.linalg.inv(D + omega * L) @ (omega * U + (omega - 1.0) * D)
        print("[SuccessiveOverRelaxation]: Iteration Matrix: S:\n", S)

    x_p = x0.copy()
    x_n = x0.copy()
    n = A.shape[0]

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


def conjugate_gradient(A, b, x0, epsilon, tolerance, max_iterations):
    solution = FixedPointSolution()

    x = x0.copy()
    r = b - A @ x
    g = r.copy()
    n = A.shape[0]

    r_old = np.zeros(n)
    Ag = np.zeros(n)

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


def minimum_residual(A, b, x0, epsilon, tolerance, max_iterations):
    solution = FixedPointSolution()

    n = A.shape[0]
    x = x0.copy()
    r = b - A @ x0

    p0 = r.copy()
    s0 = A @ p0
    p1 = p0.copy()
    s1 = s0.copy()
    p2 = np.zeros(n)
    s2 = np.zeros(n)

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
    # dtype = np.float64
    dtype = np.float32
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
    jac = jacobi(A=A, b=b, x_0=None, tolerance=1e-15, max_iterations=1000, verbose=True)
    r_jac = A @ jac.x - b
    print("----------------------------")
    print(f"Jacobi:  converged: {jac.converged}")
    print(f"Jacobi: iterations: {jac.iterations}")
    print(f"Jacobi:      error: {jac.error}")
    print("----------------------------")
    print(f"\nx_jac {jac.x.shape}[{jac.x.dtype}]:\n{jac.x}\n")

    # ----------------------------
    r_np_l2 = np.linalg.norm(r_np, ord=2)
    r_jac_l2 = np.linalg.norm(r_jac, ord=2)
    r_np_infnorm = np.linalg.norm(r_np, ord=np.inf)
    r_jac_infnorm = np.linalg.norm(r_jac, ord=np.inf)
    print("----------------------------")
    print(f"r_np_l2: {r_np_l2}")
    print(f"r_jac_l2: {r_jac_l2}")
    print("----------------------------")
    print(f"r_np_infnorm: {r_np_infnorm}")
    print(f"r_jac_infnorm: {r_jac_infnorm}")
