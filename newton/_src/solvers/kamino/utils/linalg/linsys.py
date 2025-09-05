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

"""TODO"""

import numpy as np


class FixedPointSolution:

    def __init__(self):
        self.x: np.ndarray | None = None
        self.error: float | None = None
        self.iterations: int = 0
        self.converged: bool = False


def jacobi(A, b, x0, tolerance, max_iterations, verbose=False):
    solution = FixedPointSolution()

    if verbose:
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        S = -np.linalg.inv(D) @ (L + U)
        print("[Jacobi]: Iteration Matrix: S:\n", S)

    x_p = x0.copy()
    x_n = x0.copy()
    n = A.shape[0]

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
