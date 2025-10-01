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

"""KAMINO: Utilities: Linear Algebra: LLT (a.k.a. Cholesky) factorization"""

import numpy as np

from ..matrix import (
    _make_tolerance,
    assert_is_square_matrix,
    assert_is_symmetric_matrix,
)

###
# Module interface
###

__all__ = [
    "llt_std_lower",
    "llt_std_lower_reconstruct",
    "llt_std_lower_solve",
    "llt_std_lower_with_tolerance",
    "llt_std_lower_without_conditionals",
    "llt_std_upper",
    "llt_std_upper_reconstruct",
    "llt_std_upper_solve",
    "llt_std_upper_with_tolerance",
    "llt_std_upper_without_conditionals",
]


###
# Factorize
###


def llt_std_lower(A: np.ndarray, check_symmetry: bool = True) -> np.ndarray:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    L = np.zeros_like(A)

    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1):
            sum = A.dtype.type(0)
            for k in range(j):
                sum += L[i, k] * L[j, k]
            if i == j:
                val = A[i, i] - sum
                if val <= 0:
                    raise np.linalg.LinAlgError(
                        f"Matrix is not positive definite: Non-positive diagonal element detected at index {i}: {val}"
                    )
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - sum) / L[j, j]
    return L


def llt_std_upper(A: np.ndarray, check_symmetry: bool = True) -> np.ndarray:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    U = np.zeros_like(A)

    n = A.shape[0]
    for i in range(n):
        for j in range(i, n):
            sum = 0.0
            for k in range(i):
                sum += U[k, i] * U[k, j]
            if i == j:
                val = A[i, i] - sum
                if val <= 0.0:
                    raise np.linalg.LinAlgError(
                        f"Matrix is not positive definite: Non-positive diagonal element detected at index {i}: {val}"
                    )
                U[i, j] = np.sqrt(val)
            else:
                U[i, j] = (A[i, j] - sum) / U[i, i]
    return U


def llt_std_lower_with_tolerance(A: np.ndarray, tol: float | None = None, check_symmetry: bool = True) -> np.ndarray:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    tol = _make_tolerance(tol, dtype=A.dtype)

    L = np.zeros_like(A)

    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1):
            sum = A[i, j]
            for k in range(j):
                sum -= L[i, k] * L[j, k]
            if i == j:
                if sum < tol:
                    sum = tol  # Adjust small values to avoid negative sqrt
                L[i, j] = np.sqrt(sum)
            else:
                L[i, j] = sum / L[j, j]
    return L


def llt_std_upper_with_tolerance(A: np.ndarray, tol: float | None = None, check_symmetry: bool = True) -> np.ndarray:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    tol = _make_tolerance(tol, dtype=A.dtype)

    U = np.zeros_like(A)

    n = A.shape[0]
    for i in range(n):
        for j in range(i, n):
            sum = A[i, j]
            for k in range(i):
                sum -= U[k, i] * U[k, j]
            if i == j:
                if sum < tol:
                    sum = tol  # Adjust small values to avoid negative sqrt
                U[i, j] = np.sqrt(sum)
            else:
                U[i, j] = sum / U[i, i]
    return U


def llt_std_lower_without_conditionals(
    A: np.ndarray, tol: float | None = None, check_symmetry: bool = True
) -> np.ndarray:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    epsilon = _make_tolerance(tol, dtype=A.dtype)

    L = np.zeros_like(A)

    # Compute the Cholesky factorization sequentially
    n = A.shape[0]
    for i in range(n):
        # Compute diagonal element L[i, i]
        sum = A[i, i]
        for k in range(i):
            L_ik = L[i, k]
            sum -= L_ik * L_ik
        if tol:
            sum = max(sum, epsilon)  # Adjust small values to avoid negative sqrt
        else:
            if sum <= 0.0:
                raise np.linalg.LinAlgError(
                    f"Negative partial sum detected at diagonal i={i}: sum={sum}. Matrix may be indefinite."
                )
        L_ii = np.sqrt(sum)
        L[i, i] = L_ii

        # Compute off-diagonal elements in column i
        for j in range(i + 1, n):
            sum = A[j, i]
            for k in range(i):
                sum -= L[j, k] * L[i, k]
            L[j, i] = sum / L_ii
    return L


def llt_std_upper_without_conditionals(
    A: np.ndarray, tol: float | None = None, check_symmetry: bool = True
) -> np.ndarray:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    epsilon = _make_tolerance(tol, dtype=A.dtype)

    U = np.zeros_like(A)

    # Compute the Cholesky factorization sequentially
    n = A.shape[0]
    for i in range(n):
        # Compute diagonal element U[i, i]
        sum = A[i, i]
        for k in range(i):
            U_ik = U[k, i]
            sum -= U_ik * U_ik
        if tol:
            sum = max(sum, epsilon)  # Adjust small values to avoid negative sqrt
        else:
            if sum <= 0.0:
                raise np.linalg.LinAlgError(
                    f"Negative partial sum detected at diagonal i={i}: sum={sum}. Matrix may be indefinite."
                )
        U_ii = np.sqrt(sum)
        U[i, i] = U_ii

        # Compute off-diagonal elements in row i
        for j in range(i + 1, n):
            sum = A[i, j]
            for k in range(i):
                sum -= U[k, i] * U[k, j]
            U[i, j] = sum / U_ii
    return U


###
# Solve
###


def llt_std_lower_solve_inplace(L: np.ndarray, x: np.ndarray) -> None:
    # Ensure rhs is a numpy array with the same dtype as L
    x = np.asarray(x, dtype=L.dtype)

    # Allocate solution
    n = L.shape[0]
    y = np.zeros(n, dtype=L.dtype)

    # Forward pass: L @ y = b
    for i in range(n):
        sum = x[i]
        for j in range(i):
            sum -= L[i, j] * y[j]
        y[i] = sum / L[i, i]

    # Backward pass: L.T @ x = y
    for i in range(n - 1, -1, -1):
        sum = y[i]
        for j in range(i + 1, n):
            sum -= L[j, i] * x[j]
        x[i] = sum / L[i, i]


def llt_std_lower_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Ensure rhs is a numpy array with the same dtype as L
    b = np.asarray(b, dtype=L.dtype)

    # Allocate solution
    n = L.shape[0]
    y = np.zeros(n, dtype=L.dtype)
    x = np.zeros(n, dtype=L.dtype)

    # Forward pass: L @ y = b
    for i in range(n):
        sum = b[i]
        for j in range(i):
            sum -= L[i, j] * y[j]
        y[i] = sum / L[i, i]

    # Backward pass: L.T @ x = y
    for i in range(n - 1, -1, -1):
        sum = y[i]
        for j in range(i + 1, n):
            sum -= L[j, i] * x[j]
        x[i] = sum / L[i, i]

    # Return the final solution array
    return x


def llt_std_upper_solve(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Ensure rhs is a numpy array with the same dtype as U
    b = np.asarray(b, dtype=U.dtype)

    # Allocate solution
    n = U.shape[0]
    y = np.zeros(n, dtype=U.dtype)
    x = np.zeros(n, dtype=U.dtype)

    # Forward pass: U.T @ y = b
    for i in range(n):
        sum = b[i]
        for j in range(i):
            sum -= U[j, i] * y[j]
        y[i] = sum / U[i, i]

    # Backward pass: U @ x = y
    for i in range(n - 1, -1, -1):
        sum = y[i]
        for j in range(i + 1, n):
            sum -= U[i, j] * x[j]
        x[i] = sum / U[i, i]

    # Return the final solution array
    return x


###
# Reconstruction
###


def llt_std_lower_reconstruct(L: np.ndarray) -> np.ndarray:
    return L @ L.T


def llt_std_upper_reconstruct(U: np.ndarray) -> np.ndarray:
    return U @ U.T
