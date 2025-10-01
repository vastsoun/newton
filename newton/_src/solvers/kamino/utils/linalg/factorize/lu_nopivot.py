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

"""KAMINO: Utilities: Linear Algebra: LU factorization w/o pivoting"""

import numpy as np

###
# Module interface
###

__all__ = [
    "lu_nopiv",
    "lu_nopiv_solve",
    "lu_nopiv_solve_backward_upper",
    "lu_nopiv_solve_forward_lower",
]


###
# Factorize
###


def lu_nopiv(A: np.ndarray, tol: float = 1e-12):
    """
    Doolittle-style LU factorization *without pivoting* (records the exact steps
    of Gaussian elimination without row swaps).

    For near-zero pivots, we *skip* elimination in that column (still a faithful
    record of no-pivot elimination, though not a strict LU if A is singular).
    """
    tol = A.dtype.type(tol)

    m, n = A.shape
    U = A.copy()
    L = np.eye(m, dtype=A.dtype)

    for k in range(min(m, n)):
        piv = U[k, k]
        if np.abs(piv) <= tol:
            # No pivoting allowed; skip this column (can't form multipliers here).
            continue
        for i in range(k + 1, m):
            L[i, k] = U[i, k] / piv
            U[i, k:n] -= L[i, k] * U[k, k:n]
            U[i, k] = 0.0
    return L, U


###
# Solve
###


def lu_nopiv_solve_forward_lower(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ly = b for unit-lower-triangular L (diag(L)=1).
    Works even if some columns of L were 'skipped' (multipliers left at 0).
    """
    if b.ndim == 1:
        b = b[:, None]
    m = L.shape[0]
    y = np.zeros_like(b)
    for i in range(m):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def lu_nopiv_solve_backward_upper(U: np.ndarray, y: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Solve Ux = y by back-substitution (for square, nonsingular U).
    """
    tol = U.dtype.type(tol)
    if y.ndim == 1:
        y = y[:, None]
    m, n = U.shape
    if m != n:
        raise ValueError("compute_lu_backward_upper expects square U.")
    x = np.zeros((n, y.shape[1]), dtype=U.dtype)
    for j in range(y.shape[1]):
        for i in range(n - 1, -1, -1):
            if np.abs(U[i, i]) <= tol:
                raise np.linalg.LinAlgError("Singular U encountered in back-substitution.")
            x[i, j] = (y[i, j] - U[i, i + 1 :] @ x[i + 1 :, j]) / U[i, i]
    return x


def lu_nopiv_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Solve Ax = b given LU factorization A = LU (w/o pivoting).
    """
    y = lu_nopiv_solve_forward_lower(L, b)
    x = lu_nopiv_solve_backward_upper(U, y, tol=tol)
    return x
