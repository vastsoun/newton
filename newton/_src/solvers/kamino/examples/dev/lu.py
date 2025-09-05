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


def lu_nopiv(A: np.ndarray, tol: float = 1e-12):
    """
    Doolittle-style LU factorization *without pivoting* (records the exact steps
    of Gaussian elimination without row swaps).

    For near-zero pivots, we *skip* elimination in that column (still a faithful
    record of no-pivot elimination, though not a strict LU if A is singular).
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    U = A.copy()
    L = np.eye(m, dtype=float)

    for k in range(min(m, n)):
        piv = U[k, k]
        if abs(piv) <= tol:
            # No pivoting allowed; skip this column (can’t form multipliers here).
            continue
        for i in range(k + 1, m):
            L[i, k] = U[i, k] / piv
            U[i, k:n] -= L[i, k] * U[k, k:n]
            U[i, k] = 0.0
    return L, U


def forward_sub_unit_lower(L: np.ndarray, b: np.ndarray):
    """
    Solve Ly = b for unit-lower-triangular L (diag(L)=1).
    Works even if some columns of L were 'skipped' (multipliers left at 0).
    """
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b[:, None]
    m = L.shape[0]
    y = np.zeros_like(b)
    for i in range(m):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def back_sub_upper(U: np.ndarray, y: np.ndarray, tol: float = 1e-12):
    """
    Solve Ux = y by back-substitution (for square, nonsingular U).
    """
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    m, n = U.shape
    if m != n:
        raise ValueError("back_sub_upper expects square U.")
    x = np.zeros((n, y.shape[1]), dtype=float)
    for j in range(y.shape[1]):
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) <= tol:
                raise np.linalg.LinAlgError("Singular U encountered in back-substitution.")
            x[i, j] = (y[i, j] - U[i, i + 1:] @ x[i + 1:, j]) / U[i, i]
    return x.squeeze() if x.shape[1] == 1 else x


def in_range_via_lu(A: np.ndarray, b: np.ndarray, tol: float = 1e-12):
    """
    Decide if b ∈ R(A) using the LU viewpoint:
      - Perform *no-pivot* LU (i.e., Gaussian elimination steps).
      - Apply the same row ops to b via y = L^{-1} b (forward-sub).
      - Inspect rows where U is (numerically) zero:
            if such a row has |y_i| > tol, then [A|b] has larger rank ⇒ b ∉ R(A).

    Returns:
        in_range: bool
        ranks: (rank(A), rank([A|b]))
        debug: dict with L, U, y
    """
    L, U = lu_nopiv(A, tol=tol)
    y = forward_sub_unit_lower(L, b)

    # Helper: row is (numerically) zero if all entries ≤ tol in magnitude
    zero_row = lambda row: np.all(np.abs(row) <= tol)

    m = U.shape[0]
    zero_mask = np.array([zero_row(U[i, :]) for i in range(m)], dtype=bool)
    rank_A = int(np.sum(~zero_mask))

    # If a zero row in U has a nonzero y_i, then the augmented rank increases.
    inconsistent = any(z and (abs(y[i]) > tol if np.ndim(y) == 1 else np.any(np.abs(y[i, :]) > tol))
                       for i, z in enumerate(zero_mask))
    rank_Ab = rank_A + (1 if inconsistent else 0)

    return (not inconsistent), (rank_A, rank_Ab), {"L": L, "U": U, "y": y}


# ---------------------------
# Example usage / sanity checks
# ---------------------------
if __name__ == "__main__":
    A = np.array([[1., 2.], [2., 4.], [3., 6.]])
    b1 = np.array([3., 6., 9.])    # in the span
    b2 = np.array([3., 6., 10.])   # not in the span

    ok1, ranks1, dbg1 = in_range_via_lu(A, b1)
    ok2, ranks2, dbg2 = in_range_via_lu(A, b2)
    print(ok1, ranks1)  # True,  (rankA, rankAb) like (1,1)
    print(ok2, ranks2)  # False, (1,2)

    # For a square nonsingular A, we can also solve Ax=b with LU:
    A2 = np.array([[2., 1.], [3., 4.]])
    b3 = np.array([5., 11.])
    L, U = lu_nopiv(A2)
    y = forward_sub_unit_lower(L, b3)
    x = back_sub_upper(U, y)
    print("x =", x)  # should solve A2 @ x ≈ b3
