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


def in_range_no_pivot(A: np.ndarray, b: np.ndarray, tol: float = 1e-12):
    """
    Check if b is in the range (column space) of A by forming the augmented
    matrix Ab = [A | b] and performing Gaussian elimination without pivoting.

    Parameters
    ----------
    A : (m, n) ndarray
        Coefficient matrix.
    b : (m,) or (m,1) ndarray
        Right-hand side vector.
    tol : float
        Threshold for treating values as zero (numerical tolerance).

    Returns
    -------
    in_range : bool
        True iff rank(A) == rank([A|b]) under Gaussian elimination w/o pivoting.
    ranks : tuple[int, int]
        (rank_A, rank_Ab) computed from the row-echelon form obtained w/o pivoting.
    UAb : ndarray
        The upper-triangular (row-echelon-like) matrix after elimination on [A|b]
        (useful for debugging/inspection).

    Notes
    -----
    - No row swaps (no pivoting) are used, per the requirement.
    - This procedure is less numerically robust than pivoted elimination.
    - Rank is computed as the number of nonzero rows (by `tol`) in the echelon form.
    """
    tol = A.dtype.type(tol)
    if b.ndim == 1:
        b = b[:, None]
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have the same number of rows.")
    if A.dtype != b.dtype:
        raise ValueError("A and b must have the same dtype.")

    # Form augmented matrix [A | b]
    UAb = np.concatenate([A, b], axis=1)
    m, n_aug = UAb.shape
    n = n_aug - 1  # number of columns in A portion

    # Gaussian elimination without pivoting
    # (Equivalent to LU factorization steps without P; we only keep the U-like result.)
    for k in range(min(m, n)):
        pivot = UAb[k, k]
        if abs(pivot) <= tol:
            # No row swap allowed; skip elimination for this column
            continue
        for i in range(k + 1, m):
            factor = UAb[i, k] / pivot
            # subtract factor * row k from row i (only on the trailing part for efficiency)
            UAb[i, k:n_aug] -= factor * UAb[k, k:n_aug]

    # Helper: count nonzero rows under tolerance
    def rank_from_row_echelon(M, tol):
        # A row is nonzero if any absolute entry exceeds tol
        return int(np.sum(np.any(np.abs(M) > tol, axis=1)))

    # Rank of A: evaluate on the left block after the same row ops
    rank_A = rank_from_row_echelon(UAb[:, :n], tol)
    # Rank of augmented matrix
    rank_Ab = rank_from_row_echelon(UAb, tol)

    return (rank_A == rank_Ab), (rank_A, rank_Ab), UAb


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
    def zero_row(row):
        return np.all(np.abs(row) <= tol)

    m = U.shape[0]
    zero_mask = np.array([zero_row(U[i, :]) for i in range(m)], dtype=bool)
    rank_A = int(np.sum(~zero_mask))

    # If a zero row in U has a nonzero y_i, then the augmented rank increases.
    inconsistent = any(
        z and (abs(y[i]) > tol if np.ndim(y) == 1 else np.any(np.abs(y[i, :]) > tol)) for i, z in enumerate(zero_mask)
    )
    rank_Ab = rank_A + (1 if inconsistent else 0)

    return (not inconsistent), (rank_A, rank_Ab), {"L": L, "U": U, "y": y}


# --- Example ---
if __name__ == "__main__":
    # dtype = np.float64
    dtype = np.float32

    A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=dtype)
    print(f"\nA {A.shape}[{A.dtype}]:\n{A}\n")
    b1 = np.array([3.0, 6.0, 9.0], dtype=dtype)  # in the span
    print(f"\nb1 {b1.shape}[{b1.dtype}]:\n{b1}\n")
    b2 = np.array([3.0, 6.0, 10.0], dtype=dtype)  # not in the span
    print(f"\nb2 {b2.shape}[{b2.dtype}]:\n{b2}\n")

    print(f"b1 ∈ range(A): {in_range_no_pivot(A, b1)[0]}")  # True
    print(f"b2 ∈ range(A): {in_range_no_pivot(A, b2)[0]}")  # False
