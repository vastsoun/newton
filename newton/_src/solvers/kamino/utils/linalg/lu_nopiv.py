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

from typing import Any

import numpy as np

from newton._src.solvers.kamino.utils.linalg.factorizer import MatrixFactorizer, MatrixSign

###
# Module interface
###

__all__ = [
    "LUNoPivot",
    "compute_lu_backward_upper",
    "compute_lu_forward_lower",
    "lu_nopiv",
]


###
# Factorization
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


def compute_lu_forward_lower(L: np.ndarray, b: np.ndarray):
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


def compute_lu_backward_upper(U: np.ndarray, y: np.ndarray, tol: float = 1e-12):
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
    return x.squeeze() if x.shape[1] == 1 else x


###
# Factorizer
###


class LUNoPivot(MatrixFactorizer):
    def __init__(
        self,
        A: np.ndarray | None = None,
        tol: float | None = None,
        dtype: np.dtype | None = None,
        itype: np.dtype | None = None,
        upper: bool = False,
        check_symmetry: bool = False,
        compute_error: bool = False,
    ):
        # Declare internal data structures
        self._L: np.ndarray | None = None
        self._U: np.ndarray | None = None

        # Call the parent constructor
        super().__init__(
            A=A,
            tol=tol,
            dtype=dtype,
            itype=itype,
            upper=upper,
            check_symmetry=check_symmetry,
            compute_error=compute_error,
        )

    @property
    def L(self) -> np.ndarray | None:
        return self._L

    @property
    def U(self) -> np.ndarray | None:
        return self._U

    def _factorize_impl(self, A: np.ndarray) -> None:
        # Attempt factorization of A
        try:
            self._L, self._U = lu_nopiv(A, self._tolerance)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"LU factorization failed: {e!s}") from e

        # Update internal meta-data
        self._matrix = self._L
        self._sign = MatrixSign.ZeroSign

    def _unpack_impl(self) -> None:
        pass

    def _get_unpacked_impl(self) -> Any:
        return self._matrix

    def _solve_inplace_impl(self, x: np.ndarray):
        b = np.asarray(x, dtype=self._matrix.dtype)
        y = compute_lu_forward_lower(self._L, b)
        x[:] = compute_lu_backward_upper(self._U, y, tol=self._tolerance)

    def _reconstruct_impl(self) -> np.ndarray:
        return self._L @ self._U


# # ---------------------------
# # Example usage / sanity checks
# # ---------------------------
# if __name__ == "__main__":
#     # dtype = np.float64
#     dtype = np.float32

#     A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=dtype)
#     print(f"\nA {A.shape}[{A.dtype}]:\n{A}\n")
#     b1 = np.array([3.0, 6.0, 9.0], dtype=dtype)  # in the span
#     print(f"\nb1 {b1.shape}[{b1.dtype}]:\n{b1}\n")
#     b2 = np.array([3.0, 6.0, 10.0], dtype=dtype)  # not in the span
#     print(f"\nb2 {b2.shape}[{b2.dtype}]:\n{b2}\n")

#     # ok1, ranks1, dbg1 = in_range_via_lu(A, b1)
#     # ok2, ranks2, dbg2 = in_range_via_lu(A, b2)
#     # print(ok1, ranks1)  # True,  (rankA, rankAb) like (1,1)
#     # print(ok2, ranks2)  # False, (1,2)

#     # For a square nonsingular A, we can also solve Ax=b with LU:
#     A2 = np.array([[2.0, 1.0], [3.0, 4.0]], dtype=dtype)
#     b3 = np.array([5.0, 11.0], dtype=dtype)
#     L, U = lu_nopiv(A2)
#     print(f"\nL {L.shape}[{L.dtype}]:\n{L}\n")
#     print(f"\nU {U.shape}[{U.dtype}]:\n{U}\n")
#     y = compute_lu_forward_lower(L, b3)
#     print(f"\ny {y.shape}[{y.dtype}]:\n{y}\n")
#     x = compute_lu_backward_upper(U, y)
#     print(f"\nx {x.shape}[{x.dtype}]:\n{x}\n")

#     lu = FactorizerLUNoPivot(A=A2, tol=1e-12)
#     x = lu.solve(b3)
#     print(f"\nx {x.shape}[{x.dtype}]:\n{x}\n")
