###########################################################################
# KAMINO: Utilities: Linear Algebra: LDLT w/ Bunch-Kaufman pivoting
###########################################################################

from typing import Any

import numpy as np

from newton._src.solvers.kamino.utils.linalg.factorizer import MatrixFactorizer, MatrixSign
from newton._src.solvers.kamino.utils.linalg.matrix import (
    _make_tolerance,
    assert_is_square_matrix,
    assert_is_symmetric_matrix,
)

###
# Module interface
###

__all__ = [
    "LDLTBunchKaufman",
    "compute_ldlt_bk_lower",
    "compute_ldlt_bk_lower_reconstruct",
    "compute_ldlt_bk_lower_solve",
    "unpack_ldlt_bk_lower",
]


###
# Constants
###

DEFAULT_ALPHA = float((1.0 + np.sqrt(17.0)) / 8.0)
"""Default alpha for Bunch-Kaufman pivoting."""


###
# Utilities
###


def _swap_rows_cols_sym(A, i, j):
    if i == j:
        return
    n = A.shape[0]
    for k in range(n):
        A[i, k], A[j, k] = A[j, k], A[i, k]
    for k in range(n):
        A[k, i], A[k, j] = A[k, j], A[k, i]


###
# Factorization
###


def compute_ldlt_bk_lower(
    A: np.ndarray,
    tol: float | None = None,
    itype: np.dtype = np.int64,
    alpha: float = DEFAULT_ALPHA,
    check_symmetry: bool = True,
    use_zero_correction: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    tol = _make_tolerance(tol, dtype=A.dtype)

    n = A.shape[0]

    L = np.eye(n, dtype=A.dtype)
    D = np.zeros((n, n), dtype=A.dtype)
    perm = np.arange(n, dtype=itype)

    k = 0
    while k < n:
        absakk = np.abs(A[k, k])
        imax = k
        colmax = A.dtype.type(0)
        if k + 1 < n:
            imax = k + 1
            colmax = np.abs(A[imax, k])
            for i in range(k + 2, n):
                v = np.abs(A[i, k])
                if v > colmax:
                    colmax = v
                    imax = i
        if np.max((absakk, colmax)) < tol:
            kp = k
            pivot_size = 1
        elif absakk >= alpha * colmax:
            kp = k
            pivot_size = 1
        else:
            rowmax = A.dtype.type(0)
            for j in range(k, n):
                if j != imax:
                    v = abs(A[imax, j])
                    if v > rowmax:
                        rowmax = v
            if np.abs(A[imax, imax]) >= alpha * rowmax:
                kp = imax
                pivot_size = 1
            else:
                kp = imax
                pivot_size = 2
        if pivot_size == 1:
            if kp != k:
                _swap_rows_cols_sym(A, k, kp)
                perm[k], perm[kp] = perm[kp], perm[k]
                if k > 0:
                    for j in range(k):
                        L[k, j], L[kp, j] = L[kp, j], L[k, j]
            Dkk = A[k, k]
            if np.abs(Dkk) < tol:
                if use_zero_correction:
                    Dkk = tol
                else:
                    raise np.linalg.LinAlgError(f"Near-zero 1x1 pivot at k={k}: {Dkk}")
            D[k, k] = Dkk
            invDkk = A.dtype.type(1) / Dkk
            for i in range(k + 1, n):
                L[i, k] = A[i, k] * invDkk
            for i in range(k + 1, n):
                li = L[i, k]
                for j in range(i, n):
                    A[j, i] = A[j, i] - li * Dkk * L[j, k]
                    A[i, j] = A[j, i]
            k += 1
        else:
            if kp != k + 1:
                _swap_rows_cols_sym(A, k + 1, kp)
                perm[k + 1], perm[kp] = perm[kp], perm[k + 1]
                if k > 0:
                    for j in range(k):
                        L[k + 1, j], L[kp, j] = L[kp, j], L[k + 1, j]
            a = A[k, k]
            b = A[k, k + 1]
            d = A[k + 1, k + 1]
            det = a * d - b * b
            if np.abs(det) < tol:
                if use_zero_correction:
                    det = tol
                else:
                    raise np.linalg.LinAlgError(f"Near-singular 2x2 pivot at k={k}: det={det}")
            D[k, k] = a
            D[k, k + 1] = b
            D[k + 1, k] = b
            D[k + 1, k + 1] = d
            for i in range(k + 2, n):
                Aik = A[i, k]
                Aip = A[i, k + 1]
                Lik = (Aik * d - Aip * b) / det
                Lip = (-Aik * b + Aip * a) / det
                L[i, k] = Lik
                L[i, k + 1] = Lip
            for i in range(k + 2, n):
                Lik = L[i, k]
                Lip = L[i, k + 1]
                for j in range(i, n):
                    Ljk = L[j, k]
                    Ljp = L[j, k + 1]
                    delta = (a * Lik * Ljk) + (b * Lik * Ljp) + (b * Lip * Ljk) + (d * Lip * Ljp)
                    A[j, i] = A[j, i] - delta
                    A[i, j] = A[j, i]
            L[k + 1, k] = 0.0
            k += 2

    return L, D, perm


###
# Linear systems
###


def compute_ldlt_bk_lower_solve(
    L: np.ndarray, D: np.ndarray, perm: np.ndarray, b: np.ndarray, tol: float | None = None
) -> np.ndarray:
    n = L.shape[0]

    tol = _make_tolerance(tol, dtype=L.dtype)

    b = np.asarray(b, dtype=L.dtype)
    y = b[perm]

    u = np.zeros(n, dtype=L.dtype)
    for i in range(n):
        s = y[i]
        for j in range(i):
            s -= L[i, j] * u[j]
        u[i] = s

    w = np.zeros(n, dtype=L.dtype)
    k = 0
    while k < n:
        two_by_two = (k + 1 < n) and (abs(D[k, k + 1]) > tol or abs(D[k + 1, k]) > tol)
        if not two_by_two:
            if abs(D[k, k]) < tol:
                raise np.linalg.LinAlgError(f"Singular 1x1 block at k={k}")
            w[k] = u[k] / D[k, k]
            k += 1
        else:
            a = D[k, k]
            b2 = D[k, k + 1]
            d = D[k + 1, k + 1]
            det = a * d - b2 * b2
            if abs(det) < tol:
                raise np.linalg.LinAlgError(f"Singular 2x2 block at k={k}")
            u0 = u[k]
            u1 = u[k + 1]
            w0 = (d * u0 - b2 * u1) / det
            w1 = (-b2 * u0 + a * u1) / det
            w[k] = w0
            w[k + 1] = w1
            k += 2

    z = np.zeros(n, dtype=L.dtype)
    for i in range(n - 1, -1, -1):
        s = w[i]
        for j in range(i + 1, n):
            s -= L[j, i] * z[j]
        z[i] = s

    x = np.zeros(n, dtype=L.dtype)
    for i in range(n):
        x[perm[i]] = z[i]

    return x


###
# Reconstruction
###


def compute_ldlt_bk_lower_reconstruct(L: np.ndarray, D: np.ndarray, perm: np.ndarray) -> np.ndarray:
    S = L @ D @ L.T
    A_hat = np.zeros_like(S)
    A_hat[np.ix_(perm, perm)] = S
    return A_hat


def unpack_ldlt_bk_lower(
    matrix: np.ndarray, diagonals: np.ndarray, permutations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P = np.eye(matrix.shape[0], dtype=matrix.dtype)[:, permutations]
    return matrix, diagonals, P


###
# Factorizer
###


class LDLTBunchKaufman(MatrixFactorizer):
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
        self._diagonals: np.ndarray | None = None
        self._permutations: np.ndarray | None = None

        # Declare optional unpacked factors
        self.L: np.ndarray | None = None
        self.D: np.ndarray | None = None
        self.P: np.ndarray | None = None

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

    def _factorize_impl(self, A: np.ndarray) -> None:
        try:
            self._matrix, self._diagonals, self._permutations = compute_ldlt_bk_lower(
                A=A, tol=self._tolerance, itype=self._itype, check_symmetry=False, use_zero_correction=True
            )
            print(f"LDLT BK: matrix:\n{self._matrix}\n")
            print(f"LDLT BK: diagonals:\n{self._diagonals}\n")
            print(f"LDLT BK: permutations:\n{self._permutations}\n")
            self._sign = MatrixSign.PositiveDef  # TODO: parse D to determine sign
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {e!s}") from e

    def _unpack_impl(self) -> None:
        self.L, self.D, self.P = unpack_ldlt_bk_lower(self._matrix, self._diagonals, self._permutations)

    def _get_unpacked_impl(self) -> Any:
        return self.L, self.D, self.P

    def _solve_inplace_impl(self, x: np.ndarray):
        b = np.asarray(x, dtype=self._matrix.dtype, copy=True)
        tmp = compute_ldlt_bk_lower_solve(self._matrix, self._diagonals, self._permutations, b, self._tolerance)
        x[:] = tmp

    def _reconstruct_impl(self) -> np.ndarray:
        return compute_ldlt_bk_lower_reconstruct(self._matrix, self._diagonals, self._permutations)
