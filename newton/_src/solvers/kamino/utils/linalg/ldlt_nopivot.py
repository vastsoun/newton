###########################################################################
# KAMINO: Utilities: Linear Algebra: standard LDLT w/o pivoting
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
    "LDLTNoPivot",
    "compute_ldlt_lower_reconstruct",
    "compute_ldlt_lower_solve_inplace",
    "compute_ldlt_nopivot_lower",
    "compute_ldlt_upper_solve_inplace",
]


###
# LDLT w/o pivoting
###


def compute_ldlt_nopivot_lower(
    A: np.ndarray, tol: float | None = None, check_symmetry: bool = True, use_zero_correction: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    tol = _make_tolerance(tol, dtype=A.dtype)

    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    D = np.zeros(n, dtype=A.dtype)
    for k in range(n):
        accum = A.dtype.type(0)
        for s in range(k):
            accum += (L[k, s] * L[k, s]) * D[s]
        Dk = A[k, k] - accum
        if np.abs(Dk) < tol:
            if use_zero_correction:
                Dk = tol
            else:
                raise np.linalg.LinAlgError(
                    f"Zero (or tiny) pivot encountered at k={k}: D[k]={Dk}. "
                    "Matrix may be indefinite or require pivoting (e.g., Bunch-Kaufman)."
                )
        D[k] = Dk
        invDk = A.dtype.type(1) / Dk
        for i in range(k + 1, n):
            accum2 = A.dtype.type(0)
            for j in range(k):
                accum2 += L[i, j] * L[k, j] * D[j]
            L[i, k] = (A[i, k] - accum2) * invDk
    return L, D


def compute_ldlt_lower_solve_inplace(L: np.ndarray, D: np.ndarray, x: np.ndarray, *, unit_diagonal: bool = True):
    """
    In-place solve for A x = b given LDLᵀ factorization A = L diag(D) Lᵀ.
    Overwrites x (initially b) with the solution.

    Parameters
    ----------
    L : (n,n) ndarray
        Lower-triangular factor (unit diagonal if unit_diagonal=True).
    D : (n,) or (n,n) ndarray
        Diagonal of D or full diagonal matrix.
    x : (n,) ndarray
        On input: rhs b. On output: solution x.
    unit_diagonal : bool, default True
        If False, divides by L[i,i] during forward/backward passes.

    Returns
    -------
    x : (n,) ndarray
        The same array, solved in-place.
    """
    n = L.shape[0]
    # --- Forward substitution: solve L y = b, store y in x ---
    for i in range(n):
        s = x[i]
        # subtract L[i, :i] @ x[:i]
        for j in range(i):
            s -= L[i, j] * x[j]
        if not unit_diagonal:
            s /= L[i, i]
        x[i] = s

    # --- Diagonal scaling: solve D z = y, store z in x ---
    if D.ndim == 1:
        for i in range(n):
            x[i] /= D[i]
    else:
        for i in range(n):
            x[i] /= D[i, i]

    # --- Backward substitution: solve Lᵀ x = z, store back in x ---
    for i in range(n - 1, -1, -1):
        s = x[i]
        # subtract (Lᵀ)[i, i+1:] @ x[i+1:] == L[i+1:, i] @ x[i+1:]
        for j in range(i + 1, n):
            s -= L[j, i] * x[j]
        if not unit_diagonal:
            s /= L[i, i]
        x[i] = s
    return x


def compute_ldlt_upper_solve_inplace(U: np.ndarray, D: np.ndarray, x: np.ndarray, *, unit_diagonal: bool = True):
    """
    In-place solve for A x = b given LDLᵀ factorization A = L diag(D) Lᵀ.
    Overwrites x (initially b) with the solution.

    Parameters
    ----------
    U : (n,n) ndarray
        Upper-triangular factor (unit diagonal if unit_diagonal=True).
    D : (n,) or (n,n) ndarray
        Diagonal of D or full diagonal matrix.
    x : (n,) ndarray
        On input: rhs b. On output: solution x.
    unit_diagonal : bool, default True
        If False, divides by U[i,i] during forward/backward passes.

    Returns
    -------
    x : (n,) ndarray
        The same array, solved in-place.
    """
    n = U.shape[0]
    # --- Forward substitution: solve Uᵀ y = b, store y in x ---
    for i in range(n - 1, -1, -1):
        s = x[i]
        # subtract (Uᵀ)[i, i+1:] @ x[i+1:] == U[i+1:, i] @ x[i+1:]
        for j in range(i + 1, n):
            s -= U[j, i] * x[j]
        if not unit_diagonal:
            s /= U[i, i]
        x[i] = s

    # --- Diagonal scaling: solve D z = y, store z in x ---
    if D.ndim == 1:
        for i in range(n):
            x[i] /= D[i]
    else:
        for i in range(n):
            x[i] /= D[i, i]

    # --- Backward substitution: solve U x = z, store back in x ---
    for i in range(n - 1, -1, -1):
        s = x[i]
        # subtract U[i, i+1:] @ x[i+1:]
        for j in range(i + 1, n):
            s -= U[i, j] * x[j]
        if not unit_diagonal:
            s /= U[i, i]
        x[i] = s
    return x


###
# Reconstruction
###


def compute_ldlt_lower_reconstruct(L: np.ndarray, D: np.ndarray) -> np.ndarray:
    return L @ np.diag(D) @ L.T


def compute_ldlt_upper_reconstruct(U: np.ndarray, D: np.ndarray) -> np.ndarray:
    return U @ np.diag(D) @ U.T


###
# Factorizer
###


class LDLTNoPivot(MatrixFactorizer):
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

        # Declare optional unpacked factors
        self.L: np.ndarray | None = None
        self.D: np.ndarray | None = None

        # Raise error if upper requested since it's not supported
        if upper:
            raise ValueError("Upper triangular form is not yet supported")

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
            self._matrix, self._diagonals = compute_ldlt_nopivot_lower(
                A=A, tol=self._tolerance, use_zero_correction=True
            )
            self._sign = MatrixSign.PositiveDef
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {e!s}") from e

    def _unpack_impl(self) -> None:
        self.L, self.D = self._matrix, np.diag(self._diagonals)

    def _get_unpacked_impl(self) -> Any:
        return self.L, self.D

    def _solve_inplace_impl(self, x: np.ndarray):
        compute_ldlt_lower_solve_inplace(self._matrix, self._diagonals, x)

    def _reconstruct_impl(self) -> np.ndarray:
        L, D = self.unpacked()
        return L @ D @ L.T
