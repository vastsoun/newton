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

"""
KAMINO: Utilities: Linear Algebra: Block LDLT
"""

from typing import Any

import numpy as np

from newton._src.solvers.kamino.utils.linalg.factorizer import MatrixFactorizer
from newton._src.solvers.kamino.utils.linalg.ldlt_nopivot import (
    compute_ldlt_lower_solve_inplace,
    compute_ldlt_upper_solve_inplace,
)
from newton._src.solvers.kamino.utils.linalg.matrix import (
    assert_is_square_matrix,
    assert_is_symmetric_matrix,
)

###
# Module interface
###

__all__ = [
    "compute_ldlt_blocked_lower",
    "compute_ldlt_blocked_upper",
]


###
# Factorization
###


def compute_ldlt_blocked_lower(
    A: np.ndarray, block_size: int = 1, check_symmetric: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a block column-based LDLT factorization of a symmetric positive definite matrix A.

    Parameters:
    A (ndarray): Symmetric positive definite matrix of shape (n, n)
    block_size (int): Block size for the factorization

    Returns:
    L (ndarray): Unit lower triangular matrix (n, n)
    D (ndarray): Block diagonal matrix (n, n)
    """
    assert_is_square_matrix(A)
    if check_symmetric:
        assert_is_symmetric_matrix(A)

    n = A.shape[0]

    if n % block_size != 0:
        raise ValueError("Block size must divide the matrix size evenly.")

    L = np.eye(n)
    D = np.zeros_like(A)

    for k in range(0, n, block_size):
        # Define current block A_kk
        end = k + block_size
        A_kk = A[k:end, k:end].copy()

        # Update A_kk using previous blocks
        for s in range(0, k, block_size):
            D_ss = D[s : s + block_size, s : s + block_size]
            L_ks = L[k:end, s : s + block_size]
            A_kk -= L_ks @ D_ss @ L_ks.T

        # Factorize A_kk into L_kk and D_kk
        # We use the simple LDLᵀ factorization on the block A_kk
        L_kk = np.eye(block_size)
        D_kk = np.zeros((block_size, block_size))

        for j in range(block_size):
            sum_LDL = sum(L_kk[j, p] ** 2 * D_kk[p, p] for p in range(j))
            D_kk[j, j] = A_kk[j, j] - sum_LDL

            for i in range(j + 1, block_size):
                sum_LDL = sum(L_kk[i, p] * L_kk[j, p] * D_kk[p, p] for p in range(j))
                L_kk[i, j] = (A_kk[i, j] - sum_LDL) / D_kk[j, j]

        # Store L_kk and D_kk
        L[k:end, k:end] = L_kk
        D[k:end, k:end] = D_kk

        # Update below-diagonal blocks
        for i in range(end, n, block_size):
            A_ik = A[i : i + block_size, k:end].copy()
            for s in range(0, k, block_size):
                D_ss = D[s : s + block_size, s : s + block_size]
                L_is = L[i : i + block_size, s : s + block_size]
                L_ks = L[k:end, s : s + block_size]
                A_ik -= L_is @ D_ss @ L_ks.T

            # Solve for L_ik: A_ik = L_ik * D_kk * L_kk.T → L_ik = A_ik @ inv(D_kk) @ inv(L_kk.T)
            L_ik = A_ik @ np.linalg.inv(D_kk) @ np.linalg.inv(L_kk.T)
            L[i : i + block_size, k:end] = L_ik

    return L, D


def compute_ldlt_blocked_upper(
    A: np.ndarray, block_size: int = 1, check_symmetric: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a block column-based UDUT factorization of a symmetric positive definite matrix A.

    Parameters:
    A (ndarray): Symmetric positive definite matrix of shape (n, n)
    block_size (int): Block size for the factorization

    Returns:
    U (ndarray): Unit upper triangular matrix (n, n)
    D (ndarray): Block diagonal matrix (n, n)
    """
    assert_is_square_matrix(A)
    if check_symmetric:
        assert_is_symmetric_matrix(A)

    n = A.shape[0]

    if n % block_size != 0:
        raise ValueError("Block size must divide the matrix size evenly.")

    U = np.eye(n)
    D = np.zeros_like(A)

    for k in range(0, n, block_size):
        # Define current block A_kk
        end = k + block_size
        A_kk = A[k:end, k:end].copy()

        # Update A_kk using previous blocks
        for s in range(0, k, block_size):
            D_ss = D[s : s + block_size, s : s + block_size]
            U_sk = U[s : s + block_size, k:end]
            A_kk -= U_sk.T @ D_ss @ U_sk

        # Factorize A_kk into U_kk and D_kk
        # We use the simple UDUT factorization on the block A_kk
        U_kk = np.eye(block_size)
        D_kk = np.zeros((block_size, block_size))

        for j in range(block_size):
            sum_UDU = sum(U_kk[p, j] ** 2 * D_kk[p, p] for p in range(j))
            D_kk[j, j] = A_kk[j, j] - sum_UDU

            for i in range(j + 1, block_size):
                sum_UDU = sum(U_kk[p, i] * U_kk[p, j] * D_kk[p, p] for p in range(j))
                U_kk[j, i] = (A_kk[j, i] - sum_UDU) / D_kk[j, j]

        # Store U_kk and D_kk
        U[k:end, k:end] = U_kk
        D[k:end, k:end] = D_kk

        # Update above-diagonal blocks
        for i in range(end, n, block_size):
            A_ki = A[k:end, i : i + block_size].copy()
            for s in range(0, k, block_size):
                D_ss = D[s : s + block_size, s : s + block_size]
                U_sk = U[s : s + block_size, k:end]
                U_si = U[s : s + block_size, i : i + block_size]
                A_ki -= U_sk.T @ D_ss @ U_si

            # Solve for U_ki: A_ki = U_kk * D_kk * U_ki.T → U_ki = inv(U_kk) @ inv(D_kk) @ A_ki
            U_ki = np.linalg.inv(U_kk) @ np.linalg.inv(D_kk) @ A_ki
            U[k:end, i : i + block_size] = U_ki

    return U, D


###
# Factorizer
###


class LDLTBlocked(MatrixFactorizer):
    def __init__(
        self,
        A: np.ndarray | None = None,
        tol: float | None = None,
        dtype: np.dtype | None = None,
        itype: np.dtype | None = None,
        blocksize: int = 1,
        upper: bool = False,
        check_symmetry: bool = False,
        compute_error: bool = False,
    ):
        # Declare internal data structures
        self._diagonals: np.ndarray | None = None
        self._blocksize: int = blocksize

        # Declare optional unpacked factors
        self.L: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.D: np.ndarray | None = None

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
        if self._upper:
            self._matrix, self._diagonals = compute_ldlt_blocked_upper(A, self._blocksize)
        else:
            self._matrix, self._diagonals = compute_ldlt_blocked_lower(A, self._blocksize)

    def _unpack_impl(self) -> None:
        if self._upper:
            self.U, self.D = self._matrix, self._diagonals
        else:
            self.L, self.D = self._matrix, self._diagonals

    def _get_unpacked_impl(self) -> Any:
        if self._upper:
            return self.U, self.D
        return self.L, self.D

    def _solve_inplace_impl(self, x: np.ndarray):
        b = np.asarray(x, dtype=self._matrix.dtype, copy=True)
        if self._upper:
            compute_ldlt_upper_solve_inplace(self._matrix, self._diagonals, b)
        else:
            compute_ldlt_lower_solve_inplace(self._matrix, self._diagonals, b)
        x[:] = b

    def _reconstruct_impl(self) -> np.ndarray:
        if self._upper:
            U, D = self.unpacked()
            return U @ D @ U.T
        L, D = self.unpacked()
        return L @ D @ L.T
