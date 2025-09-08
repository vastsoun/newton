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

"""KAMINO: Utilities: Linear Algebra: Factorizer base class"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from newton._src.solvers.kamino.utils.linalg.linear import ComputationInfo
from newton._src.solvers.kamino.utils.linalg.matrix import (
    MatrixSign,
    _make_tolerance,
    assert_is_square_matrix,
    assert_is_symmetric_matrix,
)

###
# Types
###


class MatrixFactorizer(ABC):
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
        self._source: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._errors: np.ndarray | None = None

        # Initialize internal meta-data
        self._tolerance: float | None = tol
        self._dtype: np.dtype | None = dtype
        self._itype: np.dtype | None = itype
        self._sign: MatrixSign = MatrixSign.ZeroSign
        self._info: ComputationInfo = ComputationInfo.Success
        self._upper: bool = upper

        # Initialize internal flags
        self._success: bool = False
        self._has_factors: bool = False
        self._has_unpacked: bool = False

        # If a matrix is provided, proceed with its factorization
        if A is not None:
            self.factorize(A=A, tol=tol, check_symmetry=check_symmetry, compute_error=compute_error)

    def _check_has_factorization(self) -> None:
        """Checks if the factorization has been computed, otherwise raises error."""
        if not self._has_factors:
            raise ValueError("A factorization has not been computed!")

    def _compute_errors(self, A: np.ndarray) -> float:
        """Computes the reconstruction error of the factorization."""
        A_rec = self.reconstructed()
        return A - A_rec

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def itype(self) -> np.dtype:
        return self._itype

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    @property
    def errors(self) -> np.ndarray | None:
        return self._errors

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def sign(self) -> MatrixSign:
        return self._sign

    @property
    def success(self) -> bool:
        return self._success

    ###
    # Internals
    ###

    @abstractmethod
    def _factorize_impl(self, A: np.ndarray) -> None:
        raise NotImplementedError("Factorization implementation is not provided.")

    @abstractmethod
    def _unpack_impl(self) -> None:
        raise NotImplementedError("Unpacking implementation is not provided.")

    @abstractmethod
    def _get_unpacked_impl(self) -> Any:
        raise NotImplementedError("Getting unpacked factors implementation is not provided.")

    @abstractmethod
    def _solve_inplace_impl(self, x: np.ndarray):
        raise NotImplementedError("In-place solving implementation is not provided.")

    @abstractmethod
    def _reconstruct_impl(self) -> np.ndarray:
        raise NotImplementedError("Reconstruction implementation is not provided.")

    ###
    # Public API
    ###

    def factorize(
        self,
        A: np.ndarray,
        tol: float | None = None,
        itype: np.dtype | None = None,
        check_symmetry: bool = False,
        compute_error: bool = False,
    ):
        """
        Performs the factorization of a square symmetric matrix `A`.

        Args:
            A (np.ndarray): The input square symmetric matrix to factorize.
            tol (float, optional): The tolerance for convergence.
            itype (np.dtype, optional): The integer type to use for internal computations.
            check_symmetry (bool, optional): Whether to check if the matrix is symmetric.
            compute_error (bool, optional): Whether to compute the reconstruction error.

        Raises:
            ValueError: If the input matrix is not square or not symmetric (if checked).
        """
        assert_is_square_matrix(A)
        if check_symmetry:
            assert_is_symmetric_matrix(A)

        # Configure data types
        self._dtype = A.dtype
        if itype is not None:
            self._itype = itype
        else:
            self._itype = np.int64

        # Override the current tolerance if provided
        if tol is not None:
            self._tolerance = _make_tolerance(tol, dtype=self._dtype)

        # Factorize the specified matrix (i.e. as np.ndarray)
        self._factorize_impl(A)

        # Update internal meta-data
        self._source = A
        self._success = True
        self._has_factors = True
        self._has_unpacked = False

        # Optionally compute the reconstruction error
        if compute_error:
            self._errors = self._compute_errors(A)

    def solve_inplace(self, x: np.ndarray, tol: float | None = None):
        """Solves the linear system `A@x = b` using the LDLT factorization in-place."""
        self._check_has_factorization()
        if tol is not None:
            self._tolerance = _make_tolerance(tol, dtype=self._dtype)
        self._solve_inplace_impl(x)

    def solve(self, b: np.ndarray, tol: float | None = None) -> np.ndarray:
        """Solves the linear system `A@x = b` using the LDLT factorization."""
        x = b.astype(self._matrix.dtype, copy=True)
        self.solve_inplace(x, tol)
        return x

    def unpacked(self) -> Any:
        """Unpacks the factorization into the conventional LDLT form: L, D, P"""
        if not self._has_unpacked:
            self._check_has_factorization()
            self._has_unpacked = True
            self._unpack_impl()
        return self._get_unpacked_impl()

    def reconstructed(self) -> np.ndarray:
        """Reconstructs the original matrix from the factorization."""
        self._check_has_factorization()
        return self._reconstruct_impl()
