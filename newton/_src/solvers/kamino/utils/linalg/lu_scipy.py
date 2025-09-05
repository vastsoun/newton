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

"""KAMINO: Utilities: Linear Algebra: LU factorization using SciPy"""

from typing import Any

import numpy as np
from scipy import linalg

from newton._src.solvers.kamino.utils.linalg.factorizer import MatrixFactorizer, MatrixSign

###
# Module interface
###

__all__ = [
    "LUSciPy",
]


###
# Factorizer
###


class LUSciPy(MatrixFactorizer):
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
        self._P: np.ndarray | None = None
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

    def _factorize_impl(self, A: np.ndarray) -> None:
        # Attempt factorization of A
        try:
            self._P, self._L, self._U = linalg.lu(A, permute_l=False)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {e!s}") from e

        # Update internal meta-data
        self._matrix = self._L
        self._sign = MatrixSign.ZeroSign

    def _unpack_impl(self) -> None:
        pass

    def _get_unpacked_impl(self) -> Any:
        return self._matrix

    def _solve_inplace_impl(self, x: np.ndarray):
        b = np.asarray(x, dtype=self._matrix.dtype)
        y = linalg.solve_triangular(self._L, self._P.T @ b, lower=True)
        x[:] = linalg.solve_triangular(self._U, y, lower=False)

    def _reconstruct_impl(self) -> np.ndarray:
        return self._P @ self._L @ self._U
