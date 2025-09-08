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

"""KAMINO: Utilities: Linear Algebra: Linear system solver base class"""

from abc import ABC, abstractmethod

import numpy as np
from enum import IntEnum
from newton._src.solvers.kamino.utils.linalg.matrix import MatrixSign, _make_tolerance


###
# Types
###

class ComputationInfo(IntEnum):
    Success = 0
    Uninitialized = 1
    NumericalIssue = 2
    NoConvergence = 3
    InvalidInput = 4


class LinearSolver(ABC):
    def __init__(
        self,
        tol: float | None = None,
        dtype: np.dtype | None = None,
    ):
        # Declare internal data structures
        self._residuals: np.ndarray | None = None
        self._error_l2: np.ndarray | None = None
        self._error_inf: np.ndarray | None = None

        # Initialize internal solver meta-data
        self._tolerance: float | None = tol
        self._dtype: np.dtype | None = dtype

        # Initialize internal solution meta-data
        self._info: ComputationInfo = ComputationInfo.Uninitialized
        self._success: bool = False

    def _compute_errors(self) -> float:
        """TODO"""
        # A_rec = self.reconstructed()
        return 0.0

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

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
    def _solve_inplace_impl(self, A: np.ndarray, x: np.ndarray):
        raise NotImplementedError("In-place solving implementation is not provided.")

    ###
    # Public API
    ###

    def solve_inplace(self, A: np.ndarray, x: np.ndarray, tol: float | None = None, compute_errors: bool = False):
        """Solves the linear system `A@x = b` in-place"""
        # TODO: Check that A, x are compatible types
        # TODO: Check that A, x are compatible shapes
        if tol is not None:
            self._tolerance = _make_tolerance(tol, dtype=self._dtype)
        self._solve_inplace_impl(A, x)

    def solve(self, A: np.ndarray, b: np.ndarray, tol: float | None = None, compute_errors: bool = False) -> np.ndarray:
        """Solves the linear system `A@x = b`"""
        x = b.astype(A.dtype, copy=True)
        self.solve_inplace(A, x, tol)
        return x
