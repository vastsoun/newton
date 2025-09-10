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
from enum import IntEnum
from typing import Any

import numpy as np

from newton._src.solvers.kamino.utils.linalg.matrix import MatrixSign

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
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        **kwargs: dict[str, Any],
    ):
        # Declare internal data structures
        self._residuals: np.ndarray | None = None
        self._error_l2: np.ndarray | None = None
        self._error_inf: np.ndarray | None = None

        # Initialize internal solver meta-data
        self._dtype: np.dtype | None = dtype

        # Initialize internal solution meta-data
        self._info: ComputationInfo = ComputationInfo.Uninitialized
        self._compute_success: bool = False
        if kwargs:
            raise TypeError(f"Unused kwargs: {list(kwargs)}")

    def _compute_errors(self) -> float:
        """TODO"""
        # A_rec = self.reconstructed()
        return 0.0

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def compute_success(self) -> bool:
        return self._compute_success

    ###
    # Internals
    ###

    @abstractmethod
    def _compute_impl(self, A: np.ndarray):
        raise NotImplementedError("Missing compute implementation.")

    @abstractmethod
    def _solve_inplace_impl(self, b: np.ndarray, compute_errors: bool, **kwargs):
        raise NotImplementedError("In-place solving implementation is not provided.")

    ###
    # Public API
    ###

    def solve_inplace(self, b: np.ndarray, compute_errors: bool = False, **kwargs):
        """Solves the linear system `A@x = b` in-place"""
        # TODO: Check that A, b are compatible types
        # TODO: Check that A, b are compatible shapes
        return self._solve_inplace_impl(b, compute_errors=compute_errors, **kwargs)

    def compute(self, A: np.ndarray):
        """Ingest matrix and precompute rhs-independent intermediate."""
        self._compute_impl(A)

    def solve(self, b: np.ndarray, compute_errors: bool = False, **kwargs) -> np.ndarray:
        """Solves the linear system `A@x = b`"""
        return self.solve_inplace(b.copy(), compute_errors=compute_errors, **kwargs)
