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

"""KAMINO: Utilities: Linear Algebra: Eigenvalue Estimation"""

import math

import numpy as np


class PowerIteration:
    """TODO"""

    def __init__(
        self,
        atol: float = 0.0,
        rtol: float = 1.0e-8,
        max_iterations: int = 1000,
        dtype: np.dtype = np.float64,
    ) -> None:
        # Initialize solver configurations
        self._dtype = dtype
        self._atol = dtype(atol)
        self._rtol = dtype(rtol)
        self._max_iterations = int(max_iterations)

        # Initialize solver data
        self._max_eigval: float = np.inf
        self._min_eigval: float = -np.inf
        self._max_eigval_residual: float = 0.0
        self._min_eigval_residual: float = 0.0
        self._iterations: int = 0
        self._converged: bool = False
        self._has_largest: bool = False

        # Declare internal solver data
        self._max_eigvec: np.ndarray | None = None
        self._min_eigvec: np.ndarray | None = None
        self._tmp_eigvec: np.ndarray | None = None

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def atol(self) -> float:
        return self._atol

    @atol.setter
    def atol(self, tol: float) -> None:
        self._atol = self._dtype(tol)

    @property
    def rtol(self) -> float:
        return self._rtol

    @rtol.setter
    def rtol(self, tol: float) -> None:
        self._rtol = self._dtype(tol)

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iter: int) -> None:
        self._max_iterations = int(max_iter)

    @property
    def max_eigenvalue(self) -> float:
        return float(self._max_eigval)

    @property
    def min_eigenvalue(self) -> float:
        return float(self._min_eigval)

    @property
    def max_residual(self) -> float:
        return float(self._max_eigval_residual)

    @property
    def min_residual(self) -> float:
        return float(self._min_eigval_residual)

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def converged(self) -> bool:
        return self._converged

    ###
    # Operations
    ###

    def largest(self, A: np.ndarray) -> float:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        self._reset(A)
        self._compute_largest(A)
        self._has_largest = True
        return self._max_eigval

    def smallest(self, A: np.ndarray) -> float:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        if not self._has_largest:
            self._reset(A)
            self._compute_largest(A)
            self._has_largest = True
        self._compute_smallest(A)
        return self._min_eigval

    ###
    # Internals
    ###

    def _reset(self, A: np.ndarray) -> None:
        self._dtype = A.dtype
        self._atol = self._dtype.type(self._atol)
        self._rtol = self._dtype.type(self._rtol)
        self._max_eigval = np.inf
        self._min_eigval = -np.inf
        self._max_eigval_residual = np.inf
        self._min_eigval_residual = np.inf
        self._iterations = 0
        self._converged = False
        self._has_largest = False

    def _compute_largest(self, A: np.ndarray) -> None:
        n = A.shape[0]
        lambda_max_0 = self._dtype.type(1.0 / math.sqrt(n))
        self._max_eigvec = self._dtype.type(1) * lambda_max_0 * np.ones(n, dtype=self._dtype)
        lambda_max = np.linalg.norm(self._max_eigvec)
        self._converged = False
        for it in range(self._max_iterations):
            lambda_max_p = lambda_max
            self._max_eigvec /= lambda_max
            self._tmp_eigvec = self._max_eigvec.copy()
            self._max_eigvec = A @ self._tmp_eigvec
            lambda_max = np.linalg.norm(self._max_eigvec)
            self._max_eigval_residual = np.abs(lambda_max_p - lambda_max)
            tol = self._atol + self._rtol * max(np.abs(lambda_max_p), np.abs(lambda_max))
            self._iterations = it + 1
            if self._max_eigval_residual <= tol:
                self._converged = True
                break
        self._max_eigval = lambda_max

    def _compute_smallest(self, A: np.ndarray) -> None:
        n = A.shape[0]
        lambda_min_0 = self._dtype.type(1.0 / math.sqrt(n))
        self._min_eigvec = self._dtype.type(1) * lambda_min_0 * np.ones(n, dtype=self._dtype)
        lambda_min = np.linalg.norm(self._min_eigvec)
        self._converged = False
        for it in range(self._max_iterations):
            lambda_min_p = lambda_min
            self._min_eigvec /= lambda_min
            self._tmp_eigvec = self._min_eigvec.copy()
            self._min_eigvec = A @ self._tmp_eigvec
            self._min_eigvec = self._min_eigvec - self._max_eigval * self._tmp_eigvec
            lambda_min = np.linalg.norm(self._min_eigvec)
            self._min_eigval_residual = np.abs(lambda_min_p - lambda_min)
            tol = self._atol + self._rtol * max(np.abs(lambda_min_p), np.abs(lambda_min))
            self._iterations = it + 1
            if self._min_eigval_residual <= tol:
                self._converged = True
                break
        self._min_eigval = self._max_eigval - lambda_min


class GramIteration:
    """Gram iteration for the spectral radius (largest eigenvalue magnitude).

    The algorithm repeatedly forms M <- M^T M with normalization and keeps
    a running log of norms to estimate the dominant eigenvalue. Mirrors the
    original C++ logic using Frobenius norms and relative difference tests.
    """

    def __init__(
        self,
        atol: float = 0.0,
        rtol: float = 1.0e-8,
        max_iterations: int = 1000,
        dtype: np.dtype = np.float64,
    ) -> None:
        # Initialize solver configurations
        self._dtype = dtype
        self._atol = dtype(atol)
        self._rtol = dtype(rtol)
        self._max_iterations = int(max_iterations)

        # Initialize solver data
        self._max_eigval: float = np.inf
        self._max_eigval_residual: float = 0.0
        self._iterations: int = 0
        self._converged: bool = False

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def atol(self) -> float:
        return self._atol

    @atol.setter
    def atol(self, tol: float) -> None:
        self._atol = self._dtype(tol)

    @property
    def rtol(self) -> float:
        return self._rtol

    @rtol.setter
    def rtol(self, tol: float) -> None:
        self._rtol = self._dtype(tol)

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iter: int) -> None:
        self._max_iterations = int(max_iter)

    @property
    def max_eigenvalue(self) -> float:
        return float(self._max_eigval)

    @property
    def max_residual(self) -> float:
        return float(self._max_eigval_residual)

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def converged(self) -> bool:
        return self._converged

    ###
    # Operations
    ###

    def largest(self, A: np.ndarray) -> float:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        self._reset(A)
        self._compute_largest(A)
        return self._max_eigval

    ###
    # Internals
    ###

    def _reset(self, A: np.ndarray) -> None:
        self._dtype = A.dtype
        self._atol = self._dtype.type(self._atol)
        self._rtol = self._dtype.type(self._rtol)
        self._max_eigval = np.inf
        self._max_eigval_residual = np.inf
        self._iterations = 0
        self._converged = False

    # Operations
    def _compute_largest(self, A: np.ndarray) -> float:
        M = A.copy()
        inverse_power = self._dtype.type(1)
        log_curr_norm = self._dtype.type(0)
        product = self._dtype.type(0)
        product_p = self._dtype.type(0)
        self._converged = False
        for it in range(self._max_iterations):
            product_p = product
            M_norm = np.linalg.norm(M)
            if M_norm == 0.0:
                self._max_eigval = self._dtype.type(0)
                self._max_eigval_residual = self._dtype.type(0)
                self._iterations = it
                self._converged = True
                return self._max_eigval
            M = M / M_norm
            M = M.T @ M
            log_curr_norm = self._dtype.type(2) * (log_curr_norm + np.log(M_norm))
            inverse_power /= self._dtype.type(2)
            product = log_curr_norm * inverse_power
            self._max_eigval_residual = abs(product_p - product)
            error_scale = self._atol + self._rtol * max(np.abs(product_p), np.abs(product))
            self._iterations = it + 1
            if self._max_eigval_residual <= error_scale:
                self._converged = True
                break

        # Final estimate uses the last M norm plus accumulated log scaling
        M_norm_final = float(np.linalg.norm(M))
        self._max_eigval = (M_norm_final**inverse_power) * np.exp(log_curr_norm * inverse_power)
        return float(self._max_eigval)
