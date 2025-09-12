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

"""KAMINO: Utilities: Linear Algebra: Linear system solver classes"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import scipy.linalg

from ...core.types import override
from .ldlt_bk import (
    compute_ldlt_bk_lower,
    compute_ldlt_bk_lower_reconstruct,
    compute_ldlt_bk_lower_solve,
    unpack_ldlt_bk_lower,
)
from .ldlt_blocked import compute_ldlt_blocked_lower, compute_ldlt_blocked_upper
from .ldlt_eigen3 import (
    compute_ldlt_eigen3_lower,
    compute_ldlt_eigen3_solve_inplace,
    unpack_ldlt_eigen3,
)
from .ldlt_nopivot import (
    compute_ldlt_lower_solve_inplace,
    compute_ldlt_nopivot_lower,
    compute_ldlt_upper_solve_inplace,
)
from .llt_std import (
    compute_cholesky_lower,
    compute_cholesky_lower_solve,
)
from .lu_nopiv import compute_lu_backward_upper, compute_lu_forward_lower, lu_nopiv
from .matrix import (
    MatrixSign,
    _make_tolerance,
    assert_is_square_matrix,
    assert_is_symmetric_matrix,
)

###
# Types
###


class ComputationInfo(IntEnum):
    Success = 0
    Uninitialized = 1
    NumericalIssue = 2
    NoConvergence = 3
    InvalidInput = 4


@dataclass
class FixedPointSolution:
    x: np.ndarray | None = None
    error: float = np.inf
    iterations: int = 0
    converged: bool = False
    res_history: list[float] = field(default_factory=list)


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
        self._success: bool = False
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
    def success(self) -> bool:
        return self._success

    ###
    # Internals
    ###

    @abstractmethod
    def _compute_impl(self, A: np.ndarray, **kwargs):
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

    def compute(self, A: np.ndarray, **kwargs):
        """Ingest matrix and precompute rhs-independent intermediate."""
        self._compute_impl(A, **kwargs)

    def solve(self, b: np.ndarray, compute_errors: bool = False, **kwargs) -> np.ndarray:
        """Solves the linear system `A@x = b`"""
        return self.solve_inplace(b.copy(), compute_errors=compute_errors, **kwargs)


class MatrixFactorizer(LinearSolver):
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
        super().__init__(dtype, compute_error)
        # Declare internal data structures
        self._source: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._errors: np.ndarray | None = None

        # Initialize internal meta-data
        self._tolerance: float | None = tol
        self._itype: np.dtype | None = itype
        self._sign: MatrixSign = MatrixSign.ZeroSign
        self._info: ComputationInfo = ComputationInfo.Success
        self._upper: bool = upper

        # Initialize internal flags
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

    ###
    # Internals
    ###

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self.factorize(A, **kwargs)

    @abstractmethod
    def _factorize_impl(self, A: np.ndarray) -> None:
        raise NotImplementedError("Factorization implementation is not provided.")

    @abstractmethod
    def _unpack_impl(self) -> None:
        raise NotImplementedError("Unpacking implementation is not provided.")

    @abstractmethod
    def _get_unpacked_impl(self) -> Any:
        raise NotImplementedError("Getting unpacked factors implementation is not provided.")

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

    @override
    def solve_inplace(self, x: np.ndarray, tol: float | None = None):
        """Solves the linear system `A@x = b` using the LDLT factorization in-place."""
        self._check_has_factorization()
        if tol is not None:
            self._tolerance = _make_tolerance(tol, dtype=self._dtype)
        self._solve_inplace_impl(x)

    @override
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


###
# Utilities
###


def _check_system_compatibility(A: np.ndarray, b: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix A must be square (n x n) but has shape {A.shape}.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError(f"Vector b ({b.shape}) must have compatible dimensions with A ({A.shape}).")
    if b.dtype != A.dtype:
        raise ValueError(f"Vector b ({b.dtype}) must have the same data type as matrix A ({A.dtype}).")


def _check_initial_guess(A: np.ndarray, x_0: np.ndarray | None) -> np.ndarray:
    if x_0 is None:
        return np.zeros(A.shape[1], dtype=A.dtype)
    if x_0.ndim != 1 or x_0.shape[0] != A.shape[1]:
        raise ValueError(f"Initial guess x_0 ({x_0.shape}) must have compatible dimensions with A ({A.shape}).")
    if x_0.dtype != A.dtype:
        raise ValueError(f"Initial guess x_0 ({x_0.dtype}) must have the same data type as matrix A ({A.dtype}).")
    return x_0


###
# Default numpy / scipy solvers
###


class NumpySolve(LinearSolver):
    """Direct solver using numpy.linalg.solve. Uses LAPACK routine _gesv internally."""

    def __init__(
        self,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype=dtype, compute_errors=compute_errors, **kwargs)
        self._A: np.ndarray | None = None

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(self, b: np.ndarray, compute_errors: bool, **kwargs) -> FixedPointSolution:
        x = np.linalg.solve(self._A, b)

        solution = FixedPointSolution()
        solution.x = x
        solution.iterations = 1
        solution.converged = True

        b[:] = x
        return solution


class ScipySolve(LinearSolver):
    """Direct solver using scipy.linalg.solve. From scipy docs:
    The general, symmetric, Hermitian and positive definite solutions are obtained via
    calling GESV, SYSV, HESV, and POSV routines of LAPACK respectively.
    """

    def __init__(
        self,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype=dtype, compute_errors=compute_errors, **kwargs)
        self._A: np.ndarray | None = None

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(self, b: np.ndarray, compute_errors: bool, **kwargs) -> FixedPointSolution:
        x = scipy.linalg.solve(self._A, b, assume_a="gen", check_finite=False, overwrite_b=False)

        solution = FixedPointSolution()
        solution.x = x
        solution.iterations = 1
        solution.converged = True

        b[:] = x
        return solution


###
# Functions
###


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    tolerance = A.dtype.type(tolerance)

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_p[k]
            x_n[j] = (b[j] - sum) / A[j, j]

        solution.error = np.max(np.abs(x_n - x_p))
        if compute_residuals:
            solution.res_history.append(A @ x_n - b)
        if solution.error < tolerance:
            solution.converged = True
            break

        x_p[:] = x_n

    solution.x = x_n
    return solution


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    tolerance = A.dtype.type(tolerance)

    n = A.shape[0]
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_n[k]
            x_n[j] = (b[j] - sum) / A[j, j]

        solution.error = np.max(np.abs(x_n - x_0))
        if compute_residuals:
            solution.res_history.append(A @ x_n - b)
        if solution.error < tolerance:
            solution.converged = True
            break

        x_0[:] = x_n

    solution.x = x_n
    return solution


def successive_over_relaxation(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    omega: float = 1.0,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    if not (0.0 < omega < 2.0):
        raise ValueError(f"Relaxation factor omega must be in the range (0, 2) but is {omega}.")
    omega = A.dtype.type(omega)
    tolerance = A.dtype.type(tolerance)

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_n[k]
            x_n[j] = x_p[j] + omega * ((b[j] - sum) / A[j, j] - x_p[j])

        solution.error = np.max(np.abs(x_n - x_p))
        if compute_residuals:
            solution.res_history.append(A @ x_n - b)
        if solution.error < tolerance:
            solution.converged = True
            break

        x_p[:] = x_n

    solution.x = x_n
    return solution


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    epsilon: float = 1e-12,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    epsilon = A.dtype.type(epsilon)
    tolerance = A.dtype.type(tolerance)

    r = b - A @ x_0
    g = r.copy()
    Ag = np.zeros_like(g)
    x = x_0.copy()
    rsold = np.dot(r, r)
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        Ag[:] = A @ g
        gAg = np.dot(g, Ag)
        alpha = rsold / max(gAg, epsilon)  # cap denom to avoid div-by-zero
        x += alpha * g
        r -= alpha * Ag
        rsnew = np.dot(r, r)

        solution.error = np.max(np.abs(r))
        if compute_residuals:
            solution.res_history.append(A @ x - b)
        if solution.error < tolerance:
            solution.converged = True
            break

        beta = rsnew / max(rsold, epsilon)  # cap denom to avoid div-by-zero
        g = r + beta * g
        rsold = rsnew

    solution.x = x
    return solution


def minimum_residual(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None,
    epsilon: float = 1e-12,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
    compute_residuals: bool = False,
) -> FixedPointSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    epsilon = A.dtype.type(epsilon)
    tolerance = A.dtype.type(tolerance)

    n = A.shape[0]
    r = b - A @ x_0
    p0 = r.copy()
    s0 = A @ p0
    p1 = p0.copy()
    s1 = s0.copy()
    x = x_0.copy()
    p2 = np.zeros(n)
    s2 = np.zeros(n)
    solution = FixedPointSolution()

    for i in range(max_iterations):
        solution.iterations = i + 1

        p2[:] = p1
        p1[:] = p0
        s2[:] = s1
        s1[:] = s0
        s1_prod = np.dot(s1, s1)
        s2_prod = np.dot(s2, s2)

        alpha = np.dot(r, s1) / max(s1_prod, epsilon)  # cap denom to avoid div-by-zero
        x += alpha * p1
        r -= alpha * s1

        solution.error = np.max(np.abs(r))
        if solution.error < tolerance:
            solution.converged = True
            break

        p0[:] = s1
        s0[:] = A @ s1

        beta = np.dot(s0, s1) / max(s1_prod, epsilon)  # cap denom to avoid div-by-zero
        p0 -= beta * p1
        s0 -= beta * s1

        if i > 0:
            gamma = np.dot(s0, s2) / max(s2_prod, epsilon)  # cap denom to avoid div-by-zero
            p0 -= gamma * p2
            s0 -= gamma * s2

    solution.x = x
    return solution


###
# Classes
###


class JacobiSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, compute_errors, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        result = jacobi(
            self._A,
            b,
            x_0,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class GaussSeidelSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        result = gauss_seidel(
            self._A,
            b,
            x_0,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class SORSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        omega: float = 1.0,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations
        self._omega: float = omega

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        result = successive_over_relaxation(
            self._A,
            b,
            x_0,
            omega=self._omega,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class ConjugateGradientSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        epsilon: float = 1e-12,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations
        self._epsilon: float = epsilon

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        epsilon = self._epsilon
        result = conjugate_gradient(
            self._A,
            b,
            x_0,
            epsilon=epsilon,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class MinimumResidualSolver(LinearSolver):
    def __init__(
        self,
        tol: float = 1e-12,
        dtype: np.dtype | None = None,
        compute_errors: bool = False,
        max_iterations: int = 1000,
        epsilon: float = 1e-12,
        **kwargs: dict[str, Any],
    ):
        super().__init__(dtype, **kwargs)
        self._tolerance: float = tol
        self._max_iterations: int = max_iterations
        self._epsilon: float = epsilon

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self._A = A
        self._compute_success = True

    @override
    def _solve_inplace_impl(
        self, b: np.ndarray, compute_errors: bool, x_0: np.ndarray | None = None, **kwargs
    ) -> FixedPointSolution:
        tol = self._tolerance
        max_iter = self._max_iterations
        epsilon = self._epsilon
        result = minimum_residual(
            self._A,
            b,
            x_0,
            epsilon=epsilon,
            tolerance=tol,
            max_iterations=max_iter,
            compute_residuals=compute_errors,
        )
        b[:] = result.x
        return result


class LDLTEigen3(MatrixFactorizer):
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
        self._transpositions: np.ndarray | None = None
        self._scratch: np.ndarray | None = None

        # Declare optional unpacked factors
        self.L: np.ndarray | None = None
        self.D: np.ndarray | None = None
        self.P: np.ndarray | None = None

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

    @property
    def transpositions(self) -> np.ndarray | None:
        return self._transpositions

    def _factorize_impl(self, A: np.ndarray) -> None:
        self._matrix, self._transpositions, self._scratch, self._sign, self._success = compute_ldlt_eigen3_lower(
            A, self._itype
        )

    def _unpack_impl(self) -> None:
        self.L, self.D, self.P = unpack_ldlt_eigen3(self._matrix, self._transpositions)

    def _get_unpacked_impl(self) -> Any:
        return self.L, self.D, self.P

    def _solve_inplace_impl(self, x: np.ndarray):
        compute_ldlt_eigen3_solve_inplace(self._matrix, self._transpositions, x, self._tolerance)

    def _reconstruct_impl(self) -> np.ndarray:
        L, D, P = self.unpacked()
        return P @ (L @ D @ L.T) @ P.T


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


class LLT(MatrixFactorizer):
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
            self._matrix = compute_cholesky_lower(A, False)
            self._sign = MatrixSign.PositiveDef
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {e!s}") from e

        # Update internal meta-data
        self._sign = MatrixSign.ZeroSign

    def _unpack_impl(self) -> None:
        pass

    def _get_unpacked_impl(self) -> Any:
        return self._matrix

    def _solve_inplace_impl(self, x: np.ndarray):
        b = np.asarray(x, dtype=self._matrix.dtype)
        y = compute_cholesky_lower_solve(self._matrix, b)
        x[:] = y

    def _reconstruct_impl(self) -> np.ndarray:
        return self._matrix @ self._matrix.T


class LLTNumPy(MatrixFactorizer):
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
            self._matrix = np.linalg.cholesky(A, upper=self._upper)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {e!s}") from e

        # Update internal meta-data
        self._sign = MatrixSign.ZeroSign

    def _unpack_impl(self) -> None:
        pass

    def _get_unpacked_impl(self) -> Any:
        return self._matrix

    def _solve_inplace_impl(self, x: np.ndarray):
        b = np.asarray(x, dtype=self._matrix.dtype)
        y = scipy.linalg.solve_triangular(self._matrix, b, lower=True)
        x[:] = scipy.linalg.solve_triangular(self._matrix.T, y, lower=False)

    def _reconstruct_impl(self) -> np.ndarray:
        return self._matrix @ self._matrix.T


class LLTSciPy(MatrixFactorizer):
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
            self._matrix = linalg.cholesky(A, lower=not self._upper)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {e!s}") from e

        # Update internal meta-data
        self._sign = MatrixSign.ZeroSign

    def _unpack_impl(self) -> None:
        pass

    def _get_unpacked_impl(self) -> Any:
        return self._matrix

    def _solve_inplace_impl(self, x: np.ndarray):
        b = np.asarray(x, dtype=self._matrix.dtype)
        y = linalg.solve_triangular(self._matrix, b, lower=True)
        x[:] = linalg.solve_triangular(self._matrix.T, y, lower=False)

    def _reconstruct_impl(self) -> np.ndarray:
        return self._matrix @ self._matrix.T


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
            self._P, self._L, self._U = scipy.linalg.lu(A, permute_l=False)
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
        y = scipy.linalg.solve_triangular(self._L, self._P.T @ b, lower=True)
        x[:] = scipy.linalg.solve_triangular(self._U, y, lower=False)

    def _reconstruct_impl(self) -> np.ndarray:
        return self._P @ self._L @ self._U


# ---------------------------
# Example usage / sanity checks
# ---------------------------
if __name__ == "__main__":
    # ----------------------------
    np.set_printoptions(linewidth=2000, precision=10, threshold=10000, suppress=False)  # Suppress scientific notation

    # ----------------------------
    # dtype = np.float64
    dtype = np.float32
    print("----------------------------")
    print(f"dtype: {dtype}")

    # ----------------------------
    epsilon = np.finfo(dtype).eps
    tolerance = np.finfo(dtype).eps
    max_iterations = 1000
    print("----------------------------")
    print(f"epsilon: {epsilon}")
    print(f"tolerance: {tolerance}")
    print(f"max_iterations: {max_iterations}")

    # ----------------------------
    A = np.array([[2.0, 1.0], [1.0, 4.0]], dtype=dtype)  # symmetric positive-definite matrix
    b = np.array([1.0, 2.0], dtype=dtype)  # right-hand side in range-space of A
    print("----------------------------")
    print(f"\nA {A.shape}[{A.dtype}]:\n{A}\n")
    print(f"\nb {b.shape}[{b.dtype}]:\n{b}\n")

    # ----------------------------
    lambdas_A = np.linalg.eigvals(A)
    rank_A = np.linalg.matrix_rank(A)
    cond_A = np.linalg.cond(A)
    print("----------------------------")
    print(f"lambda(A): {lambdas_A}")
    print(f"rank(A): {rank_A}")
    print(f"cond(A): {cond_A}")

    # ---------------------------- Reference
    x_np = np.linalg.solve(A, b)
    r_np = A @ x_np - b
    print("----------------------------")
    print(f"\nx_np {x_np.shape}[{x_np.dtype}]:\n{x_np}\n")

    # ---------------------------- Jacobi
    # jac = jacobi(A=A, b=b, x_0=None, tolerance=tolerance, max_iterations=max_iterations)
    jac_solver = JacobiSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations)
    jac_solver.compute(A)
    jac = jac_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_jac = A @ jac.x - b
    print("----------------------------")
    print(f"Jacobi:  converged: {jac.converged}")
    print(f"Jacobi: iterations: {jac.iterations}")
    print(f"Jacobi:      error: {jac.error}")
    print("----------------------------")
    print(f"\nx_jac {jac.x.shape}[{jac.x.dtype}]:\n{jac.x}\n")

    # ---------------------------- Gauss-Seidel
    # gs = gauss_seidel(A=A, b=b, x_0=None, tolerance=tolerance, max_iterations=max_iterations)
    gs_solver = GaussSeidelSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations)
    gs_solver.compute(A)
    gs = gs_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_gs = A @ gs.x - b
    print("----------------------------")
    print(f"Gauss-Seidel:  converged: {gs.converged}")
    print(f"Gauss-Seidel: iterations: {gs.iterations}")
    print(f"Gauss-Seidel:      error: {gs.error}")
    print("----------------------------")
    print(f"\nx_gs {gs.x.shape}[{gs.x.dtype}]:\n{gs.x}\n")

    # ---------------------------- Successive Over-Relaxation
    # sor = successive_over_relaxation(A=A, b=b, x_0=None, omega=1.25, tolerance=tolerance, max_iterations=max_iterations)
    sor_solver = SORSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations, omega=1.25)
    sor_solver.compute(A)
    sor = sor_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_sor = A @ sor.x - b
    print("----------------------------")
    print(f"Successive Over-Relaxation:  converged: {sor.converged}")
    print(f"Successive Over-Relaxation: iterations: {sor.iterations}")
    print(f"Successive Over-Relaxation:      error: {sor.error}")
    print("----------------------------")
    print(f"\nx_sor {sor.x.shape}[{sor.x.dtype}]:\n{sor.x}\n")

    # ---------------------------- Conjugate Gradient
    # cg = conjugate_gradient(A=A, b=b, x_0=None, epsilon=epsilon, tolerance=tolerance, max_iterations=max_iterations)
    cg_solver = ConjugateGradientSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations, epsilon=epsilon)
    cg_solver.compute(A)
    cg = cg_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_cg = A @ cg.x - b
    print("----------------------------")
    print(f"Conjugate Gradient:  converged: {cg.converged}")
    print(f"Conjugate Gradient: iterations: {cg.iterations}")
    print(f"Conjugate Gradient:      error: {cg.error}")
    print("----------------------------")
    print(f"\nx_cg {cg.x.shape}[{cg.x.dtype}]:\n{cg.x}\n")

    # ---------------------------- Minimum Residual
    # minres = minimum_residual(A=A, b=b, x_0=None, epsilon=epsilon, tolerance=tolerance, max_iterations=max_iterations)
    minres_solver = MinimumResidualSolver(tol=tolerance, dtype=dtype, max_iterations=max_iterations, epsilon=epsilon)
    minres_solver.compute(A)
    minres = minres_solver.solve_inplace(b.copy(), compute_errors=False, x_0=None)
    r_minres = A @ minres.x - b
    print("----------------------------")
    print(f"Minimum Residual:  converged: {minres.converged}")
    print(f"Minimum Residual: iterations: {minres.iterations}")
    print(f"Minimum Residual:      error: {minres.error}")
    print("----------------------------")
    print(f"\nx_minres {minres.x.shape}[{minres.x.dtype}]:\n{minres.x}\n")

    # ----------------------------
    r_np_l2 = np.linalg.norm(r_np, ord=2)
    r_jac_l2 = np.linalg.norm(r_jac, ord=2)
    r_gs_l2 = np.linalg.norm(r_gs, ord=2)
    r_sor_l2 = np.linalg.norm(r_sor, ord=2)
    r_cg_l2 = np.linalg.norm(r_cg, ord=2)
    r_minres_l2 = np.linalg.norm(r_minres, ord=2)

    # ----------------------------
    r_np_infnorm = np.linalg.norm(r_np, ord=np.inf)
    r_jac_infnorm = np.linalg.norm(r_jac, ord=np.inf)
    r_gs_infnorm = np.linalg.norm(r_gs, ord=np.inf)
    r_sor_infnorm = np.linalg.norm(r_sor, ord=np.inf)
    r_cg_infnorm = np.linalg.norm(r_cg, ord=np.inf)
    r_minres_infnorm = np.linalg.norm(r_minres, ord=np.inf)

    # ----------------------------
    print("----------------------------")
    print(f"r_np_l2: {r_np_l2}")
    print(f"r_jac_l2: {r_jac_l2}")
    print(f"r_gs_l2: {r_gs_l2}")
    print(f"r_sor_l2: {r_sor_l2}")
    print(f"r_cg_l2: {r_cg_l2}")
    print(f"r_minres_l2: {r_minres_l2}")
    print("----------------------------")
    print(f"r_np_infnorm: {r_np_infnorm}")
    print(f"r_jac_infnorm: {r_jac_infnorm}")
    print(f"r_gs_infnorm: {r_gs_infnorm}")
    print(f"r_sor_infnorm: {r_sor_infnorm}")
    print(f"r_cg_infnorm: {r_cg_infnorm}")
    print(f"r_minres_infnorm: {r_minres_infnorm}")
