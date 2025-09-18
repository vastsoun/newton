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
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import scipy.linalg

from ...core.types import override
from . import factorize
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
class LinearSolution:
    converged: bool = False
    iterations: int = 0
    x: np.ndarray | None = None
    e: np.ndarray | None = None
    r: np.ndarray | None = None


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


def _make_tolerances(atol: float, rtol: float, dtype: np.dtype) -> tuple[float, float]:
    eps = np.finfo(dtype).eps
    atol = dtype.type(max(atol, eps))
    rtol = dtype.type(max(rtol, eps))
    return atol, rtol


###
# Error Metrics
###


def norm_l2(x: np.ndarray) -> float:
    return np.linalg.norm(x)


def norm_inf(x: np.ndarray) -> float:
    return np.max(np.abs(x))


def linsys_error_l2(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return np.linalg.norm(A @ x - b)


def linsys_error_inf(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return np.max(np.abs(A @ x - b))


def iterate_error_l2(x_n: np.ndarray, x_p: np.ndarray) -> float:
    return np.linalg.norm(x_n - x_p)


def iterate_error_inf(x_n: np.ndarray, x_p: np.ndarray) -> float:
    return np.max(np.abs(x_n - x_p))


###
# Convergence Criteria
###


def has_target_absolute_update_l2(
    A: np.ndarray, b: np.ndarray, x_n: np.ndarray, x_p: np.ndarray, atol: float, rtol: float
) -> bool:
    return iterate_error_l2(x_n, x_p) < atol


def has_target_absolute_update_inf(
    A: np.ndarray, b: np.ndarray, x_n: np.ndarray, x_p: np.ndarray, atol: float, rtol: float
) -> bool:
    return iterate_error_inf(x_n, x_p) < atol


def has_target_relative_update_l2(
    A: np.ndarray, b: np.ndarray, x_n: np.ndarray, x_p: np.ndarray, atol: float, rtol: float
) -> bool:
    eps = np.finfo(A.dtype).eps
    norm_xn = norm_l2(x_n)
    norm_xp = norm_l2(x_p)
    denom = max(norm_xn, norm_xp, eps)
    return iterate_error_l2(x_n, x_p) < rtol * denom


def has_target_relative_update_inf(
    A: np.ndarray, b: np.ndarray, x_n: np.ndarray, x_p: np.ndarray, atol: float, rtol: float
) -> bool:
    eps = np.finfo(A.dtype).eps
    norm_xn = norm_inf(x_n)
    norm_xp = norm_inf(x_p)
    denom = max(norm_xn, norm_xp, eps)
    return iterate_error_inf(x_n, x_p) < rtol * denom


def has_target_absrel_update_l2(
    A: np.ndarray, b: np.ndarray, x_n: np.ndarray, x_p: np.ndarray, atol: float, rtol: float
) -> bool:
    eps = np.finfo(A.dtype).eps
    norm_xn = norm_l2(x_n)
    norm_xp = norm_l2(x_p)
    denom = max(norm_xn, norm_xp, eps)
    return iterate_error_l2(x_n, x_p) < max(atol, rtol * denom)


def has_target_absrel_update_inf(
    A: np.ndarray, b: np.ndarray, x_n: np.ndarray, x_p: np.ndarray, atol: float, rtol: float
) -> bool:
    eps = np.finfo(A.dtype).eps
    norm_xn = norm_inf(x_n)
    norm_xp = norm_inf(x_p)
    denom = max(norm_xn, norm_xp, eps)
    return iterate_error_inf(x_n, x_p) < max(atol, rtol * denom)


###
# Iterative Solvers
###


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    max_iterations: int = 1000,
    record_errors: bool = False,
    record_residuals: bool = False,
    convergence_op: callable = has_target_absolute_update_inf,
) -> LinearSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    atol, rtol = _make_tolerances(atol, rtol, A.dtype)

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()

    solution = LinearSolution()
    if record_errors:
        solution.e = np.empty(max_iterations, dtype=A.dtype)
    if record_residuals:
        solution.r = np.empty(max_iterations, dtype=A.dtype)

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_p[k]
            x_n[j] = (b[j] - sum) / A[j, j]

        if record_errors:
            solution.e[i] = norm_inf(x_n - x_p)
        if record_residuals:
            solution.r[i] = norm_inf(A @ x_n - b)

        if convergence_op(A, b, x_n, x_p, atol, rtol):
            solution.converged = True
            break

        x_p[:] = x_n

    if record_errors:
        solution.e = solution.e[: solution.iterations]
    if record_residuals:
        solution.r = solution.r[: solution.iterations]

    solution.x = x_n
    return solution


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    max_iterations: int = 1000,
    record_errors: bool = False,
    record_residuals: bool = False,
    convergence_op: callable = has_target_absolute_update_inf,
) -> LinearSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    atol, rtol = _make_tolerances(atol, rtol, A.dtype)

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()

    solution = LinearSolution()
    if record_errors:
        solution.e = np.empty(max_iterations, dtype=A.dtype)
    if record_residuals:
        solution.r = np.empty(max_iterations, dtype=A.dtype)

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_n[k]
            x_n[j] = (b[j] - sum) / A[j, j]

        if record_errors:
            solution.e[i] = norm_inf(x_n - x_p)
        if record_residuals:
            solution.r[i] = norm_inf(A @ x_n - b)

        if convergence_op(A, b, x_n, x_p, atol, rtol):
            solution.converged = True
            break

        x_p[:] = x_n

    if record_errors:
        solution.e = solution.e[: solution.iterations]
    if record_residuals:
        solution.r = solution.r[: solution.iterations]

    solution.x = x_n
    return solution


def successive_over_relaxation(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None = None,
    omega: float = 1.0,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    max_iterations: int = 1000,
    record_errors: bool = False,
    record_residuals: bool = False,
    convergence_op: callable = has_target_absolute_update_inf,
) -> LinearSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    if not (0.0 < omega < 2.0):
        raise ValueError(f"Relaxation factor omega must be in the range (0, 2) but is {omega}.")
    omega = A.dtype.type(omega)
    atol, rtol = _make_tolerances(atol, rtol, A.dtype)

    n = A.shape[0]
    x_p = x_0.copy()
    x_n = x_0.copy()

    solution = LinearSolution()
    if record_errors:
        solution.e = np.empty(max_iterations, dtype=A.dtype)
    if record_residuals:
        solution.r = np.empty(max_iterations, dtype=A.dtype)

    for i in range(max_iterations):
        solution.iterations = i + 1

        for j in range(n):
            sum = A.dtype.type(0)
            for k in range(n):
                if k != j:
                    sum += A[j, k] * x_n[k]
            x_n[j] = x_p[j] + omega * ((b[j] - sum) / A[j, j] - x_p[j])

        if record_errors:
            solution.e[i] = norm_inf(x_n - x_p)
        if record_residuals:
            solution.r[i] = norm_inf(A @ x_n - b)

        if convergence_op(A, b, x_n, x_p, atol, rtol):
            solution.converged = True
            break

        x_p[:] = x_n

    if record_errors:
        solution.e = solution.e[: solution.iterations]
    if record_residuals:
        solution.r = solution.r[: solution.iterations]

    solution.x = x_n
    return solution


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None = None,
    epsilon: float = 1e-12,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    max_iterations: int = 1000,
    record_errors: bool = False,
    record_residuals: bool = False,
    convergence_op: callable = has_target_absolute_update_inf,
) -> LinearSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    epsilon = A.dtype.type(epsilon)
    atol, rtol = _make_tolerances(atol, rtol, A.dtype)

    r = b - A @ x_0
    g = r.copy()
    Ag = np.zeros_like(g)
    x_p = x_0.copy()
    x_n = x_0.copy()
    rsold = np.dot(r, r)

    solution = LinearSolution()
    if record_errors:
        solution.e = np.empty(max_iterations, dtype=A.dtype)
    if record_residuals:
        solution.r = np.empty(max_iterations, dtype=A.dtype)

    for i in range(max_iterations):
        solution.iterations = i + 1

        Ag[:] = A @ g
        gAg = np.dot(g, Ag)
        alpha = rsold / max(gAg, epsilon)  # cap denom to avoid div-by-zero
        x_n += alpha * g
        r -= alpha * Ag
        rsnew = np.dot(r, r)

        if record_errors:
            solution.e[i] = norm_inf(x_n - x_p)
        if record_residuals:
            solution.r[i] = norm_inf(A @ x_n - b)

        if convergence_op(A, b, x_n, x_p, atol, rtol):
            solution.converged = True
            break

        x_p[:] = x_n
        beta = rsnew / max(rsold, epsilon)  # cap denom to avoid div-by-zero
        g = r + beta * g
        rsold = rsnew

    if record_errors:
        solution.e = solution.e[: solution.iterations]
    if record_residuals:
        solution.r = solution.r[: solution.iterations]

    solution.x = x_n
    return solution


def minimum_residual(
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray | None = None,
    epsilon: float = 1e-12,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    max_iterations: int = 1000,
    record_errors: bool = False,
    record_residuals: bool = False,
    convergence_op: callable = has_target_absolute_update_inf,
) -> LinearSolution:
    _check_system_compatibility(A, b)
    x_0 = _check_initial_guess(A, x_0)
    epsilon = A.dtype.type(epsilon)
    atol, rtol = _make_tolerances(atol, rtol, A.dtype)

    n = A.shape[0]
    r = b - A @ x_0
    p0 = r.copy()
    s0 = A @ p0
    p1 = p0.copy()
    s1 = s0.copy()
    x_p = x_0.copy()
    x_n = x_0.copy()
    p2 = np.zeros(n)
    s2 = np.zeros(n)

    solution = LinearSolution()
    if record_errors:
        solution.e = np.empty(max_iterations, dtype=A.dtype)
    if record_residuals:
        solution.r = np.empty(max_iterations, dtype=A.dtype)

    for i in range(max_iterations):
        solution.iterations = i + 1

        p2[:] = p1
        p1[:] = p0
        s2[:] = s1
        s1[:] = s0
        s1_prod = np.dot(s1, s1)
        s2_prod = np.dot(s2, s2)

        alpha = np.dot(r, s1) / max(s1_prod, epsilon)  # cap denom to avoid div-by-zero
        x_n += alpha * p1
        r -= alpha * s1

        if record_errors:
            solution.e[i] = norm_inf(x_n - x_p)
        if record_residuals:
            solution.r[i] = norm_inf(A @ x_n - b)

        if convergence_op(A, b, x_n, x_p, atol, rtol):
            solution.converged = True
            break

        x_p[:] = x_n
        p0[:] = s1
        s0[:] = A @ s1

        beta = np.dot(s0, s1) / max(s1_prod, epsilon)  # cap denom to avoid div-by-zero
        p0 -= beta * p1
        s0 -= beta * s1

        if i > 0:
            gamma = np.dot(s0, s2) / max(s2_prod, epsilon)  # cap denom to avoid div-by-zero
            p0 -= gamma * p2
            s0 -= gamma * s2

    if record_errors:
        solution.e = solution.e[: solution.iterations]
    if record_residuals:
        solution.r = solution.r[: solution.iterations]

    solution.x = x_n
    return solution


###
# Solver Interfaces
###


class LinearSolver(ABC):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
        **kwargs: dict[str, Any],
    ):
        # Override dtype if matrix A is provided
        if A is not None:
            dtype = A.dtype.type

        # Default tolerances to machine epsilon if not provided
        eps = np.finfo(dtype).eps
        atol = dtype(atol if atol is not None else eps)
        rtol = dtype(rtol if rtol is not None else eps)

        # Initialize internal solver meta-data
        self._dtype: np.dtype = dtype
        self._atol: float = atol
        self._rtol: float = rtol

        # Initialize internal solution meta-data
        self._info: ComputationInfo = ComputationInfo.Uninitialized  # TODO: currently not used
        self._error_abs: float | None = None
        self._error_rel: float | None = None

        # Declare internal solver data
        self._matrix: np.ndarray | None = None
        self._rhs: np.ndarray | None = None

        # If a matrix is provided, proceed with its pre-computation
        if A is not None:
            self.compute(A, **kwargs)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def info(self) -> ComputationInfo:
        return self._info

    @property
    def error_abs(self) -> float | None:
        return self._error_abs

    @property
    def error_rel(self) -> float | None:
        return self._error_rel

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    ###
    # Internals
    ###

    def _set_tolerance_dtype(self):
        eps = np.finfo(self._dtype).eps
        self._atol = self._dtype.type(max(self._atol, eps))
        self._rtol = self._dtype.type(max(self._rtol, eps))

    def _compute_solve_error(self, A: np.ndarray, b: np.ndarray, x: np.ndarray):
        """Computes the absolute and relative solution error."""
        eps = np.finfo(A.dtype).eps
        norm_x = np.linalg.norm(x, ord=np.inf)
        norm_b = np.linalg.norm(b, ord=np.inf)
        norm_A = np.linalg.norm(A, ord=np.inf)
        denom = max(norm_A * norm_x, norm_b, eps)
        self._error_abs = linsys_error_inf(A, b, x)
        self._error_rel = self._error_abs / denom

    ###
    # Implementation API
    ###

    @abstractmethod
    def _compute_impl(self, A: np.ndarray, **kwargs) -> None:
        raise NotImplementedError("Compute operation is not implemented.")

    @abstractmethod
    def _solve_inplace_impl(self, x: np.ndarray, **kwargs) -> None:
        raise NotImplementedError("Solve in-place operation is not implemented.")

    ###
    # Public API
    ###

    def compute(self, A: np.ndarray, **kwargs):
        """Ingest matrix and precompute rhs-independent intermediate."""
        self._matrix = A
        self._dtype = A.dtype
        self._set_tolerance_dtype()
        self._compute_impl(A, **kwargs)

    def solve_inplace(self, x: np.ndarray, compute_error: bool = False, **kwargs):
        """Solves the linear system `A@x = b` in-place, where `x` is initialized with the system rhs."""
        _check_system_compatibility(self._matrix, x)
        if compute_error:
            self._rhs = x.copy()
        self._solve_inplace_impl(x, **kwargs)
        if compute_error:
            self._compute_solve_error(self._matrix, self._rhs, x)
        else:
            self._error_abs = None
            self._error_rel = None

    def solve(self, b: np.ndarray, compute_error: bool = False, **kwargs) -> np.ndarray:
        """Solves the linear system `A@x = b`"""
        x = b.copy()
        self.solve_inplace(x, compute_error=compute_error, **kwargs)
        return x


class IndirectSolver(LinearSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
        record_errors: bool = False,
        record_residuals: bool = False,
        **kwargs: dict[str, Any],
    ):
        # Initialize solver configurations
        self._record_errors: bool = record_errors
        self._record_residuals: bool = record_residuals

        # Initialize solver internal data
        self._solution: LinearSolution = LinearSolution()

        # Initialize base class members
        super().__init__(A=A, atol=atol, rtol=rtol, dtype=dtype, **kwargs)

        # Check for unused kwargs
        if kwargs:
            raise TypeError(f"Unused kwargs: {list(kwargs)}")

    @property
    def solution(self) -> LinearSolution:
        return self._solution

    ###
    # Implementation API
    ###

    @abstractmethod
    def _solve_iterative_impl(self, b: np.ndarray, **kwargs) -> LinearSolution:
        raise NotImplementedError("Iterative solve implementation is not provided.")

    ###
    # Internals
    ###

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs) -> None:
        # TODO: We can potentially add matrix pre-conditioners here
        pass  # No-op for indirect solvers

    @override
    def _solve_inplace_impl(self, x: np.ndarray, **kwargs) -> None:
        self._solution = self._solve_iterative_impl(x, **kwargs)
        x[:] = self._solution.x


class DirectSolver(LinearSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
        **kwargs: dict[str, Any],
    ):
        # Default factorization tolerance to machine epsilon if not provided
        eps = np.finfo(dtype).eps
        ftol = dtype(ftol if ftol is not None else eps)

        # Initialize internal meta-data
        self._ftol: float | None = ftol
        self._sign: MatrixSign = MatrixSign.ZeroSign
        self._has_factors: bool = False
        self._has_unpacked: bool = False
        self._success: bool = False

        # Declare additional internal data
        self._factorization_error_abs: float | None = None
        self._factorization_error_rel: float | None = None

        # Initialize base class members
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
            **kwargs,
        )

        # Check for unused kwargs
        if kwargs:
            raise TypeError(f"Unused kwargs: {list(kwargs)}")

    @property
    def sign(self) -> MatrixSign:
        return self._sign

    @property
    def factorization_error_abs(self) -> float | None:
        return self._factorization_error_abs

    @property
    def factorization_error_rel(self) -> float | None:
        return self._factorization_error_rel

    ###
    # Internals
    ###

    def _check_has_factorization(self):
        """Checks if the factorization has been computed, otherwise raises error."""
        if not self._has_factors:
            raise ValueError("A factorization has not been computed!")

    def _compute_factorization_errors(self, A: np.ndarray):
        """Computes the matrix factorization error."""
        A_rec = self.reconstructed()
        norm_A = np.linalg.norm(A, ord=np.inf)
        self._factorization_error_abs = np.linalg.norm(A - A_rec, ord=np.inf)
        self._factorization_error_rel = self._factorization_error_abs / norm_A if norm_A > 0 else np.inf

    ###
    # Implementation API
    ###

    @abstractmethod
    def _factorize_impl(self, A: np.ndarray, **kwargs) -> None:
        raise NotImplementedError("Factorization implementation is not provided.")

    @abstractmethod
    def _unpack_impl(self) -> None:
        raise NotImplementedError("Unpacking implementation is not provided.")

    @abstractmethod
    def _get_unpacked_impl(self) -> Any:
        raise NotImplementedError("Getting unpacked factors implementation is not provided.")

    @abstractmethod
    def _reconstruct_impl(self) -> np.ndarray:
        raise NotImplementedError("Reconstruction implementation is not provided.")

    ###
    # Internals
    ###

    @override
    def _compute_impl(self, A: np.ndarray, **kwargs):
        self._factorize(A, **kwargs)

    def _factorize(
        self,
        A: np.ndarray,
        ftol: float | None = None,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
        **kwargs: dict[str, Any],
    ):
        # Perform basic checks on the input matrix
        assert_is_square_matrix(A)
        if check_symmetry:
            assert_is_symmetric_matrix(A)

        # Override the current tolerance if provided otherwise ensure
        # it does not exceed machine precision for the current dtype
        if ftol is not None:
            self._ftol = _make_tolerance(ftol, dtype=self._dtype)
        else:
            self._ftol = _make_tolerance(self._ftol, dtype=self._dtype)

        # Factorize the specified matrix (i.e. as np.ndarray)
        self._factorize_impl(A, **kwargs)

        # Update internal meta-data
        self._success = True
        self._has_factors = True
        self._has_unpacked = False

        # Optionally compute the matrix factorization error
        if compute_error or check_error:
            self._compute_factorization_errors(A)
            if check_error:
                if self._factorization_error_rel > self._ftol:
                    raise ValueError(
                        f"L∞ matrix factorization error {self._factorization_error_rel} exceeds tolerance {self._ftol}."
                    )
        else:
            self._factorization_error_abs = None
            self._factorization_error_rel = None

    ###
    # Public API
    ###

    def unpacked(self) -> Any:
        """Unpacks a potentially condensed factorization into a conventional format."""
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
# Reference Solvers
###


class NumPySolver(LinearSolver):
    """Direct solver using `numpy.linalg.solve`. Uses LAPACK routine _gesv internally."""

    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
    ):
        super().__init__(A=A, atol=atol, rtol=rtol, dtype=dtype)

    @override
    def _compute_impl(self, A: np.ndarray) -> None:
        pass  # No-op for NumPy solver

    @override
    def _solve_inplace_impl(self, b: np.ndarray) -> None:
        b[:] = np.linalg.solve(self._matrix, b)


class SciPySolver(LinearSolver):
    """Direct solver using scipy.linalg.solve. From scipy docs:
    The general, symmetric, Hermitian and positive definite solutions are obtained via
    calling GESV, SYSV, HESV, and POSV routines of LAPACK respectively.
    """

    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
    ):
        super().__init__(A=A, atol=atol, rtol=rtol, dtype=dtype)

    @override
    def _compute_impl(self, A: np.ndarray) -> None:
        pass  # No-op for NumPy solver

    @override
    def _solve_inplace_impl(self, b: np.ndarray) -> None:
        b[:] = scipy.linalg.solve(self._matrix, b, assume_a="symmetric", check_finite=False, overwrite_b=False)


class LLTNumPySolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # Declare internal solver data
        self._L: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._L = np.linalg.cholesky(A, upper=False)

    @override
    def _unpack_impl(self) -> None:
        pass

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._L @ self._L.T

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        scipy.linalg.cho_solve((self._L, True), x, overwrite_b=True)


class LLTSciPySolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # Declare internal solver data
        self._L: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._L = scipy.linalg.cholesky(A, lower=True)

    @override
    def _unpack_impl(self) -> None:
        pass

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._L @ self._L.T

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        scipy.linalg.cho_solve((self._L, True), x, overwrite_b=True)


class LDLTSciPySolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # LDLT packed factors
        self._PL: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._p: np.ndarray | None = None

        # Unpacked factors
        self._L: np.ndarray | None = None
        self._P: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._PL, self._D, self._p = scipy.linalg.ldl(A, lower=True)
        self._unpack_impl()
        self._has_unpacked = True

    @override
    def _unpack_impl(self) -> None:
        self._L = self._PL[self._p, :]
        self._P = np.eye(self._PL.shape[0])[:, self._p]

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L, self._D, self._P

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._PL @ self._D @ self._PL.T

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        b = x.copy()
        b = b[self._p]
        z = scipy.linalg.solve_triangular(self._L, b, lower=True)
        y = z / np.diag(self._D)
        xsol = scipy.linalg.solve_triangular(self._L.T, y, lower=False)
        x[:] = xsol[np.argsort(self._p)]


class LUSciPySolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # Declare internal data structures
        self._P: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._U: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._P, self._L, self._U = scipy.linalg.lu(A, permute_l=False)

    @override
    def _unpack_impl(self) -> None:
        pass

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._P, self._L, self._U

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._P @ self._L @ self._U

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        y = scipy.linalg.solve_triangular(self._L, self._P.T @ x, lower=True)
        x[:] = scipy.linalg.solve_triangular(self._U, y, lower=False)


###
# Indirect Solvers
###


class JacobiSolver(IndirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
        record_errors: bool = False,
        record_residuals: bool = False,
        max_iterations: int = 1000,
    ):
        # Declare internal solver data
        self._max_iterations: int = int(max_iterations)

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            record_errors=record_errors,
            record_residuals=record_residuals,
        )

    @override
    def _solve_iterative_impl(self, b: np.ndarray, x_0: np.ndarray | None = None) -> LinearSolution:
        return jacobi(
            A=self._matrix,
            b=b,
            x_0=x_0,
            atol=self._atol,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
            record_errors=self._record_errors,
            record_residuals=self._record_residuals,
        )


class GaussSeidelSolver(IndirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
        record_errors: bool = False,
        record_residuals: bool = False,
        max_iterations: int = 1000,
    ):
        # Declare internal solver data
        self._max_iterations: int = int(max_iterations)

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            record_errors=record_errors,
            record_residuals=record_residuals,
        )

    @override
    def _solve_iterative_impl(self, b: np.ndarray, x_0: np.ndarray | None = None) -> LinearSolution:
        return gauss_seidel(
            A=self._matrix,
            b=b,
            x_0=x_0,
            atol=self._atol,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
            record_errors=self._record_errors,
            record_residuals=self._record_residuals,
        )


class SORSolver(IndirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
        record_errors: bool = False,
        record_residuals: bool = False,
        max_iterations: int = 1000,
        omega: float = 1.0,
    ):
        # Declare internal solver data
        self._omega: float = float(omega)
        self._max_iterations: int = int(max_iterations)

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            record_errors=record_errors,
            record_residuals=record_residuals,
        )

    @override
    def _solve_iterative_impl(self, b: np.ndarray, x_0: np.ndarray | None = None) -> LinearSolution:
        return successive_over_relaxation(
            A=self._matrix,
            b=b,
            x_0=x_0,
            omega=self._omega,
            atol=self._atol,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
            record_errors=self._record_errors,
            record_residuals=self._record_residuals,
        )


class ConjugateGradientSolver(IndirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
        record_errors: bool = False,
        record_residuals: bool = False,
        max_iterations: int = 1000,
        epsilon: float | None = None,
    ):
        # Default epsilon to machine epsilon if not provided
        if epsilon is None:
            epsilon = np.finfo(dtype).eps

        # Declare internal solver data
        self._epsilon: float = float(epsilon)
        self._max_iterations: int = int(max_iterations)

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            record_errors=record_errors,
            record_residuals=record_residuals,
        )

    @override
    def _solve_iterative_impl(self, b: np.ndarray, x_0: np.ndarray | None = None) -> LinearSolution:
        return conjugate_gradient(
            A=self._matrix,
            b=b,
            x_0=x_0,
            epsilon=self._epsilon,
            atol=self._atol,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
            record_errors=self._record_errors,
            record_residuals=self._record_residuals,
        )


class MinimumResidualSolver(IndirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: np.dtype = np.float64,
        record_errors: bool = False,
        record_residuals: bool = False,
        max_iterations: int = 1000,
        epsilon: float | None = None,
    ):
        # Default epsilon to machine epsilon if not provided
        if epsilon is None:
            epsilon = np.finfo(dtype).eps

        # Declare internal solver data
        self._epsilon: float = float(epsilon)
        self._max_iterations: int = int(max_iterations)

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            record_errors=record_errors,
            record_residuals=record_residuals,
        )

    @override
    def _solve_iterative_impl(self, b: np.ndarray, x_0: np.ndarray | None = None) -> LinearSolution:
        return minimum_residual(
            A=self._matrix,
            b=b,
            x_0=x_0,
            epsilon=self._epsilon,
            atol=self._atol,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
            record_errors=self._record_errors,
            record_residuals=self._record_residuals,
        )


###
# Direct Solvers
###


class LLTStdSolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # Declare matrix factorization data
        self._L: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._L = factorize.llt_std_lower(A, False)

    @override
    def _unpack_impl(self) -> None:
        pass

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._L @ self._L.T

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        x[:] = factorize.llt_std_lower_solve(self._L, x)


class LDLTNoPivotSolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # Declare matrix factorization data
        self._L: np.ndarray | None = None
        self._d: np.ndarray | None = None

        # Declare optional unpacked factors
        self._D: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._L, self._d = factorize.ldlt_nopivot_lower(A=A, tol=self._ftol, use_zero_correction=False)

    @override
    def _unpack_impl(self) -> None:
        self._D = np.diag(self._d)

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L, self._D

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._L @ np.diag(self._d) @ self._L.T

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        factorize.ldlt_nopivot_lower_solve_inplace(self._L, self._d, x)


class LDLTBlockedSolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
        blocksize: int = 1,
        upper: bool = False,
    ):
        # Initialize solver configurations
        self._blocksize: int = int(blocksize)
        self._upper: bool = bool(upper)

        # Declare matrix factorization data
        self._LU: np.ndarray | None = None
        self._D: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        if self._upper:
            self._LU, self._D = factorize.ldlt_blocked_upper(A, self._blocksize)
        else:
            self._LU, self._D = factorize.ldlt_blocked_lower(A, self._blocksize)

    @override
    def _unpack_impl(self) -> None:
        pass

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._LU, self._D

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._LU @ self._D @ self._LU.T

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        if self._upper:
            factorize.ldlt_nopivot_upper_solve_inplace(self._LU, self._D, x)
        else:
            factorize.ldlt_nopivot_lower_solve_inplace(self._LU, self._D, x)


class LDLTBunchKaufmanSolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        itype: np.dtype = np.int64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # Initialize solver configurations
        self._itype: np.dtype = itype

        # Declare matrix factorization data
        self._L: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._p: np.ndarray | None = None

        # Declare optional unpacked factors
        self._P: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._L, self._D, self._p = factorize.ldlt_bk_lower(
            A=A, tol=self._ftol, itype=self._itype, check_symmetry=False, use_zero_correction=False
        )

    @override
    def _unpack_impl(self) -> None:
        self._P = np.eye(self._matrix.shape[0], dtype=self._dtype)[:, self._p]

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L, self._D, self._P

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return factorize.ldlt_bk_lower_reconstruct(self._L, self._D, self._p)

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        x[:] = factorize.ldlt_bk_lower_solve(self._matrix, self._D, self._p, x, self._atol)


class LDLTEigen3Solver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        itype: np.dtype = np.int64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # Initialize solver configurations
        self._itype: np.dtype = itype

        # Declare internal solver data
        self._scratch: np.ndarray | None = None

        # Declare matrix factorization data
        self._LD: np.ndarray | None = None
        self._p: np.ndarray | None = None

        # Declare optional unpacked factors
        self._L: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._P: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._LD, self._p, self._scratch, self._sign, self._success = factorize.ldlt_eigen3_lower(
            A=A, itype=self._itype
        )
        # Immediately unpack the factors for user access
        self._unpack_impl()

    @override
    def _unpack_impl(self) -> None:
        self._L, self._D, self._P = factorize.ldlt_eigen3_lower_unpack(self._LD, self._p)

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L, self._D, self._P

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._P @ (self._L @ self._D @ self._L.T) @ self._P.T

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        factorize.ldlt_eigen3_lower_solve_inplace(self._LD, self._p, x, self._atol)


class LUNoPivotSolver(DirectSolver):
    def __init__(
        self,
        A: np.ndarray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: np.dtype = np.float64,
        compute_error: bool = False,
        check_error: bool = False,
        check_symmetry: bool = False,
    ):
        # Declare matrix factorization data
        self._L: np.ndarray | None = None
        self._U: np.ndarray | None = None

        # Initialize base class
        super().__init__(
            A=A,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            compute_error=compute_error,
            check_error=check_error,
            check_symmetry=check_symmetry,
        )

    @override
    def _factorize_impl(self, A: np.ndarray) -> None:
        self._L, self._U = factorize.lu_nopiv(A, self._ftol)

    @override
    def _unpack_impl(self) -> None:
        pass

    @override
    def _get_unpacked_impl(self) -> Any:
        return self._L, self._U

    @override
    def _reconstruct_impl(self) -> np.ndarray:
        return self._L @ self._U

    @override
    def _solve_inplace_impl(self, x: np.ndarray) -> None:
        y = factorize.lu_nopiv_solve_forward_lower(self._L, x)
        x[:] = factorize.lu_nopiv_solve_backward_upper(self._U, y, tol=self._atol).squeeze()


###
# Summary
###

LinearSolverType = (
    NumPySolver
    | SciPySolver
    | LLTNumPySolver
    | LLTSciPySolver
    | LDLTSciPySolver
    | LUSciPySolver
    | LDLTNoPivotSolver
    | JacobiSolver
    | GaussSeidelSolver
    | SORSolver
    | ConjugateGradientSolver
    | MinimumResidualSolver
    | LLTStdSolver
    | LDLTBlockedSolver
    | LDLTBunchKaufmanSolver
    | LDLTEigen3Solver
    | LUNoPivotSolver
)
"""Type alias over all linear solvers."""
