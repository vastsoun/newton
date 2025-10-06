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
Linear system solvers for multiple independent linear systems.

This module provides interfaces for and implementations of linear
system solvers, that can solve multiple independent linear systems
in parallel, with support for both rectangular and square systems.
Depending on the particular solver implementation, both inter- and
intra-system parallelism may be exploited.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import warp as wp
from warp.context import Devicelike

from ..core.types import FloatType, float32, override
from . import conjugate, factorize
from .core import DenseLinearOperatorData, DenseSquareMultiLinearInfo, make_dtype_tolerance

###
# Module interface
###

__all__ = [
    "ConjugateGradientSolver",
    "DirectSolver",
    "LLTBlockedSolver",
    "LLTSequentialSolver",
    "LinearSolver",
    "LinearSolverType",
]


###
# Interfaces
###


class LinearSolver(ABC):
    """
    An abstract base class for linear system solvers.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: FloatType = float32,
        device: Devicelike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Declare and initialize the internal reference to the matrix/operator data
        self._operator: DenseLinearOperatorData | None = operator

        # Override dtype if linear operator is provided
        if operator is not None:
            dtype = operator.info.dtype

        # Declare and initialize internal meta-data
        self._dtype: Any = dtype
        self._atol: float = atol
        self._rtol: float = rtol

        # Declare and initialize the device identifier
        self._device: Devicelike = device

        # If an operator is provided, proceed with any necessary memory allocations
        if operator is not None:
            self.allocate(operator, **kwargs)

    ###
    # Properties
    ###

    @property
    def operator(self) -> DenseLinearOperatorData:
        if self._operator is None:
            raise ValueError("No linear operator has been allocated!")
        return self._operator

    @property
    def dtype(self) -> FloatType:
        return self._dtype

    @property
    def device(self) -> Devicelike:
        return self._device

    ###
    # Internals
    ###

    def _set_tolerance_dtype(self):
        self._atol = make_dtype_tolerance(self._atol, dtype=self._dtype)
        self._rtol = make_dtype_tolerance(self._rtol, dtype=self._dtype)

    ###
    # Implementation API
    ###

    @abstractmethod
    def _allocate_impl(self, operator: DenseLinearOperatorData, **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("An allocation operation is not implemented.")

    @abstractmethod
    def _reset_impl(self, A: wp.array, **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A reset operation is not implemented.")

    @abstractmethod
    def _compute_impl(self, A: wp.array, **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A compute operation is not implemented.")

    @abstractmethod
    def _solve_impl(self, b: wp.array, x: wp.array, **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A solve operation is not implemented.")

    @abstractmethod
    def _solve_inplace_impl(self, x: wp.array, **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A solve-in-place operation is not implemented.")

    ###
    # Public API
    ###

    def allocate(self, operator: DenseLinearOperatorData, **kwargs: dict[str, Any]) -> None:
        """
        Ingest a linear operator and allocate any necessary internal memory
        based on the multi-linear layout specified by the operator's info.
        """
        # Check the operator is valid
        if operator is None:
            raise ValueError("A valid linear operator must be provided!")
        if not isinstance(operator, DenseLinearOperatorData):
            raise ValueError("The provided operator is not a DenseLinearOperatorData instance!")
        if operator.info is None:
            raise ValueError("The provided operator does not have any associated info!")
        self._operator = operator
        self._dtype = operator.info.dtype
        self._set_tolerance_dtype()
        self._allocate_impl(operator, **kwargs)

    def reset(self) -> None:
        """Resets the internal solver data (e.g. possibly to zeros)."""
        self._reset_impl()

    def compute(self, A: wp.array, **kwargs: dict[str, Any]) -> None:
        """Ingest matrix data and pre-compute any rhs-independent intermediate data."""
        if not self._operator.info.is_matrix_compatible(A):
            raise ValueError("The provided flat matrix data array does not have enough memory!")
        self._compute_impl(A=A, **kwargs)

    def solve(self, b: wp.array, x: wp.array, **kwargs: dict[str, Any]) -> None:
        """Solves the multi-linear systems `A @ x = b`."""
        if not self._operator.info.is_rhs_compatible(b):
            raise ValueError("The provided flat rhs vector data array does not have enough memory!")
        if not self._operator.info.is_input_compatible(x):
            raise ValueError("The provided flat input vector data array does not have enough memory!")
        self._solve_impl(b=b, x=x, **kwargs)

    def solve_inplace(self, x: wp.array, **kwargs: dict[str, Any]) -> None:
        """Solves the multi-linear systems `A @ x = b` in-place, where `x` is initialized with rhs data."""
        if not self._operator.info.is_input_compatible(x):
            raise ValueError("The provided flat input vector data array does not have enough memory!")
        self._solve_inplace_impl(x=x, **kwargs)

    def _solve_inplace_impl(self, x: wp.array, **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A solve-in-place operation is not implemented.")


class DirectSolver(LinearSolver):
    """
    An abstract base class for direct linear system solvers based on matrix factorization.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: FloatType = float32,
        device: Devicelike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Default factorization tolerance to machine epsilon if not provided
        ftol = make_dtype_tolerance(ftol, dtype=dtype)

        # Initialize internal meta-data
        self._ftol: float | None = ftol
        self._has_factors: bool = False

        # Initialize base class members
        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    ###
    # Internals
    ###

    def _check_has_factorization(self):
        """Checks if the factorization has been computed, otherwise raises error."""
        if not self._has_factors:
            raise ValueError("A factorization has not been computed!")

    ###
    # Implementation API
    ###

    @abstractmethod
    def _factorize_impl(self, A: wp.array, **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A matrix factorization implementation is not provided.")

    @abstractmethod
    def _reconstruct_impl(self, A: wp.array, **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A matrix reconstruction implementation is not provided.")

    ###
    # Internals
    ###

    @override
    def _compute_impl(self, A: wp.array, **kwargs: dict[str, Any]):
        self._factorize(A, **kwargs)

    def _factorize(self, A: wp.array, ftol: float | None = None, **kwargs: dict[str, Any]) -> None:
        # Override the current tolerance if provided otherwise ensure
        # it does not exceed machine precision for the current dtype
        if ftol is not None:
            self._ftol = make_dtype_tolerance(ftol, dtype=self._dtype)
        else:
            self._ftol = make_dtype_tolerance(self._ftol, dtype=self._dtype)

        # Factorize the specified matrix data and store any intermediate data
        self._factorize_impl(A, **kwargs)
        self._has_factors = True

    ###
    # Public API
    ###

    def reconstruct(self, A: wp.array, **kwargs: dict[str, Any]) -> None:
        """Reconstructs the original matrix from the current factorization."""
        self._check_has_factorization()
        self._reconstruct_impl(A, **kwargs)


###
# Direct solvers
###


class LLTSequentialSolver(DirectSolver):
    """
    A LLT (i.e. Cholesky) factorization class computing each matrix block sequentially.\n
    This parallelizes the factorization and solve operations over each block\n
    and supports heterogeneous matrix block sizes.\n
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: FloatType = float32,
        device: Devicelike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Declare LLT-specific internal data
        self._L: wp.array | None = None
        """A flat array containing the Cholesky factorization of each matrix block."""
        self._y: wp.array | None = None
        """A flat array containing the intermediate results for the solve operation."""

        # Initialize base class members
        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    ###
    # Properties
    ###

    @property
    def L(self) -> wp.array:
        if self._L is None:
            raise ValueError("The factorization array has not been allocated!")
        return self._L

    @property
    def y(self) -> wp.array:
        if self._y is None:
            raise ValueError("The intermediate result array has not been allocated!")
        return self._y

    ###
    # Implementation
    ###

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData, **kwargs: dict[str, Any]) -> None:
        # Check the operator has info
        if A.info is None:
            raise ValueError("The provided operator does not have any associated info!")

        # Ensure that the underlying operator is compatible with LLT
        if not isinstance(A.info, DenseSquareMultiLinearInfo):
            raise ValueError("LLT factorization requires a square matrix.")

        # Allocate the Cholesky factorization matrix and the
        # intermediate result buffer on the specified device
        with wp.ScopedDevice(self._device):
            self._L = wp.zeros(shape=(self._operator.info.total_mat_size,), dtype=self._dtype)
            self._y = wp.zeros(shape=(self._operator.info.total_vec_size,), dtype=self._dtype)

    @override
    def _reset_impl(self) -> None:
        self._L.zero_()
        self._y.zero_()
        self._has_factors = False

    @override
    def _factorize_impl(self, A: wp.array) -> None:
        factorize.llt_sequential_factorize(
            num_blocks=self._operator.info.num_blocks,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            A=A,
            L=self._L,
            device=self._device,
        )

    @override
    def _reconstruct_impl(self, A: wp.array) -> None:
        raise NotImplementedError("LLT matrix reconstruction is not yet implemented.")

    @override
    def _solve_impl(self, b: wp.array, x: wp.array) -> None:
        # Solve the system L * y = b and L^T * x = y
        factorize.llt_sequential_solve(
            num_blocks=self._operator.info.num_blocks,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            vio=self._operator.info.vio,
            L=self._L,
            b=b,
            y=self._y,
            x=x,
            device=self._device,
        )

    @override
    def _solve_inplace_impl(self, x: wp.array) -> None:
        # Solve the system L * y = x and L^T * x = y
        factorize.llt_sequential_solve_inplace(
            num_blocks=self._operator.info.num_blocks,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            vio=self._operator.info.vio,
            L=self._L,
            x=x,
        )


class LLTBlockedSolver(DirectSolver):
    """
    A Blocked LLT (i.e. Cholesky) factorization class computing each matrix block with Tile-based parallelism.\n
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData | None = None,
        block_size: int = 16,
        solve_block_dim: int = 64,
        factortize_block_dim: int = 128,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: FloatType = float32,
        device: Devicelike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Declare LLT-specific internal data
        self._L: wp.array | None = None
        """A flat array containing the Cholesky factorization of each matrix block."""
        self._y: wp.array | None = None
        """A flat array containing the intermediate results for the solve operation."""

        # Cache the fixed block size
        self._block_size: int = block_size
        self._solve_block_dim: int = solve_block_dim
        self._factortize_block_dim: int = factortize_block_dim

        # Create the factorization and solve kernels
        self._factorize_kernel = factorize.make_llt_blocked_factorize_kernel(block_size)
        self._solve_kernel = factorize.make_llt_blocked_solve_kernel(block_size)
        self._solve_inplace_kernel = factorize.make_llt_blocked_solve_inplace_kernel(block_size)

        # Initialize base class members
        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    ###
    # Properties
    ###

    @property
    def L(self) -> wp.array:
        if self._L is None:
            raise ValueError("The factorization array has not been allocated!")
        return self._L

    @property
    def y(self) -> wp.array:
        if self._y is None:
            raise ValueError("The intermediate result array has not been allocated!")
        return self._y

    ###
    # Implementation
    ###

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData, **kwargs: dict[str, Any]) -> None:
        # Check the operator has info
        if A.info is None:
            raise ValueError("The provided operator does not have any associated info!")

        # Ensure that the underlying operator is compatible with LLT
        if not isinstance(A.info, DenseSquareMultiLinearInfo):
            raise ValueError("LLT factorization requires a square matrix.")

        # Allocate the Cholesky factorization matrix and the
        # intermediate result buffer on the specified device
        with wp.ScopedDevice(self._device):
            self._L = wp.zeros(shape=(self._operator.info.total_mat_size,), dtype=self._dtype)
            self._y = wp.zeros(shape=(self._operator.info.total_vec_size,), dtype=self._dtype)

    @override
    def _reset_impl(self) -> None:
        self._L.zero_()
        self._y.zero_()
        self._has_factors = False

    @override
    def _factorize_impl(self, A: wp.array) -> None:
        factorize.llt_blocked_factorize(
            kernel=self._factorize_kernel,
            num_blocks=self._operator.info.num_blocks,
            block_dim=self._factortize_block_dim,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            A=A,
            L=self._L,
        )

    @override
    def _reconstruct_impl(self, A: wp.array) -> None:
        raise NotImplementedError("LLT matrix reconstruction is not yet implemented.")

    @override
    def _solve_impl(self, b: wp.array, x: wp.array) -> None:
        # Solve the system L * y = b and L^T * x = y
        factorize.llt_blocked_solve(
            kernel=self._solve_kernel,
            num_blocks=self._operator.info.num_blocks,
            block_dim=self._solve_block_dim,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            vio=self._operator.info.vio,
            L=self._L,
            b=b,
            y=self._y,
            x=x,
            device=self._device,
        )

    @override
    def _solve_inplace_impl(self, x: wp.array) -> None:
        # Solve the system L * y = b and L^T * x = y
        factorize.llt_blocked_solve_inplace(
            kernel=self._solve_inplace_kernel,
            num_blocks=self._operator.info.num_blocks,
            block_dim=self._solve_block_dim,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            vio=self._operator.info.vio,
            L=self._L,
            y=self._y,
            x=x,
            device=self._device,
        )


###
# Summary
###


class ConjugateGradientSolver(LinearSolver):
    """
    A wrapper around the batched Conjugate Gradient implementation in `conjugate.cg`.

    This solves multiple independent SPD systems using a batched operator.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: FloatType = float32,
        device: Devicelike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Iterative-solver specific buffers
        self._A_op = None
        self._env_active: wp.array | None = None
        self._atol_sq: wp.array | None = None
        self._num_envs: int = 0
        self._max_dim: int = 0

        # Solve metadata caches (device arrays)
        self._last_iter: wp.array | None = None
        self._last_resid_sq: wp.array | None = None

        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    @override
    def _allocate_impl(self, operator: DenseLinearOperatorData, **kwargs: dict[str, Any]) -> None:
        if operator.info is None:
            raise ValueError("The provided operator does not have any associated info!")
        if not isinstance(operator.info, DenseSquareMultiLinearInfo):
            raise ValueError("ConjugateGradientSolver requires a square matrix operator.")

        dim_values = set(operator.info.maxdim.numpy().tolist())
        if len(dim_values) > 1:
            raise ValueError(f"ConjugateGradientSolver requires all blocks to have the same dimension ({dim_values}).")

        # Cache env count and per-env padded dimension
        self._num_envs = operator.info.num_blocks
        self._max_dim = int(operator.info.maxdim.numpy()[0])

        with wp.ScopedDevice(self._device):
            self._env_active = wp.full(operator.info.num_blocks, True, dtype=wp.bool)
            # Initialize absolute tolerance from dtype if not provided; store squared
            self._set_tolerance_dtype()
            atol_val = self._atol if self._atol is not None else make_dtype_tolerance(None, dtype=self._dtype)
            print(f"atol: {atol_val}")
            self._atol_sq = wp.full(operator.info.num_blocks, float(atol_val) ** 2, dtype=self._dtype)

        self._A_op = conjugate.make_dense_square_matrix_operator(
            A=operator.mat.reshape((self._num_envs, self._max_dim * self._max_dim)),
            active_dims=self._operator.info.dim,
            max_dims=self._max_dim,
            matrix_stride=self._max_dim,
        )

        self.solver = conjugate.CGSolver(
            A=self._A_op,
            active_dims=self._operator.info.dim,
            env_active=self._env_active,
            atol_sq=self._atol_sq,
            maxiter=None,
            M=None,
            callback=None,
            check_every=0,
            use_cuda_graph=True,
        )

    @override
    def _reset_impl(self, A: wp.array | None = None, **kwargs: dict[str, Any]) -> None:
        self._last_iter = None
        self._last_resid_sq = None

    @override
    def _compute_impl(self, A: wp.array, **kwargs: dict[str, Any]) -> None:
        pass
        # $print(f"Linear solve(): active_dims={self._operator.info.dim}, maxdims={self._operator.info.maxdim}")
        # TODO: check that data remains in same place

    @override
    def _solve_impl(self, b: wp.array, x: wp.array, **kwargs: dict[str, Any]) -> None:
        if self._A_op is None:
            raise ValueError("ConjugateGradientSolver.compute(A) must be called before solve().")

        zero_x = bool(kwargs.get("zero_x"))
        if zero_x:
            x.zero_()

        self._last_iter, self._last_resid_sq, _ = self.solver.solve(
            b=b.reshape((self._num_envs, self._max_dim)),
            x=x.reshape((self._num_envs, self._max_dim)),
        )

    def solve_metadata(self) -> dict[str, Any]:
        if self._last_iter is None or self._last_resid_sq is None:
            raise ValueError("No solve metadata available; call solve() first.")
        # Host summaries
        iters = self._last_iter.numpy()
        resid_sq = self._last_resid_sq.numpy()
        return {
            "final_iteration": int(iters.max() if len(iters) > 0 else 0),
            "residual_norm": float(np.sqrt(resid_sq.max() if len(resid_sq) > 0 else 0.0)),
            "atol": float(np.sqrt(self._atol_sq.numpy().max() if self._atol_sq is not None else 0.0)),
        }


LinearSolverType = LLTSequentialSolver | LLTBlockedSolver | ConjugateGradientSolver
"""Type alias over all linear solvers."""
