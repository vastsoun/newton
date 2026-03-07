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
TODO
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..core.types import FloatType, IntType, int32
from ..linalg.linear import LinearSolverType
from ..linalg.sparse_matrix import BlockSparseMatrices

###
# Module interface
###

__all__ = [
    # "TODO",
]

###
# TEMP
###


@dataclass
class MatricesSize:
    """
    TODO
    """

    num_matrices: int = 0
    """
    Host-side cache of the number of matrices described by this container.\n
    Alternatively, it can be set directly if the container is constructed explicitly.
    """

    sum_of_max_size: int = 0
    """
    Host-side cache of the sum of the maximum matrix sizes over all matrices.\n
    The size is defined as the number of dtype elements allocated to represent the matrix.
    """

    max_of_max_size: int = 0
    """
    Host-side cache of the maximum of the maximum matrix sizes over all matrices.\n
    The size is defined as the number of dtype elements allocated to represent the matrix.
    """

    max_of_max_dims: tuple[int, int] = (0, 0)
    """
    Host-side cache of the maximum of the maximum matrix dimensions over all matrices.
    """


@dataclass
class MatricesInfo:
    """
    TODO
    """

    device: wp.DeviceLike | None = None
    """Host-side cache of the device on which all data arrays are allocated."""

    index_dtype: IntType = int32
    """Host-side cache of the integer type used for indexing the underlying data arrays."""

    dims: wp.array | None = None
    """
    The active dimensions of each sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`vec2(index_dtype)`.
    """

    max_dims: wp.array | None = None
    """
    The maximum dimensions of each sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`vec2(index_dtype)`.
    """

    row_start: wp.array | None = None
    """
    The start index of each row vector block in a flattened data array of size sum_of_max_rows.\n
    Shape of ``(num_matrices,)`` and type :attr:`index_dtype`.
    """

    col_start: wp.array | None = None
    """
    The start index of each column vector block in a flattened data array of size sum_of_max_cols.\n
    Shape of ``(num_matrices,)`` and type :attr:`index_dtype`.
    """


class DenseMatrices:
    """
    A container to represent multiple dense matrices.
    """

    def __init__(
        self,
        max_dims: list[tuple[int, int]],
        index_dtype: IntType | None = None,
        device: wp.DeviceLike | None = None,
    ):
        """
        TODO
        """

        self.size: MatricesSize
        """
        Meta-data container holding both host-side caches and on-device
        arrays describing the dimensions and indexing of the dense matrices.
        """

        self.info: MatricesInfo
        """
        Meta-data container holding both host-side caches and on-device
        arrays describing the dimensions and indexing of the dense matrices.
        """

        self.data: wp.array
        """
        Flattened array containing all scalar values over all dense matrices.\n
        Shape of ``(sum_of_num_elements,)`` and type :class:`float | vector | matrix`.
        """

        if max_dims is not None:
            self.finalize(max_dims, index_dtype=index_dtype, device=device)

    ###
    # Operations
    ###

    def finalize(
        self,
        max_dims: list[tuple[int, int]],
        index_dtype: IntType | None = None,
        device: wp.DeviceLike | None = None,
    ):
        """
        TODO
        """
        pass

    def clear(self):
        """Clears all variable data."""
        pass

    def zero(self):
        """Sets all variable data to zero."""
        pass

    def assign(self, matrices: list[np.ndarray]):
        """
        Assigns data to all dense matrices from a list of dense NumPy arrays.

        This operation assumes that:
        - the dense matrices have been finalized
        - the provided dense arrays match the active dimensions of each dense matrix specified in `dims`
        - values are stored in row-major order

        Args:
            data (list[np.ndarray]):
                A list of dense NumPy arrays to assign to each dense matrix.
        """
        pass

    def numpy(self) -> list[np.ndarray]:
        """Converts all dense matrices to a list of dense NumPy arrays."""
        pass


###
# Interfaces
###


class LinearOperators:
    def __init__(self):
        pass

    @property
    def device(self) -> wp.DeviceLike:
        pass

    @property
    def dtype(self) -> FloatType:
        pass

    @property
    def num_matrices(self) -> int:
        pass

    @property
    def sum_of_max_dims(self) -> tuple[int, int]:
        pass

    @property
    def max_of_max_dims(self) -> tuple[int, int]:
        pass

    @property
    def sum_of_max_size(self) -> int:
        pass

    @property
    def max_of_max_size(self) -> int:
        pass

    @property
    def info(self) -> MatricesInfo:
        pass

    @property
    def solver(self) -> LinearSolverType:
        pass

    def finalize(self):
        pass

    def zero(self):
        pass

    def compute(self, reset_to_zero: bool = True):
        pass

    def solve(self, v: wp.array, x: wp.array):
        pass

    def solve_inplace(self, x: wp.array):
        pass

    def matvec(self, x: wp.array, y: wp.array, world_mask: wp.array):
        pass

    def matvec_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array):
        pass

    def gemv(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        pass

    def gemv_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        pass


class DenseLinearOperators:
    def __init__(self):
        pass

    @property
    def device(self) -> wp.DeviceLike:
        pass

    @property
    def dtype(self) -> FloatType:
        pass

    @property
    def num_matrices(self) -> int:
        pass

    @property
    def sum_of_max_dims(self) -> tuple[int, int]:
        pass

    @property
    def max_of_max_dims(self) -> tuple[int, int]:
        pass

    @property
    def sum_of_max_size(self) -> int:
        pass

    @property
    def max_of_max_size(self) -> int:
        pass

    @property
    def info(self) -> MatricesInfo:
        pass

    @property
    def matrices(self) -> DenseMatrices:
        pass

    @property
    def solver(self) -> LinearSolverType:
        pass

    def finalize(self):
        pass

    def zero(self):
        pass

    def compute(self, reset_to_zero: bool = True):
        pass

    def solve(self, v: wp.array, x: wp.array):
        pass

    def solve_inplace(self, x: wp.array):
        pass

    def matvec(self, x: wp.array, y: wp.array, world_mask: wp.array):
        pass

    def matvec_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array):
        pass

    def gemv(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        pass

    def gemv_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        pass

    # -------- DENSE ONLY --------- #

    def regularize(self, eta: wp.array):
        pass


class BlockSparseLinearOperators:
    def __init__(self):
        pass

    @property
    def device(self) -> wp.DeviceLike:
        pass

    @property
    def dtype(self) -> FloatType:
        pass

    @property
    def num_matrices(self) -> int:
        pass

    @property
    def sum_of_max_dims(self) -> tuple[int, int]:
        pass

    @property
    def max_of_max_dims(self) -> tuple[int, int]:
        pass

    @property
    def sum_of_max_size(self) -> int:
        pass

    @property
    def max_of_max_size(self) -> int:
        pass

    @property
    def info(self) -> MatricesInfo:
        pass

    @property
    def matrices(self) -> BlockSparseMatrices:
        pass

    def finalize(self):
        pass

    def compute(self, reset_to_zero: bool = True):
        pass

    def solve(self, v: wp.array, x: wp.array):
        pass

    def solve_inplace(self, x: wp.array):
        pass

    def matvec(self, x: wp.array, y: wp.array, world_mask: wp.array):
        pass

    def matvec_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array):
        pass

    def gemv(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        pass

    def gemv_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        pass

    # -------- SPARSE ONLY --------- #

    def set_regularization(self, eta: wp.array | None):
        pass

    def set_preconditioner(self, preconditioner: wp.array | None):
        pass

    def get_diagonal(self, diag: wp.array):
        pass
