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
KAMINO: Linear Algebra: Core types and utilities for multi-linear systems

This module provides data structures and utilities for managing multiple
independent linear systems, including rectangular and square systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import warp as wp
from warp.context import Devicelike

from ..core.types import FloatType, IntType

###
# Module interface
###

__all__ = [
    "LinearInfo",
    "LinearOperator",
]

###
# Types
###


@dataclass
class LinearInfo(ABC):
    @property
    @abstractmethod
    def num_matrices(self):
        raise NotImplementedError("`num_matrices` property is not implemented.")


#     dtype: FloatType = float32
#     """The data type of the underlying matrix and vector data arrays."""

#     itype: IntType = int32
#     """The integer type used for indexing the underlying data arrays."""

#     device: Devicelike | None = None
#     """The device on which the data arrays are allocated."""

#     dimensions: list[tuple[int, int]] | None = None
#     """Host-side cache of the dimensions of each rectangular linear system."""

#     max_dimensions: tuple[int, int] = (0, 0)
#     """Host-side cache of the maximum dimension over all matrix blocks."""

#     total_mat_size: int = 0
#     """
#     Host-side cache of the total size of the flat matrix data array.
#     This is equal to `sum(maxdim[i][0]*maxdim[i][1] for i in range(num_blocks))`.
#     """

#     total_rhs_size: int = 0
#     """
#     Host-side cache of the total size of the flat data array of rhs vectors.
#     This is equal to `sum(maxdim[i][1] for i in range(num_blocks))`.
#     """

#     total_inp_size: int = 0
#     """
#     Host-side cache of the total size of the flat data array of input vectors.
#     This is equal to `sum(maxdim[i][0] for i in range(num_blocks))`.
#     """

#     maxdim: wp.array | None = None
#     """
#     The maximum dimensions of each rectangular matrix block.
#     Shape of ``(num_blocks,)`` and type :class:`vec2i`.
#     Each entry corresponds to the shape `(max_rows, max_cols)`.
#     """

#     dim: wp.array | None = None
#     """
#     The active dimensions of each rectangular matrix block.
#     Shape of ``(num_blocks,)`` and type :class:`vec2i`.
#     Each entry corresponds to the shape `(rows, cols)`.
#     """

#     mio: wp.array | None = None
#     """
#     The matrix index offset (mio) of each block in the flat data array.
#     Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
#     """

#     rvio: wp.array | None = None
#     """
#     The rhs vector index offset (vio) of each block in the flat data array.
#     Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
#     """

#     ivio: wp.array | None = None
#     """
#     The input vector index offset (vio) of each block in the flat data array.
#     Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
#     """

###
# Types
###


@dataclass
class LinearOperator(ABC):
    """
    An abstract base class for a linear operator with support for matrix-vector products.
    """

    info: LinearInfo | None = None
    """The multi-linear data structure describing the operator."""

    ###
    # Properties
    ###

    @property
    def dtype(self) -> FloatType | None:
        if self.info is None:
            return None
        return self.info.dtype

    @property
    def itype(self) -> IntType | None:
        if self.info is None:
            return None
        return self.info.itype

    @property
    def device(self) -> Devicelike | None:
        if self.info is None:
            return None
        return self.info.device

    @property
    def active_dims(self) -> wp.array | None:
        if self.info is None:
            return None
        return self.info.dim

    @property
    def n_worlds(self) -> int:
        if self.info is None:
            return 0
        return self.info.num_blocks

    @property
    def max_dim(self) -> int:
        if self.info is None:
            return 0
        return self.info.max_dimension

    ###
    # Operations
    ###

    @abstractmethod
    def gemv(self, x: wp.array, y: wp.array, matrix_mask: wp.array, alpha: float = 1.0, beta: float = 0.0) -> None:
        raise NotImplementedError("General matrix-vector product is not implemented.")

    def matvec(self, x: wp.array, y: wp.array, matrix_mask: wp.array) -> None:
        self.gemv(x, y, matrix_mask)
