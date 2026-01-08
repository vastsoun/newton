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
KAMINO: Linear Algebra: Core types and utilities for sparse multi-world linear systems

This module provides data structures and utilities for managing multiple
independent linear systems, including rectangular and square systems.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp
from warp.context import Devicelike

from ..core.types import FloatType, IntType, float32, int32

# from ..utils import logger as msg

###
# Module interface
###

__all__ = [
    "BlockSparseLinearOperators",
    "BlockSparseMatrices",
]


###
# Types
###


class SparseBlockType:
    """A utility type for bundling meta-data about sparse-block types."""

    _supported_types = float | wp.vector | wp.matrix


@dataclass
class BlockSparseMatrices:
    """
    A container for representing multiple block-sparse matrices of fixed block size.
    """

    ###
    # Host-side Metadata
    ###

    device: Devicelike | None = None
    """The device on which the data arrays are allocated."""

    dtype: FloatType = float32
    """The data type of the underlying matrix and vector data arrays."""

    itype: IntType = int32
    """The integer type used for indexing the underlying data arrays."""

    num_matrices: int = 0
    """Host-side cache of the number of sparse matrices represented by this container."""

    sum_of_num_nzb: int = 0
    """Host-side cache of the sum of the number of non-zero sub-blocks over all sparse matrices."""

    max_of_num_nzb: int = 0
    """Host-side cache of the maximum number of non-zero sub-blocks over all sparse matrices."""

    nzb_size: tuple[int, int] | None = None
    """Host-side cache of the fixed non-zero sub-block dimensions contained in all sparse matrices."""

    ###
    # On-device Data
    ###

    max_dims: wp.array | None = None
    """
    The maximum dimensions of each sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`vec2i`.
    """

    dims: wp.array | None = None
    """
    The active dimensions of each sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`vec2i`.
    """

    max_nzb: wp.array | None = None
    """
    The maximum number of non-zero blocks per sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`int`.
    """

    num_nzb: wp.array | None = None
    """
    The active number of non-zero blocks per sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`int`.
    """

    nzb_start: wp.array | None = None
    """
    The index of the first non-zero sub-block of each sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`int`.
    """

    nzb_coords: wp.array | None = None
    """
    The row-column coordinates of each sparse sub-block within its corresponding matrices.\n
    Shape of ``(sum_of_num_nzb,)`` and type :class:`vec2i`.
    """

    nzb_values: wp.array | None = None
    """
    The flattened array containing all sparse non-zero blocks over all matrices.\n
    Shape of ``(sum_of_num_nzb,)`` and type :class:`float | vector | matrix`.
    """

    ###
    # Operations
    ###

    def finalize(capacities: list[int], block_type: SparseBlockType, device: Devicelike | None = None):
        pass

    def clear(self):
        """Clears all variable sub-blocks."""
        self._assert_is_finalized()
        self.dims.zero_()
        self.num_nzb.zero_()
        self.nzb_coords.zero_()

    def zero(self):
        """Sets all sub-block data to zero."""
        self._assert_is_finalized()
        self.nzb_values.zero_()

    def numpy(self) -> list[np.ndarray]:
        pass

    ###
    # Internals
    ###

    def _assert_is_finalized(self):
        # TODO: Check all array attributes
        if self.nzb_values is None:
            raise RuntimeError("No data has been allocated. Call `finalize()` before use.")


@dataclass
class BlockSparseLinearOperators:
    """
    A Block-Sparse Linear Operator container for representing
    and operating on multiple independent sparse linear systems.
    """

    ###
    # On-device Data
    ###

    bsm: BlockSparseMatrices | None = None
    """
    The underlying block-sparse matrix used by this operator.
    """

    row_start: wp.array | None = None
    """
    The start index of each row vector block in flattened data arrays.\n
    Shape of ``(num_superblocks,)`` and type :class:`int`.
    """

    col_start: wp.array | None = None
    """
    The start index of each column vector block in flattened data arrays.\n
    Shape of ``(num_superblocks,)`` and type :class:`int`.
    """

    ###
    # Operators
    ###

    Ax_op: Callable[["BlockSparseLinearOperators", wp.array, wp.array], None] | None = None
    """
    The operator function for performing sparse matrix-vector products `y = A @ x`.\n
    Signature: ``Ax_op(A: BlockSparseLinearOperators, x: wp.array, y: wp.array)``.
    """

    ATy_op: Callable[["BlockSparseLinearOperators", wp.array, wp.array], None] | None = None
    """
    The operator function for performing sparse matrix-transpose-vector products `x = A^T @ y`.\n
    Signature: ``ATy_op(A: BlockSparseLinearOperators, y: wp.array, x: wp.array)``.
    """

    gemv_op: Callable[["BlockSparseLinearOperators", wp.array, wp.array, float, float, bool], None] | None = None
    """
    The operator function for performing generalized sparse matrix-vector products `y = alpha * A @ x + beta * y`.\n
    Signature: ``gemv_op(A: BlockSparseLinearOperators, x: wp.array, y: wp.array, alpha: float, beta: float)``.
    """

    gemvt_op: Callable[["BlockSparseLinearOperators", wp.array, wp.array, float, float, bool], None] | None = None
    """
    The operator function for performing generalized sparse matrix-transpose-vector products `x = alpha * A^T @ y + beta * x`.\n
    Signature: ``gemvt_op(A: BlockSparseLinearOperators, y: wp.array, x: wp.array, alpha: float, beta: float)``.
    """

    ###
    # Operations
    ###

    def clear(self):
        """Clears all variable sub-blocks."""
        self.bsm.clear()

    def zero(self):
        """Sets all sub-block data to zero."""
        self.bsm.zero()

    def matvec(self, x: wp.array, y: wp.array):
        """Performs the sparse matrix-vector product `y = A @ x`."""
        if self.Ax_op is None:
            raise RuntimeError("No `A@x` operator has been assigned.")
        self.Ax_op(self, x, y)

    def matvec_transpose(self, y: wp.array, x: wp.array):
        """Performs the sparse matrix-transpose-vector product `x = A^T @ y`."""
        if self.ATy_op is None:
            raise RuntimeError("No `A^T@y` operator has been assigned.")
        self.ATy_op(self, y, x)

    def gemv(self, x: wp.array, y: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """Performs a BLAS-like generalized sparse matrix-vector product `y = alpha * A @ x + beta * y`."""
        if self.gemv_op is None:
            raise RuntimeError("No BLAS-like `GEMV` operator has been assigned.")
        self.gemv_op(self, x, y, alpha, beta)

    def gemv_transpose(self, y: wp.array, x: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """Performs a BLAS-like generalized sparse matrix-transpose-vector product `x = alpha * A^T @ y + beta * x`."""
        if self.gemvt_op is None:
            raise RuntimeError("No BLAS-like transposed `GEMV` operator has been assigned.")
        self.gemvt_op(self, y, x, alpha, beta)
