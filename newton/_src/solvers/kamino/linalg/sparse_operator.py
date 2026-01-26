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

import warp as wp

from .blas import (
    block_sparse_gemv,
    block_sparse_matvec,
    block_sparse_transpose_gemv,
    block_sparse_transpose_matvec,
)
from .blas2d import (
    block_sparse_gemv_2d,
    block_sparse_matvec_2d,
    block_sparse_transpose_gemv_2d,
    block_sparse_transpose_matvec_2d,
)
from .sparse_matrix import BlockSparseMatrices

###
# Module interface
###

__all__ = [
    "BlockSparseLinearOperators",
]


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

    ###
    # Operators
    ###

    Ax_op: Callable | None = None
    """
    The operator function for performing sparse matrix-vector products `y = A @ x`.\n
    Example signature: ``Ax_op(A: BlockSparseLinearMatrices, x: wp.array, y: wp.array, matrix_mask: wp.array)``.
    """

    ATy_op: Callable | None = None
    """
    The operator function for performing sparse matrix-transpose-vector products `x = A^T @ y`.\n
    Example signature: ``ATy_op(A: BlockSparseLinearMatrices, y: wp.array, x: wp.array, matrix_mask: wp.array)``.
    """

    gemv_op: Callable | None = None
    """
    The operator function for performing generalized sparse matrix-vector products `y = alpha * A @ x + beta * y`.\n
    Example signature: ``gemv_op(A: BlockSparseLinearMatrices, x: wp.array, y: wp.array, alpha: float, beta: float, matrix_mask: wp.array)``.
    """

    gemvt_op: Callable | None = None
    """
    The operator function for performing generalized sparse matrix-transpose-vector products `x = alpha * A^T @ y + beta * x`.\n
    Example signature: ``gemvt_op(A: BlockSparseLinearMatrices, y: wp.array, x: wp.array, alpha: float, beta: float, matrix_mask: wp.array)``.
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

    def initialize_default_operators(self, flat=True):
        """Sets all operator functions to their default implementations.
        Uses versions expecting flattened stacks of input and output vectors if flat is True;
        otherwise uses versions expecting 2d arrays for stacks of vectors."""
        if flat:
            self.Ax_op = block_sparse_matvec
            self.ATy_op = block_sparse_transpose_matvec
            self.gemv_op = block_sparse_gemv
            self.gemvt_op = block_sparse_transpose_gemv
        else:
            self.Ax_op = block_sparse_matvec_2d
            self.ATy_op = block_sparse_transpose_matvec_2d
            self.gemv_op = block_sparse_gemv_2d
            self.gemvt_op = block_sparse_transpose_gemv_2d

    def matvec(self, x: wp.array, y: wp.array, matrix_mask: wp.array):
        """Performs the sparse matrix-vector product `y = A @ x`."""
        if self.Ax_op is None:
            raise RuntimeError("No `A@x` operator has been assigned.")
        self.Ax_op(self.bsm, x, y, matrix_mask)

    def matvec_transpose(self, y: wp.array, x: wp.array, matrix_mask: wp.array):
        """Performs the sparse matrix-transpose-vector product `x = A^T @ y`."""
        if self.ATy_op is None:
            raise RuntimeError("No `A^T@y` operator has been assigned.")
        self.ATy_op(self.bsm, y, x, matrix_mask)

    def gemv(self, x: wp.array, y: wp.array, matrix_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """Performs a BLAS-like generalized sparse matrix-vector product `y = alpha * A @ x + beta * y`."""
        if self.gemv_op is None:
            raise RuntimeError("No BLAS-like `GEMV` operator has been assigned.")
        self.gemv_op(self.bsm, x, y, alpha, beta, matrix_mask)

    def gemv_transpose(self, y: wp.array, x: wp.array, matrix_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """Performs a BLAS-like generalized sparse matrix-transpose-vector product `x = alpha * A^T @ y + beta * x`."""
        if self.gemvt_op is None:
            raise RuntimeError("No BLAS-like transposed `GEMV` operator has been assigned.")
        self.gemvt_op(self.bsm, y, x, alpha, beta, matrix_mask)
