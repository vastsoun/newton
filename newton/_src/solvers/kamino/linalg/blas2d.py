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

"""Sparse BLAS operations, 2d version where stacks of vectors are not flattened"""

import functools
from typing import Any

import warp as wp

from ..core.types import FloatType, int32
from .sparse_matrix import BlockDType, BlockSparseMatrices

###
# Module interface
###

__all__ = [
    "block_sparse_gemv_2d",
    "block_sparse_matvec_2d",
    "block_sparse_transpose_gemv_2d",
    "block_sparse_transpose_matvec_2d",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


##
# Kernels
##


@functools.cache
def _make_block_sparse_matvec_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_matvec_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector:
        x: wp.array2d(dtype=block_type.dtype),
        y: wp.array2d(dtype=block_type.dtype),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: y_block += A_block @ x_block
        if wp.static(n_block_rows == 1):
            x_idx_base = block_coord[1]
            acc = block_type.dtype(0.0)

            for j in range(n_block_cols):
                acc += block[j] * x[mat_id, x_idx_base + j]

            wp.atomic_add(y, mat_id, block_coord[0], acc)

        else:
            x_idx_base = block_coord[1]
            y_idx_base = block_coord[0]

            for i in range(n_block_rows):
                acc = block_type.dtype(0.0)

                for j in range(n_block_cols):
                    acc += block[i, j] * x[mat_id, x_idx_base + j]

                wp.atomic_add(y, mat_id, y_idx_base + i, acc)

    return block_sparse_matvec_kernel


@functools.cache
def _make_block_sparse_transpose_matvec_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_transpose_matvec_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector:
        y: wp.array2d(dtype=block_type.dtype),
        x: wp.array2d(dtype=block_type.dtype),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: x_block += A_block^T @ y_block
        if wp.static(n_block_rows == 1):
            x_idx_base = block_coord[1]
            y_val = y[mat_id, block_coord[0]]

            for i in range(n_block_cols):
                wp.atomic_add(x, mat_id, x_idx_base + i, block[i] * y_val)

        else:
            x_idx_base = block_coord[1]
            y_idx_base = block_coord[0]

            for i in range(n_block_cols):
                acc = block_type.dtype(0.0)

                for j in range(n_block_rows):
                    acc += block[j, i] * y[mat_id, y_idx_base + j]

                wp.atomic_add(x, mat_id, x_idx_base + i, acc)

    return block_sparse_transpose_matvec_kernel


@functools.cache
def _make_scale_vector_kernel(space_dim: int):
    """Creates a kernel that scales a vector, taking into account a matrix mask and how the current
    size of a matrix affects the active entries of the vector.

    Parameters
    ----------
    space_dim : int
        Space of the vector in reference to the matrices (0: row space, 1: column space).
    """

    sp_dim = wp.constant(space_dim)

    @wp.kernel
    def scale_vector_kernel(
        # Matrix data:
        matrix_dims: wp.array2d(dtype=int32),
        # Inputs:
        x: wp.array2d(dtype=Any),
        beta: Any,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, entry_id = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0 or entry_id >= matrix_dims[mat_id, sp_dim]:
            return

        x[mat_id, entry_id] = beta * x[mat_id, entry_id]

    return scale_vector_kernel


@functools.cache
def _make_block_sparse_gemv_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_gemv_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector:
        x: wp.array2d(dtype=block_type.dtype),
        y: wp.array2d(dtype=block_type.dtype),
        # Scaling:
        alpha: block_type.dtype,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: z += alpha * A_block @ x_block
        if wp.static(n_block_rows == 1):
            x_idx_base = block_coord[1]
            acc = block_type.dtype(0.0)

            for j in range(n_block_cols):
                acc += alpha * block[j] * x[mat_id, x_idx_base + j]

            wp.atomic_add(y, mat_id, block_coord[0], acc)

        else:
            x_idx_base = block_coord[1]
            y_idx_base = block_coord[0]

            for i in range(n_block_rows):
                acc = block_type.dtype(0.0)

                for j in range(n_block_cols):
                    acc += alpha * block[i, j] * x[mat_id, x_idx_base + j]

                wp.atomic_add(y, mat_id, y_idx_base + i, acc)

    return block_sparse_gemv_kernel


@functools.cache
def _make_block_sparse_transpose_gemv_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_transpose_gemv_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector:
        y: wp.array2d(dtype=block_type.dtype),
        x: wp.array2d(dtype=block_type.dtype),
        # Scaling:
        alpha: block_type.dtype,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: z += alpha * A_block^T @ y_block
        if wp.static(n_block_rows == 1):
            x_idx_base = block_coord[1]
            y_val = y[mat_id, block_coord[0]]

            for i in range(n_block_cols):
                wp.atomic_add(x, mat_id, x_idx_base + i, alpha * block[i] * y_val)

        else:
            x_idx_base = block_coord[1]
            y_idx_base = block_coord[0]

            for i in range(n_block_cols):
                acc = block_type.dtype(0.0)

                for j in range(n_block_rows):
                    acc += alpha * block[j, i] * y[mat_id, y_idx_base + j]

                wp.atomic_add(x, mat_id, x_idx_base + i, acc)

    return block_sparse_transpose_gemv_kernel


@functools.cache
def _make_block_sparse_ATA_diagonal_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_ATA_diagonal_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Output:
        diag: wp.array2d(dtype=block_type.dtype),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        """
        For a block sparse matrix (stack) A, computes the diagonal of A^T * A
        """
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_col = nzb_coords[global_block_idx][1]
        block = nzb_values[global_block_idx]

        # Accumulate coefficients contributed by non-zero block
        if wp.static(n_block_rows == 1):
            for j in range(n_block_cols):
                val = block[j]
                wp.atomic_add(diag, mat_id, block_col + j, val * val)
        else:
            for j in range(n_block_cols):
                acc = block_type.dtype(0.0)
                for i in range(n_block_rows):
                    val = block[i, j]
                    acc += val * val
                wp.atomic_add(diag, mat_id, block_col + j, acc)

    return block_sparse_ATA_diagonal_kernel


@functools.cache
def _make_cwise_inverse_kernel(dtype: FloatType):
    @wp.kernel
    def cwise_inverse_kernel(
        # Inputs
        x: wp.array2d(dtype=dtype),
        dim: wp.array(dtype=wp.int32),
        mask: wp.array(dtype=wp.int32),
    ):
        mat_id, coeff_id = wp.tid()

        if mat_id >= mask.shape[0] or mask[mat_id] == 0 or coeff_id >= dim[mat_id]:
            return

        x[mat_id, coeff_id] = 1.0 / x[mat_id, coeff_id]

    return cwise_inverse_kernel


##
# Launchers
##


def block_sparse_matvec_2d(
    A: BlockSparseMatrices,
    x: wp.array,
    y: wp.array,
    matrix_mask: wp.array,
):
    """
    Launch kernel for block-sparse matrix-vector product: y = A * x

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        x (wp.array): Stack of input vectors, expects shape (num_matrices, max_of_max_cols).
        y (wp.array): Stack of output vectors, expects shape (num_matrices, max_of_max_rows).
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    y.zero_()

    wp.launch(
        kernel=_make_block_sparse_matvec_kernel(A.nzb_dtype),
        dim=(A.num_matrices, A.max_of_num_nzb),
        inputs=[
            A.num_nzb,
            A.nzb_start,
            A.nzb_coords,
            A.nzb_values,
            x,
            y,
            matrix_mask,
        ],
        device=A.device,
    )


def block_sparse_transpose_matvec_2d(
    A: BlockSparseMatrices,
    y: wp.array,
    x: wp.array,
    matrix_mask: wp.array,
):
    """
    Launch kernel for block-sparse transpose matrix-vector product: x = A^T * y

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        y (wp.array): Stack of input vectors, expects shape (num_matrices, max_of_max_rows).
        x (wp.array): Stack of output vectors, expects shape (num_matrices, max_of_max_cols).
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    x.zero_()

    wp.launch(
        kernel=_make_block_sparse_transpose_matvec_kernel(A.nzb_dtype),
        dim=(A.num_matrices, A.max_of_num_nzb),
        inputs=[
            A.num_nzb,
            A.nzb_start,
            A.nzb_coords,
            A.nzb_values,
            y,
            x,
            matrix_mask,
        ],
        device=A.device,
    )


def block_sparse_gemv_2d(
    A: BlockSparseMatrices,
    x: wp.array,
    y: wp.array,
    alpha: Any,
    beta: Any,
    matrix_mask: wp.array,
):
    """
    Launch kernel for generalized block-sparse matrix-vector product: y = alpha * (A * x) + beta * y

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        x (wp.array): Stack of input vectors, expects shape (num_matrices, max_of_max_cols).
        y (wp.array): Stack of input-output vectors, expects shape (num_matrices, max_of_max_rows).
        alpha (Any): Input scaling for matrix-vector multiplication.
        beta (Any): Input scaling for linear offset.
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    # Compute y <= beta * y
    wp.launch(
        kernel=_make_scale_vector_kernel(0),
        dim=(A.num_matrices, A.max_of_max_dims[0]),
        inputs=[A.dims, A.row_start, A.col_start, y, beta, matrix_mask],
        device=A.device,
    )

    # Compute y += alpha * A @ x
    wp.launch(
        kernel=_make_block_sparse_gemv_kernel(A.nzb_dtype),
        dim=(A.num_matrices, A.max_of_num_nzb),
        inputs=[
            A.num_nzb,
            A.nzb_start,
            A.nzb_coords,
            A.nzb_values,
            x,
            y,
            alpha,
            matrix_mask,
        ],
        device=A.device,
    )


def block_sparse_transpose_gemv_2d(
    A: BlockSparseMatrices,
    y: wp.array,
    x: wp.array,
    alpha: Any,
    beta: Any,
    matrix_mask: wp.array,
):
    """
    Launch kernel for generalized block-sparse transpose matrix-vector product: x = alpha * (A^T * y) + beta * x

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        y (wp.array): Stack of input vectors, expects shape (num_matrices, max_of_max_rows).
        x (wp.array): Stack of input-output vectors, expects shape (num_matrices, max_of_max_cols).
        alpha (Any): Input scaling for matrix-vector multiplication.
        beta (Any): Input scaling for linear offset.
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    # Compute x <= beta * x
    wp.launch(
        kernel=_make_scale_vector_kernel(1),
        dim=(A.num_matrices, A.max_of_max_dims[1]),
        inputs=[A.dims, A.row_start, A.col_start, x, beta, matrix_mask],
        device=A.device,
    )

    # Compute y += alpha * A^T @ y
    wp.launch(
        kernel=_make_block_sparse_transpose_gemv_kernel(A.nzb_dtype),
        dim=(A.num_matrices, A.max_of_num_nzb),
        inputs=[
            A.num_nzb,
            A.nzb_start,
            A.nzb_coords,
            A.nzb_values,
            y,
            x,
            alpha,
            matrix_mask,
        ],
        device=A.device,
    )


def block_sparse_ATA_inv_diagonal_2d(A: BlockSparseMatrices, inv_diag: wp.array, matrix_mask: wp.array):
    """
    Function computing the inverse of the diagonal of A^T * A given sparse matrix (stack) A.

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        inv_diag (wp.array): Stack of output vectors, expects shape (num_matrices, max_of_max_cols).
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    inv_diag.zero_()
    wp.launch(
        kernel=_make_block_sparse_ATA_diagonal_kernel(A.nzb_dtype),
        dim=(A.num_matrices, A.max_of_num_nzb),
        inputs=[
            A.num_nzb,
            A.nzb_start,
            A.nzb_coords,
            A.nzb_values,
            inv_diag,
            matrix_mask,
        ],
        device=A.device,
    )
    int_size_bytes = 4  # Size of wp.int32 in bytes
    cols = wp.array(
        dtype=wp.int32,
        shape=(A.num_matrices,),
        ptr=A.dims.ptr + int_size_bytes,
        strides=(2 * int_size_bytes,),
        copy=False,
    )
    wp.launch(
        kernel=_make_cwise_inverse_kernel(A.nzb_dtype.dtype),
        dim=(A.num_matrices, A.max_of_max_dims[1]),
        inputs=[
            inv_diag,
            cols,
            matrix_mask,
        ],
        device=A.device,
    )
