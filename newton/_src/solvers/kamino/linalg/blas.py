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

"""BLAS-like operations for multi-linear systems"""

import functools
from typing import Any

import warp as wp

from ..core.types import float32, int32
from .sparse import BlockDType, BlockSparseMatrices

###
# Module interface
###

__all__ = [
    "block_sparse_gemv",
    "block_sparse_matvec",
    "block_sparse_transpose_gemv",
    "block_sparse_transpose_matvec",
    "dense_gemv",
    "diag_gemv",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


##
# Kernels
##


@wp.kernel
def _mult_left_right_diag_matrix_with_matrix(
    # Inputs:
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    D: wp.array(dtype=float32),
    X: wp.array(dtype=float32),
    # Outputs:
    Y: wp.array(dtype=float32),
):
    # Retrieve the thread indices
    wid, tid = wp.tid()

    # Retrieve the number of active dimensions in the world
    n = dim[wid]

    # Compute i (row) and j (col) indices from the tid
    i = tid // n
    j = tid % n

    # Skip if indices exceed the problem size
    if i >= n or j >= n:
        return

    # Retrieve the matrix index offset of the world
    m_0 = mio[wid]

    # Retrieve the vector index offset of the world
    v_0 = vio[wid]

    # Compute the global index of the matrix entry
    m_ij = m_0 + n * i + j

    # Retrieve the ij entry of the input matrix
    X_ij = X[m_ij]

    # Retrieve the i,j entries of the diagonal matrix
    D_i = D[v_0 + i]
    D_j = D[v_0 + j]

    # Compute the i,j entry of the output matrix
    Y[m_ij] = D_i * D_j * X_ij


@wp.kernel
def _mult_left_diag_matrix_with_vector(
    # Inputs:
    dim: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    D: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    # Outputs:
    y: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    n = dim[wid]

    # Skip if row index exceed the problem size
    if tid >= n:
        return

    # Retrieve the vector index offset of the world
    v_0 = vio[wid]

    # Compute the global index of the vector entry
    v_i = v_0 + tid

    # Retrieve the i-th entry of the input vector
    x_i = x[v_i]

    # Retrieve the i-th entry of the diagonal matrix
    D_i = D[v_i]

    # Compute the i-th entry of the output vector
    y[v_i] = D_i * x_i


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
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Vector:
        x: wp.array(dtype=block_type.dtype),
        y: wp.array(dtype=block_type.dtype),
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
            x_idx_base = col_start[mat_id] + block_coord[1]
            acc = block_type.dtype(0.0)

            for j in range(n_block_cols):
                acc += block[j] * x[x_idx_base + j]

            wp.atomic_add(y, row_start[mat_id] + block_coord[0], acc)

        else:
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_idx_base = row_start[mat_id] + block_coord[0]

            for i in range(n_block_rows):
                acc = block_type.dtype(0.0)

                for j in range(n_block_cols):
                    acc += block[i, j] * x[x_idx_base + j]

                wp.atomic_add(y, y_idx_base + i, acc)

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
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Vector:
        y: wp.array(dtype=block_type.dtype),
        x: wp.array(dtype=block_type.dtype),
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
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_val = y[row_start[mat_id] + block_coord[0]]

            for i in range(n_block_cols):
                wp.atomic_add(x, x_idx_base + i, block[i] * y_val)

        else:
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_idx_base = row_start[mat_id] + block_coord[0]

            for i in range(n_block_cols):
                acc = block_type.dtype(0.0)

                for j in range(n_block_rows):
                    acc += block[j, i] * y[y_idx_base + j]

                wp.atomic_add(x, x_idx_base + i, acc)

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
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Inputs:
        x: wp.array(dtype=Any),
        beta: Any,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, entry_id = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0 or entry_id >= matrix_dims[mat_id, sp_dim]:
            return

        if wp.static(space_dim == 0):
            idx = row_start[mat_id] + entry_id
            x[idx] = beta * x[idx]
        else:
            idx = col_start[mat_id] + entry_id
            x[idx] = beta * x[idx]

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
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Vector:
        x: wp.array(dtype=block_type.dtype),
        y: wp.array(dtype=block_type.dtype),
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
            x_idx_base = col_start[mat_id] + block_coord[1]
            acc = block_type.dtype(0.0)

            for j in range(n_block_cols):
                acc += alpha * block[j] * x[x_idx_base + j]

            wp.atomic_add(y, row_start[mat_id] + block_coord[0], acc)

        else:
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_idx_base = row_start[mat_id] + block_coord[0]

            for i in range(n_block_rows):
                acc = block_type.dtype(0.0)

                for j in range(n_block_cols):
                    acc += alpha * block[i, j] * x[x_idx_base + j]

                wp.atomic_add(y, y_idx_base + i, acc)

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
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Vector:
        y: wp.array(dtype=block_type.dtype),
        x: wp.array(dtype=block_type.dtype),
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
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_val = y[row_start[mat_id] + block_coord[0]]

            for i in range(n_block_cols):
                wp.atomic_add(x, x_idx_base + i, alpha * block[i] * y_val)

        else:
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_idx_base = row_start[mat_id] + block_coord[0]

            for i in range(n_block_cols):
                acc = block_type.dtype(0.0)

                for j in range(n_block_rows):
                    acc += alpha * block[j, i] * y[y_idx_base + j]

                wp.atomic_add(x, x_idx_base + i, acc)

    return block_sparse_transpose_gemv_kernel


@wp.kernel
def _diag_gemv_kernel(
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    D: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    world_active: wp.array(dtype=wp.bool),
    alpha: Any,
    beta: Any,
):
    """Computes y[w] = alpha * D[w] * x[w] + beta * y[w] for each world w."""
    world, row = wp.tid()
    assert world < len(active_dims)
    if not world_active[world] or row >= active_dims[world]:
        return

    zero = type(alpha)(0)
    s = y.dtype(0)

    if alpha != zero:
        s += alpha * D[world, row] * x[world, row]
    if beta != zero:
        s += beta * y[world, row]
    y[world, row] = s


@wp.kernel
def _dense_gemv_kernel(
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    A: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    world_active: wp.array(dtype=wp.bool),
    alpha: Any,
    beta: Any,
    matrix_stride: int,
    tile_size: int,
):
    """Computes y[w] = alpha * (A[w] @ x[w]) + beta * y[w] in-place for each world w."""
    world, row, lane = wp.tid()
    assert world < len(active_dims)
    dim = active_dims[world]
    if not world_active[world] or row >= dim:
        return

    row_stride = active_dims[world]
    zero = type(alpha)(0)
    s = zero
    if alpha != zero:
        for col in range(lane, dim, tile_size):
            s += A[world, row * row_stride + col] * x[world, col]
    row_tile = wp.tile_sum(wp.tile(s * alpha))
    if beta != zero:
        row_tile += beta * wp.tile_load(y[world], shape=1, offset=row)
    wp.tile_store(y[world], row_tile, offset=row)


##
# Launchers
##


def diag_gemv(
    D: wp.array2d,
    x: wp.array2d,
    y: wp.array2d,
    active_dims: wp.array,
    world_active: wp.array,
    alpha: float,
    beta: float,
):
    """
    Launch kernel for diagonal matrix gemv: y = alpha * D * x + beta * y

    Args:
        D: Diagonal matrices stored as 2D array (n_worlds, max_dim).
        x: Input vectors (n_worlds, max_dim).
        y: Output vectors (n_worlds, max_dim), modified in-place.
        active_dims: Active dimension per world.
        world_active: Boolean mask for active worlds.
        alpha: Scalar multiplier for D * x.
        beta: Scalar multiplier for y.
    """
    n_worlds, max_dim = x.shape
    dtype = x.dtype
    wp.launch(
        _diag_gemv_kernel,
        dim=(n_worlds, max_dim),
        inputs=[x, y, D, active_dims, world_active, dtype(alpha), dtype(beta)],
        device=x.device,
    )


def dense_gemv(
    A: wp.array2d,
    x: wp.array2d,
    y: wp.array2d,
    active_dims: wp.array,
    world_active: wp.array,
    alpha: float,
    beta: float,
    matrix_stride: int,
    block_dim: int = 64,
):
    """
    Launch kernel for dense matrix gemv: y = alpha * A @ x + beta * y

    Args:
        A: Dense matrices stored as 2D array (n_worlds, max_dim * max_dim).
        x: Input vectors (n_worlds, max_dim).
        y: Output vectors (n_worlds, max_dim), modified in-place.
        active_dims: Active dimension per world.
        world_active: Boolean mask for active worlds.
        alpha: Scalar multiplier for A * x.
        beta: Scalar multiplier for y.
        matrix_stride: Stride for matrix row indexing.
        block_dim: Block dimension for tiled computation.
    """
    n_worlds, max_dim = x.shape
    dtype = x.dtype
    tile_size = block_dim if x.device.is_cuda else 1
    wp.launch(
        _dense_gemv_kernel,
        dim=(n_worlds, max_dim, block_dim),
        inputs=[x, y, A, active_dims, world_active, dtype(alpha), dtype(beta), matrix_stride, tile_size],
        device=x.device,
        block_dim=tile_size if tile_size > 1 else 256,
    )


def block_sparse_matvec(
    A: BlockSparseMatrices,
    x: wp.array,
    y: wp.array,
    matrix_mask: wp.array,
):
    """
    Launch kernel for block-sparse matrix-vector product: y = A * x

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        x (wp.array): Input vector.
        y (wp.array): Output vector.
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
            A.row_start,
            A.col_start,
            x,
            y,
            matrix_mask,
        ],
        device=A.device,
    )


def block_sparse_transpose_matvec(
    A: BlockSparseMatrices,
    y: wp.array,
    x: wp.array,
    matrix_mask: wp.array,
):
    """
    Launch kernel for block-sparse transpose matrix-vector product: x = A^T * y

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        y (wp.array): Input vector.
        x (wp.array): Output vector.
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
            A.row_start,
            A.col_start,
            y,
            x,
            matrix_mask,
        ],
        device=A.device,
    )


def block_sparse_gemv(
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
        x (wp.array): Input vector for matrix-vector multiplication.
        y (wp.array): Input vector for linear offset and output vector.
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
            A.row_start,
            A.col_start,
            x,
            y,
            alpha,
            matrix_mask,
        ],
        device=A.device,
    )


def block_sparse_transpose_gemv(
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
        y (wp.array): Input vector for matrix-vector multiplication.
        x (wp.array): Input vector for linear offset and output vector.
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
            A.row_start,
            A.col_start,
            y,
            x,
            alpha,
            matrix_mask,
        ],
        device=A.device,
    )
