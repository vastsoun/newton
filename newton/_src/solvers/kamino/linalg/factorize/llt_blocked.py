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

"""KAMINO: Linear Algebra: Blocked LLT (i.e. Cholesky) factorization using Warp's Tile API."""

from __future__ import annotations

from functools import cache

import warp as wp
from warp.context import Devicelike

from ...core.types import float32, int32

###
# Module interface
###

__all__ = [
    "llt_blocked_factorize",
    "llt_blocked_solve",
    "llt_blocked_solve_inplace",
    "make_llt_blocked_factorize_kernel",
    "make_llt_blocked_solve_inplace_kernel",
    "make_llt_blocked_solve_kernel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@cache
def make_llt_blocked_factorize_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_factorize_kernel(
        # Inputs:
        maxdim: wp.array(dtype=int32),
        dim: wp.array(dtype=int32),
        mio: wp.array(dtype=int32),
        A: wp.array(dtype=float32),
        # Outputs:
        L: wp.array2d(dtype=float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        # Retrieve the matrix block dimensions and size
        maxdim = maxdim[tid]
        dim = dim[tid]
        mio = mio[tid]

        # Round up active_matrix_size to next multiple of block_size
        active_matrix_size = dim * dim
        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, n, block_size):
            end = k + block_size

            # Load current diagonal block A[k:end, k:end]
            # and update with contributions from previously computed blocks.
            A_kk_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(mio + k, k), storage="shared")
            # The following if pads the matrix if it is not divisible by block_size
            if k + block_size > active_matrix_size or k + block_size > active_matrix_size:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block

                for i in range(num_iterations):
                    linear_index = tid_block + i * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= active_matrix_size or k + col >= active_matrix_size:
                        value = wp.where(row == col, float32(1), float32(0))
                    A_kk_tile[row, col] = value

            if k > 0:
                for j in range(0, k, block_size):
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(mio + k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    L_L_T_block = wp.tile_matmul(L_block, L_block_T)
                    A_kk_tile -= L_L_T_block

            # Compute the Cholesky factorization for the block
            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L, L_kk_tile, offset=(mio + k, k))

            # Process the blocks below the current block
            for i in range(end, n, block_size):
                A_ik_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(mio + i, k), storage="shared")
                # The following if pads the matrix if it is not divisible by block_size
                if i + block_size > active_matrix_size or k + block_size > active_matrix_size:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block

                    for ii in range(num_iterations):
                        linear_index = tid_block + ii * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= active_matrix_size or k + col >= active_matrix_size:
                            value = wp.where(i + row == k + col, float32(1), float32(0))
                        A_ik_tile[row, col] = value

                if k > 0:
                    for j in range(0, k, block_size):
                        L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(mio + i, j))
                        L_2_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(mio + k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        L_L_T_tile = wp.tile_matmul(L_tile, L_T_tile)
                        A_ik_tile -= L_L_T_tile

                t = wp.tile_transpose(A_ik_tile)
                tmp = wp.tile_lower_solve(L_kk_tile, t)
                sol_tile = wp.tile_transpose(tmp)

                wp.tile_store(L, sol_tile, offset=(mio + i, k))

    # Return the kernel function
    return llt_blocked_factorize_kernel


@cache
def make_llt_blocked_solve_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_solve_kernel(
        # Inputs:
        dim_in: wp.array(dtype=int32),
        rio_in: wp.array(dtype=int32),
        L_in: wp.array2d(dtype=wp.float32),
        b_in: wp.array2d(dtype=wp.float32),
        # Outputs:
        y_out: wp.array2d(dtype=wp.float32),
        x_out: wp.array2d(dtype=wp.float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()

        # Retrieve the matrix block dimensions and size
        dim = dim_in[tid]
        rio = rio_in[tid]

        # Round up active_matrix_size to next multiple of block_size
        active_matrix_size = dim * dim
        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L y = b
        for i in range(0, n, block_size):
            i_end = i + block_size
            rhs_tile = wp.tile_load(b_in, shape=(block_size, 1), offset=(rio + i, 0))
            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L_in, shape=(block_size, block_size), offset=(rio + i, j))
                    y_block = wp.tile_load(y_out, shape=(block_size, 1), offset=(rio + j, 0))
                    Ly_block = wp.tile_matmul(L_block, y_block)
                    rhs_tile -= Ly_block
            L_tile = wp.tile_load(L_in, shape=(block_size, block_size), offset=(rio + i, i))
            y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
            wp.tile_store(y_out, y_tile, offset=(rio + i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(n - block_size, -1, -block_size):
            i_start = i
            i_end = i_start + block_size
            rhs_tile = wp.tile_load(y_out, shape=(block_size, 1), offset=(rio + i_start, 0))
            if i_end < n:
                for j in range(i_end, n, block_size):
                    L_tile = wp.tile_load(L_in, shape=(block_size, block_size), offset=(rio + j, i_start))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_out, shape=(block_size, 1), offset=(rio + j, 0))
                    L_T_x_tile = wp.tile_matmul(L_T_tile, x_tile)
                    rhs_tile -= L_T_x_tile
            L_tile = wp.tile_load(L_in, shape=(block_size, block_size), offset=(rio + i_start, i_start))
            x_tile = wp.tile_upper_solve(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x_out, x_tile, offset=(rio + i_start, 0))

    # Return the kernel function
    return llt_blocked_solve_kernel


@cache
def make_llt_blocked_solve_inplace_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_solve_inplace_kernel(
        # Inputs:
        dim_in: wp.array(dtype=int32),
        rio_in: wp.array(dtype=int32),
        L_in: wp.array2d(dtype=wp.float32),
        # Outputs:
        x_inout: wp.array2d(dtype=wp.float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()

        # Retrieve the matrix block dimensions and size
        dim = dim_in[tid]
        rio = rio_in[tid]
        wp.printf("tid=%d, rio=%d, dim=%d\n", tid, rio, dim)

        # Round up active_matrix_size to next multiple of block_size
        active_matrix_size = dim * dim
        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L y = b
        for i in range(0, n, block_size):
            i_end = i + block_size
            rhs_tile = wp.tile_load(x_inout, shape=(block_size, 1), offset=(rio + i, 0))
            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L_in, shape=(block_size, block_size), offset=(rio + i, j))
                    y_block = wp.tile_load(x_inout, shape=(block_size, 1), offset=(rio + j, 0))
                    Ly_block = wp.tile_matmul(L_block, y_block)
                    rhs_tile -= Ly_block
            L_tile = wp.tile_load(L_in, shape=(block_size, block_size), offset=(rio + i, i))
            y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
            wp.tile_store(x_inout, y_tile, offset=(rio + i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(n - block_size, -1, -block_size):
            i_start = i
            i_end = i_start + block_size
            rhs_tile = wp.tile_load(x_inout, shape=(block_size, 1), offset=(rio + i_start, 0))
            if i_end < n:
                for j in range(i_end, n, block_size):
                    L_tile = wp.tile_load(L_in, shape=(block_size, block_size), offset=(rio + j, i_start))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_inout, shape=(block_size, 1), offset=(rio + j, 0))
                    L_T_x_tile = wp.tile_matmul(L_T_tile, x_tile)
                    rhs_tile -= L_T_x_tile
            L_tile = wp.tile_load(L_in, shape=(block_size, block_size), offset=(rio + i_start, i_start))
            x_tile = wp.tile_upper_solve(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x_inout, x_tile, offset=(rio + i_start, 0))

    # Return the kernel function
    return llt_blocked_solve_inplace_kernel


###
# Launchers
###


def llt_blocked_factorize(
    kernel,
    maxdim: wp.array(dtype=int32),
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    A: wp.array(dtype=float32),
    L: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 128,  # TODO: Rename this to be clearer that this is the number of threads per TILE block and not matrix block
    device: Devicelike = None,
):
    """
    Launches the blocked Cholesky factorization kernel for a block partitioned matrix.

    Args:
        kernel: The kernel function to use for the blocked factorization.
        num_blocks (int): The number of matrix blocks to process.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
        maxdim (wp.array): An array of shape `(num_blocks,)` containing the maximum dimensions of each matrix block.
        dim (wp.array): An array of shape `(num_blocks,)` containing the active dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the matrix index offset (mio) of each matrix block.
        A (wp.array): The flat input array containing the input matrix blocks to be factorized.
        L (wp.array): The flat output array containing the factorization of each matrix block.
    """
    wp.launch_tiled(
        kernel=kernel, dim=num_blocks, inputs=[maxdim, dim, mio, A], outputs=[L], block_dim=block_dim, device=device
    )


def llt_blocked_solve(
    kernel,
    dim: wp.array(dtype=int32),
    rio: wp.array(dtype=int32),
    L: wp.array2d(dtype=float32),
    b: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 64,
    device: Devicelike = None,
):
    """
    Launches the blocked Cholesky solve kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        rio (wp.array): An array of shape `(num_blocks,)` containing the row index offsets of each matrix block.
        L (wp.array2d): The flat input array containing the Cholesky factorization of each matrix block.
        b (wp.array): The flat input array containing the stacked right-hand side vectors.
        y (wp.array): The output array where the intermediate result will be stored.
        x (wp.array): The output array where the solution to the linear system `A @ x = b` will be stored.
        kernel: The kernel function to use for the blocked solve.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
    """
    wp.launch_tiled(
        kernel=kernel, dim=num_blocks, inputs=[dim, rio, L, b], outputs=[y, x], block_dim=block_dim, device=device
    )


def llt_blocked_solve_inplace(
    kernel,
    dim: wp.array(dtype=int32),
    rio: wp.array(dtype=int32),
    L: wp.array2d(dtype=float32),
    x: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 64,
    device: Devicelike = None,
):
    """
    Launches the blocked Cholesky in-place solve kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        rio (wp.array): An array of shape `(num_blocks,)` containing the row index offsets of each matrix block.
        L (wp.array2d): The flat input array containing the Cholesky factorization of each matrix block.
        x (wp.array): The input/output array where the solution to the linear system `A @ x = b` will be stored in-place.
        kernel: The kernel function to use for the blocked in-place solve.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
    """
    wp.launch_tiled(kernel=kernel, dim=num_blocks, inputs=[dim, rio, L, x], block_dim=block_dim, device=device)
