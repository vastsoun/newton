###########################################################################
# KAMINO: Cholesky factorization module
###########################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import Union

import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.math import FLOAT32_EPS
from newton._src.solvers.kamino.core.types import float32, int32

###
# Module interface
###

__all__ = [
    "BlockedCholeskyFactorizer",
    "CholeskyData",
    "CholeskyFactorizer",
    "CholeskyFactorizerBase",
    "SequentialCholeskyFactorizer",
    "cholesky_blocked_factorize",
    "cholesky_blocked_solve",
    "cholesky_blocked_solve_inplace",
    "cholesky_sequential_factorize",
    "cholesky_sequential_solve",
    "cholesky_sequential_solve_backward",
    "cholesky_sequential_solve_forward",
    "cholesky_sequential_solve_inplace",
    "make_cholesky_blocked_factorize_kernel",
    "make_cholesky_blocked_solve_inplace_kernel",
    "make_cholesky_blocked_solve_kernel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _cholesky_sequential_factorize(
    # Inputs:
    maxdim_in: wp.array(dtype=int32),
    dim_in: wp.array(dtype=int32),
    mio_in: wp.array(dtype=int32),
    A_in: wp.array(dtype=float32),
    # Outputs:
    L_out: wp.array(dtype=float32),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Define minimum value for diagonal elements
    min_L_ii = wp.static(1000.0 * FLOAT32_EPS)

    # Retrieve the matrix start offset and dimension
    mio = mio_in[tid]
    maxn = maxdim_in[tid]
    n = dim_in[tid]

    # Compute the Cholesky factorization sequentially
    for i in range(n):
        # Compute diagonal element L[i, i]
        m_i = mio + maxn * i
        m_ii = m_i + i
        sum = A_in[m_ii]
        for k in range(i):
            L_ik = L_out[m_i + k]
            sum -= L_ik * L_ik
        sum = wp.max(sum, min_L_ii)
        L_ii = wp.sqrt(sum)
        L_out[m_ii] = L_ii

        # Compute off-diagonal elements in column i
        for j in range(i + 1, n):
            m_j = mio + maxn * j
            m_ji = m_j + i
            sum = A_in[m_ji]
            for k in range(i):
                m_jk = m_j + k
                m_ik = m_i + k
                sum -= L_out[m_jk] * L_out[m_ik]
            L_out[m_ji] = sum / L_ii

    # for i in range(n):
    #     m_i = mio + maxn * i
    #     m_ii = m_i + i
    #     for j in range(i + 1):
    #         m_j = mio + maxn * j
    #         m_jj = m_j + j
    #         m_ij = m_i + j
    #         sum = float32(0.0)
    #         for k in range(j):
    #             m_ik = m_i + k
    #             m_jk = m_j + k
    #             sum += L_out[m_ik] * L_out[m_jk]
    #         if i == j:
    #             val = A_in[m_ii] - sum
    #             L_out[m_ij] = wp.sqrt(val)
    #         else:
    #             L_out[m_ij] = (A_in[m_ij] - sum) / L_out[m_jj]


@wp.kernel
def _cholesky_sequential_solve_forward(
    # Inputs:
    maxdim_in: wp.array(dtype=int32),
    dim_in: wp.array(dtype=int32),
    mio_in: wp.array(dtype=int32),
    vio_in: wp.array(dtype=int32),
    L_in: wp.array(dtype=float32),
    b_in: wp.array(dtype=float32),
    # Outputs:
    y_out: wp.array(dtype=float32),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the start offsets and problem dimension
    mio = mio_in[tid]
    vio = vio_in[tid]
    maxn = maxdim_in[tid]
    n = dim_in[tid]

    # Forward substitution to solve L * y = b
    for i in range(n):
        m_i = mio + maxn * i
        m_ii = m_i + i
        L_ii = L_in[m_ii]
        sum_i = b_in[vio + i]
        for j in range(i):
            m_ij = m_i + j
            sum_i -= L_in[m_ij] * y_out[vio + j]
        y_out[vio + i] = sum_i / L_ii


@wp.kernel
def _cholesky_sequential_solve_backward(
    # Inputs:
    maxdim_in: wp.array(dtype=int32),
    dim_in: wp.array(dtype=int32),
    mio_in: wp.array(dtype=int32),
    vio_in: wp.array(dtype=int32),
    L_in: wp.array(dtype=float32),
    y_in: wp.array(dtype=float32),
    # Outputs:
    x_out: wp.array(dtype=float32),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the start offsets and problem dimension
    mio = mio_in[tid]
    vio = vio_in[tid]
    maxn = maxdim_in[tid]
    n = dim_in[tid]

    # Backward substitution to solve L^T * x = y
    for i in range(n - 1, -1, -1):
        m_i = mio + maxn * i
        m_ii = m_i + i
        LT_ii = L_in[m_ii]
        sum_i = y_in[vio + i]
        for j in range(i + 1, n):
            m_ji = mio + maxn * j + i
            sum_i -= L_in[m_ji] * x_out[vio + j]
        x_out[vio + i] = sum_i / LT_ii


@wp.kernel
def _cholesky_sequential_solve(
    # Inputs:
    maxdim_in: wp.array(dtype=int32),
    dim_in: wp.array(dtype=int32),
    mio_in: wp.array(dtype=int32),
    vio_in: wp.array(dtype=int32),
    L_in: wp.array(dtype=float32),
    b_in: wp.array(dtype=float32),
    # Outputs:
    y_out: wp.array(dtype=float32),
    x_out: wp.array(dtype=float32),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the start offsets and problem dimension
    mio = mio_in[tid]
    vio = vio_in[tid]
    maxn = maxdim_in[tid]
    n = dim_in[tid]

    # Forward substitution to solve L * y = b
    for i in range(n):
        m_i = mio + maxn * i
        m_ii = m_i + i
        L_ii = L_in[m_ii]
        sum_i = b_in[vio + i]
        for j in range(i):
            m_ij = m_i + j
            sum_i -= L_in[m_ij] * y_out[vio + j]
        y_out[vio + i] = sum_i / L_ii

    # Backward substitution to solve L^T * x = y
    for i in range(n - 1, -1, -1):
        m_i = mio + maxn * i
        m_ii = m_i + i
        LT_ii = L_in[m_ii]
        sum_i = y_out[vio + i]
        for j in range(i + 1, n):
            m_ji = mio + maxn * j + i
            sum_i -= L_in[m_ji] * x_out[vio + j]
        x_out[vio + i] = sum_i / LT_ii


@wp.kernel
def _cholesky_sequential_solve_inplace(
    # Inputs:
    maxdim_in: wp.array(dtype=int32),
    dim_in: wp.array(dtype=int32),
    mio_in: wp.array(dtype=int32),
    vio_in: wp.array(dtype=int32),
    L_in: wp.array(dtype=float32),
    x_inout: wp.array(dtype=float32),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the start offsets and problem dimension
    mio = mio_in[tid]
    vio = vio_in[tid]
    maxn = maxdim_in[tid]
    n = dim_in[tid]

    # Forward substitution to solve L * y = b
    for i in range(n):
        m_i = mio + maxn * i
        m_ii = m_i + i
        L_ii = L_in[m_ii]
        sum_i = x_inout[vio + i]
        for j in range(i):
            m_ij = m_i + j
            sum_i -= L_in[m_ij] * x_inout[vio + j]
        x_inout[vio + i] = sum_i / L_ii

    # Backward substitution to solve L^T * x = y
    for i in range(n - 1, -1, -1):
        m_i = mio + maxn * i
        m_ii = m_i + i
        LT_ii = L_in[m_ii]
        sum_i = x_inout[vio + i]
        for j in range(i + 1, n):
            m_ji = mio + maxn * j + i
            sum_i -= L_in[m_ji] * x_inout[vio + j]
        x_inout[vio + i] = sum_i / LT_ii


@cache
def make_cholesky_blocked_factorize_kernel(block_size: int):
    @wp.kernel
    def cholesky_blocked_factorize_kernel(
        # Inputs:
        dim_in: wp.array(dtype=int32),
        rio_in: wp.array(dtype=int32),
        A_in: wp.array2d(dtype=float32),
        # Outputs:
        L_out: wp.array2d(dtype=float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        # Retrieve the matrix block dimensions and size
        dim = dim_in[tid]
        rio = rio_in[tid]

        # Round up active_matrix_size to next multiple of block_size
        active_matrix_size = dim * dim
        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, n, block_size):
            end = k + block_size

            # Load current diagonal block A[k:end, k:end]
            # and update with contributions from previously computed blocks.
            A_kk_tile = wp.tile_load(A_in, shape=(block_size, block_size), offset=(rio + k, k), storage="shared")
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
                    L_block = wp.tile_load(L_out, shape=(block_size, block_size), offset=(rio + k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    L_L_T_block = wp.tile_matmul(L_block, L_block_T)
                    A_kk_tile -= L_L_T_block

            # Compute the Cholesky factorization for the block
            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L_out, L_kk_tile, offset=(rio + k, k))

            # Process the blocks below the current block
            for i in range(end, n, block_size):
                A_ik_tile = wp.tile_load(A_in, shape=(block_size, block_size), offset=(rio + i, k), storage="shared")
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
                        L_tile = wp.tile_load(L_out, shape=(block_size, block_size), offset=(rio + i, j))
                        L_2_tile = wp.tile_load(L_out, shape=(block_size, block_size), offset=(rio + k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        L_L_T_tile = wp.tile_matmul(L_tile, L_T_tile)
                        A_ik_tile -= L_L_T_tile

                t = wp.tile_transpose(A_ik_tile)
                tmp = wp.tile_lower_solve(L_kk_tile, t)
                sol_tile = wp.tile_transpose(tmp)

                wp.tile_store(L_out, sol_tile, offset=(rio + i, k))

    # Return the kernel function
    return cholesky_blocked_factorize_kernel


@cache
def make_cholesky_blocked_solve_kernel(block_size: int):
    @wp.kernel
    def cholesky_blocked_solve_kernel(
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
        num_threads_per_block = wp.block_dim()

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
    return cholesky_blocked_solve_kernel


@cache
def make_cholesky_blocked_solve_inplace_kernel(block_size: int):
    @wp.kernel
    def cholesky_blocked_solve_inplace_kernel(
        # Inputs:
        dim_in: wp.array(dtype=int32),
        rio_in: wp.array(dtype=int32),
        L_in: wp.array2d(dtype=wp.float32),
        # Outputs:
        x_inout: wp.array2d(dtype=wp.float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

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
    return cholesky_blocked_solve_inplace_kernel


###
# Launchers
###


def cholesky_sequential_factorize(
    num_blocks: int,
    maxdim: wp.array(dtype=int32),
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    A: wp.array(dtype=float32),
    L: wp.array(dtype=float32),
):
    """
    Launches the sequential Cholesky factorization kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each matrix block.
        A (wp.array): The flat input array containing the input matrix blocks to be factorized.
        L (wp.array): The flat output array containing the Cholesky factorization of each matrix block.
    """
    wp.launch(kernel=_cholesky_sequential_factorize, dim=num_blocks, inputs=[maxdim, dim, mio, A, L])


def cholesky_sequential_solve_forward(
    num_blocks: int,
    maxdim: wp.array(dtype=int32),
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    b: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
):
    """
    Launches the sequential forward solve kernel using the Cholesky factorization of a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each vector block.
        L (wp.array): The flat input array containing the Cholesky factorization of each matrix block.
        b (wp.array): The flat input array containing the stacked right-hand side vectors.
        y (wp.array): The output array where the intermediate result will be stored.
    """
    wp.launch(kernel=_cholesky_sequential_solve_forward, dim=num_blocks, inputs=[maxdim, dim, mio, vio, L, b, y])


def cholesky_sequential_solve_backward(
    num_blocks: int,
    maxdim: wp.array(dtype=int32),
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
):
    """
    Launches the sequential backward solve kernel using the Cholesky factorization of a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each vector block.
        L (wp.array): The flat input array containing the Cholesky factorization of each matrix block.
        y (wp.array): The flat input array containing the intermediate result from the forward solve.
        x (wp.array): The output array where the solution to the linear system `A @ x = b` will be stored.
    """
    wp.launch(kernel=_cholesky_sequential_solve_backward, dim=num_blocks, inputs=[maxdim, dim, mio, vio, L, y, x])


def cholesky_sequential_solve(
    num_blocks: int,
    maxdim: wp.array(dtype=int32),
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    b: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
):
    """ "
    Launches the sequential solve kernel using the Cholesky factorization of a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each vector block.
        L (wp.array): The flat input array containing the Cholesky factorization of each matrix block.
        b (wp.array): The flat input array containing the stacked right-hand side vectors.
        y (wp.array): The output array where the intermediate result will be stored.
        x (wp.array): The output array where the solution to the linear system `A @ x = b` will be stored.
    """
    wp.launch(kernel=_cholesky_sequential_solve, dim=num_blocks, inputs=[maxdim, dim, mio, vio, L, b, y, x])


def cholesky_sequential_solve_inplace(
    num_blocks: int,
    maxdim: wp.array(dtype=int32),
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
):
    """
    Launches the sequential in-place solve kernel using the Cholesky factorization of a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each vector block.
        L (wp.array): The flat input array containing the Cholesky factorization of each matrix block.
        x (wp.array): The input/output array where the solution to the linear system `A @ x = b` will be stored in-place.
    """
    wp.launch(
        kernel=_cholesky_sequential_solve_inplace,
        dim=num_blocks,
        inputs=[maxdim, dim, mio, vio, L, x],
    )


def cholesky_blocked_factorize(
    num_blocks: int,
    dim: wp.array(dtype=int32),
    rio: wp.array(dtype=int32),
    A: wp.array2d(dtype=float32),
    L: wp.array2d(dtype=float32),
    kernel,
    block_dim: int = 128,
):
    """
    Launches the blocked Cholesky factorization kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        rio (wp.array): An array of shape `(num_blocks,)` containing the row index offsets of each matrix block.
        A (wp.array2d): The flat input array containing the input matrix blocks to be factorized.
        L (wp.array2d): The flat output array containing the Cholesky factorization of each matrix block.
        kernel: The kernel function to use for the blocked factorization.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
    """
    wp.launch_tiled(kernel=kernel, dim=num_blocks, inputs=[dim, rio, A], outputs=[L], block_dim=block_dim)


def cholesky_blocked_solve(
    num_blocks: int,
    dim: wp.array(dtype=int32),
    rio: wp.array(dtype=int32),
    L: wp.array2d(dtype=float32),
    b: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    kernel,
    block_dim: int = 64,
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
    wp.launch_tiled(kernel=kernel, dim=num_blocks, inputs=[dim, rio, L, b], outputs=[y, x], block_dim=block_dim)


def cholesky_blocked_solve_inplace(
    num_blocks: int,
    dim: wp.array(dtype=int32),
    rio: wp.array(dtype=int32),
    L: wp.array2d(dtype=float32),
    x: wp.array(dtype=float32),
    kernel,
    block_dim: int = 64,
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
    wp.launch_tiled(kernel=kernel, dim=num_blocks, inputs=[dim, rio, L, x], block_dim=block_dim)


###
# Containers
###


class CholeskyData:
    """
    A container to hold the data of multiple Cholesky factorization matrix blocks.
    """

    def __init__(self):
        self.num_blocks: int = 0
        """Host-side cache of the number of matrix blocks in the Cholesky factorization."""

        self.dimensions: list[int] = []
        """Host-side cache of the dimensions of each symmetric positive-definite matrix block."""

        self.msize: int = 0
        """Host-side cache of the total size of the flat matrix data array, which is the sum of the sizes of all matrix blocks."""

        self.vsize: int = 0
        """Host-side cache of the total size of the flat vector data array, which is the sum of the sizes of all vector blocks."""

        self.maxdim: wp.array(dtype=int32) | None = None
        """The maximum dimensions of each symmetric positive-definite matrix block. Shape of ``(num_blocks,)`` and type :class:`int32`."""

        self.dim: wp.array(dtype=int32) | None = None
        """The dimensions of each symmetric positive-definite matrix block. Shape of ``(num_blocks,)`` and type :class:`int32`."""

        self.mio: wp.array(dtype=int32) | None = None
        """The matrix index offset (mio) of each block in the flat data array. Shape of ``(num_blocks,)`` and type :class:`int32`."""

        self.vio: wp.array(dtype=int32) | None = None
        """The vector index offset (vio) of each block in the flat data array. Shape of ``(num_blocks,)`` and type :class:`int32`."""

        self.L: wp.array(dtype=float32) | None = None
        """The flat array containing the Cholesky factorizations of each matrix block. Shape of ``(total_size,)`` and type :class:`float32`."""

        self.y: wp.array(dtype=float32) | None = None
        """Buffer for the intermediate result of the Cholesky forward/backward solve. Shape of ``(total_size, 1)`` and type :class:`float32`."""


###
# Factorizers
###


class CholeskyFactorizerBase(ABC):
    """
    The base class to manage Cholesky factorizations of one or several matrix blocks.\n
    Each matrix block is assumed to contain symmetric positive-definite of arbitrary dimensions.\n
    """

    @staticmethod
    def _check_dims(dims: list[int]) -> list[int]:
        if isinstance(dims, int):
            dims = [dims]
        elif isinstance(dims, list):
            if len(dims) > 0 and not all(isinstance(d, int) for d in dims):
                raise ValueError("All dimensions must be integers.")
        else:
            raise TypeError("Dimensions must be an integer or a list of integers.")
        return dims

    def __init__(self, dims: list[int] | None = None, allocate_info=True, device: Devicelike = None):
        """
        Creates a new Cholesky factorization container.\n

        This class supports both RAII and manual memory management.\n
        If the `dims` argument is not empty, it allocates the Cholesky\n
        factorization data on the specified device, otherwise deferred\n
        allocation is permitted by calling the `allocate()` method.\n

        Args
        ----
            dims (List[int]): A list of dimensions for the symmetric positive-definite matrix blocks.
            device (Devicelike): The device on which to allocate the Cholesky factorization data.

        Raises
        ------
            ValueError: If the dimensions are not valid (e.g., empty list).
            TypeError: If the dimensions are not integers or a list of integers.
        """
        # Initialize the device identifier cache
        self._device: Devicelike = None

        # Initialize the Delassus state data container
        self._data: CholeskyData = CholeskyData()

        # Allocate the Delassus operator data if dims are provided
        if dims is not None:
            self.allocate(dims=dims, allocate_info=allocate_info, device=device)

    @property
    def num_blocks(self) -> int:
        """
        Returns the number of matrix blocks in the Cholesky factorization.
        """
        return self._data.num_blocks

    @property
    def dimensions(self) -> list[int]:
        """
        Returns the total capacity of the Cholesky factorization memory allocation.
        """
        return self._data.dimensions

    @property
    def msize(self) -> int:
        """
        Returns the total size of the flat matrix data array.
        """
        return self._data.msize

    @property
    def vsize(self) -> int:
        """
        Returns the total size of the flat vector data array.
        """
        return self._data.vsize

    @property
    def maxdim(self) -> wp.array:
        """
        Returns the maximum dimensions of each symmetric positive-definite matrix block.
        """
        if self._data.maxdim is None:
            raise ValueError("Cholesky factorization dimensions `maxdim` are not allocated.")
        return self._data.maxdim

    @property
    def dim(self) -> wp.array:
        """
        Returns the dimensions of each symmetric positive-definite matrix block.
        """
        if self._data.dim is None:
            raise ValueError("Cholesky factorization dimensions `dim` are not allocated.")
        return self._data.dim

    @property
    def mio(self) -> wp.array:
        """
        Returns the matrix index offset (mio) of each block in the flat data array.
        """
        if self._data.mio is None:
            raise ValueError("Cholesky factorization matrix index offset `mio` is not allocated.")
        return self._data.mio

    @property
    def vio(self) -> wp.array:
        """
        Returns the vector index offset (vio) of each block in the flat data array.
        """
        if self._data.vio is None:
            raise ValueError("Cholesky factorization vector index offset `vio` is not allocated.")
        return self._data.vio

    @property
    def L(self) -> wp.array:
        """
        Returns the Cholesky factorization matrix.
        """
        if self._data.L is None:
            raise ValueError("Cholesky factorization matrix `L` is not allocated.")
        return self._data.L

    @property
    def y(self) -> wp.array:
        """
        Returns the intermediate result buffer for the Cholesky forward/backward solve.
        """
        if self._data.y is None:
            raise ValueError("Cholesky intermediate result buffer `y` is not allocated.")
        return self._data.y

    @property
    def data(self) -> CholeskyData:
        """
        Returns a reference to the Cholesky factorization data container.
        """
        return self._data

    @abstractmethod
    def _allocate(self):
        pass

    def allocate(self, dims: list[int], allocate_info=True, device: Devicelike = None):
        """
        Allocates the Cholesky factorization data on the specified device.
        """
        # Ensure the problem dimensions are valid
        dims = self._check_dims(dims)

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self._device = device

        # Compute the allocation sizes and offsets for the flat data arrays
        mat_sizes = [n * n for n in dims]
        mat_offsets = [0] + [sum(mat_sizes[:i]) for i in range(1, len(mat_sizes) + 1)]
        mat_flat_size = sum(mat_sizes)
        vec_sizes = dims
        vec_offsets = [0] + [sum(vec_sizes[:i]) for i in range(1, len(vec_sizes) + 1)]
        vec_flat_size = sum(vec_sizes)

        # Update the allocation meta-data the specified constraint dimensions
        self._data.num_blocks = len(dims)
        self._data.dimensions = dims
        self._data.msize = mat_flat_size
        self._data.vsize = vec_flat_size

        # Allocate the Cholesky factorization data on the specified device
        with wp.ScopedDevice(self._device):
            # Allocate the Cholesky factorization matrix and the intermediate result buffer
            self._data.L = wp.zeros(shape=(self._data.msize,), dtype=float32)
            self._data.y = wp.zeros(shape=(self._data.vsize,), dtype=float32)

            # Optionally allocate the meta-data arrays if requested
            if allocate_info:
                self._data.maxdim = wp.array(self._data.dimensions, dtype=int32)
                self._data.dim = wp.array(self._data.dimensions, dtype=int32)
                self._data.mio = wp.array(mat_offsets[: self._data.num_blocks], dtype=int32)
                self._data.vio = wp.array(vec_offsets[: self._data.num_blocks], dtype=int32)

        # Call the implementation-specific allocation method for post-processing
        self._allocate()

    def zero(self):
        """
        Resets the Cholesky factorization data (L, y) to zero.\n
        """
        self._data.L.zero_()
        self._data.y.zero_()

    @abstractmethod
    def factorize(self, A: wp.array(dtype=float32)):
        """
        Performs the Cholesky factorization of the given matrix block.\n

        Args
        ----
            A (wp.array): The input flat array containing the matrix blocks to be factorized.

        Raises
        ------
            ValueError: If the input matrix is not square or exceeds the allocated size.
        """
        pass

    @abstractmethod
    def solve(self, b: wp.array(dtype=float32), x: wp.array(dtype=float32)):
        """
        Solves the linear system `A @ x = b` using the Cholesky factorization.\n

        Args
        ----
            b (wp.array): The right-hand-side vector of the linear system.
            x (wp.array): The output vector where the solution will be stored.

        Raises
        ------
            ValueError: If the right-hand-side vector `b` or the solution vector `x` exceeds the allocated size.
        """
        pass

    @abstractmethod
    def solve_inplace(self, x: wp.array(dtype=float32)):
        """
        Solves the linear system `A @ x = b` using the Cholesky factorization, storing the solution in-place.\n
        In-place means that the input vector `x` should initially contain the right-hand-side vector `b`.\n

        Args
        ----
            x (wp.array): The input vector containing the right-hand-side vector `b`, and where the solution will be stored in-place.

        Raises
        ------
            ValueError: If the input vector `x` exceeds the allocated size.
        """
        pass


class SequentialCholeskyFactorizer(CholeskyFactorizerBase):
    """
    A Cholesky factorization class computing each matrix block sequentially.\n
    This parallelizes the factorization and solve operations over each block\n
    and supports heterogeneous matrix block sizes.\n
    """

    def __init__(self, dims: list[int] = [], allocate_info=True, device: Devicelike = None):
        super().__init__(dims=dims, allocate_info=allocate_info, device=device)

    def _allocate(self):
        pass

    def factorize(self, A: wp.array(dtype=float32)):
        # Ensure the input matrix is square and matches the allocated size
        if A.shape[0] > self.data.msize:
            raise ValueError(f"Input matrix must be square and not exceed the allocated size of {(self.data.msize,)}.")

        # Perform the Cholesky factorization
        cholesky_sequential_factorize(
            num_blocks=self._data.num_blocks,
            maxdim=self._data.maxdim,
            dim=self._data.dim,
            mio=self._data.mio,
            A=A,
            L=self._data.L,
        )

    def solve(self, b: wp.array(dtype=float32), x: wp.array(dtype=float32)):
        # Ensure that the right-hand-side vector matches the allocated size
        if b.shape[0] > self.data.vsize:
            raise ValueError(
                f"Right-hand-side vector `b` is larger (shape={b.shape}) than the allocated size of {self.data.vsize}."
            )

        # Ensure that the solution vector matches the allocated size
        if x.shape[0] > self.data.vsize:
            raise ValueError(
                f"Solution vector is `x` larger (shape={x.shape}) than the allocated size of {self.data.vsize}."
            )

        # Solve the system L * y = b and L^T * x = y
        cholesky_sequential_solve(
            num_blocks=self._data.num_blocks,
            maxdim=self._data.maxdim,
            dim=self._data.dim,
            mio=self._data.mio,
            vio=self._data.vio,
            L=self._data.L,
            b=b,
            y=self._data.y,
            x=x,
        )

    def solve_inplace(self, x: wp.array(dtype=float32)):
        # Ensure that the solution vector matches the allocated size
        if x.shape[0] > self.data.vsize:
            raise ValueError(
                f"Solution vector is `x` larger (shape={x.shape}) than the allocated size of {self.data.vsize}."
            )

        # Solve the system L * y = x and L^T * x = y
        cholesky_sequential_solve_inplace(
            num_blocks=self._data.num_blocks,
            maxdim=self._data.maxdim,
            dim=self._data.dim,
            mio=self._data.mio,
            vio=self._data.vio,
            L=self._data.L,
            x=x,
        )


class BlockedCholeskyFactorizer(CholeskyFactorizerBase):
    """
    A Blocked Cholesky factorization class computing each matrix block in parallel.\n
    This implementation currently only supports homogeneous matrix block sizes,\n
    and can thus parallelize over both each outer and inner matrix blocks.\n
    """

    def __init__(
        self,
        dims: list[int] = [],
        block_size: int = 16,
        solve_block_dim: int = 64,
        factortize_block_dim: int = 128,
        allocate_info=True,
        device: Devicelike = None,
    ):
        # Initialize the base class with the specified capacity and size
        super().__init__(dims=dims, allocate_info=allocate_info, device=device)

        # Cache the block size
        self._block_size = block_size

        # Set default values for the kernel thread and block dimensions
        self._factortize_block_dim = factortize_block_dim
        self._solve_block_dim = solve_block_dim

        # Create the factorization and solve kernels
        self._factorize_kernel = make_cholesky_blocked_factorize_kernel(block_size)
        self._solve_kernel = make_cholesky_blocked_solve_kernel(block_size)
        self._solve_inplace_kernel = make_cholesky_blocked_solve_inplace_kernel(block_size)

    @property
    def block_size(self) -> int:
        """
        Returns the block size used for the blocked Cholesky factorization.
        """
        return self._block_size

    @property
    def factortize_block_dim(self):
        """
        Returns the block dimension used for the blocked Cholesky factorization.
        """
        return self._factortize_block_dim

    @factortize_block_dim.setter
    def factortize_block_dim(self, value: int):
        """
        Sets the block dimension used for the blocked Cholesky factorization.
        """
        if value <= 0:
            raise ValueError(
                f"Invalid block dimension for blocked Cholesky factorization: {value}. Must be greater than zero."
            )
        self._factortize_block_dim = value

    @property
    def solve_block_dim(self):
        """
        Returns the block dimension used for the blocked Cholesky solve.
        """
        return self._solve_block_dim

    @solve_block_dim.setter
    def solve_block_dim(self, value: int):
        """
        Sets the block dimension used for the blocked Cholesky solve.
        """
        if value <= 0:
            raise ValueError(f"Invalid block dimension for blocked Cholesky solve: {value}. Must be greater than zero.")
        self._solve_block_dim = value

    def _allocate(self):
        # First check if all dimensions are the same
        # NOTE: This is a requirement for the blocked Cholesky factorization
        mbdim = self._data.dimensions[0]
        for i in range(1, self._data.num_blocks):
            if self._data.dimensions[i] != mbdim:
                raise ValueError(
                    f"Blocked Cholesky factorization requires all dimensions to be the same: {self._data.dimensions}."
                )

        # Compute and cache the necessary matrix-vector reshaping info
        # required by the blocked Cholesky factorization kernels
        mbsize = self._data.L.size
        mbrows = mbsize // mbdim
        vbsize = self._data.y.size
        self._mshape = (mbrows, mbdim)
        self._vshape = (vbsize, 1)

    def factorize(self, A: wp.array(dtype=float32)):
        # Ensure the input matrix is square and matches the allocated size
        if A.shape[0] > self.data.msize:
            raise ValueError(f"Input matrix must be square and not exceed the allocated size of {self.data.msize}.")

        # Perform the blocked Cholesky factorization
        cholesky_blocked_factorize(
            num_blocks=self._data.num_blocks,
            dim=self._data.dim,
            rio=self._data.vio,
            A=A.reshape(self._mshape),
            L=self._data.L.reshape(self._mshape),
            kernel=self._factorize_kernel,
            block_dim=self.factortize_block_dim,
        )

    def solve(self, b: wp.array(dtype=float32), x: wp.array(dtype=float32)):
        # Ensure that the right-hand-side vector matches the allocated size
        if b.shape[0] > self.data.vsize:
            raise ValueError(
                f"Right-hand-side vector `b` is larger (shape={b.shape}) than the allocated size of {self.data.vsize}."
            )

        # Ensure that the solution vector matches the allocated size
        if x.shape[0] > self.data.vsize:
            raise ValueError(
                f"Solution vector is `x` larger (shape={x.shape}) than the allocated size of {self.data.vsize}."
            )

        # Solve the system L * y = b and L^T * x = y
        cholesky_blocked_solve(
            num_blocks=self._data.num_blocks,
            dim=self._data.dim,
            rio=self._data.vio,
            L=self._data.L.reshape(self._mshape),
            b=b.reshape(self._vshape),
            y=self._data.y.reshape(self._vshape),
            x=x.reshape(self._vshape),
            kernel=self._solve_kernel,
            block_dim=self.solve_block_dim,
        )

    def solve_inplace(self, x: wp.array(dtype=float32)):
        raise NotImplementedError(
            "In-place solve is not yet implemented for BlockedCholeskyFactorization (it is a WIP)"
        )

        # # Ensure that the solution vector matches the allocated size
        # if x.shape[0] > self.data.vsize:
        #     raise ValueError(f"Solution vector is `x` larger (shape={x.shape}) than the allocated size of {self.data.vsize}.")

        # # Solve the system L * y = b and L^T * x = y
        # cholesky_blocked_solve_inplace(
        #     num_blocks=self._data.num_blocks,
        #     dim=self._data.dim,
        #     rio=self._data.vio,
        #     L=self._data.L.reshape(self._mshape),
        #     x=x.reshape(self._vshape),
        #     kernel=self._solve_inplace_kernel,
        #     block_dim=self.solve_block_dim
        # )


CholeskyFactorizer = Union[SequentialCholeskyFactorizer, BlockedCholeskyFactorizer, None]
"""A type alias for the Cholesky factorizer, which can be either a sequential or blocked implementation."""
