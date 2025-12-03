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
KAMINO: Conjugate gradient and conjugate residual solvers
"""

import functools
import math
from collections.abc import Callable
from typing import Any

import warp as wp

# No need to auto-generate adjoint code for linear solvers
wp.set_module_options({"enable_backward": False})

# based on the warp.optim.linear implementation


class BatchedLinearOperator:
    """
    Linear operator to be used as left-hand-side of linear iterative solvers for a batch of independent envs(problems).
    The rhs and x vectors are stored as 2d arrays with the first dimension corresponding to the env.

    Args:
        shape: Tuple of (n_envs, max_rows, max_cols)
        dtype: Type of the operator elements
        device: Device on which computations involving the operator should be performed
        matvec: Matrix-vector multiplication routine
        ndim: int,

    The matrix-vector multiplication routine should have the following signature:

    .. code-block:: python

        def matvec(
            x: wp.array,
            y: wp.array,
            z: wp.array,
            dims: wp.array,
            alpha: Scalar,
            beta: Scalar,
        ):
            '''Perform a generalized matrix-vector product.

            This function computes the operation z = alpha * (A @ x) + beta * y, where 'A'
            is the linear operator represented by this class.

            The additional arguments enable batched solving across multiple env:
              - 'ndim' is the number of independent env processed in parallel
            '''
            ...

    For performance reasons, by default the iterative linear solvers in this module will try to capture the calls
    for one or more iterations in CUDA graphs. If the `matvec` routine of a custom :class:`BatchedLinearOperator`
    cannot be graph-captured, the ``use_cuda_graph=False`` parameter should be passed to the solver function.

    For performance reasons, no assumptions may be made about elements outside the active dims.

    """

    def __init__(self, shape: tuple[int, int, int], dtype: type, device: wp.context.Device, matvec: Callable):
        # TODO: active shapes per env, mostly for correctness
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._matvec: Callable[int, int] = matvec
        self._active_array = wp.full(shape[0], True, dtype=wp.bool, device=self._device)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def device(self) -> wp.context.Device:
        return self._device

    @property
    def matvec(self) -> Callable:
        return self._matvec

    @property
    def scalar_type(self):
        return wp.types.type_scalar_type(self.dtype)


# Implementations
# ---------------


@wp.kernel
def _check_env_termination(
    maxiter: wp.array(dtype=int),
    cycle_size: int,
    r_norm_sq: wp.array(dtype=Any),
    atol_sq: wp.array(dtype=Any),
    env_active: wp.array(dtype=wp.bool),
    cur_iter: wp.array(dtype=int),
    env_condition: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    active = wp.where(env_active[env], 1, 0)
    if active == 0 or env_condition[env] == 0:
        return
    cur_iter[env] += cycle_size
    env_condition[env] = wp.where(r_norm_sq[env] <= atol_sq[env] or cur_iter[env] >= maxiter[env], 0, active)


@wp.kernel
def _update_batch_condition(
    env_conditions: wp.array(dtype=wp.int32),
    n_envs: int,
    batch_condition: wp.array(dtype=wp.int32),
):
    for i in range(n_envs):
        if env_conditions[i]:
            batch_condition[0] = 1
            return
    batch_condition[0] = 0


@functools.cache
def make_termination_kernel(n_envs):
    @wp.kernel
    def check_termination(
        maxiter: wp.array(dtype=int),
        cycle_size: int,
        r_norm_sq: wp.array(dtype=Any),
        atol_sq: wp.array(dtype=Any),
        env_active: wp.array(dtype=wp.bool),
        cur_iter: wp.array(dtype=int),
        env_condition: wp.array(dtype=wp.int32),
        batch_condition: wp.array(dtype=wp.int32),
    ):
        thread = wp.tid()
        active = wp.tile_astype(wp.tile_load(env_active, (n_envs,)), wp.int32)
        condition = wp.tile_load(env_condition, (n_envs,))
        env_stepped = wp.tile_map(wp.mul, active, condition)
        iter = env_stepped * cycle_size + wp.tile_load(cur_iter, (n_envs,))

        wp.tile_store(cur_iter, iter)
        cont_norm = wp.tile_astype(
            wp.tile_map(lt_mask, wp.tile_load(atol_sq, (n_envs,)), wp.tile_load(r_norm_sq, (n_envs,))), wp.int32
        )
        cont_iter = wp.tile_map(lt_mask, iter, wp.tile_load(maxiter, (n_envs,)))
        cont = wp.tile_map(wp.mul, wp.tile_map(wp.mul, cont_iter, cont_norm), env_stepped)
        wp.tile_store(env_condition, cont)
        batch_cont = wp.where(wp.tile_sum(cont)[0] > 0, 1, 0)
        if thread == 0:
            batch_condition[0] = batch_cont

    return check_termination


@wp.kernel
def _dense_mv_kernel(
    A: wp.array2d(dtype=Any),
    x: wp.array1d(dtype=Any),
    y: wp.array1d(dtype=Any),
    z: wp.array1d(dtype=Any),
    alpha: Any,
    beta: Any,
):
    row, lane = wp.tid()

    zero = type(alpha)(0)
    s = zero
    if alpha != zero:
        for col in range(lane, A.shape[1], wp.block_dim()):
            s += A[row, col] * x[col]

    row_tile = wp.tile_sum(wp.tile(s * alpha))

    if beta != zero:
        row_tile += wp.tile_load(y, shape=1, offset=row) * beta

    wp.tile_store(z, row_tile, offset=row)


@wp.kernel
def _diag_mv_kernel(
    A: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    alpha: Any,
    beta: Any,
):
    i = wp.tid()
    zero = type(alpha)(0)
    s = z.dtype(zero)
    if alpha != zero:
        s += alpha * (A[i] * x[i])
    if beta != zero:
        s += beta * y[i]
    z[i] = s


@wp.kernel
def _cg_kernel_1(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    p_Ap: wp.array(dtype=Any),
    p: wp.array2d(dtype=Any),
    Ap: wp.array2d(dtype=Any),
    x: wp.array2d(dtype=Any),
    r: wp.array2d(dtype=Any),
):
    e, i = wp.tid()

    alpha = wp.where(resid[e] > tol[e], rz_old[e] / p_Ap[e], rz_old.dtype(0.0))

    x[e, i] = x[e, i] + alpha * p[e, i]
    r[e, i] = r[e, i] - alpha * Ap[e, i]


@wp.kernel
def _cg_kernel_2(
    tol: wp.array(dtype=Any),
    resid_new: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    rz_new: wp.array(dtype=Any),
    z: wp.array2d(dtype=Any),
    p: wp.array2d(dtype=Any),
):
    #    p = r + (rz_new / rz_old) * p;
    e, i = wp.tid()

    cond = resid_new[e] > tol[e]
    beta = wp.where(cond, rz_new[e] / rz_old[e], rz_old.dtype(0.0))

    p[e, i] = z[e, i] + beta * p[e, i]


@wp.kernel
def _cr_kernel_1(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    zAz_old: wp.array(dtype=Any),
    y_Ap: wp.array(dtype=Any),
    p: wp.array2d(dtype=Any),
    Ap: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    x: wp.array2d(dtype=Any),
    r: wp.array2d(dtype=Any),
    z: wp.array2d(dtype=Any),
):
    e, i = wp.tid()

    alpha = wp.where(resid[e] > tol[e] and y_Ap[e] > 0.0, zAz_old[e] / y_Ap[e], zAz_old.dtype(0.0))

    x[e, i] = x[e, i] + alpha * p[e, i]
    r[e, i] = r[e, i] - alpha * Ap[e, i]
    z[e, i] = z[e, i] - alpha * y[e, i]


@wp.kernel
def _cr_kernel_2(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    zAz_old: wp.array(dtype=Any),
    zAz_new: wp.array(dtype=Any),
    z: wp.array2d(dtype=Any),
    Az: wp.array2d(dtype=Any),
    p: wp.array2d(dtype=Any),
    Ap: wp.array2d(dtype=Any),
):
    #    p = r + (rz_new / rz_old) * p;
    e, i = wp.tid()

    beta = wp.where(resid[e] > tol[e] and zAz_old[e] > 0.0, zAz_new[e] / zAz_old[e], zAz_old.dtype(0.0))

    p[e, i] = z[e, i] + beta * p[e, i]
    Ap[e, i] = Az[e, i] + beta * Ap[e, i]


# Diagonal matrices
@wp.kernel
def diag_matvec_kernel(
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    D: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    env_active: wp.array(dtype=wp.bool),
    alpha: Any,
    beta: Any,
    z: wp.array2d(dtype=Any),
):
    env, row = wp.tid()
    assert env < len(active_dims)
    if not env_active[env] or row >= active_dims[env]:
        return

    zero = type(alpha)(0)
    s = z.dtype(0)

    if alpha != zero:
        s += alpha * D[env, row] * x[env, row]
    if beta != zero:
        s += beta * y[env, row]
    z[env, row] = s


def make_diag_matrix_operator(
    diag: wp.array2d(dtype=Any),
    max_dims: int,
    active_dims: wp.array(dtype=Any),
) -> BatchedLinearOperator:
    # int_type = active_dims.dtype
    n = len(active_dims)
    device = diag.device
    assert active_dims.device == device

    dtype = diag.dtype
    batch_size = n

    def matvec(
        x: wp.array2d(dtype=Any),
        y: wp.array2d(dtype=Any),
        z: wp.array2d(dtype=Any),
        env_active: wp.array(dtype=wp.bool),
        alpha: Any,
        beta: Any,
    ):
        wp.launch(
            diag_matvec_kernel,
            dim=(batch_size, max_dims),
            inputs=[x, y, diag, active_dims, env_active, dtype(alpha), dtype(beta)],
            outputs=[z],
            device=device,
        )

    shape = (n, max_dims, max_dims)
    return BatchedLinearOperator(shape=shape, dtype=dtype, device=device, matvec=matvec)


# Dense matrices
@wp.kernel
def _dense_matvec_kernel(
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    A: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    env_active: wp.array(dtype=wp.bool),
    alpha: Any,
    beta: Any,
    matrix_stride: int,
    tile_size: int,
    z: wp.array2d(dtype=Any),
):
    """Computes z[i] = alpha * (A[i] @ x[i]) + beta * y[i]"""
    env, row, lane = wp.tid()
    assert env < len(active_dims)
    dim = active_dims[env]
    if not env_active[env] or row >= dim:
        return

    row_stride = active_dims[env]  # Active elements are contiguous in memory
    zero = type(alpha)(0)
    s = zero
    if alpha != zero:
        for col in range(lane, dim, tile_size):
            s += A[env, row * row_stride + col] * x[env, col]
    row_tile = wp.tile_sum(wp.tile(s * alpha))
    if beta != zero:
        row_tile += beta * wp.tile_load(y[env], shape=1, offset=row)
    wp.tile_store(z[env], row_tile, offset=row)


def make_dense_square_matrix_operator(
    A: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    max_dims: int,
    matrix_stride: int,
    block_dim: int = 64,
) -> BatchedLinearOperator:
    """Create a wp.optim.linalg.LinearOperator computing `z = \\alpha (A x) + \\beta y`
    for multiple, differently sized square matrices."""
    dtype = A.dtype
    device = A.device

    tile_size = block_dim if device.is_cuda else 1
    n = len(active_dims)
    assert A.shape[0] == n

    def matvec(
        x: wp.array(dtype=Any),
        y: wp.array(dtype=Any),
        z: wp.array(dtype=Any),
        env_active: wp.array(dtype=wp.bool),
        alpha: Any,
        beta: Any,
    ):
        wp.launch(
            _dense_matvec_kernel,
            dim=(len(active_dims), max_dims, block_dim),
            inputs=[x, y, A, active_dims, env_active, dtype(alpha), dtype(beta), matrix_stride, tile_size],
            outputs=[z],
            device=device,
            block_dim=tile_size if tile_size > 1 else 256,
        )

    shape = (n, max_dims, max_dims)
    return BatchedLinearOperator(shape=shape, dtype=dtype, device=device, matvec=matvec)


def _run_capturable_loop(
    do_cycle: Callable,
    r_norm_sq: wp.array,
    env_active: wp.array(dtype=wp.bool),
    cur_iter: wp.array(dtype=wp.int32),
    conditions: wp.array(dtype=wp.int32),
    maxiter: wp.array(dtype=int),
    atol_sq: wp.array,
    callback: Callable | None,
    check_every: int,
    use_cuda_graph: bool,
    cycle_size: int = 1,
    termination_kernel=None,
):
    # TODO: check-every > 0 without python-space code
    assert check_every == 0
    device = atol_sq.device

    n_envs = maxiter.shape[0]
    cur_iter.fill_(-1)
    conditions.fill_(1)

    env_condition, global_condition = conditions[:n_envs], conditions[n_envs:]

    update_condition_launch = wp.launch(
        termination_kernel,
        dim=(1, n_envs),
        block_dim=n_envs,
        device=device,
        inputs=[maxiter, cycle_size, r_norm_sq, atol_sq, env_active, cur_iter],
        outputs=[env_condition, global_condition],
        record_cmd=True,
    )

    if isinstance(callback, wp.Kernel):
        callback_launch = wp.launch(
            callback, dim=n_envs, device=device, inputs=[cur_iter, r_norm_sq, atol_sq], record_cmd=True
        )
    else:
        callback_launch = None

    # TODO: consider using a spinlock for fusing kernels
    # update_env_condition_launch.launch()
    # update_global_condition_launch.launch()
    update_condition_launch.launch()

    if callback_launch is not None:
        callback_launch.launch()

    def do_cycle_with_condition():
        # print("Global cond:", global_condition.numpy())
        do_cycle()
        update_condition_launch.launch()
        if callback_launch is not None:
            callback_launch.launch()

    if use_cuda_graph and device.is_cuda and device.is_capturing:
        wp.capture_while(global_condition, do_cycle_with_condition)
    else:
        for _ in range(0, int(maxiter.numpy().max()), cycle_size):
            do_cycle_with_condition()
            if not global_condition.numpy()[0]:
                # print("Exiting")
                break

    return cur_iter, r_norm_sq, atol_sq


@wp.kernel
def dot_kernel(
    a: wp.array3d(dtype=Any),
    b: wp.array3d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    env_active: wp.array(dtype=wp.bool),
    dot: wp.array3d(dtype=Any),
):
    # called with (n_cols, n_envs)
    col, env = wp.tid()
    assert col < a.shape[0]
    assert col < b.shape[0]
    assert env < a.shape[1]
    assert env < b.shape[1]
    assert env_active.shape[0] == a.shape[1]

    if not env_active[env]:
        dot[col, env, 0] = a.dtype(0)
        return

    # use float64 to control error
    z = wp.float64(0)

    for i in range(active_dims[env]):
        z += wp.float64(a[col, env, i] * b[col, env, i])

    dot[col, env, 0] = a.dtype(z)


@wp.func
def lt_mask(a: Any, b: Any):
    """Return 1 if a < b, else 0"""
    return wp.where(a < b, type(a)(1), type(a)(0))


WP_NO_TILE_FULL = True  # TODO: remove w/ warp upgrade


@wp.func
def mul_mask(mask: Any, value: Any):
    """Return value if mask is positive, else 0"""
    return wp.where(mask > type(mask)(0), value, type(value)(0))


@functools.cache
def make_dot_kernel(tile_size: int, maxdim: int):
    second_tile_size = (maxdim + tile_size - 1) // tile_size

    @wp.kernel(enable_backward=False)
    def dot(
        a: wp.array3d(dtype=Any),
        b: wp.array3d(dtype=Any),
        env_size: wp.array(dtype=wp.int32),
        env_active: wp.array(dtype=wp.bool),
        result: wp.array2d(dtype=Any),
    ):
        """Compute the dot products between the trailing-dim arrays in a and b using tiles and pairwise summation."""
        col, env, _ = wp.tid()
        if not env_active[env]:
            return
        n = env_size[env]

        ts = wp.tile_zeros((second_tile_size,), dtype=a.dtype, storage="shared")
        o_src = wp.int32(0)

        for block in range(second_tile_size):
            if o_src >= n:
                break
            ta = wp.tile_load(a[col, env], shape=tile_size, offset=o_src)
            tb = wp.tile_load(b[col, env], shape=tile_size, offset=o_src)
            # TODO: consider using ts[block] twice, look into += data race in wp
            prod = wp.tile_map(wp.mul, ta, tb)
            if o_src > n - tile_size:
                if wp.static(WP_NO_TILE_FULL):
                    thresh_scalar = wp.tile_zeros(1, dtype=a.dtype)
                    thresh_scalar[0] = a.dtype(n - o_src)
                    thresh = wp.tile_broadcast(thresh_scalar, (tile_size,))
                else:
                    thresh = wp.tile_full((tile_size,), a.dtype(n - o_src), dtype=a.dtype)
                mask = wp.tile_map(lt_mask, wp.tile_arange(tile_size, dtype=a.dtype), thresh)
                prod = wp.tile_map(mul_mask, mask, prod)
            s = wp.tile_sum(prod)
            ts[block] = s[0]
            o_src += tile_size
        wp.tile_store(result[col], wp.tile_sum(ts), offset=env)

    return dot


@functools.cache
def create_tiled_dot_kernels(tile_size):
    @wp.kernel
    def block_dot_kernel(
        a: wp.array2d(dtype=Any),
        b: wp.array2d(dtype=Any),
        partial_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, tid_block = wp.tid()

        start = block_id * tile_size

        a_block = wp.tile_load(a[column], shape=tile_size, offset=start)
        b_block = wp.tile_load(b[column], shape=tile_size, offset=start)
        t = wp.tile_map(wp.mul, a_block, b_block)

        tile_sum = wp.tile_sum(t)
        wp.tile_store(partial_sums[column], tile_sum, offset=block_id)

    @wp.kernel
    def block_sum_kernel(
        data: wp.array2d(dtype=Any),
        partial_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, tid_block = wp.tid()
        start = block_id * tile_size

        t = wp.tile_load(data[column], shape=tile_size, offset=start)

        tile_sum = wp.tile_sum(t)
        wp.tile_store(partial_sums[column], tile_sum, offset=block_id)

    return block_dot_kernel, block_sum_kernel


class CGSolver:
    def __init__(
        self,
        A: BatchedLinearOperator,
        active_dims: wp.array(dtype=Any),
        env_active: wp.array(dtype=wp.bool),
        atol_sq: wp.array(dtype=Any),
        maxiter: wp.array = None,
        M: BatchedLinearOperator | None = None,
        callback: Callable | None = None,
        check_every=10,
        use_cuda_graph=True,
    ):
        if not isinstance(A, BatchedLinearOperator):
            raise ValueError("A must be a BatchedLinearOperator")
        if not isinstance(M, BatchedLinearOperator | None):
            raise ValueError("M must be a BatchedLinearOperator or None")

        self.scalar_type = A.scalar_type

        self.n_envs, self.maxdims, _ = A.shape
        if self.maxdims != A.shape[2]:
            raise ValueError("A must be a square BatchedLinearOperator")
        if M is not None and A.shape != M.shape:
            raise ValueError("M and A must have the same dimensions")

        self.A = A
        self.M = M
        self.device = A.device

        self.active_dims = active_dims
        self.env_active = env_active
        self.atol_sq = atol_sq
        self.maxiter = maxiter

        self.callback = callback
        self.check_every = check_every
        self.use_cuda_graph = use_cuda_graph

        self.dot_tile_size = min(2048, 2 ** math.ceil(math.log(241, 2)))
        self.tiled_dot_kernel = make_dot_kernel(self.dot_tile_size, self.maxdims)
        self._allocate()

    def _allocate(self):
        # Temp storage
        self.r_and_z = wp.empty((2, *self.A.shape[:2]), dtype=self.scalar_type, device=self.device)
        self.p_and_Ap = wp.empty_like(self.r_and_z)
        self.residual = wp.empty((self.n_envs), dtype=self.scalar_type, device=self.device)

        if self.maxiter is None:
            self.maxiter = wp.full(self.n_envs, int(1.5 * self.maxdims), dtype=int, device=self.device)

        # TODO: non-tiled variant for CPU
        self.dot_product = wp.empty((2, self.n_envs), dtype=self.scalar_type, device=self.device)

        # (r, r) -- so we can compute r.z and r.r at once
        self.r_repeated = _repeat_first(self.r_and_z)
        if self.M is None:
            # without preconditioner r == z
            # TODO: allocate r_and_z here
            self.r_and_z = self.r_repeated
            self.rz_new = self.dot_product[0]
        else:
            self.rz_new = self.dot_product[1]

        self.cur_iter = wp.empty(self.n_envs, dtype=wp.int32, device=self.device)
        self.conditions = wp.empty(self.n_envs + 1, dtype=wp.int32, device=self.device)
        self.termination_kernel = make_termination_kernel(self.n_envs)

    def compute_dot(self, a, b, col_offset=0):
        block_dim = 256
        if a.ndim == 2:
            a = a.reshape((1, *a.shape))
            b = b.reshape((1, *b.shape))

        result = self.dot_product[col_offset:]

        wp.launch(
            self.tiled_dot_kernel,
            dim=(a.shape[0], self.n_envs, block_dim),
            block_dim=min(256, self.dot_tile_size // 8),
            inputs=[a, b, self.active_dims, self.env_active],
            outputs=[result],
            device=self.device,
        )

    def update_rr_rz(self, r, z, r_repeated):
        # z = M r
        if self.M is None:
            self.compute_dot(r, r)
        else:
            self.M.matvec(r, z, z, self.env_active, alpha=1.0, beta=0.0)
            self.compute_dot(r_repeated, self.r_and_z)

    def solve(self, b: wp.array, x: wp.array):
        r, z = self.r_and_z[0], self.r_and_z[1]
        r_norm_sq = self.dot_product[0]
        p, Ap = self.p_and_Ap[0], self.p_and_Ap[1]

        # Not strictly necessary, but makes it more robust to user-provided BatchedLinearOperators
        Ap.zero_()
        z.zero_()

        # Initialize residual
        self.A.matvec(x, b, r, self.env_active, alpha=-1.0, beta=1.0)
        self.update_rr_rz(r, z, self.r_repeated)
        p.assign(z)

        do_iteration = functools.partial(
            self.do_iteration, p=p, Ap=Ap, rz_old=self.residual, rz_new=self.rz_new, z=z, x=x, r=r, r_norm_sq=r_norm_sq
        )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq,
            self.env_active,
            self.cur_iter,
            self.conditions,
            self.maxiter,
            self.atol_sq,
            self.callback,
            self.check_every,
            self.use_cuda_graph,
            termination_kernel=self.termination_kernel,
        )

    def do_iteration(self, p, Ap, rz_old, rz_new, z, x, r, r_norm_sq):
        rz_old.assign(rz_new)

        # Ap = A * p;
        self.A.matvec(p, Ap, Ap, self.env_active, alpha=1, beta=0)
        self.compute_dot(p, Ap, col_offset=1)
        p_Ap = self.dot_product[1]

        wp.launch(
            kernel=_cg_kernel_1,
            dim=(self.n_envs, self.maxdims),
            inputs=[self.atol_sq, r_norm_sq, rz_old, p_Ap, p, Ap],
            outputs=[x, r],
            device=self.device,
        )

        self.update_rr_rz(r, z, self.r_repeated)

        wp.launch(
            kernel=_cg_kernel_2,
            dim=(self.n_envs, self.maxdims),
            inputs=[self.atol_sq, r_norm_sq, rz_old, rz_new, z],
            outputs=[p],
            device=self.device,
        )


class CRSolver:
    def __init__(
        self,
        A: BatchedLinearOperator,
        active_dims: wp.array(dtype=Any),
        env_active: wp.array(dtype=wp.bool),
        atol_sq: wp.array(dtype=Any),
        maxiter: wp.array = None,
        M: BatchedLinearOperator | None = None,
        callback: Callable | None = None,
        check_every=10,
        use_cuda_graph=True,
    ):
        if not isinstance(A, BatchedLinearOperator):
            raise ValueError("A must be a BatchedLinearOperator")
        if not isinstance(M, BatchedLinearOperator | None):
            raise ValueError("M must be a BatchedLinearOperator or None")

        self.scalar_type = A.scalar_type

        self.n_envs, self.maxdims, _ = A.shape
        if self.maxdims != A.shape[2]:
            raise ValueError("A must be a square BatchedLinearOperator")
        if M is not None and A.shape != M.shape:
            raise ValueError("M and A must have the same dimensions")

        self.A = A
        self.M = M
        self.device = A.device

        self.active_dims = active_dims
        self.env_active = env_active
        self.atol_sq = atol_sq
        self.maxiter = maxiter

        self.callback = callback
        self.check_every = check_every
        self.use_cuda_graph = use_cuda_graph

        self.dot_tile_size = min(2048, 2 ** math.ceil(math.log(241, 2)))
        self.tiled_dot_kernel = make_dot_kernel(self.dot_tile_size, self.maxdims)
        self._allocate()

    def _allocate(self):
        # Temp storage
        self.r_and_z = wp.empty((2, *self.A.shape[:2]), dtype=self.scalar_type, device=self.device)
        self.r_and_Az = wp.empty_like(self.r_and_z)
        self.y_and_Ap = wp.empty_like(self.r_and_z)
        self.p = wp.empty(self.A.shape[:2], dtype=self.scalar_type, device=self.device)
        self.residual = wp.empty((self.n_envs), dtype=self.scalar_type, device=self.device)

        if self.maxiter is None:
            self.maxiter = wp.full(self.n_envs, int(1.5 * self.maxdims), dtype=int, device=self.device)

        self.dot_product = wp.empty((2, self.n_envs), dtype=self.scalar_type, device=self.device)

        # (r, r) -- so we can compute r.z and r.r at once
        if self.M is None:
            # For the unpreconditioned case, z == r and y == Ap
            self.r_and_z = _repeat_first(self.r_and_z)
            self.y_and_Ap = _repeat_first(self.y_and_Ap)

        self.cur_iter = wp.empty(self.n_envs, dtype=wp.int32, device=self.device)
        self.conditions = wp.empty(self.n_envs + 1, dtype=wp.int32, device=self.device)
        self.termination_kernel = make_termination_kernel(self.n_envs)

        # Notations below follow roughly pseudo-code from https://en.wikipedia.org/wiki/Conjugate_residual_method
        # with z := M^-1 r and y := M^-1 Ap

    def compute_dot(self, a, b, col_offset=0):
        block_dim = 256
        if a.ndim == 2:
            a = a.reshape((1, *a.shape))
            b = b.reshape((1, *b.shape))

        result = self.dot_product[col_offset:]

        wp.launch(
            self.tiled_dot_kernel,
            dim=(a.shape[0], self.n_envs, block_dim),
            block_dim=min(256, self.dot_tile_size // 8),
            inputs=[a, b, self.active_dims, self.env_active],
            outputs=[result],
            device=self.device,
        )

    def update_rr_zAz(self, z, Az, r, r_copy):
        self.A.matvec(z, Az, Az, self.env_active, alpha=1, beta=0)
        r_copy.assign(r)
        self.compute_dot(self.r_and_z, self.r_and_Az)

    def solve(self, b: wp.array, x: wp.array):
        # named views
        r, z = self.r_and_z[0], self.r_and_z[1]
        r_copy, Az = self.r_and_Az[0], self.r_and_Az[1]
        y, Ap = self.y_and_Ap[0], self.y_and_Ap[1]

        r_norm_sq = self.dot_product[0]
        zAz_new = self.dot_product[1]
        zAz_old = self.residual

        # Initialize tolerance from right-hand-side norm
        self.A.matvec(x, b, r, self.env_active, alpha=-1.0, beta=1.0)

        # Not strictly necessary, but makes it more robust to user-provided LinearOperators
        # y.zero_()
        # Ap.zero_()
        # self.y_and_Ap.zero_()

        # z = M r
        if self.M is not None:
            z.zero_()
            self.M.matvec(r, z, z, self.env_active, alpha=1.0, beta=0.0)

        self.update_rr_zAz(z, Az, r, r_copy)

        self.p.assign(z)
        Ap.assign(Az)

        do_iteration = functools.partial(
            self.do_iteration,
            p=self.p,
            Ap=Ap,
            Az=Az,
            zAz_old=self.residual,
            zAz_new=self.dot_product[1],
            z=z,
            y=y,
            x=x,
            r=r,
            r_copy=r_copy,
            r_norm_sq=r_norm_sq,
        )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq,
            self.env_active,
            self.cur_iter,
            self.conditions,
            self.maxiter,
            self.atol_sq,
            self.callback,
            self.check_every,
            self.use_cuda_graph,
            termination_kernel=self.termination_kernel,
        )

    def do_iteration(self, p, Ap, Az, zAz_old, zAz_new, z, y, x, r, r_copy, r_norm_sq):
        zAz_old.assign(zAz_new)

        if self.M is not None:
            self.M.matvec(Ap, y, y, self.env_active, alpha=1.0, beta=0.0)
        self.compute_dot(Ap, y, col_offset=1)
        y_Ap = self.dot_product[1]

        if self.M is None:
            # In non-preconditioned case, first kernel is same as CG
            wp.launch(
                kernel=_cg_kernel_1,
                dim=(self.n_envs, self.maxdims),
                inputs=[self.atol_sq, r_norm_sq, zAz_old, y_Ap, p, Ap],
                outputs=[x, r],
                device=self.device,
            )
        else:
            # In preconditioned case, we have one more vector to update
            wp.launch(
                kernel=_cr_kernel_1,
                dim=(self.n_envs, self.maxdims),
                inputs=[self.atol_sq, r_norm_sq, zAz_old, y_Ap, p, Ap, y],
                outputs=[x, r, z],
                device=self.device,
            )

        self.update_rr_zAz(z, Az, r, r_copy)

        wp.launch(
            kernel=_cr_kernel_2,
            dim=(self.n_envs, self.maxdims),
            inputs=[self.atol_sq, r_norm_sq, zAz_old, zAz_new, z, Az],
            outputs=[p, Ap],
            device=self.device,
        )


def _repeat_first(arr: wp.array):
    # returns a view of the first element repeated arr.shape[0] times
    view = wp.array(
        ptr=arr.ptr,
        shape=arr.shape,
        dtype=arr.dtype,
        strides=(0, *arr.strides[1:]),
        device=arr.device,
    )
    view._ref = arr
    return view
