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

from abc import abstractmethod
from typing import Any

import warp as wp
from warp.context import Devicelike
from warp.optim.linear import LinearOperator, cg, cr

from ..core.types import Floatlike
from .core import DenseLinearOperatorData
from .linear import LinearSolver

__all__ = [
    "ConjugateGradientSolver",
    # "ConjugateResidualSolver",
]


wp.set_module_options({"enable_backward": False})


@wp.kernel
def assign_active(
    max_dims: wp.array(dtype=wp.int32),
    vio: wp.array(dtype=wp.int32),
    dims: wp.array(dtype=wp.int32),
    a: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
):
    """Assign active entries of vector a to vector b, zeroing inactive ones. a and be need not be distinct."""
    env, row = wp.tid()

    assert env < len(dims)
    dim, maxdim = dims[env], max_dims[env]
    v0 = vio[env]
    if row < dim:
        if b != a:
            b[v0 + row] = a[v0 + row]
    elif row < maxdim:
        b[v0 + row] = type(a[0])(0)


@wp.kernel
def batched_matrix_vector(
    row_stride: wp.array(dtype=wp.int32),
    mio: wp.array(dtype=wp.int32),
    vio: wp.array(dtype=wp.int32),
    dims: wp.array(dtype=wp.int32),
    A: wp.array(dtype=Any),
    alpha: Any,
    beta: Any,
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
):
    """Computes z = alpha * (A @ x) + beta * y"""
    env, row, lane = wp.tid()
    assert env < len(dims)
    dim = dims[env]
    if row >= dim:
        return

    m0, v0 = mio[env], vio[env]
    # TODO
    # sr = row_stride[env]
    sr = dims[env]
    zero = type(alpha)(0)
    s = zero
    if alpha != zero:
        for col in range(lane, dim, wp.block_dim()):
            s += A[m0 + row * sr + col] * x[v0 + col]
    row_tile = wp.tile_sum(wp.tile(s * alpha))
    if beta != zero:
        row_tile += wp.tile_load(y, shape=1, offset=v0 + row)
    wp.tile_store(z, row_tile, offset=v0 + row)


def make_linear_operator(
    A: wp.array(dtype=Any),
    max_dims: wp.array(dtype=wp.int32),
    mio: wp.array(dtype=wp.int32),
    vio: wp.array(dtype=wp.int32),
    dims: wp.array(dtype=wp.int32),
    dtype,
    block_dim: int = 64,
    device: Devicelike | None = None,
):
    """Create a wp.optim.linalg.LinearOperator computing `z = \\alpha (A x) + \\beta y`
    for multiple, differently sized square matrices."""
    max_dims_l = max_dims.numpy().tolist()
    total_dim = sum(max_dims_l)

    def linop(x: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any), alpha: Any, beta: Any):
        wp.launch(
            batched_matrix_vector,
            inputs=[max_dims, mio, vio, dims, A, A.dtype(alpha), A.dtype(beta), x, y],
            outputs=[z],
            dim=(len(dims), max(max_dims_l), block_dim),
            block_dim=block_dim,
        )

    if device is None:
        device = max_dims.device
    return LinearOperator(shape=(total_dim, total_dim), dtype=dtype, device=device, matvec=linop)


class ConjugateSolver(LinearSolver):
    """Base class for conjugate gradient and conjugate residual solvers using warp.optim.linalg."""

    def __init__(
        self,
        operator: DenseLinearOperatorData | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: Floatlike = wp.float32,
        device: Devicelike | None = None,
        max_iter: int = 0,
        **kwargs: dict[str, Any],
    ):
        self._max_iter = max_iter
        self._A = None
        super().__init__(operator, atol, rtol, dtype, device, **kwargs)  # initializes _device and _operator

    @classmethod
    @abstractmethod
    def _solver_func(cls):
        raise NotImplementedError("Solver dispatch is not implemented.")

    def _allocate_impl(self, operator: DenseLinearOperatorData) -> None:
        if operator is not self._operator:
            raise ValueError("Cannot allocate with a different operator than used in __init__.")
        with wp.ScopedDevice(self._device):
            # Allocate the x buffer
            self._b = wp.zeros(shape=int(operator.info.maxdim.numpy().sum()))

        mi = self._operator.info
        self._A_op = make_linear_operator(
            operator.mat, mi.maxdim, mi.mio, mi.vio, mi.dim, dtype=self.dtype, device=self.device
        )

    def _callback(self):
        pass

    def _reset_impl(self) -> None:
        self._b.zero_()

    def _compute_impl(self, A: wp.array) -> None:
        self._A = A

    def _assign_active(self, a, b):
        """Assign the active elements of vector a to vector b, setting the remainder to zero."""
        mi = self._operator.info
        max_dims_l = mi.maxdim.numpy().tolist()
        wp.launch(assign_active, inputs=[mi.maxdim, mi.vio, mi.dim, a, b], dim=(len(max_dims_l), max(max_dims_l)))

    def _solve_impl(self, b: wp.array, x: wp.array, zero_x: bool = True) -> None:
        if self._A is None:
            raise ValueError("Must call compute() before calling solve()")
        if zero_x:
            x.zero_()
        else:
            self._assign_active(x, x)

        self._assign_active(b, self._b)

        self._solver_func()(
            self._A_op, self._b, x, tol=float(self._rtol), atol=float(self._atol), maxiter=self._max_iter, check_every=0
        )

    def _solve_inplace_impl(self, x: wp.array) -> None:
        if self._A is None:
            raise ValueError("Must call compute() before calling solve()")

        self._assign_active(x, self._b)
        x.zero_()
        self._solver_func()(self._A_op, self._b, x, tol=self._rtol**2)


class ConjugateGradientSolver(ConjugateSolver):
    """Conjugate gradient solver using warp.optim.linalg.cg."""

    @classmethod
    def _solver_func(cls):
        return cg


class ConjugateResidualSolver(ConjugateSolver):
    """Conjugate residual solver using warp.optim.linalg.cr."""

    @classmethod
    def _solver_func(cls):
        return cr
