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
from typing import Any

import warp as wp

from .....core.types import override
from ...core.model import Model as ModelKamino
from ...core.model import ModelData as DataKamino
from ...core.types import float32, int32
from ...geometry.contacts import Contacts as ContactsKamino
from ...kinematics.limits import Limits as LimitsKamino
from ...linalg import LinearSolverType
from ..linalg import BlockSparseLinearOperators, DenseLinearOperators, LinearOperatorsType
from ..problem import (
    ConstrainedDynamicsConfig,
    ConstrainedDynamicsProblem,
    ConstrainedDynamicsProblemData,
    SystemJacobiansType,
)

###
# Module interface
###

__all__ = [
    "PrimalProblem",
    "PrimalProblemData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class PrimalProblemData(ConstrainedDynamicsProblemData):
    """
    A container to hold the the constrained forward dynamics primal problem data over multiple worlds.
    """

    ###
    # Problem Info
    ###

    u_start: wp.array | None = None
    """
    The vector index offset of each generalized velocity vector block.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    ###
    # Problem Data
    ###

    A: LinearOperatorsType | None = None
    """
    Array of per-world primal schur complement lhs matrix blocks.\n

    Computed as
    ```
    A := M + (eta + rho)^{-1} * J^T @ J
    ```

    and together with the rhs vector
    ```
    b := w_f + (eta + rho)^{-1} * J^T @ v
    ```

    forms the linear system `A @ x = b` that defines the primal problem.

    where:
    - `M` is the block-diagonal generalized mass matrix
    - `J` is the constraint Jacobian matrix
    - `eta` is the scalar proximal regularization parameter
    - `rho` is the scalar ALM penalty parameter
    """

    b: wp.array | None = None
    """
    Array of per-world primal schur complement rhs vector blocks.

    Shape of `(info.sum_of_max_problem_dims,)` and type :class:`float32`.

    Computed as
    ```
    b := w_f + (eta + rho)^{-1} * J^T @ v
    ```

    where `w_f` is the free-motion generalized force vector
    ```
    w_f := M @ u_minus + dt * h
    ```

    `v` is the vector of constraint-space bias terms
    ```
    v := -(v^{*} + v_{rho,eta})
    ```

    with `v^{*}` being the constraint-space reference velocity
    ```
    v^{*} := v_b + v_i
    ```

    and `v_{rho,eta}` being the ALM constraint-space unbiasing velocity.

    The constraint-space bias velocities `v_b` and `v_i` are defined as
    ```
    v_b = [alpha * inv_dt * r_joints; beta * inv_dt * r_limits; gamma * inv_dt * r_contacts]
    v_i = epsilon @ (J_cts @ u_minus)
    ```

    Additionally:
    - `dt` is the time step
    - `u_minus` is the stack of per-body generalized velocities at the beginning of the time step
    - `M^{-1}` is the block-diagonal inverse generalized mass matrix
    - `J` is the constraint Jacobian matrix
    - `h` is the stack of non-linear generalized forces vectors
    - `eta` is the scalar proximal regularization parameter
    - `rho` is the scalar ALM penalty parameter
    """

    @override
    def reset(self):
        """
        Resets the problem data to zero (or identity for the preconditioner).
        """
        super().reset()
        self.A.reset()
        self.b.zero_()


###
# Functions
###

# TODO


###
# Kernels
###

# TODO


###
# Interfaces
###


class PrimalProblem(ConstrainedDynamicsProblem):
    """
    A container to manage data and operate a forward dynamics primal problem.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SystemJacobiansType | None = None,
        linear_solver_type: type[LinearSolverType] | None = None,
        linear_solver_kwargs: dict[str, Any] | None = None,
        configs: list[ConstrainedDynamicsConfig] | ConstrainedDynamicsConfig | None = None,
        device: wp.DeviceLike = None,
    ):
        """
        TODO
        """
        self._primal_data: PrimalProblemData | None = None
        """The primal forward dynamics-specific problem data container."""

        # Initialize the base class which will perform the rest of the setup work
        # NOTE: The base class constructor will call the `_finalize_impl` method
        super().__init__(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            jacobians=jacobians,
            configs=configs,
            device=device,
            linear_solver_type=linear_solver_type,
            linear_solver_kwargs=linear_solver_kwargs,
        )

    ###
    # Properties
    ###

    @property
    def data(self) -> PrimalProblemData:
        """
        Returns the primal problem data container.
        """
        return self._primal_data

    ###
    # Implementation
    ###

    @override
    def _get_sum_max_problem_dims_impl(self) -> tuple[int, int]:
        # NOTE: The primal problem is defined solely in the space of body-DoF
        world_num_body_dofs = [self._system.model.worlds[w].num_body_dofs for w in range(self.info.num_worlds)]
        return sum(world_num_body_dofs), max(world_num_body_dofs)

    @override
    def _finalize_impl(self, **kwargs) -> PrimalProblemData:
        # Construct the primal problem offset indices based on the problem dimensions
        u_size = [0] * (self.info.num_worlds)
        u_start = [0] * (self.info.num_worlds + 1)
        u_start[-1] = self.info.sum_of_max_problem_dims
        for w in range(1, self.info.num_worlds + 1):
            max_dim_w = self.system.model.worlds[w].num_body_dofs
            u_size[w - 1] = max_dim_w
            u_start[w] = u_start[w - 1] + max_dim_w
        if u_start[-1] != self.info.sum_of_max_problem_dims:
            raise ValueError(
                f"The computed primal problem offset indices do not match the total max problem dimension. "
                f"Expected last entry to be {self.info.sum_of_max_problem_dims} but got {u_start[-1]}."
            )
        if sum(u_size) != self.info.sum_of_max_problem_dims:
            raise ValueError(
                f"The computed primal problem max sizes do not match the total max problem dimension. "
                f"Expected sum of sizes to be {self.info.sum_of_max_problem_dims} but got {sum(u_size)}."
            )

        # Construct the primal system linear operator based on the constrained system
        if self.sparse:
            A_operator = BlockSparseLinearOperators()  # TODO: CONSTRUCT THIS BASED ON PROBLEM INFO
        else:
            A_operator = DenseLinearOperators()  # TODO: CONSTRUCT THIS BASED ON PROBLEM INFO

        # Construct the primal system data container
        with wp.ScopedDevice(self.device):
            self._primal_data = PrimalProblemData(
                u_start=wp.array(u_start, dtype=int32),
                A=A_operator,
                b=wp.zeros(shape=(self.info.sum_of_max_problem_dims,), dtype=float32),
                v_star=wp.zeros(shape=(self.info.sum_of_max_total_cts,), dtype=float32),
                mu=wp.zeros(shape=(self.info.sum_of_max_contacts,), dtype=float32),
                P=wp.ones(shape=(self.info.sum_of_max_problem_dims,), dtype=float32),
            )

        # Retrieve from kwargs any additional data needed for creating the linear
        # system solver instance and construct it if a supported type is specified
        linear_solver_type: type[LinearSolverType] = kwargs.pop("linear_solver_type", None)
        linear_solver_kwargs: dict[str, Any] = kwargs.pop("linear_solver_kwargs", {})
        if linear_solver_type is not None and issubclass(linear_solver_type, LinearSolverType):
            self._solver = linear_solver_type(operator=self._primal_data.A, device=self.device, **linear_solver_kwargs)
        else:
            raise ValueError("A valid linear solver type must be specified for the primal problem.")

        # Return the primal system data container to the base class constructor
        return self._primal_data

    @override
    def _reset_impl(self):
        # NOTE: The default implementation is a no-op since
        # not all CFD problem formulations may require resetting.
        pass

    @override
    def _build_impl(self):
        raise NotImplementedError("Implementation-specific `_build_impl` method is not defined.")

    @override
    def _build_preconditioner_impl(self):
        # NOTE: The default implementation is a no-op since not
        # all CFD problem formulations may require preconditioning.
        pass

    @override
    def _apply_preconditioner_impl(self):
        # NOTE: The default implementation is a no-op since not
        # all CFD problem formulations may require preconditioning.
        pass

    @override
    def _precompute_impl(self, world_mask: wp.array) -> None:
        raise NotImplementedError("The `_precompute_impl` operation is not implemented.")

    @override
    def _solve_impl(self, v: wp.array, x: wp.array, world_mask: wp.array) -> None:
        raise NotImplementedError("The `_solve_impl` operation is not implemented.")

    @override
    def _matvec_impl(self, x: wp.array, y: wp.array, world_mask: wp.array):
        raise NotImplementedError("The `_matvec_impl` operation is not implemented.")

    @override
    def _matvec_transpose_impl(self, y: wp.array, x: wp.array, world_mask: wp.array):
        raise NotImplementedError("The `_matvec_transpose_impl` operation is not implemented.")

    @override
    def _gemv_impl(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        raise NotImplementedError("The `_gemv_impl` operation is not implemented.")

    @override
    def _gemv_transpose_impl(
        self, y: wp.array, x: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0
    ):
        raise NotImplementedError("The `_gemv_transpose_impl` operation is not implemented.")
