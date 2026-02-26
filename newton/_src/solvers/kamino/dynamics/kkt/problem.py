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

import numpy as np
import warp as wp

from .....core.types import override
from ...core.model import Model as ModelKamino
from ...core.model import ModelData as DataKamino
from ...core.types import float32, int32
from ...geometry.contacts import Contacts as ContactsKamino
from ...kinematics.constraints import get_max_constraints_per_world
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
    "KKTProblem",
    "KKTProblemData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class KKTProblemData(ConstrainedDynamicsProblemData):
    """
    A container to hold the the KKT system data over multiple worlds.
    """

    ###
    # Problem Info
    ###

    ux_start: wp.array | None = None
    """
    The vector index offset of each problem vector block.\n
    Shape of `(num_worlds + 1,)` and type :class:`int32`.\n
    The last entry is equal to `sum_of_max_problem_dims` the total max problem dimension across all worlds.
    """

    ###
    # Problem Data
    ###

    K: LinearOperatorsType | None = None
    """
    The KKT system linear operator.

    The KKT problem is defined as
    ```
    K := [ M   J^T       ] , k := [ w_f               ] , ux := [ u^{+}    ]
         [ J   sigma * I ]        [ -v^* + v_{sigma}  ]         [ -lambdas ]
    ```

    forming the linear system
    ```
    K @ ux = k
    ```

    where:
    - `M` is the generalized mass matrix
    - `J` is the constraint Jacobian matrix
    - `sigma` is the scalar primal diagonal regularization parameter
    - `w_f` is the stack of free-motion generalized forces vector
    - `v^*` is the stack of constraint-space reference velocity bias term
    - `v_{sigma}` is the stack of constraint-space ALM unbiasing velocity vector
    - `u^{+}` is the stack of per-body generalized velocities
       (i.e. body twists) at the end of the time step
    - `lambdas` is the stack of constraint reactions vector
    """

    k: wp.array | None = None
    """
    Stack of KKT rhs vectors.

    Shape of `(info.sum_of_max_problem_dims,)` and type :class:`float32`.

    Defined as
    ```
    k := [ w_f               ]
         [ -v^{*} + v_{sigma}  ]
    ```

    Where the free-motion generalized forces vector `w_f` is defined as
    ```
    w_f := M @ u_minus + dt * h
    ```

    and `v^{*}` is the constraint-space reference velocity
    ```
    v^{*} := v_b + v_i
    ```

    The constraint-space bias velocities `v_b` and `v_i` are defined as
    ```
    v_b = [alpha * inv_dt * r_joints; beta * inv_dt * r_limits; gamma * inv_dt * r_contacts]
    v_i = epsilon @ (J @ u_minus)
    ```

    Additionally:
    - `M` is the generalized mass matrix
    - `J` is the constraint Jacobian matrix
    - `h` is the stack of non-linear generalized forces vectors
    - `u_minus` is the stack of per-body generalized velocities at the beginning of the time step
    - `dt` is the time step
    - `alpha`, `beta` and `gamma` are the scalar constraint-space bias scaling parameters
    - `epsilon` is the scalar constraint-space bias unregularized compliance parameter
    """

    @override
    def reset(self):
        """
        Resets the problem data to zero (or identity for the preconditioner).
        """
        super().reset()
        self.K.reset()
        self.k.zero_()


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


class KKTProblem(ConstrainedDynamicsProblem):
    """
    A container to manage data and operate a constrained forward dynamics KKT problem.
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
        self._kkt_data: KKTProblemData | None = None
        """The KKT-system-specific problem data container."""

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
    def data(self) -> KKTProblemData:
        """
        Returns the KKT system data container.
        """
        return self._kkt_data

    ###
    # Implementation
    ###

    @override
    def _get_sum_max_problem_dims_impl(self) -> tuple[int, int]:
        # NOTE: The KKT problem is defined in the space of concatenation of body-DoF + constraints
        world_num_body_dofs = [self._system.model.worlds[w].num_body_dofs for w in range(self.info.num_worlds)]
        world_max_cts = get_max_constraints_per_world(self._system.model, self._system.limits, self._system.contacts)
        world_max_problem_dims = np.array(world_num_body_dofs, dtype=np.int32)
        world_max_problem_dims += np.array(world_max_cts, dtype=np.int32)
        return int(world_max_problem_dims.sum()), int(world_max_problem_dims.max())

    @override
    def _finalize_impl(self, **kwargs: dict[str, Any]) -> KKTProblemData:
        # Construct the KKT problem offset indices based on the problem dimensions
        ux_size = [0] * (self.info.num_worlds)
        ux_start = [0] * (self.info.num_worlds + 1)
        ux_start[-1] = self.info.sum_of_max_problem_dims
        for w in range(1, self.info.num_worlds + 1):
            max_dim_w = self.system.model.worlds[w].num_body_dofs + self.info.world_max_total_cts[w]
            ux_size[w - 1] = max_dim_w
            ux_start[w] = ux_start[w - 1] + max_dim_w
        if ux_start[-1] != self.info.sum_of_max_problem_dims:
            raise ValueError(
                f"The computed KKT problem offset indices do not match the total max problem dimension. "
                f"Expected last entry to be {self.info.sum_of_max_problem_dims} but got {ux_start[-1]}."
            )
        if sum(ux_size) != self.info.sum_of_max_problem_dims:
            raise ValueError(
                f"The computed KKT problem max sizes do not match the total max problem dimension. "
                f"Expected sum of sizes to be {self.info.sum_of_max_problem_dims} but got {sum(ux_size)}."
            )

        # Construct the KKT system linear operator based on the constrained system
        if self.sparse:
            K_operator = BlockSparseLinearOperators()  # TODO: CONSTRUCT THIS BASED ON PROBLEM INFO
        else:
            K_operator = DenseLinearOperators()  # TODO: CONSTRUCT THIS BASED ON PROBLEM INFO

        # Construct the KKT system data container
        with wp.ScopedDevice(self.device):
            self._kkt_data = KKTProblemData(
                ux_start=wp.array(ux_start, dtype=int32),
                K=K_operator,
                k=wp.zeros(shape=(self.info.sum_of_max_problem_dims,), dtype=float32),
                v_star=wp.zeros(shape=(self.info.sum_of_max_total_cts,), dtype=float32),
                mu=wp.zeros(shape=(self.info.sum_of_max_contacts,), dtype=float32),
                P=wp.ones(shape=(self.info.sum_of_max_problem_dims,), dtype=float32),
            )

        # Retrieve from kwargs any additional data needed for creating the linear
        # system solver instance and construct it if a supported type is specified
        linear_solver_type: type[LinearSolverType] = kwargs.pop("linear_solver_type", None)
        linear_solver_kwargs: dict[str, Any] = kwargs.pop("linear_solver_kwargs", {})
        if linear_solver_type is not None and issubclass(linear_solver_type, LinearSolverType):
            self._solver = linear_solver_type(operator=self._kkt_data.K, device=self.device, **linear_solver_kwargs)
        else:
            raise ValueError("A valid linear solver type must be specified for the KKT problem.")

        # Return the KKT system data container to the base class constructor
        return self._kkt_data

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
