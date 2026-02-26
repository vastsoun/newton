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
from ...core.types import float32, int32, vec6f
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
from .kernels import (
    _add_matrix_diag_product,
    _apply_dual_preconditioner_to_matrix_dense,
    _apply_dual_preconditioner_to_vector,
    _build_dense_delassus_elementwise,
    _build_free_velocity_bias_contacts,
    _build_free_velocity_bias_joints,
    _build_free_velocity_bias_limits,
    _build_free_velocity_dense,
    _build_free_velocity_sparse,
    _build_dual_preconditioner_dense,
    _build_dual_preconditioner_sparse,
    _build_generalized_free_velocity,
    _build_nonlinear_generalized_force,
    _compute_delassus_diagonal_sparse,
    _inverse_mass_matrix_matvec,
    _regularize_dense_delassus_diagonal,
    _scaled_vector_sum,
    _set_matrix_diag_product,
)

###
# Module interface
###

__all__ = [
    "DualProblem",
    "DualProblemData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class DualProblemData(ConstrainedDynamicsProblemData):
    """
    A container to hold the the constrained forward dynamics dual problem data over multiple worlds.
    """

    ###
    # Problem Info
    ###

    x_start: wp.array | None = None
    """
    The vector index offset of each generalized velocity vector block.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    ###
    # Problem Data
    ###

    D: LinearOperatorsType | None = None
    """
    TODO
    """

    h: wp.array | None = None
    """
    Stack of non-linear generalized forces vectors of each world.\n

    Computed as:
    `h = dt * (w_e + w_gc + w_a)`

    where:
    - `dt` is the simulation time step
    - `w_e` is the stack of per-body purely external wrenches
    - `w_gc` is the stack of per-body gravitational + Coriolis wrenches
    - `w_a` is the stack of per-body jointactuation wrenches

    Construction of this term is optional, as it's contributions are already
    incorporated in the computation of the generalized free-velocity `u_f`.
    It is can be optionally built for analysis or debugging purposes.

    Shape of `(sum_of_num_body_dofs,)` and type :class:`vec6f`.
    """

    u_f: wp.array | None = None
    """
    Stack of unconstrained generalized velocity vectors.\n

    Computed as:
    `u_f = u_minus + dt * M^{-1} @ h`

    where:
    - `u_minus` is the stack of per-body generalized velocities at the beginning of the time step
    - `M^{-1}` is the block-diagonal inverse generalized mass matrix
    - `h` is the stack of non-linear generalized forces vectors

    Shape of `(sum_of_num_body_dofs,)` and type :class:`vec6f`.
    """

    # TODO: Remove and just accumulate in `v_f`
    v_b: wp.array | None = None
    """
    Stack of free-velocity statbilization biases vectors (in constraint-space).\n

    Computed as:
    `v_b = [alpha * inv_dt * r_joints; beta * inv_dt * r_limits; gamma * inv_dt * r_contacts]`

    where:
    - `inv_dt` is the inverse simulation time step
    - `r_joints` is the stack of joint constraint residuals
    - `r_limits` is the stack of limit constraint residuals
    - `r_contacts` is the stack of contact constraint residuals
    - `alpha`, `beta`, `gamma` are the Baumgarte stabilization
        parameters for joints, limits and contacts, respectively

    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    # TODO: Remove and just accumulate in `v_f`
    v_i: wp.array | None = None
    """
    The stack of free-velocity impact biases vector (in constraint-space).\n

    Computed as:
    `v_i = epsilon @ (J_cts @ u_minus)`

    where:
    - `epsilon` is the stack of per-contact restitution coefficients
    - `J_cts` is the constraint Jacobian matrix
    - `u_minus` is the stack of per-body generalized velocities at the beginning of the time step

    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    # TODO: Maybe alias this as `v_star` and accumulate everything there directly
    v_f: wp.array | None = None
    """
    Stack of free-velocity vectors (constraint-space unconstrained velocity).\n

    Computed as:
    ```
    v_f = J_cts @ u_f + v_star
    ```

    where:
    - `J_cts` is the constraint Jacobian matrix
    - `u_f` is the stack of unconstrained generalized velocity vectors
    - `v_star` is the stack of reference velocity vectors

    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    ###
    # Buffers
    ###

    # TODO: Do we need three buffers? Can we just allocate
    # two of max(sum_of_num_body_dofs, sum_of_max_total_cts)?

    _buffer_dofs: wp.array | None = None
    """
    A buffer to store intermediate results in body-DoF space.\n
    This is only allocated if the problem is sparse.\n
    Shape of `(sum_of_num_body_dofs,)` and type :class:`float32`.
    """

    _buffer_cts_A: wp.array | None = None
    """
    A buffer to store intermediate results in constraint-space.\n
    This is only allocated if the problem is sparse.\n
    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    _buffer_cts_B: wp.array | None = None
    """
    A second buffer to store intermediate results in constraint-space.\n
    This is only allocated if the problem is sparse.\n
    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    _buffer_P: wp.array | None = None
    """
    A problem preconditioner buffer to cache the preconditioner array.\n
    This never actually allocated, but is used to store
    references to the active preconditioner array.\n
    Shape of `(sum_of_num_body_dofs,)` and type :class:`float32`.
    """

    ###
    # Operations
    ###

    @override
    def reset(self):
        """
        Resets the problem data to zero (or identity for the preconditioner).
        """
        super().reset()
        self.D.reset()
        self.v_f.zero_()
        self.v_b.zero_()
        self.v_i.zero_()


###
# Interfaces
###


class DualProblem(ConstrainedDynamicsProblem):
    """
    A container to manage data and operate a forward dynamics dual problem.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SystemJacobiansType | None = None,
        configs: list[ConstrainedDynamicsConfig] | ConstrainedDynamicsConfig | None = None,
        linear_solver_type: type[LinearSolverType] | None = None,
        linear_solver_kwargs: dict[str, Any] | None = None,
        device: wp.DeviceLike = None,
        compute_h: bool = False,
    ):
        """
        TODO
        """
        self._dual_data: DualProblemData | None = None
        """The dual forward dynamics-specific problem data container."""

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
            compute_h=compute_h,
        )

    ###
    # Properties
    ###

    @property
    def data(self) -> DualProblemData:
        """
        Returns the dual problem data container.
        """
        if self._dual_data is None:
            raise ValueError("DualProblemData is not available. Ensure that `finalize()` has been called.")
        return self._dual_data

    ###
    # Implementation
    ###

    @override
    def _get_sum_max_problem_dims_impl(self) -> tuple[int, int]:
        # NOTE: The dual problem is defined solely in constraint space
        world_max_cts = get_max_constraints_per_world(self._system.model, self._system.limits, self._system.contacts)
        return sum(world_max_cts), max(world_max_cts)

    @override
    def _finalize_impl(self, **kwargs: dict[str, Any]) -> DualProblemData:
        # Construct the dual problem offset indices based on the problem dimensions
        x_size = [0] * (self.info.num_worlds)
        x_start = [0] * (self.info.num_worlds + 1)
        x_start[-1] = self.info.sum_of_max_problem_dims
        for w in range(1, self.info.num_worlds + 1):
            x_size[w - 1] = self.info.world_max_total_cts[w]
            x_start[w] = x_start[w - 1] + self.info.world_max_total_cts[w]
        if x_start[-1] != self.info.sum_of_max_problem_dims:
            raise ValueError(
                f"The computed dual problem offset indices do not match the total max problem dimension. "
                f"Expected last entry to be {self.info.sum_of_max_problem_dims} but got {x_start[-1]}."
            )
        if sum(x_size) != self.info.sum_of_max_problem_dims:
            raise ValueError(
                f"The computed dual problem max sizes do not match the total max problem dimension. "
                f"Expected sum of sizes to be {self.info.sum_of_max_problem_dims} but got {sum(x_size)}."
            )

        # Construct the dual system linear operator based on the constrained system
        if self.sparse:
            D_operator = BlockSparseLinearOperators()  # TODO: CONSTRUCT THIS BASED ON PROBLEM INFO
        else:
            D_operator = DenseLinearOperators()  # TODO: CONSTRUCT THIS BASED ON PROBLEM INFO

        # Retrieve dual-problem-specific arguments from kwargs
        linear_solver_type: type[LinearSolverType] = kwargs.pop("linear_solver_type", None)
        linear_solver_kwargs: dict[str, Any] = kwargs.pop("linear_solver_kwargs", {})
        compute_h: bool = kwargs.pop("compute_h", False)

        # Construct the dual system data container
        with wp.ScopedDevice(self.device):
            # TODO: Allocate `v_f` and alias as `v_star`
            self._dual_data = DualProblemData(
                x_start=wp.array(x_start, dtype=int32),
                D=D_operator,
                v_f=wp.zeros(shape=(self.info.sum_of_max_problem_dims,), dtype=float32),
                h=wp.zeros(shape=(self.system.model.size.sum_of_num_bodies,), dtype=vec6f) if compute_h else None,
                v_star=wp.zeros(shape=(self.info.sum_of_max_total_cts,), dtype=float32),
                mu=wp.zeros(shape=(self.info.sum_of_max_contacts,), dtype=float32),
                P=wp.ones(shape=(self.info.sum_of_max_problem_dims,), dtype=float32),
            )

        # Create the linear system solver instance if a supported type is specified
        if linear_solver_type is not None and issubclass(linear_solver_type, LinearSolverType):
            self._solver = linear_solver_type(operator=self._dual_data.D, device=self.device, **linear_solver_kwargs)
        else:
            raise ValueError("A valid linear solver type must be specified for the dual problem.")

        # Return the dual system data container to the base class constructor
        return self._dual_data

    @override
    def _reset_impl(self):
        # NOTE: The default implementation is a no-op since
        # not all CFD problem formulations may require resetting.
        pass

    @override
    def _build_impl(self, reset_to_zero: bool = True):
        # Build the Delassus operator
        if self.sparse:
            # Sparse delassus is built implicitly within the matvec
            # operation, so we don't need to explicitly build it here.
            pass
        else:
            # Optionally initialize the Delassus matrix to zero
            if reset_to_zero:
                self._dual_data.D.zero()

            # Build the Delassus matrix parallelized element-wise
            wp.launch(
                kernel=_build_dense_delassus_elementwise,
                dim=(self._info.num_worlds, self._dual_data.D.info.max_of_max_size),
                inputs=[
                    # Inputs:
                    self.system.model.info.num_bodies,
                    self.system.model.info.bodies_offset,
                    self.system.model.bodies.inv_m_i,
                    self.system.data.bodies.inv_I_i,
                    self.system.jacobians.data.J_cts_offsets,
                    self.system.jacobians.data.J_cts_data,
                    self._info.constraints_count,
                    self._dual_data.D.start,
                    # Outputs:
                    self._dual_data.D.data,
                ],
            )

        # Optionally also build the non-linear generalized force vector
        if self._dual_data.h is not None:
            self._build_nonlinear_generalized_force()

        # Build the generalized free-velocity vector
        self._build_generalized_free_velocity()

        # Build the free-velocity bias terms
        self._build_free_velocity_bias()

        # Build the free-velocity vector
        if self.sparse:
            wp.copy(self._dual_data.v_f, self._dual_data.v_b)
            J_cts = self.system.jacobians._J_cts.bsm
            wp.launch(
                _build_free_velocity_sparse,
                dim=(self._info.num_worlds, J_cts.max_of_num_nzb),
                inputs=[
                    # Inputs:
                    self.system.model.info.bodies_offset,
                    self.system.data.bodies.u_i,
                    J_cts.num_nzb,
                    J_cts.nzb_start,
                    J_cts.nzb_coords,
                    J_cts.nzb_values,
                    self._info.constraints_start,
                    self._dual_data.u_f,
                    self._dual_data.v_i,
                    # Outputs:
                    self._dual_data.v_f,
                ],
            )
        else:
            wp.launch(
                _build_free_velocity_dense,
                dim=(self._info.num_worlds, self._info.max_of_max_total_cts),
                inputs=[
                    # Inputs:
                    self.system.model.info.num_bodies,
                    self.system.model.info.bodies_offset,
                    self.system.data.bodies.u_i,
                    self.system.jacobians.data.J_cts_offsets,
                    self.system.jacobians.data.J_cts_data,
                    self._info.constraints_start,
                    self._info.constraints_count,
                    self._dual_data.u_f,
                    self._dual_data.v_b,
                    self._dual_data.v_i,
                    # Outputs:
                    self._dual_data.v_f,
                ],
            )

    @override
    def _build_preconditioner_impl(self):
        """
        Builds the diagonal preconditioner 'P' according to the current Delassus operator.
        """
        if self.sparse:
            self._delassus.diagonal(self._data.P)
            wp.launch(
                _build_dual_preconditioner_sparse,
                dim=(self._info.num_worlds, self._info.max_of_max_total_cts),
                inputs=[
                    # Inputs:
                    self._info.configs,
                    self._info.constraints_start,
                    self._info.constraints_count,
                    self._info.njc,
                    self._info.nl,
                    # Outputs:
                    self._dual_data.P,
                ],
            )
        else:
            wp.launch(
                _build_dual_preconditioner_dense,
                dim=(self._info.num_worlds, self._info.max_of_max_total_cts),
                inputs=[
                    # Inputs:
                    self._info.configs,
                    self._info.constraints_start,
                    self._info.constraints_count,
                    self._info.njc,
                    self._info.nl,
                    self._dual_data.D.dense.start,
                    self._dual_data.D.dense.data,
                    # Outputs:
                    self._dual_data.P,
                ],
            )

    @override
    def _apply_preconditioner_impl(self):
        """
        Applies the diagonal preconditioner 'P' to the
        Delassus operator 'D' and free-velocity vector `v_f`.
        """
        if self.sparse:
            # Preconditioner has already been connected to appropriate array
            pass
        else:
            wp.launch(
                _apply_dual_preconditioner_to_matrix_dense,
                dim=(self._info.num_worlds, self._dual_data.D.info.max_of_max_size),
                inputs=[
                    # Inputs:
                    self._info.constraints_start,
                    self._info.constraints_count,
                    self._dual_data.P,
                    self._dual_data.D.dense.start,
                    # Outputs:
                    self._dual_data.D.dense.data,
                ],
            )

        wp.launch(
            _apply_dual_preconditioner_to_vector,
            dim=(self._info.num_worlds, self._info.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self._info.constraints_start,
                self._info.constraints_count,
                self._dual_data.P,
                # Outputs:
                self._dual_data.v_f,
            ],
        )

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
        """
        Performs a BLAS-like generalized sparse matrix-transpose-vector product
        `x = alpha * D^T @ y + beta * x`.

        Note:
            Since the Delassus matrix is symmetric, this is equivalent to `gemv` with swapped arguments.
        """
        self._gemv_impl(y, x, world_mask, alpha, beta)

    ###
    # Internals
    ###

    def _apply_regularization(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0):
        """
        Adds diagonal regularization to the matrix-vector product result.

        If regularization values have been set via `set_regularization()`, this method computes
        `y += alpha * diag(sigma) @ x`, where `sigma` contains the regularization.

        Args:
            x (wp.array): Input vector to be scaled by the diagonal regularization matrix.
                Shape `(sum_of_max_total_cts,)` and type :class:`float32`.
            y (wp.array): Output vector to which the regularization contribution is added.
                Shape `(sum_of_max_total_cts,)` and type :class:`float32`.
            world_mask (wp.array): Array of integers indicating which worlds to process (0 = skip).
                Shape `(num_worlds,)` and type :class:`int32`.
            alpha (float, optional): Scalar multiplier for the regularization term. Defaults to 1.0.
        """
        # Skip if no regularization is set
        if self._data.sigma is None:
            return

        if self.sparse:
            wp.launch(
                kernel=_add_matrix_diag_product,
                dim=(self._info.num_worlds, self._info.max_of_max_total_cts),
                inputs=[
                    self._info.constraints_count,
                    self._data.D.bsm.row_start,
                    self._data.sigma,
                    x,
                    y,
                    alpha,
                    world_mask,
                ],
                device=self._device,
            )
        else:
            raise NotImplementedError("Regularization is only implemented for the sparse formulation at the moment.")

    def _apply_preconditioning(self, x: wp.array, world_mask: wp.array):
        """
        Applies diagonal preconditioning to a vector.

        If a preconditioner has been set via `set_preconditioner()`, this method computes
        `x = diag(P) @ x`, where `P` contains the diagonal elements of a preconditioning matrix.

        Args:
            x (wp.array): Vector to be scaled by the diagonal preconditioning matrix.
                Shape `(sum_of_max_total_cts,)` and type :class:`float32`.
            world_mask (wp.array): Array of integers indicating which worlds to process (0 = skip).
                Shape `(num_worlds,)` and type :class:`int32`.
        """
        # Skip if no preconditioner is set
        if self._data.P is None:
            return

        if self.sparse:
            wp.launch(
                kernel=_set_matrix_diag_product,
                dim=(self._info.num_worlds, self._info.max_of_max_total_cts),
                inputs=[
                    self._info.constraints_count,
                    self._data.D.bsm.row_start,
                    self._data.P,
                    x,
                    world_mask,
                ],
                device=self._device,
            )
        else:
            raise NotImplementedError("Preconditioning is only implemented for the sparse formulation at the moment.")

    def _build_nonlinear_generalized_force(self):
        """
        Builds the nonlinear generalized force vector `h`.
        """
        wp.launch(
            _build_nonlinear_generalized_force,
            dim=self.system.model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                self.system.model.time.dt,
                self.system.model.gravity.vector,
                self.system.model.bodies.wid,
                self.system.model.bodies.m_i,
                self.system.data.bodies.u_i,
                self.system.data.bodies.I_i,
                self.system.data.bodies.w_e_i,
                self.system.data.bodies.w_a_i,
                # Outputs:
                self._dual_data.h,
            ],
        )

    def _build_generalized_free_velocity(self):
        """
        Builds the generalized free-velocity vector (i.e. unconstrained) `u_f`.
        """
        wp.launch(
            _build_generalized_free_velocity,
            dim=self.system.model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                self.system.model.time.dt,
                self.system.model.gravity.vector,
                self.system.model.bodies.wid,
                self.system.model.bodies.m_i,
                self.system.model.bodies.inv_m_i,
                self.system.data.bodies.u_i,
                self.system.data.bodies.I_i,
                self.system.data.bodies.inv_I_i,
                self.system.data.bodies.w_e_i,
                self.system.data.bodies.w_a_i,
                # Outputs:
                self._dual_data.u_f,
            ],
        )

    def _build_free_velocity_bias(self):
        """
        Builds the free-velocity bias vector `v_b`.
        """

        if self.system.model.size.sum_of_num_joints > 0:
            wp.launch(
                _build_free_velocity_bias_joints,
                dim=self.system.model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
                    self.system.model.info.joint_cts_offset,
                    self.system.model.time.inv_dt,
                    self.system.model.joints.wid,
                    self.system.model.joints.num_cts,
                    self.system.model.joints.cts_offset,
                    self.system.data.joints.r_j,
                    self._info.configs,
                    self._info.constraints_start,
                    # Outputs:
                    self._dual_data.v_b,
                ],
            )

        if self.system.limits is not None and self.system.limits.model_max_limits_host > 0:
            wp.launch(
                _build_free_velocity_bias_limits,
                dim=self.system.limits.model_max_limits_host,
                inputs=[
                    # Inputs:
                    self.system.model.time.inv_dt,
                    self.system.data.info.limit_cts_group_offset,
                    self.system.limits.model_max_limits_host,
                    self.system.limits.model_active_limits,
                    self.system.limits.wid,
                    self.system.limits.lid,
                    self.system.limits.r_q,
                    self._info.configs,
                    self._info.constraints_start,
                    # Outputs:
                    self._dual_data.v_b,
                ],
            )

        if self.system.contacts is not None and self.system.contacts.model_max_contacts_host > 0:
            wp.launch(
                _build_free_velocity_bias_contacts,
                dim=self.system.contacts.model_max_contacts_host,
                inputs=[
                    # Inputs:
                    self.system.model.time.inv_dt,
                    self.system.model.info.contacts_offset,
                    self.system.data.info.contact_cts_group_offset,
                    self.system.contacts.model_max_contacts_host,
                    self.system.contacts.model_active_contacts,
                    self.system.contacts.wid,
                    self.system.contacts.cid,
                    self.system.contacts.gapfunc,
                    self.system.contacts.material,
                    self._info.configs,
                    self._info.constraints_start,
                    # Outputs:
                    self._dual_data.v_b,
                    self._dual_data.v_i,
                    self._dual_data.mu,
                ],
            )

    def _build_free_velocity(self):
        """
        Builds the free-velocity vector `v_f`.
        """
        wp.launch(
            _build_free_velocity_dense,
            dim=(self._info.num_worlds, self._info.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self.system.model.info.num_bodies,
                self.system.model.info.bodies_offset,
                self.system.data.bodies.u_i,
                self.system.jacobians.data.J_cts_offsets,
                self.system.jacobians.data.J_cts_data,
                self._info.constraints_start,
                self._info.constraints_count,
                self._dual_data.u_f,
                self._dual_data.v_b,
                self._dual_data.v_i,
                # Outputs:
                self._dual_data.v_f,
            ],
        )
