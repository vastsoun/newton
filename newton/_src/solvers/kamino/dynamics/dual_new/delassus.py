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

from typing import Any

import warp as wp

from ...core.model import Model, ModelData, ModelSize
from ...core.types import FloatType, float32, int32
from ...geometry.contacts import Contacts
from ...kinematics.constraints import get_max_constraints_per_world
from ...kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ...kinematics.limits import Limits
from ...linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LinearSolverType
from ...linalg.linear import IterativeSolver
from ...linalg.sparse_operator import BlockSparseLinearOperators
from .kernels import (
    _add_matrix_diag_product,
    _build_dense_delassus_elementwise,
    _build_dual_preconditioner_dense,
    _build_dual_preconditioner_sparse,
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
    "DelassusOperatorSparse",
    "DelassusOperatorDense",
]


###
# Interfaces
###


class DelassusOperatorDense:
    """
    A container to represent the Delassus matrix operator.
    """

    def __init__(
        self,
        model: Model | None = None,
        data: ModelData | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        solver: LinearSolverType = None,
        solver_kwargs: dict[str, Any] | None = None,
        device: wp.DeviceLike = None,
    ):
        """
        Creates a Delassus operator for the given model, limits and contacts containers.

        This class also supports deferred allocation, i.e. it can be initialized without
        a model, limits, or contacts, and the allocation can be performed later using the
        `allocate` method. This is useful for scenarios where the model or constraints are
        not known at the time of Delassus operator creation, but will be available later.

        The dimension of a Delassus matrix is defined as the sum over active
        joint, limit, and contact constraints, and the maximum dimension is
        the maximum number of constraints that can be active in each world.

        Args:
            model (Model): The model container for which the Delassus operator is built.
            data (ModelData, optional): The model data container holding the state info and data.
            limits (Limits, optional): The container holding the allocated joint-limit data.
            contacts (Contacts, optional): The container holding the allocated contacts data.
            device (wp.DeviceLike, optional): The device identifier for the Delassus operator. Defaults to None.
            factorizer (CholeskyFactorizer, optional): An optional Cholesky factorization object. Defaults to None.
        """
        # Declare and initialize the host-side cache of the necessary memory allocations
        self._num_worlds: int = 0
        self._model_maxdims: int = 0
        self._model_maxsize: int = 0
        self._world_maxdims: list[int] = []
        self._world_maxsize: list[int] = []
        self._max_of_max_total_D_size: int = 0

        # Cache the requested device
        self._device: wp.DeviceLike = device

        # Declare the model size cache
        self._size: ModelSize | None = None

        # Initialize the Delassus data container
        self._operator: DenseLinearOperatorData | None = None

        # Declare the optional Cholesky factorization
        self._solver: LinearSolverType | None = None

        # Allocate the Delassus operator data if at least the model is provided
        if model is not None:
            self.finalize(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                solver=solver,
                solver_kwargs=solver_kwargs,
                device=device,
            )

    @property
    def num_worlds(self) -> int:
        """
        Returns the number of worlds represented by the Delassus operator.
        This is equal to the number of matrix blocks contained in the flat array.
        """
        return self._num_worlds

    @property
    def num_maxdims(self) -> int:
        """
        Returns the maximum dimension of the Delassus matrix across all worlds.
        This is the sum of per matrix block maximum dimensions.
        """
        return self._model_maxdims

    @property
    def num_maxsize(self) -> int:
        """
        Returns the maximum size of the Delassus matrix across all worlds.
        This is the sum over the sizes of all matrix blocks.
        """
        return self._model_maxsize

    @property
    def operator(self) -> DenseLinearOperatorData:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator

    @property
    def solver(self) -> LinearSolverType:
        """
        The linear solver object for the Delassus operator.
        This is used to perform the factorization of the Delassus matrix.
        """
        return self._solver

    @property
    def info(self) -> DenseSquareMultiLinearInfo:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator.info

    @property
    def D(self) -> wp.array:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator.mat

    def finalize(
        self,
        model: Model,
        data: ModelData,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        solver: LinearSolverType = None,
        device: wp.DeviceLike = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Allocates the Delassus operator with the specified dimensions and device.

        Args
        ----
            dims (List[int]): The dimensions of the Delassus matrix for each world.
            device (wp.DeviceLike, optional): The device identifier for the Delassus operator. Defaults to None.
            factorizer (CholeskyFactorizer, optional): An optional Cholesky factorization object. Defaults to None.
        """

        # Ensure the model container is valid
        if model is None:
            raise ValueError("A model container of type `Model` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, Model):
            raise ValueError("Invalid model provided. Must be an instance of `Model`.")

        # Ensure the data container is valid if provided
        if data is None:
            raise ValueError("A data container of type `ModelData` must be provided to allocate the Delassus operator.")
        elif not isinstance(data, ModelData):
            raise ValueError("Invalid data container provided. Must be an instance of `ModelData`.")

        # Ensure the limits container is valid if provided
        if limits is not None:
            if not isinstance(limits, Limits):
                raise ValueError("Invalid limits container provided. Must be an instance of `Limits`.")

        # Ensure the contacts container is valid if provided
        if contacts is not None:
            if not isinstance(contacts, Contacts):
                raise ValueError("Invalid contacts container provided. Must be an instance of `Contacts`.")

        # Capture reference to the model size
        self._size = model.size

        # Extract required maximum number of constraints for each world
        maxdims = get_max_constraints_per_world(model, limits, contacts)

        # Update the allocation meta-data the specified constraint dimensions
        self._num_worlds = model.size.num_worlds
        self._world_dims = maxdims
        self._world_size = [maxdims[i] * maxdims[i] for i in range(self._num_worlds)]
        self._model_maxdims = sum(self._world_dims)
        self._model_maxsize = sum(self._world_size)
        self._max_of_max_total_D_size = max(self._world_size)

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self._device = device

        # Construct the Delassus operator data structure
        self._operator = DenseLinearOperatorData()
        self._operator.info = DenseSquareMultiLinearInfo()
        self._operator.mat = wp.zeros(shape=(self._model_maxsize,), dtype=float32, device=self._device)
        if (model.info is not None) and (data.info is not None):
            mat_offsets = [0] + [sum(self._world_size[:i]) for i in range(1, self._num_worlds + 1)]
            self._operator.info.assign(
                maxdim=model.info.max_total_cts,
                dim=data.info.num_total_cts,
                vio=model.info.total_cts_offset,
                mio=wp.array(mat_offsets[: self._num_worlds], dtype=int32, device=self._device),
                dtype=float32,
                device=self._device,
            )
        else:
            self._operator.info.finalize(dimensions=maxdims, dtype=float32, itype=int32, device=self._device)

        # Optionally initialize the linear system solver if one is specified
        if solver is not None:
            if not issubclass(solver, LinearSolverType):
                raise ValueError("Invalid solver provided. Must be a subclass of `LinearSolverType`.")
            solver_kwargs = solver_kwargs or {}
            self._solver = solver(operator=self._operator, device=self._device, **solver_kwargs)

    def zero(self):
        """
        Sets all values of the Delassus matrix to zero.
        This is useful for resetting the operator before recomputing it.
        """
        self._operator.mat.zero_()

    def build(self, model: Model, data: ModelData, jacobians: DenseSystemJacobians, reset_to_zero: bool = True):
        """
        Builds the Delassus matrix using the provided Model, ModelData, and constraint Jacobians.

        Args:
            model (Model): The model for which the Delassus operator is built.
            data (ModelData): The current data of the model.
            reset_to_zero (bool, optional): If True (default), resets the Delassus matrix to zero before building.

        Raises:
            ValueError: If the model, data, or Jacobians are not valid.
            ValueError: If the Delassus matrix is not allocated.
        """
        # Ensure the model is valid
        if model is None or not isinstance(model, Model):
            raise ValueError("A valid model of type `Model` must be provided to build the Delassus operator.")

        # Ensure the data is valid
        if data is None or not isinstance(data, ModelData):
            raise ValueError("A valid model data of type `ModelData` must be provided to build the Delassus operator.")

        # Ensure the Jacobians are valid
        if jacobians is None or not isinstance(jacobians, DenseSystemJacobians):
            raise ValueError(
                "A valid Jacobians data container of type `DenseSystemJacobians` "
                "must be provided to build the Delassus operator."
            )

        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call finalize() first.")

        # Initialize the Delassus matrix to zero
        if reset_to_zero:
            self.zero()

        # Build the Delassus matrix parallelized element-wise
        wp.launch(
            kernel=_build_dense_delassus_elementwise,
            dim=(self._size.num_worlds, self._max_of_max_total_D_size),
            inputs=[
                # Inputs:
                model.info.num_bodies,
                model.info.bodies_offset,
                model.bodies.inv_m_i,
                data.bodies.inv_I_i,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                self._operator.info.dim,
                self._operator.info.mio,
                # Outputs:
                self._operator.mat,
            ],
        )

    def regularize(self, eta: wp.array):
        """
        Adds diagonal regularization to each matrix block of the Delassus operator.

        Args:
            eta (wp.array): The regularization values to add to the diagonal of each matrix block.
            This should be an array of shape `(maxdims,)` and type :class:`float32`.
            Each value in `eta` corresponds to the regularization along each constraint.
        """
        wp.launch(
            kernel=_regularize_dense_delassus_diagonal,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[self._operator.info.dim, self._operator.info.vio, self._operator.info.mio, eta, self._operator.mat],
        )

    def compute(self, reset_to_zero: bool = True):
        """
        Runs Delassus pre-computation operations in preparation for linear systems solves.

        Depending on the configured solver type, this may perform different
        pre-computation, e.g. Cholesky factorization for direct solvers.

        Args:
            reset_to_zero (bool):
                If True, resets the Delassus matrix to zero.\n
                This is useful for ensuring that the matrix is in a clean state before pre-computation.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call finalize() first.")

        # Ensure the solver is available if pre-computation is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Optionally initialize the factorization matrix before factorizing
        if reset_to_zero:
            self._solver.reset()

        # Perform the Cholesky factorization
        self._solver.compute(A=self._operator.mat)

    def solve(self, v: wp.array, x: wp.array):
        """
        Solves the linear system D * x = v using the Cholesky factorization.

        Args:
            v (wp.array): The right-hand side vector of the linear system.
            x (wp.array): The array to hold the solution.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the factorizer is not available.
            ValueError: If a factorizer has not been configured set.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call finalize() first.")

        # Ensure the solver is available if solving is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Solve the linear system using the factorized matrix
        return self._solver.solve(b=v, x=x)

    def solve_inplace(self, x: wp.array):
        """
        Solves the linear system D * x = v in-place.\n
        This modifies the input array x to contain the solution assuming it is initialized as x=v.

        Args:
            x (wp.array): The array to hold the solution. It should be initialized with the right-hand side vector v.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the factorizer is not available.
            ValueError: If a factorizer has not been configured set.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call finalize() first.")

        # Ensure the solvers is available if solving in-place is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Solve the linear system in-place
        return self._solver.solve_inplace(x=x)


class DelassusOperatorSparse(BlockSparseLinearOperators):
    """
    TODO
    """

    def __init__(
        self,
        model: Model | None = None,
        data: ModelData | None = None,
        jacobians: SparseSystemJacobians | None = None,
        linear_solver: LinearSolverType = None,
        linear_solver_kwargs: dict[str, Any] | None = None,
        device: wp.DeviceLike = None,
    ):
        """
        TODO
        """
        super().__init__()

        # self.bsm represents the constraint Jacobian
        self._model: Model | None = None
        self._data: ModelData | None = None

        # Problem info object
        # TODO: Create more general info object independent of dense matrix representation
        self._info: DenseSquareMultiLinearInfo | None = None

        # Cache the requested device
        self._device: wp.DeviceLike = device

        # Declare the optional (iterative) solver
        self._solver: LinearSolverType | None = None

        # TODO
        self._preconditioner: wp.array | None = None
        self._sigma: wp.array | None = None

        # Buffers
        self._vec_temp_body_space: wp.array | None = None
        self._vec_temp_cts_space_A: wp.array | None = None
        self._vec_temp_cts_space_B: wp.array | None = None

        # Allocate the Delassus operator data if at least the model is provided
        if model is not None:
            self.finalize(
                model=model,
                data=data,
                jacobians=jacobians,
                linear_solver=linear_solver,
                linear_solver_kwargs=linear_solver_kwargs,
                device=device,
            )

    def finalize(
        self,
        model: Model,
        data: ModelData,
        jacobians: SparseSystemJacobians | None = None,
        linear_solver_type: LinearSolverType = None,
        linear_solver_kwargs: dict[str, Any] | None = None,
        device: wp.DeviceLike = None,
    ):
        """
        TODO
        """
        # Ensure the model container is valid
        if model is None:
            raise ValueError("A model container of type `Model` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, Model):
            raise ValueError("Invalid model provided. Must be an instance of `Model`.")

        # Ensure the data container is valid if provided
        if data is None:
            raise ValueError("A data container of type `ModelData` must be provided to allocate the Delassus operator.")
        elif not isinstance(data, ModelData):
            raise ValueError("Invalid data container provided. Must be an instance of `ModelData`.")

        # Ensure the solver is iterative if provided
        if linear_solver_type is not None and not issubclass(linear_solver_type, IterativeSolver):
            raise ValueError("Invalid solver provided. Must be a subclass of `IterativeSolver`.")

        self._model = model
        self._data = data

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self._device = device

        self._info = DenseSquareMultiLinearInfo()
        if model.info is not None and data.info is not None:
            self._info.assign(
                maxdim=model.info.max_total_cts,
                dim=data.info.num_total_cts,
                vio=model.info.total_cts_offset,
                mio=wp.empty((self.num_matrices,), dtype=int32, device=self._device),
                dtype=float32,
                device=self._device,
            )
        else:
            self._info.finalize(
                dimensions=model.info.max_total_cts.numpy(),
                dtype=float32,
                itype=int32,
                device=self._device,
            )

        # TODO: Why not just capture references here?
        self._active_rows = wp.array(
            dtype=wp.int32,
            shape=(self._model.size.num_worlds,),
            ptr=self._data.info.num_total_cts.ptr,
            copy=False,
        )
        self._active_cols = wp.array(
            dtype=wp.int32,
            shape=(self._model.size.num_worlds,),
            ptr=self._data.info.num_total_cts.ptr,
            copy=False,
        )

        # Initialize temporary memory
        self._vec_temp_body_space = wp.empty(
            (self._model.size.sum_of_num_body_dofs,), dtype=float32, device=self._device
        )

        # Assign Jacobian if specified
        if jacobians is not None:
            self.bsm = jacobians._J_cts.bsm

        # Optionally initialize the iterative linear system solver if one is specified
        if linear_solver_type is not None:
            linear_solver_kwargs = linear_solver_kwargs or {}
            self._solver = linear_solver_type(operator=self, device=self._device, **linear_solver_kwargs)

    def set_regularization(self, eta: wp.array | None):
        """
        Adds diagonal regularization to each matrix block of the Delassus operator, replacing any
        previously set regularization.

        The regularized Delassus matrix is defined as D = J @ M^-1 @ J^T + diag(eta)

        Args:
            eta (wp.array): The regularization values to add to the diagonal of each matrix block,
                with each value corresponding to the regularization along a constraint.
                This should be an array of shape `(sum_of_max_total_cts,)` and type :class:`float32`,
                or `None` if no regularization should be applied.
        """
        self._sigma = eta

    def set_preconditioner(self, preconditioner: wp.array | None):
        """
        Sets the diagonal preconditioner for the Delassus operator, replacing any previously set
        preconditioner.

        With preconditioning, the effective operator becomes P @ D @ P, where P = diag(preconditioner).

        Args:
            preconditioner (wp.array): The diagonal preconditioner values to apply to the Delassus
                operator, with each value corresponding to a constraint. This should be an array of
                shape `(sum_of_max_total_cts,)` and type :class:`float32`, or `None` to disable
                preconditioning.
        """
        self._preconditioner = preconditioner

        # Initialize memory to store intermediate results with preconditioning
        if self._preconditioner is not None:
            if self._vec_temp_cts_space_A is None:
                self._vec_temp_cts_space_A = wp.empty(
                    (self._model.size.sum_of_max_total_cts,), dtype=float32, device=self._device
                )
            if self._vec_temp_cts_space_B is None:
                self._vec_temp_cts_space_B = wp.empty(
                    (self._model.size.sum_of_max_total_cts,), dtype=float32, device=self._device
                )

    def diagonal(self, diag: wp.array):
        """Stores the diagonal of the Delassus matrix in the given array.

        Note:
            This uses the diagonal of the pure Delassus matrix, without any regularization or
            preconditioning.

        Args:
            diag (wp.array): Output vector for the Delassus matrix diagonal entries.
                Shape `(sum_of_max_total_cts,)` and type :class:`float32`.
        """
        if self._model is None or self._data is None:
            raise RuntimeError("Model and data must be assigned before computing diagonal.")
        if self.bsm is None:
            raise RuntimeError("Jacobian must be assigned before computing diagonal.")

        diag.zero_()

        # Launch kernel over all non-zero blocks
        wp.launch(
            kernel=_compute_delassus_diagonal_sparse,
            dim=(self._model.size.num_worlds, self.bsm.max_of_num_nzb),
            inputs=[
                self._model.info.bodies_offset,
                self._model.bodies.inv_m_i,
                self._data.bodies.inv_I_i,
                self.bsm.nzb_start,
                self.bsm.num_nzb,
                self.bsm.nzb_coords,
                self.bsm.nzb_values,
                self.bsm.row_start,
                diag,
            ],
            device=self._device,
        )

    def compute(self, reset_to_zero: bool = True):
        """
        Runs Delassus pre-computation operations in preparation for linear systems solves.

        Depending on the configured solver type, this may perform different pre-computation.

        Args:
            reset_to_zero (bool):
                If True, resets the Delassus matrix to zero.\n
                This is useful for ensuring that the matrix is in a clean state before pre-computation.
        """
        # Ensure that `finalize()` was called
        if self._info is None:
            raise ValueError("Data structure is not allocated. Call finalize() first.")

        # Ensure the Jacobian is set
        if self.bsm is None:
            raise ValueError("Jacobian matrix is not set. Call assign() first.")

        # Ensure the solver is available if pre-computation is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Optionally initialize the solver
        if reset_to_zero:
            self._solver.reset()

        # Perform the pre-computation
        self._solver.compute()

    def solve(self, v: wp.array, x: wp.array):
        """
        Solves the linear system D * x = v using the assigned solver.

        Args:
            v (wp.array): The right-hand side vector of the linear system.
            x (wp.array): The array to hold the solution.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the solver is not available.
        """
        # Ensure that `finalize()` was called
        if self._info is None:
            raise ValueError("Data structure is not allocated. Call finalize() first.")

        # Ensure the Jacobian is set
        if self.bsm is None:
            raise ValueError("Jacobian matrix is not set. Call assign() first.")

        # Ensure the solver is available
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Solve the linear system
        return self._solver.solve(b=v, x=x)

    def solve_inplace(self, x: wp.array):
        """
        Solves the linear system D * x = v in-place.\n
        This modifies the input array x to contain the solution assuming it is initialized as x=v.

        Args:
            x (wp.array): The array to hold the solution. It should be initialized with the right-hand side vector v.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the solver is not available.
        """
        # Ensure that `finalize()` was called
        if self._info is None:
            raise ValueError("Data structure is not allocated. Call finalize() first.")

        # Ensure the Jacobian is set
        if self.bsm is None:
            raise ValueError("Jacobian matrix is not set. Call assign() first.")

        # Ensure the solver is available if pre-computation is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Solve the linear system in-place
        return self._solver.solve_inplace(x=x)

    ###
    # Properties
    ###

    @property
    def info(self) -> DenseSquareMultiLinearInfo | None:
        """
        Returns the info object for the Delassus problem dimensions and sizes.
        """
        return self._info

    @property
    def num_matrices(self) -> int:
        """
        Returns the number of matrices represented by the Delassus operator.
        """
        return self._model.size.num_worlds

    @property
    def max_of_max_dims(self) -> tuple[int, int]:
        """
        Returns the maximum dimension of any Delassus matrix across all worlds.
        """
        max_jac_rows = self._model.size.max_of_max_total_cts
        return (max_jac_rows, max_jac_rows)

    @property
    def sum_of_max_dims(self) -> int:
        """
        Returns the sum of maximum dimensions of the Delassus matrix across all worlds.
        """
        return self._model.size.sum_of_max_total_cts

    @property
    def dtype(self) -> FloatType:
        return self._info.dtype

    @property
    def device(self) -> wp.DeviceLike:
        return self._model.device

    ###
    # Operations
    ###

    def matvec(self, x: wp.array, y: wp.array, world_mask: wp.array):
        """
        Performs the sparse matrix-vector product `y = D @ x`, applying regularization and
        preconditioning if configured.
        """
        if self.Ax_op is None:
            raise RuntimeError("No `A@x` operator has been assigned.")
        if self.ATy_op is None:
            raise RuntimeError("No `A^T@y` operator has been assigned.")

        v = self._vec_temp_body_space
        v.zero_()

        if self._preconditioner is None:
            # Compute first Jacobian matrix-vector product: v <- J^T @ x
            self.ATy_op(self.bsm, x, v, world_mask)

            # Multiply by inverse mass matrix (in-place): v <- M^-1 @ v
            self._apply_inverse_mass_matrix(v, world_mask)

            # Compute second Jacobian matrix-vector product: y <- J @ v
            self.Ax_op(self.bsm, v, y, world_mask)

            # Add regularization, if provided: y <- y + diag(eta) @ x
            self._apply_regularization(x, y, world_mask)

        else:
            x_preconditioned = self._vec_temp_cts_space_A

            # Apply preconditioning to input vector: x_p <- diag(P) @ x
            wp.copy(x_preconditioned, x)
            self._apply_preconditioning(x_preconditioned, world_mask)

            # Compute first Jacobian matrix-vector product: v <- J^T @ x_p
            self.ATy_op(self.bsm, x_preconditioned, v, world_mask)

            # Multiply by inverse mass matrix (in-place): v <- M^-1 @ v
            self._apply_inverse_mass_matrix(v, world_mask)

            # Compute second Jacobian matrix-vector product: y <- J @ v
            self.Ax_op(self.bsm, v, y, world_mask)

            # Apply preconditioning to output vector: y <- diag(P) @ y
            self._apply_preconditioning(y, world_mask)

            # Add regularization, if provided: y <- y + diag(eta) @ x
            self._apply_regularization(x, y, world_mask)

    def matvec_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array):
        """
        Performs the sparse matrix-transpose-vector product `x = D^T @ y`.

        Note:
            Since the Delassus matrix is symmetric, this is equivalent to `matvec`.
        """
        self.matvec(x, y, world_mask)

    def gemv(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """
        Performs a BLAS-like generalized sparse matrix-vector product `y = alpha * D @ x + beta * y`,
        applying regularization and preconditioning if configured.
        """
        if self.gemv_op is None:
            raise RuntimeError("No BLAS-like `GEMV` operator has been assigned.")
        if self.ATy_op is None:
            raise RuntimeError("No `A^T@y` operator has been assigned.")

        v = self._vec_temp_body_space
        v.zero_()

        if self._preconditioner is None:
            # Compute first Jacobian matrix-vector product: v <- J^T @ x
            self.ATy_op(self.bsm, x, v, world_mask)

            # Multiply by inverse mass matrix (in-place): v <- M^-1 @ v
            self._apply_inverse_mass_matrix(v, world_mask)

            # Compute second Jacobian matrix-vector product as general matrix-vector product:
            #   y <- alpha * J @ v + beta * y
            self.gemv_op(self.bsm, v, y, alpha, beta, world_mask)

            # Add regularization, if provided: y <- y + alpha * diag(eta) @ x
            self._apply_regularization(x, y, world_mask, alpha)

        else:
            x_preconditioned = self._vec_temp_cts_space_A
            z = self._vec_temp_cts_space_B

            # Apply preconditioning to input vector: x_p <- diag(P) @ x
            wp.copy(x_preconditioned, x)
            self._apply_preconditioning(x_preconditioned, world_mask)

            # Compute first Jacobian matrix-vector product: v <- J^T @ x_p
            self.ATy_op(self.bsm, x_preconditioned, v, world_mask)

            # Multiply by inverse mass matrix (in-place): v <- M^-1 @ v
            self._apply_inverse_mass_matrix(v, world_mask)

            if beta == 0.0:
                # Special case: If `beta` is zero, we can skip some of the steps and use `y`
                # directly without having to use an additional temporary.

                # Compute second Jacobian matrix-vector product using general matrix-vector product:
                #   y <- alpha * J @ v
                self.gemv_op(self.bsm, v, y, alpha, 0.0, world_mask)

                # Apply preconditioning: y <- diag(P) @ y
                self._apply_preconditioning(y, world_mask)

                # Add regularization, if provided: y <- y + alpha * diag(eta) @ x
                self._apply_regularization(x, y, world_mask, alpha)

            else:
                # Compute second Jacobian matrix-vector product: z <- J @ v
                self.Ax_op(self.bsm, v, z, world_mask)

                # Apply preconditioning: z <- diag(P) @ z
                self._apply_preconditioning(z, world_mask)

                # Add regularization, if provided: z <- z + diag(eta) @ x
                self._apply_regularization(x, z, world_mask)

                # Add scaling and offset: y <- alpha * z + beta * y
                wp.launch(
                    kernel=_scaled_vector_sum,
                    dim=(self._model.size.num_worlds, self._model.size.max_of_max_total_cts),
                    inputs=[
                        self._data.info.num_total_cts,
                        self.bsm.row_start,
                        z,
                        y,
                        alpha,
                        beta,
                        world_mask,
                    ],
                    device=self._device,
                )

    def gemv_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """
        Performs a BLAS-like generalized sparse matrix-transpose-vector product
        `x = alpha * D^T @ y + beta * x`.

        Note:
            Since the Delassus matrix is symmetric, this is equivalent to `gemv` with swapped arguments.
        """
        self.gemv(y, x, world_mask, alpha, beta)

    def _apply_inverse_mass_matrix(self, x: wp.array, world_mask: wp.array):
        """
        Applies the inverse mass matrix to a vector in-place: x = M^-1 @ x.

        Args:
            x (wp.array): Input/output vector in body coordinate space.
            world_mask (wp.array): Array of integers indicating which worlds to process (0 = skip).
        """
        if self._model is None or self._data is None:
            raise RuntimeError("Model and data must be assigned before applying inverse mass matrix.")

        wp.launch(
            kernel=_inverse_mass_matrix_matvec,
            dim=(self._model.size.num_worlds, self._model.size.max_of_num_bodies),
            inputs=[
                self._model.info.num_bodies,
                self._model.info.bodies_offset,
                self._model.bodies.inv_m_i,
                self._data.bodies.inv_I_i,
                x,
                world_mask,
            ],
            device=self._device,
        )

    def _apply_regularization(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0):
        """
        Adds diagonal regularization to the matrix-vector product result.

        If regularization values have been set via `set_regularization()`, this method computes
        `y += alpha * diag(eta) @ x`, where `eta` contains the regularization.

        Args:
            x (wp.array): Input vector to be scaled by the diagonal regularization matrix.
                Shape `(sum_of_max_total_cts,)` and type :class:`float32`.
            y (wp.array): Output vector to which the regularization contribution is added.
                Shape `(sum_of_max_total_cts,)` and type :class:`float32`.
            world_mask (wp.array): Array of integers indicating which worlds to process (0 = skip).
                Shape `(num_worlds,)` and type :class:`int32`.
            alpha (float, optional): Scalar multiplier for the regularization term. Defaults to 1.0.
        """
        if self._model is None or self._data is None:
            raise RuntimeError("Model and data must be assigned before applying regularization.")

        # Skip if no regularization is set
        if self._sigma is None:
            return

        wp.launch(
            kernel=_add_matrix_diag_product,
            dim=(self._model.size.num_worlds, self._model.size.max_of_max_total_cts),
            inputs=[
                self._data.info.num_total_cts,
                self.bsm.row_start,
                self._sigma,
                x,
                y,
                alpha,
                world_mask,
            ],
            device=self._device,
        )

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
        if self._model is None or self._data is None:
            raise RuntimeError("Model and data must be assigned before applying preconditioning.")

        # Skip if no preconditioner is set
        if self._preconditioner is None:
            return

        wp.launch(
            kernel=_set_matrix_diag_product,
            dim=(self._model.size.num_worlds, self._model.size.max_of_max_total_cts),
            inputs=[
                self._data.info.num_total_cts,
                self.bsm.row_start,
                self._preconditioner,
                x,
                world_mask,
            ],
            device=self._device,
        )
