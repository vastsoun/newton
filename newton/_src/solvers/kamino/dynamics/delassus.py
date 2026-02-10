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
Provides containers to represent and operate Delassus operators.

A Delassus operator is a symmetric semi-positive-definite matrix that
represents the apparent inertia within the space defined by the set of
active constraints imposed on a constrained rigid multi-body system.

This module thus provides building-blocks to realize Delassus operators across multiple
worlds contained in a :class:`Model`. The :class:`DelassusOperator` class provides a
high-level interface to encapsulate both the data representation as well as the
relevant operations. It provides methods to allocate the necessary data arrays, build
the Delassus matrix given the current state of the model and the active constraints,
add diagonal regularization, and solve linear systems of the form `D @ x = v` given
arrays holding the right-hand-side (rhs) vectors v. Moreover, it supports the use of
different linear solvers as a back-end for performing the aforementioned linear system
solve. Construction of the Delassus operator is realized using a set of Warp kernels
that parallelize the computation using various strategies.

Typical usage example:
    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilder()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = builder.finalize()
    data = model.data()
    limits = Limits(model)
    contacts = Contacts(builder)
    jacobians = DenseSystemJacobians(model, limits, contacts)

    # Define a linear solver type to use as a back-end for the
    # Delassus operator computations such as factorization and
    # solving the linear system when a rhs vector is provided
    linear_solver = LLTBlockedSolver
    ...

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a Delassus operator and build it using the current model data
    # and active unilateral constraints (i.e. for limits and contacts).
    delassus = DelassusOperator(model, limits, contacts, linear_solver)
    delassus.build(model, data, jacobians)

    # Add diagonal regularization the Delassus matrix
    eta = ...
    delassus.regularize(eta=eta)

    # Factorize the Delassus matrix using the Cholesky factorization
    delassus.compute()

    # Solve a linear system using the Delassus operator
    rhs = ...
    solution = ...
    delassus.solve(b=rhs, x=solution)
"""

from typing import Any

import warp as wp
from warp.context import Devicelike

from ..core.model import Model, ModelData, ModelSize
from ..core.types import FloatType, float32, int32, mat33f, vec3f, vec6f
from ..geometry.contacts import Contacts
from ..kinematics.constraints import get_max_constraints_per_world
from ..kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..kinematics.limits import Limits
from ..linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LinearSolverType
from ..linalg.linear import IterativeSolver

# from ..linalg import Matrices, LinearOperators
# from ..linalg.dense import DenseMatrices, DenseLinearOperators
from ..linalg.sparse_operator import BlockSparseLinearOperators

###
# Module interface
###

__all__ = [
    "BlockSparseMatrixFreeDelassusOperator",
    "DelassusOperator",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _build_delassus_elementwise(
    # Inputs:
    model_info_num_bodies: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    data_bodies_inv_I_i: wp.array(dtype=mat33f),
    jacobians_cts_offset: wp.array(dtype=int32),
    jacobians_cts_data: wp.array(dtype=float32),
    delassus_dim: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    # Outputs:
    delassus_D: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index and Delassus element index
    wid, tid = wp.tid()

    # Retrieve the world dimensions
    nb = model_info_num_bodies[wid]
    bio = model_info_bodies_offset[wid]

    # Retrieve the problem dimensions
    ncts = delassus_dim[wid]

    # Skip if world has no constraints
    if ncts == 0:
        return

    # Compute i (row) and j (col) indices from the tid
    i = tid // ncts
    j = tid % ncts

    # Skip if indices exceed the problem size
    if i >= ncts or j >= ncts:
        return

    # Retrieve the world's matrix offsets
    dmio = delassus_mio[wid]
    cjmio = jacobians_cts_offset[wid]

    # Compute the number of body DoFs of the world
    nbd = 6 * nb

    # Buffers
    # tmp = vec3f(0.0)
    Jv_i = vec3f(0.0)
    Jv_j = vec3f(0.0)
    Jw_i = vec3f(0.0)
    Jw_j = vec3f(0.0)
    D_ij = float32(0.0)
    D_ji = float32(0.0)

    # Loop over rigid body blocks
    # NOTE: k is the body index w.r.t the world
    for k in range(nb):
        # Body index (bid) of body k w.r.t the model
        bid_k = bio + k
        # DoF index offset (dio) of body k in the flattened Jacobian matrix
        # NOTE: Equivalent to the column index in the matrix-form of the Jacobian matrix
        dio_k = 6 * k
        # Jacobian index offsets
        jio_ik = cjmio + nbd * i + dio_k
        jio_jk = cjmio + nbd * j + dio_k

        # Load the Jacobian blocks of body k
        for d in range(3):
            # Load the i-th row block
            Jv_i[d] = jacobians_cts_data[jio_ik + d]
            Jw_i[d] = jacobians_cts_data[jio_ik + d + 3]
            # Load the j-th row block
            Jv_j[d] = jacobians_cts_data[jio_jk + d]
            Jw_j[d] = jacobians_cts_data[jio_jk + d + 3]

        # Linear term: inv_m_k * dot(Jv_i, Jv_j)
        inv_m_k = model_bodies_inv_m_i[bid_k]
        lin_ij = inv_m_k * wp.dot(Jv_i, Jv_j)
        lin_ji = inv_m_k * wp.dot(Jv_j, Jv_i)

        # Angular term: dot(Jw_i.T * I_k, Jw_j)
        inv_I_k = data_bodies_inv_I_i[bid_k]
        ang_ij = float32(0.0)
        ang_ji = float32(0.0)
        for r in range(3):  # Loop over rows of A (and elements of v)
            for c in range(r, 3):  # Loop over upper triangular part of A (including diagonal)
                ang_ij += Jw_i[r] * inv_I_k[r, c] * Jw_j[c]
                ang_ji += Jw_j[r] * inv_I_k[r, c] * Jw_i[c]
                if r != c:
                    ang_ij += Jw_i[c] * inv_I_k[r, c] * Jw_j[r]
                    ang_ji += Jw_j[c] * inv_I_k[r, c] * Jw_i[r]

        # Accumulate
        D_ij += lin_ij + ang_ij
        D_ji += lin_ji + ang_ji

    # Store the result in the Delassus matrix
    delassus_D[dmio + ncts * i + j] = 0.5 * (D_ij + D_ji)


@wp.kernel
def _regularize_delassus_diagonal(
    # Inputs:
    delassus_dim: wp.array(dtype=int32),
    delassus_vio: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    eta: wp.array(dtype=float32),
    # Outputs:
    delassus_D: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the problem dimensions and matrix block index offset
    dim = delassus_dim[wid]
    vio = delassus_vio[wid]
    mio = delassus_mio[wid]

    # Skip if row index exceed the problem size
    if tid >= dim:
        return

    # Regularize the diagonal element
    delassus_D[mio + dim * tid + tid] += eta[vio + tid]


@wp.kernel
def _inverse_mass_matrix_matvec(
    # Model:
    model_info_num_bodies: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    # Mass properties:
    model_bodies_inv_m_i: wp.array(dtype=float32),
    data_bodies_inv_I_i: wp.array(dtype=mat33f),
    # Vector:
    x: wp.array(dtype=float32),
    # Mask:
    world_mask: wp.array(dtype=int32),
):
    """
    Applies the inverse mass matrix to a vector in body coordinate space: x = M^-1 @ x.
    """
    # Retrieve the thread index as the world index and body index
    world_id, body_id = wp.tid()

    # Skip if world is inactive or body index exceeds the number of bodies in the world
    if world_mask[world_id] == 0 or body_id >= model_info_num_bodies[world_id]:
        return

    # Index of body w.r.t the model
    global_body_id = model_info_bodies_offset[world_id] + body_id

    # Body dof index offset in the flattened vector
    body_dof_index = 6 * global_body_id

    # Load the inverse mass and inverse inertia for this body
    inv_m = model_bodies_inv_m_i[global_body_id]
    inv_I = data_bodies_inv_I_i[global_body_id]

    # Load the input vector components for this body
    v = x[body_dof_index : body_dof_index + 6]
    v_lin = wp.vec3(*v[0:3])
    v_ang = wp.vec3(*v[3:6])

    # Apply inverse mass to linear velocity component
    v_lin_out = inv_m * v_lin

    # Apply inverse inertia to angular velocity component
    v_ang_out = inv_I @ v_ang

    # Store the result
    x[body_dof_index + 0] = v_lin_out[0]
    x[body_dof_index + 1] = v_lin_out[1]
    x[body_dof_index + 2] = v_lin_out[2]
    x[body_dof_index + 3] = v_ang_out[0]
    x[body_dof_index + 4] = v_ang_out[1]
    x[body_dof_index + 5] = v_ang_out[2]


@wp.kernel
def _compute_block_sparse_delassus_diagonal(
    # Inputs:
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    data_bodies_inv_I_i: wp.array(dtype=mat33f),
    bsm_nzb_start: wp.array(dtype=int32),
    bsm_num_nzb: wp.array(dtype=int32),
    bsm_nzb_coords: wp.array2d(dtype=int32),
    bsm_nzb_values: wp.array(dtype=vec6f),
    vec_start: wp.array(dtype=int32),
    # Outputs:
    diag: wp.array(dtype=float32),
):
    """
    Computes the diagonal entries of the Delassus matrix by summing up the contributions of each
    non-zero block of the Jacobian: D_ii = sum_k J_ik @ M_kk^-1 @ (J_ik)^T

    This kernel processes one non-zero block per thread and accumulates all contributions.
    """
    # Retrieve the thread index as the world index and block index
    world_id, block_idx_local = wp.tid()

    # Skip if block index exceeds the number of non-zero blocks
    if block_idx_local >= bsm_num_nzb[world_id]:
        return

    # Compute the global block index
    block_idx = bsm_nzb_start[world_id] + block_idx_local

    # Get the row and column for this block
    row = bsm_nzb_coords[block_idx, 0]
    col = bsm_nzb_coords[block_idx, 1]

    # Get the body index offset for this world
    body_index_offset = model_info_bodies_offset[world_id]

    # Get the Jacobian block and extract linear and angular components
    J_block = bsm_nzb_values[block_idx]
    Jv = J_block[0:3]
    Jw = J_block[3:6]

    # Get the body index from the column
    body_idx = col // 6
    body_idx_global = body_index_offset + body_idx

    # Load the inverse mass and inverse inertia for this body
    inv_m = model_bodies_inv_m_i[body_idx_global]
    inv_I = data_bodies_inv_I_i[body_idx_global]

    # Compute linear contribution: Jv^T @ inv_m @ Jv
    diag_kk = inv_m * wp.dot(Jv, Jv)

    # Compute angular contribution: Jw^T @ inv_I @ Jw
    diag_kk += wp.dot(Jw, inv_I @ Jw)

    # Atomically add contribution to the diagonal element
    wp.atomic_add(diag, vec_start[world_id] + row, diag_kk)


@wp.kernel
def _add_matrix_diag_product(
    model_data_num_total_cts: wp.array(dtype=int32),
    row_start: wp.array(dtype=int32),
    d: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    alpha: float,
    world_mask: wp.array(dtype=int32),
):
    """
    Adds the product of a vector with a diagonal matrix to another vector: y += alpha * diag(d) @ x
    This is used to apply a regularization to the Delassus matrix-vector product.
    """
    # Retrieve the thread index as the world index and constraint index
    world_id, ct_id = wp.tid()

    # Terminate early if world or constraint is inactive
    if world_mask[world_id] == 0 or ct_id >= model_data_num_total_cts[world_id]:
        return

    idx = row_start[world_id] + ct_id
    y[idx] += alpha * d[idx] * x[idx]


@wp.kernel
def _set_matrix_diag_product(
    model_data_num_total_cts: wp.array(dtype=int32),
    row_start: wp.array(dtype=int32),
    d: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    world_mask: wp.array(dtype=int32),
):
    """
    Applies a diagonal matrix to a vector: x = diag(d) @ x
    This is used to apply preconditioning to a vector.
    """
    # Retrieve the thread index as the world index and constraint index
    world_id, ct_id = wp.tid()

    # Terminate early if world or constraint is inactive
    if world_mask[world_id] == 0 or ct_id >= model_data_num_total_cts[world_id]:
        return

    idx = row_start[world_id] + ct_id
    x[idx] = d[idx] * x[idx]


@wp.kernel
def _scaled_vector_sum(
    model_data_num_total_cts: wp.array(dtype=int32),
    row_start: wp.array(dtype=int32),
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    alpha: float,
    beta: float,
    world_mask: wp.array(dtype=int32),
):
    """
    Computes the scaled vector sum: y = alpha * x + beta * y.
    """
    # Retrieve the thread index as the world index and constraint index
    world_id, ct_id = wp.tid()

    # Terminate early if world or constraint is inactive
    if world_mask[world_id] == 0 or ct_id >= model_data_num_total_cts[world_id]:
        return

    idx = row_start[world_id] + ct_id
    y[idx] = alpha * x[idx] + beta * y[idx]


###
# Interfaces
###


class DelassusOperator:
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
        device: Devicelike = None,
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
            device (Devicelike, optional): The device identifier for the Delassus operator. Defaults to None.
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
        self._device: Devicelike = device

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
        device: Devicelike = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Allocates the Delassus operator with the specified dimensions and device.

        Args
        ----
            dims (List[int]): The dimensions of the Delassus matrix for each world.
            device (Devicelike, optional): The device identifier for the Delassus operator. Defaults to None.
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
            kernel=_build_delassus_elementwise,
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
            kernel=_regularize_delassus_diagonal,
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


class BlockSparseMatrixFreeDelassusOperator(BlockSparseLinearOperators):
    """
    A matrix-free Delassus operator for representing and operating on multiple independent sparse
    linear systems.

    In contrast to the dense :class:`DelassusOperator`, this operator only provides functions to
    compute matrix-vector products with the Delassus matrix, not solve linear systems.

    The Delassus operator D is implicitly defined as D = J @ M^-1 @ J^T, where J is the constraint
    Jacobian and M is the mass matrix. It supports diagonal regularization and diagonal
    preconditioning.

    For a given diagonal regularization matrix R and a diagonal preconditioning matrix P, the
    final operator is defined by the matrix P @ D @ P + R.

    Typical usage example:

    .. code-block:: python

        # Create a model builder and add bodies, joints, geoms, etc.
        builder = ModelBuilder()
        ...

        # Create a model from the builder and construct additional
        # containers to hold joint-limits, contacts, Jacobians
        model = builder.finalize()
        data = model.data()
        limits = Limits(model)
        contacts = Contacts(builder)
        jacobians = SparseSystemJacobians(model, limits, contacts)

        # Build the Jacobians for the model and active limits and contacts
        jacobians.build(model, data, limits, contacts)
        ...

        # Create a Delassus operator from the model data and Jacobians
        delassus = BlockSparseMatrixFreeDelassusOperator(model, data, jacobians)

        # Add diagonal regularization to the Delassus operator
        eta = ...
        delassus.set_regularization(eta=eta)

        # Add preconditioning to the Delassus operator
        P = ...
        delassus.set_preconditioner(preconditioner=P)

        # Compute the matrix-vector product `y = D @ x` using the Delassus operator
        x = ...
        y = ...
        world_mask = ...
        delassus.matvec(x=x, y=y, world_mask=world_mask)
    """

    def __init__(
        self,
        model: Model | None = None,
        data: ModelData | None = None,
        jacobians: SparseSystemJacobians | None = None,
        solver: LinearSolverType = None,
        solver_kwargs: dict[str, Any] | None = None,
        device: Devicelike = None,
    ):
        """
        Creates a Delassus operator for the given model.

        This class also supports deferred allocation, i.e. it can be initialized without a model,
        and the allocation can be performed later using the `finalize` method. This is useful for
        scenarios where the model or constraints are not known at the time of Delassus operator
        creation, but will be available later.

        The dimension of a Delassus matrix is defined as the sum over active joint, limit, and
        contact constraints, and the maximum dimension is the maximum number of constraints that can
        be active in each world.

        Args:
            model (Model): The model container for which the Delassus operator is built.
            data (ModelData, optional): The model data container holding the state info and data.
            jacobians (SparseSystemJacobians, optional): The sparse Jacobians container. Can be assigned later via the `assign` method.
            solver (LinearSolverType, optional): The linear solver class to use for solving linear systems. Must be a subclass of `IterativeSolver`.
            solver_kwargs (dict, optional): Additional keyword arguments to pass to the solver constructor.
            device (Devicelike, optional): The device identifier for the Delassus operator. Defaults to None.
        """
        super().__init__()

        # self.bsm represents the constraint Jacobian
        self._model: Model | None = None
        self._data: ModelData | None = None
        self._preconditioner: wp.array | None = None
        self._eta: wp.array | None = None

        # Problem info object
        # TODO: Create more general info object independent of dense matrix representation
        self._info: DenseSquareMultiLinearInfo | None = None

        # Cache the requested device
        self._device: Devicelike = device

        # Declare the optional (iterative) solver
        self._solver: LinearSolverType | None = None

        # Temporary vector to store results, sized to the number of body dofs in a model.
        self._vec_temp_body_space: wp.array | None = None
        # Temporary vectors to store results, sized to the maximum number of constraints in a model.
        self._vec_temp_cts_space_A: wp.array | None = None
        self._vec_temp_cts_space_B: wp.array | None = None

        # Allocate the Delassus operator data if at least the model is provided
        if model is not None:
            self.finalize(
                model=model,
                data=data,
                jacobians=jacobians,
                solver=solver,
                device=device,
                solver_kwargs=solver_kwargs,
            )

    def finalize(
        self,
        model: Model,
        data: ModelData,
        jacobians: SparseSystemJacobians | None = None,
        solver: LinearSolverType = None,
        device: Devicelike = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Allocates the Delassus operator with the specified dimensions and device.

        Args:
            model (Model): The model container for which the Delassus operator is built.
            data (ModelData): The model data container holding the state info and data.
            jacobians (SparseSystemJacobians, optional): The sparse Jacobians container. Can be assigned later via the `assign` method.
            solver (LinearSolverType, optional): The linear solver class to use for solving linear systems. Must be a subclass of `IterativeSolver`.
            device (Devicelike, optional): The device identifier for the Delassus operator. Defaults to None.
            solver_kwargs (dict, optional): Additional keyword arguments to pass to the solver constructor.
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
        if solver is not None and not issubclass(solver, IterativeSolver):
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

        # Initialize temporary memory
        self._vec_temp_body_space = wp.empty(
            (self._model.size.sum_of_num_body_dofs,), dtype=float32, device=self._device
        )

        # Assign Jacobian if specified
        if jacobians is not None:
            self.bsm = jacobians._J_cts.bsm

        # Optionally initialize the iterative linear system solver if one is specified
        if solver is not None:
            solver_kwargs = solver_kwargs or {}
            self._solver = solver(operator=self, device=self._device, **solver_kwargs)

    def assign(self, jacobian: SparseSystemJacobians):
        """
        Assigns the constraint Jacobian to the Delassus operator.

        This method links the Delassus operator to the block sparse Jacobian matrix,
        which is used to compute the implicit Delassus matrix-vector products.

        Args:
            jacobian (SparseSystemJacobians): The sparse system Jacobians container holding the
                constraint Jacobian matrix in block sparse format.
        """
        self.bsm = jacobian._J_cts.bsm

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
        self._eta = eta

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
            kernel=_compute_block_sparse_delassus_diagonal,
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
    def device(self) -> Devicelike:
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
        if self._eta is None:
            return

        wp.launch(
            kernel=_add_matrix_diag_product,
            dim=(self._model.size.num_worlds, self._model.size.max_of_max_total_cts),
            inputs=[
                self._data.info.num_total_cts,
                self.bsm.row_start,
                self._eta,
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
