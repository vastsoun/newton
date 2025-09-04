###########################################################################
# KAMINO: Delassus Operator
###########################################################################
"""
Provides containers and operations to respectively represent and compute Delassus operators.

A Delassus operator is a symmetric matrix that represents the apparent inertia in the space
defined by the set of active constraints imposed on a constrained rigid multi-body system.

This module provides building-blocks to compute the Delassus operator in parallel across
multiple worlds represented within a :class:`Model`. A set of Warp kernels that compute the
Delassus matrix using various parallelization strategies The :class:`DelassusData` data
structure to bundle the necessary data arrays, either by allocation or by references to
existing arrays provided by the user. The :class:`DelassusOperator` class provides a
high-level interface to encapsulate both the data representations as well as the relevant
operations involving the Delassus operator. It provides methods to allocate the necessary
data arrays, build the Delassus matrix given the current state of the model and the active
constraints, add diagonal regularization, and solve linear systems of the form `D @ x = v`
given arrays holding the right-hand-side (rhs) vectors v.

Typical usage example:

    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilder()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = builder.finalize()
    state = model.data()
    limits = Limits(builder)
    contacts = Contacts(builder)
    jacobians = DenseSystemJacobians(model, limits, contacts)
    factorizer = CholeskyFactorizer(model)
    ...

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, state, limits, contacts)
    ...

    # Create a Delassus operator and build it using the current model state
    # and active unilateral constraints (i.e. for limits and contacts).
    delassus = DelassusOperator(model, limits, contacts, factorizer)
    delassus.build(model, state, jacobians)

    # Add diagonal regularization the Delassus matrix
    eta = ...
    delassus.regularize(eta=)

    # Factorize the Delassus matrix using the Cholesky factorization
    delassus.factorize()

    # Solve a linear system using the Delassus operator
    rhs = ...
    delassus.solve(rhs)
"""


from __future__ import annotations

import warp as wp

from typing import List
from warp.context import Devicelike
from newton._src.solvers.kamino.core.types import int32, float32, vec3f, mat33f
from newton._src.solvers.kamino.core.model import ModelSize, ModelData, Model
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.linalg.cholesky import CholeskyFactorizer
from newton._src.solvers.kamino.kinematics.jacobians import DenseSystemJacobiansData
from newton._src.solvers.kamino.kinematics.constraints import max_constraints_per_world


###
# Module interface
###

__all__ = [
    "DelassusData",
    "DelassusOperator",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###

class DelassusData:
    """
    A container to hold the time-varying data of the Delassus operator.
    """
    def __init__(self):
        self.maxdim: wp.array(dtype=int32) | None = None
        """
        The max dimensions of the symmetric Delassus matrix of each world.\n
        This is equal to the sum of the maximum constraints in each world: joints, max limits, and max contacts.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.dim: wp.array(dtype=int32) | None = None
        """
        The active dimensions of the symmetric Delassus matrix of each world.\n
        This is equal to the sum of all active constraint in each world: joints, limits, and contact.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.mio: wp.array(dtype=int32) | None = None
        """
        The matrix index offsets (MIO) of the Delassus matrix block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.vio: wp.array(dtype=int32) | None = None
        """
        The vector index offsets (VIO) of the active constraints in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.D: wp.array(dtype=float32) | None = None
        """
        The flat Delassus matrix (constraint-space apparent inertia).\n
        Shape of ``(sum(maxdim_w * maxdim_w),)`` and type :class:`float32`.\n
        `maxdim_w` is the maximum dimensions of the Delassus matrix block of each world.
        """


###
# Kernels
###

@wp.kernel
def _build_delassus_elementwise(
    # Inputs:
    model_info_num_bodies: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    state_bodies_inv_I_i: wp.array(dtype=mat33f),
    jacobians_cts_offset: wp.array(dtype=int32),
    jacobians_cts_data: wp.array(dtype=float32),
    delassus_maxdim: wp.array(dtype=int32),
    delassus_dim: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    # Outputs:
    delassus_D: wp.array(dtype=float32)
):
    # Retrieve the thread index as the world index and Delassus element index
    wid, tid = wp.tid()

    # Retrieve the world dimensions
    nb = model_info_num_bodies[wid]
    bio = model_info_bodies_offset[wid]

    # Retrieve the problem dimensions
    maxncts = delassus_maxdim[wid]
    ncts = delassus_dim[wid]

    # Compute i (row) and j (col) indices from the tid
    i = tid // ncts
    j = tid % ncts

    # Skip if indices exceed the problem size
    if i >= ncts or j >= ncts:
        return
    # wp.printf("[wid: %d]: [tid: %d]: i= %d, j= %d\n", wid, tid, i, j)

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
        # NOTE: Equievalent to the column index in the matrix-form of the Jacobian matrix
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

        # Linear term: m_k * dot(Jv_i, Jv_j)
        lin_ij = model_bodies_inv_m_i[bid_k] * wp.dot(Jv_i, Jv_j)
        lin_ji = model_bodies_inv_m_i[bid_k] * wp.dot(Jv_j, Jv_i)

        # Angular term: dot(Jw_i.T * I_k, Jw_j)
        I_k = state_bodies_inv_I_i[bid_k]
        # Jw_i_norm = wp.length(Jw_i)
        # Jw_j_norm = wp.length(Jw_j)
        # Jw_i_scaled = (1.0 / Jw_i_norm) * Jw_i if Jw_i_norm > 0 else Jw_i
        # Jw_j_scaled = (1.0 / Jw_j_norm) * Jw_j if Jw_j_norm > 0 else Jw_j
        # I_k_trace = wp.trace(I_k)
        # I_k_scaled = (1.0 / I_k_trace) * I_k

        # wp.printf("[wid: %d]: [tid: %d]: I_%d:\n", wid, tid, bid_k)
        # print(I_k)
        # wp.printf("[wid: %d]: [tid: %d]: Jw_%d:\n", wid, tid, i)
        # print(Jw_i)
        # wp.printf("[wid: %d]: [tid: %d]: Jw_%d:\n", wid, tid, j)
        # print(Jw_j)
        # print("\n\n\n")

        # tmp = vec3f(0.0)
        # for r in range(3):
        #     for c in range(3):
        #         tmp[r] += Jw_i[c] * I_k[c, r]  # tmp = I_k^T * Jw_i
        # ang = wp.dot(tmp, Jw_j)

        ang_ij = float32(0.0)
        ang_ji = float32(0.0)
        for r in range(3):  # Loop over rows of A (and elements of v)
            for c in range(r, 3):  # Loop over upper triangular part of A (including diagonal)
                ang_ij += Jw_i[r] * I_k[r, c] * Jw_j[c]
                ang_ji += Jw_j[r] * I_k[r, c] * Jw_i[c]
                # ang += Jw_i_scaled[r] * I_k[r, c] * Jw_j_scaled[c]
                if r != c:
                    ang_ij += Jw_i[c] * I_k[r, c] * Jw_j[r]
                    ang_ji += Jw_j[c] * I_k[r, c] * Jw_i[r]
                    # ang += Jw_i_scaled[c] * I_k[r, c] * Jw_j_scaled[r]
        # ang *= Jw_i_norm * Jw_j_norm

        # ang = float32(0.0)
        # for r in range(3):  # Loop over rows of A (and elements of v)
        #     for c in range(3):  # Loop over columns of A (and elements of u)
        #         ang += Jw_i[r] * I_k[r, c] * Jw_j[c]
        #         ang += Jw_i_scaled[r] * I_k[r, c] * Jw_j_scaled[c]
        #         ang += Jw_i[r] * I_k_scaled[r, c] * Jw_j[c]
        # ang *= I_k_trace
        # ang *= Jw_i_norm * Jw_j_norm

        # Accumulate
        D_ij += lin_ij + ang_ij
        D_ji += lin_ji + ang_ji

    # print("\n")
    # Store the result in the Delassus matrix
    # delassus_D[dmio + maxncts * i + j] = D_ij
    delassus_D[dmio + maxncts * i + j] = 0.5 * (D_ij + D_ji)


@wp.kernel
def _regularize_delassus_diagonal(
    # Inputs:
    delassus_maxdim: wp.array(dtype=int32),
    delassus_dim: wp.array(dtype=int32),
    delassus_vio: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    eta: wp.array(dtype=float32),
    # Outputs:
    delassus_D: wp.array(dtype=float32)
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the problem dimensions and matrix block index offset
    maxdim = delassus_maxdim[wid]
    dim = delassus_dim[wid]
    vio = delassus_vio[wid]
    mio = delassus_mio[wid]

    # Skip if row index exceed the problem size
    if tid >= dim:
        return

    # Regularize the diagonal element
    delassus_D[mio + maxdim * tid + tid] += eta[vio + tid]


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
        state: ModelData | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        factorizer: CholeskyFactorizer = None,
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
            state (ModelData, optional): The model state container holding the state info and data.
            limits (Limits, optional): The container holding the allocated joint-limit data.
            contacts (Contacts, optional): The container holding the allocated contacts data.
            device (Devicelike, optional): The device identifier for the Delassus operator. Defaults to None.
            factorizer (CholeskyFactorizer, optional): An optional Cholesky factorization object. Defaults to None.
        """
        # Declare and initialize the host-side cache of the necessary memory allocations
        self._num_worlds: int = 0
        self._model_maxdims: int = 0
        self._model_maxsize: int = 0
        self._world_maxdims: List[int] = []
        self._world_maxsize: List[int] = []
        self._max_of_max_total_D_size: int = 0

        # Cache the requested device
        self._device: Devicelike = device

        # Declare the model size cache
        self._size: ModelSize | None = None

        # Initialize the Delassus state container
        self._data: DelassusData = DelassusData()

        # Declare the optional Cholesky factorization
        self._factorizer: CholeskyFactorizer = None

        # Allocate the Delassus operator data if at least the model is provided
        if model is not None:
            self.allocate(
                model=model,
                state=state,
                limits=limits,
                contacts=contacts,
                factorizer=factorizer,
                device=device,
            )

    @property
    def num_worlds(self) -> int:
        """
        Returns the number of worlds represented by the Delassus operater.
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
    def data(self) -> DelassusData:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._data

    @property
    def factorizer(self) -> CholeskyFactorizer:
        """
        The Cholesky factorization object for the Delassus operator.
        This is used to perform the factorization of the Delassus matrix.
        """
        return self._factorizer

    def allocate(
        self,
        model: Model,
        state: ModelData,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        factorizer: CholeskyFactorizer = None,
        device: Devicelike = None,
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
            raise ValueError("A model of type `Model` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, Model):
            raise ValueError("Invalid model provided. Must be an instance of `Model`.")

        # Ensure the state container is valid if provided
        if state is not None:
            if not isinstance(state, ModelData):
                raise ValueError("Invalid state container provided. Must be an instance of `ModelData`.")

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
        maxdims = max_constraints_per_world(model, limits, contacts)

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

        # First allocate the Delassus matrix data array given the total maximum size
        self._data.D = wp.zeros(shape=(self._model_maxsize,), dtype=float32, device=self._device)

        # If the model info contains the maximum total constraints array use that,
        # otherwise allocate a new array with the world dimensions.
        if model.info.max_total_cts is not None:
            self._data.maxdim = model.info.max_total_cts
        else:
            self._data.maxdim = wp.array(self._world_dims, dtype=int32, device=self._device)

        # If the state info contains the total active constraints array
        # use that, otherwise allocate a new array with zeros.
        if state is not None:
            self._data.dim = state.info.num_total_cts
        else:
            self._data.dim = wp.zeros(shape=(self._num_worlds,), dtype=int32, device=self._device)

        # If the model info contains the total constraints offset array use that,
        # otherwise allocate a new array with the world offsets.
        if model.info.total_cts_offset is not None:
            self._data.vio = model.info.total_cts_offset
        else:
            vec_offsets = [0] + [sum(self._world_dims[:i]) for i in range(1, self._num_worlds + 1)]
            self._data.vio = wp.array(vec_offsets[:self._num_worlds], dtype=int32, device=self._device)

        # Allocate the matrix index offsets (MIO) for each world
        mat_offsets = [0] + [sum(self._world_size[:i]) for i in range(1, self._num_worlds + 1)]
        self._data.mio = wp.array(mat_offsets[:self._num_worlds], dtype=int32, device=self._device)

        # Optionally initialize the factorizer if one is specified
        if factorizer is not None:
            # NOTE: Since the dimensions of the factorizer are the same as the Delassus operator,
            # we can re-use the same info arrays, and propagate them by reference to the factorizer.
            # This is possible by passing `allocate_info=False` to the factorizer constructor.
            self._factorizer = factorizer(dims=self._world_dims, allocate_info=False, device=self._device)
            self._factorizer._data.maxdim = self._data.maxdim
            self._factorizer._data.dim = self._data.dim
            self._factorizer._data.mio = self._data.mio
            self._factorizer._data.vio = self._data.vio

    def zero(self):
        """
        Sets all values of the Delassus matrix to zero.
        This is useful for resetting the operator before recomputing it.
        """
        self._data.D.zero_()

    def build(
        self,
        model: Model,
        state: ModelData,
        jacobians: DenseSystemJacobiansData,
        reset_to_zero: bool = True
    ):
        """
        Builds the Delassus matrix using the provided Model, ModelData, and constraint Jacobians.

        Args:
            model (Model): The model for which the Delassus operator is built.
            state (ModelData): The current state of the model.
            reset_to_zero (bool, optional): If True, resets the Delassus matrix to zero before building. Defaults to True.

        Raises:
            ValueError: If the model, state, or Jacobians are not valid.
            ValueError: If the Delassus matrix is not allocated.
        """
        # Ensure the model is valid
        if model is None or not isinstance(model, Model):
            raise ValueError("A valid model of type `Model` must be provided to build the Delassus operator.")

        # Ensure the state is valid
        if state is None or not isinstance(state, ModelData):
            raise ValueError("A valid model state of type `ModelData` must be provided to build the Delassus operator.")

        # Ensure the Jacobians are valid
        if jacobians is None or not isinstance(jacobians, DenseSystemJacobiansData):
            raise ValueError("A valid Jacobians state of type `DenseSystemJacobiansData` must be provided to build the Delassus operator.")

        # Ensure the Delassus matrix is allocated
        if self._data.D is None:
            raise ValueError("Delassus matrix is not allocated. Call allocate() first.")

        # Initialze the Delassus matrix to zero
        if reset_to_zero:
            self._data.D.zero_()

        # Build the Delassus matrix parrallelized element-wise
        wp.launch(
            kernel=_build_delassus_elementwise,
            dim=(self._size.num_worlds, self._max_of_max_total_D_size),
            inputs=[
                # Inputs:
                model.info.num_bodies,
                model.info.bodies_offset,
                model.bodies.inv_m_i,
                state.bodies.inv_I_i,
                jacobians.J_cts_offsets,
                jacobians.J_cts_data,
                self._data.maxdim,
                self._data.dim,
                self._data.mio,
                # Outputs:
                self._data.D
            ]
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
            inputs=[
                self._data.maxdim,
                self._data.dim,
                self._data.vio,
                self._data.mio,
                eta,
                self._data.D
            ]
        )

    def factorize(self, reset_to_zero: bool = True):
        """
        Factorizes the Delassus matrix using the Cholesky factorization.\n
        Returns True if the factorization was successful, False otherwise.

        Args:
            reset_to_zero (bool): If True, resets the Delassus matrix to zero before factorizing.
            This is useful for ensuring that the matrix is in a clean state before factorization.
        """
        # Ensure the Delassus matrix is allocated
        if self._data.D is None:
            raise ValueError("Delassus matrix is not allocated. Call allocate() first.")

        # Ensure the factorizer is available if factorization is requested
        if self._factorizer is None:
            raise ValueError("Cholesky factorizer is not available. Allocate with factorizer=CholeskyFactorizer.")

        # Optionally initialize the factorization matrix before factorizing
        if reset_to_zero:
            self.factorizer.zero()

        # Perform the Cholesky factorization
        self._factorizer.factorize(self._data.D)

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
        if self._data.D is None:
            raise ValueError("Delassus matrix is not allocated. Call allocate() first.")

        # Ensure the factorizer is available if solving is requested
        if self._factorizer is None:
            raise ValueError("Cholesky factorizer is not available. Allocate with factorizer=CholeskyFactorizer.")

        # Solve the linear system using the factorized matrix
        return self._factorizer.solve(b=v, x=x)

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
        if self._data.D is None:
            raise ValueError("Delassus matrix is not allocated. Call allocate() first.")

        # Ensure the factorizer is available if solving is requested
        if self._factorizer is None:
            raise ValueError("Cholesky factorizer is not available. Allocate with factorizer=CholeskyFactorizer.")

        # Solve the linear system in-place
        return self._factorizer.solve_inplace(x=x)
