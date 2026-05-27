# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import warp as wp

from ..core.model import ModelKamino
from ..kinematics.jacobians import (
    DenseSystemJacobians,
    SparseSystemJacobians,
    SystemJacobiansType,
)

###
# Module interface
###

__all__ = ["compute_constraint_space_velocities"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _compute_constraint_velocities_dense(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    model_info_max_total_cts: wp.array[wp.int32],
    jacobian_cts_offsets: wp.array[wp.int32],
    jacobian_cts_data: wp.array[wp.float32],
    v_start: wp.array[wp.int32],
    body_u: wp.array[wp.spatial_vectorf],
    # Outputs:
    v: wp.array[wp.float32],
):
    # Retrieve the thread index as the world and constraint row index
    wid, tid = wp.tid()

    # Skip if row index exceeds the per-world allocated number of constraints
    if tid >= model_info_max_total_cts[wid]:
        return

    # Retrieve the world-specific data
    bio = model_info_bodies_offset[wid]
    nb = model_info_bodies_offset[wid + 1] - bio

    # Compute the number of body DoFs of the world
    nbd = 6 * nb

    # Append the row offset to the world's Jacobian block start index
    cjmio = jacobian_cts_offsets[wid] + nbd * tid

    # Compute the output vector index for the constraint row
    cts_offset = v_start[wid] + tid

    # Buffers
    J_i = wp.spatial_vectorf(0.0)
    v_i = wp.float32(0.0)

    # Iterate over each body to accumulate constraint velocity contributions
    for i in range(nb):
        # Compute the Jacobian block index
        m_ji = cjmio + 6 * i

        # Extract the body twist
        u_i = body_u[bio + i]

        # Extract the Jacobian block J_ji
        for d in range(6):
            J_i[d] = jacobian_cts_data[m_ji + d]

        # Accumulate J_ji @ u_i
        v_i += wp.dot(J_i, u_i)

    # Store the resulting constraint-space velocity
    v[cts_offset] = v_i


@wp.kernel
def _compute_constraint_velocities_sparse(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    jac_num_nzb: wp.array[wp.int32],
    jac_nzb_start: wp.array[wp.int32],
    jac_nzb_coords: wp.array2d[wp.int32],
    jac_nzb_values: wp.array[wp.spatial_vectorf],
    v_start: wp.array[wp.int32],
    body_u: wp.array[wp.spatial_vectorf],
    # Outputs:
    v: wp.array[wp.float32],
):
    # Retrieve the thread index as the world and non-zero block index
    wid, nzb_id = wp.tid()

    # Skip if the non-zero block index exceeds the number of active blocks for the world
    if nzb_id >= jac_num_nzb[wid]:
        return

    # Retrieve the Jacobian block coordinates and values
    global_block_idx = jac_nzb_start[wid] + nzb_id
    block_coord = jac_nzb_coords[global_block_idx]
    jac_block = jac_nzb_values[global_block_idx]

    # Retrieve the world-specific body index offset
    bio = model_info_bodies_offset[wid]

    # Decode the constraint row and body index from the block coordinates
    cts_row = block_coord[0]
    bid = block_coord[1] // 6

    # Extract the body twist
    u_i = body_u[bio + bid]

    # Atomically accumulate J_ji @ u_i contribution into the output constraint row
    wp.atomic_add(v, v_start[wid] + cts_row, wp.dot(jac_block, u_i))


###
# Launchers
###


def compute_constraint_velocities_dense(
    model: ModelKamino,
    jacobians: DenseSystemJacobians,
    u: wp.array[wp.spatial_vectorf],
    v_start: wp.array[wp.int32],
    v: wp.array[wp.float32],
    reset_to_zero: bool = True,
) -> None:
    """
    Computes the constraint-space velocities `v = J_cts @ u_i` for the dense Jacobian case.

    Args:
        model:
            The model containing the time-invariant data of the simulation.
        jacobians:
            The dense system Jacobians container providing ``J_cts``.
        u:
            The input per-body twists ``u_i``.
        v_start:
            The per-world starting offsets into the flat constraint-space output array.
        v:
            The output flat array to store the constraint-space velocities.
        reset_to_zero:
            Whether to reset the output array to zero before launching the kernel.
            Defaults to ``True``.
    """
    # First check that the Jacobians are dense
    if not isinstance(jacobians, DenseSystemJacobians):
        raise ValueError(f"Expected `DenseSystemJacobians` but got {type(jacobians)}.")

    # Optionally clear the output array if requested
    if reset_to_zero:
        v.zero_()

    # Then launch the kernel to compute the constraint-space velocities
    wp.launch(
        _compute_constraint_velocities_dense,
        dim=(model.size.num_worlds, model.size.max_of_max_total_cts),
        inputs=[
            model.info.bodies_offset,
            model.info.max_total_cts,
            jacobians.data.J_cts_offsets,
            jacobians.data.J_cts_data,
            v_start,
            u,
        ],
        outputs=[v],
        device=model.device,
    )


def compute_constraint_velocities_sparse(
    model: ModelKamino,
    jacobians: SparseSystemJacobians,
    u: wp.array[wp.spatial_vectorf],
    v_start: wp.array[wp.int32],
    v: wp.array[wp.float32],
    reset_to_zero: bool = True,
) -> None:
    """
    Computes the constraint-space velocities `v = J_cts @ u_i` for the sparse Jacobian case.

    Args:
        model:
            The model containing the time-invariant data of the simulation.
        jacobians:
            The sparse system Jacobians container providing ``J_cts``.
        u:
            The input per-body twists ``u_i``.
        v_start:
            The per-world starting offsets into the flat constraint-space output array.
        v:
            The output flat array to store the constraint-space velocities.
        reset_to_zero:
            Whether to reset the output array to zero before launching the kernel.
            The kernel uses ``wp.atomic_add`` so the output must be zero before each call.
            Defaults to ``True``.
    """
    # First check that the Jacobians are sparse
    if not isinstance(jacobians, SparseSystemJacobians):
        raise ValueError(f"Expected `SparseSystemJacobians` but got {type(jacobians)}.")

    # Optionally clear the output array if requested
    if reset_to_zero:
        v.zero_()

    # Then launch the kernel to compute the constraint-space velocities
    wp.launch(
        _compute_constraint_velocities_sparse,
        dim=(model.size.num_worlds, jacobians._J_cts.bsm.max_of_num_nzb),
        inputs=[
            model.info.bodies_offset,
            jacobians._J_cts.bsm.num_nzb,
            jacobians._J_cts.bsm.nzb_start,
            jacobians._J_cts.bsm.nzb_coords,
            jacobians._J_cts.bsm.nzb_values,
            v_start,
            u,
        ],
        outputs=[v],
        device=model.device,
    )


def compute_constraint_space_velocities(
    model: ModelKamino,
    jacobians: SystemJacobiansType,
    u: wp.array[wp.spatial_vectorf],
    v_start: wp.array[wp.int32],
    v: wp.array[wp.float32],
    reset_to_zero: bool = True,
) -> None:
    """
    Computes the constraint-space velocities `v = J_cts @ u_i`.

    This realizes the `v^{+/-} = J(q^{+/-}) @ u^{+/-}` operation.

    Depending on whether we use the pre-event (-) or post-event (+) state,
    we will compute the respective constraint-space velocities as:
    - `v^{+} = J(q) @ u^{+}`, i.e. `v^{+} := v_plus`
    - `v^{-} = J(q) @ u^{-}`, i.e. `v^{-} := v_minus`
    - All computations also depend on whether the pre- or post-event coordinates are use
        to evaluate the constraint Jacobian `J(q)`. However, for the purposes of evaluating
        physical correctness, we will use the system coordinates `q` that are coincident
        with the given state.

    This operation dispatches to the dense or sparse implementation based on the type of ``jacobians``.

    Args:
        model:
            The model containing the time-invariant data of the simulation.
        jacobians:
            The system Jacobians container providing ``J_cts``.
        u:
            The input per-body twists ``u_i``.
        v_start:
            The per-world starting offsets into the flat constraint-space output array.
        v:
            The output flat array to store the constraint-space velocities.
        reset_to_zero:
            Whether to reset the output array to zero before launching the kernel.
            Defaults to ``True``.
    """
    if isinstance(jacobians, DenseSystemJacobians):
        compute_constraint_velocities_dense(
            model=model, jacobians=jacobians, u=u, v_start=v_start, v=v, reset_to_zero=reset_to_zero
        )
    elif isinstance(jacobians, SparseSystemJacobians):
        compute_constraint_velocities_sparse(
            model=model, jacobians=jacobians, u=u, v_start=v_start, v=v, reset_to_zero=reset_to_zero
        )
    else:
        raise ValueError(f"Expected `DenseSystemJacobians` or `SparseSystemJacobians` but got {type(jacobians)}.")
