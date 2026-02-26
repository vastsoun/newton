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

import warp as wp

from ...core.math import FLOAT32_EPS, UNIT_Z, gravity_plus_coriolis_wrench, screw, screw_angular, screw_linear
from ...core.types import float32, int32, mat33f, vec2f, vec3f, vec4f, vec6f
from ..problem import ConstrainedDynamicsCfg

###
# Module interface
###

__all__ = [
    "_add_matrix_diag_product",
    "_apply_dual_preconditioner_to_matrix_dense",
    "_apply_dual_preconditioner_to_vector",
    "_build_dense_delassus_elementwise",
    "_build_dual_preconditioner_dense",
    "_build_dual_preconditioner_sparse",
    "_build_free_velocity_bias_contacts",
    "_build_free_velocity_bias_joints",
    "_build_free_velocity_bias_limits",
    "_build_free_velocity_dense",
    "_build_free_velocity_sparse",
    "_build_generalized_free_velocity",
    "_build_nonlinear_generalized_force",
    "_compute_delassus_diagonal_sparse",
    "_inverse_mass_matrix_matvec",
    "_regularize_dense_delassus_diagonal",
    "_scaled_vector_sum",
    "_set_matrix_diag_product",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _build_dense_delassus_elementwise(
    # Inputs:
    model_info_num_bodies: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    data_bodies_inv_I_i: wp.array(dtype=mat33f),
    jacobians_cts_start: wp.array(dtype=int32),
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
    cjmio = jacobians_cts_start[wid]

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
def _regularize_dense_delassus_diagonal(
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
    v_lin = wp.vec3(v[0], v[1], v[2])
    v_ang = wp.vec3(v[3], v[4], v[5])

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
def _compute_delassus_diagonal_sparse(
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


@wp.kernel
def _build_nonlinear_generalized_force(
    # Inputs:
    model_time_dt: wp.array(dtype=float32),
    model_gravity_vector: wp.array(dtype=vec4f),
    model_bodies_wid: wp.array(dtype=int32),
    model_bodies_m_i: wp.array(dtype=float32),
    state_bodies_u_i: wp.array(dtype=vec6f),
    state_bodies_I_i: wp.array(dtype=mat33f),
    state_bodies_w_e_i: wp.array(dtype=vec6f),
    state_bodies_w_a_i: wp.array(dtype=vec6f),
    # Outputs:
    problem_h: wp.array(dtype=vec6f),
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Retrieve the body model and data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    u_i = state_bodies_u_i[bid]
    w_e_i = state_bodies_w_e_i[bid]
    w_a_i = state_bodies_w_a_i[bid]

    # Get world data
    dt = model_time_dt[wid]
    gv = model_gravity_vector[wid]

    # Extract the effective gravity vector
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Extract the linear and angular components of the generalized velocity
    omega_i = screw_angular(u_i)

    # Compute the net external wrench on the body
    h_i = w_e_i + w_a_i + gravity_plus_coriolis_wrench(g, m_i, I_i, omega_i)

    # Store the generalized free-velocity vector
    problem_h[bid] = dt * h_i


@wp.kernel
def _build_generalized_free_velocity(
    # Inputs:
    model_time_dt: wp.array(dtype=float32),
    model_gravity_vector: wp.array(dtype=vec4f),
    model_bodies_wid: wp.array(dtype=int32),
    model_bodies_m_i: wp.array(dtype=float32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    state_bodies_u_i: wp.array(dtype=vec6f),
    state_bodies_I_i: wp.array(dtype=mat33f),
    state_bodies_inv_I_i: wp.array(dtype=mat33f),
    state_bodies_w_e_i: wp.array(dtype=vec6f),
    state_bodies_w_a_i: wp.array(dtype=vec6f),
    # Outputs:
    problem_u_f: wp.array(dtype=vec6f),
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Retrieve the body model and data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    inv_m_i = model_bodies_inv_m_i[bid]
    inv_I_i = state_bodies_inv_I_i[bid]
    u_i = state_bodies_u_i[bid]
    w_e_i = state_bodies_w_e_i[bid]
    w_a_i = state_bodies_w_a_i[bid]

    # Get world data
    dt = model_time_dt[wid]
    gv = model_gravity_vector[wid]

    # Extract the effective gravity vector
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Extract the linear and angular components of the generalized velocity
    v_i = screw_linear(u_i)
    omega_i = screw_angular(u_i)

    # Compute the net external wrench on the body
    h_i = w_e_i + w_a_i + gravity_plus_coriolis_wrench(g, m_i, I_i, omega_i)
    f_h_i = screw_linear(h_i)
    tau_h_i = screw_angular(h_i)

    # Compute the generalized free-velocity vector components
    v_f_i = v_i + dt * (inv_m_i * f_h_i)
    omega_f_i = omega_i + dt * (inv_I_i @ tau_h_i)

    # Store the generalized free-velocity vector
    problem_u_f[bid] = screw(v_f_i, omega_f_i)


@wp.kernel
def _build_free_velocity_bias_joints(
    # Inputs:
    model_info_joint_cts_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_num_cts: wp.array(dtype=int32),
    model_joints_cts_offset: wp.array(dtype=int32),
    state_joints_r_j: wp.array(dtype=float32),
    problem_config: wp.array(dtype=ConstrainedDynamicsCfg),
    problem_cts_start: wp.array(dtype=int32),
    # Outputs:
    problem_v_b: wp.array(dtype=float32),
):
    # Retrieve the joint index as the thread index
    jid = wp.tid()

    # Retrieve the world index from the joint
    wid = model_joints_wid[jid]

    # Retrieve the joint constraint index offset
    cts_offset = model_info_joint_cts_offset[wid]

    # Retrieve the model time step
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the dual problem config
    config = problem_config[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_cts_start[wid]

    # Retrieve the joint constraint index and offset
    ncts_j = model_joints_num_cts[jid]
    ctsio_j = model_joints_cts_offset[jid]

    # Compute baumgarte constraint stabilization coefficient
    c_b = config.alpha * inv_dt

    # Compute block offsets for the constraint and residual vectors
    vio_j = vio + ctsio_j
    rio_j = cts_offset + ctsio_j

    # Compute the free-velocity bias for the joint
    for j in range(ncts_j):
        problem_v_b[vio_j + j] = c_b * state_joints_r_j[rio_j + j]


@wp.kernel
def _build_free_velocity_bias_limits(
    # Inputs:
    model_time_inv_dt: wp.array(dtype=float32),
    state_info_limit_cts_group_offset: wp.array(dtype=int32),
    limits_model_max: int32,
    limits_model_num: wp.array(dtype=int32),
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_r_q: wp.array(dtype=float32),
    problem_config: wp.array(dtype=ConstrainedDynamicsCfg),
    problem_cts_start: wp.array(dtype=int32),
    # Outputs:
    problem_v_b: wp.array(dtype=float32),
):
    # Retrieve the limit index as the thread index
    tid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nl = wp.min(limits_model_num[0], limits_model_max)

    # Skip if cid is greater than the number of contacts active in the world
    if tid >= model_nl:
        return

    # Retrieve the limit entity data
    wid = limits_wid[tid]
    lid = limits_lid[tid]
    r_q = limits_r_q[tid]

    # Retrieve the world-specific data
    inv_dt = model_time_inv_dt[wid]
    config = problem_config[wid]
    vio = problem_cts_start[wid]
    lcio = state_info_limit_cts_group_offset[wid]

    # Compute the total constraint index offset of the current contact
    lcio_l = vio + lcio + lid

    # Compute the contact constraint stabilization bias
    problem_v_b[lcio_l] = config.beta * inv_dt * wp.min(0.0, r_q)


@wp.kernel
def _build_free_velocity_bias_contacts(
    # Inputs:
    model_time_inv_dt: wp.array(dtype=float32),
    model_info_contacts_offset: wp.array(dtype=int32),
    state_info_contact_cts_group_offset: wp.array(dtype=int32),
    contacts_model_max: int32,
    contacts_model_num: wp.array(dtype=int32),
    contacts_wid: wp.array(dtype=int32),
    contacts_cid: wp.array(dtype=int32),
    contacts_gapfunc: wp.array(dtype=vec4f),
    contacts_material: wp.array(dtype=vec2f),
    problem_config: wp.array(dtype=ConstrainedDynamicsCfg),
    problem_cts_start: wp.array(dtype=int32),
    # Outputs:
    problem_v_b: wp.array(dtype=float32),
    problem_v_i: wp.array(dtype=float32),
    problem_mu: wp.array(dtype=float32),
):
    # Retrieve the contact index as the thread index
    tid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nc = wp.min(contacts_model_num[0], contacts_model_max)

    # Skip if cid is greater than the number of contacts active in the world
    if tid >= model_nc:
        return

    # Retrieve the contact entity data
    wid_k = contacts_wid[tid]
    cid_k = contacts_cid[tid]
    material_k = contacts_material[tid]
    penetration_k = contacts_gapfunc[tid][3]

    # Retrieve the world-specific data
    inv_dt = model_time_inv_dt[wid_k]
    cio = model_info_contacts_offset[wid_k]
    ccio = state_info_contact_cts_group_offset[wid_k]
    vio = problem_cts_start[wid_k]
    config = problem_config[wid_k]

    # Compute the total constraint index offset of the current contact
    ccio_k = vio + ccio + 3 * cid_k

    # Compute the total contact index offset of the current contact
    cio_k = cio + cid_k

    # Retrieve the contact material properties
    mu_k = material_k.x  # Friction coefficient
    epsilon_k = material_k.y  # Penetration reduction coefficient

    # Compute the constraint residuals for unilateral contact constraints
    # NOTE#1: The residuals correspond to configuration-level constraint
    # violation of each contact along the corresponding normal direction
    # NOTE#2: contact penetration is assumed to be represented using
    # non-positive values (d <= 0), and since the penetration value
    # in the container is positive (p >= 0) we need to invert the sign
    # TODO: How to best use config.delta?
    distance_k = wp.min(0.0, penetration_k)

    # Compute the per-contact penetration error reduction term
    # NOTE#1: Penetrations are represented as xi < 0 (hence the sign inversion)
    # NOTE#2: xi_p_relaxed corresponds to one-sided Baumgarte-like stabilization
    xi = inv_dt * distance_k
    xi_relaxed = config.gamma * wp.min(0.0, xi) + wp.max(0.0, xi)
    if epsilon_k == 1.0:
        alpha = 0.0
    else:
        alpha = 1.0

    # Compute the contact constraint stabilization bias
    v_b_k = alpha * xi_relaxed * UNIT_Z

    # Store the contact constraint stabilization bias in the output vector
    for i in range(3):
        problem_v_b[ccio_k + i] = v_b_k[i]

    # Initialize the restitutive Newton-type impact model term
    problem_v_i[ccio_k] = 0.0
    problem_v_i[ccio_k + 1] = 0.0
    problem_v_i[ccio_k + 2] = epsilon_k

    # Store the contact friction coefficient in the output vector
    problem_mu[cio_k] = mu_k


@wp.kernel
def _build_free_velocity_dense(
    # Inputs:
    model_info_num_bodies: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    state_bodies_u_i: wp.array(dtype=vec6f),
    jacobians_J_cts_offsets: wp.array(dtype=int32),
    jacobians_J_cts_data: wp.array(dtype=float32),
    problem_cts_start: wp.array(dtype=int32),
    problem_cts_count: wp.array(dtype=int32),
    problem_u_f: wp.array(dtype=vec6f),
    problem_v_b: wp.array(dtype=float32),
    problem_v_i: wp.array(dtype=float32),
    # Outputs:
    problem_v_f: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the problem dimensions and matrix block index offset
    ncts = problem_cts_count[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the world-specific data
    cjmio = jacobians_J_cts_offsets[wid]
    bio = model_info_bodies_offset[wid]
    nb = model_info_num_bodies[wid]
    vio = problem_cts_start[wid]

    # Compute the number of Jacobian rows, i.e. the number of body DoFs
    nbd = 6 * nb

    # Compute the thread-specific index offset
    thread_offset = vio + tid

    # Append the column offset to the Jacobian index
    cjmio += nbd * tid

    # Extract the cached impact bias scaling (i.e. restitution coefficient)
    # NOTE: This is a quick hack to avoid multiple kernels. The
    # proper way would be to perform this op only for contacts
    epsilon_j = problem_v_i[thread_offset]

    # Buffers
    J_i = vec6f(0.0)
    v_f_j = float32(0.0)

    # Iterate over each body to accumulate velocity contributions
    for i in range(nb):
        # Compute the Jacobian block index
        m_ji = cjmio + 6 * i

        # Extract the twist and unconstrained velocity of the body
        u_i = state_bodies_u_i[bio + i]
        u_f_i = problem_u_f[bio + i]

        # Extract the Jacobian block J_ji
        # TODO: use slicing operation when available
        for d in range(6):
            J_i[d] = jacobians_J_cts_data[m_ji + d]

        # Accumulate J_i @ u_i
        v_f_j += wp.dot(J_i, u_f_i)

        # Accumulate the impact bias term
        v_f_j += epsilon_j * wp.dot(J_i, u_i)

    # Store sum of velocity bias terms
    problem_v_f[thread_offset] = v_f_j + problem_v_b[thread_offset]


@wp.kernel
def _build_free_velocity_sparse(
    # Inputs:
    model_info_bodies_offset: wp.array(dtype=int32),
    state_bodies_u_i: wp.array(dtype=vec6f),
    jac_num_nzb: wp.array(dtype=int32),
    jac_nzb_start: wp.array(dtype=int32),
    jac_nzb_coords: wp.array2d(dtype=int32),
    jac_nzb_values: wp.array(dtype=vec6f),
    problem_cts_start: wp.array(dtype=int32),
    problem_u_f: wp.array(dtype=vec6f),
    problem_v_i: wp.array(dtype=float32),
    # Outputs:
    problem_v_f: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, nzb_id = wp.tid()

    # Skip if block index exceed the number of blocks
    if nzb_id >= jac_num_nzb[wid]:
        return

    # Retrieve block data
    global_block_idx = jac_nzb_start[wid] + nzb_id
    jac_block_coord = jac_nzb_coords[global_block_idx]
    jac_block = jac_nzb_values[global_block_idx]

    # Retrieve the world-specific data
    bio = model_info_bodies_offset[wid]
    vio = problem_cts_start[wid]

    # Compute the thread-specific index offset
    thread_offset = vio + jac_block_coord[0]

    # Extract the cached impact bias scaling (i.e. restitution coefficient)
    # NOTE: This is a quick hack to avoid multiple kernels. The
    # proper way would be to perform this op only for contacts
    epsilon_j = problem_v_i[thread_offset]

    # Buffers
    v_f_j = float32(0.0)

    # Iterate over each body to accumulate velocity contributions
    bid = jac_block_coord[1] // 6

    # Extract the twist and unconstrained velocity of the body
    u_i = state_bodies_u_i[bio + bid]
    u_f_i = problem_u_f[bio + bid]

    # Accumulate J_i @ u_i
    v_f_j += wp.dot(jac_block, u_f_i)

    # Accumulate the impact bias term
    v_f_j += epsilon_j * wp.dot(jac_block, u_i)

    # Store sum of velocity bias terms
    wp.atomic_add(problem_v_f, thread_offset, v_f_j)


@wp.kernel
def _build_dual_preconditioner_dense(
    # Inputs:
    problem_config: wp.array(dtype=ConstrainedDynamicsCfg),
    problem_cts_start: wp.array(dtype=int32),
    problem_cts_count: wp.array(dtype=int32),
    problem_njc: wp.array(dtype=int32),
    problem_nl: wp.array(dtype=int32),
    problem_D_start: wp.array(dtype=int32),
    problem_D_data: wp.array(dtype=float32),
    # Outputs:
    problem_P: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the world-specific problem config
    config = problem_config[wid]

    # Retrieve the number of active constraints in the world
    ncts = problem_cts_count[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts or not config.preconditioning:
        return

    # Retrieve the matrix index offset of the world
    mio = problem_D_start[wid]

    # Retrieve the vector index offset of the world
    vio = problem_cts_start[wid]

    # Retrieve the number of active joint and limit constraints of the world
    njc = problem_njc[wid]
    nl = problem_nl[wid]
    njlc = njc + nl

    # Compute the preconditioner entry for the current constraint
    # First handle joint and limit constraints, then contact constraints
    if tid < njlc:
        # Retrieve the diagonal entry of the Delassus matrix
        D_ii = problem_D_data[mio + ncts * tid + tid]
        # Compute the corresponding Jacobi preconditioner entry
        problem_P[vio + tid] = wp.sqrt(1.0 / (wp.abs(D_ii) + FLOAT32_EPS))
    else:
        # Compute the contact constraint index
        ccid = tid - njlc
        # Only the thread of the first contact constraint dimension computes the preconditioner
        if ccid % 3 == 0:
            # Retrieve the diagonal entries of the Delassus matrix for the contact constraint set
            D_kk_0 = problem_D_data[mio + ncts * (tid + 0) + (tid + 0)]
            D_kk_1 = problem_D_data[mio + ncts * (tid + 1) + (tid + 1)]
            D_kk_2 = problem_D_data[mio + ncts * (tid + 2) + (tid + 2)]
            # Compute the effective diagonal entry
            # D_kk = (D_kk_0 + D_kk_1 + D_kk_2) / 3.0
            # D_kk = wp.min(vec3f(D_kk_0, D_kk_1, D_kk_2))
            D_kk = wp.max(vec3f(D_kk_0, D_kk_1, D_kk_2))
            # Compute the corresponding Jacobi preconditioner entry
            P_k = wp.sqrt(1.0 / (wp.abs(D_kk) + FLOAT32_EPS))
            problem_P[vio + tid] = P_k
            problem_P[vio + tid + 1] = P_k
            problem_P[vio + tid + 2] = P_k


@wp.kernel
def _build_dual_preconditioner_sparse(
    # Inputs:
    problem_config: wp.array(dtype=ConstrainedDynamicsCfg),
    problem_cts_start: wp.array(dtype=int32),
    problem_cts_count: wp.array(dtype=int32),
    problem_njc: wp.array(dtype=int32),
    problem_nl: wp.array(dtype=int32),
    # Outputs:
    problem_P: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the world-specific problem config
    config = problem_config[wid]

    # Retrieve the number of active constraints in the world
    ncts = problem_cts_count[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts or not config.preconditioning:
        return

    # Retrieve the vector index offset of the world
    vio = problem_cts_start[wid]

    # Retrieve the number of active joint and limit constraints of the world
    njc = problem_njc[wid]
    nl = problem_nl[wid]
    njlc = njc + nl

    # Compute the preconditioner entry for the current constraint
    # First handle joint and limit constraints, then contact constraints
    if tid < njlc:
        # Retrieve the diagonal entry of the Delassus matrix
        D_ii = problem_P[vio + tid]
        # Compute the corresponding Jacobi preconditioner entry
        problem_P[vio + tid] = wp.sqrt(1.0 / (wp.abs(D_ii) + FLOAT32_EPS))
    else:
        # Compute the contact constraint index
        ccid = tid - njlc
        # Only the thread of the first contact constraint dimension computes the preconditioner
        if ccid % 3 == 0:
            # Retrieve the diagonal entries of the Delassus matrix for the contact constraint set
            D_kk_0 = problem_P[vio + tid]
            D_kk_1 = problem_P[vio + tid + 1]
            D_kk_2 = problem_P[vio + tid + 2]
            # Compute the effective diagonal entry
            # D_kk = (D_kk_0 + D_kk_1 + D_kk_2) / 3.0
            # D_kk = wp.min(vec3f(D_kk_0, D_kk_1, D_kk_2))
            D_kk = wp.max(vec3f(D_kk_0, D_kk_1, D_kk_2))
            # Compute the corresponding Jacobi preconditioner entry
            P_k = wp.sqrt(1.0 / (wp.abs(D_kk) + FLOAT32_EPS))
            problem_P[vio + tid] = P_k
            problem_P[vio + tid + 1] = P_k
            problem_P[vio + tid + 2] = P_k


@wp.kernel
def _apply_dual_preconditioner_to_matrix_dense(
    # Inputs:
    problem_cts_start: wp.array(dtype=int32),
    problem_cts_count: wp.array(dtype=int32),
    problem_P: wp.array(dtype=float32),
    A_start: wp.array(dtype=int32),
    # Outputs:
    A_data: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_cts_count[wid]

    # Skip if there no constraints ar active
    if ncts == 0:
        return

    # Compute i (row) and j (col) indices from the tid
    i = tid // ncts
    j = tid % ncts

    # Skip if indices exceed the problem size
    if i >= ncts or j >= ncts:
        return

    # Retrieve the matrix index offset of the world
    mio = A_start[wid]

    # Retrieve the vector index offset of the world
    vio = problem_cts_start[wid]

    # Compute the global index of the matrix entry
    m_ij = mio + ncts * i + j

    # Retrieve the i,j-th entry of the target matrix
    X_ij = A_data[m_ij]

    # Retrieve the i,j-th entries of the diagonal preconditioner
    P_i = problem_P[vio + i]
    P_j = problem_P[vio + j]

    # Store the preconditioned i,j-th entry of the matrix
    A_data[m_ij] = P_i * (P_j * X_ij)


@wp.kernel
def _apply_dual_preconditioner_to_vector(
    # Inputs:
    problem_cts_start: wp.array(dtype=int32),
    problem_cts_count: wp.array(dtype=int32),
    problem_P: wp.array(dtype=float32),
    # Outputs:
    x: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_cts_count[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the vector index offset of the world
    vio = problem_cts_start[wid]

    # Compute the global index of the vector entry
    v_i = vio + tid

    # Retrieve the i-th entry of the target vector
    x_i = x[v_i]

    # Retrieve the i-th entry of the diagonal preconditioner
    P_i = problem_P[v_i]

    # Store the preconditioned i-th entry of the vector
    x[v_i] = P_i * x_i
