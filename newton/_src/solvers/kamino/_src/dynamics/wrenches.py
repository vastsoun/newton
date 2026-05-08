# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Dynamics: Wrenches
"""

import warp as wp

from ..core.control import ControlKamino
from ..core.data import DataKamino
from ..core.joints import JointDoFType
from ..core.math import (
    concat6d,
    expand6d,
    screw_transform_matrix_from_points,
)
from ..core.model import ModelKamino
from ..core.types import float32, int32, mat33f, mat63f, transformf, vec2i, vec3f, vec6f
from ..geometry.contacts import ContactsKamino
from ..kinematics.jacobians import (
    DenseSystemJacobians,
    SparseSystemJacobians,
    compute_intermediate_body_frame_universal_joint,
    compute_joint_relative_quaternion,
)
from ..kinematics.limits import LimitsKamino

###
# Module interface
###

__all__ = [
    "compute_constraint_body_wrenches",
    "compute_joint_dof_body_wrenches",
    "convert_body_parent_wrenches_to_joint_reactions",
    "convert_joint_wrenches_to_body_parent_wrenches",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def joint_dof_axis_from_index(dof_type: int32, dof_within_joint: int32) -> int32:
    """
    Maps a joint's local DoF index (i.e. ``dof_within_joint``) to the corresponding
    6D axis index in the joint frame, based on the joint's DoF type.
    """
    if dof_type == JointDoFType.REVOLUTE:
        return 3
    elif dof_type == JointDoFType.PRISMATIC:
        return 0
    elif dof_type == JointDoFType.CYLINDRICAL:
        # CYLINDRICAL DoFs are: T_x (axis 0), R_x (axis 3)
        if dof_within_joint == 0:
            return 0
        return 3
    elif dof_type == JointDoFType.UNIVERSAL:
        return 3 + dof_within_joint
    elif dof_type == JointDoFType.SPHERICAL:
        return 3 + dof_within_joint
    elif dof_type == JointDoFType.GIMBAL:
        return 3 + dof_within_joint
    elif dof_type == JointDoFType.CARTESIAN:
        return dof_within_joint
    elif dof_type == JointDoFType.FREE:
        return dof_within_joint
    return -1


def make_typed_write_joint_kinematic_lambdas(dof_type: JointDoFType):
    """
    Generates a per-joint-type Warp function that writes the kinematic-constraint
    Lagrange multipliers of a single joint into the global ``state_lambda_j`` array.
    """
    cts_axes = dof_type.cts_axes
    num_cts = dof_type.num_cts

    @wp.func
    def _typed_write_joint_kinematic_lambdas(
        cts_offset_j: int32,
        j_w_j: vec6f,
        state_lambda_j: wp.array[float32],
    ):
        for k in range(num_cts):
            state_lambda_j[cts_offset_j + k] = j_w_j[cts_axes[k]]

    return _typed_write_joint_kinematic_lambdas


def make_write_joint_kinematic_lambdas():
    """
    Generates a Warp function that dispatches the per-joint-type writer of the
    joint's kinematic-constraint Lagrange multipliers, based on the joint's DoF type.
    """

    @wp.func
    def _write_joint_kinematic_lambdas(
        dof_type: int32,
        cts_offset_j: int32,
        j_w_j: vec6f,
        state_lambda_j: wp.array[float32],
    ):
        if dof_type == JointDoFType.REVOLUTE:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.REVOLUTE))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.PRISMATIC:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.PRISMATIC))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.CYLINDRICAL:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.CYLINDRICAL))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.UNIVERSAL:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.UNIVERSAL))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.SPHERICAL:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.SPHERICAL))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.GIMBAL:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.GIMBAL))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.CARTESIAN:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.CARTESIAN))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.FIXED:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.FIXED))(cts_offset_j, j_w_j, state_lambda_j)
        # FREE: no kinematic constraints; nothing to write

    return _write_joint_kinematic_lambdas


###
# Kernels
###


@wp.kernel
def _compute_joint_dof_body_wrenches_dense(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    model_info_joint_dofs_offset: wp.array[int32],
    model_joints_dofs_offset: wp.array[int32],
    model_joints_wid: wp.array[int32],
    model_joints_bid_B: wp.array[int32],
    model_joints_bid_F: wp.array[int32],
    data_joints_tau_j: wp.array[float32],
    jacobian_dofs_offsets: wp.array[int32],
    jacobian_dofs_data: wp.array[float32],
    # Outputs:
    data_bodies_w_a: wp.array[vec6f],
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the world index of the joint
    wid = model_joints_wid[jid]

    # Retrieve the body indices of the joint
    # NOTE: these indices are w.r.t the model
    bid_F_j = model_joints_bid_F[jid]
    bid_B_j = model_joints_bid_B[jid]

    # Retrieve the size and index offset of the joint DoFs
    dio_j = model_joints_dofs_offset[jid]
    d_j = model_joints_dofs_offset[jid + 1] - dio_j

    # Retrieve the element index offset of the bodies of the world
    bio = model_info_bodies_offset[wid]

    # Compute the number of body DoFs in the world
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)

    # Compute the DoF block index offsets of the world's actuation
    # Jacobian matrix and generalized joint actuation force vector
    mio = jacobian_dofs_offsets[wid]
    dio_j_world = dio_j - model_info_joint_dofs_offset[wid]
    mio += nbd * dio_j_world
    vio = dio_j

    # Compute and store the joint actuation wrench for the Follower body
    w_j_F = vec6f(0.0)
    dio_F = 6 * (bid_F_j - bio)
    for j in range(d_j):
        mio_j = mio + nbd * j + dio_F
        vio_j = vio + j
        tau_j = data_joints_tau_j[vio_j]
        for i in range(6):
            w_j_F[i] += jacobian_dofs_data[mio_j + i] * tau_j
    wp.atomic_add(data_bodies_w_a, bid_F_j, w_j_F)

    # Compute and store the joint actuation wrench for the Base body if bid_B >= 0
    if bid_B_j >= 0:
        w_j_B = vec6f(0.0)
        dio_B = 6 * (bid_B_j - bio)
        for j in range(d_j):
            mio_j = mio + nbd * j + dio_B
            vio_j = vio + j
            tau_j = data_joints_tau_j[vio_j]
            for i in range(6):
                w_j_B[i] += jacobian_dofs_data[mio_j + i] * tau_j
        wp.atomic_add(data_bodies_w_a, bid_B_j, w_j_B)


@wp.kernel
def _compute_joint_dof_body_wrenches_sparse(
    # Inputs:
    model_joints_num_dofs: wp.array[int32],
    model_joints_dofs_offset: wp.array[int32],
    model_joints_wid: wp.array[int32],
    model_joints_bid_B: wp.array[int32],
    model_joints_bid_F: wp.array[int32],
    data_joints_tau_j: wp.array[float32],
    jac_joint_nzb_offsets: wp.array[int32],
    jac_nzb_values: wp.array[vec6f],
    # Outputs:
    data_bodies_w_a: wp.array[vec6f],
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the body indices of the joint
    # NOTE: these indices are w.r.t the model
    bid_F_j = model_joints_bid_F[jid]
    bid_B_j = model_joints_bid_B[jid]

    # Retrieve the size and index offset of the joint DoFs
    d_j = model_joints_num_dofs[jid]
    dio_j = model_joints_dofs_offset[jid]

    # Retrieve the starting index for the non-zero blocks for the current joint
    jac_j_nzb_start = jac_joint_nzb_offsets[jid]

    # Compute and store the joint actuation wrench for the Follower body
    w_j_F = vec6f(0.0)
    for j in range(d_j):
        jac_block = jac_nzb_values[jac_j_nzb_start + j]
        vio_j = dio_j + j
        tau_j = data_joints_tau_j[vio_j]
        w_j_F += jac_block * tau_j
    wp.atomic_add(data_bodies_w_a, bid_F_j, w_j_F)

    # Compute and store the joint actuation wrench for the Base body if bid_B >= 0
    if bid_B_j >= 0:
        w_j_B = vec6f(0.0)
        for j in range(d_j):
            jac_block = jac_nzb_values[jac_j_nzb_start + d_j + j]
            vio_j = dio_j + j
            tau_j = data_joints_tau_j[vio_j]
            w_j_B += jac_block * tau_j
        wp.atomic_add(data_bodies_w_a, bid_B_j, w_j_B)


@wp.kernel
def _compute_joint_cts_body_wrenches_dense(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    model_info_joint_dynamic_cts_offset: wp.array[int32],
    model_info_joint_kinematic_cts_offset: wp.array[int32],
    model_info_joint_dynamic_cts_group_offset: wp.array[int32],
    model_info_joint_kinematic_cts_group_offset: wp.array[int32],
    model_time_inv_dt: wp.array[float32],
    model_joints_wid: wp.array[int32],
    model_joints_dynamic_cts_offset: wp.array[int32],
    model_joints_kinematic_cts_offset: wp.array[int32],
    model_joints_bid_B: wp.array[int32],
    model_joints_bid_F: wp.array[int32],
    jacobian_cts_offset: wp.array[int32],
    jacobian_cts_data: wp.array[float32],
    lambdas_offsets: wp.array[int32],
    lambdas_data: wp.array[float32],
    # Outputs:
    data_bodies_w_j: wp.array[vec6f],
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the world index of the joint
    wid = model_joints_wid[jid]

    # Retrieve the body indices of the joint
    # NOTE: these indices are w.r.t the model
    bid_F_j = model_joints_bid_F[jid]
    bid_B_j = model_joints_bid_B[jid]

    # Retrieve the size and index offset of the joint constraint
    dyn_cts_start_j = model_joints_dynamic_cts_offset[jid]
    num_dyn_cts_j = model_joints_dynamic_cts_offset[jid + 1] - dyn_cts_start_j
    kin_cts_start_j = model_joints_kinematic_cts_offset[jid]
    num_kin_cts_j = model_joints_kinematic_cts_offset[jid + 1] - kin_cts_start_j

    # Retrieve the element index offset of the bodies of the world
    bio = model_info_bodies_offset[wid]

    # Compute the number of body DoFs in the world
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)

    # Retrieve the index offsets of the active joint dynamic and kinematic constraints of the world
    world_jdcgo = model_info_joint_dynamic_cts_group_offset[wid]
    world_jkcgo = model_info_joint_kinematic_cts_group_offset[wid]

    # Compute local (within-world) constraint offsets for Jacobian matrix indexing
    local_dyn_cts_start_j = dyn_cts_start_j - model_info_joint_dynamic_cts_offset[wid]
    local_kin_cts_start_j = kin_cts_start_j - model_info_joint_kinematic_cts_offset[wid]

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the constraint block index offsets of the
    # Jacobian matrix and multipliers vector of the world
    world_jacobian_start = jacobian_cts_offset[wid]
    world_cts_start = lambdas_offsets[wid]

    # Compute and store the joint constraint wrench for the Follower body
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    w_j_F = vec6f(0.0)
    col_F_start = 6 * (bid_F_j - bio)
    for j in range(num_dyn_cts_j):
        row_j = world_jdcgo + local_dyn_cts_start_j + j
        mio_j = world_jacobian_start + nbd * row_j + col_F_start
        vio_j = world_cts_start + row_j
        lambda_j = inv_dt * lambdas_data[vio_j]
        for i in range(6):
            w_j_F[i] += jacobian_cts_data[mio_j + i] * lambda_j
    for j in range(num_kin_cts_j):
        row_j = world_jkcgo + local_kin_cts_start_j + j
        mio_j = world_jacobian_start + nbd * row_j + col_F_start
        vio_j = world_cts_start + row_j
        lambda_j = inv_dt * lambdas_data[vio_j]
        for i in range(6):
            w_j_F[i] += jacobian_cts_data[mio_j + i] * lambda_j
    wp.atomic_add(data_bodies_w_j, bid_F_j, w_j_F)

    # Compute and store the joint constraint wrench for the Base body if bid_B >= 0
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    if bid_B_j >= 0:
        w_j_B = vec6f(0.0)
        col_B_start = 6 * (bid_B_j - bio)
        for j in range(num_dyn_cts_j):
            row_j = world_jdcgo + local_dyn_cts_start_j + j
            mio_j = world_jacobian_start + nbd * row_j + col_B_start
            vio_j = world_cts_start + row_j
            lambda_j = inv_dt * lambdas_data[vio_j]
            for i in range(6):
                w_j_B[i] += jacobian_cts_data[mio_j + i] * lambda_j
        for j in range(num_kin_cts_j):
            row_j = world_jkcgo + local_kin_cts_start_j + j
            mio_j = world_jacobian_start + nbd * row_j + col_B_start
            vio_j = world_cts_start + row_j
            lambda_j = inv_dt * lambdas_data[vio_j]
            for i in range(6):
                w_j_B[i] += jacobian_cts_data[mio_j + i] * lambda_j
        wp.atomic_add(data_bodies_w_j, bid_B_j, w_j_B)


@wp.kernel
def _compute_limit_cts_body_wrenches_dense(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    data_info_limit_cts_group_offset: wp.array[int32],
    model_time_inv_dt: wp.array[float32],
    limits_model_num: wp.array[int32],
    limits_model_max: int32,
    limits_wid: wp.array[int32],
    limits_lid: wp.array[int32],
    limits_bids: wp.array[vec2i],
    jacobian_cts_offset: wp.array[int32],
    jacobian_cts_data: wp.array[float32],
    lambdas_offsets: wp.array[int32],
    lambdas_data: wp.array[float32],
    # Outputs:
    data_bodies_w_l: wp.array[vec6f],
):
    # Retrieve the thread index
    tid = wp.tid()

    # Skip if tid is greater than the number of active limits in the model
    if tid >= wp.min(limits_model_num[0], limits_model_max):
        return

    # Retrieve the limit index of the limit w.r.t the world
    lid = limits_lid[tid]

    # Retrieve the world index of the limit
    wid = limits_wid[tid]

    # Extract the body indices associated with the limit
    # NOTE: These indices are w.r.t the model
    bids = limits_bids[tid]
    bid_B = bids[0]
    bid_F = bids[1]

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the world-specific info
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)
    mio = jacobian_cts_offset[wid]
    vio = lambdas_offsets[wid]

    # Retrieve the index offset of the active limit constraints of the world
    lcgo = data_info_limit_cts_group_offset[wid]

    # Compute the index offsets of the limit constraint
    cio_l = lcgo + lid
    vio_l = vio + cio_l
    mio_l = mio + nbd * cio_l

    # Extract the limit force/torque from the impulse
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    lambda_l = inv_dt * lambdas_data[vio_l]

    # Extract the limit constraint Jacobian for the follower body
    JT_l_F = vec6f(0.0)
    dio_F = 6 * (bid_F - bio)
    mio_lF = mio_l + dio_F
    for i in range(6):
        JT_l_F[i] = jacobian_cts_data[mio_lF + i]

    # Compute the limit constraint wrench for the follower body
    w_l_F = JT_l_F * lambda_l

    # Store the limit constraint wrench for the follower body
    wp.atomic_add(data_bodies_w_l, bid_F, w_l_F)

    # Compute the limit constraint wrench for the joint base body if bid_B >= 0
    if bid_B >= 0:
        # Extract the limit constraint Jacobian for the base body
        JT_l_B = vec6f(0.0)
        dio_B = 6 * (bid_B - bio)
        mio_lB = mio_l + dio_B
        for i in range(6):
            JT_l_B[i] = jacobian_cts_data[mio_lB + i]

        # Compute the limit constraint wrench for the base body
        w_l_B = JT_l_B * lambda_l

        # Store the limit constraint wrench for the base body
        wp.atomic_add(data_bodies_w_l, bid_B, w_l_B)


@wp.kernel
def _compute_contact_cts_body_wrenches_dense(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    data_info_contact_cts_group_offset: wp.array[int32],
    model_time_inv_dt: wp.array[float32],
    contacts_model_num: wp.array[int32],
    contacts_model_max: int32,
    contacts_wid: wp.array[int32],
    contacts_cid: wp.array[int32],
    contacts_bid_AB: wp.array[vec2i],
    jacobian_cts_offset: wp.array[int32],
    jacobian_cts_data: wp.array[float32],
    lambdas_offsets: wp.array[int32],
    lambdas_data: wp.array[float32],
    # Outputs:
    data_bodies_w_c: wp.array[vec6f],
):
    # Retrieve the thread index
    tid = wp.tid()

    # Skip if tid is greater than the number of active contacts in the model
    if tid >= wp.min(contacts_model_num[0], contacts_model_max):
        return

    # Retrieve the contact index of the contact w.r.t the world
    cid = contacts_cid[tid]

    # Retrieve the world index of the contact
    wid = contacts_wid[tid]

    # Extract the body indices associated with the contact
    # NOTE: These indices are w.r.t the model
    bid_AB = contacts_bid_AB[tid]
    bid_A = bid_AB[0]
    bid_B = bid_AB[1]

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the world-specific info data
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)
    mio = jacobian_cts_offset[wid]
    vio = lambdas_offsets[wid]

    # Retrieve the index offset of the active contact constraints of the world
    ccgo = data_info_contact_cts_group_offset[wid]

    # Compute the index offsets of the contact constraint
    k = 3 * cid
    cio_k = ccgo + k
    vio_k = vio + cio_k
    mio_k = mio + nbd * cio_k

    # Extract the 3D contact force
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    lambda_c = inv_dt * vec3f(lambdas_data[vio_k], lambdas_data[vio_k + 1], lambdas_data[vio_k + 2])

    # Extract the contact constraint Jacobian for body B
    JT_c_B = mat63f(0.0)
    dio_B = 6 * (bid_B - bio)
    for j in range(3):
        mio_kj = mio_k + nbd * j + dio_B
        for i in range(6):
            JT_c_B[i, j] = jacobian_cts_data[mio_kj + i]

    # Compute the contact constraint wrench for body B
    w_c_B = JT_c_B @ lambda_c

    # Store the contact constraint wrench for body B
    wp.atomic_add(data_bodies_w_c, bid_B, w_c_B)

    # Compute the contact constraint wrench for body A if bid_A >= 0
    if bid_A >= 0:
        # Extract the contact constraint Jacobian for body A
        JT_c_A = mat63f(0.0)
        dio_A = 6 * (bid_A - bio)
        for j in range(3):
            mio_kj = mio_k + nbd * j + dio_A
            for i in range(6):
                JT_c_A[i, j] = jacobian_cts_data[mio_kj + i]

        # Compute the contact constraint wrench for body A
        w_c_A = JT_c_A @ lambda_c

        # Store the contact constraint wrench for body A
        wp.atomic_add(data_bodies_w_c, bid_A, w_c_A)


@wp.kernel
def _compute_cts_body_wrenches_sparse(
    # Inputs:
    model_time_inv_dt: wp.array[float32],
    model_info_bodies_offset: wp.array[int32],
    data_info_limit_cts_group_offset: wp.array[int32],
    data_info_contact_cts_group_offset: wp.array[int32],
    jac_num_nzb: wp.array[int32],
    jac_nzb_start: wp.array[int32],
    jac_nzb_coords: wp.array2d[int32],
    jac_nzb_values: wp.array[vec6f],
    lambdas_offsets: wp.array[int32],
    lambdas_data: wp.array[float32],
    # Outputs:
    data_bodies_w_j_i: wp.array[vec6f],
    data_bodies_w_l_i: wp.array[vec6f],
    data_bodies_w_c_i: wp.array[vec6f],
):
    # Retrieve the world and non-zero
    # block indices from the thread grid
    wid, nzbid = wp.tid()

    # Skip if the non-zero block index is greater than
    # the number of active non-zero blocks for the world
    if nzbid >= jac_num_nzb[wid]:
        return

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve world-specific index offsets
    world_bid_start = model_info_bodies_offset[wid]
    J_cts_nzb_start = jac_nzb_start[wid]
    world_cts_start = lambdas_offsets[wid]
    limit_cts_group_start = data_info_limit_cts_group_offset[wid]
    contact_cts_group_start = data_info_contact_cts_group_offset[wid]

    # Retrieve the Jacobian matrix block coordinates
    # and values for the current non-zero block
    global_nzb_idx = J_cts_nzb_start + nzbid
    J_ji_coords = jac_nzb_coords[global_nzb_idx]
    J_ji = jac_nzb_values[global_nzb_idx]

    # Get constraint and body from the block coordinates
    cts_row = J_ji_coords[0]
    bid_j = J_ji_coords[1] // 6

    # Get global body index, i.e. w.r.t the model
    global_bid_j = world_bid_start + bid_j

    # Retrieve the constraint reaction of the current constraint row
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    lambda_j = inv_dt * lambdas_data[world_cts_start + cts_row]

    # Compute the joint constraint wrench for the body
    w_ij = lambda_j * J_ji

    # Add the wrench to the appropriate array
    if cts_row >= contact_cts_group_start:
        wp.atomic_add(data_bodies_w_c_i, global_bid_j, w_ij)
    elif cts_row >= limit_cts_group_start:
        wp.atomic_add(data_bodies_w_l_i, global_bid_j, w_ij)
    else:
        wp.atomic_add(data_bodies_w_j_i, global_bid_j, w_ij)


@wp.kernel
def _convert_joint_wrenches_to_body_parent_wrenches(
    # Inputs:
    model_joints_dof_type: wp.array[int32],
    model_joints_bid_F: wp.array[int32],
    state_body_w_a_i: wp.array[vec6f],
    state_body_w_j_i: wp.array[vec6f],
    state_body_w_l_i: wp.array[vec6f],
    # Outputs:
    body_parent_f: wp.array[wp.spatial_vectorf],
):
    # Retrieve the joint index from the thread grid
    jid = wp.tid()

    # Retrieve the DoF type of the joint
    dof_type_j = model_joints_dof_type[jid]

    # Skip if the joint is a FREE joint because otherwise
    # this will include purely external body forces
    if dof_type_j == JointDoFType.FREE:
        return

    # Retrieve the follower body index
    bid_F = model_joints_bid_F[jid]

    # Retrieve the body-specific data
    w_a_F = state_body_w_a_i[bid_F]
    w_j_F = state_body_w_j_i[bid_F]
    w_l_F = state_body_w_l_i[bid_F]

    # Compute the net wrench applied to the
    # follower body from the current joint
    w_F_j = w_a_F + w_j_F + w_l_F

    # Accumulate the wrench into the body parent wrench array
    wp.atomic_add(body_parent_f, bid_F, w_F_j)


@wp.kernel
def _compute_joint_wrenches_from_body_parent_wrenches(
    # Inputs:
    model_joints_dof_type: wp.array[int32],
    model_joints_kinematic_cts_offset_joint_cts: wp.array[int32],
    model_joints_bid_F: wp.array[int32],
    model_joints_bid_B: wp.array[int32],
    model_joints_X_j: wp.array[mat33f],
    data_joints_p_j: wp.array[transformf],
    data_bodies_q_i: wp.array[transformf],
    body_parent_f: wp.array[wp.spatial_vectorf],
    # Outputs:
    data_joints_j_w_j: wp.array[vec6f],
    data_joints_lambda_j: wp.array[float32],
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint model data
    dof_type = model_joints_dof_type[jid]

    # Skip FREE joints: they have no kinematic constraints
    # and `joint_parent_f`is not accumulated for FREE joints
    if dof_type == JointDoFType.FREE:
        return

    # Retrieve the body indices of the joint
    bid_F = model_joints_bid_F[jid]
    bid_B = model_joints_bid_B[jid]

    # Retrieve the joint frame pose (in world coords)
    T_j = data_joints_p_j[jid]
    r_j = wp.transform_get_translation(T_j)
    R_X_j = wp.quat_to_matrix(wp.transform_get_rotation(T_j))

    # Retrieve the follower body's pose (CoM in world coords)
    T_F_j = data_bodies_q_i[bid_F]
    r_F_j = wp.transform_get_translation(T_F_j)

    # Compute the inverse wrench-transform from the follower CoM to the joint frame.
    # Since `W_j_F = screw_transform_matrix_from_points(r_j, r_F_j)` transforms a wrench
    # from the joint frame to the body's CoM, its inverse swaps the role of the two points.
    inv_W_j_F = screw_transform_matrix_from_points(r_F_j, r_j)

    # General case: 6D extension of the constant joint-frame rotation matrix
    if dof_type != JointDoFType.UNIVERSAL:
        R_X_bar_j = expand6d(R_X_j)
    # Universal joint: replace R_X_j with the frame of the intermediate body for rotation constraints
    else:
        # The base body's pose is needed to compute the relative quaternion;
        # for unary joints (bid_B == -1), use the world identity transform.
        T_B_j = wp.transform_identity()
        if bid_B > -1:
            T_B_j = data_bodies_q_i[bid_B]
        j_q_j = compute_joint_relative_quaternion(T_B_j, T_F_j, model_joints_X_j[jid])
        R_intermediate = compute_intermediate_body_frame_universal_joint(j_q_j)
        R_X_bar_j = concat6d(R_X_j, R_X_j @ R_intermediate)

    # Read the world-frame wrench applied on body F by joint j (at body F's CoM).
    w_ij_sv = body_parent_f[bid_F]
    w_ij = vec6f(w_ij_sv[0], w_ij_sv[1], w_ij_sv[2], w_ij_sv[3], w_ij_sv[4], w_ij_sv[5])

    # Transform the wrench from body-F CoM to the joint frame (world-aligned),
    # then express it in the joint-local frame.
    w_j = inv_W_j_F @ w_ij
    j_w_j = wp.transpose(R_X_bar_j) @ w_j

    # Store the joint-local wrench
    data_joints_j_w_j[jid] = j_w_j

    # Write the kinematic-constraint Lagrange multipliers for this joint
    cts_offset_j = model_joints_kinematic_cts_offset_joint_cts[jid]
    wp.static(make_write_joint_kinematic_lambdas())(dof_type, cts_offset_j, j_w_j, data_joints_lambda_j)


@wp.kernel
def _compute_joint_wrenches_from_joint_parent_wrenches(
    # Inputs:
    model_joints_dof_type: wp.array[int32],
    model_joints_kinematic_cts_offset_joint_cts: wp.array[int32],
    model_joints_bid_F: wp.array[int32],
    model_joints_bid_B: wp.array[int32],
    model_joints_X_j: wp.array[mat33f],
    data_joints_p_j: wp.array[transformf],
    data_bodies_q_i: wp.array[transformf],
    joint_parent_f: wp.array[wp.spatial_vectorf],
    # Outputs:
    data_joints_j_w_j: wp.array[vec6f],
    data_lambda_j: wp.array[float32],
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint model data
    dof_type = model_joints_dof_type[jid]

    # Skip FREE joints: they have no kinematic constraints
    # and `joint_parent_f`is not accumulated for FREE joints
    if dof_type == JointDoFType.FREE:
        return

    # Retrieve the body indices of the joint
    bid_F = model_joints_bid_F[jid]
    bid_B = model_joints_bid_B[jid]

    # Retrieve the joint frame pose (in world coords)
    T_j = data_joints_p_j[jid]
    r_j = wp.transform_get_translation(T_j)
    R_X_j = wp.quat_to_matrix(wp.transform_get_rotation(T_j))

    # Retrieve the follower body's pose (CoM in world coords)
    T_F_j = data_bodies_q_i[bid_F]
    r_F_j = wp.transform_get_translation(T_F_j)

    # Compute the inverse wrench-transform from the follower CoM to the joint frame.
    # Since `W_j_F = screw_transform_matrix_from_points(r_j, r_F_j)` transforms a wrench
    # from the joint frame to the body's CoM, its inverse swaps the role of the two points.
    inv_W_j_F = screw_transform_matrix_from_points(r_F_j, r_j)

    # General case: 6D extension of the constant joint-frame rotation matrix
    if dof_type != JointDoFType.UNIVERSAL:
        R_X_bar_j = expand6d(R_X_j)
    # Universal joint: replace R_X_j with the frame of the intermediate body for rotation constraints
    else:
        # The base body's pose is needed to compute the relative quaternion;
        # for unary joints (bid_B == -1), use the world identity transform.
        T_B_j = wp.transform_identity()
        if bid_B > -1:
            T_B_j = data_bodies_q_i[bid_B]
        j_q_j = compute_joint_relative_quaternion(T_B_j, T_F_j, model_joints_X_j[jid])
        R_intermediate = compute_intermediate_body_frame_universal_joint(j_q_j)
        R_X_bar_j = concat6d(R_X_j, R_X_j @ R_intermediate)

    # Read the world-frame wrench applied on body F by joint j (at body F's CoM).
    w_ij_sv = joint_parent_f[jid]
    w_ij = vec6f(w_ij_sv[0], w_ij_sv[1], w_ij_sv[2], w_ij_sv[3], w_ij_sv[4], w_ij_sv[5])

    # Transform the wrench from body-F CoM to the joint frame (world-aligned),
    # then express it in the joint-local frame.
    w_j = inv_W_j_F @ w_ij
    j_w_j = wp.transpose(R_X_bar_j) @ w_j

    # Store the joint-local wrench
    data_joints_j_w_j[jid] = j_w_j

    # Write the kinematic-constraint Lagrange multipliers for this joint
    cts_offset_j = model_joints_kinematic_cts_offset_joint_cts[jid]
    wp.static(make_write_joint_kinematic_lambdas())(dof_type, cts_offset_j, j_w_j, data_lambda_j)


@wp.kernel
def _compute_limit_reactions_from_joint_wrenches(
    # Inputs:
    model_joints_dof_type: wp.array[int32],
    model_joints_dofs_offset: wp.array[int32],
    limits_model_num: wp.array[int32],
    limits_model_max: int32,
    limits_jid: wp.array[int32],
    limits_dof: wp.array[int32],
    limits_side: wp.array[float32],
    data_joints_j_w_j: wp.array[vec6f],
    control_tau_j: wp.array[float32],
    # Outputs:
    limits_reaction: wp.array[float32],
):
    # Retrieve the limit index from the thread grid
    lid = wp.tid()

    # Skip if lid is greater than the number of active limits in the model
    if lid >= wp.min(limits_model_num[0], limits_model_max):
        return

    # Retrieve the joint and DoF indices for this active limit
    jid = limits_jid[lid]
    dof_l = limits_dof[lid]
    side_l = limits_side[lid]

    # Map the global DoF index to the joint-local DoF index, then to the 6D joint-frame axis
    dof_within_joint = dof_l - model_joints_dofs_offset[jid]
    axis = joint_dof_axis_from_index(model_joints_dof_type[jid], dof_within_joint)

    # Recover the limit reaction: the joint-frame total wrench at the DoF axis is
    # `tau_total = tau_actuation + side * lambda_l`, so `lambda_l = side * (tau_total - tau_actuation)`.
    j_w_j = data_joints_j_w_j[jid]
    tau_total = j_w_j[axis]
    tau_act = control_tau_j[dof_l]
    limits_reaction[lid] = side_l * (tau_total - tau_act)


###
# Launchers
###


def compute_joint_dof_body_wrenches_dense(
    model: ModelKamino, data: DataKamino, jacobians: DenseSystemJacobians, reset_to_zero: bool = True
):
    """
    Update the actuation wrenches of the bodies based on the active joint torques.
    """
    # First check that the Jacobians are dense
    if not isinstance(jacobians, DenseSystemJacobians):
        raise ValueError(f"Expected `DenseSystemJacobians` but got {type(jacobians)}.")

    # Clear the previous actuation wrenches, because the kernel computing them
    # uses an atomic add to accumulate contributions from each joint DoF, and
    # thus assumes the target array is zeroed out before each call
    if reset_to_zero:
        data.bodies.w_a_i.zero_()

    # Then compute the body wrenches resulting from the current generalized actuation forces
    wp.launch(
        _compute_joint_dof_body_wrenches_dense,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.info.bodies_offset,
            model.info.joint_dofs_offset,
            model.joints.dofs_offset,
            model.joints.wid,
            model.joints.bid_B,
            model.joints.bid_F,
            data.joints.tau_j,
            jacobians.data.J_dofs_offsets,
            jacobians.data.J_dofs_data,
            # Outputs:
            data.bodies.w_a_i,
        ],
        device=model.device,
    )


def compute_joint_dof_body_wrenches_sparse(
    model: ModelKamino, data: DataKamino, jacobians: SparseSystemJacobians, reset_to_zero: bool = True
) -> None:
    """
    Update the actuation wrenches of the bodies based on the active joint torques.
    """
    # First check that the Jacobians are sparse
    if not isinstance(jacobians, SparseSystemJacobians):
        raise ValueError(f"Expected `SparseSystemJacobians` but got {type(jacobians)}.")

    # Clear the previous actuation wrenches, because the kernel computing them
    # uses an atomic add to accumulate contributions from each joint DoF, and
    # thus assumes the target array is zeroed out before each call
    if reset_to_zero:
        data.bodies.w_a_i.zero_()

    # Then compute the body wrenches resulting from the current generalized actuation forces
    wp.launch(
        _compute_joint_dof_body_wrenches_sparse,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.joints.num_dofs,
            model.joints.dofs_offset,
            model.joints.wid,
            model.joints.bid_B,
            model.joints.bid_F,
            data.joints.tau_j,
            jacobians._J_dofs_joint_nzb_offsets,
            jacobians._J_dofs.bsm.nzb_values,
            # Outputs:
            data.bodies.w_a_i,
        ],
        device=model.device,
    )


def compute_joint_dof_body_wrenches(
    model: ModelKamino,
    data: DataKamino,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    reset_to_zero: bool = True,
) -> None:
    """
    Update the actuation wrenches of the bodies based on the active joint torques.
    """
    if isinstance(jacobians, DenseSystemJacobians):
        compute_joint_dof_body_wrenches_dense(model, data, jacobians, reset_to_zero)
    elif isinstance(jacobians, SparseSystemJacobians):
        compute_joint_dof_body_wrenches_sparse(model, data, jacobians, reset_to_zero)
    else:
        raise ValueError(f"Expected `DenseSystemJacobians` or `SparseSystemJacobians` but got {type(jacobians)}.")


def compute_constraint_body_wrenches_dense(
    model: ModelKamino,
    data: DataKamino,
    jacobians: DenseSystemJacobians,
    lambdas_offsets: wp.array,
    lambdas_data: wp.array,
    limits: LimitsKamino | None = None,
    contacts: ContactsKamino | None = None,
    reset_to_zero: bool = True,
):
    """
    Launches the kernels to compute the body-wise constraint wrenches.
    """
    # First check that the Jacobians are dense
    if not isinstance(jacobians, DenseSystemJacobians):
        raise ValueError(f"Expected `DenseSystemJacobians` but got {type(jacobians)}.")

    # Proceed by constraint type, since the Jacobian and lambda data are
    # stored in separate blocks for each constraint type in the dense case
    if model.size.sum_of_num_joints > 0:
        if reset_to_zero:
            data.bodies.w_j_i.zero_()
        wp.launch(
            _compute_joint_cts_body_wrenches_dense,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.info.bodies_offset,
                model.info.joint_dynamic_cts_offset,
                model.info.joint_kinematic_cts_offset,
                model.info.joint_dynamic_cts_group_offset,
                model.info.joint_kinematic_cts_group_offset,
                model.time.inv_dt,
                model.joints.wid,
                model.joints.dynamic_cts_offset,
                model.joints.kinematic_cts_offset,
                model.joints.bid_B,
                model.joints.bid_F,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_j_i,
            ],
            device=model.device,
        )

    if limits is not None and limits.model_max_limits_host > 0:
        if reset_to_zero:
            data.bodies.w_l_i.zero_()
        wp.launch(
            _compute_limit_cts_body_wrenches_dense,
            dim=limits.model_max_limits_host,
            inputs=[
                # Inputs:
                model.info.bodies_offset,
                data.info.limit_cts_group_offset,
                model.time.inv_dt,
                limits.model_active_limits,
                limits.model_max_limits_host,
                limits.wid,
                limits.lid,
                limits.bids,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_l_i,
            ],
            device=model.device,
        )

    if contacts is not None and contacts.model_max_contacts_host > 0:
        if reset_to_zero:
            data.bodies.w_c_i.zero_()
        wp.launch(
            _compute_contact_cts_body_wrenches_dense,
            dim=contacts.model_max_contacts_host,
            inputs=[
                # Inputs:
                model.info.bodies_offset,
                data.info.contact_cts_group_offset,
                model.time.inv_dt,
                contacts.model_active_contacts,
                contacts.model_max_contacts_host,
                contacts.wid,
                contacts.cid,
                contacts.bid_AB,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_c_i,
            ],
            device=model.device,
        )


def compute_constraint_body_wrenches_sparse(
    model: ModelKamino,
    data: DataKamino,
    jacobians: SparseSystemJacobians,
    lambdas_offsets: wp.array,
    lambdas_data: wp.array,
    reset_to_zero: bool = True,
):
    """
    Launches the kernels to compute the body-wise constraint wrenches.
    """
    # First check that the Jacobians are sparse
    if not isinstance(jacobians, SparseSystemJacobians):
        raise ValueError(f"Expected `SparseSystemJacobians` but got {type(jacobians)}.")

    # Optionally clear the previous constraint wrenches, because the kernel computing them
    # uses an `wp.atomic_add` op to accumulate contributions from each constraint non-zero
    # block, and thus assumes the target arrays are zeroed out before each call
    if reset_to_zero:
        data.bodies.w_j_i.zero_()
        data.bodies.w_l_i.zero_()
        data.bodies.w_c_i.zero_()

    # Then compute the body wrenches resulting from the current active constraints
    wp.launch(
        _compute_cts_body_wrenches_sparse,
        dim=(model.size.num_worlds, jacobians._J_cts.bsm.max_of_num_nzb),
        inputs=[
            # Inputs:
            model.time.inv_dt,
            model.info.bodies_offset,
            data.info.limit_cts_group_offset,
            data.info.contact_cts_group_offset,
            jacobians._J_cts.bsm.num_nzb,
            jacobians._J_cts.bsm.nzb_start,
            jacobians._J_cts.bsm.nzb_coords,
            jacobians._J_cts.bsm.nzb_values,
            lambdas_offsets,
            lambdas_data,
            # Outputs:
            data.bodies.w_j_i,
            data.bodies.w_l_i,
            data.bodies.w_c_i,
        ],
        device=model.device,
    )


def compute_constraint_body_wrenches(
    model: ModelKamino,
    data: DataKamino,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    lambdas_offsets: wp.array,
    lambdas_data: wp.array,
    limits: LimitsKamino | None = None,
    contacts: ContactsKamino | None = None,
    reset_to_zero: bool = True,
):
    """
    Launches the kernels to compute the body-wise constraint wrenches.
    """
    if isinstance(jacobians, DenseSystemJacobians):
        compute_constraint_body_wrenches_dense(
            model=model,
            data=data,
            jacobians=jacobians,
            lambdas_offsets=lambdas_offsets,
            lambdas_data=lambdas_data,
            limits=limits,
            contacts=contacts,
            reset_to_zero=reset_to_zero,
        )
    elif isinstance(jacobians, SparseSystemJacobians):
        compute_constraint_body_wrenches_sparse(
            model=model,
            data=data,
            jacobians=jacobians,
            lambdas_offsets=lambdas_offsets,
            lambdas_data=lambdas_data,
            reset_to_zero=reset_to_zero,
        )
    else:
        raise ValueError(f"Expected `DenseSystemJacobians` or `SparseSystemJacobians` but got {type(jacobians)}.")


###
# Conversions
###


def convert_joint_wrenches_to_body_parent_wrenches(
    model: ModelKamino,
    data: DataKamino,
    body_parent_f: wp.array,
):
    """
    Converts the joint actuation and constraint wrenches to the body parent wrenches.

    Args:
        model: The model containing the time-invariant data of the simulation.
        data: The internal solver data container holding the time-varying data of the simulation.
        body_parent_f: The output array to store the body parent wrenches.
    """
    # First clear the body parent wrench array
    body_parent_f.zero_()

    # First convert the joint constraint reactions to the body parent wrenches
    wp.launch(
        kernel=_convert_joint_wrenches_to_body_parent_wrenches,
        dim=model.joints.num_joints,
        inputs=[
            model.joints.dof_type,
            model.joints.bid_F,
            data.bodies.w_a_i,
            data.bodies.w_j_i,
            data.bodies.w_l_i,
        ],
        outputs=[body_parent_f],
        device=model.device,
    )


def convert_body_parent_wrenches_to_joint_reactions(
    body_parent_f: wp.array[wp.spatial_vectorf],
    model: ModelKamino,
    data: DataKamino,
    control: ControlKamino,
    limits: LimitsKamino | None = None,
    reset_to_zero: bool = True,
):
    """
    Converts Newton body-parent wrenches `newton.State.body_parent_f`
    data to Kamino `StateKamino.lambda_j` and `LimitsKamino.reaction`.

    This operation also updates per-joint wrenches arrays `DataKamino.joints.j_w_j` as a byproduct.

    Definitions:
    - `body_parent_f` contains the wrench applied on each body by its parent body, referenced w.r.
        the child body's center of mass (COM) and expressed in the world frame (i.e. world coordinates).
        Each entry is equal to `w_ij`, the world wrench applied by parent body `i` joint `j`.
    - `w_j` is the wrench applied by joint `j` on its follower/child
        body, referenced w.r.t. the joint frame in world coordinates.
    - `j_w_j` is the wrench applied by joint `j` on its follower/child
        body, expressed in the local coordinates of the joint frame.
    - `lambda_j` contains the constraint reaction impulses
        applied by each joint, expressed in the joint frame.
    - `lambda_l_j` contains the joint-limit constraint reactions.
    - `tau_c_j` is the joint-space actuation generalized forces.
    - `tau_j` is the joint-space generaralized forces. However, as any acting joint-limit constraint
        reactions also lie in the same space (i.e. DoF-space), we will consider this to be equal to
        the total joint-space generalized forces `tau_j := tau_c_j + lambda_l_j`
    - `dt` is the simulation time step.

    The conversion is performed parallel over joints as follows:
    - We use the relation `w_j = inv(W_ij) @ w_ij` to compute `w_j`, i.e. the joint wrench
        referenced w.r.t. the joint frame in world coordinates, where `W_ij` is the `6x6` wrench
        transform matrix transforming `w_j` from the joint frame to the COM frame of body `i`.
        When body `i` is the  follower/child we use the absolute pose of the body and joint
        frames to compute `W_ij`.
    - Having `w_j`, we compute `j_w_j` as `j_w_j = X_bar_j.T @ R_bar_j.T @ w_j`, where `X_bar_j`
        is the `6x6` constant joint frame transform matrix extended to 6D (via 3x3 on both diagonals)
        and similarly `R_bar_j` is the `6x6` extended joint frame rotation matrix extended to 6D
        computed from the absolute pose of the joint frame `p_j`.
    - Having `j_w_j`, we compute `lambda_j` as `[lambda_j; tau_j] = inv(S_j) @ j_w_j`, where `S_j`
        is the `6x6` joint constraint/dof selection matrix. `tau_j` is the sum of the joint-space actuation
        generalized forces plus the joint-limit constraint reactions. Thus to recover `lambda_l_j`, and
        assuming we know `tau_c_j`, we can simply compute `lambda_l_j := tau_j - tau_c_j`.

    Correspondences between data containers and conversion inputs/outputs:
    - body_parent_f --> w_ij
    - control.tau_j --> tau_c_j
    - data.joints.j_w_j --> j_w_j
    - data.joints.lambda_j --> lambda_j
    - limits.reaction --> lambda_l_j

    Args:
        body_parent_f:
            The input array of per-body parent wrenches (world frame, at child body's center of mass (COM)).
        model:
            The model containing the time-invariant data of the simulation.
        data:
            The internal solver data container holding the time-varying data of the simulation.
        control:
            The input control data containing the current control inputs of
            the simulation. Used to compute the joint actuation forces `tau_j`.
        limits:
            The active joint-limits container. Optional; if ``None`` (or empty), ``limits.reaction`` is not updated.

    """
    # Early exit if there are no joints, so there is nothing to convert
    if model.size.sum_of_num_joints == 0:
        return

    # Helper function to check if a limit container
    # is provided and if limits have been allocated
    def _has_limits(limits: LimitsKamino | None) -> bool:
        return limits is not None and limits.model_max_limits_host > 0

    # Optionally clear the previous joint wrenches and limit reactions
    if reset_to_zero:
        data.joints.j_w_j.zero_()
        data.joints.lambda_j.zero_()
        if _has_limits(limits):
            limits.reaction.zero_()

    # First convert the body parent wrenches to joint wrenches
    wp.launch(
        kernel=_compute_joint_wrenches_from_body_parent_wrenches,
        dim=model.size.sum_of_num_joints,
        inputs=[
            model.joints.dof_type,
            model.joints.kinematic_cts_offset_joint_cts,
            model.joints.bid_F,
            model.joints.bid_B,
            model.joints.X_j,
            data.joints.p_j,
            data.bodies.q_i,
            body_parent_f,
        ],
        outputs=[data.joints.j_w_j, data.joints.lambda_j],
        device=model.device,
    )

    # Then convert the joint wrenches to limit reactions, if limits are provided
    if _has_limits(limits):
        wp.launch(
            kernel=_compute_limit_reactions_from_joint_wrenches,
            dim=limits.model_max_limits_host,
            inputs=[
                model.joints.dof_type,
                model.joints.dofs_offset,
                limits.model_active_limits,
                limits.model_max_limits_host,
                limits.jid,
                limits.dof,
                limits.side,
                data.joints.j_w_j,
                control.tau_j,
            ],
            outputs=[limits.reaction],
            device=model.device,
        )


def convert_joint_parent_wrenches_to_joint_reactions(
    joint_parent_f: wp.array[wp.spatial_vectorf],
    model: ModelKamino,
    data: DataKamino,
    control: ControlKamino,
    limits: LimitsKamino | None = None,
    reset_to_zero: bool = True,
):
    """
    Converts Newton body-parent wrenches `newton.State.joint_parent_f`
    data to Kamino `StateKamino.lambda_j` and `LimitsKamino.reaction`.

    This operation also updates per-joint wrenches arrays `DataKamino.joints.j_w_j` as a byproduct.

    Definitions:
    - `joint_parent_f` contains the wrench applied via each joint onto its child body by the parent body, referenced
       w.r.t. the child body's center of mass (COM) and expressed in the world frame (i.e. world coordinates).
       Each entry is equal to `w_ij`, the world wrench applied onto child body `i` via joint `j` by the parent body.
    - `w_j` is the wrench applied by joint `j` on its follower/child
        body, referenced w.r.t. the joint frame in world coordinates.
    - `j_w_j` is the wrench applied by joint `j` on its follower/child
        body, expressed in the local coordinates of the joint frame.
    - `lambda_j` contains the constraint reaction impulses
        applied by each joint, expressed in the joint frame.
    - `lambda_l_j` contains the joint-limit constraint reactions.
    - `tau_c_j` is the joint-space actuation generalized forces.
    - `tau_j` is the joint-space generaralized forces. However, as any acting joint-limit constraint
        reactions also lie in the same space (i.e. DoF-space), we will consider this to be equal to
        the total joint-space generalized forces `tau_j := tau_c_j + lambda_l_j`
    - `dt` is the simulation time step.

    The conversion is performed parallel over joints as follows:
    - We use the relation `w_j = inv(W_ij) @ w_ij` to compute `w_j`, i.e. the joint wrench
        referenced w.r.t. the joint frame in world coordinates, where `W_ij` is the `6x6` wrench
        transform matrix transforming `w_j` from the joint frame to the COM frame of body `i`.
        When body `i` is the  follower/child we use the absolute pose of the body and joint
        frames to compute `W_ij`.
    - Having `w_j`, we compute `j_w_j` as `j_w_j = X_bar_j.T @ R_bar_j.T @ w_j`, where `X_bar_j`
        is the `6x6` constant joint frame transform matrix extended to 6D (via 3x3 on both diagonals)
        and similarly `R_bar_j` is the `6x6` extended joint frame rotation matrix extended to 6D
        computed from the absolute pose of the joint frame `p_j`.
    - Having `j_w_j`, we compute `lambda_j` as `[lambda_j; tau_j] = inv(S_j) @ j_w_j`, where `S_j`
        is the `6x6` joint constraint/dof selection matrix. `tau_j` is the sum of the joint-space actuation
        generalized forces plus the joint-limit constraint reactions. Thus to recover `lambda_l_j`, and
        assuming we know `tau_c_j`, we can simply compute `lambda_l_j := tau_j - tau_c_j`.

    Correspondences between data containers and conversion inputs/outputs:
    - joint_parent_f --> w_ij
    - control.tau_j --> tau_c_j
    - data.joints.j_w_j --> j_w_j
    - data.joints.lambda_j --> lambda_j
    - limits.reaction --> lambda_l_j

    Args:
        joint_parent_f:
            The input array of per-joint parent wrenches (world frame, at child body's center of mass (COM)).
        model:
            The model containing the time-invariant data of the simulation.
        data:
            The internal solver data container holding the time-varying data of the simulation.
        control:
            The input control data containing the current control inputs of
            the simulation. Used to compute the joint actuation forces `tau_j`.
        limits:
            The active joint-limits container. Optional; if ``None`` (or empty), ``limits.reaction`` is not updated.

    """
    # Early exit if there are no joints, so there is nothing to convert
    if model.size.sum_of_num_joints == 0:
        return

    # Helper function to check if a limit container
    # is provided and if limits have been allocated
    def _has_limits(limits: LimitsKamino | None) -> bool:
        return limits is not None and limits.model_max_limits_host > 0

    # Optionally clear the previous joint wrenches and limit reactions
    if reset_to_zero:
        data.joints.j_w_j.zero_()
        data.joints.lambda_j.zero_()
        if _has_limits(limits):
            limits.reaction.zero_()

    # First convert the joint parent wrenches to joint wrenches
    wp.launch(
        kernel=_compute_joint_wrenches_from_joint_parent_wrenches,
        dim=model.size.sum_of_num_joints,
        inputs=[
            model.joints.dof_type,
            model.joints.kinematic_cts_offset_joint_cts,
            model.joints.bid_F,
            model.joints.bid_B,
            model.joints.X_j,
            data.joints.p_j,
            data.bodies.q_i,
            joint_parent_f,
        ],
        outputs=[data.joints.j_w_j, data.joints.lambda_j],
        device=model.device,
    )

    # Then convert the joint wrenches to limit reactions, if limits are provided
    if _has_limits(limits):
        wp.launch(
            kernel=_compute_limit_reactions_from_joint_wrenches,
            dim=limits.model_max_limits_host,
            inputs=[
                model.joints.dof_type,
                model.joints.dofs_offset,
                limits.model_active_limits,
                limits.model_max_limits_host,
                limits.jid,
                limits.dof,
                limits.side,
                data.joints.j_w_j,
                control.tau_j,
            ],
            outputs=[limits.reaction],
            device=model.device,
        )
