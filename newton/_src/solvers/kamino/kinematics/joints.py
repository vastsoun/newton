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
KAMINO: Kinematics: Joints
"""

from __future__ import annotations

import warp as wp

from ..core.joints import JointDoFType
from ..core.math import (
    TWO_PI,  # noqa: F401
    quat_apply,
    quat_conj,
    quat_log,
    quat_product,
    quat_to_euler_xyz,
    screw,
    screw_angular,
    screw_linear,
)
from ..core.model import Model, ModelData
from ..core.types import (
    float32,
    int32,
    mat33f,
    quatf,
    transformf,
    vec1f,
    vec2f,
    vec3f,
    vec6f,
    vec7f,
)

###
# Module interface
###

__all__ = [
    "compute_joints_state",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions - Coordinate Mappings
###


@wp.func
def map_to_joint_coords_revolute(j_r_j: vec3f, j_q_j: quatf) -> vec1f:
    j_p_j = quat_log(j_q_j)
    return vec1f(j_p_j[0])


@wp.func
def map_to_joint_coords_prismatic(j_r_j: vec3f, j_q_j: quatf) -> vec1f:
    return vec1f(j_r_j[0])


@wp.func
def map_to_joint_coords_cylindrical(j_r_j: vec3f, j_q_j: quatf) -> vec2f:
    return vec2f(j_r_j[0], j_r_j[1])


@wp.func
def map_to_joint_coords_universal(j_r_j: vec3f, j_q_j: quatf) -> vec2f:
    j_theta_j = quat_log(j_q_j)
    return vec2f(j_theta_j[0], j_theta_j[1])


@wp.func
def map_to_joint_coords_spherical(j_r_j: vec3f, j_q_j: quatf) -> vec3f:
    return j_q_j


@wp.func
def map_to_joint_coords_gimbal(j_r_j: vec3f, j_q_j: quatf) -> vec3f:
    return quat_to_euler_xyz(j_q_j)


@wp.func
def map_to_joint_coords_cartesian(j_r_j: vec3f, j_q_j: quatf) -> vec3f:
    return j_r_j


@wp.func
def map_to_joint_coords_free(j_r_j: vec3f, j_q_j: quatf) -> vec7f:
    return vec7f(j_r_j[0], j_r_j[1], j_r_j[2], j_q_j.x, j_q_j.y, j_q_j.z, j_q_j.w)


###
# Functions - State Writes
###


def make_write_joint_state_generic(dof_type: JointDoFType):
    """
    Generates functions to store the joint state according to the
    constraint and DoF dimensions specific to the type of joint.
    """

    # Retrieve the joint constraint and DoF axes
    dof_axes = dof_type.dofs_axes
    cts_axes = dof_type.cts_axes

    # Retrieve the number of constraints and dofs
    # num_coords = dof_type.num_coords
    num_dof = dof_type.num_dofs
    num_cts = dof_type.num_cts

    @wp.func
    def write_joint_state_generic(
        # Inputs:
        cts_offset: int32,  # Index offset of the joint constraints
        coords_offset: int32,  # Index offset of the joint coordinates
        dofs_offset: int32,  # Index offset of the joint DoFs
        j_p_j: vec6f,  # 6D vector of the joint-local relative pose
        j_u_j: vec6f,  # 6D vector ofthe joint-local relative twist
        # Outputs:
        r_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
        dr_j_out: wp.array(dtype=float32),  # Flat array of joint constraint velocities
        q_j_out: wp.array(dtype=float32),  # Flat array of joint DoF coordinates
        dq_j_out: wp.array(dtype=float32),  # Flat array of joint DoF velocities
    ):
        # Store the joint constraint residuals
        for j in range(num_cts):
            r_j_out[cts_offset + j] = j_p_j[cts_axes[j]]
            dr_j_out[cts_offset + j] = j_u_j[cts_axes[j]]

        # Store the joint DoF coordinates and velocities
        for j in range(num_dof):
            q_j_out[coords_offset + j] = j_p_j[dof_axes[j]]
            dq_j_out[dofs_offset + j] = j_u_j[dof_axes[j]]

    # Return the function
    return write_joint_state_generic


write_joint_state_fixed = make_write_joint_state_generic(JointDoFType.FIXED)
"""Function to store the joint state for 0-DoF fixed joints."""

write_joint_state_revolute = make_write_joint_state_generic(JointDoFType.REVOLUTE)
"""Function to store the joint state for 1-DoF revolute joints."""

write_joint_state_prismatic = make_write_joint_state_generic(JointDoFType.PRISMATIC)
"""Function to store the joint state for 1-DoF prismatic joints."""

write_joint_state_cylindrical = make_write_joint_state_generic(JointDoFType.CYLINDRICAL)
"""Function to store the joint state for 2-DoF cylindrical joints."""

write_joint_state_universal = make_write_joint_state_generic(JointDoFType.UNIVERSAL)
"""Function to store the joint state for 2-DoF universal joints."""

write_joint_state_spherical = make_write_joint_state_generic(JointDoFType.SPHERICAL)
"""Function to store the joint state for 3-DoF spherical joints."""

write_joint_state_gimbal = make_write_joint_state_generic(JointDoFType.GIMBAL)
"""Function to store the joint state for 3-DoF gimbal joints."""

write_joint_state_cartesian = make_write_joint_state_generic(JointDoFType.CARTESIAN)
"""Function to store the joint state for 3-DoF cartesian joints."""

write_joint_state_free = make_write_joint_state_generic(JointDoFType.FREE)
"""Function to store the joint state for 6-DoF free joints."""


@wp.func
def write_joint_state_universal_new(
    # Inputs:
    cts_offset: int32,  # Index offset of the joint constraints
    coords_offset: int32,  # Index offset of the joint coordinates
    dofs_offset: int32,  # Index offset of the joint DoFs
    j_r_j: vec3f,  # 3D vector of the joint-local relative pose
    j_q_j: quatf,  # 4D unit-quaternion of the joint-local relative pose
    j_u_j: vec6f,  # 6D vector ofthe joint-local relative twist
    # Outputs:
    r_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
    dr_j_out: wp.array(dtype=float32),  # Flat array of joint constraint velocities
    q_j_out: wp.array(dtype=float32),  # Flat array of joint DoF coordinates
    dq_j_out: wp.array(dtype=float32),  # Flat array of joint DoF velocities
):
    # Retrieve the joint constraint and DoF axes
    dof_axes = wp.static(JointDoFType.UNIVERSAL.dofs_axes)
    cts_axes = wp.static(JointDoFType.UNIVERSAL.cts_axes)

    # Retrieve the number of constraints and dofs
    num_cts = wp.static(JointDoFType.UNIVERSAL.num_cts)
    num_dof = wp.static(JointDoFType.UNIVERSAL.num_dofs)
    num_coord = wp.static(JointDoFType.UNIVERSAL.num_coords)

    # Convert the joint rotation quaternion to a rotation vector
    j_theta_j = quat_log(j_q_j)

    # Construct a 6D relative pose vector using a rotation vector
    j_p_j = screw(j_r_j, j_theta_j)

    # Store the joint constraint residuals
    for j in range(num_cts):
        r_j_out[cts_offset + j] = j_p_j[cts_axes[j]]
        dr_j_out[cts_offset + j] = j_u_j[cts_axes[j]]

    # Store the joint DoF velocities
    for j in range(num_dof):
        dq_j_out[dofs_offset + j] = j_u_j[dof_axes[j]]

    # Store the joint DoF coordinates (i.e. XY components of the 3D rotation vector)
    for j in range(num_coord):
        q_j_out[coords_offset + j] = j_theta_j[j]


@wp.func
def write_joint_state_spherical_new(
    # Inputs:
    cts_offset: int32,  # Index offset of the joint constraints
    coords_offset: int32,  # Index offset of the joint coordinates
    dofs_offset: int32,  # Index offset of the joint DoFs
    j_r_j: vec3f,  # 3D vector of the joint-local relative pose
    j_q_j: quatf,  # 4D unit-quaternion of the joint-local relative pose
    j_u_j: vec6f,  # 6D vector ofthe joint-local relative twist
    # Outputs:
    r_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
    dr_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
    q_j_out: wp.array(dtype=float32),  # Flat array of joint DoF coordinates
    dq_j_out: wp.array(dtype=float32),  # Flat array of joint DoF velocities
):
    # Retrieve the joint constraint and DoF axes
    dof_axes = wp.static(JointDoFType.SPHERICAL.dofs_axes)
    cts_axes = wp.static(JointDoFType.SPHERICAL.cts_axes)

    # Retrieve the number of constraints and dofs
    num_cts = wp.static(JointDoFType.SPHERICAL.num_cts)
    num_dof = wp.static(JointDoFType.SPHERICAL.num_dofs)
    num_coord = wp.static(JointDoFType.SPHERICAL.num_coords)

    # Construct a 6D relative pose vector using a rotation vector
    j_p_j = screw(j_r_j, quat_log(j_q_j))

    # Store the joint constraint residuals
    for j in range(num_cts):
        r_j_out[cts_offset + j] = j_p_j[cts_axes[j]]
        dr_j_out[cts_offset + j] = j_u_j[cts_axes[j]]

    # Store the joint DoF velocities
    for j in range(num_dof):
        dq_j_out[dofs_offset + j] = j_u_j[dof_axes[j]]

    # Store the joint DoF coordinates (i.e. unit quaternion components)
    for j in range(num_coord):
        q_j_out[coords_offset + j] = j_q_j[j]


@wp.func
def write_joint_state_gimbal_new(
    # Inputs:
    cts_offset: int32,  # Index offset of the joint constraints
    coords_offset: int32,  # Index offset of the joint coordinates
    dofs_offset: int32,  # Index offset of the joint DoFs
    j_r_j: vec3f,  # 3D vector of the joint-local relative pose
    j_q_j: quatf,  # 4D unit-quaternion of the joint-local relative pose
    j_u_j: vec6f,  # 6D vector ofthe joint-local relative twist
    # Outputs:
    r_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
    dr_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
    q_j_out: wp.array(dtype=float32),  # Flat array of joint DoF coordinates
    dq_j_out: wp.array(dtype=float32),  # Flat array of joint DoF velocities
):
    # Retrieve the joint constraint and DoF axes
    dof_axes = wp.static(JointDoFType.GIMBAL.dofs_axes)
    cts_axes = wp.static(JointDoFType.GIMBAL.cts_axes)

    # Retrieve the number of constraints and dofs
    num_cts = wp.static(JointDoFType.GIMBAL.num_cts)
    num_dof = wp.static(JointDoFType.GIMBAL.num_dofs)
    num_coord = wp.static(JointDoFType.GIMBAL.num_coords)

    # Convert the joint rotation quaternion to a rotation vector
    rpy = quat_to_euler_xyz(j_q_j)

    # Construct a 6D relative pose vector using a rotation vector
    j_p_j = screw(j_r_j, quat_log(j_q_j))

    # Store the joint constraint residuals
    for j in range(num_cts):
        r_j_out[cts_offset + j] = j_p_j[cts_axes[j]]
        dr_j_out[cts_offset + j] = j_u_j[cts_axes[j]]

    # Store the joint DoF velocities
    for j in range(num_dof):
        dq_j_out[dofs_offset + j] = j_u_j[dof_axes[j]]

    # Store the joint DoF coordinates (i.e. 3D XYZ Euler angles)
    for j in range(num_coord):
        q_j_out[coords_offset + j] = rpy[j]


###
# Functions - Coordinate Correction
###


# @wp.func
# def correct_joint_coord_revolute(
#     # Inputs:
#     cio_j: int32,  # Index offset of the joint constraints
#     qio_j: int32,  # Index offset of the joint coordinates
#     dio_j: int32,  # Index offset of the joint DoFs
#     j_r_j: vec3f,  # 3D vector of the joint-local relative pose
#     j_q_j: quatf,  # 4D unit-quaternion of the joint-local relative pose
#     j_u_j: vec6f,  # 6D vector ofthe joint-local relative twist
#     # Outputs:
#     r_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
#     dr_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
#     q_j_out: wp.array(dtype=float32),  # Flat array of joint DoF coordinates
#     dq_j_out: wp.array(dtype=float32),  # Flat array of joint DoF velocities
# ):
#     # TODO: How to clean this up?
#     # Enforce continuity of revolute joint angles by adding or subtracting 2*PI as needed
#     # to minimize the difference between the current angle and the previous angle
#     r_j_rx = r_j[3]
#     r_j_rx_corr = wp.round((q_j_p[qio_j] - r_j_rx) / TWO_PI) * TWO_PI
#     r_j[3] += r_j_rx_corr


###
# Functions - State Computation
###


@wp.func
def compute_joint_pose_and_relative_motion(
    T_B_j: transformf,
    T_F_j: transformf,
    u_B_j: vec6f,
    u_F_j: vec6f,
    B_r_Bj: vec3f,
    F_r_Fj: vec3f,
    X_j: mat33f,
) -> tuple[transformf, vec6f, vec6f]:
    """
    Computes the relative motion of a joint given the states of its Base and Follower bodies.

    Args:
        T_B_j (transformf): The absolute pose of the Base body in world coordinates.
        T_F_j (transformf): The absolute pose of the Follower body in world coordinates.
        u_B_j (vec6f): The absolute twist of the Base body in world coordinates.
        u_F_j (vec6f): The absolute twist of the Follower body in world coordinates.
        B_r_Bj (vec3f): The position of the joint frame in the Base body's local coordinates.
        F_r_Fj (vec3f): The position of the joint frame in the Follower body's local coordinates.
        X_j (mat33f): The joint transformation matrix.

    Returns:
        tuple[transformf, vec6f, vec6f]: The absolute pose of the joint frame in world coordinates,
        and two 6D vectors encoding the relative motion of the bodies in the frame of the joint.
    """
    # TODO: skip rotmat and rotate everything with quaternions

    # Extract the decomposed state of the Base body
    r_B_j = wp.transform_get_translation(T_B_j)
    q_B_j = wp.transform_get_rotation(T_B_j)
    R_B_j = wp.quat_to_matrix(q_B_j)
    v_B_j = screw_linear(u_B_j)
    omega_B_j = screw_angular(u_B_j)

    # Extract the decomposed state of the Follower body
    r_F_j = wp.transform_get_translation(T_F_j)
    q_F_j = wp.transform_get_rotation(T_F_j)
    v_F_j = screw_linear(u_F_j)
    omega_F_j = screw_angular(u_F_j)

    # Compute the pose of the joint frame via the Base body
    r_j_B = r_B_j + R_B_j @ B_r_Bj
    p_j = wp.transformation(r_j_B, q_B_j, dtype=float32)

    # Pre-compute transforms to joint-space
    X_j_T = wp.transpose(X_j)
    R_B_j_T = wp.transpose(R_B_j)
    X_j_T_R_B_j_T = X_j_T @ R_B_j_T

    # Compute the 6D relative pose vector between the representations of joint frame w.r.t. the two bodies
    q_B_j_conj = quat_conj(q_B_j)
    j_dr_j = X_j_T @ (quat_apply(q_B_j_conj, r_F_j + quat_apply(q_F_j, F_r_Fj) - r_B_j) - B_r_Bj)
    j_dR_j = X_j_T @ quat_log(quat_product(q_B_j_conj, q_F_j))
    j_dp_j = screw(j_dr_j, j_dR_j)

    # Compute the 6D relative twist vector between the representations of joint frame w.r.t. the two bodies
    # TODO: How can we simplify this expression and make it more efficient?
    r_Bj = quat_apply(q_B_j, B_r_Bj)
    r_Fj = quat_apply(q_F_j, F_r_Fj)
    j_v_j = X_j_T_R_B_j_T @ (v_F_j - v_B_j + wp.cross(omega_F_j, r_Fj) - wp.cross(omega_B_j, r_Bj))
    j_omega_j = X_j_T_R_B_j_T @ (omega_F_j - omega_B_j)
    j_u_j = screw(j_v_j, j_omega_j)

    # Return the computed joint frame pose and relative motion vectors
    return p_j, j_dp_j, j_u_j


@wp.func
def write_joint_state(
    # Inputs:
    dof_type: int32,
    cts_offset: int32,
    coords_offset: int32,
    dofs_offset: int32,
    j_p_j: vec6f,
    j_u_j: vec6f,
    # Outputs:
    data_r_j: wp.array(dtype=float32),
    data_dr_j: wp.array(dtype=float32),
    data_q_j: wp.array(dtype=float32),
    data_dq_j: wp.array(dtype=float32),
):
    """
    Stores the joint constraint residuals and DoF motion based on the joint type.

    Args:
        dof_type (int32): The type of joint DoF.
        cts_offset (int32): Index offset of the joint constraints.
        coords_offset (int32): Index offset of the joint coordinates.
        dofs_offset (int32): Index offset of the joint DoFs.
        r_j (vec6f): 6D vector of the joint-local relative pose.
        dr_j (vec6f): 6D vector ofthe joint-local relative twist.
        data_r_j (wp.array): Flat array of joint constraint residuals.
        data_dr_j (wp.array): Flat array of joint constraint residuals.
        data_q_j (wp.array): Flat array of joint DoF coordinates.
        data_dq_j (wp.array): Flat array of joint DoF velocities.
    """
    # TODO: Use wp.static to include conditionals at compile time based on the joint types present in the builder

    if dof_type == JointDoFType.REVOLUTE:
        write_joint_state_revolute(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )

    elif dof_type == JointDoFType.PRISMATIC:
        write_joint_state_prismatic(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        write_joint_state_cylindrical(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        write_joint_state_universal(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )

    elif dof_type == JointDoFType.SPHERICAL:
        write_joint_state_spherical(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )

    elif dof_type == JointDoFType.GIMBAL:
        write_joint_state_gimbal(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )

    elif dof_type == JointDoFType.CARTESIAN:
        write_joint_state_cartesian(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )

    elif dof_type == JointDoFType.FIXED:
        write_joint_state_fixed(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )

    elif dof_type == JointDoFType.FREE:
        write_joint_state_free(
            cts_offset, coords_offset, dofs_offset, j_p_j, j_u_j, data_r_j, data_dr_j, data_q_j, data_dq_j
        )


###
# Kernels
###


@wp.kernel
def _compute_joints_state(
    # Inputs:
    model_info_joint_coords_offset: wp.array(dtype=int32),
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_cts_offset: wp.array(dtype=int32),
    model_joint_wid: wp.array(dtype=int32),
    model_joint_dof_type: wp.array(dtype=int32),
    model_joint_coords_offset: wp.array(dtype=int32),
    model_joint_dofs_offset: wp.array(dtype=int32),
    model_joint_cts_offset: wp.array(dtype=int32),
    model_joint_bid_B: wp.array(dtype=int32),
    model_joint_bid_F: wp.array(dtype=int32),
    model_joint_B_r_Bj: wp.array(dtype=vec3f),
    model_joint_F_r_Fj: wp.array(dtype=vec3f),
    model_joint_X_j: wp.array(dtype=mat33f),
    state_body_q_i: wp.array(dtype=transformf),
    state_body_u_i: wp.array(dtype=vec6f),
    data_q_j_p: wp.array(dtype=float32),
    # Outputs:
    data_p_j: wp.array(dtype=transformf),
    data_r_j: wp.array(dtype=float32),
    data_dr_j: wp.array(dtype=float32),
    data_q_j: wp.array(dtype=float32),
    data_dq_j: wp.array(dtype=float32),
):
    """
    Reset the current state to the initial state defined in the model.
    """
    # Retrieve the thread index
    jid = wp.tid()

    # Retrieve the joint model data
    wid = model_joint_wid[jid]
    dof_type = model_joint_dof_type[jid]
    coords_offset = model_joint_coords_offset[jid]
    dofs_offset = model_joint_dofs_offset[jid]
    cts_offset = model_joint_cts_offset[jid]
    bid_B = model_joint_bid_B[jid]
    bid_F = model_joint_bid_F[jid]
    B_r_Bj = model_joint_B_r_Bj[jid]
    F_r_Fj = model_joint_F_r_Fj[jid]
    X_j = model_joint_X_j[jid]

    # Retrieve the index offsets of the joint's constraint and DoF dimensions
    world_joint_coords_offset = model_info_joint_coords_offset[wid]
    world_joint_dofs_offset = model_info_joint_dofs_offset[wid]
    world_joint_cts_offset = model_info_joint_cts_offset[wid]

    # Append the index offsets of the world's joint blocks
    coords_offset += world_joint_coords_offset
    dofs_offset += world_joint_dofs_offset
    cts_offset += world_joint_cts_offset

    # If the Base body is the world (bid=-1), use the identity transform (frame
    # of the world's origin), otherwise retrieve the Base body's pose and twist
    T_B_j = wp.transform_identity(dtype=float32)
    u_B_j = vec6f(0.0)
    if bid_B > -1:
        T_B_j = state_body_q_i[bid_B]
        u_B_j = state_body_u_i[bid_B]

    # Retrieve the Follower body's pose and twist
    T_F_j = state_body_q_i[bid_F]
    u_F_j = state_body_u_i[bid_F]

    # Compute the joint frame pose and relative motion
    p_j, j_p_j, j_u_j = compute_joint_pose_and_relative_motion(T_B_j, T_F_j, u_B_j, u_F_j, B_r_Bj, F_r_Fj, X_j)

    # Store the absolute pose of the joint frame in world coordinates
    data_p_j[jid] = p_j

    # Store the joint constraint residuals and motion
    write_joint_state(
        dof_type,
        cts_offset,
        dofs_offset,
        coords_offset,
        j_p_j,
        j_u_j,
        data_r_j,
        data_dr_j,
        data_q_j,
        data_dq_j,
    )


###
# Launchers
###


def compute_joints_state(model: Model, q_j_ref: wp.array, data: ModelData) -> None:
    """
    Computes the states of the joints based on the current body states.

    The joint state data to computed includes both the generalized coordinates and velocities
    corresponding to the respective degrees of freedom (DoFs), as well as the constraint-space
    residuals and velocities of the applied bilateral constraints.

    Args:
        model (`Model`): The model container holding the time-invariant data of the simulation.
        q_j_p (`wp.array`): An array of reference joint DoF coordinates used for coordinate correction.\n
            Only used for revolute DoFs of the relevant joints to enforce angle continuity.\n
            Shape of ``(sum_of_num_joint_coords,)`` and type :class:`float`.
        data (`ModelData`): The solver data container holding the internal time-varying state of the simulation.
    """
    wp.launch(
        _compute_joints_state,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.info.joint_coords_offset,
            model.info.joint_dofs_offset,
            model.info.joint_cts_offset,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.cts_offset,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_j,
            data.bodies.q_i,
            data.bodies.u_i,
            q_j_ref,
            # Outputs:
            data.joints.p_j,
            data.joints.r_j,
            data.joints.dr_j,
            data.joints.q_j,
            data.joints.dq_j,
        ],
    )
