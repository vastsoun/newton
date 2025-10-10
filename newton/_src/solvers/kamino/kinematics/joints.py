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

from typing import Any

import warp as wp

from ..core.joints import JointDoFType
from ..core.math import (
    quat_apply,
    quat_conj,
    quat_log,
    quat_product,
    screw,
    screw_angular,
    screw_linear,
)
from ..core.model import Model, ModelData
from ..core.types import (
    float32,
    int32,
    mat33f,
    transformf,
    vec1i,
    vec2i,
    vec3f,
    vec3i,
    vec4i,
    vec5i,
    vec6f,
    vec6i,
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
# Joint constraint selectors (as vectors of indices)
###

S_cts_fixed = wp.constant(vec6i(0, 1, 2, 3, 4, 5))
"""Constraint column selection for 0-DoF fixed joints."""

S_cts_revolute = wp.constant(vec5i(0, 1, 2, 4, 5))
"""Constraint column selection for 1-DoF revolute joints."""

S_cts_prismatic = wp.constant(vec5i(1, 2, 3, 4, 5))
"""Constraint column selection for 1-Dof prismatic joints."""

S_cts_cylindrical = wp.constant(vec4i(1, 2, 4, 5))
"""Constraint column selection for 2-DoF cylindrical joints."""

S_cts_universal = wp.constant(vec4i(2, 3, 4, 5))
"""Constraint column selection for 2-DoF universal joints."""

S_cts_spherical = wp.constant(vec3i(0, 1, 2))
"""Constraint column selection for 3-DoF spherical joints."""

S_cts_cartesian = wp.constant(vec3i(3, 4, 5))
"""Constraint column selection for 3-DoF cartesian joints."""


###
# Joint DoF selectors (as vectors of indices)
###

S_dofs_revolute = wp.constant(vec1i(3))
"""DoF column selection for 1-DoF revolute joints."""

S_dofs_prismatic = wp.constant(vec1i(0))
"""DoF column selection for 1-DoF prismatic joints."""

S_dofs_cylindrical = wp.constant(vec2i(0, 3))
"""DoF column selection for 2-DoF cylindrical joints."""

S_dofs_universal = wp.constant(vec2i(0, 1))
"""DoF column selection for 2-DoF universal joints."""

S_dofs_spherical = wp.constant(vec3i(3, 4, 5))
"""DoF column selection for 3-DoF spherical joints."""

S_dofs_cartesian = wp.constant(vec3i(0, 1, 2))
"""DoF column selection for 3-DoF cartesian joints."""

S_dofs_free = wp.constant(vec6i(0, 1, 2, 3, 4, 5))
"""DoF column selection for 6-DoF free joints."""


###
# Functions
###


def make_store_joint_state_func(cst_selection: Any, dof_selection: Any):
    """
    Generates functions to store the joint state according to the
    constraint and DoF dimensions specific to the type of joint.
    """

    @wp.func
    def store_joint_state(
        # Inputs:
        cio_j: int32,  # Index offset of the joint constraints dimensions
        dio_j: int32,  # Index offset of the joint DoF dimensions
        r_j: vec6f,  # 6D vector of the joint-local relative pose
        dr_j: vec6f,  # 6D vector ofthe joint-local relative twist
        # Outputs:
        r_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
        dr_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
        q_j_out: wp.array(dtype=float32),  # Flat array of joint DoF coordinates
        dq_j_out: wp.array(dtype=float32),  # Flat array of joint DoF velocities
    ):
        # Compute the number of constraints and dofs
        num_cst = wp.static(len(cst_selection))
        num_dof = wp.static(len(dof_selection))

        # Store the joint constraint residuals
        for j in range(num_cst):
            r_j_out[cio_j + j] = r_j[cst_selection[j]]
            dr_j_out[cio_j + j] = dr_j[cst_selection[j]]

        # Store the joint DoF coordinates and velocities
        for j in range(num_dof):
            q_j_out[dio_j + j] = r_j[dof_selection[j]]
            dq_j_out[dio_j + j] = dr_j[dof_selection[j]]

    # Return the function
    return store_joint_state


store_joint_state_fixed = make_store_joint_state_func(S_cts_fixed, [])
"""Function to store the joint state for 0-DoF fixed joints."""

store_joint_state_revolute = make_store_joint_state_func(S_cts_revolute, S_dofs_revolute)
"""Function to store the joint state for 1-DoF revolute joints."""

store_joint_state_prismatic = make_store_joint_state_func(S_cts_prismatic, S_dofs_prismatic)
"""Function to store the joint state for 1-DoF prismatic joints."""

store_joint_state_cylindrical = make_store_joint_state_func(S_cts_cylindrical, S_dofs_cylindrical)
"""Function to store the joint state for 2-DoF cylindrical joints."""

store_joint_state_universal = make_store_joint_state_func(S_cts_universal, S_dofs_universal)
"""Function to store the joint state for 2-DoF universal joints."""

store_joint_state_spherical = make_store_joint_state_func(S_cts_spherical, S_dofs_spherical)
"""Function to store the joint state for 3-DoF spherical joints."""

store_joint_state_cartesian = make_store_joint_state_func(S_cts_cartesian, S_dofs_cartesian)
"""Function to store the joint state for 3-DoF cartesian joints."""

store_joint_state_free = make_store_joint_state_func([], S_dofs_free)
"""Function to store the joint state for 6-DoF free joints."""


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
    # Outputs:
    state_joint_p_j: wp.array(dtype=transformf),
    state_joint_r_j: wp.array(dtype=float32),
    state_joint_dr_j: wp.array(dtype=float32),
    state_joint_q_j: wp.array(dtype=float32),
    state_joint_dq_j: wp.array(dtype=float32),
):
    """
    Reset the current state to the initial state defined in the model.
    """
    # TODO: skip rotmat and rotate everything with quaternions

    # Retrieve the thread index
    jid = wp.tid()

    # Retrieve the joint model data
    wid_j = model_joint_wid[jid]
    dof_type_j = model_joint_dof_type[jid]
    qio_j = model_joint_coords_offset[jid]
    dio_j = model_joint_dofs_offset[jid]
    cio_j = model_joint_cts_offset[jid]
    bid_B = model_joint_bid_B[jid]
    bid_F = model_joint_bid_F[jid]
    B_r_Bj = model_joint_B_r_Bj[jid]
    F_r_Fj = model_joint_F_r_Fj[jid]
    X_j = model_joint_X_j[jid]

    # Retrieve the index offsets of the joint's constraint and DoF dimensions
    jqio = model_info_joint_coords_offset[wid_j]
    jdio = model_info_joint_dofs_offset[wid_j]
    jcio = model_info_joint_cts_offset[wid_j]

    # Append the index offsets of the world's joint blocks
    qio_j += jqio
    dio_j += jdio
    cio_j += jcio

    # If the base body is the world (bid=-1), use the identity transform (frame of the world's origin)
    T_B_j = wp.transform_identity()
    u_F_j = vec6f(0.0)
    if bid_B > -1:
        T_B_j = state_body_q_i[bid_B]
        u_B_j = state_body_u_i[bid_B]

    # Retrive the Follower body frames and twists
    T_F_j = state_body_q_i[bid_F]
    u_F_j = state_body_u_i[bid_F]

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
    T_j_B = wp.transformation(r_j_B, q_B_j, dtype=float32)

    # Compute the 6D relative pose vector between the representations of joint frame w.r.t. the two bodies
    q_B_j_conj = quat_conj(q_B_j)
    r_j_t = wp.transpose(X_j) @ (quat_apply(q_B_j_conj, r_F_j + quat_apply(q_F_j, F_r_Fj) - r_B_j) - B_r_Bj)
    r_j_r = wp.transpose(X_j) @ quat_log(quat_product(q_B_j_conj, q_F_j))
    r_j = screw(r_j_t, r_j_r)

    # Compute the 6D relative twist vector between the representations of joint frame w.r.t. the two bodies
    # TODO: How can we simplify this expression and make it more efficient?
    r_Bj = quat_apply(q_B_j, B_r_Bj)
    r_Fj = quat_apply(q_F_j, F_r_Fj)
    R_B_j_X_j_T = wp.transpose(X_j) @ wp.transpose(R_B_j)
    dr_j_t = R_B_j_X_j_T @ (v_F_j - v_B_j + wp.cross(omega_F_j, r_Fj) - wp.cross(omega_B_j, r_Bj))
    dr_j_r = R_B_j_X_j_T @ (omega_F_j - omega_B_j)
    dr_j = screw(dr_j_t, dr_j_r)

    # Store the pose of the joint frame
    state_joint_p_j[jid] = T_j_B

    # Store the joint state depending the kinematic (i.e. DoF) type
    if dof_type_j == JointDoFType.REVOLUTE:
        store_joint_state_revolute(
            cio_j, dio_j, r_j, dr_j, state_joint_r_j, state_joint_dr_j, state_joint_q_j, state_joint_dq_j
        )

    elif dof_type_j == JointDoFType.PRISMATIC:
        store_joint_state_prismatic(
            cio_j, dio_j, r_j, dr_j, state_joint_r_j, state_joint_dr_j, state_joint_q_j, state_joint_dq_j
        )

    elif dof_type_j == JointDoFType.CYLINDRICAL:
        store_joint_state_cylindrical(
            cio_j, dio_j, r_j, dr_j, state_joint_r_j, state_joint_dr_j, state_joint_q_j, state_joint_dq_j
        )

    elif dof_type_j == JointDoFType.UNIVERSAL:
        store_joint_state_universal(
            cio_j, dio_j, r_j, dr_j, state_joint_r_j, state_joint_dr_j, state_joint_q_j, state_joint_dq_j
        )

    elif dof_type_j == JointDoFType.SPHERICAL:
        store_joint_state_spherical(
            cio_j, dio_j, r_j, dr_j, state_joint_r_j, state_joint_dr_j, state_joint_q_j, state_joint_dq_j
        )

    elif dof_type_j == JointDoFType.CARTESIAN:
        store_joint_state_cartesian(
            cio_j, dio_j, r_j, dr_j, state_joint_r_j, state_joint_dr_j, state_joint_q_j, state_joint_dq_j
        )

    elif dof_type_j == JointDoFType.FIXED:
        store_joint_state_fixed(
            cio_j, dio_j, r_j, dr_j, state_joint_r_j, state_joint_dr_j, state_joint_q_j, state_joint_dq_j
        )

    elif dof_type_j == JointDoFType.FREE:
        store_joint_state_free(
            cio_j, dio_j, r_j, dr_j, state_joint_r_j, state_joint_dr_j, state_joint_q_j, state_joint_dq_j
        )


###
# Launchers
###


def compute_joints_state(model: Model, data: ModelData):
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
            # Outputs:
            data.joints.p_j,
            data.joints.r_j,
            data.joints.dr_j,
            data.joints.q_j,
            data.joints.dq_j,
        ],
    )
