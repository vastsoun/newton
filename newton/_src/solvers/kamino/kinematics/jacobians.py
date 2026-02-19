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
KAMINO: Kinematics: Jacobians
"""

from __future__ import annotations

from typing import Any

import warp as wp
from warp.context import Devicelike

from ..core.joints import JointDoFType
from ..core.math import (
    FLOAT32_MAX,
    FLOAT32_MIN,
    contact_wrench_matrix_from_points,
    expand6d,
    screw_transform_matrix_from_points,
)
from ..core.model import Model, ModelData
from ..core.types import (
    float32,
    int32,
    mat33f,
    mat66f,
    quatf,
    transformf,
    vec2i,
    vec3f,
    vec6f,
)
from ..geometry.contacts import Contacts
from ..kinematics.limits import Limits
from ..linalg.sparse_matrix import BlockDType, BlockSparseMatrices
from ..linalg.sparse_operator import BlockSparseLinearOperators

###
# Module interface
###

__all__ = [
    "ColMajorSparseConstraintJacobians",
    "DenseSystemJacobians",
    "DenseSystemJacobiansData",
    "SparseSystemJacobians",
    "build_dense_jacobians",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


def make_store_joint_jacobian_dense_func(axes: Any):
    """
    Generates a warp function to store body-pair Jacobian blocks into a target flat
    data array given a vector of Jacobian row indices (i.e. selection vector).
    """

    @wp.func
    def store_joint_jacobian_dense(
        J_offset: int,
        row_offset: int,
        row_size: int,
        bid_offset: int,
        bid_B: int,
        bid_F: int,
        JT_B_j: mat66f,
        JT_F_j: mat66f,
        J_data: wp.array(dtype=float32),
    ):
        """
        Stores the Jacobian blocks of a joint into the provided flat data array at the specified offset.

        Args:
            J_offset (int): The offset at which the Jacobian matrix block of the corresponding world starts.
            row_offset (int): The offset at which the first target row starts.
            row_size (int): The number of columns in the world's Jacobian block.
            bid_offset (int): The body index offset of the world's bodies w.r.t the model.
            bid_B (int): The body index of the base body of the joint w.r.t the model.
            bid_F (int): The body index of the follower body of the joint w.r.t the model.
            JT_B_j (mat66f): The 6x6 Jacobian transpose block of the joint's base body.
            JT_F_j (mat66f): The 6x6 Jacobian transpose block of the joint's follower body.
            J_data (wp.array(dtype=float32)): The flat data array holding the Jacobian matrix blocks.
        """
        # Set the number of rows in the output Jacobian block
        # NOTE: This is evaluated statically at compile time
        num_jac_rows = wp.static(len(axes))

        # Append the row offset to the Jacobian matrix block offset
        # NOTE: This sets the adjusts the start index to the first target row
        J_offset += row_size * row_offset

        # Store the Jacobian block for the follower body
        J_offset_F = J_offset + 6 * (bid_F - bid_offset)
        for j in range(num_jac_rows):
            kj = J_offset_F + row_size * j
            for i in range(6):
                J_data[kj + i] = JT_F_j[i, axes[j]]

        # If the base body is not the world (:= -1), store the respective Jacobian block
        if bid_B > -1:
            J_offset_B = J_offset + 6 * (bid_B - bid_offset)
            for j in range(num_jac_rows):
                kj = J_offset_B + row_size * j
                for i in range(6):
                    J_data[kj + i] = JT_B_j[i, axes[j]]

    # Return the function
    return store_joint_jacobian_dense


def make_store_joint_jacobian_sparse_func(axes: Any):
    """
    Generates a warp function to store body-pair Jacobian blocks into a target
    block sparse data structure.
    """

    @wp.func
    def store_joint_jacobian_sparse(
        is_binary: bool,
        JT_B_j: mat66f,
        JT_F_j: mat66f,
        J_nzb_offset: int,
        J_nzb_values: wp.array(dtype=vec6f),
    ):
        """
        Function extracting rows corresponding to joint axes from the 6x6 joint jacobians,
        and storing them into the appropriate non-zero blocks.
        Args:
            is_binary (bool): Whether the joint is binary
            JT_B_j (mat66f): The 6x6 Jacobian transpose block of the joint's base body.
            JT_F_j (mat66f): The 6x6 Jacobian transpose block of the joint's follower body.
            J_nzb_offset (int): The index of the first nzb corresponding to this joint.
            J_nzb_values (wp.array(dtype=vec6f)): Array storing the non-zero blocks of the Jacobians.
        """
        # Set the number of rows in the output Jacobian block
        # NOTE: This is evaluated statically at compile time
        num_jac_rows = wp.static(len(axes))

        # Store the Jacobian block for the follower body
        for i in range(num_jac_rows):
            nzb_id = J_nzb_offset + i
            J_nzb_values[nzb_id] = JT_F_j[:, axes[i]]

        # Store the Jacobian block for the base body, for binary joints
        if is_binary:
            for i in range(num_jac_rows):
                nzb_id = J_nzb_offset + num_jac_rows + i
                J_nzb_values[nzb_id] = JT_B_j[:, axes[i]]

    # Return the function
    return store_joint_jacobian_sparse


@wp.func
def store_joint_cts_jacobian_dense(
    dof_type: int,
    J_cts_offset: int,
    cts_offset: int,
    num_body_dofs: int,
    bid_offset: int,
    bid_B: int,
    bid_F: int,
    JT_B: mat66f,
    JT_F: mat66f,
    J_cts_data: wp.array(dtype=float32),
):
    """
    Stores the constraints Jacobian block of a joint into the provided flat data array at the given offset.
    """

    if dof_type == JointDoFType.REVOLUTE:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.REVOLUTE.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.PRISMATIC.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.CYLINDRICAL.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.UNIVERSAL.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.SPHERICAL.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.GIMBAL.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.CARTESIAN:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.CARTESIAN.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.FIXED:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.FIXED.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )


@wp.func
def store_joint_dofs_jacobian_dense(
    dof_type: int,
    J_dofs_offset: int,
    dofs_offset: int,
    num_body_dofs: int,
    bid_offset: int,
    bid_B: int,
    bid_F: int,
    JT_B: mat66f,
    JT_F: mat66f,
    J_dofs_data: wp.array(dtype=float32),
):
    """
    Stores the DoFs Jacobian block of a joint into the provided flat data array at the given offset.
    """

    if dof_type == JointDoFType.REVOLUTE:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.REVOLUTE.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.PRISMATIC.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.CYLINDRICAL.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.UNIVERSAL.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.SPHERICAL.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.GIMBAL.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.CARTESIAN:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.CARTESIAN.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.FREE:
        wp.static(make_store_joint_jacobian_dense_func(JointDoFType.FREE.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )


@wp.func
def store_joint_cts_jacobian_sparse(
    dof_type: int,
    is_binary: bool,
    JT_B_j: mat66f,
    JT_F_j: mat66f,
    J_nzb_offset: int,
    J_nzb_values: wp.array(dtype=vec6f),
):
    """
    Stores the constraints Jacobian block of a joint into the provided flat data array at the given offset.
    """

    if dof_type == JointDoFType.REVOLUTE:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.REVOLUTE.cts_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.PRISMATIC.cts_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.CYLINDRICAL.cts_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.UNIVERSAL.cts_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.SPHERICAL.cts_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.GIMBAL.cts_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.CARTESIAN:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.CARTESIAN.cts_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.FIXED:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.FIXED.cts_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )


@wp.func
def store_joint_dofs_jacobian_sparse(
    dof_type: int,
    is_binary: bool,
    JT_B_j: mat66f,
    JT_F_j: mat66f,
    J_nzb_offset: int,
    J_nzb_values: wp.array(dtype=vec6f),
):
    """
    Stores the DoFs Jacobian block of a joint into the provided flat data array at the given offset.
    """

    if dof_type == JointDoFType.REVOLUTE:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.REVOLUTE.dofs_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.PRISMATIC.dofs_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.CYLINDRICAL.dofs_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.UNIVERSAL.dofs_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.SPHERICAL.dofs_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.GIMBAL.dofs_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.CARTESIAN:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.CARTESIAN.dofs_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )

    elif dof_type == JointDoFType.FREE:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.FREE.dofs_axes))(
            is_binary, JT_B_j, JT_F_j, J_nzb_offset, J_nzb_values
        )


###
# Kernels
###


@wp.kernel
def _build_joint_jacobians_dense(
    # Inputs
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_dof_type: wp.array(dtype=int32),
    model_joints_cts_offset: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    model_joints_X: wp.array(dtype=mat33f),
    state_joints_p: wp.array(dtype=transformf),
    state_bodies_q: wp.array(dtype=transformf),
    jacobian_cts_offsets: wp.array(dtype=int32),
    jacobian_dofs_offsets: wp.array(dtype=int32),
    # Outputs
    jacobian_cts_data: wp.array(dtype=float32),
    jacobian_dofs_data: wp.array(dtype=float32),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint model data
    wid = model_joints_wid[jid]
    dof_type = model_joints_dof_type[jid]
    cio = model_joints_cts_offset[jid]
    dio = model_joints_dofs_offset[jid]
    bid_B = model_joints_bid_B[jid]
    bid_F = model_joints_bid_F[jid]
    X_j = model_joints_X[jid]

    # Retrieve the number of body DoFs for corresponding world
    nbd = model_info_num_body_dofs[wid]
    bio = model_info_bodies_offset[wid]

    # Retrieve the Jacobian block offset for this joint
    cjmio = jacobian_cts_offsets[wid]
    djmio = jacobian_dofs_offsets[wid]

    # Retrieve the pose transform of the joint
    T_j = state_joints_p[jid]
    r_j = wp.transform_get_translation(T_j)
    R_j = wp.quat_to_matrix(wp.transform_get_rotation(T_j))

    # Retrieve the pose transforms of each body
    # NOTE: If the base body is the world (bid=-1), use the identity transform (frame of the world's origin)
    T_B_j = wp.transform_identity()
    if bid_B > -1:
        T_B_j = state_bodies_q[bid_B]
    T_F_j = state_bodies_q[bid_F]
    r_B_j = wp.transform_get_translation(T_B_j)
    r_F_j = wp.transform_get_translation(T_F_j)

    # Compute the wrench matrices
    # TODO: Since the lever-arm is a relative position, can we just use B_r_Bj and F_r_Fj instead?
    W_j_B = screw_transform_matrix_from_points(r_j, r_B_j)
    W_j_F = screw_transform_matrix_from_points(r_j, r_F_j)

    # Compute the effective projector to joint frame and expand to 6D
    R_X_j = R_j @ X_j
    R_X_bar_j = expand6d(R_X_j)

    # Compute the extended jacobians, i.e. without the selection-matrix multiplication
    JT_B_j = -W_j_B @ R_X_bar_j  # Reaction is on the Base body body ; (6 x 6)
    JT_F_j = W_j_F @ R_X_bar_j  # Action is on the Follower body    ; (6 x 6)

    # Store the constraint Jacobian block
    store_joint_cts_jacobian_dense(dof_type, cjmio, cio, nbd, bio, bid_B, bid_F, JT_B_j, JT_F_j, jacobian_cts_data)

    # Store the actuation Jacobian block if the joint is actuated
    store_joint_dofs_jacobian_dense(dof_type, djmio, dio, nbd, bio, bid_B, bid_F, JT_B_j, JT_F_j, jacobian_dofs_data)


@wp.kernel
def _configure_jacobians_sparse(
    # Input:
    model_num_joint_cts: wp.array(dtype=int32),
    num_limits: wp.array(dtype=int32),
    num_contacts: wp.array(dtype=int32),
    # Output:
    jac_cts_rows: wp.array(dtype=int32),
):
    world_id = wp.tid()

    jac_cts_rows[world_id] = model_num_joint_cts[world_id] + num_limits[world_id] + 3 * num_contacts[world_id]


@wp.kernel
def _build_joint_jacobians_sparse(
    # Inputs
    model_joints_dof_type: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    model_joints_X: wp.array(dtype=mat33f),
    state_joints_p: wp.array(dtype=transformf),
    state_bodies_q: wp.array(dtype=transformf),
    jacobian_cts_nzb_offsets: wp.array(dtype=int32),
    jacobian_dofs_nzb_offsets: wp.array(dtype=int32),
    # Outputs
    jacobian_cts_nzb_values: wp.array(dtype=vec6f),
    jacobian_dofs_nzb_values: wp.array(dtype=vec6f),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint model data
    dof_type = model_joints_dof_type[jid]
    bid_B = model_joints_bid_B[jid]
    bid_F = model_joints_bid_F[jid]
    X_j = model_joints_X[jid]

    # Retrieve the pose transform of the joint
    T_j = state_joints_p[jid]
    r_j = wp.transform_get_translation(T_j)
    R_j = wp.quat_to_matrix(wp.transform_get_rotation(T_j))

    # Retrieve the pose transforms of each body
    # NOTE: If the base body is the world (bid=-1), use the identity transform (frame of the world's origin)
    T_B_j = wp.transform_identity()
    if bid_B > -1:
        T_B_j = state_bodies_q[bid_B]
    T_F_j = state_bodies_q[bid_F]
    r_B_j = wp.transform_get_translation(T_B_j)
    r_F_j = wp.transform_get_translation(T_F_j)

    # Compute the wrench matrices
    # TODO: Since the lever-arm is a relative position, can we just use B_r_Bj and F_r_Fj instead?
    W_j_B = screw_transform_matrix_from_points(r_j, r_B_j)
    W_j_F = screw_transform_matrix_from_points(r_j, r_F_j)

    # Compute the effective projector to joint frame and expand to 6D
    R_X_j = R_j @ X_j
    R_X_bar_j = expand6d(R_X_j)

    # Compute the extended jacobians, i.e. without the selection-matrix multiplication
    JT_B_j = -W_j_B @ R_X_bar_j  # Reaction is on the Base body  ; (6 x 6)
    JT_F_j = W_j_F @ R_X_bar_j  # Action is on the Follower body ; (6 x 6)

    # Store the constraint Jacobian block
    store_joint_cts_jacobian_sparse(
        dof_type,
        bid_B > -1,
        JT_B_j,
        JT_F_j,
        jacobian_cts_nzb_offsets[jid],
        jacobian_cts_nzb_values,
    )

    # Store the actuation Jacobian block if the joint is actuated
    store_joint_dofs_jacobian_sparse(
        dof_type,
        bid_B > -1,
        JT_B_j,
        JT_F_j,
        jacobian_dofs_nzb_offsets[jid],
        jacobian_dofs_nzb_values,
    )


@wp.kernel
def _build_limit_jacobians_dense(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    state_info_limit_cts_group_offset: wp.array(dtype=int32),
    limits_model_num: wp.array(dtype=int32),
    limits_model_max: int32,
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    limits_dof: wp.array(dtype=int32),
    limits_side: wp.array(dtype=float32),
    jacobian_dofs_offsets: wp.array(dtype=int32),
    jacobian_dofs_data: wp.array(dtype=float32),
    jacobian_cts_offsets: wp.array(dtype=int32),
    # Outputs:
    jacobian_cts_data: wp.array(dtype=float32),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the limit index
    lid = wp.tid()

    # Skip if cid is greater than the total number of active limits in the model
    if lid >= wp.min(limits_model_num[0], limits_model_max):
        return

    # Retrieve the world index of the active limit
    wid_l = limits_wid[lid]

    # Retrieve the limit description info
    # NOTE: *_l is used to denote a subscript for the limit index
    lid_l = limits_lid[lid]
    bids_l = limits_bids[lid]
    dof_l = limits_dof[lid]
    side_l = limits_side[lid]

    # Retrieve the relevant model info of the world
    nbd = model_info_num_body_dofs[wid_l]
    bio = model_info_bodies_offset[wid_l]
    lcgo = state_info_limit_cts_group_offset[wid_l]
    ajmio = jacobian_dofs_offsets[wid_l]
    cjmio = jacobian_cts_offsets[wid_l]

    # Append the index offsets to the corresponding rows of the Jacobians
    ajmio += nbd * dof_l
    cjmio += nbd * (lcgo + lid_l)

    # Extract the body ids
    bid_B_l = bids_l[0]
    bid_F_l = bids_l[1]

    # Set the constraint Jacobian block for the follower body from the actuation Jacobian block
    bio_F = 6 * (bid_F_l - bio)
    act_kj = ajmio + bio_F
    cts_kj = cjmio + bio_F
    for i in range(6):
        jacobian_cts_data[cts_kj + i] = side_l * jacobian_dofs_data[act_kj + i]

    # If not the world body, set the constraint Jacobian block for the base body from the actuation Jacobian block
    if bid_B_l > -1:
        bio_B = 6 * (bid_B_l - bio)
        act_kj = ajmio + bio_B
        cts_kj = cjmio + bio_B
        for i in range(6):
            jacobian_cts_data[cts_kj + i] = side_l * jacobian_dofs_data[act_kj + i]


@wp.kernel
def _build_limit_jacobians_sparse(
    # Inputs:
    model_info_bodies_offset: wp.array(dtype=int32),
    model_info_joints_offset: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    state_info_limit_cts_group_offset: wp.array(dtype=int32),
    limits_model_num: wp.array(dtype=int32),
    limits_model_max: int32,
    limits_wid: wp.array(dtype=int32),
    limits_jid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    limits_dof: wp.array(dtype=int32),
    limits_side: wp.array(dtype=float32),
    jacobian_dofs_joint_nzb_offsets: wp.array(dtype=int32),
    jacobian_dofs_nzb_values: wp.array(dtype=vec6f),
    jacobian_cts_nzb_start: wp.array(dtype=int32),
    # Outputs:
    jacobian_cts_num_nzb: wp.array(dtype=int32),
    jacobian_cts_nzb_coords: wp.array2d(dtype=int32),
    jacobian_cts_nzb_values: wp.array(dtype=vec6f),
    jacobian_cts_limit_nzb_offsets: wp.array(dtype=int32),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the limit index
    limit_id = wp.tid()

    # Skip if cid is greater than the total number of active limits in the model
    if limit_id >= wp.min(limits_model_num[0], limits_model_max):
        return

    # Retrieve the world index of the active limit
    world_id = limits_wid[limit_id]

    # Retrieve the limit description info
    # NOTE: *_l is used to denote a subscript for the limit index
    limit_id_l = limits_lid[limit_id]
    body_ids_l = limits_bids[limit_id]
    body_id_B_l = body_ids_l[0]
    body_id_F_l = body_ids_l[1]
    dof_l = limits_dof[limit_id]
    side_l = limits_side[limit_id]
    joint_id = limits_jid[limit_id] + model_info_joints_offset[world_id]

    # Resolve which NZB of the dofs Jacobian corresponds to the limit's dof (on the follower)
    dof_id = dof_l - model_joints_dofs_offset[joint_id]  # Id of the dof among the joint's dof
    jac_dofs_nzb_idx = jacobian_dofs_joint_nzb_offsets[joint_id] + dof_id

    # Retrieve the relevant model info of the world
    body_index_offset = model_info_bodies_offset[world_id]
    limit_cts_offset = state_info_limit_cts_group_offset[world_id]

    # Create NZB(s)
    num_limit_nzb = 2 if body_id_B_l > -1 else 1
    jac_cts_nzb_offset_world = wp.atomic_add(jacobian_cts_num_nzb, world_id, num_limit_nzb)
    jac_cts_nzb_idx = jacobian_cts_nzb_start[world_id] + jac_cts_nzb_offset_world
    jacobian_cts_limit_nzb_offsets[limit_id] = jac_cts_nzb_idx

    # Set the constraint Jacobian block for the follower body from the actuation Jacobian block
    jacobian_cts_nzb_values[jac_cts_nzb_idx] = side_l * jacobian_dofs_nzb_values[jac_dofs_nzb_idx]
    jacobian_cts_nzb_coords[jac_cts_nzb_idx, 0] = limit_cts_offset + limit_id_l
    jacobian_cts_nzb_coords[jac_cts_nzb_idx, 1] = 6 * (body_id_F_l - body_index_offset)

    # If not the world body, set the constraint Jacobian block for the base body from the actuation Jacobian block
    if body_id_B_l > -1:
        nzb_stride = model_joints_num_dofs[joint_id]
        jacobian_cts_nzb_values[jac_cts_nzb_idx + 1] = side_l * jacobian_dofs_nzb_values[jac_dofs_nzb_idx + nzb_stride]
        jacobian_cts_nzb_coords[jac_cts_nzb_idx + 1, 0] = limit_cts_offset + limit_id_l
        jacobian_cts_nzb_coords[jac_cts_nzb_idx + 1, 1] = 6 * (body_id_B_l - body_index_offset)


@wp.kernel
def _build_contact_jacobians_dense(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    state_info_contact_cts_group_offset: wp.array(dtype=int32),
    state_bodies_q: wp.array(dtype=transformf),
    contacts_model_num: wp.array(dtype=int32),
    contacts_model_max: int32,
    contacts_wid: wp.array(dtype=int32),
    contacts_cid: wp.array(dtype=int32),
    contacts_bid_AB: wp.array(dtype=vec2i),
    contacts_position_A: wp.array(dtype=vec3f),
    contacts_position_B: wp.array(dtype=vec3f),
    contacts_frame: wp.array(dtype=quatf),
    jacobian_cts_offsets: wp.array(dtype=int32),
    # Outputs:
    jacobian_cts_data: wp.array(dtype=float32),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the contact index
    cid = wp.tid()

    # Skip if cid is greater than the total number of active contacts in the model
    if cid >= wp.min(contacts_model_num[0], contacts_model_max):
        return

    # Retrieve the contact index w.r.t the world
    # NOTE: k denotes a notational subscript for the
    # contact index, i.e. C_k is the k-th contact entity
    cid_k = contacts_cid[cid]

    # Retrieve the the contact-specific data
    wid = contacts_wid[cid]
    q_k = contacts_frame[cid]
    bid_AB_k = contacts_bid_AB[cid]
    r_Ac_k = contacts_position_A[cid]
    r_Bc_k = contacts_position_B[cid]

    # Retrieve the relevant model info for the world
    nbd = model_info_num_body_dofs[wid]
    bio = model_info_bodies_offset[wid]
    ccgo = state_info_contact_cts_group_offset[wid]
    cjmio = jacobian_cts_offsets[wid]

    # Append the index offset for the contact Jacobian block in the constraint Jacobian
    cjmio += ccgo * nbd

    # Extract the individual body indices
    bid_A_k = bid_AB_k[0]
    bid_B_k = bid_AB_k[1]

    # Compute the rotation matrix from the contact frame quaternion
    R_k = wp.quat_to_matrix(q_k)  # (3 x 3)

    # Set the constraint index offset for this contact
    cio_k = 3 * cid_k

    # Compute and store the revolute Jacobian block for the follower body (subject of action)
    r_B_k = wp.transform_get_translation(state_bodies_q[bid_B_k])
    W_B_k = contact_wrench_matrix_from_points(r_Bc_k, r_B_k)
    JT_c_B_k = W_B_k @ R_k  # Action is on the follower body (B)  ; (6 x 3)
    bio_B = 6 * (bid_B_k - bio)
    for j in range(3):
        kj = cjmio + nbd * (cio_k + j) + bio_B
        for i in range(6):
            jacobian_cts_data[kj + i] = JT_c_B_k[i, j]

    # If not the world body, compute and store the revolute Jacobian block for the base body (subject of reaction)
    if bid_A_k > -1:
        r_A_k = wp.transform_get_translation(state_bodies_q[bid_A_k])
        W_A_k = contact_wrench_matrix_from_points(r_Ac_k, r_A_k)
        JT_c_A_k = -W_A_k @ R_k  # Reaction is on the base body (A)    ; (6 x 3)
        bio_A = 6 * (bid_A_k - bio)
        for j in range(3):
            kj = cjmio + nbd * (cio_k + j) + bio_A
            for i in range(6):
                jacobian_cts_data[kj + i] = JT_c_A_k[i, j]


@wp.kernel
def _build_contact_jacobians_sparse(
    # Inputs:
    model_info_bodies_offset: wp.array(dtype=int32),
    state_info_contact_cts_group_offset: wp.array(dtype=int32),
    state_bodies_q: wp.array(dtype=transformf),
    contacts_model_num: wp.array(dtype=int32),
    contacts_model_max: int32,
    contacts_wid: wp.array(dtype=int32),
    contacts_cid: wp.array(dtype=int32),
    contacts_bid_AB: wp.array(dtype=vec2i),
    contacts_position_A: wp.array(dtype=vec3f),
    contacts_position_B: wp.array(dtype=vec3f),
    contacts_frame: wp.array(dtype=quatf),
    jacobian_cts_nzb_start: wp.array(dtype=int32),
    # Outputs:
    jacobian_cts_num_nzb: wp.array(dtype=int32),
    jacobian_cts_nzb_coords: wp.array2d(dtype=int32),
    jacobian_cts_nzb_values: wp.array(dtype=vec6f),
    jacobian_cts_contact_nzb_offsets: wp.array(dtype=int32),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the contact index
    contact_id = wp.tid()

    # Skip if cid is greater than the total number of active contacts in the model
    if contact_id >= wp.min(contacts_model_num[0], contacts_model_max):
        return

    # Retrieve the contact index w.r.t the world
    # NOTE: k denotes a notational subscript for the
    # contact index, i.e. C_k is the k-th contact entity
    contact_id_k = contacts_cid[contact_id]

    # Retrieve the the contact-specific data
    world_id = contacts_wid[contact_id]
    q_k = contacts_frame[contact_id]
    body_ids_k = contacts_bid_AB[contact_id]
    body_id_A_k = body_ids_k[0]
    body_id_B_k = body_ids_k[1]
    r_Ac_k = contacts_position_A[contact_id]
    r_Bc_k = contacts_position_B[contact_id]

    # Retrieve the relevant model info for the world
    body_idx_offset = model_info_bodies_offset[world_id]
    contact_cts_offset = state_info_contact_cts_group_offset[world_id]

    # Compute the rotation matrix from the contact frame quaternion
    R_k = wp.quat_to_matrix(q_k)  # (3 x 3)

    # Set the start constraint index for this contact
    cts_idx_start = 3 * contact_id_k + contact_cts_offset

    # Compute and store the revolute Jacobian block for the follower body (subject of action)
    r_B_k = wp.transform_get_translation(state_bodies_q[body_id_B_k])
    W_B_k = contact_wrench_matrix_from_points(r_Bc_k, r_B_k)
    JT_c_B_k = W_B_k @ R_k  # Action is on the follower body (B)  ; (6 x 3)
    body_idx_offset_B = 6 * (body_id_B_k - body_idx_offset)
    num_contact_nzb = 6 if body_id_A_k > -1 else 3
    # Allocate non-zero blocks in the Jacobian by incrementing the number of NZB
    jac_cts_nzb_offset_world = wp.atomic_add(jacobian_cts_num_nzb, world_id, num_contact_nzb)
    jac_cts_nzb_offset = jacobian_cts_nzb_start[world_id] + jac_cts_nzb_offset_world
    jacobian_cts_contact_nzb_offsets[contact_id] = jac_cts_nzb_offset
    # Store 6x3 Jacobian block as three separate 6x1 blocks
    for j in range(3):
        jacobian_cts_nzb_values[jac_cts_nzb_offset + j] = JT_c_B_k[:, j]
        jacobian_cts_nzb_coords[jac_cts_nzb_offset + j, 0] = cts_idx_start + j
        jacobian_cts_nzb_coords[jac_cts_nzb_offset + j, 1] = body_idx_offset_B

    # If not the world body, compute and store the revolute Jacobian block for the base body (subject of reaction)
    if body_id_A_k > -1:
        r_A_k = wp.transform_get_translation(state_bodies_q[body_id_A_k])
        W_A_k = contact_wrench_matrix_from_points(r_Ac_k, r_A_k)
        JT_c_A_k = -W_A_k @ R_k  # Reaction is on the base body (A)    ; (6 x 3)
        body_idx_offset_A = 6 * (body_id_A_k - body_idx_offset)
        # Store 6x3 Jacobian block as three separate 6x1 blocks
        for j in range(3):
            jacobian_cts_nzb_values[jac_cts_nzb_offset + 3 + j] = JT_c_A_k[:, j]
            jacobian_cts_nzb_coords[jac_cts_nzb_offset + 3 + j, 0] = cts_idx_start + j
            jacobian_cts_nzb_coords[jac_cts_nzb_offset + 3 + j, 1] = body_idx_offset_A


@wp.func
def store_col_major_jacobian_block(
    nzb_id: int32,
    row_id: int32,
    col_id: int32,
    block: mat66f,
    nzb_coords: wp.array2d(dtype=int32),
    nzb_values: wp.array(dtype=wp.types.matrix(shape=(6, 1), dtype=float32)),
):
    for i in range(6):
        nzb_id_i = nzb_id + i
        nzb_coords[nzb_id_i, 0] = row_id
        nzb_coords[nzb_id_i, 1] = col_id + i
        for j in range(6):
            nzb_values[nzb_id_i][j, 0] = block[j, i]


@wp.kernel
def _update_col_major_joint_jacobians(
    # Inputs
    model_joints_wid: wp.array(dtype=int32),
    model_joints_num_cts: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    jac_cts_row_major_joint_nzb_offsets: wp.array(dtype=int32),
    jac_cts_row_major_nzb_coords: wp.array2d(dtype=int32),
    jac_cts_row_major_nzb_values: wp.array(dtype=vec6f),
    jac_cts_col_major_nzb_start: wp.array(dtype=int32),
    jac_cts_col_major_joint_nzb_offsets: wp.array(dtype=int32),
    # Outputs
    jac_cts_col_major_nzb_coords: wp.array2d(dtype=int32),
    jac_cts_col_major_nzb_values: wp.array(dtype=wp.types.matrix(shape=(6, 1), dtype=float32)),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint model data
    wid = model_joints_wid[jid]
    num_cts = model_joints_num_cts[jid]
    bid_B = model_joints_bid_B[jid]

    # Retrieve the Jacobian data
    nzb_start_rm_j = jac_cts_row_major_joint_nzb_offsets[jid]
    nzb_row_init = jac_cts_row_major_nzb_coords[nzb_start_rm_j, 0]
    nzb_col_init_F = jac_cts_row_major_nzb_coords[nzb_start_rm_j, 1]

    nzb_start_cm = jac_cts_col_major_nzb_start[wid]
    nzb_offset_cm = jac_cts_col_major_joint_nzb_offsets[jid]

    block_F = mat66f(0.0)

    # Offset the Jacobian rows within the 6x6 block to avoid exceeding matrix dimensions.
    # Since we might not fill the full 6x6 block with Jacobian entries, shifting the block upwards
    # and filling the bottom part will prevent the block lying outside the matrix dimensions.
    # We additional guard against the case where the shift would push the block above the start of
    # the matrix by taking the minimum of the full shift and `nzb_row_init`.
    block_row_init = min(6 - num_cts, nzb_row_init)
    nzb_row_init -= block_row_init
    for i in range(num_cts):
        nzb_idx_rm = nzb_start_rm_j + i
        block_rm = jac_cts_row_major_nzb_values[nzb_idx_rm]
        block_F[block_row_init + i] = block_rm

    store_col_major_jacobian_block(
        nzb_start_cm + nzb_offset_cm,
        nzb_row_init,
        nzb_col_init_F,
        block_F,
        jac_cts_col_major_nzb_coords,
        jac_cts_col_major_nzb_values,
    )

    if bid_B > -1:
        nzb_col_init_B = jac_cts_row_major_nzb_coords[nzb_start_rm_j + num_cts, 1]
        block_B = mat66f(0.0)
        for i in range(num_cts):
            nzb_idx_rm = nzb_start_rm_j + num_cts + i
            block_rm = jac_cts_row_major_nzb_values[nzb_idx_rm]
            block_B[block_row_init + i] = block_rm

        store_col_major_jacobian_block(
            nzb_start_cm + nzb_offset_cm + 6,
            nzb_row_init,
            nzb_col_init_B,
            block_B,
            jac_cts_col_major_nzb_coords,
            jac_cts_col_major_nzb_values,
        )


@wp.kernel
def _update_col_major_limit_jacobians(
    # Inputs
    limits_model_num: wp.array(dtype=int32),
    limits_model_max: int32,
    limits_wid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    jac_cts_row_major_limit_nzb_offsets: wp.array(dtype=int32),
    jac_cts_row_major_nzb_coords: wp.array2d(dtype=int32),
    jac_cts_row_major_nzb_values: wp.array(dtype=vec6f),
    jac_cts_col_major_nzb_start: wp.array(dtype=int32),
    # Outputs
    jac_cts_col_major_num_nzb: wp.array(dtype=int32),
    jac_cts_col_major_nzb_coords: wp.array2d(dtype=int32),
    jac_cts_col_major_nzb_values: wp.array(dtype=wp.types.matrix(shape=(6, 1), dtype=float32)),
):
    """
    A kernel to assemble the limit constraint Jacobian in a model.
    """

    # Retrieve the thread index as the limit index
    limit_id = wp.tid()

    # Skip if cid is greater than the total number of active limits in the model
    if limit_id >= wp.min(limits_model_num[0], limits_model_max):
        return

    # Retrieve the world index of the active limit
    world_id = limits_wid[limit_id]

    # Retrieve the limit description info
    # NOTE: *_l is used to denote a subscript for the limit index
    body_ids_l = limits_bids[limit_id]
    body_id_B_l = body_ids_l[0]

    # Set the constraint Jacobian block for the follower body from the actuation Jacobian block
    num_limit_nzb = 12 if body_id_B_l > -1 else 6
    nzb_offset_cm = wp.atomic_add(jac_cts_col_major_num_nzb, world_id, num_limit_nzb)

    # Retrieve the Jacobian data
    nzb_start_rm_l = jac_cts_row_major_limit_nzb_offsets[limit_id]
    nzb_row_init = jac_cts_row_major_nzb_coords[nzb_start_rm_l, 0]
    nzb_col_init_F = jac_cts_row_major_nzb_coords[nzb_start_rm_l, 1]

    nzb_start_cm = jac_cts_col_major_nzb_start[world_id]

    # Offset the Jacobian rows within the 6x6 block to avoid exceeding matrix dimensions.
    # Since we might not fill the full 6x6 block with Jacobian entries, shifting the block upwards
    # and filling the bottom part will prevent the block lying outside the matrix dimensions.
    # We additional guard against the case where the shift would push the block above the start of
    # the matrix by taking the minimum of the full shift and `nzb_row_init`.
    block_row_init = min(5, nzb_row_init)
    nzb_row_init -= block_row_init

    block_F = mat66f(0.0)
    block_F[block_row_init] = jac_cts_row_major_nzb_values[nzb_start_rm_l]

    store_col_major_jacobian_block(
        nzb_start_cm + nzb_offset_cm,
        nzb_row_init,
        nzb_col_init_F,
        block_F,
        jac_cts_col_major_nzb_coords,
        jac_cts_col_major_nzb_values,
    )

    if body_id_B_l > -1:
        nzb_col_init_B = jac_cts_row_major_nzb_coords[nzb_start_rm_l + 1, 1]

        block_B = mat66f(0.0)
        block_B[block_row_init] = jac_cts_row_major_nzb_values[nzb_start_rm_l + 1]

        store_col_major_jacobian_block(
            nzb_start_cm + nzb_offset_cm + 6,
            nzb_row_init,
            nzb_col_init_B,
            block_B,
            jac_cts_col_major_nzb_coords,
            jac_cts_col_major_nzb_values,
        )


@wp.kernel
def _update_col_major_contact_jacobians(
    # Inputs:
    contacts_model_num: wp.array(dtype=int32),
    contacts_model_max: int32,
    contacts_wid: wp.array(dtype=int32),
    contacts_bid_AB: wp.array(dtype=vec2i),
    jac_cts_row_major_contact_nzb_offsets: wp.array(dtype=int32),
    jac_cts_row_major_nzb_coords: wp.array2d(dtype=int32),
    jac_cts_row_major_nzb_values: wp.array(dtype=vec6f),
    jac_cts_col_major_nzb_start: wp.array(dtype=int32),
    # Outputs
    jac_cts_col_major_num_nzb: wp.array(dtype=int32),
    jac_cts_col_major_nzb_coords: wp.array2d(dtype=int32),
    jac_cts_col_major_nzb_values: wp.array(dtype=wp.types.matrix(shape=(6, 1), dtype=float32)),
):
    """
    A kernel to assemble the contact constraint Jacobian in a model.
    """

    # Retrieve the thread index as the contact index
    contact_id = wp.tid()

    # Skip if cid is greater than the total number of active contacts in the model
    if contact_id >= wp.min(contacts_model_num[0], contacts_model_max):
        return

    # Retrieve the the contact-specific data
    world_id = contacts_wid[contact_id]
    body_ids_k = contacts_bid_AB[contact_id]
    body_id_A_k = body_ids_k[0]

    # Set the constraint Jacobian block for the follower body from the actuation Jacobian block
    num_contact_nzb = 12 if body_id_A_k > -1 else 6
    nzb_offset_cm = wp.atomic_add(jac_cts_col_major_num_nzb, world_id, num_contact_nzb)

    # Retrieve the Jacobian data
    nzb_start_rm_c = jac_cts_row_major_contact_nzb_offsets[contact_id]
    nzb_row_init = jac_cts_row_major_nzb_coords[nzb_start_rm_c, 0]
    nzb_col_init_F = jac_cts_row_major_nzb_coords[nzb_start_rm_c, 1]

    nzb_start_cm = jac_cts_col_major_nzb_start[world_id]

    # Offset the Jacobian rows within the 6x6 block to avoid exceeding matrix dimensions.
    # Since we might not fill the full 6x6 block with Jacobian entries, shifting the block upwards
    # and filling the bottom part will prevent the block lying outside the matrix dimensions.
    # We additional guard against the case where the shift would push the block above the start of
    # the matrix by taking the minimum of the full shift and `nzb_row_init`.
    block_row_init = min(3, nzb_row_init)
    nzb_row_init -= block_row_init

    block_F = mat66f(0.0)
    for i in range(3):
        block_F[block_row_init + i] = jac_cts_row_major_nzb_values[nzb_start_rm_c + i]

    store_col_major_jacobian_block(
        nzb_start_cm + nzb_offset_cm,
        nzb_row_init,
        nzb_col_init_F,
        block_F,
        jac_cts_col_major_nzb_coords,
        jac_cts_col_major_nzb_values,
    )

    if body_id_A_k > -1:
        nzb_col_init_B = jac_cts_row_major_nzb_coords[nzb_start_rm_c + 3, 1]

        block_B = mat66f(0.0)
        for i in range(3):
            block_B[block_row_init + i] = jac_cts_row_major_nzb_values[nzb_start_rm_c + 3 + i]

        store_col_major_jacobian_block(
            nzb_start_cm + nzb_offset_cm + 6,
            nzb_row_init,
            nzb_col_init_B,
            block_B,
            jac_cts_col_major_nzb_coords,
            jac_cts_col_major_nzb_values,
        )


###
# Launchers
###


def build_dense_jacobians(
    model: Model,
    data: ModelData,
    limits: Limits | None,
    contacts: Contacts | None,
    jacobian_cts_offsets: wp.array,
    jacobian_cts_data: wp.array,
    jacobian_dofs_offsets: wp.array,
    jacobian_dofs_data: wp.array,
    reset_to_zero: bool = True,
):
    # Optionally reset the Jacobian array data to zero
    if reset_to_zero:
        jacobian_cts_data.zero_()
        jacobian_dofs_data.zero_()

    # Build the joint constraints and actuation Jacobians
    if model.size.sum_of_num_joints > 0:
        wp.launch(
            _build_joint_jacobians_dense,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                model.joints.wid,
                model.joints.dof_type,
                model.joints.cts_offset,
                model.joints.dofs_offset,
                model.joints.bid_B,
                model.joints.bid_F,
                model.joints.X_j,
                data.joints.p_j,
                data.bodies.q_i,
                jacobian_cts_offsets,
                jacobian_dofs_offsets,
                # Outputs:
                jacobian_cts_data,
                jacobian_dofs_data,
            ],
        )

    # Build the limit constraints Jacobians if a limits data container is provided
    if limits is not None and limits.model_max_limits_host > 0:
        wp.launch(
            _build_limit_jacobians_dense,
            dim=limits.model_max_limits_host,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                data.info.limit_cts_group_offset,
                limits.model_active_limits,
                limits.model_max_limits_host,
                limits.wid,
                limits.lid,
                limits.bids,
                limits.dof,
                limits.side,
                jacobian_dofs_offsets,
                jacobian_dofs_data,
                jacobian_cts_offsets,
                # Outputs:
                jacobian_cts_data,
            ],
        )

    # Build the contact constraints Jacobians if a contacts data container is provided
    if contacts is not None and contacts.model_max_contacts_host > 0:
        wp.launch(
            _build_contact_jacobians_dense,
            dim=contacts.model_max_contacts_host,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                data.info.contact_cts_group_offset,
                data.bodies.q_i,
                contacts.model_active_contacts,
                contacts.model_max_contacts_host,
                contacts.wid,
                contacts.cid,
                contacts.bid_AB,
                contacts.position_A,
                contacts.position_B,
                contacts.frame,
                jacobian_cts_offsets,
                # Outputs:
                jacobian_cts_data,
            ],
        )


###
# Types
###


class DenseSystemJacobiansData:
    """
    Container to hold time-varying Jacobians of the system.
    """

    def __init__(self):
        ###
        # Constraint Jacobian
        ###

        self.J_cts_offsets: wp.array(dtype=int32) | None = None
        """
        The index offset of the constraint Jacobian matrix block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        self.J_cts_data: wp.array(dtype=float32) | None = None
        """
        A flat array containing the joint constraint Jacobian matrix data of all worlds.\n
        Shape of ``(sum(ncts_w * nbd_w),)`` and type :class:`float32`.
        """

        ###
        # DoFs Jacobian
        ###

        self.J_dofs_offsets: wp.array(dtype=int32) | None = None
        """
        The index offset of the DoF Jacobian matrix block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        self.J_dofs_data: wp.array(dtype=float32) | None = None
        """
        A flat array containing the joint DoF Jacobian matrix data of all worlds.\n
        Shape of ``(sum(njad_w * nbd_w),)`` and type :class:`float32`.
        """


###
# Interfaces
###


class DenseSystemJacobians:
    """
    Container to hold time-varying Jacobians of the system.
    """

    def __init__(
        self,
        model: Model | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        device: Devicelike = None,
    ):
        # Declare and initialize the Jacobian data container
        self._data = DenseSystemJacobiansData()

        # If a model is provided, allocate the Jacobians data
        if model is not None:
            self.finalize(model=model, limits=limits, contacts=contacts, device=device)

    @property
    def data(self) -> DenseSystemJacobiansData:
        """
        Returns the internal data container holding the Jacobians data.
        """
        return self._data

    def finalize(
        self, model: Model, limits: Limits | None = None, contacts: Contacts | None = None, device: Devicelike = None
    ):
        # Ensure the model container is valid
        if model is None:
            raise ValueError("`model` is required but got `None`.")
        else:
            if not isinstance(model, Model):
                raise TypeError(f"`model` is required to be of type `Model` but got {type(model)}.")

        # Ensure the limits container is valid
        if limits is not None:
            if not isinstance(limits, Limits):
                raise TypeError(f"`limits` is required to be of type `Limits` but got {type(limits)}.")

        # Ensure the contacts container is valid
        if contacts is not None:
            if not isinstance(contacts, Contacts):
                raise TypeError(f"`contacts` is required to be of type `Contacts` but got {type(contacts)}.")

        # Extract the constraint and DoF sizes of each world
        nw = model.info.num_worlds
        nbd = [model.worlds[w].num_body_dofs for w in range(nw)]
        njc = [model.worlds[w].num_joint_cts for w in range(nw)]
        njd = [model.worlds[w].num_joint_dofs for w in range(nw)]
        maxnl = limits.world_max_limits_host if limits and limits.model_max_limits_host > 0 else [0] * nw
        maxnc = contacts.world_max_contacts_host if contacts and contacts.model_max_contacts_host > 0 else [0] * nw
        maxncts = [njc[w] + maxnl[w] + 3 * maxnc[w] for w in range(nw)]

        # Compute the sizes of the Jacobian matrix data for each world
        J_cts_sizes = [maxncts[i] * nbd[i] for i in range(nw)]
        J_dofs_sizes = [njd[i] * nbd[i] for i in range(nw)]

        # Compute the total size of the Jacobian matrix data
        total_J_cts_size = sum(J_cts_sizes)
        total_J_dofs_size = sum(J_dofs_sizes)

        # Compute matrix index offsets of each Jacobian block
        J_cts_offsets = [0] * nw
        J_dofs_offsets = [0] * nw
        for w in range(1, nw):
            J_cts_offsets[w] = J_cts_offsets[w - 1] + J_cts_sizes[w - 1]
            J_dofs_offsets[w] = J_dofs_offsets[w - 1] + J_dofs_sizes[w - 1]

        # Allocate the Jacobian arrays
        with wp.ScopedDevice(device):
            self._data.J_cts_offsets = wp.array(J_cts_offsets, dtype=int32)
            self._data.J_dofs_offsets = wp.array(J_dofs_offsets, dtype=int32)
            self._data.J_cts_data = wp.zeros(shape=(total_J_cts_size,), dtype=float32)
            self._data.J_dofs_data = wp.zeros(shape=(total_J_dofs_size,), dtype=float32)

    def build(
        self,
        model: Model,
        data: ModelData,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        reset_to_zero: bool = True,
    ):
        """
        Builds the system DoF and constraint Jacobians for the given
        data of the provided model, data, limits and contacts containers.
        """
        build_dense_jacobians(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            jacobian_cts_offsets=self._data.J_cts_offsets,
            jacobian_cts_data=self._data.J_cts_data,
            jacobian_dofs_offsets=self._data.J_dofs_offsets,
            jacobian_dofs_data=self._data.J_dofs_data,
            reset_to_zero=reset_to_zero,
        )


class SparseSystemJacobians:
    """
    Container to hold time-varying Jacobians of the system in block-sparse format.
    """

    def __init__(
        self,
        model: Model | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        device: Devicelike = None,
    ):
        # Declare and initialize the Jacobian data containers
        self._J_cts: BlockSparseLinearOperators | None = None
        self._J_dofs: BlockSparseLinearOperators | None = None
        # Local (in-world) offsets for the non-zero blocks of the constraint and dofs Jacobian for
        # each (global) joint, limit, and contact
        self._J_cts_joint_nzb_offsets: wp.array | None = None
        self._J_cts_limit_nzb_offsets: wp.array | None = None
        self._J_cts_contact_nzb_offsets: wp.array | None = None
        self._J_dofs_joint_nzb_offsets: wp.array | None = None
        # Lists of number of non-zero blocks in each world connected to joint constraints
        self._J_cts_num_joint_nzb: wp.array | None = None

        # If a model is provided, allocate the Jacobians data
        if model is not None:
            self.finalize(model=model, limits=limits, contacts=contacts, device=device)

    def finalize(
        self, model: Model, limits: Limits | None = None, contacts: Contacts | None = None, device: Devicelike = None
    ):
        # Ensure the model container is valid
        if model is None:
            raise ValueError("`model` is required but got `None`.")
        else:
            if not isinstance(model, Model):
                raise TypeError(f"`model` is required to be of type `Model` but got {type(model)}.")

        # Ensure the limits container is valid
        if limits is not None:
            if not isinstance(limits, Limits):
                raise TypeError(f"`limits` is required to be of type `Limits` but got {type(limits)}.")

        # Ensure the contacts container is valid
        if contacts is not None:
            if not isinstance(contacts, Contacts):
                raise TypeError(f"`contacts` is required to be of type `Contacts` but got {type(contacts)}.")

        # Extract the constraint and DoF sizes of each world
        num_worlds = model.info.num_worlds
        num_body_dofs = [model.worlds[w].num_body_dofs for w in range(num_worlds)]
        num_joint_cts = [model.worlds[w].num_joint_cts for w in range(num_worlds)]
        num_joint_dofs = [model.worlds[w].num_joint_dofs for w in range(num_worlds)]
        max_num_limits = (
            limits.world_max_limits_host if limits and limits.model_max_limits_host > 0 else [0] * num_worlds
        )
        max_num_contacts = (
            contacts.world_max_contacts_host if contacts and contacts.model_max_contacts_host > 0 else [0] * num_worlds
        )
        max_num_constraints = [
            num_joint_cts[w] + max_num_limits[w] + 3 * max_num_contacts[w] for w in range(num_worlds)
        ]

        # Compute the number of non-zero blocks required for each Jacobian matrix, as well as the
        # per-joint and per-dof offsets, and nzb coordinates.
        joint_wid = model.joints.wid.numpy()
        joint_bid_B = model.joints.bid_B.numpy()
        joint_bid_F = model.joints.bid_F.numpy()
        joint_num_cts = model.joints.num_cts.numpy()
        joint_num_dofs = model.joints.num_dofs.numpy()
        joint_q_j_min = model.joints.q_j_min.numpy()
        joint_q_j_max = model.joints.q_j_max.numpy()
        joint_cts_offset = model.joints.cts_offset.numpy()
        joint_dofs_offset = model.joints.dofs_offset.numpy()
        bodies_offset = model.info.bodies_offset.numpy()
        J_cts_nnzb_min = [0] * num_worlds
        J_cts_nnzb_max = [0] * num_worlds
        J_dofs_nnzb = [0] * num_worlds
        J_cts_joint_nzb_offsets = [0] * model.size.sum_of_num_joints
        J_dofs_joint_nzb_offsets = [0] * model.size.sum_of_num_joints
        J_cts_nzb_row = [[] for _ in range(num_worlds)]
        J_cts_nzb_col = [[] for _ in range(num_worlds)]
        J_dofs_nzb_row = [[] for _ in range(num_worlds)]
        J_dofs_nzb_col = [[] for _ in range(num_worlds)]
        dofs_start = 0
        # Add non-zero blocks for joints and joint limits
        for _j in range(model.size.sum_of_num_joints):
            w = joint_wid[_j]
            J_cts_joint_nzb_offsets[_j] = J_cts_nnzb_min[w]
            J_dofs_joint_nzb_offsets[_j] = J_dofs_nnzb[w]

            # Joint nzb counts
            is_binary = joint_bid_B[_j] > -1
            num_adjacent_bodies = 2 if is_binary else 1
            num_cts = int(joint_num_cts[_j])
            num_dofs = int(joint_num_dofs[_j])
            J_cts_nnzb_min[w] += num_adjacent_bodies * num_cts
            J_cts_nnzb_max[w] += num_adjacent_bodies * num_cts
            J_dofs_nnzb[w] += num_adjacent_bodies * num_dofs

            # Joint nzb coordinates
            cts_offset = joint_cts_offset[_j]
            dofs_offset = joint_dofs_offset[_j]
            for _ in range(num_adjacent_bodies):
                for i in range(num_cts):
                    J_cts_nzb_row[w].append(cts_offset + i)
                for i in range(num_dofs):
                    J_dofs_nzb_row[w].append(dofs_offset + i)
            col_id = 6 * (joint_bid_F[_j] - bodies_offset[w])
            J_cts_nzb_col[w].extend(num_cts * [col_id])
            J_dofs_nzb_col[w].extend(num_dofs * [col_id])
            if is_binary:
                col_id = 6 * (joint_bid_B[_j] - bodies_offset[w])
                J_cts_nzb_col[w].extend(num_cts * [col_id])
                J_dofs_nzb_col[w].extend(num_dofs * [col_id])

            # Limit nzb counts (maximum)
            if max_num_limits[w] > 0:
                for d_j in range(num_dofs):
                    if joint_q_j_min[dofs_start + d_j] > float(FLOAT32_MIN) or joint_q_j_max[dofs_start + d_j] < float(
                        FLOAT32_MAX
                    ):
                        J_cts_nnzb_max[w] += num_adjacent_bodies
            dofs_start += num_dofs
        # Add non-zero blocks for contacts
        # TODO: Use the candidate geom-pair info to compute maximum possible contact constraint blocks more accurately
        if contacts is not None and contacts.model_max_contacts_host > 0:
            for w in range(num_worlds):
                J_cts_nnzb_max[w] += 2 * 3 * max_num_contacts[w]

        # Compute the sizes of the Jacobian matrix data for each world
        J_cts_dims_max = [(max_num_constraints[i], num_body_dofs[i]) for i in range(num_worlds)]
        J_dofs_dims = [(num_joint_dofs[i], num_body_dofs[i]) for i in range(num_worlds)]

        # Flatten nzb coordinates
        for w in range(num_worlds):
            J_cts_nzb_row[w] += [0] * (J_cts_nnzb_max[w] - J_cts_nnzb_min[w])
            J_cts_nzb_col[w] += [0] * (J_cts_nnzb_max[w] - J_cts_nnzb_min[w])
        J_cts_nzb_row = [i for rows in J_cts_nzb_row for i in rows]
        J_cts_nzb_col = [j for cols in J_cts_nzb_col for j in cols]
        J_dofs_nzb_row = [i for rows in J_dofs_nzb_row for i in rows]
        J_dofs_nzb_col = [j for cols in J_dofs_nzb_col for j in cols]

        # Allocate the block-sparse linear-operator data to represent each system Jacobian
        with wp.ScopedDevice(device):
            # First allocate the geometric constraint Jacobian
            bsm_cts = BlockSparseMatrices(
                num_matrices=num_worlds,
                nzb_dtype=BlockDType(dtype=float32, shape=(6,)),
                device=device,
            )
            bsm_cts.finalize(max_dims=J_cts_dims_max, capacities=J_cts_nnzb_max)
            self._J_cts = BlockSparseLinearOperators(bsm=bsm_cts)

            # Then allocate the geometric DoFs Jacobian
            bsm_dofs = BlockSparseMatrices(
                num_matrices=num_worlds,
                nzb_dtype=BlockDType(dtype=float32, shape=(6,)),
                device=device,
            )
            bsm_dofs.finalize(max_dims=J_dofs_dims, capacities=J_dofs_nnzb)
            self._J_dofs = BlockSparseLinearOperators(bsm=bsm_dofs)

            # Set all constant values into BSMs (corresponding to joint dofs/cts)
            bsm_cts.nzb_row.assign(J_cts_nzb_row)
            bsm_cts.nzb_col.assign(J_cts_nzb_col)
            bsm_cts.num_cols.assign(num_body_dofs)
            bsm_dofs.nzb_row.assign(J_dofs_nzb_row)
            bsm_dofs.nzb_col.assign(J_dofs_nzb_col)
            bsm_dofs.num_rows.assign(num_joint_dofs)
            bsm_dofs.num_cols.assign(num_body_dofs)
            bsm_dofs.num_nzb.assign(J_dofs_nnzb)

            # Convert per-world nzb offsets to global nzb offsets
            J_cts_nzb_start = bsm_cts.nzb_start.numpy()
            J_dofs_nzb_start = bsm_dofs.nzb_start.numpy()
            for _j in range(model.size.sum_of_num_joints):
                w = joint_wid[_j]
                J_cts_joint_nzb_offsets[_j] += J_cts_nzb_start[w]
                J_dofs_joint_nzb_offsets[_j] += J_dofs_nzb_start[w]

            # Create/move precomputed helper arrays to device
            self._J_cts_joint_nzb_offsets = wp.array(J_cts_joint_nzb_offsets, dtype=int32, device=device)
            self._J_cts_limit_nzb_offsets = wp.zeros(shape=(model.size.sum_of_max_limits,), dtype=int32, device=device)
            self._J_cts_contact_nzb_offsets = wp.zeros(
                shape=(model.size.sum_of_max_contacts,), dtype=int32, device=device
            )
            self._J_dofs_joint_nzb_offsets = wp.array(J_dofs_joint_nzb_offsets, dtype=int32, device=device)
            self._J_cts_num_joint_nzb = wp.array(J_cts_nnzb_min, dtype=int32, device=device)

    def build(
        self,
        model: Model,
        data: ModelData,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        reset_to_zero: bool = True,
    ):
        """
        Builds the system DoF and constraint Jacobians for the given
        data of the provided model, data, limits and contacts containers.
        """
        # Ensure the Jacobians have been finalized
        if self._J_cts is None or self._J_dofs is None:
            raise RuntimeError("SparseSystemJacobians.build() called before finalize().")

        jacobian_cts = self._J_cts.bsm
        jacobian_dofs = self._J_dofs.bsm

        # Optionally reset the Jacobian array data to zero
        if reset_to_zero:
            jacobian_cts.zero()
            jacobian_dofs.zero()

        # Compute active rows of constraints Jacobian
        # TODO: Compute num_nzb and offsets for limit and contact entries to avoid atomic_add in those kernels
        wp.launch(
            _configure_jacobians_sparse,
            dim=model.size.num_worlds,
            inputs=[
                # Inputs:
                model.info.num_joint_cts,
                data.info.num_limits,
                data.info.num_contacts,
                # Outputs:
                jacobian_cts.num_rows,
            ],
        )

        # Build the joint constraints and actuation Jacobians
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                _build_joint_jacobians_sparse,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
                    model.joints.dof_type,
                    model.joints.bid_B,
                    model.joints.bid_F,
                    model.joints.X_j,
                    data.joints.p_j,
                    data.bodies.q_i,
                    self._J_cts_joint_nzb_offsets,
                    self._J_dofs_joint_nzb_offsets,
                    # Outputs:
                    jacobian_cts.nzb_values,
                    jacobian_dofs.nzb_values,
                ],
            )

            # Initialize the number of NZB with the number of NZB for all joints
            wp.copy(jacobian_cts.num_nzb, self._J_cts_num_joint_nzb)

        # Build the limit constraints Jacobians if a limits data container is provided
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                _build_limit_jacobians_sparse,
                dim=limits.model_max_limits_host,
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    model.info.joints_offset,
                    model.joints.dofs_offset,
                    model.joints.num_dofs,
                    data.info.limit_cts_group_offset,
                    limits.model_active_limits,
                    limits.model_max_limits_host,
                    limits.wid,
                    limits.jid,
                    limits.lid,
                    limits.bids,
                    limits.dof,
                    limits.side,
                    self._J_dofs_joint_nzb_offsets,
                    jacobian_dofs.nzb_values,
                    jacobian_cts.nzb_start,
                    # Outputs:
                    jacobian_cts.num_nzb,
                    jacobian_cts.nzb_coords,
                    jacobian_cts.nzb_values,
                    self._J_cts_limit_nzb_offsets,
                ],
            )

        # Build the contact constraints Jacobians if a contacts data container is provided
        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                _build_contact_jacobians_sparse,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    data.info.contact_cts_group_offset,
                    data.bodies.q_i,
                    contacts.model_active_contacts,
                    contacts.model_max_contacts_host,
                    contacts.wid,
                    contacts.cid,
                    contacts.bid_AB,
                    contacts.position_A,
                    contacts.position_B,
                    contacts.frame,
                    jacobian_cts.nzb_start,
                    # Outputs:
                    jacobian_cts.num_nzb,
                    jacobian_cts.nzb_coords,
                    jacobian_cts.nzb_values,
                    self._J_cts_contact_nzb_offsets,
                ],
            )


class ColMajorSparseConstraintJacobians(BlockSparseLinearOperators):
    """
    Container to hold a column-major version of the constraint Jacobian that uses 6x1 blocks instead
    of the regular 1x6 blocks.

    Note:
        This version of the Jacobian is more efficient when computing the product of the transpose
        Jacobian with a vector.

        If a Jacobian matrix has a maximum number of rows of fewer than six, this Jacobian variant
        might lead to issues due to potential memory access outside of the allocated arrays. Avoid
        using this Jacobian variant for such cases.
    """

    def __init__(
        self,
        model: Model | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        jacobians: SparseSystemJacobians | None = None,
        device: Devicelike | None = None,
    ):
        """
        Constructs a column-major sparse constraint Jacobian.

        Args:
            model (Model, optional): The model containing the system's kinematic structure. If provided,
                the Jacobian will be immediately finalized with the given model.
            limits (Limits, optional): Limits data container for joint limit constraints.
            contacts (Contacts, optional): Contacts data container for contact constraints.
            jacobians (SparseSystemJacobians, optional): Row-major sparse Jacobians. If provided
                along with model, the column-major Jacobian will be immediately updated with values
                from the provided Jacobians.
            device (Devicelike, optional): The device on which to allocate memory for the Jacobian
                data structures.
        """
        super().__init__()

        self._joint_nzb_offsets: wp.array | None = None
        self._num_joint_nzb: wp.array | None = None

        if model is not None:
            self.finalize(model=model, limits=limits, contacts=contacts, jacobians=jacobians, device=device)

    def finalize(
        self,
        model: Model,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        jacobians: SparseSystemJacobians | None = None,
        device: Devicelike | None = None,
    ):
        """
        Initializes the data structure of the column-major constraint Jacobian.

        Args:
            model (Model): The model containing the system's kinematic structure.
            limits (Limits, optional): Limits data container for joint limit constraints. Needs to
                be provided if the regular Jacobian also has limit constraints.
            contacts (Contacts, optional): Contacts data container for contact constraints. Needs to
                be provided if the regular Jacobian also has contact constraints.
            jacobians (SparseSystemJacobians, optional): Row-major sparse Jacobians. If provided,
                the column-major Jacobian will be immediately updated with values from the provided
                Jacobians after allocation.
            device (Devicelike, optional): The device on which to allocate memory for the Jacobian
                data structures.
        """
        # Extract the constraint and DoF sizes of each world
        num_worlds = model.info.num_worlds
        num_body_dofs = [model.worlds[w].num_body_dofs for w in range(num_worlds)]
        num_joint_cts = [model.worlds[w].num_joint_cts for w in range(num_worlds)]
        max_num_limits = (
            limits.world_max_limits_host if limits and limits.model_max_limits_host > 0 else [0] * num_worlds
        )
        max_num_contacts = (
            contacts.world_max_contacts_host if contacts and contacts.model_max_contacts_host > 0 else [0] * num_worlds
        )
        max_num_constraints = [
            num_joint_cts[w] + max_num_limits[w] + 3 * max_num_contacts[w] for w in range(num_worlds)
        ]

        # Compute the number of non-zero blocks required for Jacobian matrix, using 6 1x6 blocks per
        # body per joint/limit/contact
        joint_wid = model.joints.wid.numpy()
        joint_bid_B = model.joints.bid_B.numpy()
        joint_num_dofs = model.joints.num_dofs.numpy()
        joint_q_j_min = model.joints.q_j_min.numpy()
        joint_q_j_max = model.joints.q_j_max.numpy()
        J_cts_cm_nnzb_min = [0] * num_worlds
        J_cts_cm_nnzb_max = [0] * num_worlds
        J_cts_cm_joint_nzb_offsets = [0] * model.size.sum_of_num_joints
        dofs_start = 0
        # Add non-zero blocks for joints and joint limits
        for _j in range(model.size.sum_of_num_joints):
            w = joint_wid[_j]
            J_cts_cm_joint_nzb_offsets[_j] = J_cts_cm_nnzb_min[w]
            J_cts_cm_nnzb_min[w] += 12 if joint_bid_B[_j] > -1 else 6
            J_cts_cm_nnzb_max[w] += 12 if joint_bid_B[_j] > -1 else 6
            if max_num_limits[w] > 0:
                for d_j in range(joint_num_dofs[_j]):
                    if joint_q_j_min[dofs_start + d_j] > float(FLOAT32_MIN) or joint_q_j_max[dofs_start + d_j] < float(
                        FLOAT32_MAX
                    ):
                        J_cts_cm_nnzb_max[w] += 12 if joint_bid_B[_j] > -1 else 6
            dofs_start += joint_num_dofs[_j]
        # Add non-zero blocks for contacts
        # TODO: Use the candidate geom-pair info to compute maximum possible contact constraint blocks more accurately
        if contacts is not None and contacts.model_max_contacts_host > 0:
            for w in range(num_worlds):
                J_cts_cm_nnzb_max[w] += 12 * max_num_contacts[w]

        # Compute the sizes of the Jacobian matrix data for each world
        J_cts_cm_dims_max = [(max_num_constraints[i], num_body_dofs[i]) for i in range(num_worlds)]

        # Allocate the block-sparse linear-operator data to represent each system Jacobian
        with wp.ScopedDevice(device):
            # Allocate the column-major constraint Jacobian
            self.bsm = BlockSparseMatrices(
                num_matrices=num_worlds,
                nzb_dtype=BlockDType(dtype=float32, shape=(6, 1)),
                device=device,
            )
            self.bsm.finalize(max_dims=J_cts_cm_dims_max, capacities=J_cts_cm_nnzb_max)

            self._joint_nzb_offsets = wp.array(J_cts_cm_joint_nzb_offsets, dtype=int32, device=device)
            self._num_joint_nzb = wp.array(J_cts_cm_nnzb_min, dtype=int32, device=device)

        if jacobians is not None:
            self.update(jacobians, model, limits, contacts)

    def update(
        self,
        jacobians: SparseSystemJacobians,
        model: Model,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
    ):
        """
        Fills the column-major constraint Jacobian with the values of an already assembled row-major
        Jacobian.

        Args:
            jacobians (SparseSystemJacobians): The row-major sparse system Jacobians containing the
                constraint Jacobians.
            model (Model): The model containing the system's kinematic structure.
            limits (Limits, optional): Limits data container for joint limit constraints. Needs to
                be provided if the regular Jacobian also has limit constraints.
            contacts (Contacts, optional): Contacts data container for contact constraints. Needs to
                 be provided if the regular Jacobian also has contact constraints.

        Note:
            The finalize() method must be called before update() to allocate the necessary data structures.
            The dimensions of the column-major Jacobian will be set to match the input row-major Jacobian.
        """
        J_cts = jacobians._J_cts.bsm

        # Set dimensions from input Jacobian
        self.bsm.dims.assign(J_cts.dims)

        # Update the joint constraints Jacobians
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                kernel=_update_col_major_joint_jacobians,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
                    model.joints.wid,
                    model.joints.num_cts,
                    model.joints.bid_B,
                    jacobians._J_cts_joint_nzb_offsets,
                    J_cts.nzb_coords,
                    J_cts.nzb_values,
                    self.bsm.nzb_start,
                    self._joint_nzb_offsets,
                    # Outputs:
                    self.bsm.nzb_coords,
                    self.bsm.nzb_values,
                ],
            )

        # Initialize the number of NZB with the number of NZB for all joints
        wp.copy(self.bsm.num_nzb, self._num_joint_nzb)

        # Update the limit constraints Jacobians if a limits data container is provided
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                _update_col_major_limit_jacobians,
                dim=limits.model_max_limits_host,
                inputs=[
                    # Inputs:
                    limits.model_active_limits,
                    limits.model_max_limits_host,
                    limits.wid,
                    limits.bids,
                    jacobians._J_cts_limit_nzb_offsets,
                    J_cts.nzb_coords,
                    J_cts.nzb_values,
                    self.bsm.nzb_start,
                    # Outputs:
                    self.bsm.num_nzb,
                    self.bsm.nzb_coords,
                    self.bsm.nzb_values,
                ],
            )

        # Build the contact constraints Jacobians if a contacts data container is provided
        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                _update_col_major_contact_jacobians,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    # Inputs:
                    contacts.model_active_contacts,
                    contacts.model_max_contacts_host,
                    contacts.wid,
                    contacts.bid_AB,
                    jacobians._J_cts_contact_nzb_offsets,
                    J_cts.nzb_coords,
                    J_cts.nzb_values,
                    self.bsm.nzb_start,
                    # Outputs:
                    self.bsm.num_nzb,
                    self.bsm.nzb_coords,
                    self.bsm.nzb_values,
                ],
            )
