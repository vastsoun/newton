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
        wid: int,
        row_start: int,
        bid_offset: int,
        bid_B: int,
        bid_F: int,
        JT_B_j: mat66f,
        JT_F_j: mat66f,
        J_nzb_offset: int,
        J_nzb_start: wp.array(dtype=int32),
        J_nzb_coords: wp.array2d(dtype=int32),
        J_nzb_values: wp.array(dtype=vec6f),
    ):
        """
        Stores the Jacobian blocks of a joint into the provided block sparse data structure at the specified offset.

        Args:
            wid (int): The world index to which this joint belongs.
            row_start (int): The starting row index in the sparse Jacobian matrix for this joint.
            bid_offset (int): The body index offset of the world's bodies w.r.t the model.
            bid_B (int): The body index of the base body of the joint w.r.t the model.
            bid_F (int): The body index of the follower body of the joint w.r.t the model.
            JT_B_j (mat66f): The 6x6 Jacobian transpose block of the joint's base body.
            JT_F_j (mat66f): The 6x6 Jacobian transpose block of the joint's follower body.
            J_nzb_offset (int): The offset within the world's non-zero blocks for this joint.
            J_nzb_start (wp.array(dtype=int32)): Array containing the start index of non-zero blocks for each world.
            J_nzb_coords (wp.array2d(dtype=int32)): 2D array storing the (row, column) coordinates of each non-zero block.
            J_nzb_values (wp.array(dtype=vec6f)): Array storing the non-zero blocks of the Jacobians.
        """
        # Set the number of rows in the output Jacobian block
        # NOTE: This is evaluated statically at compile time
        num_jac_rows = wp.static(len(axes))

        # Retrieve the start index of the world's Jacobian super-block
        J_j_nzb_start = J_nzb_start[wid]

        # Store the Jacobian block for the follower body
        J_F_col = 6 * (bid_F - bid_offset)
        for i in range(num_jac_rows):
            J_F_i_nzb_start = J_j_nzb_start + J_nzb_offset + i
            J_nzb_coords[J_F_i_nzb_start, 0] = row_start + i
            J_nzb_coords[J_F_i_nzb_start, 1] = J_F_col
            J_nzb_values[J_F_i_nzb_start] = JT_F_j[:, axes[i]]

        # If the base body is not the world (:= -1), store the respective Jacobian block
        if bid_B > -1:
            J_B_col = 6 * (bid_B - bid_offset)
            for i in range(num_jac_rows):
                J_B_i_nzb_start = J_j_nzb_start + J_nzb_offset + num_jac_rows + i
                J_nzb_coords[J_B_i_nzb_start, 0] = row_start + i
                J_nzb_coords[J_B_i_nzb_start, 1] = J_B_col
                J_nzb_values[J_B_i_nzb_start] = JT_B_j[:, axes[i]]

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
    wid: int,
    dof_type: int,
    cts_offset: int,
    bid_offset: int,
    bid_B: int,
    bid_F: int,
    JT_B: mat66f,
    JT_F: mat66f,
    J_cts_nzb_start: wp.array(dtype=int32),
    J_cts_nzb_coords: wp.array2d(dtype=int32),
    J_cts_nzb_values: wp.array(dtype=vec6f),
    J_cts_nzb_offset: int,
):
    """
    Stores the constraints Jacobian block of a joint into the provided flat data array at the given offset.
    """

    if dof_type == JointDoFType.REVOLUTE:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.REVOLUTE.cts_axes))(
            wid,
            cts_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_cts_nzb_offset,
            J_cts_nzb_start,
            J_cts_nzb_coords,
            J_cts_nzb_values,
        )

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.PRISMATIC.cts_axes))(
            wid,
            cts_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_cts_nzb_offset,
            J_cts_nzb_start,
            J_cts_nzb_coords,
            J_cts_nzb_values,
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.CYLINDRICAL.cts_axes))(
            wid,
            cts_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_cts_nzb_offset,
            J_cts_nzb_start,
            J_cts_nzb_coords,
            J_cts_nzb_values,
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.UNIVERSAL.cts_axes))(
            wid,
            cts_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_cts_nzb_offset,
            J_cts_nzb_start,
            J_cts_nzb_coords,
            J_cts_nzb_values,
        )

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.SPHERICAL.cts_axes))(
            wid,
            cts_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_cts_nzb_offset,
            J_cts_nzb_start,
            J_cts_nzb_coords,
            J_cts_nzb_values,
        )

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.GIMBAL.cts_axes))(
            wid,
            cts_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_cts_nzb_offset,
            J_cts_nzb_start,
            J_cts_nzb_coords,
            J_cts_nzb_values,
        )

    elif dof_type == JointDoFType.CARTESIAN:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.CARTESIAN.cts_axes))(
            wid,
            cts_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_cts_nzb_offset,
            J_cts_nzb_start,
            J_cts_nzb_coords,
            J_cts_nzb_values,
        )

    elif dof_type == JointDoFType.FIXED:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.FIXED.cts_axes))(
            wid,
            cts_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_cts_nzb_offset,
            J_cts_nzb_start,
            J_cts_nzb_coords,
            J_cts_nzb_values,
        )


@wp.func
def store_joint_dofs_jacobian_sparse(
    wid: int,
    dof_type: int,
    dofs_offset: int,
    bid_offset: int,
    bid_B: int,
    bid_F: int,
    JT_B: mat66f,
    JT_F: mat66f,
    J_dofs_nzb_start: wp.array(dtype=int32),
    J_dofs_nzb_coords: wp.array2d(dtype=int32),
    J_dofs_nzb_values: wp.array(dtype=vec6f),
    J_dofs_nzb_offset: int,
):
    """
    Stores the DoFs Jacobian block of a joint into the provided flat data array at the given offset.
    """

    if dof_type == JointDoFType.REVOLUTE:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.REVOLUTE.dofs_axes))(
            wid,
            dofs_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_dofs_nzb_offset,
            J_dofs_nzb_start,
            J_dofs_nzb_coords,
            J_dofs_nzb_values,
        )

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.PRISMATIC.dofs_axes))(
            wid,
            dofs_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_dofs_nzb_offset,
            J_dofs_nzb_start,
            J_dofs_nzb_coords,
            J_dofs_nzb_values,
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.CYLINDRICAL.dofs_axes))(
            wid,
            dofs_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_dofs_nzb_offset,
            J_dofs_nzb_start,
            J_dofs_nzb_coords,
            J_dofs_nzb_values,
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.UNIVERSAL.dofs_axes))(
            wid,
            dofs_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_dofs_nzb_offset,
            J_dofs_nzb_start,
            J_dofs_nzb_coords,
            J_dofs_nzb_values,
        )

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.SPHERICAL.dofs_axes))(
            wid,
            dofs_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_dofs_nzb_offset,
            J_dofs_nzb_start,
            J_dofs_nzb_coords,
            J_dofs_nzb_values,
        )

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.GIMBAL.dofs_axes))(
            wid,
            dofs_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_dofs_nzb_offset,
            J_dofs_nzb_start,
            J_dofs_nzb_coords,
            J_dofs_nzb_values,
        )

    elif dof_type == JointDoFType.CARTESIAN:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.CARTESIAN.dofs_axes))(
            wid,
            dofs_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_dofs_nzb_offset,
            J_dofs_nzb_start,
            J_dofs_nzb_coords,
            J_dofs_nzb_values,
        )

    elif dof_type == JointDoFType.FREE:
        wp.static(make_store_joint_jacobian_sparse_func(JointDoFType.FREE.dofs_axes))(
            wid,
            dofs_offset,
            bid_offset,
            bid_B,
            bid_F,
            JT_B,
            JT_F,
            J_dofs_nzb_offset,
            J_dofs_nzb_start,
            J_dofs_nzb_coords,
            J_dofs_nzb_values,
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
    model_num_body_dofs: wp.array(dtype=int32),
    model_num_joint_cts: wp.array(dtype=int32),
    model_num_joint_dofs: wp.array(dtype=int32),
    num_limits: wp.array(dtype=int32),
    num_contacts: wp.array(dtype=int32),
    # Output:
    jac_cts_dims: wp.array2d(dtype=int32),
    jac_dofs_dims: wp.array2d(dtype=int32),
):
    world_id = wp.tid()

    jac_cts_dims[world_id, 0] = model_num_joint_cts[world_id] + num_limits[world_id] + 3 * num_contacts[world_id]
    jac_cts_dims[world_id, 1] = model_num_body_dofs[world_id]

    jac_dofs_dims[world_id, 0] = model_num_joint_dofs[world_id]
    jac_dofs_dims[world_id, 1] = model_num_body_dofs[world_id]


@wp.kernel
def _build_joint_jacobians_sparse(
    # Inputs
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
    jacobian_cts_nzb_start: wp.array(dtype=int32),
    jacobian_cts_nzb_offsets: wp.array(dtype=int32),
    jacobian_dofs_nzb_start: wp.array(dtype=int32),
    jacobian_dofs_nzb_offsets: wp.array(dtype=int32),
    # Outputs
    jacobian_cts_nzb_coords: wp.array2d(dtype=int32),
    jacobian_cts_nzb_values: wp.array(dtype=vec6f),
    jacobian_dofs_nzb_coords: wp.array2d(dtype=int32),
    jacobian_dofs_nzb_values: wp.array(dtype=vec6f),
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
    bio = model_info_bodies_offset[wid]

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
        wid,
        dof_type,
        cio,
        bio,
        bid_B,
        bid_F,
        JT_B_j,
        JT_F_j,
        jacobian_cts_nzb_start,
        jacobian_cts_nzb_coords,
        jacobian_cts_nzb_values,
        jacobian_cts_nzb_offsets[jid],
    )

    # Store the actuation Jacobian block if the joint is actuated
    store_joint_dofs_jacobian_sparse(
        wid,
        dof_type,
        dio,
        bio,
        bid_B,
        bid_F,
        JT_B_j,
        JT_F_j,
        jacobian_dofs_nzb_start,
        jacobian_dofs_nzb_coords,
        jacobian_dofs_nzb_values,
        jacobian_dofs_nzb_offsets[jid],
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
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    state_info_limit_cts_group_offset: wp.array(dtype=int32),
    limits_model_num: wp.array(dtype=int32),
    limits_model_max: int32,
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    limits_dof: wp.array(dtype=int32),
    limits_side: wp.array(dtype=float32),
    jacobian_dofs_nzb_start: wp.array(dtype=int32),
    jacobian_dofs_dof_nzb_offsets: wp.array(dtype=int32),
    jacobian_dofs_nzb_values: wp.array(dtype=vec6f),
    jacobian_cts_nzb_start: wp.array(dtype=int32),
    # Outputs:
    jacobian_cts_num_nzb: wp.array(dtype=int32),
    jacobian_cts_nzb_coords: wp.array2d(dtype=int32),
    jacobian_cts_nzb_values: wp.array(dtype=vec6f),
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

    # Compute the index of the blocks we need to pull from the dofs Jacobian to reuse for the limits.
    # We first compute the global dof index of the dof of the limit, then query the NZB offsets to
    # get us the (local) NZB offset for the specific dof (in the dof Jacobian) and add it to the
    # NZB offset of the world to get the global NZB offset.
    global_dof_l = model_info_joint_dofs_offset[world_id] + dof_l
    jac_dofs_nzb_idx = jacobian_dofs_nzb_start[world_id] + jacobian_dofs_dof_nzb_offsets[global_dof_l]

    # Retrieve the relevant model info of the world
    body_index_offset = model_info_bodies_offset[world_id]
    limit_cts_offset = state_info_limit_cts_group_offset[world_id]

    # Set the constraint Jacobian block for the follower body from the actuation Jacobian block
    num_limit_nzb = 2 if body_id_B_l > -1 else 1
    jac_cts_nzb_offset_world = wp.atomic_add(jacobian_cts_num_nzb, world_id, num_limit_nzb)
    jac_cts_nzb_idx = jacobian_cts_nzb_start[world_id] + jac_cts_nzb_offset_world
    jacobian_cts_nzb_values[jac_cts_nzb_idx] = side_l * jacobian_dofs_nzb_values[jac_dofs_nzb_idx]
    jacobian_cts_nzb_coords[jac_cts_nzb_idx, 0] = limit_cts_offset + limit_id_l
    jacobian_cts_nzb_coords[jac_cts_nzb_idx, 1] = 6 * (body_id_F_l - body_index_offset)

    # If not the world body, set the constraint Jacobian block for the base body from the actuation Jacobian block
    if body_id_B_l > -1:
        jacobian_cts_nzb_values[jac_cts_nzb_idx + 1] = side_l * jacobian_dofs_nzb_values[jac_dofs_nzb_idx + 1]
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
        # Local (in-world) offsets for the non-zero blocks of the constraint and dofs Jacobian for each (global) joint
        self._J_cts_joint_nzb_offsets: wp.array | None = None
        self._J_dofs_joint_nzb_offsets: wp.array | None = None
        # Local (in-world) offsets for the non-zero blocks of the dofs Jacobian for each (global) joint dof
        self._J_dofs_dof_nzb_offsets: wp.array | None = None
        # Lists of number of non-zero blocks in each world connected to joint constraints and dofs
        self._J_cts_num_joint_nzb: wp.array | None = None
        self._J_dofs_num_joint_nzb: wp.array | None = None

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
        # per-joint and per-dof offsets.
        joint_wid = model.joints.wid.numpy()
        joint_bid_B = model.joints.bid_B.numpy()
        joint_num_cts = model.joints.num_cts.numpy()
        joint_num_dofs = model.joints.num_dofs.numpy()
        joint_q_j_min = model.joints.q_j_min.numpy()
        joint_q_j_max = model.joints.q_j_max.numpy()
        J_cts_nnzb_min = [0] * num_worlds
        J_cts_nnzb_max = [0] * num_worlds
        J_dofs_nnzb = [0] * num_worlds
        J_cts_joint_nzb_offsets = [0] * model.size.sum_of_num_joints
        J_dofs_joint_nzb_offsets = [0] * model.size.sum_of_num_joints
        J_dofs_dof_nzb_offsets = [0] * model.size.sum_of_num_joint_dofs
        dofs_start = 0
        # Add non-zero blocks for joints and joint limits
        for _j in range(model.size.sum_of_num_joints):
            w = joint_wid[_j]
            J_cts_joint_nzb_offsets[_j] = J_cts_nnzb_min[w]
            J_dofs_joint_nzb_offsets[_j] = J_dofs_nnzb[w]
            for d_j in range(joint_num_dofs[_j]):
                J_dofs_dof_nzb_offsets[dofs_start + d_j] = J_dofs_nnzb[w] + d_j * (2 if joint_bid_B[_j] > -1 else 1)
            J_cts_nnzb_min[w] += 2 * int(joint_num_cts[_j]) if joint_bid_B[_j] > -1 else int(joint_num_cts[_j])
            J_cts_nnzb_max[w] += 2 * int(joint_num_cts[_j]) if joint_bid_B[_j] > -1 else int(joint_num_cts[_j])
            J_dofs_nnzb[w] += 2 * int(joint_num_dofs[_j]) if joint_bid_B[_j] > -1 else int(joint_num_dofs[_j])
            if max_num_limits[w] > 0:
                for d_j in range(joint_num_dofs[_j]):
                    if joint_q_j_min[dofs_start + d_j] > float(FLOAT32_MIN) or joint_q_j_max[dofs_start + d_j] < float(
                        FLOAT32_MAX
                    ):
                        J_cts_nnzb_max[w] += 2 if joint_bid_B[_j] > -1 else 1
            dofs_start += joint_num_dofs[_j]
        # Add non-zero blocks for contacts
        # TODO: Use the candidate geom-pair info to compute maximum possible contact constraint blocks more accurately
        if contacts is not None and contacts.model_max_contacts_host > 0:
            for w in range(num_worlds):
                J_cts_nnzb_max[w] += 2 * 3 * max_num_contacts[w]

        # Compute the sizes of the Jacobian matrix data for each world
        J_cts_dims_max = [(max_num_constraints[i], num_body_dofs[i]) for i in range(num_worlds)]
        J_dofs_dims = [(num_joint_dofs[i], num_body_dofs[i]) for i in range(num_worlds)]

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

            self._J_cts_joint_nzb_offsets = wp.array(J_cts_joint_nzb_offsets, dtype=int32, device=device)
            self._J_dofs_joint_nzb_offsets = wp.array(J_dofs_joint_nzb_offsets, dtype=int32, device=device)
            self._J_dofs_dof_nzb_offsets = wp.array(J_dofs_dof_nzb_offsets, dtype=int32, device=device)

            self._J_cts_num_joint_nzb = wp.array(J_cts_nnzb_min, dtype=int32, device=device)
            self._J_dofs_num_joint_nzb = wp.array(J_dofs_nnzb, dtype=int32, device=device)

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

        jacobian_cts.clear()
        jacobian_dofs.clear()

        # Compute active BSM dimensions
        # TODO: Compute num_nzb and offsets for limit and contact entries to avoid atomic_add in those kernels
        wp.launch(
            _configure_jacobians_sparse,
            dim=model.size.num_worlds,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.num_joint_cts,
                model.info.num_joint_dofs,
                data.info.num_limits,
                data.info.num_contacts,
                # Outputs:
                jacobian_cts.dims,
                jacobian_dofs.dims,
            ],
        )

        # Build the joint constraints and actuation Jacobians
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                _build_joint_jacobians_sparse,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
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
                    jacobian_cts.nzb_start,
                    self._J_cts_joint_nzb_offsets,
                    jacobian_dofs.nzb_start,
                    self._J_dofs_joint_nzb_offsets,
                    # Outputs:
                    jacobian_cts.nzb_coords,
                    jacobian_cts.nzb_values,
                    jacobian_dofs.nzb_coords,
                    jacobian_dofs.nzb_values,
                ],
            )

            # Initialize the number of NZB with the number of NZB for all joints
            wp.copy(jacobian_cts.num_nzb, self._J_cts_num_joint_nzb)
            wp.copy(jacobian_dofs.num_nzb, self._J_dofs_num_joint_nzb)

        # Build the limit constraints Jacobians if a limits data container is provided
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                _build_limit_jacobians_sparse,
                dim=limits.model_max_limits_host,
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    model.info.joint_dofs_offset,
                    data.info.limit_cts_group_offset,
                    limits.model_active_limits,
                    limits.model_max_limits_host,
                    limits.wid,
                    limits.lid,
                    limits.bids,
                    limits.dof,
                    limits.side,
                    jacobian_dofs.nzb_start,
                    self._J_dofs_dof_nzb_offsets,
                    jacobian_dofs.nzb_values,
                    jacobian_cts.nzb_start,
                    # Outputs:
                    jacobian_cts.num_nzb,
                    jacobian_cts.nzb_coords,
                    jacobian_cts.nzb_values,
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
                ],
            )
