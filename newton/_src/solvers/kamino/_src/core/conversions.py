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

"""Provides a set of conversion utilities to bridge Kamino and Newton."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .....geometry import GeoType, ShapeFlags
from .....sim import JointTargetMode, JointType
from .....sim.model import Model
from ..core.bodies import RigidBodiesModel
from ..core.size import SizeKamino
from ..utils import logger as msg
from .builder import JointActuationType
from .geometry import GeometriesModel
from .joints import JOINT_DQMAX, JOINT_QMAX, JOINT_QMIN, JOINT_TAUMAX, JointDoFType, JointsModel
from .materials import MaterialDescriptor, MaterialManager
from .shapes import ShapeType, max_contacts_for_shape_pair
from .types import float32, int32, mat33f, transformf, vec2i, vec3f, vec4f, vec6f

if TYPE_CHECKING:
    from ..core.model import ModelKaminoInfo

###
# Module interface
###

__all__ = [
    "compute_required_contact_capacity",
    "convert_entity_local_transforms",
    "convert_geometries",
    "convert_joints",
    "convert_model_joint_transforms",
    "convert_rigid_bodies",
]

###
# Functions
###


def _to_wpq(q):
    return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def convert_entity_local_transforms(model: Model) -> dict[str, np.ndarray]:
    """
    Converts all entity-local transforms (i.e. of bodies, joints and shapes) in the
    given  Newton model to a format that is compatible with Kamino's constraint system.

    This involves absorbing any non-identity :attr:`Model.joint_X_c`
    rotations into the child body frames, and updating all downstream
    :attr:`Model.joint_X_p` and :attr:`Model.shape_transform` accordingly.

    Args:
        model (Model): Newton model to be modified in-place.
    """
    # ---------------------------------------------------------------------------
    # Pre-processing: absorb non-identity joint_X_c rotations into child body
    # frames so that Kamino sees aligned joint frames on both sides.
    #
    # Kamino's constraint system assumes a single joint frame X_j valid for both
    # the base (parent) and follower (child) bodies.  At q = 0 it requires
    #   q_base^{-1} * q_follower = identity
    #
    # Newton, however, allows different parent / child joint-frame orientations
    # via joint_X_p and joint_X_c.  At q = 0 Newton's FK gives:
    #   q_follower = q_parent * q_pj * inv(q_cj)
    # so q_base^{-1} * q_follower = q_pj * inv(q_cj) which is generally not
    # identity.
    #
    # To fix this we apply a per-body correction rotation q_corr = q_cj * inv(q_pj)
    # (applied on the right) to each child body's frame:
    #   q_body_new = q_body_old * q_corr
    #
    # This makes q_base^{-1} * q_follower_new = identity at q = 0, and the joint
    # rotation axis R(q_pj) * axis is preserved.
    #
    # All body-local quantities (CoM, inertia, shapes) are re-expressed in the
    # rotated frame, and downstream joint_X_p transforms are updated to account
    # for the parent body's frame change.
    # ---------------------------------------------------------------------------

    def _from_wpq(q):
        return np.array([q[0], q[1], q[2], q[3]], dtype=np.float64)

    def _quat_is_identity(q, tol=1e-5):
        return abs(abs(q[3]) - 1.0) < tol

    def _quat_mul(a, b):
        return _from_wpq(_to_wpq(a) * _to_wpq(b))

    def _quat_inv(q):
        return _from_wpq(wp.quat_inverse(_to_wpq(q)))

    def _quat_rotate_vec(q, v):
        r = wp.quat_rotate(_to_wpq(q), wp.vec3(float(v[0]), float(v[1]), float(v[2])))
        return np.array([r[0], r[1], r[2]], dtype=np.float64)

    def _quat_to_mat33(q):
        return np.array(wp.quat_to_matrix(_to_wpq(q)), dtype=np.float64).reshape(3, 3)

    # Work on copies so the original Newton model is not mutated
    body_com_np: np.ndarray = model.body_com.numpy().copy()
    body_q_np: np.ndarray = model.body_q.numpy().copy()
    body_qd_np: np.ndarray = model.body_qd.numpy().copy()
    body_inertia_np: np.ndarray = model.body_inertia.numpy().copy()
    body_inv_inertia_np: np.ndarray = model.body_inv_inertia.numpy().copy()
    joint_parent_np: np.ndarray = model.joint_parent.numpy().copy()
    joint_child_np: np.ndarray = model.joint_child.numpy().copy()
    joint_X_p_np: np.ndarray = model.joint_X_p.numpy().copy()
    joint_X_c_np: np.ndarray = model.joint_X_c.numpy().copy()
    shape_transform_np: np.ndarray = model.shape_transform.numpy().copy()
    shape_body_np: np.ndarray = model.shape_body.numpy().copy()

    # Process joints in tree order (Newton stores them parent-before-child).
    # For each joint whose q_pj * inv(q_cj) is not identity, we apply a
    # correction q_corr to the child body's frame and immediately propagate
    # to all downstream joints that reference the corrected body as parent.
    body_corr: dict[int, np.ndarray] = {}  # body_index -> cumulative q_corr

    for j in range(model.joint_count):
        parent = int(joint_parent_np[j])
        child = int(joint_child_np[j])

        # If the parent body was previously corrected, first update this
        # joint's parent-side transform to the new parent frame.
        if parent >= 0 and parent in body_corr:
            q_par_corr_inv = _quat_inv(body_corr[parent])
            p_pos = joint_X_p_np[j, :3].astype(np.float64)
            joint_X_p_np[j, :3] = _quat_rotate_vec(q_par_corr_inv, p_pos)
            p_quat = joint_X_p_np[j, 3:7].astype(np.float64)
            joint_X_p_np[j, 3:7] = _quat_mul(q_par_corr_inv, p_quat)

        # Now compute the correction for this joint's child body
        q_cj = joint_X_c_np[j, 3:7].astype(np.float64)
        q_pj = joint_X_p_np[j, 3:7].astype(np.float64)
        q_corr = _quat_mul(q_cj, _quat_inv(q_pj))

        if child < 0 or _quat_is_identity(q_corr):
            continue

        if child in body_corr:
            msg.warning(
                "Body %d is the child of multiple joints requiring joint_X_c "
                "correction. The previous correction will be overwritten, which "
                "may produce incorrect joint constraints for loop-closing joints.",
                child,
            )
        body_corr[child] = q_corr.copy()

        # Update child-side joint transform: rotation becomes identity,
        # position re-expressed in the new child frame
        q_corr_inv = _quat_inv(q_corr)
        c_pos = joint_X_c_np[j, :3].astype(np.float64)
        joint_X_c_np[j, :3] = _quat_rotate_vec(q_corr_inv, c_pos)
        joint_X_c_np[j, 3:7] = [0.0, 0.0, 0.0, 1.0]

        # Rotate the child body's local quantities
        R_inv_corr = _quat_to_mat33(q_corr_inv)

        q_old = body_q_np[child, 3:7].astype(np.float64)
        body_q_np[child, 3:7] = _quat_mul(q_old, q_corr)

        body_com_np[child] = _quat_rotate_vec(q_corr_inv, body_com_np[child].astype(np.float64))

        body_inertia_np[child] = R_inv_corr @ body_inertia_np[child].astype(np.float64) @ R_inv_corr.T
        body_inv_inertia_np[child] = R_inv_corr @ body_inv_inertia_np[child].astype(np.float64) @ R_inv_corr.T

        # TODO: Do these need be converted? Aren't they already computed at body CoM?
        body_qd_np[child, :3] = R_inv_corr @ body_qd_np[child, :3].astype(np.float64)
        body_qd_np[child, 3:6] = R_inv_corr @ body_qd_np[child, 3:6].astype(np.float64)

        for s in range(model.shape_count):
            if int(shape_body_np[s]) != child:
                continue
            s_pos = shape_transform_np[s, :3].astype(np.float64)
            s_quat = shape_transform_np[s, 3:7].astype(np.float64)
            shape_transform_np[s, :3] = _quat_rotate_vec(q_corr_inv, s_pos)
            shape_transform_np[s, 3:7] = _quat_mul(q_corr_inv, s_quat)

    if body_corr:
        msg.debug(
            "Absorbed joint_X_c rotations for %d child bodies: %s",
            len(body_corr),
            list(body_corr.keys()),
        )

    # Return the converted transforms as numpy arrays
    # to be used for constructing the Kamino model
    return {
        "body_q": body_q_np,
        "body_qd": body_qd_np,
        "body_com": body_com_np,
        "body_inertia": body_inertia_np,
        "body_inv_inertia": body_inv_inertia_np,
        "shape_transform": shape_transform_np,
        "joint_X_p": joint_X_p_np,
        "joint_X_c": joint_X_c_np,
    }


def compute_required_contact_capacity(
    model: Model,
    max_contacts_per_pair: int | None = None,
    max_contacts_per_world: int | None = None,
) -> tuple[int, list[int]]:
    """
    Computes the required contact capacity for a given Newton model.

    The outputs are used to determine the minimum number of contacts
    to be allocated, according to the shapes present in the model.

    Args:
        model (Model):
            The Newton model for which to compute the required contact capacity.
        max_contacts_per_pair (int, optional):
            Optional maximum number of contacts to allocate per shape pair.
            If `None`, no per-pair limit is applied.
        max_contacts_per_world (int, optional):
            Optional maximum number of contacts to allocate per world.
            If `None`, no per-world limit is applied, otherwise it will
            override the computed per-world requirements if it is larger.

    Returns:
        (model_required_contacts, world_required_contacts):
            A tuple containing:
            - `model_required_contacts` (int):
                The total number of contacts required for the entire model.
            - `world_required_contacts` (list[int]):
                A list of required contacts per world, where the length of the
                list is equal to `model.world_count` and each entry corresponds
                to the required contacts for that world.

    """
    # Retrieve the shape types, worlds and collision candidate pairs from the model
    shape_type_np = model.shape_type.numpy()
    shape_world_np = model.shape_world.numpy()
    shape_contact_pairs = model.shape_contact_pairs.numpy().tolist()

    # First check if there are any collision geometries
    if model.shape_count == 0:
        return 0, [0] * model.world_count

    # Compute the maximum possible number of geom pairs per world.
    # Global shapes (world=-1, e.g. ground plane) can collide with shapes
    # from any world — attribute their contacts to the non-global shape's world.
    world_max_contacts = [0] * model.world_count
    for shape_pair in shape_contact_pairs:
        s1 = int(shape_pair[0])
        s2 = int(shape_pair[1])
        type1, _ = ShapeType.from_newton(shape_type_np[s1])
        type2, _ = ShapeType.from_newton(shape_type_np[s2])
        if type1 > type2:
            s1, s2 = s2, s1
            type1, type2 = type2, type1
        num_contacts_a, num_contacts_b = max_contacts_for_shape_pair(
            type_a=type1,
            type_b=type2,
        )
        num_contacts = num_contacts_a + num_contacts_b
        # Determine the world for this pair — fall back to other shape if one is global
        w1 = int(shape_world_np[s1])
        w2 = int(shape_world_np[s2])
        wid = w1 if w1 >= 0 else w2
        if wid < 0:
            continue  # Both shapes are global — skip
        if max_contacts_per_pair is not None:
            world_max_contacts[wid] += min(num_contacts, max_contacts_per_pair)
        else:
            world_max_contacts[wid] += num_contacts

    # Override the per-world maximum contacts if specified in the settings
    if max_contacts_per_world is not None:
        for w in range(model.world_count):
            world_max_contacts[w] = min(world_max_contacts[w], max_contacts_per_world)

    # Return the per-world maximum contacts list
    return sum(world_max_contacts), world_max_contacts


# TODO: Re-implement this using a kernel to run in parallel on the GPU if possible
def convert_model_joint_transforms(model: Model, joints: JointsModel) -> None:
    """
    Converts the joint model parameterization of Newton's to Kamino's format.

    This essentially involves computing the B_r_Bj, F_r_Fj and
    X_j arrays from the joint_X_p and joint_X_c transforms.

    Args:
    - model (Model):
        The input Newton model containing the joint information to be converted.
    - joints (JointsModel):
        The output JointsModel instance where the converted joint data will be stored.
        This function modifies the `joints` object in-place.
    """
    joint_X_p_np = model.joint_X_p.numpy()
    joint_X_c_np = model.joint_X_c.numpy()
    body_com_np = model.body_com.numpy()
    joint_parent_np = model.joint_parent.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_axis_np = model.joint_axis.numpy()
    joint_dof_dim_np = model.joint_dof_dim.numpy()
    joint_qd_start_np = model.joint_qd_start.numpy()
    joint_limit_lower_np = model.joint_limit_lower.numpy()
    joint_limit_upper_np = model.joint_limit_upper.numpy()
    dof_type_np = joints.dof_type.numpy()

    n_joints = model.joint_count
    B_r_Bj_np = np.zeros((n_joints, 3), dtype=np.float32)
    F_r_Fj_np = np.zeros((n_joints, 3), dtype=np.float32)
    X_j_np = np.zeros((n_joints, 9), dtype=np.float32)

    for j in range(n_joints):
        dof_type_j = JointDoFType(int(dof_type_np[j]))
        dof_dim_j = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
        dofs_start_j = int(joint_qd_start_np[j])
        ndofs_j = dof_type_j.num_dofs
        joint_axes_j = joint_axis_np[dofs_start_j : dofs_start_j + ndofs_j]
        joint_q_min_j = joint_limit_lower_np[dofs_start_j : dofs_start_j + ndofs_j]
        joint_q_max_j = joint_limit_upper_np[dofs_start_j : dofs_start_j + ndofs_j]
        R_axis_j = JointDoFType.from_newton(dof_type_j, dof_dim_j, joint_axes_j, joint_q_min_j, joint_q_max_j)

        parent_bid = int(joint_parent_np[j])
        p_r_p_com = wp.vec3f(body_com_np[parent_bid]) if parent_bid >= 0 else wp.vec3f(0.0)
        c_r_c_com = wp.vec3f(body_com_np[int(joint_child_np[j])])

        X_p_j = wp.transformf(*joint_X_p_np[j, :])
        X_c_j = wp.transformf(*joint_X_c_np[j, :])
        q_p_j = wp.transform_get_rotation(X_p_j)
        p_r_p_j = wp.transform_get_translation(X_p_j)
        c_r_c_j = wp.transform_get_translation(X_c_j)

        B_r_Bj_np[j, :] = p_r_p_j - p_r_p_com
        F_r_Fj_np[j, :] = c_r_c_j - c_r_c_com
        X_j_np[j, :] = wp.quat_to_matrix(q_p_j) @ R_axis_j

    joints.B_r_Bj.assign(B_r_Bj_np)
    joints.F_r_Fj.assign(F_r_Fj_np)
    joints.X_j.assign(X_j_np.reshape((n_joints, 3, 3)))


def _compute_entity_indices_wrt_world(entity_world: wp.array) -> np.ndarray:
    wid_np = entity_world.numpy()
    eid_np = np.zeros_like(wid_np)
    for e in range(wid_np.size):
        eid_np[e] = np.sum(wid_np[:e] == wid_np[e])
    return eid_np


def _compute_num_entities_per_world(entity_world: wp.array, num_worlds: int) -> np.ndarray:
    wid_np = entity_world.numpy()
    counts = np.zeros(num_worlds, dtype=int)
    for w in range(num_worlds):
        counts[w] = np.sum(wid_np == w)
    return counts


def convert_rigid_bodies(
    model: Model,
    model_size: SizeKamino,
    model_info: ModelKaminoInfo,
    body_com: np.ndarray,
    body_q: np.ndarray,
    body_qd: np.ndarray,
    body_inertia: np.ndarray,
    body_inv_inertia: np.ndarray,
) -> RigidBodiesModel:
    """
    Converts the rigid bodies from a Newton model into Kamino's format. The function
    will create a new `RigidBodiesModel` object and fill in the rigid body and shape
    entries of the provided `SizeKamino` and `ModelKaminoInfo` objects.

    This function requires that the preprocessing that absorbs non-identity joint
    rotations into child body frames has already been computed, and the conversion
    result is passed in as arguments.

    Args:
        model: Newton model.
        model_size: Model size object, to be filled in by the function.
        model_info: Model info object, to be filled in by the function.
        body_com: Preprocessed rigid body center of mass positions.
        body_q: Preprocessed initial rigid body poses.
        body_qd: Preprocessed initial rigid body velocities.
        body_inertia: Preprocessed rigid body inertias.
        body_inv_inertia: Preprocessed inverse rigid body inertias.

    Returns:
        Fully converted rigid bodies model in Kamino's format.
    """

    # Compute the entity indices of each body w.r.t the corresponding world
    body_bid_np = _compute_entity_indices_wrt_world(model.body_world)

    # Compute the number of entities per world
    num_bodies_np = _compute_num_entities_per_world(model.body_world, model.world_count)
    num_shapes_np = _compute_num_entities_per_world(model.shape_world, model.world_count)

    # Compute body coord/DoF counts per world
    num_body_dofs_np = num_bodies_np * 6

    # Compute offsets per world
    world_shape_offset_np = np.zeros((model.world_count,), dtype=int)
    world_body_offset_np = np.zeros((model.world_count,), dtype=int)
    world_body_dof_offset_np = np.zeros((model.world_count,), dtype=int)

    for w in range(1, model.world_count):
        world_shape_offset_np[w] = world_shape_offset_np[w - 1] + num_shapes_np[w - 1]
        world_body_offset_np[w] = world_body_offset_np[w - 1] + num_bodies_np[w - 1]
        world_body_dof_offset_np[w] = world_body_dof_offset_np[w - 1] + num_body_dofs_np[w - 1]

    # Construct per-world inertial summaries
    mass_min_np = np.zeros((model.world_count,), dtype=float)
    mass_max_np = np.zeros((model.world_count,), dtype=float)
    mass_total_np = np.zeros((model.world_count,), dtype=float)
    inertia_total_np = np.zeros((model.world_count,), dtype=float)
    body_world_np = model.body_world.numpy()
    body_mass_np = model.body_mass.numpy()
    for w in range(model.world_count):
        masses_w = []
        for b in range(model.body_count):
            if body_world_np[b] == w:
                mass_b = body_mass_np[b]
                masses_w.append(mass_b)
                mass_total_np[w] += mass_b
                inertia_total_np[w] += 3.0 * mass_b + body_inertia[b].diagonal().sum()
        mass_min_np[w] = min(masses_w)
        mass_max_np[w] = max(masses_w)

    # model.body_q stores body-origin world poses, but Kamino expects
    # COM world poses (joint attachment vectors are COM-relative).
    q_i_0_np = np.empty((model.body_count, 7), dtype=np.float32)
    for i in range(model.body_count):
        pos = body_q[i, :3]
        rot = wp.quatf(*body_q[i, 3:7])
        com_world = pos + np.array(wp.quat_rotate(rot, wp.vec3f(*body_com[i])))
        q_i_0_np[i, :3] = com_world
        q_i_0_np[i, 3:7] = body_q[i, 3:7]

    # Construct SizeKamino from the newton.Model instance
    model_size.sum_of_num_bodies = int(num_bodies_np.sum())
    model_size.max_of_num_bodies = int(num_bodies_np.max())
    model_size.sum_of_num_geoms = int(num_shapes_np.sum())
    model_size.max_of_num_geoms = int(num_shapes_np.max())
    model_size.sum_of_num_body_dofs = int(num_body_dofs_np.sum())
    model_size.max_of_num_body_dofs = int(num_body_dofs_np.max())

    # Per-world heterogeneous model info
    model_info.num_bodies = wp.array(num_bodies_np, dtype=int32)
    model_info.num_geoms = wp.array(num_shapes_np, dtype=int32)
    model_info.num_body_dofs = wp.array(num_body_dofs_np, dtype=int32)
    model_info.bodies_offset = wp.array(world_body_offset_np, dtype=int32)
    model_info.geoms_offset = wp.array(world_shape_offset_np, dtype=int32)
    model_info.body_dofs_offset = wp.array(world_body_dof_offset_np, dtype=int32)
    model_info.mass_min = wp.array(mass_min_np, dtype=float32)
    model_info.mass_max = wp.array(mass_max_np, dtype=float32)
    model_info.mass_total = wp.array(mass_total_np, dtype=float32)
    model_info.inertia_total = wp.array(inertia_total_np, dtype=float32)

    model_bodies = RigidBodiesModel(
        num_bodies=model.body_count,
        label=model.body_label,
        wid=model.body_world,
        bid=wp.array(body_bid_np, dtype=int32),  # TODO: Remove
        m_i=model.body_mass,
        inv_m_i=model.body_inv_mass,
        i_r_com_i=wp.array(body_com, dtype=vec3f),
        i_I_i=wp.array(body_inertia, dtype=mat33f),
        inv_i_I_i=wp.array(body_inv_inertia, dtype=mat33f),
        q_i_0=wp.array(q_i_0_np, dtype=wp.transformf),
        u_i_0=wp.array(body_qd, dtype=vec6f),
    )
    return model_bodies


def convert_joints(
    model: Model,
    model_size: SizeKamino,
    model_info: ModelKaminoInfo,
    body_com: np.ndarray,
    joint_X_p: np.ndarray,
    joint_X_c: np.ndarray,
) -> JointsModel:
    """
    Converts the joints from a Newton model into Kamino's format. The function will
    create a new `JointsModel` object and fill in the joint entries of the provided
    `SizeKamino` and `ModelKaminoInfo` objects.

    This function requires that the preprocessing that absorbs non-identity joint
    rotations into child body frames has already been computed, and the conversion
    result is passed in as arguments.

    Args:
        model: Newton model.
        model_size: Model size object, to be filled in by the function.
        model_info: Model info object, to be filled in by the function.
        body_com: Preprocessed rigid body center of mass positions.
        joint_X_p: Preprocessed joint frames in parent frame.
        joint_X_c: Preprocessed joint frames in child frame.

    Returns:
        Fully converted joints model in Kamino's format.
    """

    # Compute the entity indices of each body w.r.t the corresponding world
    joint_jid_np = _compute_entity_indices_wrt_world(model.joint_world)

    # Compute the number of entities per world
    num_joints_np = _compute_num_entities_per_world(model.joint_world, model.world_count)

    # Compute joint coord/DoF/constraint counts per world
    num_passive_joints_np = np.zeros((model.world_count,), dtype=int)
    num_actuated_joints_np = np.zeros((model.world_count,), dtype=int)
    num_dynamic_joints_np = np.zeros((model.world_count,), dtype=int)
    num_joint_coords_np = np.zeros((model.world_count,), dtype=int)
    num_joint_dofs_np = np.zeros((model.world_count,), dtype=int)
    num_joint_passive_coords_np = np.zeros((model.world_count,), dtype=int)
    num_joint_passive_dofs_np = np.zeros((model.world_count,), dtype=int)
    num_joint_actuated_coords_np = np.zeros((model.world_count,), dtype=int)
    num_joint_actuated_dofs_np = np.zeros((model.world_count,), dtype=int)
    num_joint_cts_np = np.zeros((model.world_count,), dtype=int)
    num_joint_dynamic_cts_np = np.zeros((model.world_count,), dtype=int)
    num_joint_kinematic_cts_np = np.zeros((model.world_count,), dtype=int)

    # TODO
    joint_dof_type_np = np.zeros((model.joint_count,), dtype=int)
    joint_act_type_np = np.zeros((model.joint_count,), dtype=int)
    joint_B_r_Bj_np = np.zeros((model.joint_count, 3), dtype=float)
    joint_F_r_Fj_np = np.zeros((model.joint_count, 3), dtype=float)
    joint_X_j_np = np.zeros((model.joint_count, 9), dtype=float)
    joint_num_coords_np = np.zeros((model.joint_count,), dtype=int)
    joint_num_dofs_np = np.zeros((model.joint_count,), dtype=int)
    joint_num_cts_np = np.zeros((model.joint_count,), dtype=int)
    joint_num_dynamic_cts_np = np.zeros((model.joint_count,), dtype=int)
    joint_num_kinematic_cts_np = np.zeros((model.joint_count,), dtype=int)
    joint_coord_start_np = np.zeros((model.joint_count,), dtype=int)
    joint_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
    joint_actuated_coord_start_np = np.zeros((model.joint_count,), dtype=int)
    joint_actuated_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
    joint_passive_coord_start_np = np.zeros((model.joint_count,), dtype=int)
    joint_passive_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
    joint_cts_start_np = np.zeros((model.joint_count,), dtype=int)
    joint_dynamic_cts_start_np = np.zeros((model.joint_count,), dtype=int)
    joint_kinematic_cts_start_np = np.zeros((model.joint_count,), dtype=int)

    # TODO
    joint_wid_np: np.ndarray = model.joint_world.numpy().copy()
    joint_type_np: np.ndarray = model.joint_type.numpy().copy()
    joint_target_mode_np: np.ndarray = model.joint_target_mode.numpy().copy()
    joint_parent_np: np.ndarray = model.joint_parent.numpy().copy()
    joint_child_np: np.ndarray = model.joint_child.numpy().copy()
    joint_axis_np: np.ndarray = model.joint_axis.numpy().copy()
    joint_dof_dim_np: np.ndarray = model.joint_dof_dim.numpy().copy()
    joint_q_start_np: np.ndarray = model.joint_q_start.numpy().copy()
    joint_qd_start_np: np.ndarray = model.joint_qd_start.numpy().copy()
    joint_limit_lower_np: np.ndarray = model.joint_limit_lower.numpy().copy()
    joint_limit_upper_np: np.ndarray = model.joint_limit_upper.numpy().copy()
    joint_velocity_limit_np = model.joint_velocity_limit.numpy().copy()
    joint_effort_limit_np = model.joint_effort_limit.numpy().copy()
    joint_armature_np: np.ndarray = model.joint_armature.numpy().copy()
    joint_friction_np: np.ndarray = model.joint_friction.numpy().copy()
    joint_target_ke_np: np.ndarray = model.joint_target_ke.numpy().copy()
    joint_target_kd_np: np.ndarray = model.joint_target_kd.numpy().copy()

    for j in range(model.joint_count):
        # TODO
        wid_j = joint_wid_np[j]

        # TODO
        joint_coord_start_np[j] = num_joint_coords_np[wid_j]
        joint_dofs_start_np[j] = num_joint_dofs_np[wid_j]
        joint_actuated_coord_start_np[j] = num_joint_actuated_coords_np[wid_j]
        joint_actuated_dofs_start_np[j] = num_joint_actuated_dofs_np[wid_j]
        joint_passive_coord_start_np[j] = num_joint_passive_coords_np[wid_j]
        joint_passive_dofs_start_np[j] = num_joint_passive_dofs_np[wid_j]
        joint_cts_start_np[j] = num_joint_cts_np[wid_j]
        joint_dynamic_cts_start_np[j] = num_joint_dynamic_cts_np[wid_j]
        joint_kinematic_cts_start_np[j] = num_joint_kinematic_cts_np[wid_j]

        # TODO
        type_j = int(joint_type_np[j])
        dof_dim_j = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
        q_count_j = int(joint_q_start_np[j + 1] - joint_q_start_np[j])
        qd_count_j = int(joint_qd_start_np[j + 1] - joint_qd_start_np[j])
        limit_upper_j = joint_limit_upper_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]].astype(float)
        limit_lower_j = joint_limit_lower_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]].astype(float)
        dof_type_j = JointDoFType.from_newton(
            JointType(type_j), q_count_j, qd_count_j, dof_dim_j, limit_lower_j, limit_upper_j
        )

        # TODO
        ncoords_j = dof_type_j.num_coords
        ndofs_j = dof_type_j.num_dofs
        ncts_j = dof_type_j.num_cts

        # TODO
        joint_dof_type_np[j] = dof_type_j.value
        num_joint_coords_np[wid_j] += ncoords_j
        num_joint_dofs_np[wid_j] += ndofs_j
        joint_num_coords_np[j] = ncoords_j
        joint_num_dofs_np[j] = ndofs_j

        # TODO
        dofs_start_j = joint_qd_start_np[j]
        dof_axes_j = joint_axis_np[dofs_start_j : dofs_start_j + ndofs_j]
        joint_dofs_target_mode_j = joint_target_mode_np[dofs_start_j : dofs_start_j + ndofs_j]
        act_type_j = JointActuationType.from_newton(
            JointTargetMode(max(joint_dofs_target_mode_j) if len(joint_dofs_target_mode_j) > 0 else 0)
        )
        joint_act_type_np[j] = act_type_j.value

        # Infer if the joint requires dynamic constraints
        is_dynamic_j = False
        if ndofs_j > 0:
            a_j = joint_armature_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
            b_j = joint_friction_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
            ke_j = joint_target_ke_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
            kd_j = joint_target_kd_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
            a_j_min = float(a_j.min())
            b_j_min = float(b_j.min())
            ke_j_min = float(ke_j.min())
            kd_j_min = float(kd_j.min())
            a_j_max = float(a_j.max())
            b_j_max = float(b_j.max())
            ke_j_max = float(ke_j.max())
            kd_j_max = float(kd_j.max())
            if (a_j_min < 0.0) or (b_j_min < 0.0) or (ke_j_min < 0.0) or (kd_j_min < 0.0):
                raise ValueError(
                    f"Joint {j} in world {wid_j} has negative armature, friction "
                    "or target stiffness/damping values, which is not supported."
                )
            if (a_j_min < a_j_max) or (b_j_min < b_j_max) or (ke_j_min < ke_j_max) or (kd_j_min < kd_j_max):
                raise ValueError(
                    f"Joint {j} in world {wid_j} has non-constant armature, friction "
                    "or target stiffness/damping values, which is not supported."
                )
            is_dynamic_j = (a_j_max > 0.0) or (b_j_max > 0.0) or (ke_j_max > 0.0) or (kd_j_max > 0.0)

        # TODO
        if is_dynamic_j:
            joint_num_dynamic_cts_np[j] = ndofs_j
            joint_dynamic_cts_start_np[j] = num_joint_dynamic_cts_np[wid_j]
            num_joint_dynamic_cts_np[wid_j] += ndofs_j
            num_joint_cts_np[wid_j] += ndofs_j
            num_dynamic_joints_np[wid_j] += 1
        else:
            joint_dynamic_cts_start_np[j] = -1

        # TODO
        num_joint_cts_np[wid_j] += ncts_j
        num_joint_kinematic_cts_np[wid_j] += ncts_j
        if act_type_j > JointActuationType.PASSIVE:
            num_actuated_joints_np[wid_j] += 1
            num_joint_actuated_coords_np[wid_j] += ncoords_j
            num_joint_actuated_dofs_np[wid_j] += ndofs_j
            joint_passive_coord_start_np[j] = -1
            joint_passive_dofs_start_np[j] = -1
        else:
            num_passive_joints_np[wid_j] += 1
            num_joint_passive_coords_np[wid_j] += ncoords_j
            num_joint_passive_dofs_np[wid_j] += ndofs_j
            joint_actuated_coord_start_np[j] = -1
            joint_actuated_dofs_start_np[j] = -1
        joint_num_kinematic_cts_np[j] = ncts_j
        joint_num_cts_np[j] = joint_num_dynamic_cts_np[j] + joint_num_kinematic_cts_np[j]

        # TODO
        parent_bid = joint_parent_np[j]
        p_r_p_com = vec3f(body_com[parent_bid]) if parent_bid >= 0 else vec3f(0.0, 0.0, 0.0)
        c_r_c_com = vec3f(body_com[joint_child_np[j]])
        X_p_j = transformf(*joint_X_p[j, :])
        X_c_j = transformf(*joint_X_c[j, :])
        q_p_j = wp.transform_get_rotation(X_p_j)
        p_r_p_j = wp.transform_get_translation(X_p_j)
        c_r_c_j = wp.transform_get_translation(X_c_j)

        # TODO
        R_axis_j = JointDoFType.axes_matrix_from_joint_type(dof_type_j, dof_dim_j, dof_axes_j)
        B_r_Bj = p_r_p_j - p_r_p_com
        F_r_Fj = c_r_c_j - c_r_c_com
        X_j = wp.quat_to_matrix(q_p_j) @ R_axis_j
        joint_B_r_Bj_np[j, :] = B_r_Bj
        joint_F_r_Fj_np[j, :] = F_r_Fj
        joint_X_j_np[j, :] = X_j

    # Convert joint limits and effort/velocity limits to np.float32 and clip to supported ranges
    np.clip(a=joint_limit_lower_np, a_min=JOINT_QMIN, a_max=JOINT_QMAX, out=joint_limit_lower_np)
    np.clip(a=joint_limit_upper_np, a_min=JOINT_QMIN, a_max=JOINT_QMAX, out=joint_limit_upper_np)
    np.clip(a=joint_velocity_limit_np, a_min=-JOINT_DQMAX, a_max=JOINT_DQMAX, out=joint_velocity_limit_np)
    np.clip(a=joint_effort_limit_np, a_min=-JOINT_TAUMAX, a_max=JOINT_TAUMAX, out=joint_effort_limit_np)

    # Compute offsets per world
    world_joint_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_dof_offset_np = np.zeros((model.world_count,), dtype=int)
    world_actuated_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_actuated_joint_dofs_offset_np = np.zeros((model.world_count,), dtype=int)
    world_passive_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_passive_joint_dofs_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_dynamic_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_kinematic_cts_offset_np = np.zeros((model.world_count,), dtype=int)

    for w in range(1, model.world_count):
        world_joint_offset_np[w] = world_joint_offset_np[w - 1] + num_joints_np[w - 1]
        world_joint_coord_offset_np[w] = world_joint_coord_offset_np[w - 1] + num_joint_coords_np[w - 1]
        world_joint_dof_offset_np[w] = world_joint_dof_offset_np[w - 1] + num_joint_dofs_np[w - 1]
        world_actuated_joint_coord_offset_np[w] = (
            world_actuated_joint_coord_offset_np[w - 1] + num_joint_actuated_coords_np[w - 1]
        )
        world_actuated_joint_dofs_offset_np[w] = (
            world_actuated_joint_dofs_offset_np[w - 1] + num_joint_actuated_dofs_np[w - 1]
        )
        world_passive_joint_coord_offset_np[w] = (
            world_passive_joint_coord_offset_np[w - 1] + num_joint_passive_coords_np[w - 1]
        )
        world_passive_joint_dofs_offset_np[w] = (
            world_passive_joint_dofs_offset_np[w - 1] + num_joint_passive_dofs_np[w - 1]
        )
        world_joint_cts_offset_np[w] = world_joint_cts_offset_np[w - 1] + num_joint_cts_np[w - 1]
        world_joint_dynamic_cts_offset_np[w] = (
            world_joint_dynamic_cts_offset_np[w - 1] + num_joint_dynamic_cts_np[w - 1]
        )
        world_joint_kinematic_cts_offset_np[w] = (
            world_joint_kinematic_cts_offset_np[w - 1] + num_joint_kinematic_cts_np[w - 1]
        )

    # Determine the base body and joint indices per world
    base_body_idx_np = np.full((model.world_count,), -1, dtype=int)
    base_joint_idx_np = np.full((model.world_count,), -1, dtype=int)
    body_world_np = model.body_world.numpy()
    joint_world_np = model.joint_world.numpy()
    body_world_start_np = model.body_world_start.numpy()

    # Check for articulations
    if model.articulation_count > 0:
        msg.warning("Model contains articulations")
        articulation_start_np = model.articulation_start.numpy()
        articulation_world_np = model.articulation_world.numpy()
        # For each articulation, assign its base body and joint to the corresponding world
        # NOTE: We only assign the first articulation found in each world
        for aid in range(model.articulation_count):
            wid = articulation_world_np[aid]
            base_joint = articulation_start_np[aid]
            base_body = joint_child_np[base_joint]
            if base_body_idx_np[wid] == -1 and base_joint_idx_np[wid] == -1:
                base_body_idx_np[wid] = base_body
                base_joint_idx_np[wid] = base_joint

    # Check for root joint (i.e. joint with no parent body (= -1))
    elif model.joint_count > 0:
        # TODO: How to handle no free joint being defined?
        # Create a list of joint indices with parent body == -1 for each world
        world_parent_joints: dict[int, list[int]] = {w: [] for w in range(model.world_count)}
        for j in range(model.joint_count):
            wid_j = joint_world_np[j]
            parent_j = joint_parent_np[j]
            if parent_j == -1:
                world_parent_joints[wid_j].append(j)
        # For each world, assign the base body and joint based on the first joint with parent == -1,
        # If no joint with parent == -1 is found in a world, then assign the first body as base
        # If multiple joints with parent == -1 are found in a world, then assign the first one as the base
        for w in range(model.world_count):
            if len(world_parent_joints[w]) > 0:
                j = world_parent_joints[w][0]
                base_joint_idx_np[w] = j
                base_body_idx_np[w] = int(joint_child_np[j])
            else:
                base_body_idx_np[w] = int(body_world_start_np[w])
                base_joint_idx_np[w] = -1

    # Fall-back: first body and joint in the world
    else:
        for w in range(model.world_count):
            # Base body: first body in the world
            for b in range(model.body_count):
                if body_world_np[b] == w:
                    base_body_idx_np[w] = b
                    break
            # Base joint: first joint in the world
            for j in range(model.joint_count):
                if joint_world_np[j] == w:
                    base_joint_idx_np[w] = j
                    break

    # Ensure that all worlds have a base body assigned
    for w in range(model.world_count):
        if base_body_idx_np[w] == -1:
            raise ValueError(f"World {w} does not have a base body assigned (index is -1).")

    # Update size object
    model_size.sum_of_num_joints = int(num_joints_np.sum())
    model_size.max_of_num_joints = int(num_joints_np.max())
    model_size.sum_of_num_passive_joints = int(num_passive_joints_np.sum())
    model_size.max_of_num_passive_joints = int(num_passive_joints_np.max())
    model_size.sum_of_num_actuated_joints = int(num_actuated_joints_np.sum())
    model_size.max_of_num_actuated_joints = int(num_actuated_joints_np.max())
    model_size.sum_of_num_dynamic_joints = int(num_dynamic_joints_np.sum())
    model_size.max_of_num_dynamic_joints = int(num_dynamic_joints_np.max())
    model_size.sum_of_num_joint_coords = int(num_joint_coords_np.sum())
    model_size.max_of_num_joint_coords = int(num_joint_coords_np.max())
    model_size.sum_of_num_joint_dofs = int(num_joint_dofs_np.sum())
    model_size.max_of_num_joint_dofs = int(num_joint_dofs_np.max())
    model_size.sum_of_num_passive_joint_coords = int(num_joint_passive_coords_np.sum())
    model_size.max_of_num_passive_joint_coords = int(num_joint_passive_coords_np.max())
    model_size.sum_of_num_passive_joint_dofs = int(num_joint_passive_dofs_np.sum())
    model_size.max_of_num_passive_joint_dofs = int(num_joint_passive_dofs_np.max())
    model_size.sum_of_num_actuated_joint_coords = int(num_joint_actuated_coords_np.sum())
    model_size.max_of_num_actuated_joint_coords = int(num_joint_actuated_coords_np.max())
    model_size.sum_of_num_actuated_joint_dofs = int(num_joint_actuated_dofs_np.sum())
    model_size.max_of_num_actuated_joint_dofs = int(num_joint_actuated_dofs_np.max())
    model_size.sum_of_num_joint_cts = int(num_joint_cts_np.sum())
    model_size.max_of_num_joint_cts = int(num_joint_cts_np.max())
    model_size.sum_of_num_dynamic_joint_cts = int(num_joint_dynamic_cts_np.sum())
    model_size.max_of_num_dynamic_joint_cts = int(num_joint_dynamic_cts_np.max())
    model_size.sum_of_num_kinematic_joint_cts = int(num_joint_kinematic_cts_np.sum())
    model_size.max_of_num_kinematic_joint_cts = int(num_joint_kinematic_cts_np.max())
    model_size.sum_of_max_total_cts = int(num_joint_cts_np.sum())
    model_size.max_of_max_total_cts = int(num_joint_cts_np.max())

    # Update per-world heterogeneous model info
    model_info.num_joints = wp.array(num_joints_np, dtype=int32)
    model_info.num_passive_joints = wp.array(num_passive_joints_np, dtype=int32)
    model_info.num_actuated_joints = wp.array(num_actuated_joints_np, dtype=int32)
    model_info.num_dynamic_joints = wp.array(num_dynamic_joints_np, dtype=int32)
    model_info.num_joint_coords = wp.array(num_joint_coords_np, dtype=int32)
    model_info.num_joint_dofs = wp.array(num_joint_dofs_np, dtype=int32)
    model_info.num_passive_joint_coords = wp.array(num_joint_passive_coords_np, dtype=int32)
    model_info.num_passive_joint_dofs = wp.array(num_joint_passive_dofs_np, dtype=int32)
    model_info.num_actuated_joint_coords = wp.array(num_joint_actuated_coords_np, dtype=int32)
    model_info.num_actuated_joint_dofs = wp.array(num_joint_actuated_dofs_np, dtype=int32)
    model_info.num_joint_cts = wp.array(num_joint_cts_np, dtype=int32)
    model_info.num_joint_dynamic_cts = wp.array(num_joint_dynamic_cts_np, dtype=int32)
    model_info.num_joint_kinematic_cts = wp.array(num_joint_kinematic_cts_np, dtype=int32)
    model_info.joints_offset = wp.array(world_joint_offset_np, dtype=int32)
    model_info.joint_coords_offset = wp.array(world_joint_coord_offset_np, dtype=int32)
    model_info.joint_dofs_offset = wp.array(world_joint_dof_offset_np, dtype=int32)
    model_info.joint_passive_coords_offset = wp.array(world_passive_joint_coord_offset_np, dtype=int32)
    model_info.joint_passive_dofs_offset = wp.array(world_passive_joint_dofs_offset_np, dtype=int32)
    model_info.joint_actuated_coords_offset = wp.array(world_actuated_joint_coord_offset_np, dtype=int32)
    model_info.joint_actuated_dofs_offset = wp.array(world_actuated_joint_dofs_offset_np, dtype=int32)
    model_info.joint_cts_offset = wp.array(world_joint_cts_offset_np, dtype=int32)
    model_info.joint_dynamic_cts_offset = wp.array(world_joint_dynamic_cts_offset_np, dtype=int32)
    model_info.joint_kinematic_cts_offset = wp.array(world_joint_kinematic_cts_offset_np, dtype=int32)
    model_info.base_body_index = wp.array(base_body_idx_np, dtype=int32)
    model_info.base_joint_index = wp.array(base_joint_idx_np, dtype=int32)

    # Joints
    model_joints = JointsModel(
        num_joints=model.joint_count,
        label=model.joint_label,
        wid=model.joint_world,
        jid=wp.array(joint_jid_np, dtype=int32),  # TODO: Remove
        dof_type=wp.array(joint_dof_type_np, dtype=int32),
        act_type=wp.array(joint_act_type_np, dtype=int32),
        bid_B=model.joint_parent,
        bid_F=model.joint_child,
        B_r_Bj=wp.array(joint_B_r_Bj_np, dtype=wp.vec3f),
        F_r_Fj=wp.array(joint_F_r_Fj_np, dtype=wp.vec3f),
        X_j=wp.array(joint_X_j_np.reshape((model.joint_count, 3, 3)), dtype=wp.mat33f),
        q_j_min=wp.array(joint_limit_lower_np, dtype=float32),
        q_j_max=wp.array(joint_limit_upper_np, dtype=float32),
        dq_j_max=wp.array(joint_velocity_limit_np, dtype=float32),
        tau_j_max=wp.array(joint_effort_limit_np, dtype=float32),
        a_j=model.joint_armature,
        b_j=model.joint_friction,  # TODO: Is this the right attribute?
        k_p_j=model.joint_target_ke,
        k_d_j=model.joint_target_kd,
        q_j_0=model.joint_q,
        dq_j_0=model.joint_qd,
        num_coords=wp.array(joint_num_coords_np, dtype=int32),
        num_dofs=wp.array(joint_num_dofs_np, dtype=int32),
        num_cts=wp.array(joint_num_cts_np, dtype=int32),
        num_dynamic_cts=wp.array(joint_num_dynamic_cts_np, dtype=int32),
        num_kinematic_cts=wp.array(joint_num_kinematic_cts_np, dtype=int32),
        coords_offset=wp.array(joint_coord_start_np, dtype=int32),
        dofs_offset=wp.array(joint_dofs_start_np, dtype=int32),
        passive_coords_offset=wp.array(joint_passive_coord_start_np, dtype=int32),
        passive_dofs_offset=wp.array(joint_passive_dofs_start_np, dtype=int32),
        actuated_coords_offset=wp.array(joint_actuated_coord_start_np, dtype=int32),
        actuated_dofs_offset=wp.array(joint_actuated_dofs_start_np, dtype=int32),
        cts_offset=wp.array(joint_cts_start_np, dtype=int32),
        dynamic_cts_offset=wp.array(joint_dynamic_cts_start_np, dtype=int32),
        kinematic_cts_offset=wp.array(joint_kinematic_cts_start_np, dtype=int32),
    )
    return model_joints


def _register_materials(model: Model, materials_manager: MaterialManager) -> np.ndarray:
    # Set up material parameter dictionary
    material_param_indices: dict[tuple[float, float], int] = {}
    for material in materials_manager.materials:
        # Adding already existing (default) materials from material manager, making sure the values
        # undergo the same transformation as any material parameters in the Newton model (conversion
        # to np.float32)
        mu = float(np.float32(material.static_friction))
        restitution = float(np.float32(material.restitution))
        material_param_indices[(mu, restitution)] = 0

    # Newton material parameters
    shape_friction = model.shape_material_mu.numpy().tolist()
    shape_restitution = model.shape_material_restitution.numpy().tolist()
    # Mapping from geom to material index
    geom_material = np.zeros((model.shape_count,), dtype=int)
    # TODO: Integrate world index for shape material
    # shape_world_np = model.shape_world.numpy()

    for s in range(model.shape_count):
        # Check if material with these parameters already exists
        material_desc = (shape_friction[s], shape_restitution[s])
        if material_desc in material_param_indices:
            material_id = material_param_indices[material_desc]
        else:
            material = MaterialDescriptor(
                name=f"{model.shape_label[s]}_material",
                restitution=shape_restitution[s],
                static_friction=shape_friction[s],
                dynamic_friction=shape_friction[s],
                # wid=shape_world_np[s],
            )
            material_id = materials_manager.register(material)
            material_param_indices[material_desc] = material_id
        geom_material[s] = material_id

    return geom_material


def convert_geometries(
    model: Model, model_size: KaminoSize, materials_manager: MaterialManager, shape_transform: np.ndarray
) -> GeometriesModel:
    # Compute the entity indices of each body w.r.t the corresponding world
    shape_sid_np = _compute_entity_indices_wrt_world(model.shape_world)

    # Set up materials
    geom_material_np = _register_materials(model, materials_manager)

    # Update size object
    model_size.sum_of_num_materials = materials_manager.num_materials
    model_size.max_of_num_materials = materials_manager.num_materials
    model_size.sum_of_num_material_pairs = materials_manager.num_material_pairs
    model_size.max_of_num_material_pairs = materials_manager.num_material_pairs

    # Convert per-shape properties from Newton to Kamino format
    shape_type_np = model.shape_type.numpy()
    shape_scale_np = model.shape_scale.numpy()
    shape_flags_np = model.shape_flags.numpy()
    geom_shape_collision_group_np = model.shape_collision_group.numpy()
    geom_shape_type_np = np.zeros((model.shape_count,), dtype=int)
    geom_shape_params_np = np.zeros((model.shape_count, 4), dtype=float)
    model_num_collidable_geoms = 0
    for s in range(model.shape_count):
        shape_type, params = ShapeType.from_newton(GeoType(int(shape_type_np[s])), vec3f(*shape_scale_np[s]))
        geom_shape_type_np[s] = shape_type
        geom_shape_params_np[s, :] = params
        if (shape_flags_np[s] & ShapeFlags.COLLIDE_SHAPES) != 0 and geom_shape_collision_group_np[s] > 0:
            model_num_collidable_geoms += 1
        else:
            geom_material_np[s] = -1  # Ensure non-collidable geoms no material assigned

    # Fix plane normals: derive from the shape transform rotation (local Z-axis)
    # instead of the hardcoded default in convert_newton_geo_to_kamino_shape.
    for s in range(model.shape_count):
        if shape_type_np[s] == GeoType.PLANE:
            tf = shape_transform[s, :]
            q_rot = _to_wpq(np.array([tf[3], tf[4], tf[5], tf[6]]))
            normal = wp.quat_rotate(q_rot, vec3f(0.0, 0.0, 1.0))
            geom_shape_params_np[s, 0] = float(normal[0])
            geom_shape_params_np[s, 1] = float(normal[1])
            geom_shape_params_np[s, 2] = float(normal[2])
            geom_shape_params_np[s, 3] = 0.0

    # Compute total number of required contacts per world
    if model.rigid_contact_max > 0:
        model_min_contacts = int(model.rigid_contact_max)
        min_contacts_per_world = model.rigid_contact_max // model.world_count
        world_min_contacts = [min_contacts_per_world] * model.world_count
    else:
        model_min_contacts, world_min_contacts = compute_required_contact_capacity(model)

    # Geometries
    model_geoms = GeometriesModel(
        num_geoms=model.shape_count,
        num_collidable=model_num_collidable_geoms,
        num_collidable_pairs=model.shape_contact_pair_count,
        num_excluded_pairs=len(model.shape_collision_filter_pairs),
        model_minimum_contacts=model_min_contacts,
        world_minimum_contacts=world_min_contacts,
        label=model.shape_label,
        wid=model.shape_world,
        gid=wp.array(shape_sid_np, dtype=int32),  # TODO: Remove
        bid=model.shape_body,
        type=wp.array(geom_shape_type_np, dtype=int32),
        flags=model.shape_flags,
        ptr=model.shape_source_ptr,
        params=wp.array(geom_shape_params_np, dtype=vec4f),
        offset=wp.zeros_like(model.shape_transform),
        material=wp.array(geom_material_np, dtype=int32),
        group=model.shape_collision_group,
        gap=model.shape_gap,
        margin=model.shape_margin,
        collidable_pairs=model.shape_contact_pairs,
        excluded_pairs=wp.array(sorted(model.shape_collision_filter_pairs), dtype=vec2i),
    )

    return model_geoms
