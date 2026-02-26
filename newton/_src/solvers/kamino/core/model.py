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

"""Defines the model container of Kamino."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from ....geometry.flags import ShapeFlags
from ....geometry.types import GeoType
from ....sim.joints import JointType
from ....sim.model import Model
from ..utils import logger as msg
from .bodies import RigidBodiesData, RigidBodiesModel
from .control import ControlKamino
from .data import DataKamino, DataKaminoInfo
from .geometry import GeometriesData, GeometriesModel
from .gravity import GravityModel
from .joints import (
    JOINT_DQMAX,
    JOINT_QMAX,
    JOINT_QMIN,
    JOINT_TAUMAX,
    JointActuationType,
    JointsData,
    JointsModel,
    axes_matrix_from_joint_type,
    newton_to_kamino_joint_actuation_type,
    newton_to_kamino_joint_dof_type,
)
from .materials import MaterialManager, MaterialPairsModel, MaterialsModel
from .shapes import convert_newton_geo_to_kamino_shape
from .state import StateKamino
from .time import TimeData, TimeModel
from .types import float32, int32, mat33f, transformf, vec4f, vec6f
from .world import WorldDescriptor

###
# Module interface
###

__all__ = [
    "ModelKamino",
    "ModelKaminoInfo",
    "ModelKaminoSize",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


@dataclass
class ModelKaminoSize:
    """
    A container to hold the summary size of memory allocations and thread dimensions.

    Notes:
    - The sums are used for memory allocations.
    - The maximums are used to define 2D thread shapes: (num_worlds, max_of_max_XXX)
    - Where `XXX` is the maximum number of limits, contacts, unilaterals, or constraints in any world.
    """

    num_worlds: int = 0
    """The number of worlds represented in the model."""

    sum_of_num_bodies: int = 0
    """The total number of bodies in the model across all worlds."""

    max_of_num_bodies: int = 0
    """The maximum number of bodies in any world."""

    sum_of_num_joints: int = 0
    """The total number of joints in the model across all worlds."""

    max_of_num_joints: int = 0
    """The maximum number of joints in any world."""

    sum_of_num_passive_joints: int = 0
    """The total number of passive joints in the model across all worlds."""

    max_of_num_passive_joints: int = 0
    """The maximum number of passive joints in any world."""

    sum_of_num_actuated_joints: int = 0
    """The total number of actuated joints in the model across all worlds."""

    max_of_num_actuated_joints: int = 0
    """The maximum number of actuated joints in any world."""

    sum_of_num_geoms: int = 0
    """The total number of geometries in the model across all worlds."""

    max_of_num_geoms: int = 0
    """The maximum number of geometries in any world."""

    sum_of_num_materials: int = 0
    """
    The total number of materials in the model across all worlds.

    In the present implementation, this will be equal to `max_of_num_materials`,
    since model materials are defined globally for all worlds. We plan to also
    introduce per-world materials in the future.
    """

    max_of_num_materials: int = 0
    """
    The maximum number of materials in any world.

    In the present implementation, this will be equal to `sum_of_num_materials`,
    since model materials are defined globally for all worlds. We plan to also
    introduce per-world materials in the future.
    """

    sum_of_num_material_pairs: int = 0
    """The total number of material pairs in the model across all worlds."""

    max_of_num_material_pairs: int = 0
    """The maximum number of material pairs in any world."""

    sum_of_num_body_dofs: int = 0
    """The total number of body DoFs in the model across all worlds."""

    max_of_num_body_dofs: int = 0
    """The maximum number of body DoFs in any world."""

    sum_of_num_joint_coords: int = 0
    """The total number of joint coordinates in the model across all worlds."""

    max_of_num_joint_coords: int = 0
    """The maximum number of joint coordinates in any world."""

    sum_of_num_joint_dofs: int = 0
    """The total number of joint DoFs in the model across all worlds."""

    max_of_num_joint_dofs: int = 0
    """The maximum number of joint DoFs in any world."""

    sum_of_num_passive_joint_coords: int = 0
    """The total number of passive joint coordinates in the model across all worlds."""

    max_of_num_passive_joint_coords: int = 0
    """The maximum number of passive joint coordinates in any world."""

    sum_of_num_passive_joint_dofs: int = 0
    """The total number of passive joint DoFs in the model across all worlds."""

    max_of_num_passive_joint_dofs: int = 0
    """The maximum number of passive joint DoFs in any world."""

    sum_of_num_actuated_joint_coords: int = 0
    """The total number of actuated joint coordinates in the model across all worlds."""

    max_of_num_actuated_joint_coords: int = 0
    """The maximum number of actuated joint coordinates in any world."""

    sum_of_num_actuated_joint_dofs: int = 0
    """The total number of actuated joint DoFs in the model across all worlds."""

    max_of_num_actuated_joint_dofs: int = 0
    """The maximum number of actuated joint DoFs in any world."""

    sum_of_num_joint_cts: int = 0
    """The total number of joint constraints in the model across all worlds."""

    max_of_num_joint_cts: int = 0
    """The maximum number of joint constraints in any world."""

    sum_of_max_limits: int = 0
    """The total maximum number of limits allocated for the model across all worlds."""

    max_of_max_limits: int = 0
    """The maximum number of active limits of any world."""

    sum_of_max_contacts: int = 0
    """The total maximum number of contacts allocated for the model across all worlds."""

    max_of_max_contacts: int = 0
    """The maximum number of active contacts of any world."""

    sum_of_max_unilaterals: int = 0
    """The maximum number of active unilateral entities, i.e. joint-limits and contacts."""

    max_of_max_unilaterals: int = 0
    """The maximum number of active unilaterals of any world."""

    sum_of_max_total_cts: int = 0
    """The maximum number of active constraints."""

    max_of_max_total_cts: int = 0
    """The maximum number of active constraints of any world."""

    def __repr__(self):
        """Returns a human-readable string representation of the ModelKaminoSize as a formatted table."""
        # List of (row title, sum attr, max attr)
        rows = [
            ("num_bodies", "sum_of_num_bodies", "max_of_num_bodies"),
            ("num_joints", "sum_of_num_joints", "max_of_num_joints"),
            ("num_passive_joints", "sum_of_num_passive_joints", "max_of_num_passive_joints"),
            ("num_actuated_joints", "sum_of_num_actuated_joints", "max_of_num_actuated_joints"),
            ("num_geoms", "sum_of_num_geoms", "max_of_num_geoms"),
            ("num_material_pairs", "sum_of_num_material_pairs", "max_of_num_material_pairs"),
            ("num_body_dofs", "sum_of_num_body_dofs", "max_of_num_body_dofs"),
            ("num_joint_coords", "sum_of_num_joint_coords", "max_of_num_joint_coords"),
            ("num_joint_dofs", "sum_of_num_joint_dofs", "max_of_num_joint_dofs"),
            ("num_passive_joint_coords", "sum_of_num_passive_joint_coords", "max_of_num_passive_joint_coords"),
            ("num_passive_joint_dofs", "sum_of_num_passive_joint_dofs", "max_of_num_passive_joint_dofs"),
            ("num_actuated_joint_coords", "sum_of_num_actuated_joint_coords", "max_of_num_actuated_joint_coords"),
            ("num_actuated_joint_dofs", "sum_of_num_actuated_joint_dofs", "max_of_num_actuated_joint_dofs"),
            ("num_joint_cts", "sum_of_num_joint_cts", "max_of_num_joint_cts"),
            ("max_limits", "sum_of_max_limits", "max_of_max_limits"),
            ("max_contacts", "sum_of_max_contacts", "max_of_max_contacts"),
            ("max_unilaterals", "sum_of_max_unilaterals", "max_of_max_unilaterals"),
            ("max_total_cts", "sum_of_max_total_cts", "max_of_max_total_cts"),
        ]

        # Compute column widths
        name_width = max(len("Name"), *(len(r[0]) for r in rows))
        sum_width = max(len("Sum"), *(len(str(getattr(self, r[1]))) for r in rows))
        max_width = max(len("Max"), *(len(str(getattr(self, r[2]))) for r in rows))

        # Write ModelKaminoSize members as a formatted table
        lines = []
        lines.append("-" * (name_width + 1 + sum_width + 1 + max_width))
        lines.append(f"{'Name':<{name_width}} {'Sum':>{sum_width}} {'Max':>{max_width}}")
        lines.append("-" * (name_width + 1 + sum_width + 1 + max_width))
        for name, sum_attr, max_attr in rows:
            sum_val = getattr(self, sum_attr)
            max_val = getattr(self, max_attr)
            line = f"{name:<{name_width}} {sum_val:>{sum_width}} {max_val:>{max_width}}"
            lines.append(line)
            lines.append("-" * (name_width + 1 + sum_width + 1 + max_width))

        # Join the lines into a single string
        return "\n".join(lines)


# TODO: Rename to also include `World` since it actually holds per-world model info.
@dataclass
class ModelKaminoInfo:
    """
    A container to hold the time-invariant information and meta-data of a model.
    """

    ###
    # Host-side Summary Counts
    ###

    num_worlds: int = 0
    """The number of worlds represented in the model."""

    ###
    # Entity Counts
    ###

    num_bodies: wp.array | None = None
    """
    The number of bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joints: wp.array | None = None
    """
    The number of joints in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_passive_joints: wp.array | None = None
    """
    The number of passive joints in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_actuated_joints: wp.array | None = None
    """
    The number of actuated joints in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_geoms: wp.array | None = None
    """
    The number of geometries in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_limits: wp.array | None = None
    """
    The maximum number of limits in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_contacts: wp.array | None = None
    """
    The maximum number of contacts in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # DoF Counts
    ###

    num_body_dofs: wp.array | None = None
    """
    The number of body DoFs of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joint_coords: wp.array | None = None
    """
    The number of joint coordinates of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joint_dofs: wp.array | None = None
    """
    The number of joint DoFs of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_passive_joint_coords: wp.array | None = None
    """
    The number of passive joint coordinates of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_passive_joint_dofs: wp.array | None = None
    """
    The number of passive joint DoFs of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_actuated_joint_coords: wp.array | None = None
    """
    The number of actuated joint coordinates of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_actuated_joint_dofs: wp.array | None = None
    """
    The number of actuated joint DoFs of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Constraint Counts
    ###

    num_joint_cts: wp.array | None = None
    """
    The number of joint constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_limit_cts: wp.array | None = None
    """
    The maximum number of active limit constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_contact_cts: wp.array | None = None
    """
    The maximum number of active contact constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_total_cts: wp.array | None = None
    """
    The maximum total number of active constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Entity Offsets
    ###

    bodies_offset: wp.array | None = None
    """
    The body index offset of each world w.r.t the model.\n
    Used to index into world-specific blocks of per-body entity arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joints_offset: wp.array | None = None
    """
    The joint index offset of each world w.r.t the model.\n
    Used to index into world-specific blocks of per-joint entity arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    limits_offset: wp.array | None = None
    """
    The limit index offset of each world w.r.t the model.\n
    Used to index into world-specific blocks of per-limit entity arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    contacts_offset: wp.array | None = None
    """
    The contact index offset of each world w.r.t the model.\n
    Used to index into world-specific blocks of per-contact entity arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    unilaterals_offset: wp.array | None = None
    """
    The index offset of the unilaterals (limits + contacts) block of each world.\n
    Used to index into world-specific blocks of per-unilateral entity arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # DoF Offsets
    ###

    # TODO: Remove
    body_dofs_offset: wp.array | None = None
    """
    The index offset of the body DoF block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_coords_offset: wp.array | None = None
    """
    The index offset of the joint coordinates block of each world.\n
    Used to index into world-specific blocks of per-joint-coordinate arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_dofs_offset: wp.array | None = None
    """
    The index offset of the joint DoF block of each world.\n
    Used to index into world-specific blocks of per-joint-DoF arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_passive_coords_offset: wp.array | None = None
    """
    The index offset of the passive joint coordinates block of each world.\n
    Used to index into world-specific blocks of per-passive-joint-coordinate arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_passive_dofs_offset: wp.array | None = None
    """
    The index offset of the passive joint DoF block of each world.\n
    Used to index into world-specific blocks of per-passive-joint-DoF arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_actuated_coords_offset: wp.array | None = None
    """
    The index offset of the actuated joint coordinates block of each world.\n
    Used to index into world-specific blocks of per-actuated-joint-coordinate arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_actuated_dofs_offset: wp.array | None = None
    """
    The index offset of the actuated joint DoF block of each world.\n
    Used to index into world-specific blocks of per-actuated-joint-DoF arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Constraint Offsets
    ###

    # TODO: move to joints section
    joint_cts_offset: wp.array | None = None
    """
    The index offset of the joint constraints block of each world.\n
    Used to index into world-specific blocks of per-joint-constraint arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    # TODO: Remove
    limit_cts_offset: wp.array | None = None
    """
    The index offset of the limit constraints block of each world.\n
    Used to index into world-specific blocks of per-limit-constraint arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    # TODO: Remove
    contact_cts_offset: wp.array | None = None
    """
    The index offset of the contact constraints block of each world.\n
    Used to index into world-specific blocks of per-contact-constraint arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    # TODO: Remove
    unilateral_cts_offset: wp.array | None = None
    """
    The index offset of the unilateral constraints block of each world.\n
    Used to index into world-specific blocks of per-unilateral-constraint arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    total_cts_offset: wp.array | None = None
    """
    The index offset of the total constraints block of each world.\n
    Used to index into world-specific blocks of per-constraint arrays.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Base Properties
    ###

    base_body_index: wp.array | None = None
    """
    The index of the base body assigned in each world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    base_joint_index: wp.array | None = None
    """
    The index of the base joint assigned in each world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Inertial Properties
    ###

    mass_min: wp.array | None = None
    """
    Smallest mass amongst all bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`float`.
    """

    mass_max: wp.array | None = None
    """
    Largest mass amongst all bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`float`.
    """

    mass_total: wp.array | None = None
    """
    Total mass over all bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`float`.
    """

    inertia_total: wp.array | None = None
    """
    Total diagonal inertia over all bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`float`.
    """


@dataclass
class ModelKamino:
    """
    A container to hold the time-invariant system model data.
    """

    _model: Model | None = None
    """The base :class:`newton.Model` instance from which this :class:`kamino.ModelKamino` was created."""

    _device: wp.DeviceLike | None = None
    """The Warp device on which the model data is allocated."""

    _requires_grad: bool = False
    """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""

    size: ModelKaminoSize = field(default_factory=ModelKaminoSize)
    """
    Host-side cache of the model summary sizes.\n
    This is used for memory allocations and kernel thread dimensions.
    """

    worlds: list[WorldDescriptor] = field(default_factory=list)
    """
    Host-side cache of the world descriptors.\n
    This is used to construct the model and for memory allocations.
    """

    info: ModelKaminoInfo | None = None
    """The model info container holding the information and meta-data of the model."""

    time: TimeModel | None = None
    """The time model container holding time-step of each world."""

    gravity: GravityModel | None = None
    """The gravity model container holding the gravity configurations for each world."""

    bodies: RigidBodiesModel | None = None
    """The rigid bodies model container holding all rigid body entities in the model."""

    joints: JointsModel | None = None
    """The joints model container holding all joint entities in the model."""

    geoms: GeometriesModel | None = None
    """The collision geometries model container holding all collision geometry entities in the model."""

    materials: MaterialsModel | None = None
    """
    The materials model container holding all material entities in the model.\n
    The materials data is currently defined globally to be shared by all worlds.
    """

    material_pairs: MaterialPairsModel | None = None
    """
    The material pairs model container holding all material pairs in the model.\n
    The material-pairs data is currently defined globally to be shared by all worlds.
    """

    @property
    def device(self) -> wp.DeviceLike:
        """The Warp device on which the model data is allocated."""
        return self._device

    @property
    def requires_grad(self) -> bool:
        """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""
        return self._requires_grad

    @classmethod
    def from_newton(cls, model: Model) -> ModelKamino:
        """
        Finalizes the ModelKamino from an existing newton.Model instance.
        """
        # Ensure the base model is valid
        if model is None:
            raise ValueError("Cannot finalize ModelKamino from a None newton.Model instance.")
        elif not isinstance(model, Model):
            raise TypeError("Cannot finalize ModelKamino from an invalid newton.Model instance.")

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

        # Compute the entity indices of each body w.r.t the corresponding world
        body_bid_np = _compute_entity_indices_wrt_world(model.body_world)
        joint_jid_np = _compute_entity_indices_wrt_world(model.joint_world)
        shape_sid_np = _compute_entity_indices_wrt_world(model.shape_world)

        # Compute the number of entities per world
        num_bodies_np = _compute_num_entities_per_world(model.body_world, model.num_worlds)
        num_joints_np = _compute_num_entities_per_world(model.joint_world, model.num_worlds)
        num_shapes_np = _compute_num_entities_per_world(model.shape_world, model.num_worlds)

        # Compute body coord/DoF counts per world
        num_body_coords_np = num_bodies_np * 7
        num_body_dofs_np = num_bodies_np * 6

        # Compute joint coord/DoF/constraint counts per world
        num_joint_coords_np = np.zeros((model.num_worlds,), dtype=int)
        num_joint_dofs_np = np.zeros((model.num_worlds,), dtype=int)
        num_joint_cts_np = np.zeros((model.num_worlds,), dtype=int)
        num_actuated_joints_np = np.zeros((model.num_worlds,), dtype=int)
        num_actuated_joint_coords_np = np.zeros((model.num_worlds,), dtype=int)
        num_actuated_joint_dofs_np = np.zeros((model.num_worlds,), dtype=int)
        num_passive_joints_np = np.zeros((model.num_worlds,), dtype=int)
        num_passive_joint_coords_np = np.zeros((model.num_worlds,), dtype=int)
        num_passive_joint_dofs_np = np.zeros((model.num_worlds,), dtype=int)
        joint_dof_type_np = np.zeros((model.joint_count,), dtype=int)
        joint_act_type_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_coords_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_dofs_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_cts_np = np.zeros((model.joint_count,), dtype=int)
        joint_B_r_Bj_np = np.zeros((model.joint_count, 3), dtype=float)
        joint_F_r_Fj_np = np.zeros((model.joint_count, 3), dtype=float)
        joint_X_j_np = np.zeros((model.joint_count, 9), dtype=float)
        joint_coord_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_cts_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_actuated_coord_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_actuated_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_passive_coord_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_passive_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_wid_np = model.joint_world.numpy()
        joint_type_np = model.joint_type.numpy()
        joint_act_mode_np = model.joint_act_mode.numpy()
        msg.error("joint_type_np: %s", joint_type_np)
        msg.error("joint_act_mode_np: %s", joint_act_mode_np)
        joint_parent_np = model.joint_parent.numpy()
        joint_child_np = model.joint_child.numpy()
        joint_X_p_np = model.joint_X_p.numpy()
        joint_X_c_np = model.joint_X_c.numpy()
        joint_axis_np = model.joint_axis.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        joint_q_start_np = model.joint_q_start.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()
        joint_limit_lower_np = model.joint_limit_lower.numpy()
        joint_limit_upper_np = model.joint_limit_upper.numpy()
        body_com_np = model.body_com.numpy()

        # ---------------------------------------------------------------------------
        # Pre-processing: absorb non-identity joint_X_c rotations into child body
        # frames so that Kamino sees aligned joint frames on both sides.
        #
        # Kamino's constraint system assumes a single joint frame X_j valid for both
        # the base (parent) and follower (child) bodies.  At q = 0 it requires
        #   q_base^{-1} * q_follower = identity
        # Newton, however, allows different parent / child joint-frame orientations
        # via joint_X_p and joint_X_c.  At q = 0 Newton's FK gives:
        #   q_follower = q_parent * q_pj * inv(q_cj)
        # so q_base^{-1} * q_follower = q_pj * inv(q_cj) which is generally not
        # identity.
        #
        # To fix this we apply a per-body correction rotation q_corr = q_cj * inv(q_pj)
        # (applied on the right) to each child body's frame:
        #   q_body_new = q_body_old * q_corr
        # This makes q_base^{-1} * q_follower_new = identity at q = 0, and the joint
        # rotation axis R(q_pj) * axis is preserved.
        #
        # All body-local quantities (CoM, inertia, shapes) are re-expressed in the
        # rotated frame, and downstream joint_X_p transforms are updated to account
        # for the parent body's frame change.
        # ---------------------------------------------------------------------------

        def _to_wpq(q):
            return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

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
        body_q_np = model.body_q.numpy().copy()
        body_qd_np = model.body_qd.numpy().copy()
        body_com_np = body_com_np.copy()
        body_inertia_np = model.body_inertia.numpy().copy()
        body_inv_inertia_np = model.body_inv_inertia.numpy().copy()
        shape_transform_np = model.shape_transform.numpy().copy()
        shape_body_np = model.shape_body.numpy()
        joint_X_p_np = joint_X_p_np.copy()
        joint_X_c_np = joint_X_c_np.copy()

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
            body_inv_inertia_np[child] = (
                R_inv_corr @ body_inv_inertia_np[child].astype(np.float64) @ R_inv_corr.T
            )

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

        # Overwrite the warp arrays on the model so that downstream Kamino code
        # (which reads model.body_q, model.body_com, etc.) picks up the corrected
        # values.  This is safe because ModelKamino.from_newton owns the conversion.
        model.body_q.assign(body_q_np)
        model.body_qd.assign(body_qd_np)
        model.body_com.assign(body_com_np)
        model.body_inertia.assign(body_inertia_np)
        model.body_inv_inertia.assign(body_inv_inertia_np)
        model.shape_transform.assign(shape_transform_np)
        model.joint_X_p.assign(joint_X_p_np)
        model.joint_X_c.assign(joint_X_c_np)

        # Re-read numpy views that are used below
        body_com_np = model.body_com.numpy()
        joint_X_p_np = model.joint_X_p.numpy()
        joint_X_c_np = model.joint_X_c.numpy()

        for j in range(model.joint_count):
            # TODO
            wid_j = joint_wid_np[j]

            # TODO
            joint_coord_start_np[j] = num_joint_coords_np[wid_j]
            joint_dofs_start_np[j] = num_joint_dofs_np[wid_j]
            joint_cts_start_np[j] = num_joint_cts_np[wid_j]
            joint_actuated_coord_start_np[j] = num_actuated_joint_coords_np[wid_j]
            joint_actuated_dofs_start_np[j] = num_actuated_joint_dofs_np[wid_j]
            joint_passive_coord_start_np[j] = num_passive_joint_coords_np[wid_j]
            joint_passive_dofs_start_np[j] = num_passive_joint_dofs_np[wid_j]

            # TODO
            type_j = int(joint_type_np[j])
            dof_dim_j = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
            q_count_j = int(joint_q_start_np[j + 1] - joint_q_start_np[j])
            qd_count_j = int(joint_qd_start_np[j + 1] - joint_qd_start_np[j])
            limit_upper_j = joint_limit_upper_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]].astype(float)
            limit_lower_j = joint_limit_lower_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]].astype(float)
            msg.error("[%s]: key_j: %s", j, model.joint_key[j])
            msg.error("[%s]: type_j: %s", j, JointType(type_j).name)
            msg.error("[%s]: dof_dim_j: %s", j, dof_dim_j)
            msg.error("[%s]: q_count_j: %s", j, q_count_j)
            msg.error("[%s]: qd_count_j: %s", j, qd_count_j)
            msg.error("[%s]: limit_lower_j: %s", j, limit_lower_j)
            msg.error("[%s]: limit_upper_j: %s", j, limit_upper_j)

            # TODO
            dof_type_j = newton_to_kamino_joint_dof_type(
                type_j, dof_dim_j, q_count_j, qd_count_j, limit_lower_j, limit_upper_j
            )
            msg.warning("[%s]: dof_type_j: %s", j, dof_type_j)
            ncoords_j = dof_type_j.num_coords
            ndofs_j = dof_type_j.num_dofs

            msg.warning("[%s]: ndofs_j: %s", j, ndofs_j)
            ncts_j = dof_type_j.num_cts
            joint_dof_type_np[j] = dof_type_j.value
            num_joint_coords_np[wid_j] += ncoords_j
            num_joint_dofs_np[wid_j] += ndofs_j
            num_joint_cts_np[wid_j] += ncts_j
            joint_num_coords_np[j] = ncoords_j
            joint_num_dofs_np[j] = ndofs_j
            joint_num_cts_np[j] = ncts_j

            # TODO
            dofs_start_j = joint_qd_start_np[j]
            msg.warning("[%s]: dofs_start_j: %s", j, dofs_start_j)
            joint_axes_j = joint_axis_np[dofs_start_j : dofs_start_j + ndofs_j]
            R_axis_j = axes_matrix_from_joint_type(dof_type_j, dof_dim_j, joint_axes_j)
            joint_dofs_act_mode_j = joint_act_mode_np[dofs_start_j : dofs_start_j + ndofs_j]
            msg.warning("[%s]: joint_dofs_act_mode_j: %s", j, joint_dofs_act_mode_j)
            joint_act_mode_j = (
                max(joint_dofs_act_mode_j) if len(joint_dofs_act_mode_j) > 0 else JointActuationType.PASSIVE
            )
            msg.warning("[%s]: joint_act_mode_j: %s", j, joint_act_mode_j)

            # TODO
            act_type_j = newton_to_kamino_joint_actuation_type(joint_act_mode_j)
            joint_act_type_np[j] = act_type_j.value
            if act_type_j > JointActuationType.PASSIVE:
                num_actuated_joints_np[wid_j] += 1
                num_actuated_joint_coords_np[wid_j] += ncoords_j
                num_actuated_joint_dofs_np[wid_j] += ndofs_j
                joint_passive_coord_start_np[j] = -1
                joint_passive_dofs_start_np[j] = -1
            else:
                num_passive_joints_np[wid_j] += 1
                num_passive_joint_coords_np[wid_j] += ncoords_j
                num_passive_joint_dofs_np[wid_j] += ndofs_j
                joint_actuated_coord_start_np[j] = -1
                joint_actuated_dofs_start_np[j] = -1

            # TODO
            parent_bid = joint_parent_np[j]
            p_r_p_com = wp.vec3f(body_com_np[parent_bid]) if parent_bid >= 0 else wp.vec3f(0.0, 0.0, 0.0)
            c_r_c_com = wp.vec3f(body_com_np[joint_child_np[j]])
            X_p_j = wp.transformf(*joint_X_p_np[j, :])
            X_c_j = wp.transformf(*joint_X_c_np[j, :])
            q_p_j = wp.transform_get_rotation(X_p_j)
            p_r_p_j = wp.transform_get_translation(X_p_j)
            c_r_c_j = wp.transform_get_translation(X_c_j)

            # TODO
            B_r_Bj = p_r_p_j - p_r_p_com
            F_r_Fj = c_r_c_j - c_r_c_com
            X_j = wp.quat_to_matrix(q_p_j) @ R_axis_j
            joint_B_r_Bj_np[j, :] = B_r_Bj
            joint_F_r_Fj_np[j, :] = F_r_Fj
            joint_X_j_np[j, :] = X_j
        msg.error("joint_act_type_np: %s", joint_act_type_np)

        # TODO
        joint_velocity_limit_np = model.joint_velocity_limit.numpy()
        joint_effort_limit_np = model.joint_effort_limit.numpy()
        np.clip(a=joint_limit_lower_np, a_min=JOINT_QMIN, a_max=JOINT_QMAX, out=joint_limit_lower_np)
        np.clip(a=joint_limit_upper_np, a_min=JOINT_QMIN, a_max=JOINT_QMAX, out=joint_limit_upper_np)
        np.clip(a=joint_velocity_limit_np, a_min=-JOINT_DQMAX, a_max=JOINT_DQMAX, out=joint_velocity_limit_np)
        np.clip(a=joint_effort_limit_np, a_min=-JOINT_TAUMAX, a_max=JOINT_TAUMAX, out=joint_effort_limit_np)
        model.joint_limit_lower.assign(joint_limit_lower_np)
        model.joint_limit_upper.assign(joint_limit_upper_np)
        model.joint_velocity_limit.assign(joint_velocity_limit_np)
        model.joint_effort_limit.assign(joint_effort_limit_np)

        # TODO
        materials_manager = MaterialManager()
        # shape_material: list[MaterialDescriptor] = []
        # shape_friction_np = model.shape_material_mu.numpy()
        # shape_restitution_np = model.shape_material_restitution.numpy()
        # shape_world_np = model.shape_world.numpy()
        # for s in range(model.shape_count):
        #     shape_material.append(
        #         MaterialDescriptor(
        #             restitution=shape_restitution_np[s],
        #             static_friction=shape_friction_np[s],
        #             dynamic_friction=shape_friction_np[s],
        #             wid=shape_world_np[s],
        #         )
        #     )
        #     materials_manager.register(shape_material[-1])

        # Convert per-shape properties from Newton to Kamino format
        shape_type_np = model.shape_type.numpy()
        shape_scale_np = model.shape_scale.numpy()
        shape_flags_np = model.shape_flags.numpy()
        shape_transform_np = model.shape_transform.numpy()
        geom_shape_collision_group_np = model.shape_collision_group.numpy()
        geom_shape_type_np = np.zeros((model.shape_count,), dtype=int)
        geom_shape_params_np = np.zeros((model.shape_count, 4), dtype=float)
        model_num_collidable_geoms = 0
        for s in range(model.shape_count):
            shape_type, params = convert_newton_geo_to_kamino_shape(shape_type_np[s], shape_scale_np[s])
            geom_shape_type_np[s] = shape_type
            geom_shape_params_np[s, :] = params
            if (shape_flags_np[s] & ShapeFlags.COLLIDE_SHAPES) != 0 and geom_shape_collision_group_np[s] > 0:
                model_num_collidable_geoms += 1

        # Fix plane normals: derive from the shape transform rotation (local Z-axis)
        # instead of the hardcoded default in convert_newton_geo_to_kamino_shape.
        for s in range(model.shape_count):
            if shape_type_np[s] == GeoType.PLANE:
                tf = shape_transform_np[s]
                q_rot = _to_wpq(np.array([tf[3], tf[4], tf[5], tf[6]]))
                normal = wp.quat_rotate(q_rot, wp.vec3(0.0, 0.0, 1.0))
                geom_shape_params_np[s, 0] = float(normal[0])
                geom_shape_params_np[s, 1] = float(normal[1])
                geom_shape_params_np[s, 2] = float(normal[2])
                geom_shape_params_np[s, 3] = 0.0

        # Compute total number of required contacts per world
        # TODO: We need to do this properly based on the actual geometries in each world
        required_contacts_per_world = model.rigid_contact_max // model.num_worlds
        world_required_contacts = [required_contacts_per_world] * model.num_worlds

        # Compute offsets per world
        world_shape_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_body_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_body_dof_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_joint_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_joint_coord_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_joint_dof_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_joint_cts_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_actuated_joint_coord_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_actuated_joint_dofs_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_passive_joint_coord_offset_np = np.zeros((model.num_worlds,), dtype=int)
        world_passive_joint_dofs_offset_np = np.zeros((model.num_worlds,), dtype=int)

        for w in range(1, model.num_worlds):
            world_shape_offset_np[w] = world_shape_offset_np[w - 1] + num_shapes_np[w - 1]
            world_body_offset_np[w] = world_body_offset_np[w - 1] + num_bodies_np[w - 1]
            world_body_dof_offset_np[w] = world_body_dof_offset_np[w - 1] + num_body_dofs_np[w - 1]
            world_joint_offset_np[w] = world_joint_offset_np[w - 1] + num_joints_np[w - 1]
            world_joint_coord_offset_np[w] = world_joint_coord_offset_np[w - 1] + num_joint_coords_np[w - 1]
            world_joint_dof_offset_np[w] = world_joint_dof_offset_np[w - 1] + num_joint_dofs_np[w - 1]
            world_joint_cts_offset_np[w] = world_joint_cts_offset_np[w - 1] + num_joint_cts_np[w - 1]
            world_actuated_joint_coord_offset_np[w] = (
                world_actuated_joint_coord_offset_np[w - 1] + num_actuated_joint_coords_np[w - 1]
            )
            world_actuated_joint_dofs_offset_np[w] = (
                world_actuated_joint_dofs_offset_np[w - 1] + num_actuated_joint_dofs_np[w - 1]
            )
            world_passive_joint_coord_offset_np[w] = (
                world_passive_joint_coord_offset_np[w - 1] + num_passive_joint_coords_np[w - 1]
            )
            world_passive_joint_dofs_offset_np[w] = (
                world_passive_joint_dofs_offset_np[w - 1] + num_passive_joint_dofs_np[w - 1]
            )

        # Determine the base body and joint indices per world
        base_body_idx_np = np.full((model.num_worlds,), -1, dtype=int)
        base_joint_idx_np = np.full((model.num_worlds,), -1, dtype=int)
        body_world_np = model.body_world.numpy()
        joint_world_np = model.joint_world.numpy()
        body_world_start_np = model.body_world_start.numpy()
        # joint_world_start_np = model.joint_world_start.numpy()

        # Check for articulations
        if model.articulation_count > 0:
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
            msg.error("joint_world_np: %s", joint_world_np)
            msg.error("joint_parent_np: %s", joint_parent_np)
            msg.error("joint_child_np: %s", joint_child_np)

            # TODO: How to handle no free joint being defined?
            # Create a list of joint indices with parent body == -1 for each world
            world_parent_joints: dict[int, list[int]] = {w: [] for w in range(model.num_worlds)}
            msg.warning("world_parent_joints: %s", world_parent_joints)
            for j in range(model.joint_count):
                wid_j = joint_world_np[j]
                msg.info("joint %s belongs to world %s", j, wid_j)
                parent_j = joint_parent_np[j]
                if parent_j == -1:
                    msg.info("BEFORE: world_parent_joints: %s", world_parent_joints)
                    world_parent_joints[wid_j].append(j)
                    msg.info("AFTER: world_parent_joints[%s]: %s", wid_j, world_parent_joints)
            msg.error("world_parent_joints: %s", world_parent_joints)

            # For each world, assign the base body and joint based on the first joint with parent == -1,
            # If no joint with parent == -1 is found in a world, then assign the first body as base
            # If multiple joints with parent == -1 are found in a world, then assign the first one as the base
            for w in range(model.num_worlds):
                if len(world_parent_joints[w]) > 0:
                    j = world_parent_joints[w][0]
                    base_joint_idx_np[w] = j
                    base_body_idx_np[w] = int(joint_child_np[j])
                else:
                    base_body_idx_np[w] = int(body_world_start_np[w])
                    base_joint_idx_np[w] = -1

        # Fall-back: first body and joint in the world
        else:
            for w in range(model.num_worlds):
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
        msg.error("base_body_idx_np: %s", base_body_idx_np)
        msg.error("base_joint_idx_np: %s", base_joint_idx_np)
        # Ensure that all worlds have a base body assigned
        for w in range(model.num_worlds):
            if base_body_idx_np[w] == -1:
                raise ValueError(f"World {w} does not have a base body assigned (index is -1).")

        # Construct per-world inertial summaries
        mass_min_np = np.zeros((model.num_worlds,), dtype=float)
        mass_max_np = np.zeros((model.num_worlds,), dtype=float)
        mass_total_np = np.zeros((model.num_worlds,), dtype=float)
        inertia_total_np = np.zeros((model.num_worlds,), dtype=float)
        body_mass_np = model.body_mass.numpy()
        body_inertia_np = model.body_inertia.numpy()
        for w in range(model.num_worlds):
            masses_w = []
            for b in range(model.body_count):
                if body_world_np[b] == w:
                    mass_b = body_mass_np[b]
                    masses_w.append(mass_b)
                    mass_total_np[w] += mass_b
                    inertia_total_np[w] += 3.0 * mass_b + body_inertia_np[b].diagonal().sum()
            mass_min_np[w] = min(masses_w)
            mass_max_np[w] = max(masses_w)

        # Construct per-world gravity
        gravity_g_dir_acc_np = np.zeros(shape=(model.num_worlds, 4), dtype=float)
        gravity_vector_np = np.zeros(shape=(model.num_worlds, 4), dtype=float)
        gravity_np = model.gravity.numpy()
        for w in range(model.num_worlds):
            gravity_accel = np.linalg.norm(gravity_np[w, :])
            if gravity_accel > 0.0:
                gravity_dir = gravity_np[w, :] / gravity_accel
            else:
                gravity_dir = np.array([0.0, 0.0, -1.0])
            gravity_g_dir_acc_np[w, :] = np.array(
                [gravity_dir[0], gravity_dir[1], gravity_dir[2], gravity_accel], dtype=float
            )
            gravity_vector_np[w, 0:3] = gravity_np[w, :]
            gravity_vector_np[w, 3] = 1.0  # Enable gravity by default in all worlds

        # Construct the per-material and per-material-pair properties
        materials_rest = [materials_manager.restitution_vector()]
        materials_static_fric = [materials_manager.static_friction_vector()]
        materials_dynamic_fric = [materials_manager.dynamic_friction_vector()]
        mpairs_rest = [materials_manager.restitution_matrix()]
        mpairs_static_fric = [materials_manager.static_friction_matrix()]
        mpairs_dynamic_fric = [materials_manager.dynamic_friction_matrix()]

        ###
        # Model Attributes
        ###

        # TODO: Construct the world descriptors from the newton.Model instance
        model_worlds: list[WorldDescriptor] = [None] * model.num_worlds
        for w in range(model.num_worlds):
            model_worlds[w] = WorldDescriptor(
                name=f"world_{w}",
                wid=w,
                num_bodies=int(num_bodies_np[w]),
                num_joints=int(num_joints_np[w]),
                num_passive_joints=int(num_passive_joints_np[w]),
                num_actuated_joints=int(num_actuated_joints_np[w]),
                num_geoms=int(num_shapes_np[w]),
                num_materials=0,  # TODO: how to handle both global and per-world materials simultaneously?
                num_body_coords=int(num_body_coords_np[w]),
                num_body_dofs=int(num_body_dofs_np[w]),
                num_joint_coords=int(num_joint_coords_np[w]),
                num_joint_dofs=int(num_joint_dofs_np[w]),
                num_joint_cts=int(num_joint_cts_np[w]),
                num_passive_joint_coords=int(num_passive_joint_coords_np[w]),
                num_passive_joint_dofs=int(num_passive_joint_dofs_np[w]),
                num_actuated_joint_coords=int(num_actuated_joint_coords_np[w]),
                num_actuated_joint_dofs=int(num_actuated_joint_dofs_np[w]),
                # TODO
                joint_coords=[],
                joint_dofs=[],
                joint_passive_coords=[],
                joint_passive_dofs=[],
                joint_actuated_coords=[],
                joint_actuated_dofs=[],
                joint_cts=[],
                # TODO
                bodies_idx_offset=int(world_body_offset_np[w]),
                joints_idx_offset=int(world_joint_offset_np[w]),
                geoms_idx_offset=int(world_shape_offset_np[w]),
                body_dofs_idx_offset=int(world_body_dof_offset_np[w]),
                joint_coords_idx_offset=int(world_joint_coord_offset_np[w]),
                joint_dofs_idx_offset=int(world_joint_dof_offset_np[w]),
                passive_joint_coords_idx_offset=int(world_passive_joint_coord_offset_np[w]),
                passive_joint_dofs_idx_offset=int(world_passive_joint_dofs_offset_np[w]),
                actuated_joint_coords_idx_offset=int(world_actuated_joint_coord_offset_np[w]),
                actuated_joint_dofs_idx_offset=int(world_actuated_joint_dofs_offset_np[w]),
                joint_cts_idx_offset=int(world_joint_cts_offset_np[w]),
                # TODO
                body_names=[],
                body_uids=[],
                joint_names=[],
                joint_uids=[],
                geom_names=[],
                geom_uids=[],
                material_names=[],
                material_uids=[],
                unary_joint_names=[],
                fixed_joint_names=[],
                passive_joint_names=[],
                actuated_joint_names=[],
                geometry_layers=["default"],
                geometry_max_contacts=[-1] * int(num_shapes_np[w]),
                # TODO
                base_body_idx=int(base_body_idx_np[w]),
                base_joint_idx=int(base_joint_idx_np[w]),
                mass_min=float(mass_min_np[w]),
                mass_max=float(mass_max_np[w]),
                mass_total=float(mass_total_np[w]),
                inertia_total=float(inertia_total_np[w]),
            )

        # Construct ModelKaminoSize from the newton.Model instance
        model_size = ModelKaminoSize(
            num_worlds=model.num_worlds,
            sum_of_num_bodies=int(num_bodies_np.sum()),
            max_of_num_bodies=int(num_bodies_np.max()),
            sum_of_num_joints=int(num_joints_np.sum()),
            max_of_num_joints=int(num_joints_np.max()),
            sum_of_num_passive_joints=int(num_passive_joints_np.sum()),
            max_of_num_passive_joints=int(num_passive_joints_np.max()),
            sum_of_num_actuated_joints=int(num_actuated_joints_np.sum()),
            max_of_num_actuated_joints=int(num_actuated_joints_np.max()),
            sum_of_num_geoms=int(num_shapes_np.sum()),
            max_of_num_geoms=int(num_shapes_np.max()),
            sum_of_num_materials=materials_manager.num_materials,
            max_of_num_materials=materials_manager.num_materials,
            sum_of_num_material_pairs=materials_manager.num_material_pairs,
            max_of_num_material_pairs=materials_manager.num_material_pairs,
            sum_of_num_body_dofs=int(num_body_dofs_np.sum()),
            max_of_num_body_dofs=int(num_body_dofs_np.max()),
            sum_of_num_joint_coords=int(num_joint_coords_np.sum()),
            max_of_num_joint_coords=int(num_joint_coords_np.max()),
            sum_of_num_joint_dofs=int(num_joint_dofs_np.sum()),
            max_of_num_joint_dofs=int(num_joint_dofs_np.max()),
            sum_of_num_passive_joint_coords=int(num_passive_joint_coords_np.sum()),
            max_of_num_passive_joint_coords=int(num_passive_joint_coords_np.max()),
            sum_of_num_passive_joint_dofs=int(num_passive_joint_dofs_np.sum()),
            max_of_num_passive_joint_dofs=int(num_passive_joint_dofs_np.max()),
            sum_of_num_actuated_joint_coords=int(num_actuated_joint_coords_np.sum()),
            max_of_num_actuated_joint_coords=int(num_actuated_joint_coords_np.max()),
            sum_of_num_actuated_joint_dofs=int(num_actuated_joint_dofs_np.sum()),
            max_of_num_actuated_joint_dofs=int(num_actuated_joint_dofs_np.max()),
            sum_of_num_joint_cts=int(num_joint_cts_np.sum()),
            max_of_num_joint_cts=int(num_joint_cts_np.max()),
            sum_of_max_limits=0,
            max_of_max_limits=0,
            sum_of_max_contacts=0,
            max_of_max_contacts=0,
            sum_of_max_unilaterals=0,
            max_of_max_unilaterals=0,
            sum_of_max_total_cts=int(num_joint_cts_np.sum()),
            max_of_max_total_cts=int(num_joint_cts_np.max()),
        )
        # msg.info("ModelKaminoSize:\n%s", model_size)

        # Construct the model entities from the newton.Model instance
        with wp.ScopedDevice(device=model.device):
            # Per-world heterogeneous model info
            model_info = ModelKaminoInfo(
                num_worlds=model.num_worlds,
                num_bodies=wp.array(num_bodies_np, dtype=int32),
                num_joints=wp.array(num_joints_np, dtype=int32),
                num_passive_joints=wp.array(num_passive_joints_np, dtype=int32),
                num_actuated_joints=wp.array(num_actuated_joints_np, dtype=int32),
                num_geoms=wp.array(num_shapes_np, dtype=int32),
                num_body_dofs=wp.array(num_body_dofs_np, dtype=int32),
                num_joint_coords=wp.array(num_joint_coords_np, dtype=int32),
                num_joint_dofs=wp.array(num_joint_dofs_np, dtype=int32),
                num_passive_joint_coords=wp.array(num_passive_joint_coords_np, dtype=int32),
                num_passive_joint_dofs=wp.array(num_passive_joint_dofs_np, dtype=int32),
                num_actuated_joint_coords=wp.array(num_actuated_joint_coords_np, dtype=int32),
                num_actuated_joint_dofs=wp.array(num_actuated_joint_dofs_np, dtype=int32),
                num_joint_cts=wp.array(num_joint_cts_np, dtype=int32),
                bodies_offset=wp.array(world_body_offset_np, dtype=int32),
                joints_offset=wp.array(world_joint_offset_np, dtype=int32),
                body_dofs_offset=wp.array(world_body_dof_offset_np, dtype=int32),
                joint_coords_offset=wp.array(world_joint_coord_offset_np, dtype=int32),
                joint_dofs_offset=wp.array(world_joint_dof_offset_np, dtype=int32),
                joint_cts_offset=wp.array(world_joint_cts_offset_np, dtype=int32),
                joint_passive_coords_offset=wp.array(world_passive_joint_coord_offset_np, dtype=int32),
                joint_passive_dofs_offset=wp.array(world_passive_joint_dofs_offset_np, dtype=int32),
                joint_actuated_coords_offset=wp.array(world_actuated_joint_coord_offset_np, dtype=int32),
                joint_actuated_dofs_offset=wp.array(world_actuated_joint_dofs_offset_np, dtype=int32),
                base_body_index=wp.array(base_body_idx_np, dtype=int32),
                base_joint_index=wp.array(base_joint_idx_np, dtype=int32),
                mass_min=wp.array(mass_min_np, dtype=float32),
                mass_max=wp.array(mass_max_np, dtype=float32),
                mass_total=wp.array(mass_total_np, dtype=float32),
                inertia_total=wp.array(inertia_total_np, dtype=float32),
            )

            # Per-world time
            model_time = TimeModel(
                dt=wp.zeros(shape=(model.num_worlds,), dtype=float32),
                inv_dt=wp.zeros(shape=(model.num_worlds,), dtype=float32),
            )

            # Per-world gravity
            model_gravity = GravityModel(
                g_dir_acc=wp.array(gravity_g_dir_acc_np, dtype=vec4f),
                vector=wp.array(gravity_vector_np, dtype=vec4f),
            )

            # Rigid bodies
            model_bodies = RigidBodiesModel(
                num_bodies=model.body_count,
                label=model.body_key,
                wid=model.body_world,
                bid=wp.array(body_bid_np, dtype=int32),  # TODO: Remove
                i_r_com_i=model.body_com,
                m_i=model.body_mass,
                inv_m_i=model.body_inv_mass,
                i_I_i=model.body_inertia,
                inv_i_I_i=model.body_inv_inertia,
                q_i_0=model.body_q,
                u_i_0=model.body_qd,
            )

            # Joints
            model_joints = JointsModel(
                num_joints=model.joint_count,
                label=model.joint_key,
                wid=model.joint_world,
                jid=wp.array(joint_jid_np, dtype=int32),  # TODO: Remove
                # dof_type=model.joint_type,
                # act_type=model.joint_act_mode,
                dof_type=wp.array(joint_dof_type_np, dtype=int32),
                act_type=wp.array(joint_act_type_np, dtype=int32),
                bid_B=model.joint_parent,
                bid_F=model.joint_child,
                B_r_Bj=wp.array(joint_B_r_Bj_np, dtype=wp.vec3f),
                F_r_Fj=wp.array(joint_F_r_Fj_np, dtype=wp.vec3f),
                X_j=wp.array(joint_X_j_np.reshape((model.joint_count, 3, 3)), dtype=wp.mat33f),
                q_j_min=model.joint_limit_lower,
                q_j_max=model.joint_limit_upper,
                dq_j_max=model.joint_velocity_limit,
                tau_j_max=model.joint_effort_limit,
                q_j_0=model.joint_q,
                dq_j_0=model.joint_qd,
                # TODO:
                # coords_start=model.joint_q_start,
                # dofs_start=model.joint_qd_start,
                num_coords=wp.array(joint_num_coords_np, dtype=int32),
                num_dofs=wp.array(joint_num_dofs_np, dtype=int32),
                num_cts=wp.array(joint_num_cts_np, dtype=int32),
                coords_offset=wp.array(joint_coord_start_np, dtype=int32),
                dofs_offset=wp.array(joint_dofs_start_np, dtype=int32),
                cts_offset=wp.array(joint_cts_start_np, dtype=int32),
                passive_coords_offset=wp.array(joint_passive_coord_start_np, dtype=int32),
                passive_dofs_offset=wp.array(joint_passive_dofs_start_np, dtype=int32),
                actuated_coords_offset=wp.array(joint_actuated_coord_start_np, dtype=int32),
                actuated_dofs_offset=wp.array(joint_actuated_dofs_start_np, dtype=int32),
            )

            # Collision geometries
            model_geoms = GeometriesModel(
                num_geoms=model.shape_count,
                num_collidable_geoms=model_num_collidable_geoms,
                num_collidable_geom_pairs=model.shape_contact_pair_count,
                model_max_contacts=model.rigid_contact_max,
                world_max_contacts=world_required_contacts,
                label=model.shape_key,
                wid=model.shape_world,
                gid=wp.array(shape_sid_np, dtype=int32),  # TODO: Remove
                lid=wp.zeros(shape=model.shape_count, dtype=int32),  # TODO: Remove this since it's not used anywhere
                bid=model.shape_body,
                sid=wp.array(geom_shape_type_np, dtype=int32),
                ptr=model.shape_source_ptr,
                offset=model.shape_transform,
                params=wp.array(geom_shape_params_np, dtype=vec4f),
                mid=wp.full(shape=(model.shape_count,), value=0, dtype=int32),  # TODO: model.shape_material_id
                group=wp.full(shape=(model.shape_count,), value=1, dtype=int32),  # TODO: model.shape_collision_group
                collides=wp.full(shape=(model.shape_count,), value=1, dtype=int32),  # TODO: model.shape_collision_group
                margin=model.shape_contact_margin,
                collidable_pairs=model.shape_contact_pairs,
            )

            # Per-material properties
            model_materials = MaterialsModel(
                num_materials=model_size.sum_of_num_materials,
                restitution=wp.array(materials_rest[0], dtype=float32),
                static_friction=wp.array(materials_static_fric[0], dtype=float32),
                dynamic_friction=wp.array(materials_dynamic_fric[0], dtype=float32),
            )

            # Per-material-pair properties
            model_material_pairs = MaterialPairsModel(
                num_material_pairs=model_size.sum_of_num_material_pairs,
                restitution=wp.array(mpairs_rest[0], dtype=float32),
                static_friction=wp.array(mpairs_static_fric[0], dtype=float32),
                dynamic_friction=wp.array(mpairs_dynamic_fric[0], dtype=float32),
            )

        # Post-processing after construction
        # TODO: Transform initial body CoM state from body frame state --> probably using kernel called in reset methods

        # Construct and return the new ModelKamino instance
        return ModelKamino(
            _model=model,
            _device=model.device,
            _requires_grad=model.requires_grad,
            size=model_size,
            worlds=model_worlds,
            info=model_info,
            time=model_time,
            gravity=model_gravity,
            bodies=model_bodies,
            joints=model_joints,
            geoms=model_geoms,
            materials=model_materials,
            material_pairs=model_material_pairs,
        )

    def data(
        self,
        unilateral_cts: bool = False,
        requires_grad: bool = False,
        device: wp.DeviceLike = None,
    ) -> DataKamino:
        """
        Creates a model data container with the initial state of the model entities.

        Parameters:
            unilateral_cts (`bool`, optional):
                Whether to include unilateral constraints (limits and contacts) in the model data. Defaults to `True`.
            requires_grad (`bool`, optional):
                Whether the model data should require gradients. Defaults to `False`.
            device (`wp.DeviceLike`, optional):
                The device to create the model data on. If not specified, the model's device is used.
                Defaults to `None`. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Retrieve entity counts
        nw = self.size.num_worlds
        nb = self.size.sum_of_num_bodies
        nj = self.size.sum_of_num_joints
        ng = self.size.sum_of_num_geoms

        # Retrieve the joint coordinate, DoF and constraint counts
        njq = self.size.sum_of_num_joint_coords
        njd = self.size.sum_of_num_joint_dofs
        njc = self.size.sum_of_num_joint_cts

        # Construct the model data on the specified device
        with wp.ScopedDevice(device=device):
            # Create a new model data info with the total constraint
            # counts initialized to the joint constraints count
            info = DataKaminoInfo(
                num_total_cts=wp.clone(self.info.num_joint_cts),
            )

            # If unilateral constraints are enabled, initialize the additional state info
            if unilateral_cts:
                info.num_limits = wp.zeros(shape=nw, dtype=int32)
                info.num_contacts = wp.zeros(shape=nw, dtype=int32)
                info.num_limit_cts = wp.zeros(shape=nw, dtype=int32)
                info.num_contact_cts = wp.zeros(shape=nw, dtype=int32)
                info.limit_cts_group_offset = wp.zeros(shape=nw, dtype=int32)
                info.contact_cts_group_offset = wp.zeros(shape=nw, dtype=int32)

            # Construct the time state
            time = TimeData(
                steps=wp.zeros(shape=nw, dtype=int32, requires_grad=requires_grad),
                time=wp.zeros(shape=nw, dtype=float32, requires_grad=requires_grad),
            )

            # Construct the rigid bodies state from the model's initial state
            bodies = RigidBodiesData(
                num_bodies=nb,
                I_i=wp.zeros(shape=nb, dtype=mat33f, requires_grad=requires_grad),
                inv_I_i=wp.zeros(shape=nb, dtype=mat33f, requires_grad=requires_grad),
                q_i=wp.clone(self.bodies.q_i_0, requires_grad=requires_grad),
                u_i=wp.clone(self.bodies.u_i_0, requires_grad=requires_grad),
                w_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_a_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_j_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_l_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_c_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_e_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
            )

            # Construct the joints state from the model's initial state
            joints = JointsData(
                num_joints=nj,
                p_j=wp.zeros(shape=nj, dtype=transformf, requires_grad=requires_grad),
                r_j=wp.zeros(shape=njc, dtype=float32, requires_grad=requires_grad),
                dr_j=wp.zeros(shape=njc, dtype=float32, requires_grad=requires_grad),
                lambda_j=wp.zeros(shape=njc, dtype=float32, requires_grad=requires_grad),
                q_j=wp.zeros(shape=njq, dtype=float32, requires_grad=requires_grad),
                q_j_p=wp.zeros(shape=njq, dtype=float32, requires_grad=requires_grad),
                dq_j=wp.zeros(shape=njd, dtype=float32, requires_grad=requires_grad),
                tau_j=wp.zeros(shape=njd, dtype=float32, requires_grad=requires_grad),
                j_w_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad),
                j_w_c_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad),
                j_w_a_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad),
                j_w_l_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad),
            )

            # Construct the geometries state from the model's initial state
            geoms = GeometriesData(
                num_geoms=ng,
                pose=wp.zeros(shape=ng, dtype=transformf, requires_grad=requires_grad),
            )

        # Assemble and return the new model data container
        return DataKamino(
            info=info,
            time=time,
            bodies=bodies,
            joints=joints,
            geoms=geoms,
        )

    def state(self, requires_grad: bool = False, device: wp.DeviceLike = None) -> StateKamino:
        """
        Creates state container initialized to the initial body state defined in the model.

        Parameters:
            requires_grad (`bool`, optional):
                Whether the state should require gradients. Defaults to `False`.
            device (`wp.DeviceLike`, optional):
                The device to create the state on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new state container with the initial state of the model entities on the specified device
        with wp.ScopedDevice(device=device):
            state = StateKamino(
                q_i=wp.clone(self.bodies.q_i_0, requires_grad=requires_grad),
                u_i=wp.clone(self.bodies.u_i_0, requires_grad=requires_grad),
                w_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                q_j=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                q_j_p=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                dq_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                lambda_j=wp.zeros(shape=self.size.sum_of_num_joint_cts, dtype=float32, requires_grad=requires_grad),
            )

        # Return the constructed state container
        return state

    def control(self, requires_grad: bool = False, device: wp.DeviceLike = None) -> ControlKamino:
        """
        Creates a control container with all values initialized to zeros.

        Parameters:
            requires_grad (`bool`, optional):
                Whether the control container should require gradients. Defaults to `False`.
            device (`wp.DeviceLike`, optional):
                The device to create the control container on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new control container on the specified device
        with wp.ScopedDevice(device=device):
            control = ControlKamino(
                tau_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad)
            )

        # Return the constructed control container
        return control
