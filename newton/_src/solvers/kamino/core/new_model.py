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

"""Constrained Rigid Multi-Body Model & Data Containers"""

from dataclasses import dataclass

import numpy as np
import warp as wp
from warp.context import Devicelike

from ....sim.model import Model
from ..utils import logger as msg
from .bodies import RigidBodiesData, RigidBodiesModel
from .geometry import CollisionGeometriesModel, GeometriesData, GeometriesModel
from .gravity import GravityModel
from .joints import JointActuationType, JointsData, JointsModel, newton_to_kamino_joint_dof_type
from .materials import MaterialPairsModel
from .new_control import KaminoControl
from .new_data import KaminoData, KaminoDataInfo
from .new_state import KaminoState
from .time import TimeData, TimeModel
from .types import float32, int32, mat33f, transformf, vec6f
from .world import WorldDescriptor

###
# Module interface
###

__all__ = [
    "KaminoModel",
    "KaminoModelInfo",
    "KaminoModelSize",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


@dataclass
class KaminoModelSize:
    """
    A container to hold the summary size of memory allocations and thread dimensions.

    Notes:
    - The sums are used for memory allocations.
    - The maximums are used to define 2D thread shapes: (num_worlds, max_of_max_XXX)
    - Where `XXX` is the maximum number of limits, contacts, unilaterals, or constraints in any world.

    Attributes:
        num_worlds (int):
            The number of worlds represented in the model.
        sum_of_num_bodies (int):
            The total number of bodies in the model across all worlds.
        max_of_num_bodies (int):
            The maximum number of bodies in any world.
        sum_of_num_joints (int):
            The total number of joints in the model across all worlds.
        max_of_num_joints (int):
            The maximum number of joints in any world.
        sum_of_num_passive_joints (int):
            The total number of passive joints in the model across all worlds.
        max_of_num_passive_joints (int):
            The maximum number of passive joints in any world.
        sum_of_num_actuated_joints (int):
            The total number of actuated joints in the model across all worlds.
        max_of_num_actuated_joints (int):
            The maximum number of actuated joints in any world.
        sum_of_num_collision_geoms (int):
            The total number of collision geometries in the model across all worlds.
        max_of_num_collision_geoms (int):
            The maximum number of collision geometries in any world.
        sum_of_num_physical_geoms (int):
            The total number of physical geometries in the model across all worlds.
        max_of_num_physical_geoms (int):
            The maximum number of physical geometries in any world.
        sum_of_num_material_pairs (int):
            The total number of material pairs in the model across all worlds.
        max_of_num_material_pairs (int):
            The maximum number of material pairs in any world.
        sum_of_num_body_dofs (int):
            The total number of body DoFs in the model across all worlds.
        max_of_num_body_dofs (int):
            The maximum number of body DoFs in any world.
        sum_of_num_joint_coords (int):
            The total number of joint coordinates in the model across all worlds.
        max_of_num_joint_coords (int):
            The maximum number of joint coordinates in any world.
        sum_of_num_joint_dofs (int):
            The total number of joint DoFs in the model across all worlds.
        max_of_num_joint_dofs (int):
            The maximum number of joint DoFs in any world.
        sum_of_num_passive_joint_coords (int):
            The total number of passive joint coordinates in the model across all worlds.
        max_of_num_passive_joint_coords (int):
            The maximum number of passive joint coordinates in any world.
        sum_of_num_passive_joint_dofs (int):
            The total number of passive joint DoFs in the model across all worlds.
        max_of_num_passive_joint_dofs (int):
            The maximum number of passive joint DoFs in any world.
        sum_of_num_actuated_joint_coords (int):
            The total number of actuated joint coordinates in the model across all worlds.
        max_of_num_actuated_joint_coords (int):
            The maximum number of actuated joint coordinates in any world.
        sum_of_num_actuated_joint_dofs (int):
            The total number of actuated joint DoFs in the model across all worlds.
        max_of_num_actuated_joint_dofs (int):
            The maximum number of actuated joint DoFs in any world.
        sum_of_num_joint_cts (int):
            The total number of joint constraints in the model across all worlds.
        max_of_num_joint_cts (int):
            The maximum number of joint constraints in any world.
        sum_of_max_limits (int):
            The total maximum number of limits allocated for the model across all worlds.
        max_of_max_limits (int):
            The maximum number of active limits of any world.
        sum_of_max_contacts (int):
            The total maximum number of contacts allocated for the model across all worlds.
        max_of_max_contacts (int):
            The maximum number of active contacts of any world.
        sum_of_max_unilaterals (int):
            The maximum number of active unilateral entities, i.e. joint-limits and contacts.
        max_of_max_unilaterals (int):
            The maximum number of active unilaterals of any world.
        sum_of_max_total_cts (int):
            The maximum number of active constraints.
        max_of_max_total_cts (int):
            The maximum number of active constraints of any world.
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

    sum_of_num_collision_geoms: int = 0
    """The total number of collision geometries in the model across all worlds."""

    max_of_num_collision_geoms: int = 0
    """The maximum number of collision geometries in any world."""

    sum_of_num_physical_geoms: int = 0
    """The total number of physical geometries in the model across all worlds."""

    max_of_num_physical_geoms: int = 0
    """The maximum number of physical geometries in any world."""

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
        """Returns a human-readable string representation of the ModelSize as a formatted table."""
        # List of (row title, sum attr, max attr)
        rows = [
            ("num_bodies", "sum_of_num_bodies", "max_of_num_bodies"),
            ("num_joints", "sum_of_num_joints", "max_of_num_joints"),
            ("num_passive_joints", "sum_of_num_passive_joints", "max_of_num_passive_joints"),
            ("num_actuated_joints", "sum_of_num_actuated_joints", "max_of_num_actuated_joints"),
            ("num_collision_geoms", "sum_of_num_collision_geoms", "max_of_num_collision_geoms"),
            ("num_physical_geoms", "sum_of_num_physical_geoms", "max_of_num_physical_geoms"),
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

        # Write ModelSize members as a formatted table
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


@dataclass
class KaminoModelInfo:
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

    num_collision_geoms: wp.array | None = None
    """
    The number of collision geometries in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_physical_geoms: wp.array | None = None
    """
    The number of physical geometries in each world.\n
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
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joints_offset: wp.array | None = None
    """
    The joint index offset of each world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    limits_offset: wp.array | None = None
    """
    The limit index offset of each world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    contacts_offset: wp.array | None = None
    """
    The contact index offset of world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    unilaterals_offset: wp.array | None = None
    """
    The index offset of the unilaterals (limits + contacts) block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # DoF Offsets
    ###

    body_dofs_offset: wp.array | None = None
    """
    The index offset of the body DoF block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_coords_offset: wp.array | None = None
    """
    The index offset of the joint coordinates block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_dofs_offset: wp.array | None = None
    """
    The index offset of the joint DoF block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_passive_coords_offset: wp.array | None = None
    """
    The index offset of the passive joint coordinates block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_passive_dofs_offset: wp.array | None = None
    """
    The index offset of the passive joint DoF block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_actuated_coords_offset: wp.array | None = None
    """
    The index offset of the actuated joint coordinates block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_actuated_dofs_offset: wp.array | None = None
    """
    The index offset of the actuated joint DoF block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Constraint Offsets
    ###

    joint_cts_offset: wp.array | None = None
    """
    The index offset of the joint constraints block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    limit_cts_offset: wp.array | None = None
    """
    The index offset of the limit constraints block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    contact_cts_offset: wp.array | None = None
    """
    The index offset of the contact constraints block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    unilateral_cts_offset: wp.array | None = None
    """
    The index offset of the unilateral constraints block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    total_cts_offset: wp.array | None = None
    """
    The index offset of the total constraints block of each world.\n
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


class KaminoModel:
    """
    A container to hold the time-invariant system model data.

    Attributes:
        device (Devicelike):
            The device on which the model data is allocated.
        requires_grad (bool):
            Whether the model requires gradients for its state. Defaults to `False`.
        size (ModelSize):
            Host-side cache of the model summary sizes.\n
            This is used for memory allocations and kernel thread dimensions.
        worlds (list[WorldDescriptor]):
            Host-side cache of the world descriptors.\n
            This is used to construct the model and for memory allocations.
        info (ModelInfo):
            The model info container holding the information and meta-data of the model.
        time (TimeModel):
            The time model container holding time-step of each world.
        gravity (GravityModel):
            The gravity model container holding the gravity configurations for each world.
        bodies (RigidBodiesModel):
            The rigid bodies model container holding all rigid body entities in the model.
        joints (JointsModel):
            The joints model container holding all joint entities in the model.
        cgeoms (CollisionGeometriesModel):
            The collision geometries model container holding all collision geometry entities in the model.
        pgeoms (GeometriesModel):
            The physical geometries model container holding all physical geometry entities in the model.
        mpairs (MaterialPairsModel):
            The material pairs model container holding all material pairs in the model.
    """

    def __init__(self, model: Model | None = None):
        """
        Create a new KaminoModel instance.

        TODO: about creating from an existing newton.Model instance.
        """

        self._model_base: Model | None = None
        """The base newton.Model instance from which this KaminoModel was created."""

        self.device: Devicelike = None
        """
        The device on which the model data is allocated.\n
        Defaults to `None`, indicating the default/preferred Warp device.
        """

        self.requires_grad: bool = False
        """Whether the model requires gradients for its state. Defaults to `False`."""

        self.size: KaminoModelSize = KaminoModelSize()
        """
        Host-side cache of the model summary sizes.\n
        This is used for memory allocations and kernel thread dimensions.
        """

        self.worlds: list[WorldDescriptor] = []
        """
        Host-side cache of the world descriptors.\n
        This is used to construct the model and for memory allocations.
        """

        self.info: KaminoModelInfo | None = None
        """The model info container holding the information and meta-data of the model."""

        self.time: TimeModel | None = None
        """The time model container holding time-step of each world."""

        self.gravity: GravityModel | None = None
        """The gravity model container holding the gravity configurations for each world."""

        self.bodies: RigidBodiesModel | None = None
        """The rigid bodies model container holding all rigid body entities in the model."""

        self.joints: JointsModel | None = None
        """The joints model container holding all joint entities in the model."""

        self.cgeoms: CollisionGeometriesModel | None = None
        """The collision geometries model container holding all collision geometry entities in the model."""

        self.pgeoms: GeometriesModel | None = None
        """The physical geometries model container holding all physical geometry entities in the model."""

        self.mpairs: MaterialPairsModel | None = None
        """The material pairs model container holding all material pairs in the model."""

        # If a base model is provided, finalize from it
        if model is not None:
            self.finalize_from_newton_model(model)

    def finalize_from_newton_model(self, model: Model):
        """
        Finalizes the KaminoModel from an existing newton.Model instance.
        """
        # Ensure the base model is valid
        if model is None:
            raise ValueError("Cannot finalize KaminoModel from a None newton.Model instance.")
        elif not isinstance(model, Model):
            raise TypeError("Cannot finalize KaminoModel from an invalid newton.Model instance.")

        # Store the base model
        self._model_base = model
        self.device = self._model_base.device
        self.requires_grad = self._model_base.requires_grad

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
        msg.info("body_bid_np: %s", body_bid_np)
        msg.info("joint_jid_np: %s", joint_jid_np)
        msg.info("shape_sid_np: %s\n", shape_sid_np)

        # Compute the number of entities per world
        num_bodies_np = _compute_num_entities_per_world(model.body_world, model.num_worlds)
        num_joints_np = _compute_num_entities_per_world(model.joint_world, model.num_worlds)
        num_shapes_np = _compute_num_entities_per_world(model.shape_world, model.num_worlds)
        msg.info("num_bodies_np: %s", num_bodies_np)
        msg.info("num_joints_np: %s", num_joints_np)
        msg.info("num_shapes_np: %s\n", num_shapes_np)

        # Compute body DoF and joint coord/DoF/constraint counts per world
        num_body_coords_np = num_bodies_np * 7
        num_body_dofs_np = num_bodies_np * 6
        num_joint_coords_np = np.zeros((model.num_worlds,), dtype=int)
        num_joint_dofs_np = np.zeros((model.num_worlds,), dtype=int)
        num_joint_cts_np = np.zeros((model.num_worlds,), dtype=int)
        joint_num_coords_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_dofs_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_cts_np = np.zeros((model.joint_count,), dtype=int)
        joint_wid_np = model.joint_world.numpy()
        joint_type_np = model.joint_type.numpy()
        for j in range(model.joint_count):
            wid_j = joint_wid_np[j]
            dof_type_j = newton_to_kamino_joint_dof_type(joint_type_np[j])
            ncoords_j = dof_type_j.num_coords
            ndofs_j = dof_type_j.num_dofs
            ncts_j = dof_type_j.num_cts
            num_joint_coords_np[wid_j] += ncoords_j
            num_joint_dofs_np[wid_j] += ndofs_j
            num_joint_cts_np[wid_j] += ncts_j
            joint_num_coords_np[j] = ncoords_j
            joint_num_dofs_np[j] = ndofs_j
            joint_num_cts_np[j] = ncts_j
        msg.info("num_body_coords_np: %s", num_body_coords_np)
        msg.info("num_body_dofs_np: %s", num_body_dofs_np)
        msg.info("num_joint_coords_np: %s", num_joint_coords_np)
        msg.info("num_joint_dofs_np: %s", num_joint_dofs_np)
        msg.info("num_joint_cts_np: %s\n", num_joint_cts_np)
        msg.info("joint_num_coords_np: %s\n", joint_num_coords_np)
        msg.info("joint_num_dofs_np: %s\n", joint_num_dofs_np)
        msg.info("joint_num_cts_np: %s\n", joint_num_cts_np)

        # Compute offsets per world
        body_offset_np = np.zeros((model.num_worlds,), dtype=int)
        body_dof_offset_np = np.zeros((model.num_worlds,), dtype=int)
        joint_offset_np = np.zeros((model.num_worlds,), dtype=int)
        joint_coord_offset_np = np.zeros((model.num_worlds,), dtype=int)
        joint_dof_offset_np = np.zeros((model.num_worlds,), dtype=int)
        joint_cts_offset_np = np.zeros((model.num_worlds,), dtype=int)
        for w in range(1, model.num_worlds):
            body_offset_np[w] = body_offset_np[w - 1] + num_bodies_np[w - 1]
            body_dof_offset_np[w] = body_dof_offset_np[w - 1] + num_body_dofs_np[w - 1]
            joint_offset_np[w] = joint_offset_np[w - 1] + num_joints_np[w - 1]
            joint_coord_offset_np[w] = joint_coord_offset_np[w - 1] + num_joint_coords_np[w - 1]
            joint_dof_offset_np[w] = joint_dof_offset_np[w - 1] + num_joint_dofs_np[w - 1]
            joint_cts_offset_np[w] = joint_cts_offset_np[w - 1] + num_joint_cts_np[w - 1]
        msg.info("body_offset_np: %s", body_offset_np)
        msg.info("body_dof_offset_np: %s", body_dof_offset_np)
        msg.info("joint_offset_np: %s", joint_offset_np)
        msg.info("joint_coord_offset_np: %s", joint_coord_offset_np)
        msg.info("joint_dof_offset_np: %s", joint_dof_offset_np)
        msg.info("joint_cts_offset_np: %s\n", joint_cts_offset_np)

        # Determine the base body and joint indices per world
        # TODO: Check for articulation
        # TODO: Check for root joint (i.e. body with no parent)
        base_body_idx_np = np.zeros((model.num_worlds,), dtype=int)
        base_joint_idx_np = np.zeros((model.num_worlds,), dtype=int)

        # TODO: Fall-back: first body and joint in the world
        for w in range(model.num_worlds):
            # Base body: first body in the world
            for b in range(model.body_count):
                if model.body_world[b] == w:
                    base_body_idx_np[w] = b
                    break
            # Base joint: first joint in the world
            for j in range(model.joint_count):
                if model.joint_world[j] == w:
                    base_joint_idx_np[w] = j
                    break
        msg.info("base_body_idx_np: %s", base_body_idx_np)
        msg.info("base_joint_idx_np: %s\n", base_joint_idx_np)

        # TODO: Construct the world descriptors from the newton.Model instance
        self.worlds: list[WorldDescriptor] = [None] * model.num_worlds
        for w in range(model.num_worlds):
            self.worlds[w] = WorldDescriptor(
                name=f"world_{w}",
                wid=w,
                num_bodies=num_bodies_np[w],
                num_joints=num_joints_np[w],
                num_passive_joints=0,
                num_actuated_joints=num_joints_np[w],
                num_collision_geoms=num_shapes_np[w],
                num_physical_geoms=num_shapes_np[w],
                num_materials=1,  # TODO: define default material
                num_body_coords=num_body_coords_np[w],
                num_body_dofs=num_body_dofs_np[w],
                num_joint_coords=num_joint_coords_np[w],
                num_joint_dofs=num_joint_dofs_np[w],
                num_passive_joint_coords=0,
                num_passive_joint_dofs=0,
                num_actuated_joint_coords=num_joint_coords_np[w],
                num_actuated_joint_dofs=num_joint_dofs_np[w],
                num_joint_cts=num_joint_cts_np[w],

                # TODO
                joint_coords=[],
                joint_dofs=[],
                joint_passive_coords=[],
                joint_passive_dofs=[],
                joint_actuated_coords=[],
                joint_actuated_dofs=[],
                joint_cts=[],

                # TODO
                bodies_idx_offset=0,
                joints_idx_offset=0,
                collision_geoms_idx_offset=0,
                physical_geoms_idx_offset=0,
                body_dofs_idx_offset=0,
                joint_coords_idx_offset=0,
                joint_dofs_idx_offset=0,
                passive_joint_coords_idx_offset=0,
                passive_joint_dofs_idx_offset=0,
                actuated_joint_coords_idx_offset=0,
                actuated_joint_dofs_idx_offset=0,
                joint_cts_idx_offset=0,

                # TODO
                body_names=[],
                body_uids=[],
                joint_names=[],
                joint_uids=[],
                collision_geom_names=[],
                collision_geom_uids=[],
                physical_geom_names=[],
                physical_geom_uids=[],
                material_names=[],
                material_uids=[],
                unary_joint_names=[],
                fixed_joint_names=[],
                passive_joint_names=[],
                actuated_joint_names=[],
                physical_geometry_layers=[],
                collision_geometry_layers=[],
                collision_geometry_max_contacts=[],

                # TODO
                base_body_idx=0,  # TODO
                base_joint_idx=0,  # TODO
                mass_min=0.0,  # TODO
                mass_max=0.0,  # TODO
                mass_total=0.0,  # TODO
                inertia_total=0.0,  # TODO
            )

        # Construct KaminoModelSize from the newton.Model instance
        self.size = KaminoModelSize(
            num_worlds=model.num_worlds,
            sum_of_num_bodies=num_bodies_np.sum(),
            max_of_num_bodies=num_bodies_np.max(),
            sum_of_num_joints=num_joints_np.sum(),
            max_of_num_joints=num_joints_np.max(),
            sum_of_num_passive_joints=0,
            max_of_num_passive_joints=0,
            sum_of_num_actuated_joints=num_joints_np.sum(),
            max_of_num_actuated_joints=num_joints_np.max(),
            sum_of_num_collision_geoms=num_shapes_np.sum(),
            max_of_num_collision_geoms=num_shapes_np.max(),
            sum_of_num_physical_geoms=num_shapes_np.sum(),
            max_of_num_physical_geoms=num_shapes_np.max(),
            sum_of_num_material_pairs=0,
            max_of_num_material_pairs=0,
            sum_of_num_body_dofs=num_body_dofs_np.sum(),
            max_of_num_body_dofs=num_body_dofs_np.max(),
            sum_of_num_joint_coords=num_joint_coords_np.sum(),
            max_of_num_joint_coords=num_joint_coords_np.max(),
            sum_of_num_joint_dofs=num_joint_dofs_np.sum(),
            max_of_num_joint_dofs=num_joint_dofs_np.max(),
            sum_of_num_passive_joint_coords=0,
            max_of_num_passive_joint_coords=0,
            sum_of_num_passive_joint_dofs=0,
            max_of_num_passive_joint_dofs=0,
            sum_of_num_actuated_joint_coords=num_joint_coords_np.sum(),
            max_of_num_actuated_joint_coords=num_joint_coords_np.max(),
            sum_of_num_actuated_joint_dofs=num_joint_dofs_np.sum(),
            max_of_num_actuated_joint_dofs=num_joint_dofs_np.max(),
            sum_of_num_joint_cts=num_joint_cts_np.sum(),
            max_of_num_joint_cts=num_joint_cts_np.max(),
            sum_of_max_limits=0,
            max_of_max_limits=0,
            sum_of_max_contacts=0,
            max_of_max_contacts=0,
            sum_of_max_unilaterals=0,
            max_of_max_unilaterals=0,
            sum_of_max_total_cts=num_joint_cts_np.sum(),
            max_of_max_total_cts=num_joint_cts_np.max(),
        )
        msg.info("KaminoModelSize:\n%s", self.size)

        # Construct the model entities from the newton.Model instance
        with wp.ScopedDevice(device=self.device):
            # Model info
            self.info = KaminoModelInfo(
                num_worlds=model.num_worlds,
                num_bodies=wp.array(num_bodies_np, dtype=int32),
                num_joints=wp.array(num_joints_np, dtype=int32),
                num_passive_joints=wp.zeros((model.num_worlds,), dtype=int32),
                num_actuated_joints=wp.array(num_joints_np, dtype=int32),
                num_collision_geoms=wp.array(num_shapes_np, dtype=int32),
                num_physical_geoms=wp.array(num_shapes_np, dtype=int32),
                max_limits=None,  # Created by Limits container
                max_contacts=None,  # Created by Contacts container
                num_body_dofs=wp.array(num_body_dofs_np, dtype=int32),
                num_joint_coords=wp.array(num_joint_coords_np, dtype=int32),
                num_joint_dofs=wp.array(num_joint_dofs_np, dtype=int32),
                num_passive_joint_coords=wp.zeros((model.num_worlds,), dtype=int32),
                num_passive_joint_dofs=wp.zeros((model.num_worlds,), dtype=int32),
                num_actuated_joint_coords=wp.array(num_joint_coords_np, dtype=int32),
                num_actuated_joint_dofs=wp.array(num_joint_dofs_np, dtype=int32),
                num_joint_cts=wp.array(num_joint_cts_np, dtype=int32),
                max_limit_cts=None,  # Created by Limits container
                max_contact_cts=None,  # Created by Contacts container
                max_total_cts=wp.array(num_joint_cts_np, dtype=int32),
                bodies_offset=wp.array(body_offset_np, dtype=int32),
                joints_offset=wp.array(joint_offset_np, dtype=int32),
                body_dofs_offset=wp.array(body_dof_offset_np, dtype=int32),
                joint_coords_offset=wp.array(joint_coord_offset_np, dtype=int32),
                joint_dofs_offset=wp.array(joint_dof_offset_np, dtype=int32),
                joint_passive_coords_offset=wp.full_like(wp.array(joint_coord_offset_np, dtype=int32), -1, dtype=int32),
                joint_passive_dofs_offset=wp.full_like(wp.array(joint_dof_offset_np, dtype=int32), -1, dtype=int32),
                joint_actuated_coords_offset=wp.array(joint_coord_offset_np, dtype=int32),
                joint_actuated_dofs_offset=wp.array(joint_dof_offset_np, dtype=int32),
                joint_cts_offset=wp.array(joint_cts_offset_np, dtype=int32),

                # TODO
                # base_body_index=wp.array(...),
                # base_joint_index=wp.array(...),
                # mass_min=wp.array(...),
                # mass_max=wp.array(...),
                # mass_total=wp.array(...),
                # inertia_total=wp.array(...),
            )

            # Time
            self.time = TimeModel(
                # dt=wp.array(...),
                # inv_dt=wp.array(...),
            )

            # Gravity
            self.gravity = GravityModel(
                # g_dir_acc=wp.array(...),
                # vector=wp.array(...),
            )

            # Rigid bodies
            self.bodies = RigidBodiesModel(
                num_bodies=model.body_count,
                wid=model.body_world,
                bid=wp.array(body_bid_np, dtype=int32),
                i_r_com_i=model.body_com,
                m_i=model.body_mass,
                inv_m_i=model.body_inv_mass,
                i_I_i=model.body_inertia,
                inv_i_I_i=model.body_inv_inertia,
                # TODO: Can we avoid the extra memory allocation here?
                q_i_0=wp.zeros_like(model.body_q),
                u_i_0=wp.zeros_like(model.body_qd),
            )

            # Joints
            self.joints = JointsModel(
                num_joints=model.joint_count,
                wid=model.joint_world,
                jid=wp.array(joint_jid_np, dtype=int32),
                dof_type=model.joint_type,
                act_type=wp.full_like(model.joint_type, JointActuationType.FORCE, dtype=int32),
                bid_B=model.joint_parent,
                bid_F=model.joint_child,
                # B_r_Bj=wp.array(...),  # From model.joint_X_p
                # F_r_Fj=wp.array(...),  # From model.joint_X_c
                # X_j=wp.array(...),  # From model.joint_X_p
                q_j_min=model.joint_limit_lower,
                q_j_max=model.joint_limit_upper,
                dq_j_max=model.joint_velocity_limit,
                tau_j_max=model.joint_effort_limit,
                q_j_0=model.joint_q,
                dq_j_0=model.joint_qd,
                num_coords=wp.array(joint_num_coords_np, dtype=int32),
                num_dofs=wp.array(joint_num_dofs_np, dtype=int32),
                num_cts=wp.array(joint_num_cts_np, dtype=int32),
                coords_offset=model.joint_q_start,
                dofs_offset=model.joint_qd_start,
                passive_coords_offset=wp.full_like(model.joint_q_start, -1, dtype=int32),
                passive_dofs_offset=wp.full_like(model.joint_qd_start, -1, dtype=int32),
                actuated_coords_offset=model.joint_q_start,
                actuated_dofs_offset=model.joint_qd_start,
                # cts_offset=wp.array(...),
            )

            # Collision geometries
            self.cgeoms = CollisionGeometriesModel(
                num_geoms=model.shape_count,
                wid=model.shape_world,
                gid=wp.array(shape_sid_np, dtype=int32),
                lid=wp.zeros(shape=model.shape_count, dtype=int32),  # TODO: No layers yet
                bid=model.shape_body,
                # sid=model.shape_type,
                ptr=model.shape_source_ptr,
                offset=model.shape_transform,
                # params=wp.array(...),  # From model.shape_scale
            )

        # Post-processing after construction
        # TODO: Transform initial body CoM state from body frame state
        # TODO: Convert limits from `JOINT_LIMIT_UNLIMITED` to `FLOAT_MAX/MIN` representation
        # TODO: Convert shape_scale to geom_params

    def data(
        self,
        unilateral_cts: bool = False,
        requires_grad: bool = False,
        device: Devicelike = None,
    ) -> KaminoData:
        """
        Creates a model data container with the initial state of the model entities.

        Parameters:
            unilateral_cts (`bool`, optional):
                Whether to include unilateral constraints (limits and contacts) in the model data. Defaults to `True`.
            requires_grad (`bool`, optional):
                Whether the model data should require gradients. Defaults to `False`.
            device (`Devicelike`, optional):
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
        ncg = self.size.sum_of_num_collision_geoms
        npg = self.size.sum_of_num_physical_geoms

        # Retrieve the joint coordinate, DoF and constraint counts
        njq = self.size.sum_of_num_joint_coords
        njd = self.size.sum_of_num_joint_dofs
        njc = self.size.sum_of_num_joint_cts

        # Construct the model data on the specified device
        with wp.ScopedDevice(device=device):
            # Create a new model data info with the total constraint
            # counts initialized to the joint constraints count
            info = KaminoDataInfo(
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

            # Construct the collision geometries state from the model's initial state
            cgeoms = GeometriesData(
                num_geoms=ncg,
                pose=wp.zeros(shape=ncg, dtype=transformf, requires_grad=requires_grad),
            )

            # Construct the physical geometries state from the model's initial state
            pgeoms = GeometriesData(
                num_geoms=npg,
                pose=wp.zeros(shape=npg, dtype=transformf, requires_grad=requires_grad),
            )

        # Assemble and return the new model data container
        return KaminoData(
            info=info,
            time=time,
            bodies=bodies,
            joints=joints,
            cgeoms=cgeoms,
            pgeoms=pgeoms,
        )

    def state(self, requires_grad: bool = False, device: Devicelike = None) -> KaminoState:
        """
        Creates state container initialized to the initial body state defined in the model.

        Parameters:
            requires_grad (`bool`, optional):
                Whether the state should require gradients. Defaults to `False`.
            device (`Devicelike`, optional):
                The device to create the state on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new state container with the initial state of the model entities on the specified device
        with wp.ScopedDevice(device=device):
            state = KaminoState(
                q_i=wp.clone(self.bodies.q_i_0, requires_grad=requires_grad),
                u_i=wp.clone(self.bodies.u_i_0, requires_grad=requires_grad),
                w_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                q_j=wp.clone(self.joints.q_j_ref, requires_grad=requires_grad),
                q_j_p=wp.clone(self.joints.q_j_ref, requires_grad=requires_grad),
                dq_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                lambda_j=wp.zeros(shape=self.size.sum_of_num_joint_cts, dtype=float32, requires_grad=requires_grad),
            )

        # Return the constructed state container
        return state

    def control(self, requires_grad: bool = False, device: Devicelike = None) -> KaminoControl:
        """
        Creates a control container with all values initialized to zeros.

        Parameters:
            requires_grad (`bool`, optional):
                Whether the control container should require gradients. Defaults to `False`.
            device (`Devicelike`, optional):
                The device to create the control container on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new control container on the specified device
        with wp.ScopedDevice(device=device):
            control = KaminoControl(
                tau_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad)
            )

        # Return the constructed control container
        return control
