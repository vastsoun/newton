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

import warp as wp
from warp.context import Devicelike

from .bodies import RigidBodiesData, RigidBodiesModel
from .control import Control
from .data import DataKamino, DataKaminoInfo
from .geometry import GeometriesData, GeometriesModel
from .gravity import GravityModel
from .joints import JointsData, JointsModel
from .materials import MaterialPairsModel, MaterialsModel
from .state import StateKamino
from .time import TimeData, TimeModel
from .types import float32, int32, mat33f, transformf, vec6f
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

    sum_of_num_dynamic_joints: int = 0
    """The total number of dynamic joints in the model across all worlds."""

    max_of_num_dynamic_joints: int = 0
    """The maximum number of dynamic joints in any world."""

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

    sum_of_num_dynamic_joint_cts: int = 0
    """The total number of dynamic joint constraints in the model across all worlds."""

    max_of_num_dynamic_joint_cts: int = 0
    """The maximum number of dynamic joint constraints in any world."""

    sum_of_num_kinematic_joint_cts: int = 0
    """The total number of kinematic joint constraints in the model across all worlds."""

    max_of_num_kinematic_joint_cts: int = 0
    """The maximum number of kinematic joint constraints in any world."""

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
    """The total maximum number of active constraints allocated for the model across all worlds."""

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
            ("num_dynamic_joints", "sum_of_num_dynamic_joints", "max_of_num_dynamic_joints"),
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
            ("num_dynamic_joint_cts", "sum_of_num_dynamic_joint_cts", "max_of_num_dynamic_joint_cts"),
            ("num_kinematic_joint_cts", "sum_of_num_kinematic_joint_cts", "max_of_num_kinematic_joint_cts"),
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

    num_dynamic_joints: wp.array | None = None
    """
    The number of dynamic joints in each world.\n
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

    # TODO: We could make this a vec2i to store dynamic
    # and kinematic joint constraint counts separately
    num_joint_cts: wp.array | None = None
    """
    The number of joint constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joint_dynamic_cts: wp.array | None = None
    """
    The number of dynamic joint constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joint_kinematic_cts: wp.array | None = None
    """
    The number of kinematic joint constraints of each world.\n
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
    Used to index into arrays that contain flattened joint coordinate-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_dofs_offset: wp.array | None = None
    """
    The index offset of the joint DoF block of each world.\n
    Used to index into arrays that contain flattened joint DoF-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_passive_coords_offset: wp.array | None = None
    """
    The index offset of the passive joint coordinates block of each world.\n
    Used to index into arrays that contain flattened passive joint coordinate-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_passive_dofs_offset: wp.array | None = None
    """
    The index offset of the passive joint DoF block of each world.\n
    Used to index into arrays that contain flattened passive joint DoF-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_actuated_coords_offset: wp.array | None = None
    """
    The index offset of the actuated joint coordinates block of each world.\n
    Used to index into arrays that contain flattened actuated joint coordinate-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_actuated_dofs_offset: wp.array | None = None
    """
    The index offset of the actuated joint DoF block of each world.\n
    Used to index into arrays that contain flattened actuated joint DoF-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Constraint Offsets
    ###

    joint_cts_offset: wp.array | None = None
    """
    The index offset of the joint constraints block of each world.\n
    Used to index into arrays that contain flattened and
    concatenated dynamic and kinematic joint constraint data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_dynamic_cts_offset: wp.array | None = None
    """
    The index offset of the dynamic joint constraints block of each world.\n
    Used to index into arrays that contain flattened dynamic joint constraint data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_kinematic_cts_offset: wp.array | None = None
    """
    The index offset of the kinematic joint constraints block of each world.\n
    Used to index into arrays that contain flattened kinematic joint constraint data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    # TODO: We could make this an array of vec5i and store the absolute
    #  startindex of each constraint group in the constraint array `lambda`:
    # - [0]: total_cts_offset
    # - [1]: joint_dynamic_cts_group_offset
    # - [2]: joint_kinematic_cts_group_offset
    # - [3]: limit_cts_group_offset
    # - [4]: contact_cts_group_offset
    # TODO: We could then provide helper functions to get the start-end of each block
    total_cts_offset: wp.array | None = None
    """
    The index offset of the total constraints block of each world.\n
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.\n

    This offset should be used together with:
    - joint_dynamic_cts_group_offset
    - joint_kinematic_cts_group_offset
    - limit_cts_group_offset
    - contact_cts_group_offset

    Example:
    ```
    # To index into the dynamic joint constraint reactions of world `w`:
    world_cts_start = model_info.total_cts_offset[w]
    local_joint_dynamic_cts_start = model_info.joint_dynamic_cts_group_offset[w]
    local_joint_kinematic_cts_start = model_info.joint_kinematic_cts_group_offset[w]
    local_limit_cts_start = model_info.limit_cts_group_offset[w]
    local_contact_cts_start = model_info.contact_cts_group_offset[w]

    # Now compute the starting index of each constraint group within the total constraints block of world `w`:
    world_dynamic_joint_cts_start = world_cts_start + local_joint_dynamic_cts_start
    world_kinematic_joint_cts_start = world_cts_start + local_joint_kinematic_cts_start
    world_limit_cts_start = world_cts_start + local_limit_cts_start
    world_contact_cts_start = world_cts_start + local_contact_cts_start
    ```

    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_dynamic_cts_group_offset: wp.array | None = None
    """
    The index offset of the dynamic joint constraints group within the constraints block of each world.\n
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_kinematic_cts_group_offset: wp.array | None = None
    """
    The index offset of the kinematic joint constraints group within the constraints block of each world.\n
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.\n
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

    device: Devicelike = None
    """The device on which the model data is allocated.\n
    Defaults to `None`, indicating the default/preferred Warp device.
    """

    requires_grad: bool = False
    """Whether the model requires gradients for its state. Defaults to `False`."""

    size: ModelKaminoSize | None = None
    """
    Host-side cache of the model summary sizes.\n
    This is used for memory allocations and kernel thread dimensions.
    """

    worlds: list[WorldDescriptor] | None = None
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
    """The geometries model container holding all geometry entities in the model."""

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

    def data(
        self,
        unilateral_cts: bool = False,
        joint_wrenches: bool = False,
        requires_grad: bool = False,
        device: Devicelike = None,
    ) -> DataKamino:
        """
        Creates a model data container with the initial state of the model entities.

        Parameters:
            unilateral_cts (`bool`, optional):
                Whether to include unilateral constraints (limits and contacts) in the model data. Defaults to `True`.
            joint_wrenches (`bool`, optional):
                Whether to include joint wrenches in the model data. Defaults to `False`.
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
        ng = self.size.sum_of_num_geoms

        # Retrieve the joint coordinate, DoF and constraint counts
        njcoords = self.size.sum_of_num_joint_coords
        njdofs = self.size.sum_of_num_joint_dofs
        njcts = self.size.sum_of_num_joint_cts
        njdyncts = self.size.sum_of_num_dynamic_joint_cts
        njkincts = self.size.sum_of_num_kinematic_joint_cts

        # Construct the model data on the specified device
        with wp.ScopedDevice(device=device):
            # Create a new model data info with the total constraint
            # counts initialized to the joint constraints count
            info = DataKaminoInfo(
                num_total_cts=wp.clone(self.info.num_joint_cts),
                num_limits=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                num_contacts=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                num_limit_cts=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                num_contact_cts=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                limit_cts_group_offset=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                contact_cts_group_offset=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
            )

            # Construct the time data with the initial step and time set to zero for all worlds
            time = TimeData(
                steps=wp.zeros(shape=nw, dtype=int32, requires_grad=requires_grad),
                time=wp.zeros(shape=nw, dtype=float32, requires_grad=requires_grad),
            )

            # Construct the rigid bodies data from the model's initial state
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

            # Construct the joints data from the model's initial state
            joints = JointsData(
                num_joints=nj,
                p_j=wp.zeros(shape=nj, dtype=transformf, requires_grad=requires_grad),
                q_j=wp.zeros(shape=njcoords, dtype=float32, requires_grad=requires_grad),
                q_j_p=wp.zeros(shape=njcoords, dtype=float32, requires_grad=requires_grad),
                dq_j=wp.zeros(shape=njdofs, dtype=float32, requires_grad=requires_grad),
                tau_j=wp.zeros(shape=njdofs, dtype=float32, requires_grad=requires_grad),
                r_j=wp.zeros(shape=njkincts, dtype=float32, requires_grad=requires_grad),
                dr_j=wp.zeros(shape=njkincts, dtype=float32, requires_grad=requires_grad),
                lambda_j=wp.zeros(shape=njcts, dtype=float32, requires_grad=requires_grad),
                m_j=wp.zeros(shape=njdyncts, dtype=float32, requires_grad=requires_grad),
                inv_m_j=wp.zeros(shape=njdyncts, dtype=float32, requires_grad=requires_grad),
                dq_b_j=wp.zeros(shape=njdyncts, dtype=float32, requires_grad=requires_grad),
                # TODO: Should we make these optional and only include them when implicit joints are present?
                q_j_ref=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                dq_j_ref=wp.clone(self.joints.dq_j_0, requires_grad=requires_grad),
                tau_j_ref=wp.zeros(shape=njdofs, dtype=float32, requires_grad=requires_grad),
                j_w_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad) if joint_wrenches else None,
                j_w_c_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad) if joint_wrenches else None,
                j_w_a_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad) if joint_wrenches else None,
                j_w_l_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad) if joint_wrenches else None,
            )

            # Construct the geometries data from the model's initial state
            geoms = GeometriesData(
                num_geoms=ng,
                pose=wp.zeros(shape=ng, dtype=transformf, requires_grad=requires_grad),
            )

        # Assemble and return the new data container
        return DataKamino(
            info=info,
            time=time,
            bodies=bodies,
            joints=joints,
            geoms=geoms,
        )

    def state(self, requires_grad: bool = False, device: Devicelike = None) -> StateKamino:
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

    def control(self, requires_grad: bool = False, device: Devicelike = None) -> Control:
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
            control = Control(
                tau_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                q_j_ref=wp.zeros(shape=self.size.sum_of_num_joint_coords, dtype=float32, requires_grad=requires_grad),
                dq_j_ref=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                tau_j_ref=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
            )

        # Return the constructed control container
        return control
