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
Contact aggregation for RL applications.

This module provides functionality to aggregate per-contact data from Kamino's
ContactsData into per-body and per-geom summaries suitable for RL observations.
The aggregation is performed on GPU using efficient atomic operations.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp
from warp.context import Devicelike

from ..core.model import Model
from ..core.types import int32, quatf, vec2i, vec3f
from .contacts import ContactMode, Contacts

###
# Module interface
###

__all__ = [
    "ContactAggregation",
    "ContactAggregationData",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _aggregate_contact_force_per_body(
    # Input: Kamino ContactsData
    wid: wp.array(dtype=int32),  # world index per contact
    bid_AB: wp.array(dtype=vec2i),  # body pair per contact (global body indices)
    reaction: wp.array(dtype=vec3f),  # force in local contact frame
    frame: wp.array(dtype=quatf),  # contact frame (rotation quaternion)
    mode: wp.array(dtype=int32),  # contact mode
    world_active_contacts: wp.array(dtype=int32),  # contacts per world
    # Model data for global to per-world body ID conversion
    model_body_bid: wp.array(dtype=int32),  # Per-world body ID for each global body
    num_worlds: int,
    max_bodies_per_world: int,
    # Output: aggregated data
    net_force: wp.array3d(dtype=wp.float32),  # [num_worlds, max_bodies, 3]
    contact_flag: wp.array2d(dtype=int32),  # [num_worlds, max_bodies]
):
    """
    Aggregate contact force and flags per body across all contacts.

    Each thread processes one contact. Forces are transformed from local
    contact frame to world frame, then atomically accumulated to both
    bodies in the contact pair. Contact flags are set for both bodies.

    Args:
        wid: World index for each contact
        bid_AB: Body index pair (A, B) for each contact
        reaction: 3D contact force in local contact frame [normal, tangent1, tangent2]
        frame: Contact frame as rotation quaternion w.r.t world
        mode: Contact mode (INACTIVE, OPENING, STICKING, SLIDING)
        world_active_contacts: Number of active contacts per world
        num_worlds: Total number of worlds
        max_bodies_per_world: Maximum number of bodies per world
        net_force: Output array for net force per body (world frame)
        contact_flag: Output array for contact flag per body
    """
    contact_idx = wp.tid()

    # Calculate total active contacts across all worlds
    total_contacts = int32(0)
    for w in range(num_worlds):
        total_contacts += world_active_contacts[w]

    # Early exit if this thread is beyond active contacts
    if contact_idx >= total_contacts:
        return

    # Skip inactive contacts
    if mode[contact_idx] == ContactMode.INACTIVE:
        return

    # Get contact data
    world_idx = wid[contact_idx]
    body_pair = bid_AB[contact_idx]
    global_body_A = body_pair[0]  # Global body index
    global_body_B = body_pair[1]  # Global body index

    # Transform force from local contact frame to world frame
    force_local = reaction[contact_idx]
    contact_quat = frame[contact_idx]
    force_world = wp.quat_rotate(contact_quat, force_local)

    # Accumulate force to both bodies (equal and opposite)
    # Skip static bodies (bid < 0, e.g., ground plane)
    # Convert global body indices to per-world body indices for array indexing
    # Need to add each component separately for atomic operations on 3D arrays
    if global_body_A >= 0:
        body_A_in_world = model_body_bid[global_body_A]  # Convert to per-world index
        for i in range(3):
            wp.atomic_add(net_force, world_idx, body_A_in_world, i, -force_world[i])
        wp.atomic_max(contact_flag, world_idx, body_A_in_world, int32(1))

    if global_body_B >= 0:
        body_B_in_world = model_body_bid[global_body_B]  # Convert to per-world index
        for i in range(3):
            wp.atomic_add(net_force, world_idx, body_B_in_world, i, force_world[i])
        wp.atomic_max(contact_flag, world_idx, body_B_in_world, int32(1))


@wp.kernel
def _aggregate_ground_contact_flag_per_body(
    # Input: Kamino ContactsData
    wid: wp.array(dtype=int32),  # world index per contact
    bid_AB: wp.array(dtype=vec2i),  # body pair per contact (global body indices)
    gid_AB: wp.array(dtype=vec2i),  # geometry pair per contact
    mode: wp.array(dtype=int32),  # contact mode
    world_active_contacts: wp.array(dtype=int32),  # contacts per world
    # Static filter
    static_geom_mask: wp.array(dtype=int32),  # 1 if geom is static, 0 otherwise
    # Model data for global to per-world body ID conversion
    model_body_bid: wp.array(dtype=int32),  # Per-world body ID for each global body
    num_worlds: int,
    max_bodies_per_world: int,
    max_geoms_per_world: int,
    # Output
    static_contact_flag: wp.array2d(dtype=int32),  # [num_worlds, max_bodies]
):
    """
    Identify which bodies are in contact with static geometries.

    Each thread processes one contact. If either geometry in the contact
    pair is marked as static, the corresponding non-static body's static
    contact flag is set.

    Args:
        wid: World index for each contact
        bid_AB: Body index pair (A, B) for each contact
        gid_AB: Geometry index pair (A, B) for each contact
        mode: Contact mode (INACTIVE, OPENING, STICKING, SLIDING)
        world_active_contacts: Number of active contacts per world
        static_geom_mask: Mask indicating which geometries are static (1=static, 0=not)
        num_worlds: Total number of worlds
        max_bodies_per_world: Maximum number of bodies per world
        static_contact_flag: Output array for static contact flag per body
    """
    contact_idx = wp.tid()

    # Calculate total active contacts across all worlds
    total_contacts = int32(0)
    for w in range(num_worlds):
        total_contacts += world_active_contacts[w]

    # Early exit if this thread is beyond active contacts
    if contact_idx >= total_contacts:
        return

    # Skip inactive contacts
    if mode[contact_idx] == ContactMode.INACTIVE:
        return

    # Get contact data
    world_idx = wid[contact_idx]
    body_pair = bid_AB[contact_idx]
    geom_pair = gid_AB[contact_idx]

    global_body_A = body_pair[0]  # Global body index
    global_body_B = body_pair[1]  # Global body index
    global_geom_A = geom_pair[0]  # Global geometry index
    global_geom_B = geom_pair[1]  # Global geometry index

    # Check if either geometry is static
    # Note: gid_AB contains global geometry indices, use directly
    geom_A_is_static = static_geom_mask[global_geom_A]
    geom_B_is_static = static_geom_mask[global_geom_B]

    # Set static contact flag for non-static body
    # Convert global body indices to per-world body indices for array indexing
    # Skip static bodies (bid < 0, e.g., static plane)
    if geom_B_is_static and global_body_A >= 0:
        # Body A is in contact with static (geom B)
        body_A_in_world = model_body_bid[global_body_A]
        wp.atomic_max(static_contact_flag, world_idx, body_A_in_world, int32(1))

    if geom_A_is_static and global_body_B >= 0:
        # Body B is in contact with static (geom A)
        body_B_in_world = model_body_bid[global_body_B]
        wp.atomic_max(static_contact_flag, world_idx, body_B_in_world, int32(1))


@wp.kernel
def _aggregate_contact_force_per_geom(
    # Input: Kamino ContactsData
    wid: wp.array(dtype=int32),  # world index per contact
    gid_AB: wp.array(dtype=vec2i),  # geometry pair per contact
    reaction: wp.array(dtype=vec3f),  # force in local contact frame
    frame: wp.array(dtype=quatf),  # contact frame (rotation quaternion)
    mode: wp.array(dtype=int32),  # contact mode
    world_active_contacts: wp.array(dtype=int32),  # contacts per world
    num_worlds: int,
    max_geoms_per_world: int,
    # Output: aggregated data
    net_force: wp.array3d(dtype=wp.float32),  # [num_worlds, max_geoms, 3]
    contact_flag: wp.array2d(dtype=int32),  # [num_worlds, max_geoms]
):
    """
    Aggregate contact force and flags per geometry across all contacts.

    Similar to _aggregate_contact_force_per_body, but aggregates to geometry
    level instead of body level. Useful for detailed contact analysis in RL.

    Args:
        wid: World index for each contact
        gid_AB: Geometry index pair (A, B) for each contact
        reaction: 3D contact force in local contact frame [normal, tangent1, tangent2]
        frame: Contact frame as rotation quaternion w.r.t world
        mode: Contact mode (INACTIVE, OPENING, STICKING, SLIDING)
        world_active_contacts: Number of active contacts per world
        num_worlds: Total number of worlds
        max_geoms_per_world: Maximum number of geometries per world
        net_force: Output array for net force per geometry (world frame)
        contact_flag: Output array for contact flag per geometry
    """
    contact_idx = wp.tid()

    # Calculate total active contacts across all worlds
    total_contacts = int32(0)
    for w in range(num_worlds):
        total_contacts += world_active_contacts[w]

    # Early exit if this thread is beyond active contacts
    if contact_idx >= total_contacts:
        return

    # Skip inactive contacts
    if mode[contact_idx] == ContactMode.INACTIVE:
        return

    # Get contact data
    world_idx = wid[contact_idx]
    geom_pair = gid_AB[contact_idx]
    geom_A = geom_pair[0]
    geom_B = geom_pair[1]

    # Transform force from local contact frame to world frame
    force_local = reaction[contact_idx]
    contact_quat = frame[contact_idx]
    force_world = wp.quat_rotate(contact_quat, force_local)

    # Accumulate force to both geometries (equal and opposite)
    # Need to add each component separately for atomic operations on 3D arrays
    for i in range(3):
        wp.atomic_add(net_force, world_idx, geom_A, i, force_world[i])
        wp.atomic_add(net_force, world_idx, geom_B, i, -force_world[i])

    # Set contact flag for both geometries
    wp.atomic_max(contact_flag, world_idx, geom_A, int32(1))
    wp.atomic_max(contact_flag, world_idx, geom_B, int32(1))


###
# Types
###


@dataclass
class ContactAggregationData:
    """
    Pre-allocated arrays for aggregating contact data per world and body.
    Designed for efficient GPU computation and zero-copy PyTorch access.
    """

    # === Per-Body Aggregated Data (for RL interface) ===

    body_net_contact_force: wp.array | None = None
    """Net contact force per body (world frame). Shape: (num_worlds, max_bodies_per_world, 3)"""

    body_contact_flag: wp.array | None = None
    """Binary contact flag per body (any contact). Shape: (num_worlds, max_bodies_per_world)"""

    body_static_contact_flag: wp.array | None = None
    """Static contact flag per body (contact with static geoms). Shape: (num_worlds, max_bodies_per_world)"""

    # === Per-Geom Detailed Data (for advanced RL) ===

    geom_net_contact_force: wp.array | None = None
    """Net contact force per geometry (world frame). Shape: (num_worlds, max_geoms_per_world, 3)"""

    geom_contact_flag: wp.array | None = None
    """Contact flags per geometry. Shape: (num_worlds, max_geoms_per_world)"""

    # === Contact Position/Normal Data (optional, for visualization) ===

    body_contact_position: wp.array | None = None
    """Average contact position per body (world frame). Shape: (num_worlds, max_bodies_per_world, 3)"""

    body_contact_normal: wp.array | None = None
    """Average contact normal per body (world frame). Shape: (num_worlds, max_bodies_per_world, 3)"""

    body_num_contacts: wp.array | None = None
    """Number of contacts per body. Shape: (num_worlds, max_bodies_per_world)"""

    # === Static Geometry Filter ===

    static_geom_mask: wp.array | None = None
    """Pre-computed mask: which geom IDs are 'static'. Shape: (total_num_geoms,)"""


###
# Interfaces
###


class ContactAggregation:
    """
    High-level interface for aggregating Kamino contact data for RL.

    This class efficiently aggregates per-contact data from Kamino's ContactsData
    into per-body and per-geom summaries suitable for RL observations. All computation
    is performed on GPU using atomic operations for efficiency.

    Usage:
        aggregation = ContactAggregation(model, contacts, static_geom_ids=[0])
        aggregation.compute()  # Call after simulator.step()

        # Access via PyTorch tensors (zero-copy)
        net_force = wp.to_torch(aggregation.body_net_force)
        contact_flag = wp.to_torch(aggregation.body_contact_flag)
    """

    def __init__(
        self,
        model: Model | None = None,
        contacts: Contacts | None = None,
        static_geom_ids: list[int] | None = None,
        device: Devicelike | None = None,
        enable_positions_normals: bool = False,
    ):
        """Initialize contact aggregation.

        Args:
            model (Model | None): The model container describing the system to be simulated.
                If None, call ``finalize()`` later.
            contacts (Contacts | None): The Contacts container with per-contact data.
                If None, call ``finalize()`` later.
            static_geom_ids: List of geometry IDs considered as 'static'. Defaults to [0].
            device: Device for computation. If None, uses model's device.
            enable_positions_normals: Whether to compute average contact positions and normals per body.
        """
        # Cache the device
        self._device: Devicelike | None = device

        # Forward declarations
        self._model: Model | None = None
        self._contacts: Contacts | None = None
        self._data: ContactAggregationData | None = None
        self._enable_positions_normals: bool = enable_positions_normals

        # Proceed with memory allocations if model and contacts are provided
        if model is not None and contacts is not None:
            self.finalize(
                model=model,
                contacts=contacts,
                static_geom_ids=static_geom_ids,
                device=device,
            )

    def finalize(
        self,
        model: Model,
        contacts: Contacts,
        static_geom_ids: list[int] | None = None,
        device: Devicelike | None = None,
    ) -> None:
        """Finalizes memory allocations for the contact aggregation data.

        Args:
            model (Model): The model container describing the system to be simulated.
            contacts (Contacts): The Contacts container with per-contact data.
            static_geom_ids (list[int] | None): List of geometry IDs considered as 'static'. Defaults to [0].
            device (Devicelike | None): Device for computation. If None, uses model's device.
        """
        # Override the device if specified
        if device is not None:
            self._device = device
        if self._device is None:
            self._device = model.device

        self._model = model
        self._contacts = contacts

        # Read dimensions from the model
        num_worlds = model.size.num_worlds
        max_bodies = model.size.max_of_num_bodies
        max_geoms = model.size.max_of_num_collision_geoms
        num_geoms = model.cgeoms.num_geoms

        # Per-body aggregated data
        body_net_contact_force = wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32, device=self._device)
        body_contact_flag = wp.zeros((num_worlds, max_bodies), dtype=wp.int32, device=self._device)
        body_static_contact_flag = wp.zeros((num_worlds, max_bodies), dtype=wp.int32, device=self._device)

        # Per-geom detailed data
        geom_net_contact_force = wp.zeros((num_worlds, max_geoms, 3), dtype=wp.float32, device=self._device)
        geom_contact_flag = wp.zeros((num_worlds, max_geoms), dtype=wp.int32, device=self._device)

        # Static geometry mask
        static_geom_mask = wp.zeros(num_geoms, dtype=wp.int32, device=self._device)

        # Contact positions and normals
        body_contact_position = None
        body_contact_normal = None
        body_num_contacts = None

        if self._enable_positions_normals:
            body_contact_position = wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32, device=self._device)
            body_contact_normal = wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32, device=self._device)
            body_num_contacts = wp.zeros((num_worlds, max_bodies), dtype=wp.int32, device=self._device)

        self._data = ContactAggregationData(
            body_net_contact_force=body_net_contact_force,
            body_contact_flag=body_contact_flag,
            body_static_contact_flag=body_static_contact_flag,
            geom_net_contact_force=geom_net_contact_force,
            geom_contact_flag=geom_contact_flag,
            static_geom_mask=static_geom_mask,
            body_contact_position=body_contact_position,
            body_contact_normal=body_contact_normal,
            body_num_contacts=body_num_contacts,
        )

        # Setup static geometry mask
        if static_geom_ids is None:
            static_geom_ids = [0]  # Default: first geometry is static
        self._setup_static_mask(static_geom_ids)

    def _setup_static_mask(self, static_geom_ids: list[int]):
        """Mark which geometries are considered 'static'.

        Args:
            static_geom_ids: List of geometry IDs (per world) that should be marked as static
        """
        # Get collision geometry data
        cgeoms = self._model.cgeoms
        num_geoms = cgeoms.num_geoms

        # Create mask on host and copy geometry per-world IDs to host
        mask_host = np.zeros(num_geoms, dtype=np.int32)
        gid_host = cgeoms.gid.numpy()  # Per-world geometry IDs

        # Mark geometries whose per-world ID is in static_geom_ids
        for i in range(num_geoms):
            if gid_host[i] in static_geom_ids:
                mask_host[i] = 1

        # Copy to device
        wp.copy(self._data.static_geom_mask, wp.array(mask_host, dtype=wp.int32, device=self._device))

    def compute(self, skip_if_no_contacts: bool = False):
        """
        Compute aggregated contact data from current ContactsData.

        This method should be called after simulator.step() to update contact
        force and flags. It launches GPU kernels to efficiently aggregate
        per-contact data into per-body and per-geom summaries.

        Args:
            skip_if_no_contacts: If True, check for zero contacts and return early.
                                 Set to False when using CUDA graphs to avoid GPU-to-CPU copies.
        """

        # Zero out previous results
        self._data.body_net_contact_force.zero_()
        self._data.body_contact_flag.zero_()
        self._data.body_static_contact_flag.zero_()
        self._data.geom_net_contact_force.zero_()
        self._data.geom_contact_flag.zero_()

        if self._enable_positions_normals:
            self._data.body_contact_position.zero_()
            self._data.body_contact_normal.zero_()
            self._data.body_num_contacts.zero_()

        # Get contact data
        contacts_data = self._contacts.data

        # Optionally check if there are any active contacts
        if skip_if_no_contacts:
            total_active = contacts_data.model_active_contacts.numpy()[0]
            if total_active == 0:
                return  # No contacts, nothing to aggregate

        # Get model dimensions
        num_worlds = self._model.size.num_worlds
        max_bodies = self._model.size.max_of_num_bodies
        max_geoms = self._model.size.max_of_num_collision_geoms

        # Launch aggregation kernel for per-body force
        wp.launch(
            _aggregate_contact_force_per_body,
            dim=contacts_data.model_max_contacts_host,
            inputs=[
                contacts_data.wid,
                contacts_data.bid_AB,
                contacts_data.reaction,
                contacts_data.frame,
                contacts_data.mode,
                contacts_data.world_active_contacts,
                self._model.bodies.bid,  # For global to per-world body ID conversion
                num_worlds,
                max_bodies,
            ],
            outputs=[
                self._data.body_net_contact_force,
                self._data.body_contact_flag,
            ],
            device=self._device,
        )

        # Launch aggregation kernel for static contact flag
        wp.launch(
            _aggregate_ground_contact_flag_per_body,
            dim=contacts_data.model_max_contacts_host,
            inputs=[
                contacts_data.wid,
                contacts_data.bid_AB,
                contacts_data.gid_AB,
                contacts_data.mode,
                contacts_data.world_active_contacts,
                self._data.static_geom_mask,
                self._model.bodies.bid,  # For global to per-world body ID conversion
                num_worlds,
                max_bodies,
                max_geoms,
            ],
            outputs=[
                self._data.body_static_contact_flag,
            ],
            device=self._device,
        )

        # Launch aggregation kernel for per-geom force
        wp.launch(
            _aggregate_contact_force_per_geom,
            dim=contacts_data.model_max_contacts_host,
            inputs=[
                contacts_data.wid,
                contacts_data.gid_AB,
                contacts_data.reaction,
                contacts_data.frame,
                contacts_data.mode,
                contacts_data.world_active_contacts,
                num_worlds,
                max_geoms,
            ],
            outputs=[
                self._data.geom_net_contact_force,
                self._data.geom_contact_flag,
            ],
            device=self._device,
        )

    @property
    def body_net_force(self) -> wp.array:
        """Net force per body [num_worlds, max_bodies, 3]"""
        return self._data.body_net_contact_force

    @property
    def body_contact_flag(self) -> wp.array:
        """Contact flags per body [num_worlds, max_bodies]"""
        return self._data.body_contact_flag

    @property
    def body_static_contact_flag(self) -> wp.array:
        """Static contact flag per body [num_worlds, max_bodies]"""
        return self._data.body_static_contact_flag

    @property
    def geom_net_force(self) -> wp.array:
        """Net force per geom [num_worlds, max_geoms, 3]"""
        return self._data.geom_net_contact_force

    @property
    def geom_contact_flag(self) -> wp.array:
        """Contact flags per geom [num_worlds, max_geoms]"""
        return self._data.geom_contact_flag

    @property
    def static_geom_mask(self) -> wp.array:
        """Static geometry mask [num_geoms]"""
        return self._data.static_geom_mask
