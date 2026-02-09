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
def aggregate_contact_forces_per_body(
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
    net_forces: wp.array3d(dtype=wp.float32),  # [num_worlds, max_bodies, 3]
    contact_flags: wp.array2d(dtype=int32),  # [num_worlds, max_bodies]
):
    """
    Aggregate contact forces and flags per body across all contacts.

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
        net_forces: Output array for net forces per body (world frame)
        contact_flags: Output array for contact flags per body
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
            wp.atomic_add(net_forces, world_idx, body_A_in_world, i, -force_world[i])
        wp.atomic_max(contact_flags, world_idx, body_A_in_world, int32(1))

    if global_body_B >= 0:
        body_B_in_world = model_body_bid[global_body_B]  # Convert to per-world index
        for i in range(3):
            wp.atomic_add(net_forces, world_idx, body_B_in_world, i, force_world[i])
        wp.atomic_max(contact_flags, world_idx, body_B_in_world, int32(1))


@wp.kernel
def aggregate_ground_contact_flags_per_body(
    # Input: Kamino ContactsData
    wid: wp.array(dtype=int32),  # world index per contact
    bid_AB: wp.array(dtype=vec2i),  # body pair per contact (global body indices)
    gid_AB: wp.array(dtype=vec2i),  # geometry pair per contact
    mode: wp.array(dtype=int32),  # contact mode
    world_active_contacts: wp.array(dtype=int32),  # contacts per world
    # Ground filter
    ground_geom_mask: wp.array(dtype=int32),  # 1 if geom is ground, 0 otherwise
    # Model data for global to per-world body ID conversion
    model_body_bid: wp.array(dtype=int32),  # Per-world body ID for each global body
    num_worlds: int,
    max_bodies_per_world: int,
    max_geoms_per_world: int,
    # Output
    ground_contact_flags: wp.array2d(dtype=int32),  # [num_worlds, max_bodies]
):
    """
    Identify which bodies are in contact with ground geometries.

    Each thread processes one contact. If either geometry in the contact
    pair is marked as ground, the corresponding non-ground body's ground
    contact flag is set.

    Args:
        wid: World index for each contact
        bid_AB: Body index pair (A, B) for each contact
        gid_AB: Geometry index pair (A, B) for each contact
        mode: Contact mode (INACTIVE, OPENING, STICKING, SLIDING)
        world_active_contacts: Number of active contacts per world
        ground_geom_mask: Mask indicating which geometries are ground (1=ground, 0=not)
        num_worlds: Total number of worlds
        max_bodies_per_world: Maximum number of bodies per world
        ground_contact_flags: Output array for ground contact flags per body
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

    # Check if either geometry is ground
    # Note: gid_AB contains global geometry indices, use directly
    geom_A_is_ground = ground_geom_mask[global_geom_A]
    geom_B_is_ground = ground_geom_mask[global_geom_B]

    # Set ground contact flag for non-ground body
    # Convert global body indices to per-world body indices for array indexing
    # Skip static bodies (bid < 0, e.g., ground plane)
    if geom_B_is_ground and global_body_A >= 0:
        # Body A is in contact with ground (geom B)
        body_A_in_world = model_body_bid[global_body_A]
        wp.atomic_max(ground_contact_flags, world_idx, body_A_in_world, int32(1))

    if geom_A_is_ground and global_body_B >= 0:
        # Body B is in contact with ground (geom A)
        body_B_in_world = model_body_bid[global_body_B]
        wp.atomic_max(ground_contact_flags, world_idx, body_B_in_world, int32(1))


@wp.kernel
def aggregate_contact_forces_per_geom(
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
    net_forces: wp.array3d(dtype=wp.float32),  # [num_worlds, max_geoms, 3]
    contact_flags: wp.array2d(dtype=int32),  # [num_worlds, max_geoms]
):
    """
    Aggregate contact forces and flags per geometry across all contacts.

    Similar to aggregate_contact_forces_per_body, but aggregates to geometry
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
        net_forces: Output array for net forces per geometry (world frame)
        contact_flags: Output array for contact flags per geometry
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
        wp.atomic_add(net_forces, world_idx, geom_A, i, force_world[i])
        wp.atomic_add(net_forces, world_idx, geom_B, i, -force_world[i])

    # Set contact flags for both geometries
    wp.atomic_max(contact_flags, world_idx, geom_A, int32(1))
    wp.atomic_max(contact_flags, world_idx, geom_B, int32(1))


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

    net_contact_forces_per_body: wp.array | None = None
    """Net contact forces per body (world frame). Shape: (num_worlds, max_bodies_per_world, 3)"""

    contact_flags_per_body: wp.array | None = None
    """Binary contact flags per body (any contact). Shape: (num_worlds, max_bodies_per_world)"""

    ground_contact_flags_per_body: wp.array | None = None
    """Ground contact flags per body (contact with ground geoms). Shape: (num_worlds, max_bodies_per_world)"""

    # === Per-Geom Detailed Data (for advanced RL) ===

    net_contact_forces_per_geom: wp.array | None = None
    """Net contact forces per geometry (world frame). Shape: (num_worlds, max_geoms_per_world, 3)"""

    contact_flags_per_geom: wp.array | None = None
    """Contact flags per geometry. Shape: (num_worlds, max_geoms_per_world)"""

    # === Contact Position/Normal Data (optional, for visualization) ===

    contact_positions_per_body: wp.array | None = None
    """Average contact position per body (world frame). Shape: (num_worlds, max_bodies_per_world, 3)"""

    contact_normals_per_body: wp.array | None = None
    """Average contact normal per body (world frame). Shape: (num_worlds, max_bodies_per_world, 3)"""

    num_contacts_per_body: wp.array | None = None
    """Number of contacts per body. Shape: (num_worlds, max_bodies_per_world)"""

    # === Ground Geometry Filter ===

    ground_geom_mask: wp.array | None = None
    """Pre-computed mask: which geom IDs are 'ground'. Shape: (total_num_geoms,)"""


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
        aggregation = ContactAggregation(model, contacts, ground_geom_ids=[0])
        aggregation.compute()  # Call after simulator.step()

        # Access via PyTorch tensors (zero-copy)
        net_forces = wp.to_torch(aggregation.net_forces_per_body)
        contact_flags = wp.to_torch(aggregation.contact_flags_per_body)
    """

    def __init__(
        self,
        model: Model,
        contacts: Contacts,
        ground_geom_ids: list[int] | None = None,
        device: Devicelike | None = None,
        enable_positions_normals: bool = False,
    ):
        """Initialize contact aggregation.

        Args:
            model: The Kamino model containing world/body/geom topology
            contacts: The Contacts container with per-contact data
            ground_geom_ids: List of geometry IDs considered as 'ground'. Defaults to [0].
            device: Device for computation. If None, uses model's device.
            enable_positions_normals: Whether to compute average contact positions and normals per body.
        """
        self._model = model
        self._contacts = contacts
        self._device = device if device is not None else model.device
        self._enable_positions_normals = enable_positions_normals

        # Allocate aggregation data
        self._data = self._allocate_data()

        # Setup ground geometry mask
        if ground_geom_ids is None:
            ground_geom_ids = [0]  # Default: first geometry is ground
        self._setup_ground_mask(ground_geom_ids)

    def _allocate_data(self) -> ContactAggregationData:
        """Allocate all necessary arrays for aggregation."""
        num_worlds = self._model.size.num_worlds
        max_bodies = self._model.size.max_of_num_bodies
        max_geoms = self._model.size.max_of_num_collision_geoms

        # Per-body aggregated data
        net_contact_forces_per_body = wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32, device=self._device)
        contact_flags_per_body = wp.zeros((num_worlds, max_bodies), dtype=wp.int32, device=self._device)
        ground_contact_flags_per_body = wp.zeros((num_worlds, max_bodies), dtype=wp.int32, device=self._device)

        # Per-geom detailed data
        net_contact_forces_per_geom = wp.zeros((num_worlds, max_geoms, 3), dtype=wp.float32, device=self._device)
        contact_flags_per_geom = wp.zeros((num_worlds, max_geoms), dtype=wp.int32, device=self._device)

        # Ground geometry mask (_setup_ground_mask)
        # Allocate based on actual number of geometries, not max_geoms * num_worlds
        num_geoms = self._model.cgeoms.num_geoms
        ground_geom_mask = wp.zeros(num_geoms, dtype=wp.int32, device=self._device)

        # Contact positions and normals
        contact_positions_per_body = None
        contact_normals_per_body = None
        num_contacts_per_body = None

        if self._enable_positions_normals:
            contact_positions_per_body = wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32, device=self._device)
            contact_normals_per_body = wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32, device=self._device)
            num_contacts_per_body = wp.zeros((num_worlds, max_bodies), dtype=wp.int32, device=self._device)

        return ContactAggregationData(
            net_contact_forces_per_body=net_contact_forces_per_body,
            contact_flags_per_body=contact_flags_per_body,
            ground_contact_flags_per_body=ground_contact_flags_per_body,
            net_contact_forces_per_geom=net_contact_forces_per_geom,
            contact_flags_per_geom=contact_flags_per_geom,
            ground_geom_mask=ground_geom_mask,
            contact_positions_per_body=contact_positions_per_body,
            contact_normals_per_body=contact_normals_per_body,
            num_contacts_per_body=num_contacts_per_body,
        )

    def _setup_ground_mask(self, ground_geom_ids: list[int]):
        """Mark which geometries are considered 'ground'.

        Args:
            ground_geom_ids: List of geometry IDs (per world) that should be marked as ground
        """
        # Get collision geometry data
        cgeoms = self._model.cgeoms
        num_geoms = cgeoms.num_geoms

        # Create mask on host and copy geometry per-world IDs to host
        mask_host = np.zeros(num_geoms, dtype=np.int32)
        gid_host = cgeoms.gid.numpy()  # Per-world geometry IDs

        # Mark geometries whose per-world ID is in ground_geom_ids
        for i in range(num_geoms):
            if gid_host[i] in ground_geom_ids:
                mask_host[i] = 1

        # Copy to device
        wp.copy(self._data.ground_geom_mask, wp.array(mask_host, dtype=wp.int32, device=self._device))

    def compute(self, skip_if_no_contacts: bool = False):
        """
        Compute aggregated contact data from current ContactsData.

        This method should be called after simulator.step() to update contact
        forces and flags. It launches GPU kernels to efficiently aggregate
        per-contact data into per-body and per-geom summaries.

        Args:
            skip_if_no_contacts: If True, check for zero contacts and return early.
                                 Set to False when using CUDA graphs to avoid GPU-to-CPU copies.
        """

        # Zero out previous results
        self._data.net_contact_forces_per_body.zero_()
        self._data.contact_flags_per_body.zero_()
        self._data.ground_contact_flags_per_body.zero_()
        self._data.net_contact_forces_per_geom.zero_()
        self._data.contact_flags_per_geom.zero_()

        if self._enable_positions_normals:
            self._data.contact_positions_per_body.zero_()
            self._data.contact_normals_per_body.zero_()
            self._data.num_contacts_per_body.zero_()

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

        # Launch aggregation kernel for per-body forces
        wp.launch(
            aggregate_contact_forces_per_body,
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
                self._data.net_contact_forces_per_body,
                self._data.contact_flags_per_body,
            ],
            device=self._device,
        )

        # Launch aggregation kernel for ground contact flags
        wp.launch(
            aggregate_ground_contact_flags_per_body,
            dim=contacts_data.model_max_contacts_host,
            inputs=[
                contacts_data.wid,
                contacts_data.bid_AB,
                contacts_data.gid_AB,
                contacts_data.mode,
                contacts_data.world_active_contacts,
                self._data.ground_geom_mask,
                self._model.bodies.bid,  # For global to per-world body ID conversion
                num_worlds,
                max_bodies,
                max_geoms,
            ],
            outputs=[
                self._data.ground_contact_flags_per_body,
            ],
            device=self._device,
        )

        # Launch aggregation kernel for per-geom forces
        wp.launch(
            aggregate_contact_forces_per_geom,
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
                self._data.net_contact_forces_per_geom,
                self._data.contact_flags_per_geom,
            ],
            device=self._device,
        )

    @property
    def net_forces_per_body(self) -> wp.array:
        """Net forces per body [num_worlds, max_bodies, 3]"""
        return self._data.net_contact_forces_per_body

    @property
    def contact_flags_per_body(self) -> wp.array:
        """Contact flags per body [num_worlds, max_bodies]"""
        return self._data.contact_flags_per_body

    @property
    def ground_contact_flags_per_body(self) -> wp.array:
        """Ground contact flags per body [num_worlds, max_bodies]"""
        return self._data.ground_contact_flags_per_body

    @property
    def net_forces_per_geom(self) -> wp.array:
        """Net forces per geom [num_worlds, max_geoms, 3]"""
        return self._data.net_contact_forces_per_geom

    @property
    def contact_flags_per_geom(self) -> wp.array:
        """Contact flags per geom [num_worlds, max_geoms]"""
        return self._data.contact_flags_per_geom

    @property
    def ground_geom_mask(self) -> wp.array:
        """Ground geometry mask [num_geoms]"""
        return self._data.ground_geom_mask
