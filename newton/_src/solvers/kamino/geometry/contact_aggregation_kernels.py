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
GPU kernels for contact aggregation.

This module contains Warp kernels for efficiently aggregating per-contact data
into per-body and per-geom summaries on GPU using atomic operations.
"""

import warp as wp

from ..core.types import int32, quatf, vec2i, vec3f
from .contacts import ContactMode

###
# Module interface
###

__all__ = [
    "aggregate_contact_forces_per_body",
    "aggregate_contact_forces_per_geom",
    "aggregate_ground_contact_flags_per_body",
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
