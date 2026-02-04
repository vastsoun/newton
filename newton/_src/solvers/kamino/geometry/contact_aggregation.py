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
from .contacts import Contacts

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
# Classes
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
        # Import kernels here to avoid circular dependency
        from .contact_aggregation_kernels import (  # noqa: PLC0415
            aggregate_contact_forces_per_body,
            aggregate_contact_forces_per_geom,
            aggregate_ground_contact_flags_per_body,
        )

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
