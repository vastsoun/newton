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
KAMINO: Collision Detector Interface
"""

from __future__ import annotations

from enum import Enum

import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.geometry import update_collision_geometries_state
from newton._src.solvers.kamino.core.model import (
    Model,
    ModelData,  # TODO: change to state.State
)
from newton._src.solvers.kamino.geometry.broadphase import nxn_broadphase
from newton._src.solvers.kamino.geometry.collisions import Collisions
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.primitives import primitive_narrowphase

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Collision Pipeline Type Enum
###


class CollisionPipelineType(Enum):
    """Type of collision pipeline to use for Kamino collision detection."""

    PRIMITIVE = "primitive"
    """
    Use Kamino's primitive collision pipeline with custom broad phase (NxN) and
    narrow phase kernels. This is the original implementation with hand-coded
    collision functions for primitive shape pairs.
    """

    UNIFIED = "unified"
    """
    Use the unified collision pipeline that shares Newton's broad phase (NXN, SAP,
    or EXPLICIT) and narrow phase (GJK/MPR via NarrowPhase class). This provides
    access to more shape types and mesh collision support.
    """


###
# Collision Detector class
###


class CollisionDetector:
    """
    Collision Detection (CD) front-end interface.

    This class is responsible for performing collision detection as well
    as managing the collision containers and their memory allocations.

    Supports two collision pipeline types:
    - PRIMITIVE: Original Kamino pipeline with custom collision kernels
    - UNIFIED: Newton's unified pipeline with GJK/MPR narrow phase
    """

    def __init__(
        self,
        builder: ModelBuilder | None = None,
        default_max_contacts: int | None = None,
        device: Devicelike = None,
        pipeline_type: CollisionPipelineType = CollisionPipelineType.PRIMITIVE,
        broad_phase_mode: str = "explicit",
    ):
        """
        Initialize the CollisionDetector.

        Args:
            builder: ModelBuilder instance containing the model definition
            default_max_contacts: Default maximum contacts per world (if not specified by builder)
            device: Device to allocate buffers on
            pipeline_type: Type of collision pipeline to use (PRIMITIVE or UNIFIED)
            broad_phase_mode: Broad phase mode for UNIFIED pipeline ("nxn", "sap", or "explicit")
        """
        # Cache the target device
        self._device = device
        self._pipeline_type = pipeline_type

        # Declare the collisions and contacts containers
        self.collisions: Collisions | None = None
        self.contacts: Contacts | None = None

        # Unified pipeline (only created if using UNIFIED mode)
        self._unified_pipeline = None

        # Declare the maximum number of contacts allocation caches
        self._model_max_contacts: int = 0
        self._world_max_contacts: list[int] = [0]

        # Retrieve the required contact capacity required by the model
        model_max_contacts, world_max_contacts = builder.required_contact_capacity

        # Allocate the collisions and contacts containers if the model requires them (indicated by >= 0)
        if model_max_contacts >= 0:
            # NOTE #1: collisions are the inputs/outputs of the broad phase (for PRIMITIVE pipeline)
            if pipeline_type == CollisionPipelineType.PRIMITIVE:
                self.collisions = Collisions(builder=builder, device=device)

            # NOTE #2: contacts are the outputs of the narrow phase
            self.contacts = Contacts(
                capacity=world_max_contacts, default_max_contacts=default_max_contacts, device=device
            )

            # Cache the maximum number of contacts allocated for the model
            self._model_max_contacts: int = self.contacts.num_model_max_contacts
            self._world_max_contacts: list[int] = self.contacts.num_world_max_contacts

            # Initialize unified pipeline if requested
            if pipeline_type == CollisionPipelineType.UNIFIED:
                from newton._src.solvers.kamino.geometry.collision_pipeline_unified import (
                    KaminoBroadPhaseMode,
                    KaminoCollisionPipelineUnified,
                )

                # Map string to enum
                broad_phase_map = {
                    "nxn": KaminoBroadPhaseMode.NXN,
                    "sap": KaminoBroadPhaseMode.SAP,
                    "explicit": KaminoBroadPhaseMode.EXPLICIT,
                }
                bp_mode = broad_phase_map.get(broad_phase_mode.lower(), KaminoBroadPhaseMode.EXPLICIT)

                self._unified_pipeline = KaminoCollisionPipelineUnified(
                    builder=builder,
                    broad_phase_mode=bp_mode,
                    device=device,
                )

    @property
    def pipeline_type(self) -> CollisionPipelineType:
        """The type of collision pipeline being used."""
        return self._pipeline_type

    @property
    def model_max_contacts(self) -> int:
        """
        The total maximum number of contacts allocated for the model across all worlds.
        """
        return self._model_max_contacts

    @property
    def world_max_contacts(self) -> int:
        """
        The maximum number of contacts allocated for each world in the model.
        """
        return self._world_max_contacts

    def collide(self, model: Model, data: ModelData):  # TODO: change to state.State
        """
        Perform collision detection for the a model with the specific state.

        Uses either the PRIMITIVE or UNIFIED pipeline depending on the configuration.
        """
        # Skip this operation if the model does not allocate contacts
        # TODO: change this to check if the model has any cgeoms
        if self._model_max_contacts <= 0:
            return

        if self._pipeline_type == CollisionPipelineType.UNIFIED:
            # Use the unified pipeline
            self._unified_pipeline.collide(model, data, self.contacts)
        else:
            # Use the primitive pipeline (original behavior)
            # Clear all current collision pairs and contacts
            self.collisions.clear()
            self.contacts.clear()

            # Update geometries states from the states of the bodies
            update_collision_geometries_state(data.bodies.q_i, model.cgeoms, data.cgeoms)

            # Perform the broad-phase collision detection to generate collision pairs
            nxn_broadphase(model.cgeoms, data.cgeoms, self.collisions.cmodel, self.collisions.cdata)

            # Perform the narrow-phase collision detection to generate active contacts
            primitive_narrowphase(model, data, self.collisions, self.contacts)
