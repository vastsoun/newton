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
Provides a unified interface for performing Collision Detection in Kamino.

TODO: More detailed description
"""

from enum import IntEnum

import warp as wp
from warp.context import Devicelike

from ..core.builder import ModelBuilder
from ..core.geometry import update_collision_geometries_state
from ..core.model import Model, ModelData
from ..core.types import override
from ..geometry.broadphase import nxn_broadphase
from ..geometry.collisions import Collisions
from ..geometry.contacts import Contacts
from ..geometry.primitives import primitive_narrowphase
from ..geometry.unified import BroadPhaseMode, CollisionPipelineUnifiedKamino

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class CollisionDetectorMode(IntEnum):
    """Defines the collision detection pipelines supported in Kamino."""

    PRIMITIVE = 0
    """
    Use the "fast" collision detection pipeline specialized for geometric
    primitives using an "explicit" broad-phase on pre-computed collision
    shape pairs and a narrow-phase using Newton's primitive colliders.
    """

    UNIFIED = 1
    """
    Use Newton's unified collision-detection pipeline using a configurable
    broad-phase that supports `NXN`, `SAP`, or `EXPLICIT` modes, and a
    unified GJK/MPR-based narrow-phase. This pipeline is more general and
    supports arbitrary collision geometries, including meshes and SDFs.
    """

    @override
    def __str__(self):
        """Returns a string representation of the collision detector mode."""
        return f"CollisionDetectorMode.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the collision detector mode."""
        return self.__str__()


###
# Interfaces
###


class CollisionDetector:
    """
    Provides a Collision Detection (CD) front-end for Kamino.

    This class is responsible for performing collision detection as well
    as managing the collision containers and their memory allocations.

    Supports two collision pipeline types:

    - `PRIMITIVE`: A fast collision pipeline with specialized for geometric
    primitives using an "explicit" broad-phase on pre-computed collision
    shape pairs and a narrow-phase using Newton's primitive colliders.

    - `UNIFIED`: Newton's unified collision-detection pipeline using a configurable
    broad-phase that supports `NXN`, `SAP`, or `EXPLICIT` modes, and a unified
    GJK/MPR-based narrow-phase. This pipeline is more general and supports arbitrary
    collision geometries, including meshes and SDFs.
    """

    def __init__(
        self,
        builder: ModelBuilder | None = None,
        default_max_contacts: int | None = None,
        mode: CollisionDetectorMode = CollisionDetectorMode.PRIMITIVE,
        broadphase: BroadPhaseMode = BroadPhaseMode.EXPLICIT,
        device: Devicelike = None,
    ):
        """
        Initialize the CollisionDetector.

        Args:
            builder(ModelBuilder):
                ModelBuilder instance containing the host-side model definition.
            default_max_contacts(int | None):
                Default maximum contacts per world (if not specified by builder).
            device(Devicelike):
                The target Warp device for allocation and execution.\n
                If `None`, uses the default device selected by Warp on the given platform.
            mode(CollisionDetectorMode):
                The type of collision-detection pipeline to use, either `PRIMITIVE` or `UNIFIED`.\n
                Defaults to `PRIMITIVE`.
            broadphase(BroadPhaseMode):
                The type of broad phase collision detection to use for the UNIFIED pipeline.\n
                May be `NXN`, `SAP`, or `EXPLICIT`, but (currently) ignored if using the `PRIMITIVE` pipeline.\n
                Defaults to `EXPLICIT`.
        """
        # Cache the target device
        self._device = device

        # Cache the collision detector configuration
        self._mode = mode
        self._broadphase = broadphase

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
            if self._mode == CollisionDetectorMode.PRIMITIVE:
                self.collisions = Collisions(builder=builder, device=device)

            # NOTE #2: contacts are the outputs of the narrow phase
            self.contacts = Contacts(
                capacity=world_max_contacts, default_max_contacts=default_max_contacts, device=device
            )

            # Cache the maximum number of contacts allocated for the model
            self._model_max_contacts: int = self.contacts.num_model_max_contacts
            self._world_max_contacts: list[int] = self.contacts.num_world_max_contacts

            # Initialize unified pipeline if requested
            if self._mode == CollisionDetectorMode.UNIFIED:
                self._unified_pipeline = CollisionPipelineUnifiedKamino(
                    builder=builder,
                    broadphase=self._broadphase,
                    device=device,
                )

    @property
    def device(self) -> Devicelike:
        """Returns the device on which the CollisionDetector data is allocated and executes."""
        return self._device

    @property
    def mode(self) -> CollisionDetectorMode:
        """Returns the type of collision pipeline being used by the CollisionDetector."""
        return self._mode

    @property
    def model_max_contacts(self) -> int:
        """Returns the total maximum number of contacts allocated for the model across all worlds."""
        return self._model_max_contacts

    @property
    def world_max_contacts(self) -> int:
        """Returns the maximum number of contacts allocated for each world in the model."""
        return self._world_max_contacts

    def collide(self, model: Model, data: ModelData):
        """
        Executes collision detection given a model and its associated data.

        This operation will use the `PRIMITIVE` or `UNIFIED` pipeline depending on
        the configuration set during the initialization of the CollisionDetector.

        Args:
            model (Model): The Model instance containing the collision geometries
            data (ModelData): The ModelData instance containing the state of the geometries
        """
        # Skip this operation if the model does not allocate contacts
        # TODO: change this to check if the model has any cgeoms
        if self._model_max_contacts <= 0:
            return

        if self._mode == CollisionDetectorMode.UNIFIED:
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
