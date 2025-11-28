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

Usage example:

    # Create a model builder
    builder = ModelBuilder()
    # ... add bodies and collision geometries to the builder ...

    # Create a collision detector with desired settings
    settings = CollisionDetectorSettings(
        pipeline=CollisionPipelineType.PRIMITIVE,
        broadphase=BroadPhaseMode.EXPLICIT,
        bvtype=BoundingVolumeType.AABB,
    )

    # Create the collision detector
    detector = CollisionDetector(builder=builder, settings=settings, device="cuda:0")
"""

from dataclasses import dataclass
from enum import IntEnum

import warp as wp
from warp.context import Devicelike

from ..core.builder import ModelBuilder
from ..core.model import Model, ModelData
from ..core.types import override
from ..geometry.contacts import Contacts
from ..geometry.primitive import BoundingVolumeType, CollisionPipelinePrimitive
from ..geometry.unified import BroadPhaseMode, CollisionPipelineUnifiedKamino

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class CollisionPipelineType(IntEnum):
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


@dataclass
class CollisionDetectorSettings:
    """Defines the settings for configuring a CollisionDetector."""

    max_contacts: int | None = None
    """
    The per-world maximum contacts to allocate.\n
    If specified as integer value, it will override the
    maximum contacts per world specified by the model.\n
    Defaults to `None`.
    """

    pipeline: CollisionPipelineType = CollisionPipelineType.PRIMITIVE
    """
    The type of collision-detection pipeline to use, either `PRIMITIVE` or `UNIFIED`.\n
    Defaults to `PRIMITIVE`.
    """

    broadphase: BroadPhaseMode = BroadPhaseMode.EXPLICIT
    """
    The broad-phase collision-detection to use (`NXN`, `SAP`, or `EXPLICIT`).\n
    Defaults to `EXPLICIT`.
    """

    bvtype: BoundingVolumeType = BoundingVolumeType.AABB
    """
    The type of bounding volume to use in the broad-phase.\n
    Defaults to `AABB`.
    """


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
        settings: CollisionDetectorSettings | None = None,
        device: Devicelike = None,
    ):
        """
        Initialize the CollisionDetector.

        Args:
            builder(ModelBuilder):
                ModelBuilder instance containing the host-side model definition.
            device(Devicelike):
                The target Warp device for allocation and execution.\n
                If `None`, uses the default device selected by Warp on the given platform.

        """
        # Cache the target device
        self._device: Devicelike = device

        # Cache the collision detector settings
        self._settings: CollisionDetectorSettings | None = settings

        # Declare the contacts container
        self._contacts: Contacts | None = None

        # Declare the collision detection pipelines
        self._unified_pipeline: CollisionPipelineUnifiedKamino | None = None
        self._primitive_pipeline: CollisionPipelinePrimitive | None = None

        # Declare and initialize the caches of contacts allocation sizes
        self._model_max_contacts: int = 0
        self._world_max_contacts: list[int] = [0]

        # Finalize the collision detector if a builder is provided
        if builder is not None:
            self.finalize(builder=builder, settings=settings, device=device)

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike:
        """Returns the device on which the CollisionDetector data is allocated and executes."""
        return self._device

    @property
    def settings(self) -> CollisionDetectorSettings:
        """Returns the settings used to configure the CollisionDetector."""
        return self._settings

    @property
    def model_max_contacts(self) -> int:
        """Returns the total maximum number of contacts allocated for the model across all worlds."""
        return self._model_max_contacts

    @property
    def world_max_contacts(self) -> list[int]:
        """Returns the maximum number of contacts allocated for each world."""
        return self._world_max_contacts

    @property
    def contacts(self) -> Contacts:
        """Returns the Contacts container managed by the CollisionDetector."""
        return self._contacts

    ###
    # Operations
    ###

    def finalize(
        self,
        builder: ModelBuilder,
        settings: CollisionDetectorSettings | None = None,
        device: Devicelike = None,
    ):
        """
        Allocates CollisionDetector data on the target device.

        Args:
            builder(ModelBuilder):
                ModelBuilder instance containing the host-side model definition.
            settings(CollisionDetectorSettings):
                Settings to configure the CollisionDetector.\n
                If `None`, uses default settings.
            device(Devicelike):
                The target Warp device for allocation and execution.\n
                If `None`, uses the default device selected by Warp on the given platform.
        """
        # Check that the builder is valid
        if builder is None:
            raise ValueError("Cannot finalize CollisionDetector: builder is None")
        if not isinstance(builder, ModelBuilder):
            raise TypeError(f"Cannot finalize CollisionDetector: expected ModelBuilder, got {type(builder)}")

        # Override the settings if specified
        if settings is not None:
            self._settings = settings

        # If no settings were configured, use default settings
        if self._settings is None:
            self._settings = CollisionDetectorSettings()

        # Override the device if specified
        if device is not None:
            self._device = device

        # Retrieve the required contact capacity required by the model
        model_max_contacts, world_max_contacts = builder.required_contact_capacity

        # Proceed with allocations only if the model allocates contacts
        if model_max_contacts >= 0:
            # Allocate the contacts container which will hold the generated contacts
            self._contacts = Contacts(
                default_max_contacts=self._settings.max_contacts, capacity=world_max_contacts, device=device
            )

            # Cache the maximum number of contacts allocated for the model
            self._model_max_contacts: int = self._contacts.num_model_max_contacts
            self._world_max_contacts: list[int] = self._contacts.num_world_max_contacts

            # Initialize the configured collision detection pipeline
            match self._settings.pipeline:
                case CollisionPipelineType.PRIMITIVE:
                    self._primitive_pipeline = CollisionPipelinePrimitive(
                        device=device,
                        builder=builder,
                        broadphase=self._settings.broadphase,
                        bvtype=self._settings.bvtype,
                    )
                case CollisionPipelineType.UNIFIED:
                    self._unified_pipeline = CollisionPipelineUnifiedKamino(
                        device=device,
                        builder=builder,
                        broadphase=self._settings.broadphase,
                        # TODO: Add support for bvtype in unified pipeline
                    )
                case _:
                    raise ValueError(f"Unsupported CollisionPipelineType: {self._settings.pipeline}")

    def reset(self):
        """
        TODO
        """
        pass

    def collide(self, model: Model, data: ModelData):
        """
        Executes collision detection given a model and its associated data.

        This operation will use the `PRIMITIVE` or `UNIFIED` pipeline depending on
        the configuration set during the initialization of the CollisionDetector.

        Args:
            model (Model): The Model instance containing the collision geometries
            data (ModelData): The ModelData instance containing the state of the geometries
        """
        # Check that the contacts container is allocated
        if self._contacts is None:
            raise RuntimeError("Cannot perform collision detection: contacts container is not allocated")

        # Check that the model and data are valid
        if model is None:
            raise ValueError("Cannot perform collision detection: model is None")
        if not isinstance(model, Model):
            raise TypeError(f"Cannot perform collision detection: expected Model, got {type(model)}")
        if data is None:
            raise ValueError("Cannot perform collision detection: data is None")
        if not isinstance(data, ModelData):
            raise TypeError(f"Cannot perform collision detection: expected ModelData, got {type(data)}")

        # Skip this operation if the model does not allocate contacts
        if self._model_max_contacts <= 0:
            return

        # Execute the configured collision detection pipeline
        match self._settings.pipeline:
            case CollisionPipelineType.PRIMITIVE:
                self._primitive_pipeline.collide(model, data, self._contacts)
            case CollisionPipelineType.UNIFIED:
                self._unified_pipeline.collide(model, data, self._contacts)
            case _:
                raise ValueError(f"Unsupported CollisionPipelineType: {self._settings.pipeline}")
