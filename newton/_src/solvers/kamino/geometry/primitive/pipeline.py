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
A collision detection pipeline optimized for primitive shapes.

This pipeline uses an `EXPLICIT` broad-phase operating on pre-computed
geometry pairs and a narrow-phase based on the primitive colliders of Newton.
"""

import warp as wp
from warp.context import Devicelike

from .....sim.collide_unified import BroadPhaseMode
from ...core.builder import ModelBuilder
from ...core.model import Model, ModelData
from ...core.types import float32, int32, mat83f, vec2i
from ..contacts import Contacts
from .broadphase import (
    BoundingVolumesData,
    BoundingVolumeType,
    CollisionCandidatesData,
    CollisionCandidatesModel,
    primitive_broadphase_explicit,
)
from .narrowphase import primitive_narrowphase

###
# Interfaces
###


class CollisionPipelinePrimitive:
    """
    A collision detection pipeline optimized for primitive shapes.

    This pipeline uses an `EXPLICIT` broad-phase operating on pre-computed
    geometry pairs and a narrow-phase based on the primitive colliders of Newton.
    """

    def __init__(
        self,
        builder: ModelBuilder | None = None,
        broadphase: BroadPhaseMode = BroadPhaseMode.EXPLICIT,
        bvtype: BoundingVolumeType = BoundingVolumeType.AABB,
        default_margin: float = 1e-5,
        device: Devicelike = None,
    ):
        """
        Initialize an instance of Kamino's optimized primitive collision detection pipeline.

        Args:
            builder (ModelBuilder | None): Optional model builder to pre-compute collision pairs.
            broadphase (BroadPhaseMode): Broad-phase collision detection algorithm to use.
            bvtype (BoundingVolumeType): Type of bounding volume to use in broad-phase.
            default_margin (float): Default collision margin for geometries.
            device (Devicelike): Device on which to allocate data and perform computations.
        """
        # Cache pipeline settings
        self._device: Devicelike = device
        self._broadphase: BroadPhaseMode = broadphase
        self._bvtype: BoundingVolumeType = bvtype
        self._default_margin: float = default_margin

        # Declare the internal data containers
        self._cmodel: CollisionCandidatesModel | None = None
        self._cdata: CollisionCandidatesData | None = None
        self._bvdata: BoundingVolumesData | None = None

        # If a builder is provided, proceed to finalize all data allocations
        if builder is not None:
            self.finalize(builder, bvtype, device)

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike:
        """Returns the Warp device the pipeline operates on."""
        return self._device

    ###
    # Operations
    ###

    def finalize(self, builder: ModelBuilder, bvtype: BoundingVolumeType | None = None, device: Devicelike = None):
        """
        Finalizes the collision detection pipeline by allocating all necessary data structures.

        Args:
            builder (ModelBuilder): The model builder used to pre-compute collision pairs.
            bvtype (BoundingVolumeType | None): Optional bounding volume type to override the default.
            device (Devicelike): The Warp device on which the pipeline will operate.
        """
        # Override the device if specified
        if device is not None:
            self._device = device

        # Override the device if specified
        if bvtype is not None:
            self._bvtype = bvtype

        # Retrieve the number of world
        num_worlds = builder.num_worlds
        num_geoms = len(builder.collision_geoms)

        # Construct collision pairs
        world_num_geom_pairs, model_geom_pair, model_pairid, model_wid = builder.make_collision_candidate_pairs()
        model_num_geom_pairs = len(model_geom_pair)

        # Allocate the collision model data
        with wp.ScopedDevice(self._device):
            # Allocate the bounding volumes data
            self._bvdata = BoundingVolumesData()
            match self._bvtype:
                case BoundingVolumeType.AABB:
                    self._bvdata.aabb = wp.zeros(shape=(num_geoms,), dtype=mat83f)
                case BoundingVolumeType.BS:
                    self._bvdata.radius = wp.zeros(shape=(num_geoms,), dtype=float32)
                case _:
                    raise ValueError(f"Unsupported BoundingVolumeType: {self._bvtype}")

            # Allocate the time-invariant collision candidates model
            self._cmodel = CollisionCandidatesModel(
                num_model_geom_pairs=model_num_geom_pairs,
                num_world_geom_pairs=world_num_geom_pairs,
                model_num_pairs=wp.array([model_num_geom_pairs], dtype=int32),
                world_num_pairs=wp.array(world_num_geom_pairs, dtype=int32),
                wid=wp.array(model_wid, dtype=int32),
                pairid=wp.array(model_pairid, dtype=int32),
                geom_pair=wp.array(model_geom_pair, dtype=vec2i),
            )

            # Allocate the time-varying collision candidates data
            self._cdata = CollisionCandidatesData(
                num_model_geom_pairs=model_num_geom_pairs,
                model_num_collisions=wp.zeros(shape=(1,), dtype=int32),
                world_num_collisions=wp.zeros(shape=(num_worlds,), dtype=int32),
                wid=wp.zeros(shape=(model_num_geom_pairs,), dtype=int32),
                geom_pair=wp.zeros(shape=(model_num_geom_pairs,), dtype=vec2i),
            )

    def collide(self, model: Model, data: ModelData, contacts: Contacts):
        """
        Runs the unified collision detection pipeline to generate discrete contacts.

        Args:
            model (Model): The model container holding the time-invariant parameters of the simulation.
            data (ModelData): The data container holding the time-varying state of the simulation.
            contacts (Contacts): Output contacts container (will be cleared and populated)
        """
        # Clear all active collision candidates and contacts
        self._cdata.clear()
        contacts.clear()

        # Perform the broad-phase collision detection to generate candidate pairs
        primitive_broadphase_explicit(
            body_poses=data.bodies.q_i,
            geoms_model=model.cgeoms,
            geoms_data=data.cgeoms,
            bv_type=self._bvtype,
            bv_data=self._bvdata,
            candidates_model=self._cmodel,
            candidates_data=self._cdata,
        )

        # Perform the narrow-phase collision detection to generate active contacts
        primitive_narrowphase(model, data, self._cdata, contacts)
