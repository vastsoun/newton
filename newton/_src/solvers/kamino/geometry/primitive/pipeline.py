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

"""TODO"""

import warp as wp
from warp.context import Devicelike

from .....sim.collide_unified import BroadPhaseMode
from ...core.builder import ModelBuilder
from ...core.model import Model, ModelData
from ...core.types import int32, vec2i
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
    TODO
    """

    def __init__(
        self,
        builder: ModelBuilder | None = None,
        broadphase: BroadPhaseMode = BroadPhaseMode.EXPLICIT,
        bvtype: BoundingVolumeType | None = None,
        device: Devicelike = None,
    ):
        """
        TODO
        """
        # TODO
        self._device: Devicelike = None
        self._broadphase: BroadPhaseMode = BroadPhaseMode.EXPLICIT
        self._bvtype: BoundingVolumeType = BoundingVolumeType.AABB

        # TODO
        self._cmodel: CollisionCandidatesModel | None = None
        self._cdata: CollisionCandidatesData | None = None
        self._bvdata: BoundingVolumesData | None = None

        # TODO
        if builder is not None:
            self.finalize(builder, bvtype, device)

    def finalize(self, builder: ModelBuilder, bvtype: BoundingVolumeType | None = None, device: Devicelike = None):
        """
        TODO
        """
        # Override the device if specified
        if device is not None:
            self._device = device

        # Override the device if specified
        if bvtype is not None:
            self._bvtype = bvtype

        # Retrieve the number of world
        num_worlds = builder.num_worlds

        # Construct collision pairs
        world_num_geom_pairs, model_geom_pair, model_pairid, model_wid = builder.make_collision_candidate_pairs()
        model_num_geom_pairs = len(model_geom_pair)

        # Allocate the collision model data
        with wp.ScopedDevice(self._device):
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

    def reset(self):
        """
        TODO
        """
        pass

    def collide(self, model: Model, data: ModelData, contacts: Contacts):
        """
        TODO
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
