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

import numpy as np
import warp as wp
from warp.context import Devicelike

from ...core.model import ModelData, ModelKamino
from ...core.shapes import ShapeType
from ...core.types import float32, int32, vec2i, vec6f
from ..contacts import DEFAULT_GEOM_PAIR_CONTACT_MARGIN, Contacts
from .broadphase import (
    PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES,
    BoundingVolumesData,
    BoundingVolumeType,
    CollisionCandidatesData,
    CollisionCandidatesModel,
    primitive_broadphase_explicit,
)
from .narrowphase import PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS, primitive_narrowphase

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
        model: ModelKamino | None = None,
        bvtype: BoundingVolumeType = BoundingVolumeType.AABB,
        default_margin: float = DEFAULT_GEOM_PAIR_CONTACT_MARGIN,
        device: Devicelike = None,
    ):
        """
        Initialize an instance of Kamino's optimized primitive collision detection pipeline.

        Args:
            model (ModelKamino | None): The model container holding the time-invariant parameters of the simulation.
            broadphase (BroadPhaseMode): Broad-phase collision detection algorithm to use.
            bvtype (BoundingVolumeType): Type of bounding volume to use in broad-phase.
            default_margin (float): Default collision margin for geometries.
            device (Devicelike): Device on which to allocate data and perform computations.
        """
        # Cache pipeline settings
        self._device: Devicelike = device
        self._bvtype: BoundingVolumeType = bvtype
        self._default_margin: float = default_margin

        # Declare the internal data containers
        self._cmodel: CollisionCandidatesModel | None = None
        self._cdata: CollisionCandidatesData | None = None
        self._bvdata: BoundingVolumesData | None = None

        # If a model is provided, proceed to finalize all data allocations
        if model is not None:
            self.finalize(model, bvtype, device)

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

    def finalize(self, model: ModelKamino, bvtype: BoundingVolumeType | None = None, device: Devicelike = None):
        """
        Finalizes the collision detection pipeline by allocating all necessary data structures.

        Args:
            model (ModelKamino): The model container holding the time-invariant parameters of the simulation.
            bvtype (BoundingVolumeType | None): Optional bounding volume type to override the default.
            device (Devicelike): The Warp device on which the pipeline will operate.
        """
        # Override the device if specified
        if device is not None:
            self._device = device

        # Override the device if specified
        if bvtype is not None:
            self._bvtype = bvtype

        # Retrieve the number of worlds and total number of geometries in the model
        num_worlds = model.size.num_worlds
        num_geoms = model.geoms.num_geoms

        # Ensure that all shape types are supported by the primitive
        # broad-phase and narrow-phase back-ends before proceeding
        model_num_geom_pairs = model.geoms.num_collidable_geom_pairs
        world_num_geom_pairs, geom_pair_wid = self._assert_shapes_supported(model)

        # Allocate the collision model data
        with wp.ScopedDevice(self._device):
            # Allocate the bounding volumes data
            self._bvdata = BoundingVolumesData()
            match self._bvtype:
                case BoundingVolumeType.AABB:
                    self._bvdata.aabb = wp.zeros(shape=(num_geoms,), dtype=vec6f)
                case BoundingVolumeType.BS:
                    self._bvdata.radius = wp.zeros(shape=(num_geoms,), dtype=float32)
                case _:
                    raise ValueError(f"Unsupported BoundingVolumeType: {self._bvtype}")

            # Allocate the time-invariant collision candidates model
            self._cmodel = CollisionCandidatesModel(
                num_model_geom_pairs=model.geoms.num_collidable_geom_pairs,
                num_world_geom_pairs=world_num_geom_pairs,
                model_num_pairs=wp.array([model_num_geom_pairs], dtype=int32),
                world_num_pairs=wp.array(world_num_geom_pairs, dtype=int32),
                wid=wp.array(geom_pair_wid, dtype=int32),
                geom_pair=model.geoms.collidable_pairs,
            )

            # Allocate the time-varying collision candidates data
            self._cdata = CollisionCandidatesData(
                num_model_geom_pairs=model_num_geom_pairs,
                model_num_collisions=wp.zeros(shape=(1,), dtype=int32),
                world_num_collisions=wp.zeros(shape=(num_worlds,), dtype=int32),
                wid=wp.zeros(shape=(model_num_geom_pairs,), dtype=int32),
                geom_pair=wp.zeros(shape=(model_num_geom_pairs,), dtype=vec2i),
            )

    def collide(self, model: ModelKamino, data: ModelData, contacts: Contacts):
        """
        Runs the unified collision detection pipeline to generate discrete contacts.

        Args:
            model (ModelKamino): The model container holding the time-invariant parameters of the simulation.
            data (ModelData): The data container holding the time-varying state of the simulation.
            contacts (Contacts): Output contacts container (will be cleared and populated)
        """
        # Ensure that the pipeline has been finalized
        # before proceeding with actual operations
        self._assert_finalized()

        # Clear all active collision candidates and contacts
        self._cdata.clear()
        contacts.clear()

        # Perform the broad-phase collision detection to generate candidate pairs
        primitive_broadphase_explicit(
            body_poses=data.bodies.q_i,
            geoms_model=model.geoms,
            geoms_data=data.geoms,
            bv_type=self._bvtype,
            bv_data=self._bvdata,
            candidates_model=self._cmodel,
            candidates_data=self._cdata,
            default_margin=self._default_margin,
        )

        # Perform the narrow-phase collision detection to generate active contacts
        primitive_narrowphase(model, data, self._cdata, contacts, default_margin=self._default_margin)

    ###
    # Internals
    ###

    def _assert_finalized(self):
        """
        Asserts that the collision detection pipeline has been finalized.

        Raises:
            RuntimeError: If the pipeline has not been finalized.
        """
        if self._cmodel is None or self._cdata is None or self._bvdata is None:
            raise RuntimeError(
                "CollisionPipelinePrimitive has not been finalized. "
                "Please call `finalize(model, device)` before using the pipeline."
            )

    def _assert_shapes_supported(self, model: ModelKamino) -> tuple[list[int], np.ndarray]:
        """
        Checks whether all collision geometries in the provided builder are supported
        by the primitive narrow-phase collider.

        Args:
            model (ModelKamino): The model container holding the time-invariant parameters of the simulation.

        Raises:
            ValueError: If any unsupported shape type is found.
        """
        # Iterate over each candidate geometry pair
        geom_type = model.geoms.sid.numpy()
        geom_wid = model.geoms.wid.numpy()
        geom_pairs = model.geoms.collidable_pairs.numpy()
        world_num_geom_pairs: list[int] = [0] * model.size.num_worlds
        geom_pair_wid: np.ndarray = np.zeros(shape=(geom_pairs.shape[0],), dtype=np.int32)
        for gid_12 in range(geom_pairs.shape[0]):
            # Retrieve the shape types and world indices of the geometry pair
            gid_1 = geom_pairs[gid_12, 0]
            gid_2 = geom_pairs[gid_12, 1]
            shape_1 = ShapeType(geom_type[gid_1])
            shape_2 = ShapeType(geom_type[gid_2])

            # First check if both shapes are supported by the primitive broad-phase
            if shape_1 not in PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES:
                raise ValueError(
                    f"Builder contains shape '{shape_1}' which is currently not supported by the primitive broad-phase."
                    "\nPlease consider using the `UNIFIED` collision pipeline, or using alternative shape types."
                )
            if shape_2 not in PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES:
                raise ValueError(
                    f"Builder contains shape '{shape_2}' which is currently not supported by the primitive broad-phase."
                    "\nPlease consider using the `UNIFIED` collision pipeline, or using alternative shape types."
                )

            # Then check if the shape-pair combination is supported by the primitive narrow-phase
            shape_12_invalid = (shape_1, shape_2) not in PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS
            shape_21_invalid = (shape_2, shape_1) not in PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS
            if shape_12_invalid and shape_21_invalid:
                raise ValueError(
                    f"Builder contains shape-pair ({shape_1}, {shape_2}) with geom indices {gid_12}, "
                    "but it is currently not supported by the primitive narrow-phase."
                    "\nPlease consider using the `UNIFIED` collision pipeline, or using alternative shape types."
                )

            # Store the world index for this geometry pair
            geom_pair_12_wid = geom_wid[gid_1]
            geom_pair_wid[gid_12] = geom_pair_12_wid
            world_num_geom_pairs[geom_pair_12_wid] += 1

        # Return the per-world geometry pair counts and the per-geom-pair world indices
        return world_num_geom_pairs, geom_pair_wid
