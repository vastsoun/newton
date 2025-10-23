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

"""The customized debug viewer of Kamino"""

from typing import ClassVar

import warp as wp

from ....viewer import ViewerGL
from ..core.builder import ModelBuilder
from ..core.geometry import CollisionGeometryDescriptor, GeometryDescriptor
from ..core.shapes import ShapeType
from ..simulation.simulator import Simulator

###
#
###


class ViewerKamino(ViewerGL):
    """
    A customized debug viewer for Kamino.
    """

    # Define a static set of colors for different bodies
    body_colors: ClassVar[list[wp.array]] = [
        wp.array([wp.vec3(0.9, 0.1, 0.3)], dtype=wp.vec3),  # Crimson Red
        wp.array([wp.vec3(0.1, 0.7, 0.9)], dtype=wp.vec3),  # Cyan Blue
        wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Orange
        wp.array([wp.vec3(0.6, 0.2, 0.8)], dtype=wp.vec3),  # Purple
        wp.array([wp.vec3(0.2, 0.8, 0.2)], dtype=wp.vec3),  # Green
        wp.array([wp.vec3(0.8, 0.8, 0.2)], dtype=wp.vec3),  # Yellow
        wp.array([wp.vec3(0.8, 0.2, 0.8)], dtype=wp.vec3),  # Magenta
        wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3),  # Gray
    ]

    def __init__(
        self,
        builder: ModelBuilder,
        simulator: Simulator,
        width: int = 1920,
        height: int = 1080,
        vsync: bool = False,
        headless: bool = False,
    ):
        # Initialize the base viewer
        super().__init__(width=width, height=height, vsync=vsync, headless=headless)

        # Cache references to the simulator
        self._simulator = simulator

        # Declare and initialize geometry info cache
        self._collision_geometry: list[CollisionGeometryDescriptor] = builder.collision_geoms
        self._physical_geometry: list[GeometryDescriptor] = builder.physical_geoms

    def render_frame(self):
        # Begin a new frame
        self.begin_frame(self._simulator.time)

        # Extract body poses from the kamino simulator
        # try:
        body_poses = self._simulator.state.q_i.numpy()

        # Render each geometry using log_shapes
        for i, cgeom in enumerate(self._collision_geometry):
            # Skip Empty/None shapes
            if cgeom.shape.typeid == ShapeType.EMPTY:
                continue

            # Extract shape info
            if cgeom.bid < len(body_poses):
                # Convert kamino transformf to warp transform
                pose = body_poses[cgeom.bid]
                # kamino transformf has [x, y, z, qx, qy, qz, qw] format
                position = wp.vec3(float(pose[0]), float(pose[1]), float(pose[2]))
                quaternion = wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
                body_transform = wp.transform(position, quaternion)

                # Combine body and offset transforms
                geom_transform = wp.transform_multiply(body_transform, cgeom.offset)

                # Choose color based on body ID
                color = self.body_colors[cgeom.bid % len(self.body_colors)]

                # Extract geometry parameters based on shape type
                geom_params = None
                geom_src = None
                geom_params = None
                params = cgeom.shape.params
                num_params = cgeom.shape._num_params_of(cgeom.shape.typeid)
                if cgeom.shape.typeid == ShapeType.SPHERE:
                    geom_params = params[0]
                elif cgeom.shape.typeid == ShapeType.CYLINDER:
                    geom_params = (params[0], 0.5 * params[1])
                elif cgeom.shape.typeid == ShapeType.CONE:
                    geom_params = (params[0], 0.5 * params[1])
                elif cgeom.shape.typeid == ShapeType.CAPSULE:
                    geom_params = (params[0], 0.5 * params[1])
                elif cgeom.shape.typeid == ShapeType.BOX:
                    geom_params = 0.5 * params[:num_params]
                elif cgeom.shape.typeid == ShapeType.ELLIPSOID:
                    geom_params = params[:num_params]
                # elif cgeom.shape.typeid in {ShapeType.MESH, ShapeType.CONVEX}:
                elif cgeom.shape.typeid == ShapeType.MESH:
                    geom_params = 1.0
                    geom_src = self._collision_geometry[cgeom.gid].shape._data

                # Update the geometry data
                self.log_shapes(
                    name=f"/body_{cgeom.bid}/geom_{i}",
                    geo_type=ShapeType(cgeom.shape.typeid).to_newton(),
                    geo_scale=geom_params,
                    xforms=wp.array([geom_transform], dtype=wp.transform),
                    geo_is_solid=True,
                    colors=color,
                    geo_src=geom_src,
                )

        # End the new frame
        self.end_frame()
