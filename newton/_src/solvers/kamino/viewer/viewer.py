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
        body_poses = self._simulator.state.q_i.numpy()

        # Render each geometry using log_shapes
        for i, cgeom in enumerate(self._collision_geometry):
            # Skip Empty/None shapes
            if cgeom.shape.typeid == ShapeType.EMPTY:
                continue

            # Handle the case of static geometry (bid < 0)
            if cgeom.bid < 0:
                body_transform = wp.transform_identity()
            else:
                body_transform = wp.transform(*body_poses[cgeom.bid])

            # Combine body and offset transforms
            geom_transform = wp.transform_multiply(body_transform, cgeom.offset)

            # Choose color based on body ID
            color = self.body_colors[cgeom.bid % len(self.body_colors)]

            # Update the geometry data
            self.log_shapes(
                name=f"/body_{cgeom.bid}/geom_{i}",
                geo_type=cgeom.shape.type.to_newton(),
                geo_scale=cgeom.shape.params,
                xforms=wp.array([geom_transform], dtype=wp.transform),
                geo_is_solid=True,
                colors=color,
                geo_src=cgeom.shape.data,
            )

        # End the new frame
        self.end_frame()
