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
Provides utility functions for model
builder composition and manipulation.

This module includes functions to add common
modifiers to model builders, such as ground
planes, as well as factory functions to create
homogeneous multi-world builders and import
USD models.
"""

from collections.abc import Callable

import warp as wp

from ...core.builder import ModelBuilder
from ...core.shapes import BoxShape, PlaneShape
from ...core.types import transformf, vec3f, vec6f
from ...utils.io.usd import USDImporter

###
# Module interface
###

__all__ = [
    "make_homogeneous_builder",
]


###
# Modifiers
###


def add_ground_plane(
    builder: ModelBuilder,
    layer: str = "world",
    group: int = 1,
    collides: int = 1,
    world_index: int = 0,
    z_offset: float = 0.0,
) -> int:
    """
    Adds a static collision layer and geometry to a given builder to represent a flat ground plane.

    Args:
        builder (ModelBuilder): The model builder to which the ground geom should be added.
        group (int): The collision group for the ground geometry.
        collides (int): The collision mask for the ground geometry.
        world_index (int): The index of the world in the builder where the ground geom should be added.

    Returns:
        int: The ID of the added ground geometry.
    """
    return builder.add_collision_geometry(
        shape=PlaneShape(vec3f(0.0, 0.0, 1.0), 0.0),
        offset=transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        name="ground",
        layer=layer,
        group=group,
        collides=collides,
        world_index=world_index,
    )


def add_ground_box(
    builder: ModelBuilder,
    layer: str = "world",
    group: int = 1,
    collides: int = 1,
    world_index: int = 0,
    z_offset: float = 0.0,
) -> int:
    """
    Adds a static collision layer and geometry to a given builder to represent a flat ground plane.

    Args:
        builder (ModelBuilder): The model builder to which the ground geom should be added.
        group (int): The collision group for the ground geometry.
        collides (int): The collision mask for the ground geometry.
        world_index (int): The index of the world in the builder where the ground geom should be added.

    Returns:
        int: The ID of the added ground geometry.
    """
    return builder.add_collision_geometry(
        shape=BoxShape(20.0, 20.0, 1.0),
        offset=transformf(0.0, 0.0, -0.5 + z_offset, 0.0, 0.0, 0.0, 1.0),
        name="ground",
        layer=layer,
        group=group,
        collides=collides,
        world_index=world_index,
    )


def set_uniform_body_pose_offset(builder: ModelBuilder, offset: transformf):
    """
    Offsets a model builder by a given transform.

    Args:
        builder (ModelBuilder): The model builder to offset.
        offset (transformf): The transform offset to apply to each body in the builder.
    """
    for i in range(builder.num_bodies):
        builder.bodies[i].q_i_0 = wp.mul(offset, builder.bodies[i].q_i_0)


def set_uniform_body_twist_offset(builder: ModelBuilder, offset: vec6f):
    """
    Offsets a model builder by a given transform.

    Args:
        builder (ModelBuilder): The model builder to offset.
        offset (vec6f): The twist offset to apply to each body in the builder.
    """
    for i in range(builder.num_bodies):
        builder.bodies[i].u_i_0 += offset


###
# Builder utilities
###


def build_usd(
    source: str,
    load_static_geometry: bool = True,
    ground: bool = True,
) -> ModelBuilder:
    """
    Imports a USD model and optionally adds a ground plane.

    Each call creates a new world with the USD model and optional ground plane.

    Args:
        source: Path to USD file
        load_static_geometry: Whether to load static geometry from USD
        ground: Whether to add a ground plane

    Returns:
        ModelBuilder with imported USD model and optional ground plane
    """
    # Import the USD model
    importer = USDImporter()
    _builder = importer.import_from(
        source=source,
        load_static_geometry=load_static_geometry,
    )

    # Optionally add ground geometry
    if ground:
        add_ground_box(builder=_builder, group=1, collides=1)

    # Return the builder constructed from the USD model
    return _builder


def make_homogeneous_builder(num_worlds: int, build_fn: Callable, **kwargs) -> ModelBuilder:
    """
    Utility factory function to create a multi-world builder with identical worlds replicated across the model.

    Args:
        num_worlds (int): The number of worlds to create.
        build_fn (callable): The model builder function to use.
        **kwargs: Additional keyword arguments to pass to the builder function.

    Returns:
        ModelBuilder: The constructed model builder.
    """
    # First build a single world
    # NOTE: We want to do this first to avoid re-constructing the same model multiple
    # times especially if the construction is expensive such as importing from USD.
    single = build_fn(**kwargs)

    # Then replicate it across the specified number of worlds
    builder = ModelBuilder(default_world=False)
    for _ in range(num_worlds):
        builder.add_builder(single)
    return builder
