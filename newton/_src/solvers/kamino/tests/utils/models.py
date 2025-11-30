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

"""Provides utility functions to build simple models for testing collision detection."""

from collections.abc import Callable

import numpy as np

from ...core.builder import ModelBuilder
from ...core.shapes import (
    BoxShape,
    CapsuleShape,
    ConeShape,
    CylinderShape,
    EllipsoidShape,
    ShapeType,
    SphereShape,
)
from ...core.types import mat33f, transformf, vec6f

###
# Same-Shape Combinations
###


def build_sphere_on_sphere(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with two spheres positioned along the z-axis.

    The first body (sphere 0) is placed below the second body (sphere 1) along the z-axis.

    The spheres are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two spheres along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two spheres.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid0, shape=SphereShape(0.5))
    builder.add_collision_geometry(body=bid1, shape=SphereShape(0.5))
    return builder


def build_box_on_box(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with two boxes positioned along the z-axis.

    The first body (box 0) is placed below the second body (box 1) along the z-axis.

    The boxes are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two boxes along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two boxes.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid0, shape=BoxShape(2.0, 2.0, 1.0))
    builder.add_collision_geometry(body=bid1, shape=BoxShape(1.0, 1.0, 1.0))
    return builder


def build_capsule_on_capsule(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with two capsules positioned along the z-axis.

    The first body (capsule 0) is placed below the second body (capsule 1) along the z-axis.

    The capsules are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two capsules along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two capsules.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid0, shape=CapsuleShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid1, shape=CapsuleShape(0.5, 1.0))
    return builder


def build_cylinder_on_cylinder(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with two cylinders positioned along the z-axis.

    The first body (cylinder 0) is placed below the second body (cylinder 1) along the z-axis.

    The cylinders are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two cylinders along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two cylinders.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid0, shape=CylinderShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid1, shape=CylinderShape(0.5, 1.0))
    return builder


def build_cone_on_cone(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with two cones positioned along the z-axis.

    The first body (cone 0) is placed below the second body (cone 1) along the z-axis.

    The cones are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two cones along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two cones.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid0, shape=ConeShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid1, shape=ConeShape(0.5, 1.0))
    return builder


def build_ellipsoid_on_ellipsoid(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with two ellipsoids positioned along the z-axis.

    The first body (ellipsoid 0) is placed below the second body (ellipsoid 1) along the z-axis.

    The ellipsoids are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two ellipsoids along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two ellipsoids.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid0, shape=EllipsoidShape(0.5, 0.5, 1.0))
    builder.add_collision_geometry(body=bid1, shape=EllipsoidShape(0.5, 0.5, 1.0))
    return builder


###
# Different-Shape Combinations
###


def build_sphere_on_box(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a sphere positioned above a box along the z-axis.

    The first body (box 0) is placed below the second body (sphere 1) along the z-axis.

    The bodies are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two boxes along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two boxes.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid0, shape=SphereShape(0.5))
    builder.add_collision_geometry(body=bid1, shape=BoxShape(1.0, 1.0, 1.0))
    return builder


def build_sphere_on_capsule(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a sphere positioned above a capsule along the z-axis.

    The first body (capsule 0) is placed below the second body (sphere 1).
    Bodies are separated by dz_0 (negative for penetration, positive for separation).
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cap = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_sph = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cap, shape=CapsuleShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_sph, shape=SphereShape(0.5))
    return builder


def build_sphere_on_cylinder(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a sphere positioned above a cylinder along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cyl = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_sph = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cyl, shape=CylinderShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_sph, shape=SphereShape(0.5))
    return builder


def build_sphere_on_cone(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a sphere positioned above a cone along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cone = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_sph = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cone, shape=ConeShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_sph, shape=SphereShape(0.5))
    return builder


def build_sphere_on_ellipsoid(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a sphere positioned above an ellipsoid along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_ell = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_sph = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_ell, shape=EllipsoidShape(0.5, 0.5, 1.0))
    builder.add_collision_geometry(body=bid_sph, shape=SphereShape(0.5))
    return builder


def build_box_on_capsule(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a box positioned above a capsule along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cap = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_box = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cap, shape=CapsuleShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_box, shape=BoxShape(1.0, 1.0, 1.0))
    return builder


def build_box_on_cylinder(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a box positioned above a cylinder along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cyl = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_box = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cyl, shape=CylinderShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_box, shape=BoxShape(1.0, 1.0, 1.0))
    return builder


def build_box_on_cone(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a box positioned above a cone along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cone = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_box = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cone, shape=ConeShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_box, shape=BoxShape(1.0, 1.0, 1.0))
    return builder


def build_box_on_ellipsoid(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a box positioned above an ellipsoid along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_ell = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_box = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_ell, shape=EllipsoidShape(0.5, 0.5, 1.0))
    builder.add_collision_geometry(body=bid_box, shape=BoxShape(1.0, 1.0, 1.0))
    return builder


def build_capsule_on_cylinder(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a capsule positioned above a cylinder along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cyl = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_cap = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cyl, shape=CylinderShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_cap, shape=CapsuleShape(0.5, 1.0))
    return builder


def build_capsule_on_cone(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a capsule positioned above a cone along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cone = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_cap = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cone, shape=ConeShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_cap, shape=CapsuleShape(0.5, 1.0))
    return builder


def build_capsule_on_ellipsoid(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a capsule positioned above an ellipsoid along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_ell = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_cap = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_ell, shape=EllipsoidShape(0.5, 0.5, 1.0))
    builder.add_collision_geometry(body=bid_cap, shape=CapsuleShape(0.5, 1.0))
    return builder


def build_cylinder_on_cone(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a cylinder positioned above a cone along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_cone = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_cyl = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_cone, shape=ConeShape(0.5, 1.0))
    builder.add_collision_geometry(body=bid_cyl, shape=CylinderShape(0.5, 1.0))
    return builder


def build_cylinder_on_ellipsoid(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a cylinder positioned above an ellipsoid along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_ell = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_cyl = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_ell, shape=EllipsoidShape(0.5, 0.5, 1.0))
    builder.add_collision_geometry(body=bid_cyl, shape=CylinderShape(0.5, 1.0))
    return builder


def build_cone_on_ellipsoid(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a cone positioned above an ellipsoid along the z-axis.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid_ell = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid_cone = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_geometry(body=bid_ell, shape=EllipsoidShape(0.5, 0.5, 1.0))
    builder.add_collision_geometry(body=bid_cone, shape=ConeShape(0.5, 1.0))
    return builder


###
# Utility Mappings
###


shape_name_to_type: dict[str, ShapeType] = {
    "sphere": ShapeType.SPHERE,
    "box": ShapeType.BOX,
    "capsule": ShapeType.CAPSULE,
    "cylinder": ShapeType.CYLINDER,
    "cone": ShapeType.CONE,
    "ellipsoid": ShapeType.ELLIPSOID,
}


shape_type_to_name: dict[ShapeType, str] = {
    ShapeType.SPHERE: "sphere",
    ShapeType.BOX: "box",
    ShapeType.CAPSULE: "capsule",
    ShapeType.CYLINDER: "cylinder",
    ShapeType.CONE: "cone",
    ShapeType.ELLIPSOID: "ellipsoid",
}


shape_combinations: list[tuple[str, str]] = [
    ("sphere", "sphere"),
    ("sphere", "box"),
    ("sphere", "capsule"),
    ("sphere", "cylinder"),
    ("sphere", "cone"),
    ("sphere", "ellipsoid"),
    ("box", "box"),
    ("box", "capsule"),
    ("box", "cylinder"),
    ("box", "cone"),
    ("box", "ellipsoid"),
    ("capsule", "capsule"),
    ("capsule", "cylinder"),
    ("capsule", "cone"),
    ("capsule", "ellipsoid"),
    ("cylinder", "cylinder"),
    ("cylinder", "cone"),
    ("cylinder", "ellipsoid"),
    ("cone", "cone"),
    ("cone", "ellipsoid"),
    ("ellipsoid", "ellipsoid"),
]


shape_combination_to_builder: dict[tuple[str, str], Callable[[float], ModelBuilder]] = {
    ("sphere", "sphere"): build_sphere_on_sphere,
    ("sphere", "box"): build_sphere_on_box,
    ("sphere", "capsule"): build_sphere_on_capsule,
    ("sphere", "cylinder"): build_sphere_on_cylinder,
    ("sphere", "cone"): build_sphere_on_cone,
    ("sphere", "ellipsoid"): build_sphere_on_ellipsoid,
    ("box", "box"): build_box_on_box,
    ("box", "capsule"): build_box_on_capsule,
    ("box", "cylinder"): build_box_on_cylinder,
    ("box", "cone"): build_box_on_cone,
    ("box", "ellipsoid"): build_box_on_ellipsoid,
    ("capsule", "capsule"): build_capsule_on_capsule,
    ("capsule", "cylinder"): build_capsule_on_cylinder,
    ("capsule", "cone"): build_capsule_on_cone,
    ("capsule", "ellipsoid"): build_capsule_on_ellipsoid,
    ("cylinder", "cylinder"): build_cylinder_on_cylinder,
    ("cylinder", "cone"): build_cylinder_on_cone,
    ("cylinder", "ellipsoid"): build_cylinder_on_ellipsoid,
    ("cone", "cone"): build_cone_on_cone,
    ("cone", "ellipsoid"): build_cone_on_ellipsoid,
    ("ellipsoid", "ellipsoid"): build_ellipsoid_on_ellipsoid,
}
