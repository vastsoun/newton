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
KAMINO: Shape Types & Containers
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import IntEnum

import numpy as np
import warp as wp

if sys.version_info >= (3, 12):
    from typing import override
else:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback no-op decorator if typing_extensions is not available
        def override(func):
            return func


from ....core.types import Vec2, Vec3, nparray
from ....geometry.types import MESH_MAXHULLVERT, Mesh
from .types import Descriptor, vec4f

###
# Module interface
###

__all__ = [
    "BoxShape",
    "CapsuleShape",
    "ConeShape",
    "ConvexShape",
    "CylinderShape",
    "EllipsoidShape",
    "EmptyShape",
    "MeshShape",
    "PlaneShape",
    "SDFShape",
    "ShapeDescriptor",
    "ShapeType",
    "SphereShape",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

SHAPE_EMPTY = wp.constant(0)
"""The empty shape type, which has no parameters and is used to represent the absence of a shape."""

SHAPE_SPHERE = wp.constant(1)
"""The 1-parameter sphere shape type. Parameters: radius."""

SHAPE_CYLINDER = wp.constant(2)
"""The 2-parameter cylinder shape type. Parameters: radius, height."""

SHAPE_CONE = wp.constant(3)
"""The 2-parameter cone shape type. Parameters: radius, height."""

SHAPE_CAPSULE = wp.constant(4)
"""The 2-parameter capsule shape type. Parameters: radius, height."""

SHAPE_BOX = wp.constant(5)
"""The 3-parameter box shape type. Parameters: depth, width, height."""

SHAPE_ELLIPSOID = wp.constant(6)
"""The 3-parameter ellipsoid shape type. Parameters: a, b, c."""

SHAPE_PLANE = wp.constant(7)
"""The 4-parameter plane shape type. Parameters: normal_x, normal_y, normal_z, distance."""

SHAPE_CONVEX = wp.constant(8)
"""The n-parameter convex shape type. Parameters: vertices, normals, etc."""

SHAPE_MESH = wp.constant(9)
"""The n-parameter mesh shape type. Parameters: vertices, normals, triangles, triangle_normals."""

SHAPE_SDF = wp.constant(10)
"""The n-parameter signed-distance-field shape type. Parameters: sdf data, etc."""


###
# Containers
###


class ShapeType(IntEnum):
    """
    An enumeration of the different shape types.
    """

    EMPTY = 0
    """The empty shape type, which has no parameters and is used to represent the absence of a shape."""
    SPHERE = 1
    """The 1-parameter sphere shape type. Parameters: radius."""
    CYLINDER = 2
    """The 2-parameter cylinder shape type. Parameters: radius, height."""
    CONE = 3
    """The 2-parameter cone shape type. Parameters: radius, height."""
    CAPSULE = 4
    """The 2-parameter capsule shape type. Parameters: radius, height."""
    BOX = 5
    """The 3-parameter box shape type. Parameters: depth, width, height."""
    ELLIPSOID = 6
    """The 3-parameter ellipsoid shape type. Parameters: a, b, c."""
    PLANE = 7
    """The 4-parameter plane shape type. Parameters: normal_x, normal_y, normal_z, distance."""
    CONVEX = 8
    """The n-parameter convex shape type. Parameters: vertices, normals, etc."""
    MESH = 9
    """The n-parameter mesh shape type. Parameters: vertices, normals, triangles, triangle_normals."""
    SDF = 10
    """The n-parameter signed-distance-field shape type. Parameters: sdf data, etc."""


class ShapeDescriptor(ABC, Descriptor):
    @staticmethod
    def _num_params_of(shape_id: int) -> int:
        """
        Returns the number of parameters given a shape ID.
        """
        nparams = None
        match shape_id:
            case ShapeType.EMPTY:
                nparams = 0
            case ShapeType.SPHERE:
                nparams = 1
            case ShapeType.CYLINDER:
                nparams = 2
            case ShapeType.CONE:
                nparams = 2
            case ShapeType.CAPSULE:
                nparams = 2
            case ShapeType.BOX:
                nparams = 3
            case ShapeType.ELLIPSOID:
                nparams = 3
            case ShapeType.PLANE:
                nparams = 4
            case ShapeType.CONVEX:
                nparams = -1
            case ShapeType.MESH:
                nparams = -1
            case ShapeType.SDF:
                nparams = -1
            case _:
                raise ValueError(f"ShapeDescriptor: Unknown shape type ID: {shape_id}")
        return nparams

    def __init__(self, typeid: int, name: str = "", uid: str | None = None):
        super().__init__(name, uid)
        self._typeid: int = typeid
        self._nparams = ShapeDescriptor._num_params_of(typeid)

    def __repr__(self):
        return (
            f"ShapeDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"typeid: {self.typeid},\n"
            f"nparams: {self.nparams}\n"
            f")"
        )

    @property
    def typeid(self) -> int:
        return self._typeid

    @property
    def nparams(self) -> int:
        return self._nparams

    @property
    @abstractmethod
    def params(self) -> vec4f:
        return vec4f()


###
# Primitive Shapes
###


class EmptyShape(ShapeDescriptor):
    def __init__(self, name: str = "empty", uuid: str | None = None):
        super().__init__(ShapeType.EMPTY, name)

    @property
    @override
    def params(self) -> vec4f:
        return vec4f()


class SphereShape(ShapeDescriptor):
    def __init__(self, radius: float, name: str = "sphere", uuid: str | None = None):
        super().__init__(ShapeType.SPHERE, name)
        self.radius = radius

    @override
    def __repr__(self):
        return f"SphereShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius}\n)"

    @property
    @override
    def params(self) -> vec4f:
        return vec4f(self.radius, 0.0, 0.0, 0.0)


class CylinderShape(ShapeDescriptor):
    def __init__(self, radius: float, height: float, name: str = "cylinder", uuid: str | None = None):
        super().__init__(ShapeType.CYLINDER, name)
        self.radius = radius
        self.height = height

    @override
    def __repr__(self):
        return f"CylinderShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nheight: {self.height}\n)"

    @property
    @override
    def params(self) -> vec4f:
        return vec4f(self.radius, self.height, 0.0, 0.0)


class ConeShape(ShapeDescriptor):
    def __init__(self, radius: float, height: float, name: str = "cone", uuid: str | None = None):
        super().__init__(ShapeType.CONE, name)
        self.radius = radius
        self.height = height

    @override
    def __repr__(self):
        return f"ConeShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nheight: {self.height}\n)"

    @property
    @override
    def params(self) -> vec4f:
        return vec4f(self.radius, self.height, 0.0, 0.0)


class CapsuleShape(ShapeDescriptor):
    def __init__(self, radius: float, height: float, name: str = "capsule", uuid: str | None = None):
        super().__init__(ShapeType.CAPSULE, name)
        self.radius = radius
        self.height = height

    @override
    def __repr__(self):
        return f"CapsuleShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nheight: {self.height}\n)"

    @property
    @override
    def params(self) -> vec4f:
        return vec4f(self.radius, self.height, 0.0, 0.0)


class BoxShape(ShapeDescriptor):
    def __init__(self, depth: float, width: float, height: float, name: str = "box", uuid: str | None = None):
        super().__init__(ShapeType.BOX, name)
        self.depth = depth
        self.width = width
        self.height = height

    @override
    def __repr__(self):
        return (
            f"BoxShape(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"depth: {self.depth},\n"
            f"width: {self.width},\n"
            f"height: {self.height}\n"
            f")"
        )

    @property
    @override
    def params(self) -> vec4f:
        return vec4f(self.depth, self.width, self.height, 0.0)


class EllipsoidShape(ShapeDescriptor):
    def __init__(self, a: float, b: float, c: float, name: str = "ellipsoid", uuid: str | None = None):
        super().__init__(ShapeType.ELLIPSOID, name)
        self.a = a
        self.b = b
        self.c = c

    @override
    def __repr__(self):
        return f"EllipsoidShape(\nname: {self.name},\nuid: {self.uid},\na: {self.a},\nb: {self.b},\nc: {self.c}\n)"

    @property
    @override
    def params(self) -> vec4f:
        return vec4f(self.a, self.b, self.c, 0.0)


class PlaneShape(ShapeDescriptor):
    def __init__(self, normal: list[float], distance: float, name: str = "plane", uuid: str | None = None):
        super().__init__(ShapeType.PLANE, name)
        self.normal = normal
        self.distance = distance

    @override
    def __repr__(self):
        return (
            f"PlaneShape(\nname: {self.name},\nuid: {self.uid},\nnormal: {self.normal},\ndistance: {self.distance}\n)"
        )

    @property
    @override
    def params(self) -> vec4f:
        return vec4f(self.normal[0], self.normal[1], self.normal[2], self.distance)


###
# Explicit Shapes
###


# TODO: Define ConvexData class to hold references to unique convex meshes
class ConvexShape(ShapeDescriptor):
    def __init__(self, sdf: np.ndarray, name: str = "convex"):
        super().__init__(ShapeType.CONVEX, name)
        pass

    @override
    def __repr__(self):
        return f"ConvexShape(\nname: {self.name},\nuid: {self.uid}\nparams: {self.params}\n)"

    # TODO: What should these parameters be?
    @property
    @override
    def params(self) -> vec4f:
        return vec4f(0.0, 0.0, 0.0, 0.0)


class MeshShape(ShapeDescriptor):
    def __init__(
        self,
        vertices: Sequence[Vec3] | nparray,
        indices: Sequence[int] | nparray,
        normals: Sequence[Vec3] | nparray | None = None,
        uvs: Sequence[Vec2] | nparray | None = None,
        color: Vec3 | None = None,
        maxhullvert: int = MESH_MAXHULLVERT,
        compute_inertia: bool = True,
        is_solid: bool = True,
        name: str = "mesh",
    ):
        super().__init__(ShapeType.MESH, name)
        self._data: Mesh = Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
            is_solid=is_solid,
            maxhullvert=maxhullvert,
            color=color,
        )

    @override
    def __repr__(self):
        return (
            f"MeshShape(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"params: {self.params},\n"
            # f"vertices: {self.vertices.shape},\n"
            # f"normals: {self.normals.shape if self.normals is not None else None},\n"
            # f"triangles: {self.triangles.shape if self.triangles is not None else None},\n"
            # f"triangle_normals: {self.triangle_normals.shape if self.triangle_normals is not None else None}\n"
            f")"
        )

    # TODO: What should these parameters be?
    @property
    @override
    def params(self) -> vec4f:
        return vec4f(0.0, 0.0, 0.0, 0.0)


# TODO: Define SDFData class to hold references to unique SDFs
class SDFShape(ShapeDescriptor):
    def __init__(self, sdf: np.ndarray, name: str = "sdf"):
        super().__init__(ShapeType.SDF, name)
        pass

    @override
    def __repr__(self):
        return f"SDFShape(\nname: {self.name},\nuid: {self.uid},\nparams: {self.params}\n)"

    # TODO: What should these parameters be?
    @property
    @override
    def params(self) -> vec4f:
        return vec4f(0.0, 0.0, 0.0, 0.0)


ShapeDescriptorType = (
    None
    | EmptyShape
    | SphereShape
    | CylinderShape
    | ConeShape
    | CapsuleShape
    | BoxShape
    | EllipsoidShape
    | PlaneShape
    | ConvexShape
    | MeshShape
    | SDFShape
)
"""Type that can be used to represent any shape descriptor, including primitive and explicit shapes."""
