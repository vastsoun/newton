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

"""Provides definitions of core joint types & containers"""

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import warp as wp
from warp._src.types import Any, Int, Vector

from .math import FLOAT32_MAX, FLOAT32_MIN, PI, TWO_PI
from .types import (
    ArrayLike,
    Descriptor,
    mat33f,
    override,
    quatf,
    transformf,
    vec1f,
    vec1i,
    vec2f,
    vec2i,
    vec3f,
    vec3i,
    vec4f,
    vec4i,
    vec5i,
    vec6i,
    vec7f,
)

###
# Module interface
###

__all__ = [
    "JointActuationType",
    "JointCorrectionMode",
    "JointDescriptor",
    "JointDoFType",
    "JointsData",
    "JointsModel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Enumerations
###


class JointActuationType(IntEnum):
    """
    An enumeration of the joint actuation types.
    """

    PASSIVE = 0
    """Passive joint type, i.e. not actuated."""

    FORCE = 1
    """Force-controlled joint type, i.e. actuated by set of forces and/or torques."""

    POSITION = 2
    """Position-controlled joint type, i.e. actuated by set of joint coordinates."""

    VELOCITY = 3
    """Velocity-controlled joint type, i.e. actuated by set of joint velocities."""

    POSITION_VELOCITY = 4
    """Position-velocity-controlled joint type, i.e. actuated by set of joint coordinates and velocities."""

    @override
    def __str__(self):
        """Returns a string representation of the joint actuation type."""
        return f"JointActuationType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint actuation type."""
        return self.__str__()


class JointCorrectionMode(IntEnum):
    """
    An enumeration of the correction modes applicable to rotational joint coordinates.
    """

    TWOPI = 0
    """
    Rotational joint coordinates are computed to always lie within ``[-2*pi, 2*pi]``.\n
    This is the default correction mode for all joints with rotational DoFs.
    """

    CONTINUOUS = 1
    """
    Rotational joint coordinates are continuously accumulated and thus unbounded.\n
    This means that joint coordinates can increase/decrease indefinitely over time,
    but are limited to numerical precision limits (i.e. ``[-FLOAT32_MAX, FLOAT32_MAX]``).
    """

    NONE = -1
    """
    No joint coordinate correction is applied.\n
    Rotational joint coordinates are computed to lie within ``[-pi, pi]``.
    """

    @property
    def bound(self) -> float:
        """
        Returns the numerical bound imposed by the correction mode.
        """
        if self.value == self.TWOPI:
            return float(TWO_PI)
        elif self.value == self.CONTINUOUS:
            return float(FLOAT32_MAX)
        elif self.value == self.NONE:
            return float(PI)
        else:
            raise ValueError(f"Unknown joint correction mode: {self.value}")

    @override
    def __str__(self):
        """Returns a string representation of the joint correction mode."""
        return f"JointCorrectionMode.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint correction mode."""
        return self.__str__()


class JointDoFType(IntEnum):
    """
    An enumeration of the supported joint Degrees-of-Freedom (DoF) types.

    Joint "DoFs" are defined as the local directions of admissible motion, and
    thus  always equal `num_dofs = 6 - num_cts`, where `6` are the number of
    DoFs for unconstrained rigid motions in SE(3) and `num_cts` is the number
    of bilateral equality constraints imposed by the joint. Thus DoFs can be
    intuited as corresponding to the velocity-level description of the motion.

    Joint "coordinates" are defined as the variables used to parameterize the
    space of configurations (i.e. translations and rotations) admissible by
    the joint. Thus, the number of coordinates `num_coords` is generally not
    equal to the number of DoFs `num_dofs`, i.e. `num_coords != num_dofs`,
    since joints may use redundant or non-minimal parameterizations. For example,
    a spherical joint has `num_dofs = 3` underlying DoFs (at velocity-level),
    yet it is commonly parameterized using a 4D unit-quaternion, i.e.
    `num_coords = 4` at configuration-level.

    This class also provides property methods to query the number of:
    - Generalized coordinates
    - Degrees of Freedom (DoFs)
    - Equality constraints

    Conventions:
    - Each joint connects a Base body `B` to a Follower body `F`.
    - The relative motion of body `F' w.r.t. body `B` defines the positive direction of the joint's DoFs.
    - `R_x`, `R_y`, `R_z`: denote rotational DoFs about the local x, y, z axes respectively.
    - `T_x`, `T_y`, `T_z`: denote translational DoFs along the local x, y, z axes respectively.
    - Joints are indexed by `j`, and we often employ the subscript notation `*_j`.
    - `c_j` | `num_coords`: denote the number of generalized coordinates defined by joint `j`.
    - `d_j` | `num_dofs`: denote the number of DoFs defined by joint `j`.
    - `e_j` | `num_dynamic_cts`: denote the number of dynamic equality constraints imposed by joint `j`.
    - `f_j` | `num_kinematic_cts`: denote the number of kinematic equality constraints imposed by joint `j`.
    """

    FREE = 0
    """
    A 6-DoF free-floating joint, with rotational + translational DoFs
    along {`R_x`, `R_y`, `R_z`, `T_x`, `T_y`, `T_z`}.

    Coordinates:
        7D transform: 3D position + 4D unit quaternion
    DoFs:
        6D twist: 3D angular velocity + 3D linear velocity
    Constraints:
        None
    """

    REVOLUTE = 1
    """
    A 1-DoF revolute joint, with rotational DoF along {`R_x`}.

    Coordinates:
        1D angle: {`R_x`}
    DoFs:
        1D angular velocity: {`R_x`}
    Constraints:
        5D vector: {`T_x`, `T_y`, `T_z`, `R_y`, `R_z`}
    """

    PRISMATIC = 2
    """
    A 1-DoF prismatic joint, with translational DoF along {`T_x`}.

    Coordinates:
        1D distance: {`T_x`}
    DoFs:
        1D linear velocity: {`T_x`}
    Constraints:
        5D vector: {`T_y`, `T_z`, `R_x`, `R_y`, `R_z`}
    """

    CYLINDRICAL = 3
    """
    A 2-DoF cylindrical joint, with rotational + translational DoFs along {`R_x`, `T_x`}.

    Coordinates:
        2D vector of angle {`R_x`} + 1D distance {`T_x`}
    DoFs:
        2D vector of angular velocity {`R_x`} + linear velocity {`T_x`}
    """

    UNIVERSAL = 4
    """
    A 2-DoF universal joint, with rotational DoFs along {`R_x`, `R_y`}.

    This universal joint is implemented as being equivalent to two consecutive
    revolute joints, rotating an intermediate (virtual) body about `R_x` w.r.t
    the Base body `B`, then rotating the Follower body `F` about `R_y` of the
    intermediate body. Thus, this implementation necessarily assumes the first
    rotation is always about `R_x` followed by the rotation about `R_y`.

    Coordinates:
        2D angles: {`R_x`, `R_y`}
    DoFs:
        2D angular velocities: {`R_x`, `R_y`}
    Constraints:
        4D vector: {`T_x`, `T_y`, `T_z`, `R_z`}
    """

    SPHERICAL = 5
    """
    A 3-DoF spherical joint, with rotational DoFs along {`R_x`, `R_y`, `R_z`}.

    Coordinates:
        4D unit-quaternion to parameterize {`R_x`, `R_y`, `R_z`}
    DoFs:
        3D angular velocities: {`R_x`, `R_y`, `R_z`}
    Constraints:
        3D vector: {`T_x`, `T_y`, `T_z`}
    """

    GIMBAL = 6
    """
    A 3-DoF gimbal joint, with rotational DoFs along {`R_x`, `R_y`, `R_z`}.

    **DISCLAIMER**: This joint is not yet fully supported, and currently behaves
    identically to the SPHERICAL joint. We do not recommend using it at present time.

    Coordinates:
        3D euler angles: {`R_x`, `R_y`, `R_z`}
    DoFs:
        3D angular velocities: {`R_x`, `R_y`, `R_z`}
    Constraints:
        3D vector: {`T_x`, `T_y`, `T_z`}
    """

    CARTESIAN = 7
    """
    A 3-DoF Cartesian joint, with translational DoFs along {`T_x`, `T_y`, `T_z`}.

    Coordinates:
        3D distances: {`T_x`, `T_y`, `T_z`}
    DoFs:
        3D linear velocities: {`T_x`, `T_y`, `T_z`}
    Constraints:
        3D vector: {`R_x`, `R_y`, `R_z`}
    """

    FIXED = 8
    """
    A 0-DoF fixed joint, fully constraining the relative motion between the connected bodies.

    Coordinates:
        None
    DoFs:
        None
    Constraints:
        6D vector: {`T_x`, `T_y`, `T_z`, `R_x`, `R_y`, `R_z`}
    """

    @override
    def __str__(self):
        """Returns a string representation of the joint DoF type."""
        return f"JointDoFType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint DoF type."""
        return self.__str__()

    @property
    def num_coords(self) -> int:
        """
        Returns the number of generalized coordinates defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 7  # 3D position + 4D quaternion
        elif self.value == self.REVOLUTE:
            return 1  # 1D angle
        elif self.value == self.PRISMATIC:
            return 1  # 1D distance
        elif self.value == self.CYLINDRICAL:
            return 2  # 2D vector of angle + distance
        elif self.value == self.UNIVERSAL:
            return 2  # 2D angles
        elif self.value == self.SPHERICAL:
            return 4  # 4D unit-quaternion
        elif self.value == self.GIMBAL:
            return 3  # 3D euler angles
        elif self.value == self.CARTESIAN:
            return 3  # 3D distances
        elif self.value == self.FIXED:
            return 0  # None
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def num_dofs(self) -> int:
        """
        Returns the number of DoFs defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 6  # 3D angular velocity + 3D linear velocity
        elif self.value == self.REVOLUTE:
            return 1  # 1D angular velocity
        elif self.value == self.PRISMATIC:
            return 1  # 1D linear velocity
        elif self.value == self.CYLINDRICAL:
            return 2  # 1D angular velocity + 1D linear velocity
        elif self.value == self.UNIVERSAL:
            return 2  # 2D angular velocities
        elif self.value == self.SPHERICAL:
            return 3  # 3D angular velocities
        elif self.value == self.GIMBAL:
            return 3  # 3D angular velocities
        elif self.value == self.CARTESIAN:
            return 3  # 3D linear velocities
        elif self.value == self.FIXED:
            return 0  # None
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def num_cts(self) -> int:
        """
        Returns the number of constraints defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 0  # None
        elif self.value == self.REVOLUTE:
            return 5  # 5D vector for `{T_x, T_y, T_z, R_y, R_z}`
        elif self.value == self.PRISMATIC:
            return 5  # 5D vector for `{T_x, T_y, T_z, R_y, R_z}`
        elif self.value == self.CYLINDRICAL:
            return 4  # 4D vector for `{T_x, T_y, R_y, R_z}`
        elif self.value == self.UNIVERSAL:
            return 4  # 4D vector for `{R_x, R_y, R_z, R_w}`
        elif self.value == self.SPHERICAL:
            return 3  # 3D vector for `{R_x, R_y, R_z}`
        elif self.value == self.GIMBAL:
            return 3  # 3D vector for `{R_x, R_y, R_z}`
        elif self.value == self.CARTESIAN:
            return 3  # 3D vector for `{T_x, T_y, T_z}`
        elif self.value == self.FIXED:
            return 6  # 6D vector for `{T_x, T_y, T_z, R_x, R_y, R_z}`
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def cts_axes(self) -> Vector[Any, Int]:
        """
        Returns the indices of the joint's constraint axes.
        """
        if self.value == self.FREE:
            return []  # Empty vector (TODO: wp.constant(vec0i()))
        if self.value == self.REVOLUTE:
            return wp.constant(vec5i(0, 1, 2, 4, 5))
        elif self.value == self.PRISMATIC:
            return wp.constant(vec5i(1, 2, 3, 4, 5))
        elif self.value == self.CYLINDRICAL:
            return wp.constant(vec4i(1, 2, 4, 5))
        elif self.value == self.UNIVERSAL:
            return wp.constant(vec4i(0, 1, 2, 5))
        elif self.value == self.SPHERICAL:
            return wp.constant(vec3i(0, 1, 2))
        elif self.value == self.GIMBAL:
            return wp.constant(vec3i(0, 1, 2))
        elif self.value == self.CARTESIAN:
            return wp.constant(vec3i(3, 4, 5))
        elif self.value == self.FIXED:
            return wp.constant(vec6i(0, 1, 2, 3, 4, 5))
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def dofs_axes(self) -> Vector[Any, Int]:
        """
        Returns the indices of the joint's DoF axes.
        """
        if self.value == self.FREE:
            return wp.constant(vec6i(0, 1, 2, 3, 4, 5))
        if self.value == self.REVOLUTE:
            return wp.constant(vec1i(3))
        elif self.value == self.PRISMATIC:
            return wp.constant(vec1i(0))
        elif self.value == self.CYLINDRICAL:
            return wp.constant(vec2i(0, 3))
        elif self.value == self.UNIVERSAL:
            return wp.constant(vec2i(3, 4))
        elif self.value == self.SPHERICAL:
            return wp.constant(vec3i(3, 4, 5))
        elif self.value == self.GIMBAL:
            return wp.constant(vec3i(3, 4, 5))
        elif self.value == self.CARTESIAN:
            return wp.constant(vec3i(0, 1, 2))
        elif self.value == self.FIXED:
            return []  # Empty vector (TODO: wp.constant(vec0i()))
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def coords_storage_type(self) -> Any:
        """
        Returns the data type required to store the joint's generalized coordinates.
        """
        if self.value == self.FREE:
            return vec7f
        elif self.value == self.REVOLUTE:
            return vec1f
        elif self.value == self.PRISMATIC:
            return vec1f
        elif self.value == self.CYLINDRICAL:
            return vec2f
        elif self.value == self.UNIVERSAL:
            return vec2f
        elif self.value == self.SPHERICAL:
            return vec4f
        elif self.value == self.GIMBAL:
            return vec3f
        elif self.value == self.CARTESIAN:
            return vec3f
        elif self.value == self.FIXED:
            return None
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def coords_physical_type(self) -> Any:
        """
        Returns the data type required to represent the joint's generalized coordinates.
        """
        if self.value == self.FREE:
            return transformf
        elif self.value == self.REVOLUTE:
            return vec1f
        elif self.value == self.PRISMATIC:
            return vec1f
        elif self.value == self.CYLINDRICAL:
            return vec2f
        elif self.value == self.UNIVERSAL:
            return vec2f
        elif self.value == self.SPHERICAL:
            return quatf
        elif self.value == self.GIMBAL:
            return vec3f
        elif self.value == self.CARTESIAN:
            return vec3f
        elif self.value == self.FIXED:
            return None
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def reference_coords(self) -> list[float]:
        """
        Returns the data type required to represent the joint's generalized coordinates.
        """
        if self.value == self.FREE:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        elif self.value == self.REVOLUTE:
            return [0.0]
        elif self.value == self.PRISMATIC:
            return [0.0]
        elif self.value == self.CYLINDRICAL:
            return [0.0, 0.0]
        elif self.value == self.UNIVERSAL:
            return [0.0, 0.0]
        elif self.value == self.SPHERICAL:
            return [0.0, 0.0, 0.0, 1.0]
        elif self.value == self.GIMBAL:
            return [0.0, 0.0, 0.0]
        elif self.value == self.CARTESIAN:
            return [0.0, 0.0, 0.0]
        elif self.value == self.FIXED:
            return []
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    def coords_bound(self, correction: JointCorrectionMode) -> list[float]:
        """
        Returns a list of numeric bounds for the generalized coordinates,
        of the joint DoF type, imposed by the specified correction mode.
        """
        rotation_bound = correction.bound

        if self.value == self.FREE:
            return [FLOAT32_MAX] * 7
        elif self.value == self.REVOLUTE:
            return [rotation_bound]
        elif self.value == self.PRISMATIC:
            return [float(FLOAT32_MAX)]
        elif self.value == self.CYLINDRICAL:
            return [float(FLOAT32_MAX), rotation_bound]
        elif self.value == self.UNIVERSAL:
            return [rotation_bound, rotation_bound]
        elif self.value == self.SPHERICAL:
            return [float(FLOAT32_MAX)] * 4
        elif self.value == self.GIMBAL:
            return [rotation_bound] * 3
        elif self.value == self.CARTESIAN:
            return [float(FLOAT32_MAX)] * 3
        elif self.value == self.FIXED:
            return []
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")


###
# Containers
###


@dataclass
class JointDescriptor(Descriptor):
    """
    A container to describe a single joint in the model builder.
    """

    ###
    # Attributes
    ###

    act_type: JointActuationType = JointActuationType.PASSIVE
    """Actuation type of the joint."""

    dof_type: JointDoFType = JointDoFType.FREE
    """DoF type of the joint."""

    bid_B: int = -1
    """
    The Base body index of the joint (-1 for world, >=0 for bodies).\n
    Defaults to `-1`, indicating that the joint has not been assigned a base body.
    """

    bid_F: int = -1
    """
    The Follower body index of the joint (must always be >=0 to index a body).\n
    Defaults to `-1`, indicating that the joint has not been assigned a follower body.
    """

    # TODO: Change this to a transformf
    B_r_Bj: vec3f = field(default_factory=vec3f)
    """The relative position of the joint in the base body coordinates."""

    # TODO: Change this to a transformf
    F_r_Fj: vec3f = field(default_factory=vec3f)
    """The relative position of the joint in the follower body coordinates."""

    # TODO: Remove this when body offsets become transforms
    X_j: mat33f = field(default_factory=mat33f)
    """The constant axes matrix of the joint."""

    q_j_min: ArrayLike | float | None = None
    """
    Minimum DoF limits of the joint.

    If `None`, then no limits are applied to the joint DoFs,
    and the maximum limits default to `-inf` for lower limits.

    If specified as a single float value, it will
    be applied uniformly to all DoFs of the joint.

    If specified as a type conforming to the `ArrayLike`
    union, then the number of elements must equal number of
    DoFs of the joint, i.e. `num_dofs = dof_type.num_dofs`.

    For rotational DoFs, limits are expected in radians,
    while for translational DoFs, limits are expected in
    the same units as the world units.

    **Warning**:
    These limits are dimensioned according to the number of `num_dofs`,
    even though joint coordinates are actually dimensioned according to
    `num_coords`. This is because some joints (e.g. SPHERICAL) may use
    redundant or non-minimal parameterizations at configuration-level.
    In order to support configuration-level limits regardless of the
    underlying parameterization, a mapping is performed in the solver
    that translates the limits from DoF space to coordinate space.
    """

    q_j_max: ArrayLike | float | None = None
    """
    Maximum DoF limits of the joint.

    If `None`, then no limits are applied to the joint DoFs,
    and the maximum limits default to `-inf` for lower limits.

    If specified as a single float value, it will
    be applied uniformly to all DoFs of the joint.

    If specified as a type conforming to the `ArrayLike`
    union, then the number of elements must equal number of
    DoFs of the joint, i.e. `num_dofs = dof_type.num_dofs`.

    **Warning**:
    These limits are dimensioned according to the number of `num_dofs`,
    even though joint coordinates are actually dimensioned according to
    `num_coords`. This is because some joints (e.g. SPHERICAL) may use
    redundant or non-minimal parameterizations at configuration-level.
    In order to support configuration-level limits regardless of the
    underlying parameterization, a mapping is performed in the solver
    that translates the limits from DoF space to coordinate space.
    """

    dq_j_max: ArrayLike | float | None = None
    """
    Maximum velocity limits of the joint.

    If `None`, then no limits are applied
    to the joint's generalized velocities.

    If specified as a single float value, it will
    be applied uniformly to all DoFs of the joint.

    If specified as a type conforming to the `ArrayLike`
    union, then the number of elements must equal number of
    DoFs of the joint, i.e. `num_dofs = dof_type.num_dofs`.
    """

    tau_j_max: ArrayLike | float | None = None
    """
    Maximum effort (i.e. generalized force) limits of the joint.

    If `None`, then no limits are applied
    to the joint's generalized forces.

    If specified as a single float value, it will
    be applied uniformly to all DoFs of the joint.

    If specified as a type conforming to the `ArrayLike`
    union, then the number of elements must equal number of
    DoFs of the joint, i.e. `num_dofs = dof_type.num_dofs`.
    """

    a_j: ArrayLike | float | None = None
    """
    Internal inertia of the joint (a.k.a. joint armature),
    used for implicit integration of joint dynamics.

    This represents effects like rotor inertia of rotary motors,
    potentially transferred over a transmission, and compounding
    the inertia of the gearbox. This is often referred to as so
    called "reflected inertia" of an actuator as seen at the joint.

    If specified as a type conforming to the `ArrayLike`
    union, then the number of elements must equal number of
    DoFs of the joint, i.e. `num_dofs = dof_type.num_dofs`.

    Defaults to `[0.0] * num_dofs` if not specified, indicating
    that the joint has no internal inertia and is thus massless.
    """

    b_j: ArrayLike | float | None = None
    """
    Internal damping of the joint used for implicit integration of joint dynamics.

    This represents effects like viscous friction in rotary motors,
    potentially transferred over a transmission, and compounding
    the friction of the gearbox.

    If specified as a type conforming to the `ArrayLike`
    union, then the number of elements must equal number of
    DoFs of the joint, i.e. `num_dofs = dof_type.num_dofs`.

    Defaults to `[0.0] * num_dofs` if not specified, indicating
    that the joint has no internal damping and is thus frictionless.
    """

    k_p_j: ArrayLike | float | None = None
    """
    Implicit PD-control proportional gain.

    If specified as a type conforming to the `ArrayLike`
    union, then the number of elements must equal number of
    DoFs of the joint, i.e. `num_dofs = dof_type.num_dofs`.

    Defaults to `[0.0] * num_dofs` if not specified, indicating
    that the joint has no implicit proportional gain.
    """

    k_d_j: ArrayLike | float | None = None
    """
    Implicit PD-control derivative gain.

    If specified as a type conforming to the `ArrayLike`
    union, then the number of elements must equal number of
    DoFs of the joint, i.e. `num_dofs = dof_type.num_dofs`.

    Defaults to `[0.0] * num_dofs` if not specified, indicating
    that the joint has no implicit derivative gain.
    """

    ###
    # Metadata - to be set by the WorldDescriptor when added
    ###

    wid: int = -1
    """
    Index of the world to which the joint belongs.\n
    Defaults to `-1`, indicating that the joint has not yet been added to a world.
    """

    jid: int = -1
    """
    Index of the joint w.r.t. its world.\n
    Defaults to `-1`, indicating that the joint has not yet been added to a world.
    """

    coords_offset: int = -1
    """Index offset of this joint's coordinates among all joint coordinates in the world it belongs to."""

    dofs_offset: int = -1
    """
    Index offset of this joint's DoFs among
    all joint DoFs in the world it belongs to.
    """

    passive_coords_offset: int = -1
    """
    Index offset of this joint's passive coordinates among all
    passive joint coordinates in the world it belongs to.
    """

    passive_dofs_offset: int = -1
    """
    Index offset of this joint's passive DoFs among all
    passive joint DoFs in the world it belongs to.
    """

    actuated_coords_offset: int = -1
    """
    Index offset of this joint's actuated coordinates among
    all actuated joint coordinates in the world it belongs to.
    """

    actuated_dofs_offset: int = -1
    """
    Index offset of this joint's actuated DoFs among
    all actuated joint DoFs in the world it belongs to.
    """

    cts_offset: int = -1
    """
    Index offset of this joint's constraints among all
    joint constraints in the world it belongs to.
    """

    dynamic_cts_offset: int = -1
    """
    Index offset of this joint's dynamic constraints among all
    dynamic joint constraints in the world it belongs to.
    """

    kinematic_cts_offset: int = -1
    """
    Index offset of this joint's kinematic constraints among all
    kinematic joint constraints in the world it belongs to.
    """

    ###
    # Properties
    ###

    @property
    def num_coords(self) -> int:
        """
        Returns the number of coordinates for this joint.
        """
        return self.dof_type.num_coords

    @property
    def num_dofs(self) -> int:
        """
        Returns the number of DoFs for this joint.
        """
        return self.dof_type.num_dofs

    @property
    def num_passive_coords(self) -> int:
        """
        Returns the number of passive coordinates for this joint.
        """
        return self.dof_type.num_coords if self.is_passive else 0

    @property
    def num_passive_dofs(self) -> int:
        """
        Returns the number of passive DoFs for this joint.
        """
        return self.dof_type.num_dofs if self.is_passive else 0

    @property
    def num_actuated_coords(self) -> int:
        """
        Returns the number of actuated coordinates for this joint.
        """
        return self.dof_type.num_coords if self.is_actuated else 0

    @property
    def num_actuated_dofs(self) -> int:
        """
        Returns the number of actuated DoFs for this joint.
        """
        return self.dof_type.num_dofs if self.is_actuated else 0

    @property
    def num_cts(self) -> int:
        """
        Returns the total number of constraints introduced by this joint.
        """
        return self.num_dynamic_cts + self.num_kinematic_cts

    @property
    def num_dynamic_cts(self) -> int:
        """
        Returns the number of dynamic constraints introduced by this joint.
        """
        return self.dof_type.num_dofs if self.is_dynamic else 0

    @property
    def num_kinematic_cts(self) -> int:
        """
        Returns the number of kinematic constraints introduced by this joint.
        """
        return self.dof_type.num_cts

    @property
    def is_binary(self) -> bool:
        """
        Returns whether the joint is binary (i.e. connected to two bodies).
        """
        return self.bid_B != -1 and self.bid_F != -1

    @property
    def is_unary(self) -> bool:
        """
        Returns whether the joint is unary (i.e. connected to the world).
        """
        return self.bid_B == -1 or self.bid_F == -1

    @property
    def is_passive(self) -> bool:
        """
        Returns whether the joint is passive.
        """
        return self.act_type == JointActuationType.PASSIVE

    @property
    def is_actuated(self) -> bool:
        """
        Returns whether the joint is actuated.
        """
        return self.act_type > JointActuationType.PASSIVE

    @property
    def is_dynamic(self) -> bool:
        """
        Returns whether the joint's dynamics is simulated implicitly.
        """
        return np.any(self.a_j) or np.any(self.b_j)

    @property
    def is_implicit_pd(self) -> bool:
        """
        Returns whether the joint's dynamics is simulated using implicit PD control.
        """
        return np.any(self.k_p_j) or np.any(self.k_d_j)

    def has_base_body(self, bid: int) -> bool:
        """
        Returns whether the joint has assigned the specified body as Base.

        The body index `bid` must be given w.r.t the world.
        """
        return self.bid_B == bid

    def has_follower_body(self, bid: int) -> bool:
        """
        Returns whether the joint has assigned the specified body as Follower.

        The body index `bid` must be given w.r.t the world.
        """
        return self.bid_F == bid

    def is_connected_to_body(self, bid: int) -> bool:
        """
        Returns whether the joint is connected to the specified body.

        The body index `bid` must be given w.r.t the world.
        """
        return self.has_base_body(bid) or self.has_follower_body(bid)

    ###
    # Operations
    ###

    def __post_init__(self):
        """Post-initialization processing to validate and set up joint limits."""
        # Ensure base descriptor post-init is called first
        # NOTE: This ensures that the UID is properly set before proceeding
        super().__post_init__()

        # Check if DoF type + actuation type are compatible
        if self.dof_type == JointDoFType.FREE and self.is_binary:
            raise ValueError(
                f"Invalid joint configuration: FREE joints cannot be binary (name={self.name}, uid={self.uid})."
            )
        if self.act_type == JointActuationType.FORCE and self.dof_type == JointDoFType.FIXED:
            raise ValueError(
                f"Invalid joint configuration: FIXED joints cannot be actuated (name={self.name}, uid={self.uid})."
            )

        # Check if DoF type + dynamic/implicit PD settings are compatible
        if self.is_implicit_pd and self.dof_type == JointDoFType.FREE:
            raise ValueError(
                f"Invalid joint configuration: FREE joints cannot have implicit PD gains (name={self.name}, uid={self.uid})."
            )
        if self.is_dynamic and self.dof_type == JointDoFType.FIXED:
            raise ValueError(
                f"Invalid joint configuration: FIXED joints cannot be dynamic (name={self.name}, uid={self.uid})."
            )
        if self.is_implicit_pd and self.dof_type == JointDoFType.FIXED:
            raise ValueError(
                f"Invalid joint configuration: FIXED joints cannot have implicit PD gains (name={self.name}, uid={self.uid})."
            )

        # Set default values for joint limits if not provided
        self.q_j_min = self._check_dofs_array(self.q_j_min, self.num_dofs, float(FLOAT32_MIN))
        self.q_j_max = self._check_dofs_array(self.q_j_max, self.num_dofs, float(FLOAT32_MAX))
        self.dq_j_max = self._check_dofs_array(self.dq_j_max, self.num_dofs, float(FLOAT32_MAX))
        self.tau_j_max = self._check_dofs_array(self.tau_j_max, self.num_dofs, float(FLOAT32_MAX))

        # Set default values for internal inertia, damping, and implicit PD gains if not provided
        self.a_j = self._check_dofs_array(self.a_j, self.num_dofs, 0.0)
        self.b_j = self._check_dofs_array(self.b_j, self.num_dofs, 0.0)
        self.k_p_j = self._check_dofs_array(self.k_p_j, self.num_dofs, 0.0)
        self.k_d_j = self._check_dofs_array(self.k_d_j, self.num_dofs, 0.0)

        # Validate that the specified parameters are valid
        self._check_parameter_values()

        # Ensure that PD gains are only specified for actuated joints
        if self.is_passive and (np.any(self.k_p_j) or np.any(self.k_d_j)):
            raise ValueError(
                f"Joint `{self.name}` has non-zero PD gains but the joint is defined as passive:"
                f"\n  k_p_j: {self.k_p_j}"
                f"\n  k_d_j: {self.k_d_j}"
            )
        if self.is_implicit_pd and not (np.any(self.k_p_j) or np.any(self.k_d_j)):
            raise ValueError(
                f"Joint `{self.name}` is defined as implicit PD but has zero-valued PD gains:"
                f"\n  k_p_j: {self.k_p_j}"
                f"\n  k_d_j: {self.k_d_j}"
            )
        if self.act_type == JointActuationType.FORCE and (np.any(self.k_p_j) or np.any(self.k_d_j)):
            raise ValueError(
                f"Joint `{self.name}` is defined as FORCE actuated but has non-zero PD gains:"
                f"\n  k_p_j: {self.k_p_j}"
                f"\n  k_d_j: {self.k_d_j}"
            )
        if self.act_type == JointActuationType.POSITION_VELOCITY and not (np.any(self.k_p_j) or np.any(self.k_d_j)):
            raise ValueError(
                f"Joint `{self.name}` is defined as POSITION_VELOCITY actuated but has zero-valued PD gains:"
                f"\n  k_p_j: {self.k_p_j}"
                f"\n  k_d_j: {self.k_d_j}"
            )

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the JointDescriptor."""
        return (
            f"JointDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            "----------------------------------------------\n"
            f"act_type: {self.act_type},\n"
            f"dof_type: {self.dof_type},\n"
            "----------------------------------------------\n"
            f"bid_B: {self.bid_B},\n"
            f"bid_F: {self.bid_F},\n"
            "----------------------------------------------\n"
            f"B_r_Bj: {self.B_r_Bj},\n"
            f"F_r_Fj: {self.F_r_Fj},\n"
            f"X_j:\n{self.X_j},\n"
            "----------------------------------------------\n"
            f"q_j_min: {self.q_j_min},\n"
            f"q_j_max: {self.q_j_max},\n"
            f"dq_j_max: {self.dq_j_max},\n"
            f"tau_j_max: {self.tau_j_max}\n"
            "----------------------------------------------\n"
            f"a_j: {self.a_j},\n"
            f"b_j: {self.b_j},\n"
            f"k_p_j: {self.k_p_j},\n"
            f"k_d_j: {self.k_d_j},\n"
            "----------------------------------------------\n"
            f"wid: {self.wid},\n"
            f"jid: {self.jid},\n"
            "----------------------------------------------\n"
            f"num_coords: {self.num_coords},\n"
            f"num_dofs: {self.num_dofs},\n"
            f"num_dynamic_cts: {self.num_dynamic_cts},\n"
            f"num_kinematic_cts: {self.num_kinematic_cts},\n"
            "----------------------------------------------\n"
            f"coords_offset: {self.coords_offset},\n"
            f"dofs_offset: {self.dofs_offset},\n"
            f"cts_dynamic_offset: {self.dynamic_cts_offset},\n"
            f"cts_kinematic_offset: {self.kinematic_cts_offset},\n"
            "----------------------------------------------\n"
            f"passive_coords_offset: {self.passive_coords_offset},\n"
            f"passive_dofs_offset: {self.passive_dofs_offset},\n"
            f"actuated_coords_offset: {self.actuated_coords_offset},\n"
            f"actuated_dofs_offset: {self.actuated_dofs_offset},\n"
            f")"
        )

    ###
    # Operations - Internal
    ###

    @staticmethod
    def _check_dofs_array(x: ArrayLike | float | None, size: int, default: float = float(FLOAT32_MAX)) -> list[float]:
        """
        Processes a specified limit value to ensure it is a list of floats.

        Notes:
        - If the input is None, a list of default values is returned.
        - If the input is a single float, it is converted to a list of the specified length.
        - If the input is an empty list, a list of default values is returned.
        - If the input is a non-empty list, it is validated to ensure it
            contains only floats and matches the specified length.

        Args:
            x (List[float] | float | None): The DOF array to be processed.
            size (int): The number of degrees of freedom to determine the length of the output list.
            default (float): The default value to use if DOF array is None or an empty list.

        Returns:
            List[float]: The processed list of DOF values.

        Raises:
            ValueError: If the length of the DOF array does not match num_dofs.
            TypeError: If the DOF array contains non-float types.
        """
        if x is None:
            return [float(default) for _ in range(size)]

        if isinstance(x, (int, float, np.floating)):
            if x == math.inf:
                return [float(FLOAT32_MAX) for _ in range(size)]
            elif x == -math.inf:
                return [float(FLOAT32_MIN) for _ in range(size)]
            else:
                return [x] * size

        if isinstance(x, ArrayLike):
            if len(x) == 0:
                return [float(default) for _ in range(size)]

            if len(x) != size:
                raise ValueError(f"Invalid DOF array length: {len(x)} != {size}")

            if all(isinstance(x, (float, np.floating)) for x in x):
                for i in range(len(x)):
                    if x[i] == math.inf:
                        x[i] = float(FLOAT32_MAX)
                    elif x[i] == -math.inf:
                        x[i] = float(FLOAT32_MIN)
                return x
            else:
                raise TypeError(f"Unsupported DOF array type: {type(x)!r}; expected float, iterable of floats, or None")

    def _check_parameter_values(self):
        """
        Validates the joint parameters to ensure they are consistent and within expected ranges.

        Raises:
            ValueError: If any of the joint parameters are invalid, such as:
                - q_j_min >= q_j_max for any DoF
                - dq_j_max <= 0 for any DoF
                - tau_j_max <= 0 for any DoF
                - a_j < 0 for any DoF
                - b_j < 0 for any DoF
                - k_p_j < 0 for any DoF
                - k_d_j < 0 for any DoF
        """
        for i in range(self.num_dofs):
            if self.q_j_min[i] >= self.q_j_max[i]:
                raise ValueError(
                    f"Invalid joint limits: q_j_min[{i}] >= q_j_max[{i}] (name={self.name}, uid={self.uid})."
                )
            if self.dq_j_max[i] <= 0:
                raise ValueError(
                    f"Invalid joint velocity limit: dq_j_max[{i}] <= 0 (name={self.name}, uid={self.uid})."
                )
            if self.tau_j_max[i] <= 0:
                raise ValueError(f"Invalid joint effort limit: tau_j_max[{i}] <= 0 (name={self.name}, uid={self.uid}).")
            if self.a_j[i] < 0:
                raise ValueError(f"Invalid joint armature: a_j[{i}] < 0 (name={self.name}, uid={self.uid}).")
            if self.b_j[i] < 0:
                raise ValueError(f"Invalid joint damping: b_j[{i}] < 0 (name={self.name}, uid={self.uid}).")
            if self.k_p_j[i] < 0:
                raise ValueError(f"Invalid joint proportional gain: k_p_j[{i}] < 0 (name={self.name}, uid={self.uid}).")
            if self.k_d_j[i] < 0:
                raise ValueError(f"Invalid joint derivative gain: k_d_j[{i}] < 0 (name={self.name}, uid={self.uid}).")


@dataclass
class JointsModel:
    """
    An SoA-based container to hold time-invariant model data of joints.
    """

    num_joints: int = 0
    """Total number of joints in the model (host-side)."""

    ###
    # Identifiers
    ###

    wid: wp.array | None = None
    """
    Index each the world in which each joint is defined.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    jid: wp.array | None = None
    """
    Index of each joint w.r.t the world.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    ###
    # Parameterization
    ###

    dof_type: wp.array | None = None
    """
    Joint DoF type ID of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    act_type: wp.array | None = None
    """
    Joint actuation type ID of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    bid_B: wp.array | None = None
    """
    Base body index of each joint w.r.t the model.\n
    Equals `-1` for world, `>=0` for bodies.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    bid_F: wp.array | None = None
    """
    Follower body index of each joint w.r.t the model.\n
    Equals `-1` for world, `>=0` for bodies.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    B_r_Bj: wp.array | None = None
    """
    Relative position of the joint, expressed in and w.r.t the base body coordinate frame.\n
    Shape of ``(num_joints, 3)`` and type :class:`vec3`.
    """

    F_r_Fj: wp.array | None = None
    """
    Relative position of the joint, expressed in and w.r.t the follower body coordinate frame.\n
    Shape of ``(num_joints, 3)`` and type :class:`vec3`.
    """

    X_j: wp.array | None = None
    """
    Joint axes matrix (local coordinates) of each joint.\n
    Indicates the relative orientation of the the joint
    frame w.r.t the base body coordinate frame.\n
    Shape of ``(num_joints, 3, 3)`` and type :class:`mat33`.
    """

    ###
    # Limits
    ###

    q_j_min: wp.array | None = None
    """
    Minimum (a.k.a. lower) joint DoF limits of each joint (as flat array).\n

    Limits are dimensioned according to the number of DoFs of each joint,
    as opposed to the number of coordinates in order to handle cases such
    where joints have more coordinates than DoFs (e.g. spherical joints).\n

    Shape of ``(sum_of_num_joint_dofs,)`` and type :class:`float`,\n
    where `sum_of_num_joint_dofs := sum(d_j)`, and ``d_j``
    is the number of DoFs of joint ``j``.
    """

    q_j_max: wp.array | None = None
    """
    Maximum (a.k.a. upper) joint DoF limits of each joint (as flat array).\n

    Limits are dimensioned according to the number of DoFs of each joint,
    as opposed to the number of coordinates in order to handle cases such
    where joints have more coordinates than DoFs (e.g. spherical joints).\n

    Shape of ``(sum_of_num_joint_dofs,)`` and type :class:`float`,\n
    where `sum_of_num_joint_dofs := sum(d_j)`, and ``d_j``
    is the number of DoFs of joint ``j``.
    """

    dq_j_max: wp.array | None = None
    """
    Maximum joint velocity limits of each joint (as flat array).\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,\n
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    tau_j_max: wp.array | None = None
    """
    Maximum joint torque limits of each joint (as flat array).\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,\n
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    ###
    # Dynamics
    ###

    a_j: wp.array | None = None
    """
    Internal inertia of each joint (as flat array), used for implicit integration of joint dynamics.\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,\n
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    b_j: wp.array | None = None
    """
    Internal damping of each joint (as flat array) used for implicit integration of joint dynamics.\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,\n
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    k_p_j: wp.array | None = None
    """
    Implicit PD-control proportional gain of each joint (as flat array).\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,\n
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    k_d_j: wp.array | None = None
    """
    Implicit PD-control derivative gain of each joint (as flat array).\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,\n
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    ###
    # Initial State
    ###

    q_j_0: wp.array | None = None
    """
    The initial coordinates of each joint (as flat array),
    indicating the "rest" or "neutral" position of each joint.

    These are used for resetting joint positions when multi-turn
    correction for revolute DoFs is enabled in the simulation.

    Shape of ``(sum(c_j),)`` and type :class:`float`,\n
    where ``c_j`` is the number of coordinates of joint ``j``.
    """

    dq_j_0: wp.array | None = None
    """
    The initial velocities of each joint (as flat array),
    indicating the "rest" or "neutral" velocity of each joint.

    These are used for resetting joint velocities when multi-turn
    correction for revolute DoFs is enabled in the simulation.

    Shape of ``(sum(c_j),)`` and type :class:`float`,\n
    where ``c_j`` is the number of coordinates of joint ``j``.
    """

    ###
    # Metadata
    ###

    num_coords: wp.array | None = None
    """
    Number of coordinates of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    num_dofs: wp.array | None = None
    """
    Number of DoFs of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    # TODO: Consider making this a vec2i containing
    # both dynamic and kinematic constraint counts
    num_cts: wp.array | None = None
    """
    Number of total constraints of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    num_dynamic_cts: wp.array | None = None
    """
    Number of dynamic constraints of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    num_kinematic_cts: wp.array | None = None
    """
    Number of kinematic constraints of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    coords_offset: wp.array | None = None
    """
    Index offset of each joint's coordinates w.r.t the start
    index of joint coordinates of the corresponding world.\n

    Used to index into joint-specific blocks of:
    - array of initial joint generalized coordinates :attr:`JointsModel.q_j_0`
    - array of joint generalized coordinates :attr:`JointsData.q_j`
    - array of previous joint generalized coordinates :attr:`JointsData.q_j_p`

    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    dofs_offset: wp.array | None = None
    """
    Index offset of each joint's DoFs w.r.t the start
    index of joint DoFs of the corresponding world.

    Used to index into joint-specific blocks of:
    - array of initial joint generalized velocities :attr:`JointsModel.dq_j_0`
    - array of joint generalized velocities :attr:`JointsData.dq_j`
    - array of joint generalized forces :attr:`JointsData.tau_j`

    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    passive_coords_offset: wp.array | None = None
    """
    Index offset of each joint's passive coordinates w.r.t the start
    index of passive joint coordinates of the corresponding world.\n
    Used to index into passive-specific blocks of flattened passive joint coordinates arrays.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    passive_dofs_offset: wp.array | None = None
    """
    Index offset of each joint's passive DoFs w.r.t the start
    index of passive joint DoFs of the corresponding world.\n
    Used to index into passive-specific blocks of flattened passive joint coordinates and DoFs arrays.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    actuated_coords_offset: wp.array | None = None
    """
    Index offset of each joint's actuated coordinates w.r.t the start
    index of actuated joint coordinates of the corresponding world.\n
    Used to index into actuator-specific blocks of flattened actuator coordinates arrays.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    actuated_dofs_offset: wp.array | None = None
    """
    Index offset of each joint's actuated DoFs w.r.t the start
    index of actuated joint DoFs of the corresponding world.\n
    Used to index into actuator-specific blocks of flattened actuator DoFs arrays.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    cts_offset: wp.array | None = None
    """
    Index offset of each joint's constraints w.r.t the start
    index of constraints of the corresponding world.

    Shape of ``(num_joints,)`` and type :class:`int`.

    Used together with :attr:`ModelInfo.joint_cts_offset`
    to index into the joint-specific blocks of:
    - array of joint constraint Lagrange multipliers `lambda_j`

    For a joint `j` of world `w`, its constraint multipliers can be accessed as:
    ```
    # Retrieve dimensions and start indices
    joint_num_cts = model.joints.num_cts[j]
    world_cts_start_idx = model.info.joint_cts_offset[w]
    joint_cts_start_idx = model.joints.cts_offset[j]

    # Compute the start and end indices for the joint's constraints
    start_idx = world_cts_start_idx + joint_cts_start_idx
    end_idx = start_idx + joint_num_cts

    # Access the joint's constraint multipliers
    lambda_j = lambda_j[start_idx:end_idx]
    ```
    """

    dynamic_cts_offset: wp.array | None = None
    """
    Index offset of each joint's dynamic constraints w.r.t the start
    index of dynamic joint constraints of the corresponding world.\n

        Used together with :attr:`ModelInfo.joint_dynamic_cts_offset`
        to index into the joint-specific blocks of:
    - array of effective joint-space inertia :attr:`JointsData.m_j`
    - array of joint-space damping :attr:`JointsData.b_j`
    - array of joint-space P gains :attr:`JointsData.k_p_j`
    - array of joint-space D gains :attr:`JointsData.k_d_j`

    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    kinematic_cts_offset: wp.array | None = None
    """
    Index offset of each joint's kinematic constraints w.r.t the start
    index of kinematic joint constraints of the corresponding world.\n

    Used together with :attr:`ModelInfo.joint_kinematic_cts_offset`
    to index into the joint-specific blocks of:
    - array of joint constraint residuals :attr:`JointsData.r_j`
    - array of joint constraint residual time-derivatives :attr:`JointsData.dr_j`

    Shape of ``(num_joints,)`` and type :class:`int`.
    """


@dataclass
class JointsData:
    """
    An SoA-based container to hold time-varying data of a joint system.
    """

    num_joints: int = 0
    """Total number of joints in the model (host-side)."""

    ###
    # State
    ###

    p_j: wp.array | None = None
    """
    Array of joint frame pose transforms in world coordinates.\n
    Shape of ``(num_joints,)`` and type :class:`transform`.
    """

    q_j: wp.array | None = None
    """
    Flat array of generalized coordinates of the joints.\n
    Shape of ``(sum_of_num_joint_coords,)`` and type :class:`float`,\n
    where `sum_of_num_joint_coords := sum(c_j)`, and ``c_j``
    is the number of coordinates of joint ``j``.
    """

    q_j_p: wp.array | None = None
    """
    Flat array of previous generalized coordinates of the joints.\n
    Shape of ``(sum_of_num_joint_coords,)`` and type :class:`float`,\n
    where `sum_of_num_joint_coords := sum(c_j)`, and ``c_j``
    is the number of coordinates of joint ``j``.
    """

    dq_j: wp.array | None = None
    """
    Flat array of generalized velocities of the joints.\n
    Shape of ``(sum_of_num_joint_dofs,)`` and type :class:`float`,\n
    where `sum_of_num_joint_dofs := sum(d_j)`, and ``d_j``
    is the number of DoFs of joint ``j``.
    """

    # TODO (@ruben): I think we still need this to be in full DoF space to inject
    # generalized forces in full DoF-space, when no implicit PD control is preset,
    # plus we could use this as a feed-forward term when implicit PD control is present.
    tau_j: wp.array | None = None
    """
    Flat array of generalized forces of the joints.\n
    Shape of ``(sum_of_num_joint_dofs,)`` and type :class:`float`,
    where `sum_of_num_joint_dofs := sum(d_j)`, and ``d_j``
    is the number of DoFs of joint ``j``.
    """

    ###
    # Constraints
    ###

    r_j: wp.array | None = None
    """
    Flat array of joint kinematic constraint residuals.

    To access the constraint residuals of a specific world `w` use:
    - to get the start index: ``model.info.joint_kinematic_cts_offset[w]``
    - to get the size: ``model.info.num_joint_kinematic_cts[w]``

    Shape of ``(sum_of_num_kinematic_joint_cts,)`` and type :class:`float`,\n
    where `sum_of_num_kinematic_joint_cts := sum(f_j)`, and ``f_j``
    is the number of kinematic constraints of joint ``j``.
    """

    dr_j: wp.array | None = None
    """
    Flat array of joint kinematic constraint residual time-derivatives.

    To access the constraint residuals of a specific world `w` use:
    - to get the start index: ``model.info.joint_kinematic_cts_offset[w]``
    - to get the size: ``model.info.num_joint_kinematic_cts[w]``

    Shape of ``(sum_of_num_kinematic_joint_cts,)`` and type :class:`float`,\n
    where `sum_of_num_kinematic_joint_cts := sum(f_j)`, and ``f_j``
    is the number of kinematic constraints of joint ``j``.
    """

    lambda_j: wp.array | None = None
    """
    Flat array of joint constraint Lagrange multipliers.

    To access the constraint multipliers of a specific world `w` use:
    - to get the start index: ``model.info.joint_cts_offset[w]``
    - to get the size: ``model.info.num_joint_cts[w]``

    Then to access the individual dynamic or kinematic constraint blocks, use:
    - dynamic constraints:
        ``model.info.joint_dynamic_cts_group_offset[w]`` and ``model.info.num_joint_dynamic_cts[w]``
    - kinematic constraints:
        ``model.info.joint_kinematic_cts_group_offset[w]`` and ``model.info.num_joint_kinematic_cts[w]``

    Shape of ``(sum_of_num_joint_cts,)`` and type :class:`float`,\n
    where `sum_of_num_joint_cts := sum(e_j) + sum(f_j)`, and ``e_j`` and ``f_j``
    are the number of dynamic and kinematic constraints of joint ``j``, respectively.
    """

    ###
    # Dynamics
    ###

    m_j: wp.array | None = None
    """
    Internal effective inertia of each joint (as flat array),
    used for implicit integration of joint dynamics.

    ``m_j := a_j + dt * (b_j + k_d_j) + dt^2 * k_p_j``,\n
    where dt is the simulation time step.

    Shape of ``(sum(e_j),)`` and type :class:`float`,\n
    where ``e_j`` is the number of dynamic constraints of joint ``j``.
    """

    inv_m_j: wp.array | None = None
    """
    Internal effective inverse inertia of each joint (as flat
    array), used for implicit integration of joint dynamics.

    ``inv_m_j := 1 / m_j``, computed element-wise,\n
    where ``m_j := a_j + dt * (b_j + k_d_j) + dt^2 * k_p_j``,
    and dt is the simulation time step.

    Note that all ``inv_m_j>0``, otherwise the DoF would not be
    part of the dynamic constraints.

    Shape of ``(sum(e_j),)`` and type :class:`float`,
    where ``e_j`` is the number of dynamic constraints of joint ``j``.
    """

    qd_b_j: wp.array | None = None
    """
    The velocity bias of the joint dynamic constraints (as flat array).

    Each joint has local actuation and PD control dynamics:\n
    ``m_j * dq_j^{+} = a_j * dq_j^{-} + dt * h_j``

    and is contributes to the dynamice of the system through the constraint equation:\n
    ``dq_j^{+} = J_a_j * u^{+}``

    where ``dq_j^{-}`` and ``dq_j^{+}`` are the pre- and post-event joint-space
    velocities, and ``u^{+}`` are the post-event generalized velocities of the
    system computed implicitly as a result of solving the forward dynamics problem
    with the joint dynamic constraints. `J_a_j` is the block of the actuation Jacobian
    matrix corresponding to the rows of DoFs of joint `j`.

    This results in the following dynamic constraint equation for each joint `j`:\n
    ``dq_j^{+} + m_j^{-1} * lambda_q_j = m_j^{-1} * (a_j * dq_j^{-} + dt * h_j)``,\n
    ``dq_j^{+} + m_j^{-1} * lambda_q_j = qd_b_j``,\n
    ``J_a_j * u^{+} + m_j^{-1} * lambda_q_j = qd_b_j``

    and thus the velocity bias term of the joint-space dynamics of each joint `j` is computed as:\n
    ``h_j := dt * ( k_p_j * ( q_j_ref - q_j^{-} ) + k_d_j * dq_j_ref ) ``,\n
    ``qd_b_j := inv_m_j * ( a_j * dq_j^{-} + dt * h_j ) ``,\n
    where dt is the simulation time step.

    Shape of ``(sum(e_j),)`` and type :class:`float`,
    where ``e_j`` is the number of dynamic constraints of joint ``j``.
    """

    ###
    # Reference State
    ###

    q_j_ref: wp.array | None = None
    """
    The reference coordinates of each joint (as flat array), indicating
    the target position of each joint for implicit PD control.\n
    Shape of ``(sum(c_j),)`` and type :class:`float`,
    where ``c_j`` is the number of coordinates of joint ``j``.
    """

    dq_j_ref: wp.array | None = None
    """
    The reference velocities of each joint (as flat array), indicating
    the target velocity of each joint for implicit PD control.\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    ###
    # Per-Body Wrenches
    #
    # TODO: Remove these (probably redundant) or make them optional via flag
    # since they are mainly useful for visualization and simulation debugging
    ###

    j_w_j: wp.array | None = None
    """
    Total wrench applied by each joint, expressed
    in and about the corresponding joint frame.\n
    Its direction follows the convention that
    joints act on the follower by the base body.\n
    Shape of ``(num_joints,)`` and type :class:`vec6`.
    """

    j_w_a_j: wp.array | None = None
    """
    Actuation wrench applied by each joint, expressed
    in and about the corresponding joint frame.\n
    Its direction is defined by the convention that positive wrenches
    in the joint frame are those inducing a positive change in the
    twist of the follower body relative to the base body.\n
    Shape of ``(num_joints,)`` and type :class:`vec6`.
    """

    j_w_c_j: wp.array | None = None
    """
    Constraint wrench applied by each joint, expressed
    in and about the corresponding joint frame.\n
    Its direction is defined by the convention that positive wrenches
    in the joint frame are those inducing a positive change in the
    twist of the follower body relative to the base body.\n
    Shape of ``(num_joints,)`` and type :class:`vec6`.
    """

    j_w_l_j: wp.array | None = None
    """
    Joint-limit wrench applied by each joint, expressed
    in and about the corresponding joint frame.\n
    Its direction is defined by the convention that positive wrenches
    in the joint frame are those inducing a positive change in the
    twist of the follower body relative to the base body.\n
    Shape of ``(num_joints,)`` and type :class:`vec6`.
    """

    ###
    # Operations
    ###

    def reset_state(self, q_j_0: wp.array | None = None):
        """
        Resets all generalized joint coordinates to either zero or the provided
        reference coordinates and all generalized joint velocities to zero.
        """
        if q_j_0 is not None:
            if q_j_0.size != self.q_j.size:
                raise ValueError(f"Invalid size of q_j_0: {q_j_0.size}. Expected: {self.q_j.size}.")
            wp.copy(self.q_j, q_j_0)
            wp.copy(self.q_j_p, q_j_0)
        else:
            self.q_j.zero_()
            self.q_j_p.zero_()
        self.dq_j.zero_()

    def reset_references(self, q_j_ref: wp.array | None = None, dq_j_ref: wp.array | None = None):
        """
        Resets all reference coordinates and velocities to either zero or the provided
        reference values.
        """
        if q_j_ref is not None:
            if q_j_ref.size != self.q_j_ref.size:
                raise ValueError(f"Invalid size of q_j_ref: {q_j_ref.size}. Expected: {self.q_j_ref.size}.")
            wp.copy(self.q_j_ref, q_j_ref)
        else:
            self.q_j_ref.zero_()

        if dq_j_ref is not None:
            if dq_j_ref.size != self.dq_j_ref.size:
                raise ValueError(f"Invalid size of dq_j_ref: {dq_j_ref.size}. Expected: {self.dq_j_ref.size}.")
            wp.copy(self.dq_j_ref, dq_j_ref)
        else:
            self.dq_j_ref.zero_()

    def clear_residuals(self):
        """
        Resets all joint state variables to zero.
        """
        self.r_j.zero_()
        self.dr_j.zero_()

    def clear_constraint_reactions(self):
        """
        Resets all joint constraint reactions to zero.
        """
        self.lambda_j.zero_()
        # TODO: self.lambda_j_q.zero_()
        # TODO: self.lambda_j_c.zero_()

    def clear_actuation_forces(self):
        """
        Resets all joint actuation forces to zero.
        """
        self.tau_j.zero_()

    def clear_wrenches(self):
        """
        Resets all joint wrenches to zero.
        """
        self.j_w_j.zero_()
        self.j_w_c_j.zero_()
        self.j_w_a_j.zero_()
        self.j_w_l_j.zero_()

    def clear_all(self):
        """
        Resets all joint state variables, constraint reactions,
        actuation forces, and wrenches to zero.
        """
        self.clear_residuals()
        self.clear_constraint_reactions()
        self.clear_actuation_forces()
        self.clear_wrenches()
