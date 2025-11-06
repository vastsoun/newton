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

import warp as wp
from warp._src.types import Any, Int, Vector

from .math import FLOAT32_MAX, FLOAT32_MIN
from .types import (
    Descriptor,
    mat33f,
    override,
    vec1i,
    vec2i,
    vec3f,
    vec3i,
    vec4i,
    vec5i,
    vec6i,
)

###
# Module interface
###

__all__ = ["JointActuationType", "JointConnectionType", "JointDescriptor", "JointDoFType", "JointsData", "JointsModel"]


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

    @override
    def __str__(self):
        """Returns a string representation of the joint actuation type."""
        return f"JointDoFType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint actuation type."""
        return self.__str__()


class JointConnectionType(IntEnum):
    """
    An enumeration of the joint connection types.
    """

    UNARY = 0
    """Unary joint connection type, connecting a body to the world."""

    BINARY = 1
    """Binary joint connection type, connecting two bodies."""

    @override
    def __str__(self):
        """Returns a string representation of the joint connection type."""
        return f"JointDoFType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint connection type."""
        return self.__str__()


class JointDoFType(IntEnum):
    """
    An enumeration of the joint degrees of freedom (DoF) types supported.

    This class also provides property methods to query the number of:
    - Generalized coordinates
    - Degrees of Freedom (DoFs)
    - Equality constraints

    Conventions:
    - Joints are indexed by `j`, and we often employ the subscript notation `*_j`.
    - `R_x`, `R_y`, `R_z`: rotational DoFs about the local x, y, z axes respectively.
    - `T_x`, `T_y`, `T_z`: translational DoFs along the local x, y, z axes respectively.
    - `c_j` | `num_coords`: the number of generalized coordinates defined by joint `j`.
    - `d_j` | `num_dofs`: the number of DoFs defined by joint `j`.
    - `e_j` | `num_cts`: the number of equality constraints imposed by joint `j`.
    """

    FREE = 0
    """
    A 6-DoF free-floating joint, with 3 rotational + 3 translational DoFs,
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
    A 1-DoF revolute joint, with 1 rotational DoF along {`R_x`}.

    Coordinates:
        1D angle: {`R_x`}
    DoFs:
        1D angular velocity: {`R_x`}
    Constraints:
        5D vector: {`T_x`, `T_y`, `T_z`, `R_y`, `R_z`}
    """

    PRISMATIC = 2
    """
    A 1-DoF prismatic joint, with 1 translational DoF along {`T_x`}.

    Coordinates:
        1D distance: {`T_x`}
    DoFs:
        1D linear velocity: {`T_x`}
    Constraints:
        5D vector: {`T_y`, `T_z`, `R_x`, `R_y`, `R_z`}
    """

    CYLINDRICAL = 3
    """
    A 2-DoF cylindrical joint, with 1 rotational and 1 translational DoF along {`R_x`, `T_x`}.

    Coordinates:
        2D vector of angle {`R_x`} + 1D distance {`T_x`}
    DoFs:
        2D vector of angular velocity {`R_x`} + linear velocity {`T_x`}
    """

    UNIVERSAL = 4
    """
    A 2-DoF universal joint, with 2 rotational DoFs, {`R_x`, `R_y`}.

    Coordinates:
        2D angles: {`R_x`, `R_y`}
    DoFs:
        2D angular velocities: {`R_x`, `R_y`}
    Constraints:
        4D vector: {`T_x`, `T_y`, `T_z`, `R_z`}
    """

    SPHERICAL = 5
    """
    A 3-DoF spherical joint, with 3 rotational DoFs along {`R_x`, `R_y`, `R_z`}.

    Coordinates:
        4D unit-quaternion to parameterize {`R_x`, `R_y`, `R_z`}
    DoFs:
        3D angular velocities: {`R_x`, `R_y`, `R_z`}
    Constraints:
        3D vector: {`T_x`, `T_y`, `T_z`}
    """

    GIMBAL = 6
    """
    A 3-DoF gimbal joint, with 3 rotational DoFs, {`R_x`, `R_y`, `R_z`}.

    Coordinates:
        3D euler angles: {`R_x`, `R_y`, `R_z`}
    DoFs:
        3D angular velocities: {`R_x`, `R_y`, `R_z`}
    Constraints:
        3D vector: {`T_x`, `T_y`, `T_z`}
    """

    CARTESIAN = 7
    """
    A 3-DoF Cartesian joint, with 3 translational DoFs, {`T_x`, `T_y`, `T_z`}.

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
        The number of generalized coordinates defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 7
        elif self.value == self.REVOLUTE:
            return 1
        elif self.value == self.PRISMATIC:
            return 1
        elif self.value == self.CYLINDRICAL:
            return 2
        elif self.value == self.UNIVERSAL:
            return 2
        elif self.value == self.SPHERICAL:
            return 3  # TODO: 4
        elif self.value == self.GIMBAL:
            return 3
        elif self.value == self.CARTESIAN:
            return 3
        elif self.value == self.FIXED:
            return 0
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def num_dofs(self) -> int:
        """
        The number of DoFs defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 6
        elif self.value == self.REVOLUTE:
            return 1
        elif self.value == self.PRISMATIC:
            return 1
        elif self.value == self.CYLINDRICAL:
            return 2
        elif self.value == self.UNIVERSAL:
            return 2
        elif self.value == self.SPHERICAL:
            return 3
        elif self.value == self.GIMBAL:
            return 3
        elif self.value == self.CARTESIAN:
            return 3
        elif self.value == self.FIXED:
            return 0
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def num_cts(self) -> int:
        """
        The number of constraints defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 0
        elif self.value == self.REVOLUTE:
            return 5
        elif self.value == self.PRISMATIC:
            return 5
        elif self.value == self.CYLINDRICAL:
            return 4
        elif self.value == self.UNIVERSAL:
            return 4
        elif self.value == self.SPHERICAL:
            return 3
        elif self.value == self.GIMBAL:
            return 3
        elif self.value == self.CARTESIAN:
            return 3
        elif self.value == self.FIXED:
            return 6
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def cts_axes(self) -> Vector[Any, Int]:
        """
        Returns the constraint axes indices of the joint.
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
        Returns the DoF axes indices of the joint.
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


###
# Containers
###


@dataclass
class JointDescriptor(Descriptor):
    """
    A container to describe a single joint in the model builder.

    Attributes:
        name (str): Name of the joint.
        uid (int): Unique identifier of the joint.
        act_type (JointActuationType): Actuation type of the joint.
        dof_type (JointDoFType): DoF type of the joint.
        bid_B (int): The Base body index of the joint (-1 for world, >=0 for bodies).
        bid_F (int): The Follower body index of the joint (-1 for world, >=0 for bodies).
        B_r_Bj (vec3f): The relative position of the joint in the base body coordinates.
        F_r_Fj (vec3f): The relative position of the joint in the follower body coordinates.
        X_j (mat33f): The constant axes matrix of the joint.
        q_j_min (list[float] | float | None): Minimum configuration limits of the joint.
        q_j_max (list[float] | float | None): Maximum configuration limits of the joint.
        dq_j_max (list[float] | float | None): Maximum velocity limits of the joint.
        tau_j_max (list[float] | float | None): Maximum effort limits of the joint.
    """

    ###
    # Attributes
    ###

    act_type: JointActuationType = JointActuationType.PASSIVE
    """Actuation type of the joint."""

    dof_type: JointDoFType = JointDoFType.FREE
    """DoF type of the joint."""

    bid_B: int = -1
    """The Base body index of the joint (-1 for world, >=0 for bodies)."""

    bid_F: int = -1
    """The Follower body index of the joint (-1 for world, >=0 for bodies)."""

    # TODO: Change this to a transformf
    B_r_Bj: vec3f = field(default_factory=vec3f)
    """The relative position of the joint in the base body coordinates."""

    # TODO: Change this to a transformf
    F_r_Fj: vec3f = field(default_factory=vec3f)
    """The relative position of the joint in the follower body coordinates."""

    # TODO: Remove this when body offsets become transforms
    X_j: mat33f = field(default_factory=mat33f)
    """The constant axes matrix of the joint."""

    q_j_min: list[float] | float | None = None
    """Minimum configuration limits of the joint."""

    q_j_max: list[float] | float | None = None
    """Maximum configuration limits of the joint."""

    dq_j_max: list[float] | float | None = None
    """Maximum velocity limits of the joint."""

    tau_j_max: list[float] | float | None = None
    """Maximum effort limits of the joint."""

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
    """Index offset of this joint's DoFs among all joint DoFs in the world it belongs to."""

    passive_coords_offset: int = -1
    """Index offset of this joint's passive coordinates among all joint coordinates in the world it belongs to."""

    passive_dofs_offset: int = -1
    """Index offset of this joint's passive DoFs among all joint DoFs in the world it belongs to."""

    actuated_coords_offset: int = -1
    """Index offset of this joint's actuated coordinates among all joint coordinates in the world it belongs to."""

    actuated_dofs_offset: int = -1
    """Index offset of this joint's actuated DoFs among all joint DoFs in the world it belongs to."""

    cts_offset: int = -1
    """Index offset of this joint's constraints among all joint constraints in the world it belongs to."""

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
    def num_cts(self) -> int:
        """
        Returns the number of constraints for this joint.
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

        # Set default values for joint limits if not provided
        self.q_j_min = self._check_limits(self.q_j_min, self.num_coords, float(FLOAT32_MIN))
        self.q_j_max = self._check_limits(self.q_j_max, self.num_coords, float(FLOAT32_MAX))
        self.dq_j_max = self._check_limits(self.dq_j_max, self.num_dofs, float(FLOAT32_MAX))
        self.tau_j_max = self._check_limits(self.tau_j_max, self.num_dofs, float(FLOAT32_MAX))

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the JointDescriptor."""
        return (
            f"JointDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"act_type: {self.act_type},\n"
            f"dof_type: {self.dof_type},\n"
            f"bid_B: {self.bid_B},\n"
            f"bid_F: {self.bid_F},\n"
            f"B_r_Bj: {self.B_r_Bj},\n"
            f"F_r_Fj: {self.F_r_Fj},\n"
            f"X_j:\n{self.X_j},\n"
            f"q_j_min: {self.q_j_min},\n"
            f"q_j_max: {self.q_j_max},\n"
            f"dq_j_max: {self.dq_j_max},\n"
            f"tau_j_max: {self.tau_j_max}\n"
            f"wid: {self.wid},\n"
            f"jid: {self.jid},\n"
            f"num_coords: {self.num_coords},\n"
            f"num_dofs: {self.num_dofs},\n"
            f"num_cts: {self.num_cts},\n"
            f"coords_offset: {self.coords_offset},\n"
            f"dofs_offset: {self.dofs_offset},\n"
            f"cts_offset: {self.cts_offset},\n"
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
    def _check_limits(
        limits: list[float] | float | None, size: int, default: float = float(FLOAT32_MAX)
    ) -> list[float]:
        """
        Processes a specified limit value to ensure it is a list of floats.

        Notes:
        - If the input is None, a list of default values is returned.
        - If the input is a single float, it is converted to a list of the specified length.
        - If the input is an empty list, a list of default values is returned.
        - If the input is a non-empty list, it is validated to ensure it
            contains only floats and matches the specified length.

        Args:
            limits (List[float] | float | None): The limits to be processed.
            num_dofs (int): The number of degrees of freedom to determine the length of the output list.
            default (float): The default value to use if limits is None or an empty list.

        Returns:
            List[float]: The processed list of limits.

        Raises:
            ValueError: If the length of the limits list does not match num_dofs.
            TypeError: If the limits list contains non-float types.
        """
        if limits is None:
            return [float(default) for _ in range(size)]

        if isinstance(limits, float):
            if limits == math.inf:
                return [float(FLOAT32_MAX) for _ in range(size)]
            elif limits == -math.inf:
                return [float(FLOAT32_MIN) for _ in range(size)]
            else:
                return [limits] * size

        if isinstance(limits, list):
            if len(limits) == 0:
                return [float(default) for _ in range(size)]

            if len(limits) != size:
                raise ValueError(f"Invalid limits length: {len(limits)} != {size}")

            if all(isinstance(x, float) for x in limits):
                return limits
            else:
                raise TypeError(
                    f"Invalid limits type: All entries must be `float`, but got:\n{[type(x) for x in limits]}"
                )


@dataclass
class JointsModel:
    """
    An SoA-based container to hold time-invariant model data of joints.

    Attributes:
        num_joints (int): Total number of joints in the model.
        wid (wp.array | None): World index of each joint.
        jid (wp.array | None): Joint index w.r.t the world of each joint.
        dof_type (wp.array | None): Joint DoF type ID of each joint.
        act_type (wp.array | None): Joint actuation type ID of each joint.
        bid_B (wp.array | None): Base body index of each joint w.r.t the model.\n
            Equals `-1` for world, `>=0` for bodies.
        bid_F (wp.array | None): Follower body index of each joint w.r.t the model.\n
            Equals `-1` for world, `>=0` for bodies.
        B_r_Bj (wp.array | None): Relative position of the joint,
            expressed in and w.r.t the base body coordinate frame.
        F_r_Fj (wp.array | None): Relative position of the joint,
            expressed in and w.r.t the follower body coordinate frame.
        X_j (wp.array | None): Joint axes matrix (local coordinates) of each joint.\n
            Indicates the relative orientation of the the joint frame w.r.t the base body coordinate frame.
        q_j_min (wp.array | None): Minimum joint position limits of each joint (as flat array).
        q_j_max (wp.array | None): Maximum joint position limits of each joint (as flat array).
        num_coords (wp.array | None): Number of configuration coordinates of each joint.
        num_dofs (wp.array | None): Number of DoFs of each joint.
        num_cts (wp.array | None): Number of constraints of each joint.
        coords_offset (wp.array | None): Index offset of each joint's coordinates
            w.r.t the start index of joint coordinates of the corresponding world.
        dofs_offset (wp.array | None): Index offset of each joint's DoFs w.r.t
            the start index of joint DoFs of the corresponding world.
        passive_coords_offset (wp.array | None): Index offset of each joint's passive coordinates
            w.r.t the start index of passive joint coordinates of the corresponding world.
        passive_dofs_offset (wp.array | None): Index offset of each joint's passive DoFs
            w.r.t the start index of passive joint DoFs of the corresponding world.
        actuated_coords_offset (wp.array | None): Index offset of each joint's actuated coordinates
            w.r.t the start index of actuated joint coordinates of the corresponding world.
        actuated_dofs_offset (wp.array | None): Index offset of each joint's actuated DoFs w.r.t
            the start index of actuated joint DoFs of the corresponding world.
        cts_offset (wp.array | None): Index offset of each joint's constraints w.r.t
            the start index of joint constraints of the corresponding world.
    """

    num_joints: int = 0
    """Total number of joints in the model (host-side)."""

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

    q_j_min: wp.array | None = None
    """
    Minimum joint position limits of each joint (as flat array).\n
    Shape of ``(sum(c_j),)`` and type :class:`float`,\n
    where ``c_j`` is the number of coordinates of joint ``j``.
    """

    q_j_max: wp.array | None = None
    """
    Maximum joint position limits of each joint (as flat array).\n
    Shape of ``(sum(c_j),)`` and type :class:`float`,\n
    where ``c_j`` is the number of coordinates of joint ``j``.
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

    num_coords: wp.array | None = None
    """
    Number of configuration coordinates of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    num_dofs: wp.array | None = None
    """
    Number of DoFs of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    num_cts: wp.array | None = None
    """
    Number of constraints of each joint.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    coords_offset: wp.array | None = None
    """
    Index offset of each joint's coordinates w.r.t the start
    index of joint coordinates of the corresponding world.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    dofs_offset: wp.array | None = None
    """
    Index offset of each joint's DoFs w.r.t the start
    index of joint DoFs of the corresponding world.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    passive_coords_offset: wp.array | None = None
    """
    Index offset of each joint's passive coordinates w.r.t the start
    index of passive joint coordinates of the corresponding world.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    passive_dofs_offset: wp.array | None = None
    """
    Index offset of each joint's passive DoFs w.r.t the start
    index of passive joint DoFs of the corresponding world.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    actuated_coords_offset: wp.array | None = None
    """
    Index offset of each joint's actuated coordinates w.r.t the start
    index of actuated joint coordinates of the corresponding world.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    actuated_dofs_offset: wp.array | None = None
    """
    Index offset of each joint's actuated DoFs w.r.t the start
    index of actuated joint DoFs of the corresponding world.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """

    cts_offset: wp.array | None = None
    """
    Index offset of each joint's constraints w.r.t the start
    index of joint constraints of the corresponding world.\n
    Shape of ``(num_joints,)`` and type :class:`int`.
    """


@dataclass
class JointsData:
    """
    An SoA-based container to hold time-varying data of a joint system.
    """

    num_joints: int = 0
    """Total number of joints in the model (host-side)."""

    p_j: wp.array | None = None
    """
    Array of joint frame pose transforms in world coordinates.\n
    Shape of ``(num_joints,)`` and type :class:`transform`.
    """

    r_j: wp.array | None = None
    """
    Flat array of joint constraint residuals.\n
    Shape of ``(sum_of_num_joint_cts,)`` and type :class:`float`,\n
    where `sum_of_num_joint_cts := sum(e_j)`, and ``e_j``
    is the number of constraints of joint ``j``.
    """

    dr_j: wp.array | None = None
    """
    Flat array of joint constraint residual time-derivatives.\n
    Shape of ``(sum_of_num_joint_cts,)`` and type :class:`float`,\n
    where `sum_of_num_joint_cts := sum(e_j)`, and ``e_j``
    is the number of constraints of joint ``j``.
    """

    lambda_j: wp.array | None = None
    """
    Flat array of joint constraint Lagrange multipliers.\n
    Shape of ``(sum_of_num_joint_cts,)`` and type :class:`float`,\n
    where `sum_of_num_joint_cts := sum(e_j)`, and ``e_j``
    is the number of constraints of joint ``j``.
    """

    q_j: wp.array | None = None
    """
    Flat array of generalized coordinates of the joints.\n
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

    tau_j: wp.array | None = None
    """
    Flat array of generalized forces of the joints.\n
    Shape of ``(sum_of_num_joint_dofs,)`` and type :class:`float`,\n
    where `sum_of_num_joint_dofs := sum(d_j)`, and ``d_j``
    is the number of DoFs of joint ``j``.
    """

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

    def clear_residuals(self):
        """
        Reset all joint state variables to zero.
        """
        self.r_j.zero_()
        self.dr_j.zero_()

    def clear_state(self):
        """
        Reset all joint state variables to zero.
        """
        self.q_j.zero_()
        self.dq_j.zero_()

    def clear_constraint_reactions(self):
        """
        Reset all joint constraint reactions to zero.
        """
        self.lambda_j.zero_()

    def clear_actuation_forces(self):
        """
        Reset all joint actuation forces to zero.
        """
        self.tau_j.zero_()

    def clear_wrenches(self):
        """
        Reset all joint wrenches to zero.
        """
        self.j_w_j.zero_()
        self.j_w_c_j.zero_()
        self.j_w_a_j.zero_()
        self.j_w_l_j.zero_()
