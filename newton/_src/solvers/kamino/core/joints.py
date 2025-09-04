###########################################################################
# KAMINO: Joint Model Types & Containers
###########################################################################

from __future__ import annotations

import sys
import warp as wp

from enum import IntEnum
from typing import List

if sys.version_info >= (3, 12):
    from typing import override
else:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback no-op decorator if typing_extensions is not available
        def override(func):
            return func

from .types import (
    int32, float32,
    vec3f, vec6f,
    mat33f,
    transformf,
)


###
# Module interface
###

__all__ = [
    "JointConnectionType",
    "JointDoFType",
    "JointActuationType",
    "JointDescriptor",
    "JointsModel",
    "JointsData"
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

JOINT_UNARY = wp.constant(0)
"""Unary joint connection type, connecting a body to the world."""

JOINT_BINARY = wp.constant(1)
"""Binary joint connection type, connecting two bodies."""

JOINT_FREE = wp.constant(0)
"""6-DoF free-floating joint, with 6 rotational and translational DoFs, {R_x, R_y, R_z, T_x, T_y, T_z}."""

JOINT_REVOLUTE = wp.constant(1)
"""1-DoF revolute joint, with 1 rotational DoF, {R_x}."""

JOINT_PRISMATIC = wp.constant(2)
"""1-DoF prismatic joint, with 1 translational DoF, {T_x}."""

JOINT_CYLINDRICAL = wp.constant(3)
"""2-DoF cylindrical joint, with 1 rotational and 1 translational DoF, {R_x, T_x}."""

JOINT_UNIVERSAL = wp.constant(4)
"""2-DoF universal joint, with 2 rotational DoFs, {TODO}."""

JOINT_SPHERICAL = wp.constant(5)
"""3-DoF spherical joint, with 3 rotational DoFs, {R_x, R_y, R_z}."""

JOINT_CARTESIAN = wp.constant(6)
"""3-DoF Cartesian joint, with 3 translational DoFs, {T_x, T_y, T_z}."""

JOINT_FIXED = wp.constant(7)
"""0-DoF fixed joint, with no relative motion between the bodies."""

JOINT_PASSIVE = wp.constant(0)
"""Passive joint type."""

JOINT_FORCE_CONTROLLED = wp.constant(1)
"""Force-controlled joint type, actuated by set of forces and/or torques."""


###
# Enumerations
###

class JointActuationType(IntEnum):
    """
    An enumeration of the joint actuation types.
    """
    PASSIVE = JOINT_PASSIVE
    """Passive joint type, i.e. not actuated."""
    FORCE = JOINT_FORCE_CONTROLLED
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
    UNARY = JOINT_UNARY
    """Unary joint connection type, connecting a body to the world."""
    BINARY = JOINT_BINARY
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
    An enumeration of the joint degrees of freedom (DoF) types.
    """
    FREE = JOINT_FREE
    """6-DoF free-floating joint, with 6 rotational and translational DoFs, {R_x, R_y, R_z, T_x, T_y, T_z}."""
    REVOLUTE = JOINT_REVOLUTE
    """1-DoF revolute joint, with 1 rotational DoF, {R_x}."""
    PRISMATIC = JOINT_PRISMATIC
    """1-DoF prismatic joint, with 1 translational DoF, {T_x}."""
    CYLINDRICAL = JOINT_CYLINDRICAL
    """2-DoF cylindrical joint, with 1 rotational and 1 translational DoF, {R_x, T_x}."""
    UNIVERSAL = JOINT_UNIVERSAL
    """2-DoF universal joint, with 2 rotational DoFs, {TODO}."""
    SPHERICAL = JOINT_SPHERICAL
    """3-DoF spherical joint, with 3 rotational DoFs, {R_x, R_y, R_z}."""
    CARTESIAN = JOINT_CARTESIAN
    """3-DoF Cartesian joint, with 3 translational DoFs, {T_x, T_y, T_z}."""
    FIXED = JOINT_FIXED
    """0-DoF fixed joint, with no relative motion between the bodies."""

    @override
    def __str__(self):
        """Returns a string representation of the joint DoF type."""
        return f"JointDoFType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint DoF type."""
        return self.__str__()

    @property
    def num_cts(self) -> int:
        """
        The number of constraints defined by the joint DoF type.
        """
        if self.value == JOINT_FREE:
            return 0
        elif self.value == JOINT_REVOLUTE:
            return 5
        elif self.value == JOINT_PRISMATIC:
            return 5
        elif self.value == JOINT_CYLINDRICAL:
            return 4
        elif self.value == JOINT_UNIVERSAL:
            return 4
        elif self.value == JOINT_SPHERICAL:
            return 3
        elif self.value == JOINT_CARTESIAN:
            return 3
        elif self.value == JOINT_FIXED:
            return 6
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def num_dofs(self) -> int:
        """
        The number of DoFs defined by the joint DoF type.
        """
        if self.value == JOINT_FREE:
            return 6
        elif self.value == JOINT_REVOLUTE:
            return 1
        elif self.value == JOINT_PRISMATIC:
            return 1
        elif self.value == JOINT_CYLINDRICAL:
            return 2
        elif self.value == JOINT_UNIVERSAL:
            return 2
        elif self.value == JOINT_SPHERICAL:
            return 3
        elif self.value == JOINT_CARTESIAN:
            return 3
        elif self.value == JOINT_FIXED:
            return 0
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def num_coord(self) -> int:
        """
        The number of generalized coordinates defined by the joint DoF type.
        """
        if self.value == JOINT_FREE:
            return 7
        elif self.value == JOINT_REVOLUTE:
            return 1
        elif self.value == JOINT_PRISMATIC:
            return 1
        elif self.value == JOINT_CYLINDRICAL:
            return 2
        elif self.value == JOINT_UNIVERSAL:
            return 2
        elif self.value == JOINT_SPHERICAL:
            return 4
        elif self.value == JOINT_CARTESIAN:
            return 7
        elif self.value == JOINT_FIXED:
            return 0
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")


###
# Containers
###

class JointDescriptor:
    """
    A container to describe a single joint in the model builder.
    """
    def __init__(self):
        self.name: str | None = None
        """Name of the joint."""

        self.uid: str | None = None
        """Unique identifier code (UID) of the joint."""

        self.wid: int = 0
        """Index of the world in which the joint is defined."""

        self.jid: int = -1
        """Index of the joint w.r.t. the world."""

        self.act_type: JointActuationType = JointActuationType.PASSIVE
        """Actuation type of the joint."""

        self.dof_type: JointDoFType = JointDoFType.FREE
        """DoF type of the joint."""

        self.num_cts: int = -1
        """Number of constraints of the joint."""

        self.num_dofs: int = -1
        """Number of DoFs of the joint."""

        self.cts_offset: int = -1
        """Index offset of the joint's constraints w.r.t the world."""

        self.dofs_offset: int = -1
        """Index offset of the joint's DoFs w.r.t the world."""

        self.passive_dofs_offset: int = -1
        """Index offset of the joint's passive DoFs w.r.t the world."""

        self.actuated_dofs_offset: int = -1
        """Index offset of the joint's actuated DoFs w.r.t the world."""

        self.bid_B: int = -1
        """The Base body index of the joint (-1 for world, >=0 for bodies)."""

        self.bid_F: int = -1
        """The Follower body index of the joint (-1 for world, >=0 for bodies)."""

        # TODO: Change this to a transformf
        self.B_r_Bj: vec3f = vec3f()
        """The relative position of the joint in the base body coordinates."""

        # TODO: Change this to a transformf
        self.F_r_Fj: vec3f = vec3f()
        """The relative position of the joint in the follower body coordinates."""

        # TODO: Remove this when body offsets become transforms
        self.X_j: mat33f = mat33f()
        """The constant axes matrix of the joint."""

        self.q_j_min: List[float] | float | None = None
        """Minimum configuration limits of the joint."""

        self.q_j_max: List[float] | float | None = None
        """Maximum configuration limits of the joint."""

        self.dq_j_max: List[float] | float | None = None
        """Maximum velocity limits of the joint."""

        self.tau_j_max: List[float] | float | None = None
        """Maximum effort limits of the joint."""

    def __repr__(self):
        return (
            f"JointDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"wid: {self.wid},\n"
            f"jid: {self.jid},\n"
            f"act_type: {self.act_type},\n"
            f"dof_type: {self.dof_type},\n"
            f"num_cts: {self.num_cts},\n"
            f"num_dofs: {self.num_dofs},\n"
            f"cts_offset: {self.cts_offset},\n"
            f"dofs_offset: {self.dofs_offset},\n"
            f"passive_dofs_offset: {self.passive_dofs_offset},\n"
            f"actuated_dofs_offset: {self.actuated_dofs_offset},\n"
            f"bid_B: {self.bid_B},\n"
            f"bid_F: {self.bid_F},\n"
            f"B_r_Bj: {self.B_r_Bj},\n"
            f"F_r_Fj: {self.F_r_Fj},\n"
            f"X_j:\n{self.X_j},\n"
            f"q_j_min: {self.q_j_min},\n"
            f"q_j_max: {self.q_j_max},\n"
            f"dq_j_max: {self.dq_j_max},\n"
            f"tau_j_max: {self.tau_j_max}\n"
            f")"
        )


class JointsModel:
    """
    An SoA-based container to hold time-invariant model data of a joint system.
    """
    def __init__(self):
        self.num_joints: int32 = 0
        """Total number of joints in the model (host-side)."""

        self.wid: wp.array(dtype=int32) | None = None
        """
        Index each the world in which each joint is defined.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.jid: wp.array(dtype=int32) | None = None
        """
        Index of each joint w.r.t the world.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.dof_type: wp.array(dtype=int32) | None = None
        """
        Joint DoF type ID of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.act_type: wp.array(dtype=int32) | None = None
        """
        Joint actuation type ID of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.num_cts: wp.array(dtype=int32) | None = None
        """
        Number of constraints of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.num_dofs: wp.array(dtype=int32) | None = None
        """
        Number of DoFs of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.cts_offset: wp.array(dtype=int32) | None = None
        """
        Index offset of the joint's constraint multipliers.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.dofs_offset: wp.array(dtype=int32) | None = None
        """
        Index offset of the joint's DoFs.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.passive_dofs_offset: wp.array(dtype=int32) | None = None
        """
        Index offset of the joint's passive DoFs.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.actuated_dofs_offset: wp.array(dtype=int32) | None = None
        """
        Index offset of the joint's actuated DoFs.\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.bid_B: wp.array(dtype=int32) | None = None
        """
        Base body index of each joint (-1 for world, >=0 for bodies).\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.bid_F: wp.array(dtype=int32) | None = None
        """
        Follower body index of each joint (-1 for world, >=0 for bodies).\n
        Shape of ``(num_joints,)`` and type :class:`int32`.
        """

        self.B_r_Bj: wp.array(dtype=vec3f) | None = None
        """
        Relative position of the joint in the base body coordinates.\n
        Shape of ``(num_joints, 3)`` and type :class:`vec3f`.
        """

        self.F_r_Fj: wp.array(dtype=vec3f) | None = None
        """
        Relative position of the joint in the follower body coordinates.\n
        Shape of ``(num_joints, 3)`` and type :class:`vec3f`.
        """

        self.X_j: wp.array(dtype=mat33f) | None = None
        """
        Joint axes matrix (local coordinates) of each joint (as flat array).\n
        Shape of ``(num_joints, 3, 3)`` and type :class:`mat33f`.
        """

        self.q_j_min: wp.array(dtype=float32)
        """
        Minimum joint position limits of each joint (as flat array).\n
        Shape of ``(sum(d_j),)`` and type :class:`float32`.
        """

        self.q_j_max: wp.array(dtype=float32)
        """
        Maximum joint position limits of each joint (as flat array).\n
        Shape of ``(sum(d_j),)`` and type :class:`float32`.
        """

        self.dq_j_max: wp.array(dtype=float32)
        """
        Maximum joint velocity limits of each joint (as flat array).\n
        Shape of ``(sum(d_j),)`` and type :class:`float32`.
        """

        self.tau_j_max: wp.array(dtype=float32)
        """
        Maximum joint torque limits of each joint (as flat array).\n
        Shape of ``(sum(d_j),)`` and type :class:`float32`.
        """


class JointsData:
    """
    An SoA-based container to hold time-varying data of a joint system.
    """
    def __init__(self):
        self.num_joints: int32 = 0
        """Total number of joints in the model (host-side)."""

        self.p_j: wp.array(dtype=transformf) | None = None
        """
        Array of joint frame pose transforms in world coordinates.\n
        Shape of ``(num_joints,)`` and type :class:`transformf`.
        """

        self.r_j: wp.array(dtype=float32) | None = None
        """
        Flat array of joint constraint residuals.\n
        Shape of ``(sum(m_j),)`` and type :class:`float32`.
        """

        self.dr_j: wp.array(dtype=float32) | None = None
        """
        Flat array of joint constraint residual time-derivatives.\n
        Shape of ``(sum(m_j),)`` and type :class:`float32`.
        """

        self.lambda_j: wp.array(dtype=float32) | None = None
        """
        Flat array of joint constraint Lagrange multipliers.\n
        Shape of ``(sum(m_j),)`` and type :class:`float32`.
        """

        self.q_j: wp.array(dtype=float32) | None = None
        """
        Flat array of generalized coordinates of the joints.\n
        Shape of ``(sum(d_j),)`` and type :class:`float32`.
        """

        self.dq_j: wp.array(dtype=float32) | None = None
        """
        Flat array of generalized velocities of the joints.\n
        Shape of ``(sum(d_j),)`` and type :class:`float32`.
        """

        self.tau_j: wp.array(dtype=float32) | None = None
        """
        Flat array of generalized forces of the joints.\n
        Shape of ``(sum(d_j),)`` and type :class:`float32`.
        """

        self.j_w_j: wp.array(dtype=vec6f) | None = None
        """
        Array of total wrenches applied by each joint.\n
        Shape of ``(num_joints,)`` and type :class:`vec6f`.
        """

        self.j_w_c_j: wp.array(dtype=vec6f) | None = None
        """
        Array of constraint wrenches applied by each joint.\n
        Shape of ``(num_joints,)`` and type :class:`vec6f`.
        """

        self.j_w_a_j: wp.array(dtype=vec6f) | None = None
        """
        Flat array of actuation wrenches applied by each joint.\n
        Shape of ``(num_joints,)`` and type :class:`vec6f`.
        """

        self.j_w_l_j: wp.array(dtype=vec6f) | None = None
        """
        Flat array of joint limit wrench applied by each joint.\n
        Shape of ``(num_joints,)`` and type :class:`vec6f`.
        """


###
# Kernels
###

@wp.kernel
def _reset_joints_state(
    # Inputs:
    model_joint_num_cts: wp.array(dtype=int32),
    model_joint_num_dofs: wp.array(dtype=int32),
    model_joint_cts_offset: wp.array(dtype=int32),
    model_joint_dofs_offset: wp.array(dtype=int32),
    model_joint_wid: wp.array(dtype=int32),
    model_joint_bid_B: wp.array(dtype=int32),
    model_joint_B_r_Bj: wp.array(dtype=vec3f),
    joint_cts_offsets: wp.array(dtype=int32),
    joint_dofs_offsets: wp.array(dtype=int32),
    state_body_q: wp.array(dtype=transformf),
    state_body_u: wp.array(dtype=vec6f),
    # Outputs:
    state_joint_p_j: wp.array(dtype=transformf),
    state_joint_r_j: wp.array(dtype=float32),
    state_joint_dr_j: wp.array(dtype=float32),
    state_joint_q_j: wp.array(dtype=float32),
    state_joint_dq_j: wp.array(dtype=float32),
    state_joint_tau_j: wp.array(dtype=float32),
):
    """
    Reset the current state to the initial state defined in the model.
    """
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the joint model data
    wid = model_joint_wid[tid]
    c_j = model_joint_num_cts[tid]
    d_j = model_joint_num_dofs[tid]
    cio_j = model_joint_cts_offset[tid]
    dio_j = model_joint_dofs_offset[tid]
    bid_B_j = model_joint_bid_B[tid]
    B_r_Bj = model_joint_B_r_Bj[tid]

    # Retrieve the world index offsets
    cio = joint_cts_offsets[wid]
    dio = joint_dofs_offsets[wid]

    # If the base body is the world (bid=-1), use the identity transform (frame of the world's origin)
    T_B_j = wp.transform_identity()
    if bid_B_j > -1:
        T_B_j = state_body_q[bid_B_j]

    # Extract the state of the base body
    r_B_j = wp.transform_get_translation(T_B_j)
    q_B_j = wp.transform_get_rotation(T_B_j)
    R_B_j = wp.quat_to_matrix(q_B_j)

    # Compute the initial pose of the joint frame
    r_j = r_B_j + R_B_j @ B_r_Bj
    T_j = wp.transformation(r_j, q_B_j, dtype=float32)
    state_joint_p_j[tid] = T_j

    # TODO: Compute initial joint velocities

    # Set the initial joint residuals and time-derivatives to zero
    cio_j += cio
    for i in range(c_j):
        state_joint_r_j[dio_j + i] = 0.0
        state_joint_dr_j[dio_j + i] = 0.0

    # Retrieve the initial joint coordinates and velocities
    # NOTE: Currently we are always resetting these to zero
    # TODO: Copy entries from the model initial gencoords of joints
    q_j = vec6f(0.0)
    dq_j = vec6f(0.0)
    tau_j = vec6f(0.0)

    # Copy the initial joint coordinates and velocities to the output arrays
    dio_j += dio
    for i in range(d_j):
        state_joint_q_j[dio_j + i] = q_j[i]
        state_joint_dq_j[dio_j + i] = dq_j[i]
        state_joint_tau_j[dio_j + i] = tau_j[i]


###
# Launchers
###

def reset_joints_state(
    model: JointsModel,
    state: JointsData,
    cts_offsets: wp.array(dtype=int32),
    dofs_offsets: wp.array(dtype=int32),
    body_q: wp.array(dtype=transformf),
    body_u: wp.array(dtype=vec6f)
):
    """
    Reset the current state to the initial state defined in the model.
    """
    wp.launch(
        _reset_joints_state,
        dim=model.num_joints,
        inputs=[
            model.num_dofs,
            model.dofs_offset,
            model.bid_B,
            model.B_r_Bj,
            cts_offsets,
            dofs_offsets,
            body_q,
            body_u,
            state.p_j,
            state.q_j,
            state.dq_j,
            state.tau_j,
        ],
    )
