###########################################################################
# KAMINO: Body Model Types & Containers
###########################################################################

from __future__ import annotations

import warp as wp

from .types import float32, int32, mat33f, transformf, vec6f

###
# Module interface
###

__all__ = [
    "RigidBodiesData",
    "RigidBodiesModel",
    "RigidBodyDescriptor",
    "update_body_inertias",
    "update_body_wrenches",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Rigid-Body Containers
###


class RigidBodyDescriptor:
    """
    A container to describe a single rigid body in the model builder.
    """

    def __init__(self):
        self.name: str | None = None
        """Name of the body."""

        self.uid: str | None = None
        """Unique identifier code (UID) of the body."""

        self.wid: int = 0
        """Index of the world to which the body belongs."""

        self.bid: int = -1
        """Body index of the body w.r.t. its world."""

        self.m_i: float = 0.0
        """Mass of the body."""

        self.i_I_i: mat33f = mat33f()
        """Moment of inertia matrix (in local coordinates) of the body."""

        self.q_i_0: transformf = transformf()
        """Initial absolute pose of the body (in world coordinates)."""

        self.u_i_0: vec6f = vec6f()
        """Initial absolute twist of the body (in world coordinates)."""

    def __repr__(self):
        return (
            f"RigidBodyDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"wid: {self.wid},\n"
            f"bid: {self.bid},\n"
            f"m_i: {self.m_i},\n"
            f"i_I_i:\n{self.i_I_i},\n"
            f"q_i_0: {self.q_i_0},\n"
            f"u_i_0: {self.u_i_0}\n"
            f")"
        )


class RigidBodiesModel:
    """
    An SoA-based container to hold time-invariant model data of a set of rigid body elements.
    """

    def __init__(self):
        self.num_bodies: int = 0
        """Total number of body elements in the model (host-side)."""

        self.wid: wp.array(dtype=int32) | None = None
        """
        World index each body.\n
        Shape of ``(num_bodies,)`` and type :class:`int32`.
        """

        self.bid: wp.array(dtype=int32) | None = None
        """
        Body index of each body w.r.t it's world.\n
        Shape of ``(num_bodies,)`` and type :class:`int32`.
        """

        self.m_i: wp.array(dtype=float32) | None = None
        """
        Mass of each body.\n
        Shape of ``(num_bodies,)`` and type :class:`float32`.
        """

        self.inv_m_i: wp.array(dtype=float32) | None = None
        """
        Inverse mass (1/m_i) of each body.\n
        Shape of ``(num_bodies,)`` and type :class:`float32`.
        """

        self.i_I_i: wp.array(dtype=mat33f) | None = None
        """
        Local moment of inertia of each body.\n
        Shape of ``(num_bodies,)`` and type :class:`mat33f`.
        """

        self.inv_i_I_i: wp.array(dtype=mat33f) | None = None
        """
        Inverse of the local moment of inertia of each body.\n
        Shape of ``(num_bodies,)`` and type :class:`mat33f`.
        """

        self.q_i_0: wp.array(dtype=transformf) | None = None
        """
        Initial pose of each body.\n
        Shape of ``(num_bodies,)`` and type :class:`transformf`.
        """

        self.u_i_0: wp.array(dtype=vec6f) | None = None
        """
        Initial twist of each body.\n
        Shape of ``(num_bodies,)`` and type :class:`vec6f`.
        """


class RigidBodiesData:
    """
    An SoA-based container to hold time-varying data of a set of rigid body elements.
    """

    def __init__(self):
        self.num_bodies: int = 0
        """Total number of body elements in the model (host-side)."""

        self.I_i: wp.array(dtype=mat33f) | None = None
        """
        Moment of inertia (in world coordinates) of each body.\n
        Shape of ``(num_bodies,)`` and type :class:`mat33f`.
        """

        self.inv_I_i: wp.array(dtype=mat33f) | None = None
        """
        Inverse moment of inertia (in world coordinates) of each body.\n
        Shape of ``(num_bodies,)`` and type :class:`mat33f`.
        """

        self.q_i: wp.array(dtype=transformf) | None = None
        """
        Absolute poses of each body (in world coordinates).\n
        Shape of ``(num_bodies,)`` and type :class:`transformf`.
        """

        self.u_i: wp.array(dtype=vec6f) | None = None
        """
        Absolute twists of each body (in world coordinates).\n
        Shape of ``(num_bodies,)`` and type :class:`vec6f`.
        """

        self.w_i: wp.array(dtype=vec6f) | None = None
        """
        Total wrench applied to each body (in world coordinates).\n
        Shape of ``(num_bodies,)`` and type :class:`vec6f`.
        """

        self.w_a_i: wp.array(dtype=vec6f) | None = None
        """
        Joint actuation wrench applied to each body (in world coordinates).\n
        Shape of ``(num_bodies,)`` and type :class:`vec6f`.
        """

        self.w_j_i: wp.array(dtype=vec6f) | None = None
        """
        Joint constraint wrench applied to each body (in world coordinates).\n
        Shape of ``(num_bodies,)`` and type :class:`vec6f`.
        """

        self.w_l_i: wp.array(dtype=vec6f) | None = None
        """
        Joint limit wrench applied to each body (in world coordinates).\n
        Shape of ``(num_bodies,)`` and type :class:`vec6f`.
        """

        self.w_c_i: wp.array(dtype=vec6f) | None = None
        """
        Contact wrench applied to each body (in world coordinates).\n
        Shape of ``(num_bodies,)`` and type :class:`vec6f`.
        """

        self.w_e_i: wp.array(dtype=vec6f) | None = None
        """
        External wrench applied to each body (in world coordinates).\n
        Shape of ``(num_bodies,)`` and type :class:`vec6f`.
        """


###
# Functions
###


###
# Kernels
###


@wp.kernel
def _update_body_inertias(
    # Inputs:
    model_bodies_i_I_i_in: wp.array(dtype=mat33f),
    model_bodies_inv_i_I_i_in: wp.array(dtype=mat33f),
    state_bodies_q_i_in: wp.array(dtype=transformf),
    # Outputs:
    state_bodies_I_i_out: wp.array(dtype=mat33f),
    state_bodies_inv_I_i_out: wp.array(dtype=mat33f),
):
    # Retrieve the thread index as the body index
    bid = wp.tid()

    # Retrieve the model data
    p_i = state_bodies_q_i_in[bid]
    i_I_i = model_bodies_i_I_i_in[bid]
    inv_i_I_i = model_bodies_inv_i_I_i_in[bid]

    # Compute the moment of inertia matrices in world coordinates
    R_i = wp.quat_to_matrix(wp.transform_get_rotation(p_i))
    I_i = R_i @ i_I_i @ wp.transpose(R_i)
    inv_I_i = R_i @ inv_i_I_i @ wp.transpose(R_i)

    # Store results in the output arrays
    state_bodies_I_i_out[bid] = I_i
    state_bodies_inv_I_i_out[bid] = inv_I_i


@wp.kernel
def _update_body_wrenches(
    # Inputs
    state_bodies_w_a_i_in: wp.array(dtype=vec6f),
    state_bodies_w_j_i_in: wp.array(dtype=vec6f),
    state_bodies_w_l_i_in: wp.array(dtype=vec6f),
    state_bodies_w_c_i_in: wp.array(dtype=vec6f),
    state_bodies_w_e_i_in: wp.array(dtype=vec6f),
    # Outputs
    state_bodies_w_i_out: wp.array(dtype=vec6f),
):
    # Retrieve the thread index as the body index
    bid = wp.tid()

    # Retrieve the model data
    w_a_i = state_bodies_w_a_i_in[bid]
    w_j_i = state_bodies_w_j_i_in[bid]
    w_l_i = state_bodies_w_l_i_in[bid]
    w_c_i = state_bodies_w_c_i_in[bid]
    w_e_i = state_bodies_w_e_i_in[bid]

    # Compute the total wrench applied to the body
    w_i = w_a_i + w_j_i + w_l_i + w_c_i + w_e_i

    # Store results in the output arrays
    state_bodies_w_i_out[bid] = w_i


###
# Launchers
###


def update_body_inertias(model: RigidBodiesModel, state: RigidBodiesData):
    wp.launch(
        _update_body_inertias,
        dim=model.num_bodies,
        inputs=[
            # Inputs:
            model.i_I_i,
            model.inv_i_I_i,
            state.q_i,
            # Outputs:
            state.I_i,
            state.inv_I_i,
        ],
    )


def update_body_wrenches(model: RigidBodiesModel, state: RigidBodiesData):
    wp.launch(
        _update_body_wrenches,
        dim=model.num_bodies,
        inputs=[
            # Inputs:
            state.w_a_i,
            state.w_j_i,
            state.w_l_i,
            state.w_c_i,
            state.w_e_i,
            # Outputs:
            state.w_i,
        ],
    )
