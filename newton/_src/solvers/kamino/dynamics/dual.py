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
KAMINO: Dual dynamics
"""

from __future__ import annotations

import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.math import UNIT_Z, screw, screw_angular, screw_linear
from newton._src.solvers.kamino.core.model import Model, ModelData, ModelSize
from newton._src.solvers.kamino.core.types import (
    float32,
    int32,
    mat33f,
    vec2f,
    vec3f,
    vec4f,
    vec6f,
)
from newton._src.solvers.kamino.dynamics.delassus import DelassusOperator
from newton._src.solvers.kamino.geometry.contacts import Contacts, ContactsData
from newton._src.solvers.kamino.kinematics.jacobians import DenseSystemJacobians, DenseSystemJacobiansData
from newton._src.solvers.kamino.kinematics.limits import Limits, LimitsData
from newton._src.solvers.kamino.linalg.cholesky import CholeskyFactorizer

###
# Module interface
###

__all__ = [
    "DualProblem",
    "DualProblemConfig",
    "DualProblemData",
    "DualProblemSettings",
    "build_free_velocity",
    "build_free_velocity_bias",
    "build_generalized_free_velocity",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


@wp.struct
class DualProblemConfig:
    """
    A struct to hold configuration parameters of a dual problem.
    """

    alpha: float32
    """Baumgarte stabilization parameter for bilateral joint constraints."""
    beta: float32
    """Baumgarte stabilization parameter for unilateral joint limit constraints."""
    gamma: float32
    """Baumgarte stabilization parameter for unilateral contact constraints."""
    delta: float32
    """Contact penetration margin used for unilateral contact constraints"""


class DualProblemSettings:
    """
    A struct to hold configuration parameters of a dual problem.
    """

    def __init__(self, alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0, delta: float = 1.0e-6):
        self.alpha: float = alpha
        """Baumgarte stabilization parameter for bilateral joint constraints."""
        self.beta: float = beta
        """Baumgarte stabilization parameter for unilateral joint limit constraints."""
        self.gamma: float = gamma
        """Baumgarte stabilization parameter for unilateral contact constraints."""
        self.delta: float = delta
        """Contact penetration margin used for unilateral contact constraints"""

    def to_config(self) -> DualProblemConfig:
        """
        Convert the settings to a DualProblemConfig struct.
        """
        config = DualProblemConfig()
        config.alpha = wp.float32(self.alpha)
        config.beta = wp.float32(self.beta)
        config.gamma = wp.float32(self.gamma)
        config.delta = wp.float32(self.delta)
        return config


class DualProblemData:
    """
    A container to hold the the dual problem of forward dynamics.
    """

    def __init__(self):
        self.num_worlds: int32 = 0
        """The number of worlds represented in the dual problem data."""

        self.num_maxdims: int32 = 0
        """The maximum number of dual problem dimensions (i.e. constraints) across all worlds.\n"""

        ###
        # Problem configurations
        ###

        self.config: wp.array(dtype=DualProblemConfig) | None = None
        """
        Problem configuration parameters for each world.\n
        Shape of ``(num_worlds,)`` and type :class:`DualProblemConfig`.
        """

        ###
        # Constraints info
        ###

        self.nl: wp.array(dtype=int32) | None = None
        """
        The number of active limit constraints in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.nc: wp.array(dtype=int32) | None = None
        """
        The number of active contact constraints in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.lio: wp.array(dtype=int32) | None = None
        """
        The limit index offset of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        self.cio: wp.array(dtype=int32) | None = None
        """
        The contact index offset of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        self.uio: wp.array(dtype=int32) | None = None
        """
        The unilateral index offset of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        self.lcgo: wp.array(dtype=int32) | None = None
        """
        The limit constraint group offset of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        self.ccgo: wp.array(dtype=int32) | None = None
        """
        The contact constraint group offset of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        ###
        # Delassus operator
        ###

        self.maxdim: wp.array(dtype=int32) | None = None
        """
        The maximum number of dual problem dimensions of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.dim: wp.array(dtype=int32) | None = None
        """
        The active number of dual problem dimensions of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.mio: wp.array(dtype=int32) | None = None
        """
        The matrix index offset of each Delassus matrix block.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        This is applicable to `D` as well as to its (optional) factorizations.\n
        """

        self.vio: wp.array(dtype=int32) | None = None
        """
        The vector index offset of each constraint dimension vector block.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        This is applicable to `v_b`, `v_i` and `v_f`.\n
        """

        self.D: wp.array(dtype=float32) | None = None
        """
        The flat array of Delassus matrix blocks (constraint-space apparent inertia).\n
        Shape of ``(sum(nd_w * nd_w),)`` and type :class:`float32`.
        """

        ###
        # Auxiliary vectors
        ###

        # TODO: remove these later
        self.h: wp.array(dtype=vec6f) | None = None
        """
        The array of non-linear generalized forces vectors.\n
        Shape of ``(sum(nb_w),)`` and type :class:`vec6f`.
        """

        ###
        # Velocity vectors
        ###

        self.u_f: wp.array(dtype=vec6f) | None = None
        """
        The array of unconstrained generalized velocity vectors.\n
        Shape of ``(sum(nb_w),)`` and type :class:`vec6f`.
        """

        self.v_b: wp.array(dtype=float32) | None = None
        """
        The stack of free-velocity statbilization biases vectors (in constraint-space).\n
        Shape of ``(sum(nd_w),)`` and type :class:`float32`.
        """

        self.v_i: wp.array(dtype=float32) | None = None
        """
        The stack of free-velocity impact biases vector (in constraint-space).\n
        Shape of ``(sum(nd_w),)`` and type :class:`float32`.
        """

        self.v_f: wp.array(dtype=float32) | None = None
        """
        The stack of free-velocity vector (constraint-space unconstrained velocity).\n
        Shape of ``(sum(nd_w),)`` and type :class:`float32`.
        """

        ###
        # Friction models
        ###

        self.mu: wp.array(dtype=float32) | None = None
        """
        The stack of friction coefficient vectors.\n
        Shape of ``(sum(nc_w),)`` and type :class:`float32`.
        """


###
# Functions
###


@wp.func
def gravity_plus_coriolis_wrench(
    g: vec3f,
    m_i: float32,
    I_i: mat33f,
    omega_i: vec3f,
) -> vec6f:
    """
    Compute the gravitational+inertial wrench on a body.
    """
    f_gi_i = m_i * g
    tau_gi_i = -wp.skew(omega_i) @ (I_i @ omega_i)
    return vec6f(f_gi_i.x, f_gi_i.y, f_gi_i.z, tau_gi_i.x, tau_gi_i.y, tau_gi_i.z)


@wp.func
def gravity_plus_coriolis_wrench_split(
    g: vec3f,
    m_i: float32,
    I_i: mat33f,
    omega_i: vec3f,
):
    """
    Compute the gravitational+inertial wrench on a body.
    """
    f_gi_i = m_i * g
    tau_gi_i = -wp.skew(omega_i) @ (I_i @ omega_i)
    return f_gi_i, tau_gi_i


###
# Kernels
###


@wp.kernel
def _build_nonlinear_generalized_force(
    # Inputs:
    model_time_dt: wp.array(dtype=float32),
    model_gravity_vector: wp.array(dtype=vec4f),
    model_bodies_wid: wp.array(dtype=int32),
    model_bodies_m_i: wp.array(dtype=float32),
    state_bodies_u_i: wp.array(dtype=vec6f),
    state_bodies_I_i: wp.array(dtype=mat33f),
    state_bodies_w_e_i: wp.array(dtype=vec6f),
    state_bodies_w_a_i: wp.array(dtype=vec6f),
    # Outputs:
    problem_h: wp.array(dtype=vec6f),
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Retrieve the body model and state data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    u_i = state_bodies_u_i[bid]
    w_e_i = state_bodies_w_e_i[bid]
    w_a_i = state_bodies_w_a_i[bid]

    # Get world data
    dt = model_time_dt[wid]
    gv = model_gravity_vector[wid]

    # Extract the effective gravity vector
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Extract the linear and angular components of the generalized velocity
    omega_i = screw_angular(u_i)

    # Compute the net external wrench on the body
    h_i = w_e_i + w_a_i + gravity_plus_coriolis_wrench(g, m_i, I_i, omega_i)

    # Store the generalized free-velocity vector
    problem_h[bid] = dt * h_i


@wp.kernel
def _build_generalized_free_velocity(
    # Inputs:
    model_time_dt: wp.array(dtype=float32),
    model_gravity_vector: wp.array(dtype=vec4f),
    model_bodies_wid: wp.array(dtype=int32),
    model_bodies_m_i: wp.array(dtype=float32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    state_bodies_u_i: wp.array(dtype=vec6f),
    state_bodies_I_i: wp.array(dtype=mat33f),
    state_bodies_inv_I_i: wp.array(dtype=mat33f),
    state_bodies_w_e_i: wp.array(dtype=vec6f),
    state_bodies_w_a_i: wp.array(dtype=vec6f),
    # Outputs:
    problem_u_f: wp.array(dtype=vec6f),
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Retrieve the body model and state data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    inv_m_i = model_bodies_inv_m_i[bid]
    inv_I_i = state_bodies_inv_I_i[bid]
    u_i = state_bodies_u_i[bid]
    w_e_i = state_bodies_w_e_i[bid]
    w_a_i = state_bodies_w_a_i[bid]

    # Get world data
    dt = model_time_dt[wid]
    gv = model_gravity_vector[wid]

    # Extract the effective gravity vector
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Extract the linear and angular components of the generalized velocity
    v_i = screw_linear(u_i)
    omega_i = screw_angular(u_i)

    # Compute the net external wrench on the body
    h_i = w_e_i + w_a_i + gravity_plus_coriolis_wrench(g, m_i, I_i, omega_i)
    f_h_i = screw_linear(h_i)
    tau_h_i = screw_angular(h_i)

    # Compute the generalized free-velocity vector components
    v_f_i = v_i + dt * (inv_m_i * f_h_i)
    omega_f_i = omega_i + dt * (inv_I_i @ tau_h_i)

    # Store the generalized free-velocity vector
    problem_u_f[bid] = screw(v_f_i, omega_f_i)


@wp.kernel
def _build_free_velocity_bias_joints(
    # Inputs:
    model_info_joint_cts_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_num_cts: wp.array(dtype=int32),
    model_joints_cts_offset: wp.array(dtype=int32),
    state_joints_r_j: wp.array(dtype=float32),
    problem_config: wp.array(dtype=DualProblemConfig),
    problem_vio: wp.array(dtype=int32),
    # Outputs:
    problem_v_b: wp.array(dtype=float32),
):
    # Retrieve the joint index as the thread index
    jid = wp.tid()

    # Retrieve the world index from the joint
    wid = model_joints_wid[jid]

    # Retrieve the joint constraint index offset
    cts_offset = model_info_joint_cts_offset[wid]

    # Retrieve the model time step
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the dual problem config
    config = problem_config[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrieve the joint constraint index and offset
    ncts_j = model_joints_num_cts[jid]
    ctsio_j = model_joints_cts_offset[jid]

    # Compute baumgarte constraint stabilization coefficient
    c_b = config.alpha * inv_dt

    # Compute block offsets for the constraint and residual vectors
    vio_j = vio + ctsio_j
    rio_j = cts_offset + ctsio_j

    # Compute the free-velocity bias for the joint
    for j in range(ncts_j):
        problem_v_b[vio_j + j] = c_b * state_joints_r_j[rio_j + j]


@wp.kernel
def _build_free_velocity_bias_limits(
    # Inputs:
    model_time_inv_dt: wp.array(dtype=float32),
    state_info_limit_cts_group_offset: wp.array(dtype=int32),
    limits_model_num: wp.array(dtype=int32),
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_r_q: wp.array(dtype=float32),
    problem_config: wp.array(dtype=DualProblemConfig),
    problem_vio: wp.array(dtype=int32),
    # Outputs:
    problem_v_b: wp.array(dtype=float32),
):
    # Retrieve the limit index as the thread index
    tid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nl = limits_model_num[0]

    # Skip if cid is greater than the number of contacts active in the world
    if tid >= model_nl:
        return

    # Retrieve the limit entity data
    wid = limits_wid[tid]
    lid = limits_lid[tid]
    r_q = limits_r_q[tid]

    # Retrieve the world-specific data
    inv_dt = model_time_inv_dt[wid]
    config = problem_config[wid]
    vio = problem_vio[wid]
    lcio = state_info_limit_cts_group_offset[wid]

    # Compute the total constraint index offset of the current contact
    lcio_l = vio + lcio + lid

    # Compute the contact constraint stabilization bias
    problem_v_b[lcio_l] = config.beta * inv_dt * wp.min(0.0, r_q)


@wp.kernel
def _build_free_velocity_bias_contacts(
    # Inputs:
    model_time_inv_dt: wp.array(dtype=float32),
    model_info_contacts_offset: wp.array(dtype=int32),
    state_info_contact_cts_group_offset: wp.array(dtype=int32),
    contacts_model_num: wp.array(dtype=int32),
    contacts_wid: wp.array(dtype=int32),
    contacts_cid: wp.array(dtype=int32),
    contacts_gapfunc: wp.array(dtype=vec4f),
    contacts_material: wp.array(dtype=vec2f),
    problem_config: wp.array(dtype=DualProblemConfig),
    problem_vio: wp.array(dtype=int32),
    # Outputs:
    problem_v_b: wp.array(dtype=float32),
    problem_v_i: wp.array(dtype=float32),
    problem_mu: wp.array(dtype=float32),
):
    # Retrieve the contact index as the thread index
    tid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nc = contacts_model_num[0]

    # Skip if cid is greater than the number of contacts active in the world
    if tid >= model_nc:
        return

    # Retrieve the contact entity data
    wid_k = contacts_wid[tid]
    cid_k = contacts_cid[tid]
    material_k = contacts_material[tid]
    penetration_k = contacts_gapfunc[tid][3]

    # Retrieve the world-specific data
    inv_dt = model_time_inv_dt[wid_k]
    cio = model_info_contacts_offset[wid_k]
    ccio = state_info_contact_cts_group_offset[wid_k]
    vio = problem_vio[wid_k]
    config = problem_config[wid_k]

    # Compute the total constraint index offset of the current contact
    ccio_k = vio + ccio + 3 * cid_k

    # Compute the total contact index offset of the current contact
    cio_k = cio + cid_k

    # Retrive the contact material properties
    mu_k = material_k.x  # Friction coefficient
    epsilon_k = material_k.y  # Penetration reduction coefficient

    # Compute the constraint residuals for unilateral contact constraints
    # NOTE#1: The residuals correspond to configuration-level constraint
    # violation of each contact along the corresponding normal direction
    # NOTE#2: contact penetration is assumed to be represented using
    # non-positive values (d <= 0), and since the penetration value
    # in the container is positive (p >= 0) we need to invert the sign
    # TODO: How to best use config.delta?
    distance_k = wp.min(0.0, penetration_k)

    # Compute the per-contact penetration error reduction term
    # NOTE#1: Penetrations are represented as xi < 0 (hence the sign inversion)
    # NOTE#2: xi_p_relaxed corresponds to one-sided Baumgarte-like stabilization
    xi = inv_dt * distance_k
    xi_relaxed = config.gamma * wp.min(0.0, xi) + wp.max(0.0, xi)
    if epsilon_k == 1.0:
        alpha = 0.0
    else:
        alpha = 1.0

    # Compute the contact constraint stabilization bias
    v_b_k = alpha * xi_relaxed * UNIT_Z

    # Store the contact constraint stabilization bias in the output vector
    for i in range(3):
        problem_v_b[ccio_k + i] = v_b_k[i]

    # Initialize the restitutive Newton-type impact model term
    problem_v_i[ccio_k] = 0.0
    problem_v_i[ccio_k + 1] = 0.0
    problem_v_i[ccio_k + 2] = epsilon_k

    # Store the contact friction coefficient in the output vector
    problem_mu[cio_k] = mu_k


@wp.kernel
def _build_free_velocity(
    # Inputs:
    model_info_num_bodies: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    state_bodies_u_i: wp.array(dtype=vec6f),
    jacobians_J_cts_offsets: wp.array(dtype=int32),
    jacobians_J_cts_data: wp.array(dtype=float32),
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_u_f: wp.array(dtype=vec6f),
    problem_v_b: wp.array(dtype=float32),
    problem_v_i: wp.array(dtype=float32),
    # Outputs:
    problem_v_f: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the problem dimensions and matrix block index offset
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the world-specific data
    cjmio = jacobians_J_cts_offsets[wid]
    bio = model_info_bodies_offset[wid]
    nb = model_info_num_bodies[wid]
    vio = problem_vio[wid]

    # Compute the number of Jacobian rows, i.e. the number of body DoFs
    nbd = 6 * nb

    # Compute the thread-specific index offset
    tio = vio + tid

    # Append the column offset to the Jacobian index
    cjmio += nbd * tid

    # Extract the cached impact bias scaling (i.e. restitution coefficient)
    # NOTE: This is a quick hack to avoid multiple kernels. The
    # proper way would be to perform this op only for contacts
    epsilon_j = problem_v_i[tio]

    # Buffers
    J_i = vec6f(0.0)
    v_f_j = float32(0.0)

    # Iterate over each body to accumulate velocity contributions
    for i in range(nb):
        # Compute the Jacobian block index
        m_ji = cjmio + 6 * i

        # Extract the twist and unconstrained velocity of the body
        u_i = state_bodies_u_i[bio + i]
        u_f_i = problem_u_f[bio + i]

        # Extract the Jacobian block J_ji
        # TODO: use slicing operation when available
        for d in range(6):
            J_i[d] = jacobians_J_cts_data[m_ji + d]

        # Accumulate J_i @ u_i
        v_f_j += wp.dot(J_i, u_f_i)

        # Accumulate the impact bias term
        v_f_j += epsilon_j * wp.dot(J_i, u_i)

    # Store sum of velocity bias terms
    problem_v_f[tio] = v_f_j + problem_v_b[tio]


###
# Launchers
###


def build_nonlinear_generalized_force(model: Model, state: ModelData, problem: DualProblemData):
    """
    Builds the generalized free-velocity vector (i.e. unconstrained) `u_f`.
    """
    wp.launch(
        _build_nonlinear_generalized_force,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            model.time.dt,
            model.gravity.vector,
            model.bodies.wid,
            model.bodies.m_i,
            state.bodies.u_i,
            state.bodies.I_i,
            state.bodies.w_e_i,
            state.bodies.w_a_i,
            # Outputs:
            problem.h,
        ],
    )


def build_generalized_free_velocity(model: Model, state: ModelData, problem: DualProblemData):
    """
    Builds the generalized free-velocity vector (i.e. unconstrained) `u_f`.
    """
    wp.launch(
        _build_generalized_free_velocity,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            model.time.dt,
            model.gravity.vector,
            model.bodies.wid,
            model.bodies.m_i,
            model.bodies.inv_m_i,
            state.bodies.u_i,
            state.bodies.I_i,
            state.bodies.inv_I_i,
            state.bodies.w_e_i,
            state.bodies.w_a_i,
            # Outputs:
            problem.u_f,
        ],
    )


def build_free_velocity_bias(
    model: Model, state: ModelData, limits: LimitsData, contacts: ContactsData, problem: DualProblemData
):
    """
    Builds the joint constraint section of the free-velocity vector.
    """
    wp.launch(
        _build_free_velocity_bias_joints,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.info.joint_cts_offset,
            model.time.inv_dt,
            model.joints.wid,
            model.joints.num_cts,
            model.joints.cts_offset,
            state.joints.r_j,
            problem.config,
            problem.vio,
            # Outputs:
            problem.v_b,
        ],
    )

    if limits is not None:
        wp.launch(
            _build_free_velocity_bias_limits,
            dim=limits.num_model_max_limits,
            inputs=[
                # Inputs:
                model.time.inv_dt,
                state.info.limit_cts_group_offset,
                limits.model_num_limits,
                limits.wid,
                limits.lid,
                limits.r_q,
                problem.config,
                problem.vio,
                # Outputs:
                problem.v_b,
            ],
        )

    if contacts is not None:
        wp.launch(
            _build_free_velocity_bias_contacts,
            dim=contacts.num_model_max_contacts,
            inputs=[
                # Inputs:
                model.time.inv_dt,
                model.info.contacts_offset,
                state.info.contact_cts_group_offset,
                contacts.model_num_contacts,
                contacts.wid,
                contacts.cid,
                contacts.gapfunc,
                contacts.material,
                problem.config,
                problem.vio,
                # Outputs:
                problem.v_b,
                problem.v_i,
                problem.mu,
            ],
        )


def build_free_velocity(model: Model, state: ModelData, jacobians: DenseSystemJacobians, problem: DualProblem):
    """
    Builds the joint constraint section of the free-velocity vector.
    """
    wp.launch(
        _build_free_velocity,
        dim=(problem._size.num_worlds, problem._size.max_of_max_total_cts),
        inputs=[
            # Inputs:
            model.info.num_bodies,
            model.info.bodies_offset,
            state.bodies.u_i,
            jacobians.data.J_cts_offsets,
            jacobians.data.J_cts_data,
            problem.data.dim,
            problem.data.vio,
            problem.data.u_f,
            problem.data.v_b,
            problem.data.v_i,
            # Outputs:
            problem.data.v_f,
        ],
    )


###
# Interfaces
###


class DualProblem:
    """
    A container to hold, manage and operate a dynamics dual problem.
    """

    @staticmethod
    def _check_settings(
        settings: list[DualProblemSettings] | DualProblemSettings | None, num_worlds: int
    ) -> list[DualProblemSettings]:
        """
        Checks and prepares the settings for the dual problem.

        If a single `DualProblemSettings` object is provided, it will be replicated for all worlds.
        If a list of settings is provided, it will ensure that the number of settings matches the number of worlds.
        """
        if settings is None:
            # If no settings are provided, use default settings
            return [DualProblemSettings()] * num_worlds
        elif isinstance(settings, DualProblemSettings):
            # If a single settings object is provided, replicate it for all worlds
            return [settings] * num_worlds
        elif isinstance(settings, list):
            # Ensure the settings are of the correct type and length
            if len(settings) != num_worlds:
                raise ValueError(f"Expected {num_worlds} settings, got {len(settings)}")
            for s in settings:
                if not isinstance(s, DualProblemSettings):
                    raise TypeError(f"Expected DualProblemSettings, got {type(s)}")
            return settings
        else:
            raise TypeError(f"Expected List[DualProblemSettings] or DualProblemSettings, got {type(settings)}")

    def __init__(
        self,
        model: Model | None = None,
        state: ModelData | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        factorizer: CholeskyFactorizer = None,
        settings: list[DualProblemSettings] | DualProblemSettings | None = None,
        device: Devicelike = None,
    ):
        """
        Constructs a dual problem interface container.

        If `model`, `limits` and/or `contacts` containers are provided, it allocates the dual problem data members.
        Only the `model` is strictly required for the allocation, but the resulting dual problem will only represent
        bilateral (i.e. equality) joint constraints and possibly some unilateral (i.e. inequality) joint limits, but
        not contact constraints. The `contacts` container is required if the dual problem is to also incorporate
        contact constraints. If no `model` is provided at construction time, then deferred allocation is possible
        by calling the `allocate()` method at a later point.

        Args:
            model (Model, optional): The model to build the dual problem for.
            contacts (Contacts, optional): The contacts container to use for the dual problem.
            device (Devicelike, optional): The device to allocate the dual problem on. Defaults to None.
            factorizer (CholeskyFactorizer, optional): The factorizer to use for the Delassus operator. Defaults to None.
        """
        # Cache the requested device
        self._device: Devicelike = device

        # Declare the model size cache
        self._size: ModelSize | None = None

        self._settings: list[DualProblemSettings] = []
        """Host-side cache of the list of per world dual problem settings."""

        self._delassus: DelassusOperator | None = None
        """The Delassus operator interface container."""

        self._data: DualProblemData = DualProblemData()
        """The dual problem state data container bundling are relevant memory allocations."""

        # Allocate the dual problem state if a model is provided
        if model is not None:
            self.allocate(
                model=model,
                state=state,
                limits=limits,
                contacts=contacts,
                factorizer=factorizer,
                settings=settings,
                device=device,
            )

    @property
    def size(self) -> ModelSize:
        """
        Returns the model size of the dual problem.
        This is the size of the model that the dual problem is built for.
        """
        if self._size is None:
            raise ValueError("Model size is not allocated. Call `allocate()` first.")
        return self._size

    @property
    def settings(self) -> list[DualProblemSettings]:
        """
        Returns the list of per world dual problem settings.
        """
        return self._settings

    @settings.setter
    def settings(self, value: list[DualProblemSettings] | DualProblemSettings):
        """
        Sets the list of per world dual problem settings.
        If a single `DualProblemSettings` object is provided, it will be replicated for all worlds.
        """
        self._settings = self._check_settings(value, self._data.num_worlds)

    @property
    def delassus(self) -> DelassusOperator:
        """
        Returns the Delassus operator interface.
        """
        if self._delassus is None:
            raise ValueError("Delassus operator is not allocated. Call `allocate()` first.")
        return self._delassus

    @property
    def data(self) -> DualProblemData:
        """
        Returns the dual problem state data container.
        """
        return self._data

    def allocate(
        self,
        model: Model,
        state: ModelData | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        factorizer: CholeskyFactorizer = None,
        settings: list[DualProblemSettings] | DualProblemSettings | None = None,
        device: Devicelike = None,
    ):
        """
        Allocates the dual problem state for the given model and required contact allocations.

        Args:
            model (Model, optional): The model to build the dual problem for.
            contacts (Contacts, optional): The contacts container to use for the dual problem.
            factorizer (CholeskyFactorizer, optional): The factorizer to use for the Delassus operator.
            settings (List[DualProblemSettings] | DualProblemSettings, optional): The settings for the dual problem.
            device (Devicelike, optional): The device to allocate the dual problem on. Defaults to None.
        """
        # Ensure the model is valid
        if model is None:
            raise ValueError("A model of type `Model` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, Model):
            raise ValueError("Invalid model provided. Must be an instance of `Model`.")

        # Ensure the state container is valid if provided
        if state is not None:
            if not isinstance(state, ModelData):
                raise ValueError("Invalid state container provided. Must be an instance of `ModelData`.")

        # Ensure the limits container is valid if provided
        if limits is not None:
            if not isinstance(limits, Limits):
                raise ValueError("Invalid limits container provided. Must be an instance of `Limits`.")

        # Ensure the contacts container is valid if provided
        if contacts is not None:
            if not isinstance(contacts, Contacts):
                raise ValueError("Invalid contacts container provided. Must be an instance of `Contacts`.")

        # Capture reference to the model size
        self._size = model.size

        # Check settings validity and update cache
        self._settings = self._check_settings(settings, model.info.num_worlds)

        # Allocate the Delassus operator first since it will already process the necessary
        # model and contacts allocation sizes and will create some of the necessary arrays
        self._delassus = DelassusOperator(
            model=model, state=state, limits=limits, contacts=contacts, factorizer=factorizer, device=device
        )

        # Update the cache of the maximal problem dimensions
        self._data.num_worlds = self._delassus.num_worlds
        self._data.num_maxdims = self._delassus.num_maxdims

        # Capture references to the model, state info arrays
        # TODO: How to handle the case where state is None?
        self.data.nl = state.info.num_limits
        self.data.nc = state.info.num_contacts
        self.data.lio = model.info.limits_offset
        self.data.cio = model.info.contacts_offset
        self.data.uio = model.info.unilaterals_offset
        self.data.lcgo = state.info.limit_cts_group_offset
        self.data.ccgo = state.info.contact_cts_group_offset

        # Capture references to arrays already create by the Delassus operator
        self._data.maxdim = self._delassus.data.maxdim
        self._data.dim = self._delassus.data.dim
        self._data.mio = self._delassus.data.mio
        self._data.vio = self._delassus.data.vio
        self._data.D = self._delassus.data.D

        # Store the specified settings
        num_worlds = model.info.num_worlds if model is not None else 1
        self._settings: list[DualProblemSettings] = self._check_settings(settings, num_worlds)

        # Allocate memory for the remaining dual problem quantities
        with wp.ScopedDevice(device):
            self._data.config = wp.array([s.to_config() for s in self.settings], dtype=DualProblemConfig)
            self._data.h = wp.zeros(shape=(model.size.sum_of_num_bodies,), dtype=vec6f)  # TODO: remove these later
            self._data.u_f = wp.zeros(shape=(model.size.sum_of_num_bodies,), dtype=vec6f)
            self._data.v_b = wp.zeros(shape=(self._delassus.num_maxdims,), dtype=float32)
            self._data.v_i = wp.zeros(shape=(self._delassus.num_maxdims,), dtype=float32)
            self._data.v_f = wp.zeros(shape=(self._delassus.num_maxdims,), dtype=float32)
            self._data.mu = wp.zeros(shape=(contacts.num_model_max_contacts,), dtype=float32)

    def zero(self):
        self._data.h.zero_()  # TODO: remove these later
        self._data.u_f.zero_()
        self._data.v_b.zero_()
        self._data.v_i.zero_()
        self._data.v_f.zero_()
        self._data.mu.zero_()

    def build(
        self,
        model: Model,
        state: ModelData,
        limits: LimitsData,
        contacts: ContactsData,
        jacobians: DenseSystemJacobiansData,
        reset_to_zero: bool = True,
    ):
        """
        Builds the dual problem for the given model, state, limits and contacts data.
        """
        # Initialize problem data
        if reset_to_zero:
            self.zero()

        # Build the Delassus operator
        # NOTE: We build this first since it will update the arrays of active constraints
        self._delassus.build(
            model=model,
            state=state,
            jacobians=jacobians,
            reset_to_zero=reset_to_zero,
        )

        # TODO: remove this later
        # Build the non-linear generalized force vector
        build_nonlinear_generalized_force(model, state, self._data)

        # Build the generalized free-velocity vector
        build_generalized_free_velocity(model, state, self._data)

        # Build the free-velocity bias terms
        build_free_velocity_bias(model, state, limits, contacts, self._data)

        # Build the free-velocity vector
        wp.launch(
            _build_free_velocity,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                model.info.num_bodies,
                model.info.bodies_offset,
                state.bodies.u_i,
                jacobians.J_cts_offsets,
                jacobians.J_cts_data,
                self._data.dim,
                self._data.vio,
                self._data.u_f,
                self._data.v_b,
                self._data.v_i,
                # Outputs:
                self._data.v_f,
            ],
        )
