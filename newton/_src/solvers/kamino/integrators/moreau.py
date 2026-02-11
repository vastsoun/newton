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
KAMINO: Moreau Midpoint Semi-Implicit Integrator
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from ..core.control import Control as ControlKamino
from ..core.math import (
    quat_box_plus,
    screw_angular,
    screw_linear,
)
from ..core.model import Model as ModelKamino
from ..core.model import ModelData as DataKamino
from ..core.state import State as StateKamino
from ..core.types import (
    float32,
    int32,
    mat33f,
    transformf,
    vec3f,
    vec4f,
    vec6f,
)
from ..geometry.contacts import Contacts as ContactsKamino
from ..geometry.detector import CollisionDetector
from ..kinematics.limits import Limits as LimitsKamino
from .euler import euler_semi_implicit_with_logmap

###
# Module interface
###

__all__ = ["IntegratorMoreauJean"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _integrate_moreau_jean_first_inplace(
    # Inputs:
    model_dt: wp.array(dtype=float32),
    model_bodies_wid: wp.array(dtype=int32),
    bodies_u: wp.array(dtype=vec6f),
    # Outputs:
    bodies_q: wp.array(dtype=transformf),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the world index
    wid = model_bodies_wid[tid]

    # Retrieve the time step and gravity vector
    dt = 0.5 * model_dt[wid]

    # Retrieve the current state of the body
    p_i = bodies_q[tid]
    u_i = bodies_u[tid]

    # Extract linear and angular parts
    r_i = wp.transform_get_translation(p_i)
    q_i = wp.transform_get_rotation(p_i)
    v_i = screw_linear(u_i)
    omega_i = screw_angular(u_i)

    # Compute configuration-level update
    r_i_n = r_i + dt * v_i
    q_i_n = quat_box_plus(q_i, dt * omega_i)
    p_i_n = wp.transformf(r_i_n, q_i_n)

    # Store the computed next pose and twist
    bodies_q[tid] = p_i_n


@wp.kernel
def _integrate_moreau_jean_second_inplace(
    # Inputs:
    alpha: float,
    model_dt: wp.array(dtype=float32),
    model_gravity: wp.array(dtype=vec4f),
    model_bodies_wid: wp.array(dtype=int32),
    model_bodies_inv_m: wp.array(dtype=float32),
    model_bodies_I: wp.array(dtype=mat33f),
    model_bodies_inv_I: wp.array(dtype=mat33f),
    state_bodies_w: wp.array(dtype=vec6f),
    # Outputs:
    state_bodies_q: wp.array(dtype=transformf),
    state_bodies_u: wp.array(dtype=vec6f),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the world index
    wid = model_bodies_wid[tid]

    # Retrieve the time step and gravity vector
    dt = model_dt[wid]
    gv = model_gravity[wid]
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Retrieve the model data
    inv_m_i = model_bodies_inv_m[tid]
    I_i = model_bodies_I[tid]
    inv_I_i = model_bodies_inv_I[tid]

    # Retrieve the current state of the body
    q_i = state_bodies_q[tid]
    u_i = state_bodies_u[tid]
    w_i = state_bodies_w[tid]

    # Compute the next pose and twist
    q_i_n, u_i_n = euler_semi_implicit_with_logmap(
        alpha,
        0.5 * dt,
        g,
        inv_m_i,
        I_i,
        inv_I_i,
        q_i,
        u_i,
        w_i,
    )

    # Store the computed next pose and twist
    state_bodies_q[tid] = q_i_n
    state_bodies_u[tid] = u_i_n


###
# Interfaces
###


class IntegratorMoreauJean:
    """
    TODO
    """

    def __init__(self, model: ModelKamino, alpha: float = 0.0):
        """
        TODO
        """
        self._model = model
        self._alpha = alpha

    ###
    # Operations
    ###

    def integrate(
        self,
        forward: Callable,
        model: ModelKamino,
        data: DataKamino,
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        detector: CollisionDetector | None = None,
    ):
        """
        Solves the time integration sub-problem to compute the next state of the system.
        """

        # Take the first semi-step until the mid-point of the step
        # NOTE: We only integrate on configuration level
        # q_M = q_i + 1/2 * dt * G(q_i) * u_i
        self._integrate1(model=model, data=data)

        # If a collision detector is provided, use it to generate
        # the set of active contacts at the current state
        if detector:
            detector.collide(model=model, data=data, contacts=contacts)

        # Solve the forward dynamics sub-problem to compute the
        # constraint reactions at the mid-point of the step
        forward(
            state_in=state_in,
            state_out=state_out,
            control=control,
            limits=limits,
            contacts=contacts,
        )

        # Take the second semi-step until the end of the step
        # u_E = u_S + dt * M(q_M)^{-1} * (dt * h(q_M, u_S) + dt * J_a^T(q_M) * tau_j + J_c^T(q_M) * lambda)
        # q_E = q_M + 1/2 * dt * G(q_M) * u_E
        self._integrate2(model=model, data=data)

    ###
    # Operations
    ###

    def _integrate1(self, model: ModelKamino, data: DataKamino):
        """
        TODO
        """
        wp.launch(
            _integrate_moreau_jean_first_inplace,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.time.dt,
                model.bodies.wid,
                data.bodies.u_i,
                # Outputs:
                data.bodies.q_i,
            ],
        )

    def _integrate2(self, model: ModelKamino, data: DataKamino):
        """
        TODO
        """
        wp.launch(
            _integrate_moreau_jean_second_inplace,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                self._alpha,  # alpha: angular damping
                model.time.dt,
                model.gravity.vector,
                model.bodies.wid,
                model.bodies.inv_m_i,
                data.bodies.I_i,
                data.bodies.inv_I_i,
                data.bodies.w_i,
                # Outputs:
                data.bodies.q_i,
                data.bodies.u_i,
            ],
        )
