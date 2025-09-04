###########################################################################
# KAMINO: Euler Integrators
###########################################################################

from __future__ import annotations

import warp as wp

from newton._src.solvers.kamino.core.types import int32, float32, vec3f, vec4f, mat33f, transformf, vec6f
from newton._src.solvers.kamino.core.math import screw, screw_linear, screw_angular, quat_box_plus
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.state import State


###
# Module interface
###

__all__ = [
    "integrate_semi_implicit_euler",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###

@wp.func
def semi_implicit_euler_with_logmap(
    dt: float32,
    g: vec3f,
    inv_m_i: float32,
    I_i: mat33f,
    inv_I_i: mat33f,
    p_i: transformf,
    u_i: vec6f,
    w_i: vec6f,
):
    # Extract linear and angular parts
    r_i = wp.transform_get_translation(p_i)
    q_i = wp.transform_get_rotation(p_i)
    v_i = screw_linear(u_i)
    omega_i = screw_angular(u_i)
    S_i = wp.skew(omega_i)
    f_i = screw_linear(w_i)
    tau_i = screw_angular(w_i)

    # Compute velocity update equations
    v_i_n = v_i + dt * (g + inv_m_i * f_i)
    omega_i_n = omega_i + dt * inv_I_i @ (-S_i @ I_i @ omega_i + tau_i)

    # Compute configuration update equations
    r_i_n = r_i + dt * v_i_n
    q_i_n = quat_box_plus(q_i, dt * omega_i_n)

    # Compute the new pose and twist
    p_i_n = wp.transformation(r_i_n, q_i_n)
    u_i_n = screw(v_i_n, omega_i_n)

    # Return the new pose and twist
    return p_i_n, u_i_n


###
# Kernels
###

@wp.kernel
def _integrate_semi_implicit_euler(
    # Model inputs:
    model_dt_in: wp.array(dtype=float32),
    model_gravity_in: wp.array(dtype=vec4f),
    model_bodies_wid_in: wp.array(dtype=int32),
    model_bodies_inv_m_in: wp.array(dtype=float32),
    model_bodies_I_in: wp.array(dtype=mat33f),
    model_bodies_inv_I_in: wp.array(dtype=mat33f),
    # State inputs:
    state_bodies_q_in: wp.array(dtype=transformf),
    state_bodies_u_in: wp.array(dtype=vec6f),
    state_bodies_w_in: wp.array(dtype=vec6f),
    # State outputs:
    state_bodies_q_out: wp.array(dtype=transformf),
    state_bodies_u_out: wp.array(dtype=vec6f),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrive the world index
    wid = model_bodies_wid_in[tid]

    # Retrieve the time step and gravity vector
    dt = model_dt_in[wid]
    gv = model_gravity_in[wid]
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Retrieve the model data
    inv_m_i = model_bodies_inv_m_in[tid]
    I_i = model_bodies_I_in[tid]
    inv_I_i = model_bodies_inv_I_in[tid]

    # Retrieve the current state of the body
    q_i = state_bodies_q_in[tid]
    u_i = state_bodies_u_in[tid]
    w_i = state_bodies_w_in[tid]

    # Compute the next pose and twist
    q_i_n, u_i_n = semi_implicit_euler_with_logmap(
        dt,
        g,
        inv_m_i,
        I_i,
        inv_I_i,
        q_i,
        u_i,
        w_i,
    )

    # Store the computed next pose and twist in the output arrays
    state_bodies_q_out[tid] = q_i_n
    state_bodies_u_out[tid] = u_i_n


###
# Launchers
###

def integrate_semi_implicit_euler(model: Model, state: ModelData, s_n: State):
    wp.launch(
        _integrate_semi_implicit_euler,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            model.time.dt,
            model.gravity.vector,
            model.bodies.wid,
            model.bodies.inv_m_i,
            state.bodies.I_i,
            state.bodies.inv_I_i,
            state.bodies.q_i,
            state.bodies.u_i,
            state.bodies.w_i,
            # Outputs:
            s_n.q_i,
            s_n.u_i,
        ]
    )
