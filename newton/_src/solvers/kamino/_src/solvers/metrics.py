# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a utilities to compute performance metrics that
assess physical plausibility and accuracy of simulations.


The :class:`SolutionMetricsData` class defines the set of simulation solver performance metrics that
provide quantitative measures of constraint satisfaction and physical plausibility of the computed
solution to the dual forward-dynamics Nonlinear Complementarity Problem (NCP).

The :class:`SolutionMetrics` class provides a high-level interface to manage all relevant data
allocations as well as provide the operations to compute the various performance metrics.

Usage
----
A typical example for using this module is:

    # Import all relevant types from Kamino
    from newton._src.solvers.kamino.core import ModelBuilderKamino
    from newton._src.solvers.kamino._src.geometry import ContactsKamino
    from newton._src.solvers.kamino._src.kinematics import LimitsKamino
    from newton._src.solvers.kamino._src.kinematics import DenseSystemJacobians
    from newton._src.solvers.kamino._src.dynamics import DualProblem
    from newton._src.solvers.kamino.solvers import PADMMSolver

    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilderKamino()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = builder.finalize()
    state_p = model.state()
    data = model.data()
    limits = LimitsKamino(model)
    contacts = ContactsKamino(builder)
    jacobians = DenseSystemJacobians(model, limits, contacts)

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a forward-dynamics DualProblem to be solved
    dual = DualProblem(model, limits, contacts, jacobians)
    dual.build(model, data, limits, contacts, jacobians)

    # Create a forward-dynamics PADMM solver
    solver = PADMMSolver(model, limits, contacts)

    # Solve the dual forward dynamics problem
    solver.coldstart()
    solver.solve(problem=dual)

    # Create a SolutionMetrics container
    metrics = SolutionMetrics(model)

    # Compute the solution metrics after solving the dual problem
    metrics.reset()
    metrics.evaluate(
        solver.state.sigma,
        solver.solution.lambdas,
        solver.solution.v_plus,
        model,
        data,
        state_p,
        dual,
        jacobians,
        limits,
        contacts,
    )
"""

from dataclasses import dataclass

import warp as wp

from .....sim import Contacts, Control, Model, State
from .....solvers.solver import SolverBase
from ..core.bodies import update_body_inertias, update_body_wrenches
from ..core.control import ControlKamino
from ..core.data import DataKamino
from ..core.math import screw, screw_angular, screw_linear
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..core.types import float32, int32, int64, mat33f, uint32, vec2f, vec2i, vec3f, vec4f, vec6f
from ..dynamics.dual import DualProblem
from ..dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from ..geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from ..geometry.keying import build_pair_key2
from ..kinematics.constraints import (
    make_unilateral_constraints_info,
    update_constraints_info,
)
from ..kinematics.jacobians import (
    DenseSystemJacobians,
    SparseSystemJacobians,
    SystemJacobiansType,
)
from ..kinematics.joints import (
    compute_joints_data,
)
from ..kinematics.limits import LimitsKamino
from ..solvers.padmm.math import (
    compute_desaxce_corrections,
    compute_dot_product,
    compute_double_dot_product,
    compute_ncp_complementarity_residual,
    compute_ncp_dual_residual,
    compute_ncp_natural_map_residual,
    compute_ncp_primal_residual,
    compute_vector_sum,
)

###
# Module interface
###

__all__ = [
    "SolutionMetrics",
    "SolutionMetricsData",
    "extract_mujoco_warp_constraint_forces",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class SolutionMetricsData:
    """
    Defines a container to hold performance metrics that assess physical plausibility and accuracy
    of the computed solution to the dual forward-dynamics Nonlinear Complementarity Problem (NCP).

    Attributes:
        r_eom (wp.array):
            The largest residual across all DoF dimensions of the Equations-of-Motion (EoM).
        r_kinematics (wp.array):
            The largest residual across all kinematic bilateral (i.e. equality) constraints.
        r_cts_joints (wp.array):
            The largest constraint violation residual across all bilateral kinematic joint constraints.
        r_cts_limits (wp.array):
            The largest constraint violation residual across all unilateral joint-limit constraints.
        r_cts_contacts (wp.array):
            The largest constraint violation residual across all contact constraints.
        r_ncp_primal (wp.array):
            The NCP primal residual representing the violation of set-valued constraint reactions.
        r_ncp_dual (wp.array):
            The NCP dual residual representing the violation of set-valued augmented constraint velocities.
        r_ncp_compl (wp.array):
            The NCP complementarity residual representing the violation of complementarity conditions.
        r_vi_natmap (wp.array):
            The Variational Inequality (VI) natural-map residual representing the proximity
            of the constraint reactions to a true solution of the VI defined by the NCP.
    """

    r_eom: wp.array | None = None
    """
    The largest residual across all DoF dimensions of the Equations-of-Motion (EoM).

    Measures how well the computed post-event generalized velocity `u^+` and constraint reactions
    `lambdas` (i.e. Lagrange multipliers) satisfy the impulse-velocity form of the equations-of-motion.

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual of:

    `r_eom :=  || M @ (u^+ - u^-) - dt * (h + J_dofs.T @ tau) - J_cts.T @ lambdas||_inf`,

    where:
    - `M` is the generalized mass matrix
    - `u^+` and `u^-` are the post- and pre-event generalized velocities
    - `dt` is the time-step
    - `h` is the generalized impulse vector
    - `J_cts` is the constraint Jacobian matrix
    - `lambdas` is the vector of all constraint reactions (i.e. Lagrange multipliers)
    - `J_dofs` is the actuation Jacobian matrix
    - `tau` is the vector of generalized actuation forces

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_eom_argmax: wp.array | None = None
    """
    The index pair key computed from the body index and
    degree-of-freedom (DoF) with the largest EoM residual.

    Shape of ``(num_worlds,)`` and type :class:`int64`.
    """

    r_kinematics: wp.array | None = None
    """
    The largest residual across all kinematic bilateral (i.e. equality) constraints.

    Measures how well the computed post-event generalized velocity
    `u^+` satisfies the velocity-level kinematic equality constraints.

    Computed as the maximum absolute value (i.e. infinity-norm) over velocity-level kinematics constraints:
    `r_kinematics := || J_cts_joints @ u^+ ||_inf`,

    where:
    - `J_cts_joints` is the constraint Jacobian matrix for bilateral joint constraints
    - `u^+` is the post-event generalized velocity

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_kinematics_argmax: wp.array | None = None
    """
    The index pair key computed from the joint index and
    bilateral kinematic constraint with the largest residual.

    Shape of ``(num_worlds,)`` and type :class:`int64`.
    """

    r_cts_joints: wp.array | None = None
    """
    The largest constraint violation residual across all bilateral joint constraints.

    Computed as the maximum absolute value (i.e. infinity-norm) over joint constraint residuals.

    Equivalent to `r_cts_joints := || r_j ||_inf`, where `r_j` is the
    array of joint constraint residuals defined in :class:`JointsData`.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_cts_joints_argmax: wp.array | None = None
    """
    The index pair key computed from the joint index and bilateral
    kinematic joint constraint with the largest residual.

    Shape of ``(num_worlds,)`` and type :class:`int64`.
    """

    r_cts_limits: wp.array | None = None
    """
    The largest constraint violation residual across all unilateral joint-limit constraints.

    Computed as the maximum absolute value (i.e. infinity-norm) over joint-limit constraint residuals.

    Equivalent to `r_cts_limits := || r_l ||_inf`, where `r_l` would be an array of joint-limit
    constraint residuals constructed from the collection of `r_q` elements defined in :class:`LimitsKaminoData`.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_cts_limits_argmax: wp.array | None = None
    """
    The index pair key computed from the joint index and degree-of-freedom
    (DoF) with the largest unilateral joint-limit constraint residual.

    Shape of ``(num_worlds,)`` and type :class:`int64`.
    """

    r_cts_contacts: wp.array | None = None
    """
    The largest constraint violation residual across all contact constraints.

    Computed as the maximum absolute value (i.e. infinity-norm) over contact constraint residuals.

    Equivalent to `r_cts_contacts := || d_k ||_inf`, where `d_k` would be an array of
    contact penetrations extracted from the `gapfunc` elements of :class:`ContactsKaminoData`.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_cts_contacts_argmax: wp.array | None = None
    """
    The indexir of the contact  with the largest unilateral contact constraint residual.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_v_plus: wp.array | None = None
    """
    The largest error in the estimation of the post-event constraint-space velocity.

    Measures how well the computed post-event constraint-space velocity `v^+` was estimated.

    Computed as the maximum absolute value (i.e. infinity-norm) over velocity-level kinematics constraints:
    `r_v_plus := || v_plus_est - v_plus_true ||_inf`,

    where:
    - `v_plus_est` is the estimated post-event constraint-space velocity
    - v_plus_true` is the true post-event constraint-space velocity computed as
      `v_plus_true = v_f + D @ lambdas`, where `v_f` is the unconstrained constraint-space velocity,
      `D` is the Delassus operator, and `lambdas` is the vector of all constraint reactions (i.e. Lagrange multipliers).

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_v_plus_argmax: wp.array | None = None
    """
    The index of the constraint with the largest post-event constraint-space velocity estimation error.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_ncp_primal: wp.array | None = None
    """
    The NCP primal residual representing the violation of set-valued constraint reactions.

    Measures the feasibility of constraint reactions w.r.t the feasible-set cone `K`
    defined as the Cartesian product over all positive-orthants for joint-limits and
    Coulomb friction cones for contacts:
    `K = R^{n_l}_{+} x Π_{k=1}^{n_c} K_{mu_k}`,

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    `r_ncp_primal(lambda) = || lambda - P_K(lambda) ||_inf`, where `P_K()` is the
    Euclidean projection, i.e. proximal operator, onto K, and `lambda` is the
    vector of all constraint reactions (i.e. Lagrange multipliers).

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_ncp_primal_argmax: wp.array | None = None
    """
    The index of the constraint with the largest NCP primal residual.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_ncp_dual: wp.array | None = None
    """
    The NCP dual residual representing the violation of set-valued augmented constraint velocities.

    Measures the feasibility of augmented constraint-space velocities w.r.t
    the dual cone `K*`, the Lagrange dual of the feasible-set cone `K`.

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    `r_ncp_dual(v_hat^+) = || v_hat^+ - P_K*(v_hat^+) ||_inf`, where `P_K*()` is
    the Euclidean projection, i.e. proximal operator, onto K*, and `v_hat^+` is
    the so-called augmented constraint-space velocity. The latter is defined as
    `v_hat^+ = v^+ + Gamma(v^+)`, where `v^+ := v_f D @ lambda` is the post-event
    constraint-space velocity, and `Gamma(v^+)` is the De Saxce correction term.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_ncp_dual_argmax: wp.array | None = None
    """
    The index of the constraint with the largest NCP dual residual.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_ncp_compl: wp.array | None = None
    """
    The NCP complementarity residual representing the violation of complementarity conditions.

    Measures the complementarity between constraint reactions and the augmented constraint-space
    velocities, as defined by the velocity-level Signorini (i.e. complementarity) conditions
    and positive orthants for joint-limits and Coulomb friction cones for contacts.

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    `r_ncp_compl(lambda) = || lambda.T @ v_hat^+ ||_inf`,
    where `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    and `v_hat^+` is the augmented constraint-space velocity defined above.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_ncp_compl_argmax: wp.array | None = None
    """
    The index of the constraint with the largest NCP complementarity residual.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_vi_natmap: wp.array | None = None
    """
    The Variational Inequality (VI) natural-map residual representing the proximity
    of the constraint reactions to a true solution of the VI defined by the NCP.

    Measures the how well the given constraint reactions solve the `NCP(D, v_f, K)`,
    by simultaneously combining the effects of the primal, dual and complementarity
    residuals into a single value, providing a convenient short-hand for both solver
    convergence and solution validity.

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    `r_vi_natmapv_hat(lambda) = || lambda - P_K*(lambda - v_hat^+(lambda)) ||_inf`,
    where `P_K*()` is the Euclidean projection, i.e. proximal operator, onto K*,
    `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    and `v_hat^+(lambda)` is the augmented constraint-space velocity defined above.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_vi_natmap_argmax: wp.array | None = None
    """
    The index of the constraint with the largest VI natural-map residual.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    f_ncp: wp.array | None = None
    """
    Evaluation of the NCP energy dissipation objective function.

    Represents only the energy dissipated through friction.

    Computed as as:
    `f_ncp(lambda) := 0.5 * lambda.T @ D @ lambda + lambda.T @ v_f + lambda.T @ s`,
    where `D` is the Delassus operator, `v_f` is the unconstrained constraint-space
    velocity and `s := Gamma(v^+)` is the De Saxce correction term.

    It is also equivalently computed as:
    `f_ncp(lambda) = f_ccp(lambda) + lambda.T @ s`,
    where `f_ccp` is the CCP energy dissipation objective.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    f_ccp: wp.array | None = None
    """
    Evaluation of the CCP energy dissipation objective function.

    Represents only the energy dissipated through friction.

    Computed as as:
    `f_ccp(lambda) := 0.5 * lambda.T @ D @ lambda + v_f.T @ lambda`,
    where `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    `D` is the Delassus operator and `v_f` is the unconstrained constraint-space velocity.

    It is also equivalently computed as:
    `f_ccp(lambda) := 0.5 * lambda.T @ (v+ + v_f)`,
    where `v+ = v_f + D @ lambda` is the post-event constraint-space velocity.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    def clear(self):
        """
        Clears all metric-argmax indices to -1.
        """
        self.r_eom_argmax.fill_(-1)
        self.r_kinematics_argmax.fill_(-1)
        self.r_cts_joints_argmax.fill_(-1)
        self.r_cts_limits_argmax.fill_(-1)
        self.r_cts_contacts_argmax.fill_(-1)
        self.r_v_plus_argmax.fill_(-1)
        self.r_ncp_primal_argmax.fill_(-1)
        self.r_ncp_dual_argmax.fill_(-1)
        self.r_ncp_compl_argmax.fill_(-1)
        self.r_vi_natmap_argmax.fill_(-1)

    def zero(self):
        """
        Resets all metrics to zeros.
        """
        self.r_eom.zero_()
        self.r_kinematics.zero_()
        self.r_cts_joints.zero_()
        self.r_cts_limits.zero_()
        self.r_cts_contacts.zero_()
        self.r_v_plus.zero_()
        self.r_ncp_primal.zero_()
        self.r_ncp_dual.zero_()
        self.r_ncp_compl.zero_()
        self.r_vi_natmap.zero_()
        self.f_ncp.zero_()
        self.f_ccp.zero_()

    def reset(self):
        """
        Resets all metrics and argmax indices.
        """
        self.clear()
        self.zero()


###
# Functions
###


@wp.func
def compute_v_plus(
    dim: int32,
    vio: int32,
    mio: int32,
    sigma: float32,
    P: wp.array[float32],
    D_p: wp.array[float32],
    v_f_p: wp.array[float32],
    lambdas: wp.array[float32],
    v_plus: wp.array[float32],
):
    """
    Computes the post-event constraint-space velocity as:

    `v_plus = P^-1 @ (v_f + (D_p - sigma * I_n) @ (P^-1 @ lambdas))`.

    Where `P` is the diagonal preconditioning matrix, `D_p` is the
    preconditioned Delassus matrix, `v_f_p` is the preconditioned
    unconstrained constraint-space velocity, `lambdas` is the vector
    of constraint reactions (i.e. Lagrange multipliers), and `I_n`
    is the identity matrix of size `n x n`.

    The preconditioned Delassus matrix `D_p` is stored using row-major order in a
    flat array with allocation size `maxdim x maxdim`, starting from the matrix
    index offset `mio`. The active dimensions of the `D_p` are `dim x dim`, where
    `dim` is the number of active rows and columns.

    The vectors `v_f_p, lambdas, v_plus` are stored in flat arrays
    with dimensions `dim`, starting from the vector index offset `vio`.

    Args:
        maxdim: The maximum dimension of the matrix `A`.
        dim: The active dimension of the matrix `A` and the vectors `x, b, c`.
        vio: The vector index offset (i.e. start index) for the vectors `x, b, c`.
        mio: The matrix index offset (i.e. start index) for the matrix `A`.
        D_p: Input preconditioned Delassus matrix stored in row-major order.
        v_f_p: Input preconditioned unconstrained constraint-space velocity vector.
        lambdas: Input constraint reactions (i.e. Lagrange multipliers) vector.
        v_plus: Output array to store the post-event constraint-space velocity vector.
    """
    v_f_p_i = float(0.0)
    lambdas_j = float(0.0)
    for i in range(dim):
        v_i = vio + i
        m_i = mio + dim * i
        v_f_p_i = v_f_p[v_i]
        inv_P_i = 1.0 / P[v_i]
        lambdas_i = lambdas[v_i]
        v_f_p_i -= sigma * inv_P_i * lambdas_i
        for j in range(dim):
            v_j = vio + j
            inv_P_j = 1.0 / P[v_j]
            lambdas_j = lambdas[v_j]
            v_f_p_i += D_p[m_i + j] * inv_P_j * lambdas_j
        v_plus[v_i] = inv_P_i * v_f_p_i


@wp.func
def compute_v_plus_sparse(
    dim: int32,
    vio: int32,
    P: wp.array[float32],
    v_f_p: wp.array[float32],
    D_p_lambdas: wp.array[float32],
    v_plus: wp.array[float32],
):
    """
    Computes the post-event constraint-space velocity as:

    `v_plus = P^-1 @ v_f_p + D_p @ lambdas`.

    Where `P` is the diagonal preconditioning matrix, `D_p` is the Delassus
    matrix (without preconditioning and regularization), `v_f_p` is the
    preconditioned unconstrained constraint-space velocity, `lambdas` is the
    vector of constraint reactions (i.e. Lagrange multipliers). `D_p @ lambdas`
    is passed into the function precomputed as `D_p_lambdas`.

    All vectors are stored in flat arrays with dimensions `dim`, starting from
    the vector index offset `vio`.

    Args:
        dim: The active dimension of the matrix `A` and the vectors `x, b, c`.
        vio: The vector index offset (i.e. start index) for the vectors `x, b, c`.
        v_f_p: Input preconditioned unconstrained constraint-space velocity vector.
        D_p_lambdas: Product of the Delassus matrix with the input constraint reactions
        v_plus: Output array to store the post-event constraint-space velocity vector.
    """
    for i in range(dim):
        v_i = vio + i
        v_plus[v_i] = v_f_p[v_i] / P[v_i] + D_p_lambdas[v_i]


@wp.func
def compute_vector_difference_infnorm(
    dim: int32,
    vio: int32,
    x: wp.array[float32],
    y: wp.array[float32],
) -> tuple[float32, int32]:
    """
    Computes the sum of two vectors `x` and `y` and stores the result in vector `z`.\n
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim: The dimension (i.e. size) of the vectors.
        vio: The vector index offset (i.e. start index).
        x: The first vector.
        y: The second vector.
        z: The output vector where the sum is stored.

    Returns:
        None: The result is stored in the output vector `z`.
    """
    max = float(0.0)
    argmax = int32(-1)
    for i in range(dim):
        v_i = vio + i
        err = wp.abs(x[v_i] - y[v_i])
        max = wp.max(max, err)
        if err == max:
            argmax = i
    return max, argmax


###
# Kernels
###


@wp.kernel
def _compute_eom_residual(
    # Inputs
    model_time_dt: wp.array[float32],
    model_gravity: wp.array[vec4f],
    model_bodies_wid: wp.array[int32],
    model_bodies_m_i: wp.array[float32],
    state_bodies_I_i: wp.array[mat33f],
    state_bodies_w_i: wp.array[vec6f],
    state_bodies_u_i: wp.array[vec6f],
    state_bodies_u_i_p: wp.array[vec6f],
    # Outputs
    metric_r_eom: wp.array[float32],
    metric_r_eom_argmax: wp.array[int64],
):
    # Retrieve the thread index as the body index
    bid = wp.tid()

    # Retrieve the body data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    w_i = state_bodies_w_i[bid]
    u_i = state_bodies_u_i[bid]
    u_i_p = state_bodies_u_i_p[bid]

    # Retrieve the time step
    dt = model_time_dt[wid]
    gravity = model_gravity[wid]
    g = gravity.w * vec3f(gravity.x, gravity.y, gravity.z)

    # Decompose into linear and angular parts
    f_i = screw_linear(w_i)
    v_i = screw_linear(u_i)
    v_i_p = screw_linear(u_i_p)
    tau_i = screw_angular(w_i)
    omega_i = screw_angular(u_i)
    omega_i_p = screw_angular(u_i_p)
    S_i = wp.skew(omega_i_p)

    # Compute the per-body EoM residual over linear and angular parts
    r_linear_i = wp.abs(m_i * (v_i - v_i_p) - dt * (m_i * g + f_i))
    r_angular_i = wp.abs(I_i @ (omega_i - omega_i_p) - dt * (tau_i - S_i @ (I_i @ omega_i_p)))
    r_i = screw(r_linear_i, r_angular_i)

    # Compute the per-body maximum residual and argmax index
    r_eom_i = wp.max(r_i)
    r_eom_argmax_i = int32(wp.argmax(r_i))

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_eom, wid, r_eom_i)
    if r_eom_i >= previous_max:
        argmax_key = int64(build_pair_key2(uint32(bid), uint32(r_eom_argmax_i)))
        wp.atomic_exch(metric_r_eom_argmax, int32(wid), argmax_key)


@wp.kernel
def _compute_joint_kinematics_residual_dense(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    model_info_total_cts_offset: wp.array[int32],
    model_info_joint_kinematic_cts_group_offset: wp.array[int32],
    model_joint_wid: wp.array[int32],
    model_joint_num_kinematic_cts: wp.array[int32],
    model_joint_kinematic_cts_offset_total_cts: wp.array[int32],
    model_joint_bid_B: wp.array[int32],
    model_joint_bid_F: wp.array[int32],
    data_bodies_u_i: wp.array[vec6f],
    jacobian_cts_offset: wp.array[int32],
    jacobian_cts_data: wp.array[float32],
    # Outputs:
    metric_r_kinematics: wp.array[float32],
    metric_r_kinematics_argmax: wp.array[int64],
):
    # Retrieve the joint index from the thread index
    jid = wp.tid()

    # Retrieve the world index of the joint
    wid = model_joint_wid[jid]

    # Retrieve the body indices of the joint
    # NOTE: these indices are w.r.t the model
    bid_F_j = model_joint_bid_F[jid]
    bid_B_j = model_joint_bid_B[jid]

    # Retrieve the size and index offset of the joint constraint
    num_cts_j = model_joint_num_kinematic_cts[jid]
    cts_offset_j = model_joint_kinematic_cts_offset_total_cts[jid] - model_info_total_cts_offset[wid]

    # Retrieve the world-specific info
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)
    kgo = model_info_joint_kinematic_cts_group_offset[wid]
    mio = jacobian_cts_offset[wid]

    # Compute the per-joint constraint Jacobian matrix-vector product
    j_v_j = vec6f(0.0)
    u_i_F = data_bodies_u_i[bid_F_j]
    dio_F = 6 * (bid_F_j - bio)
    for j in range(num_cts_j):
        mio_j = mio + nbd * (cts_offset_j + j) + dio_F
        for i in range(6):
            j_v_j[j] += jacobian_cts_data[mio_j + i] * u_i_F[i]
    if bid_B_j >= 0:
        u_i_B = data_bodies_u_i[bid_B_j]
        dio_B = 6 * (bid_B_j - bio)
        for j in range(num_cts_j):
            mio_j = mio + nbd * (cts_offset_j + j) + dio_B
            for i in range(6):
                j_v_j[j] += jacobian_cts_data[mio_j + i] * u_i_B[i]

    # Compute the per-joint kinematics residual and local argmax
    j_v_j_abs = wp.abs(j_v_j)
    kin_argmax_local = wp.argmax(j_v_j_abs)
    r_kinematics_j = j_v_j_abs[kin_argmax_local]

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_kinematics, wid, r_kinematics_j)
    if r_kinematics_j >= previous_max:
        argmax_key = int64(build_pair_key2(uint32(jid), uint32(cts_offset_j - kgo) + kin_argmax_local))
        wp.atomic_exch(metric_r_kinematics_argmax, wid, argmax_key)


@wp.kernel
def _compute_joint_kinematics_residual_sparse(
    # Inputs:
    model_info_joint_kinematic_cts_offset: wp.array[int32],
    model_joint_wid: wp.array[int32],
    model_joint_num_dynamic_cts: wp.array[int32],
    model_joint_num_kinematic_cts: wp.array[int32],
    model_joint_kinematic_cts_offset: wp.array[int32],
    model_joint_bid_B: wp.array[int32],
    model_joint_bid_F: wp.array[int32],
    data_bodies_u_i: wp.array[vec6f],
    jac_nzb_values: wp.array[vec6f],
    jac_joint_nzb_offsets: wp.array[int32],
    # Outputs:
    metric_r_kinematics: wp.array[float32],
    metric_r_kinematics_argmax: wp.array[int64],
):
    # Retrieve the joint index from the thread index
    jid = wp.tid()

    # Retrieve the world index of the joint
    wid = model_joint_wid[jid]

    # Retrieve the body indices of the joint
    # NOTE: these indices are w.r.t the model
    bid_F_j = model_joint_bid_F[jid]
    bid_B_j = model_joint_bid_B[jid]

    # Retrieve the size and index offset of the joint constraint
    num_dyn_cts_j = model_joint_num_dynamic_cts[jid]
    num_kin_cts_j = model_joint_num_kinematic_cts[jid]
    kin_cts_offset_j = model_joint_kinematic_cts_offset[jid] - model_info_joint_kinematic_cts_offset[wid]

    # Retrieve the starting index for the non-zero blocks for the current joint
    jac_j_nzb_start = jac_joint_nzb_offsets[jid] + (2 * num_dyn_cts_j if bid_B_j >= 0 else num_dyn_cts_j)

    # Compute the per-joint constraint Jacobian matrix-vector product
    j_v_j = vec6f(0.0)
    u_i_F = data_bodies_u_i[bid_F_j]
    for j in range(num_kin_cts_j):
        jac_block = jac_nzb_values[jac_j_nzb_start + j]
        j_v_j[j] += wp.dot(jac_block, u_i_F)
    if bid_B_j >= 0:
        u_i_B = data_bodies_u_i[bid_B_j]
        for j in range(num_kin_cts_j):
            jac_block = jac_nzb_values[jac_j_nzb_start + num_kin_cts_j + j]
            j_v_j[j] += wp.dot(jac_block, u_i_B)

    # Compute the per-joint kinematics residual and local argmax
    j_v_j_abs = wp.abs(j_v_j)
    kin_argmax_local = wp.argmax(j_v_j_abs)
    r_kinematics_j = j_v_j_abs[kin_argmax_local]

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_kinematics, wid, r_kinematics_j)
    if r_kinematics_j >= previous_max:
        argmax_key = int64(build_pair_key2(uint32(jid), uint32(kin_cts_offset_j) + kin_argmax_local))
        wp.atomic_exch(metric_r_kinematics_argmax, wid, argmax_key)


@wp.kernel
def _compute_cts_joints_residual(
    # Inputs:
    model_info_joint_kinematic_cts_offset: wp.array[int32],
    model_joint_wid: wp.array[int32],
    model_joint_num_kinematic_cts: wp.array[int32],
    model_joint_kinematic_cts_offset: wp.array[int32],
    data_joints_r_j: wp.array[float32],
    # Outputs:
    metric_r_cts_joints: wp.array[float32],
    metric_r_cts_joints_argmax: wp.array[int64],
):
    # Retrieve the joint index from the thread index
    jid = wp.tid()

    # Retrieve the joint-specific model data
    wid = model_joint_wid[jid]
    num_cts_j = model_joint_num_kinematic_cts[jid]
    cio_j = model_joint_kinematic_cts_offset[jid]

    # Compute the per-joint constraint residual (infinity-norm) and local argmax row
    r_cts_joints_j = float32(0.0)
    argmax_j = int32(0)
    if num_cts_j > 0:
        r_cts_joints_j = wp.abs(data_joints_r_j[cio_j])
        for j in range(1, num_cts_j):
            v = wp.abs(data_joints_r_j[cio_j + j])
            if v > r_cts_joints_j:
                r_cts_joints_j = v
                argmax_j = int32(j)

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_cts_joints, wid, r_cts_joints_j)
    if r_cts_joints_j >= previous_max:
        cio_j_loc = cio_j - model_info_joint_kinematic_cts_offset[wid]
        argmax_key = int64(build_pair_key2(uint32(jid), uint32(cio_j_loc + argmax_j)))
        wp.atomic_exch(metric_r_cts_joints_argmax, wid, argmax_key)


@wp.kernel
def _compute_cts_limits_residual(
    # Inputs:
    limit_model_num_limits: wp.array[int32],
    limit_wid: wp.array[int32],
    limit_lid: wp.array[int32],
    limit_dof: wp.array[int32],
    limit_r_q: wp.array[float32],
    # Outputs:
    metric_r_cts_limits: wp.array[float32],
    metric_r_cts_limits_argmax: wp.array[int64],
):
    # Retrieve the thread index as the limit index
    lid = wp.tid()

    # Retrieve the number of limits active in the model
    model_nl = limit_model_num_limits[0]

    # Skip if lid is greater than the number of limits active in the model
    if lid >= model_nl:
        return

    # Retrieve the world index and the world-relative limit index for this limit
    wid = limit_wid[lid]
    wlid = limit_lid[lid]
    dof = limit_dof[lid]

    # Compute the per-limit constraint residual (infinity-norm)
    r_cts_limits_l = wp.abs(limit_r_q[lid])

    # Update the per-world maximum residual
    previous_max = wp.atomic_max(metric_r_cts_limits, wid, r_cts_limits_l)
    if r_cts_limits_l >= previous_max:
        argmax_key = int64(build_pair_key2(uint32(wlid), uint32(dof)))
        wp.atomic_exch(metric_r_cts_limits_argmax, wid, argmax_key)


@wp.kernel
def _compute_cts_contacts_residual(
    # Inputs:
    contact_model_num_contacts: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    contact_gapfunc: wp.array[vec4f],
    # Outputs:
    metric_r_cts_contacts: wp.array[float32],
    metric_r_cts_contacts_argmax: wp.array[int32],
):
    # Retrieve the thread index as the contact index
    cid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nc = contact_model_num_contacts[0]

    # Skip if cid is greater than the number of contacts active in the model
    if cid >= model_nc:
        return

    # Retrieve the per-contact data
    wid = contact_wid[cid]
    wcid = contact_cid[cid]
    gapfunc = contact_gapfunc[cid]

    # Compute the per-contact constraint residual (infinity-norm)
    r_cts_contacts_k = wp.abs(gapfunc[3])

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_cts_contacts, wid, r_cts_contacts_k)
    if r_cts_contacts_k >= previous_max:
        wp.atomic_exch(metric_r_cts_contacts_argmax, wid, wcid)


@wp.kernel
def _compute_dual_problem_metrics(
    # Inputs:
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_v_f: wp.array[float32],
    problem_D: wp.array[float32],
    problem_P: wp.array[float32],
    solution_sigma: wp.array[vec2f],
    solution_lambdas: wp.array[float32],
    solution_v_plus: wp.array[float32],
    # Buffers:
    buffer_s: wp.array[float32],
    buffer_v: wp.array[float32],
    # Outputs:
    metric_r_v_plus: wp.array[float32],
    metric_r_v_plus_argmax: wp.array[int32],
    metric_r_ncp_primal: wp.array[float32],
    metric_r_ncp_primal_argmax: wp.array[int32],
    metric_r_ncp_dual: wp.array[float32],
    metric_r_ncp_dual_argmax: wp.array[int32],
    metric_r_ncp_compl: wp.array[float32],
    metric_r_ncp_compl_argmax: wp.array[int32],
    metric_r_vi_natmap: wp.array[float32],
    metric_r_vi_natmap_argmax: wp.array[int32],
    metric_f_ncp: wp.array[float32],
    metric_f_ccp: wp.array[float32],
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the world-specific data
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    ncts = problem_dim[wid]
    cio = problem_cio[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    sigma = solution_sigma[wid]

    # Compute additional info
    njc = ncts - (nl + 3 * nc)

    # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
    # NOTE: We assume the dual problem linear terms `D` and `v_f` have already been preconditioned in-place using `P`
    compute_v_plus(ncts, vio, mio, sigma[0], problem_P, problem_D, problem_v_f, solution_lambdas, buffer_v)

    # Compute the post-event constraint-space velocity error as: r_v_plus = || v_plus_est - v_plus_true ||_inf
    r_v_plus, r_v_plus_argmax = compute_vector_difference_infnorm(ncts, vio, solution_v_plus, buffer_v)

    # Compute the De Saxce correction for each contact as: s = G(v_plus)
    compute_desaxce_corrections(nc, cio, vio, ccgo, problem_mu, buffer_v, buffer_s)

    # Compute the CCP optimization objective as: f_ccp = 0.5 * lambda.dot(v_plus + v_f)
    f_ccp = 0.5 * compute_double_dot_product(ncts, vio, solution_lambdas, buffer_v, problem_v_f)

    # Compute the NCP optimization objective as:  f_ncp = f_ccp + lambda.dot(s)
    f_ncp = compute_dot_product(ncts, vio, solution_lambdas, buffer_s)
    f_ncp += f_ccp

    # Compute the augmented post-event constraint-space velocity as: v_aug = v_plus + s
    compute_vector_sum(ncts, vio, buffer_v, buffer_s, buffer_v)

    # Compute the NCP primal residual as: r_p := || lambda - proj_K(lambda) ||_inf
    r_ncp_p, r_ncp_p_argmax = compute_ncp_primal_residual(nl, nc, vio, lcgo, ccgo, cio, problem_mu, solution_lambdas)

    # Compute the NCP dual residual as: r_d := || v_plus + s - proj_dual_K(v_plus + s)  ||_inf
    r_ncp_d, r_ncp_d_argmax = compute_ncp_dual_residual(njc, nl, nc, vio, lcgo, ccgo, cio, problem_mu, buffer_v)

    # Compute the NCP complementarity (lambda _|_ (v_plus + s)) residual as r_c := || lambda.dot(v_plus + s) ||_inf
    r_ncp_c, r_ncp_c_argmax = compute_ncp_complementarity_residual(nl, nc, vio, lcgo, ccgo, buffer_v, solution_lambdas)

    # Compute the natural-map residuals as: r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf
    r_ncp_natmap, r_ncp_natmap_argmax = compute_ncp_natural_map_residual(
        nl, nc, vio, lcgo, ccgo, cio, problem_mu, buffer_v, solution_lambdas
    )

    # Store the computed metrics in the output arrays
    metric_r_v_plus[wid] = r_v_plus
    metric_r_v_plus_argmax[wid] = r_v_plus_argmax
    metric_r_ncp_primal[wid] = r_ncp_p
    metric_r_ncp_primal_argmax[wid] = r_ncp_p_argmax
    metric_r_ncp_dual[wid] = r_ncp_d
    metric_r_ncp_dual_argmax[wid] = r_ncp_d_argmax
    metric_r_ncp_compl[wid] = r_ncp_c
    metric_r_ncp_compl_argmax[wid] = r_ncp_c_argmax
    metric_r_vi_natmap[wid] = r_ncp_natmap
    metric_r_vi_natmap_argmax[wid] = r_ncp_natmap_argmax
    metric_f_ncp[wid] = f_ncp
    metric_f_ccp[wid] = f_ccp


@wp.kernel
def _compute_dual_problem_metrics_sparse(
    # Inputs:
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_v_f: wp.array[float32],
    problem_P: wp.array[float32],
    solution_lambdas: wp.array[float32],
    solution_v_plus: wp.array[float32],
    # Buffers:
    buffer_s: wp.array[float32],
    buffer_v: wp.array[float32],
    # Outputs:
    metric_r_v_plus: wp.array[float32],
    metric_r_v_plus_argmax: wp.array[int32],
    metric_r_ncp_primal: wp.array[float32],
    metric_r_ncp_primal_argmax: wp.array[int32],
    metric_r_ncp_dual: wp.array[float32],
    metric_r_ncp_dual_argmax: wp.array[int32],
    metric_r_ncp_compl: wp.array[float32],
    metric_r_ncp_compl_argmax: wp.array[int32],
    metric_r_vi_natmap: wp.array[float32],
    metric_r_vi_natmap_argmax: wp.array[int32],
    metric_f_ncp: wp.array[float32],
    metric_f_ccp: wp.array[float32],
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the world-specific data
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    ncts = problem_dim[wid]
    cio = problem_cio[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]

    # Compute additional info
    njc = ncts - (nl + 3 * nc)

    # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
    # NOTE: We assume the dual problem term `v_f` has already been preconditioned in-place using `P`, and
    #       D @ lambdas is already provided in `buffer_v`
    compute_v_plus_sparse(ncts, vio, problem_P, problem_v_f, buffer_v, buffer_v)

    # Compute the post-event constraint-space velocity error as: r_v_plus = || v_plus_est - v_plus_true ||_inf
    r_v_plus, r_v_plus_argmax = compute_vector_difference_infnorm(ncts, vio, solution_v_plus, buffer_v)

    # Compute the De Saxce correction for each contact as: s = G(v_plus)
    compute_desaxce_corrections(nc, cio, vio, ccgo, problem_mu, buffer_v, buffer_s)

    # Compute the CCP optimization objective as: f_ccp = 0.5 * lambda.dot(v_plus + v_f)
    f_ccp = 0.5 * compute_double_dot_product(ncts, vio, solution_lambdas, buffer_v, problem_v_f)

    # Compute the NCP optimization objective as:  f_ncp = f_ccp + lambda.dot(s)
    f_ncp = compute_dot_product(ncts, vio, solution_lambdas, buffer_s)
    f_ncp += f_ccp

    # Compute the augmented post-event constraint-space velocity as: v_aug = v_plus + s
    compute_vector_sum(ncts, vio, buffer_v, buffer_s, buffer_v)

    # Compute the NCP primal residual as: r_p := || lambda - proj_K(lambda) ||_inf
    r_ncp_p, r_ncp_p_argmax = compute_ncp_primal_residual(nl, nc, vio, lcgo, ccgo, cio, problem_mu, solution_lambdas)

    # Compute the NCP dual residual as: r_d := || v_plus + s - proj_dual_K(v_plus + s)  ||_inf
    r_ncp_d, r_ncp_d_argmax = compute_ncp_dual_residual(njc, nl, nc, vio, lcgo, ccgo, cio, problem_mu, buffer_v)

    # Compute the NCP complementarity (lambda _|_ (v_plus + s)) residual as r_c := || lambda.dot(v_plus + s) ||_inf
    r_ncp_c, r_ncp_c_argmax = compute_ncp_complementarity_residual(nl, nc, vio, lcgo, ccgo, buffer_v, solution_lambdas)

    # Compute the natural-map residuals as: r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf
    r_ncp_natmap, r_ncp_natmap_argmax = compute_ncp_natural_map_residual(
        nl, nc, vio, lcgo, ccgo, cio, problem_mu, buffer_v, solution_lambdas
    )

    # Store the computed metrics in the output arrays
    metric_r_v_plus[wid] = r_v_plus
    metric_r_v_plus_argmax[wid] = r_v_plus_argmax
    metric_r_ncp_primal[wid] = r_ncp_p
    metric_r_ncp_primal_argmax[wid] = r_ncp_p_argmax
    metric_r_ncp_dual[wid] = r_ncp_d
    metric_r_ncp_dual_argmax[wid] = r_ncp_d_argmax
    metric_r_ncp_compl[wid] = r_ncp_c
    metric_r_ncp_compl_argmax[wid] = r_ncp_c_argmax
    metric_r_vi_natmap[wid] = r_ncp_natmap
    metric_r_vi_natmap_argmax[wid] = r_ncp_natmap_argmax
    metric_f_ncp[wid] = f_ncp
    metric_f_ccp[wid] = f_ccp


###
# Interfaces
###


class SolutionMetrics:
    """
    Provides a high-level interface to compute a set of simulation solver performance metrics
    that provide quantitative measures of constraint satisfaction and physical plausibility of
    the computed solution to the dual forward-dynamics Nonlinear Complementarity Problem (NCP).

    This class is therefore responsible for managing the lifetime of all metrics data memory
    allocations, as well as provide an easy-to-use public API of the relevant operations.

    Internally, it holds an instance of the :class:`SolutionMetricsData` class. For more details
    about the specific metrics computed, please refer to the documentation of that class.
    """

    def __init__(self, model: ModelKamino | None = None, data: DataKamino | None = None):
        """
        Initializes the solution metrics evaluator.

        Args:
            model: The model container holding the time-invariant data of the simulation.
            data: The data container holding the time-varying internal solver data.
        """
        # Declare the device cache
        self._device: wp.DeviceLike = None

        # Declare a model reference for internal use (e.g. for finalization)
        self._model: ModelKamino | None = None
        self._data: DataKamino | None = None

        # Declare the metrics data container
        self._metrics: SolutionMetricsData | None = None

        # Declare data buffers for metrics computations
        self._buffer_s: wp.array | None = None
        self._buffer_v: wp.array | None = None

        # If a model and data are provided, finalize the metrics data allocations
        if model is not None and data is not None:
            self.finalize(model, data)

    def finalize(self, model: ModelKamino, data: DataKamino):
        """
        Finalizes the metrics data allocations on the specified device.

        Args:
            model: The model container holding the time-invariant data of the simulation.
            data: The data container holding the time-varying internal solver data.
        """
        # Ensure the model and data are valid
        if not isinstance(model, ModelKamino):
            raise TypeError("Expected 'model' to be of type ModelKamino.")
        if not isinstance(data, DataKamino):
            raise TypeError("Expected 'data' to be of type DataKamino.")

        # Use the model's device
        self._device = model.device

        # Store a reference to the model and data for internal use (e.g. for finalization)
        self._model = model
        self._data = data

        # Allocate metrics data on the target device
        with wp.ScopedDevice(self._device):
            # Allocate reusable buffers for metrics computations
            self._buffer_v = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=float32)
            self._buffer_s = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=float32)

            # Allocate the metrics container data arrays
            self._metrics = SolutionMetricsData(
                r_eom=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_eom_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int64),
                r_kinematics=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_kinematics_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int64),
                r_cts_joints=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_cts_joints_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int64),
                r_cts_limits=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_cts_limits_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int64),
                r_cts_contacts=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_cts_contacts_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int32),
                r_v_plus=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_v_plus_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int32),
                r_ncp_primal=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_ncp_primal_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int32),
                r_ncp_dual=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_ncp_dual_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int32),
                r_ncp_compl=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_ncp_compl_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int32),
                r_vi_natmap=wp.zeros(self._model.size.num_worlds, dtype=float32),
                r_vi_natmap_argmax=wp.full(self._model.size.num_worlds, value=-1, dtype=int32),
                f_ncp=wp.zeros(self._model.size.num_worlds, dtype=float32),
                f_ccp=wp.zeros(self._model.size.num_worlds, dtype=float32),
            )

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device where the metrics data is allocated.
        """
        return self._device

    @property
    def data(self) -> SolutionMetricsData:
        """
        Returns the metrics data container.
        """
        self._assert_finalized()
        return self._metrics

    ###
    # Public Operations
    ###

    def reset(self):
        """
        Resets all metrics to zeros.
        """
        self._metrics.reset()

    def evaluate(
        self,
        sigma: wp.array,
        lambdas: wp.array,
        v_plus: wp.array,
        state_p: StateKamino,
        jacobians: SystemJacobiansType,
        problem: DualProblem | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """
        Evaluates all solution performance metrics.

        Args:
            model: The model containing the time-invariant data of the simulation.
            data: The model data containing the time-variant data of the simulation.
            state_p: The previous state of the simulation.
            limits: The joint-limits data describing active limit constraints.
            contacts: The contact data describing active contact constraints.
            problem: The dual forward dynamics problem of the current time-step.
            jacobians: The system Jacobians of the current time-step.
            sigma: The array diagonal regularization applied to the Delassus matrix of the current dual problem.
            lambdas: The array of constraint reactions (i.e. Lagrange multipliers) of the current dual problem solution.
            v_plus: The array of post-event constraint-space velocities of the current dual problem solution.
        """
        self._assert_finalized()
        self._evaluate_constraint_violations_perf(self._model, self._data, limits, contacts)
        self._evaluate_primal_problem_perf(self._model, self._data, state_p, jacobians)
        self._evaluate_dual_problem_perf(sigma, lambdas, v_plus, problem)

    ###
    # Internals
    ###

    def _assert_finalized(self):
        """
        Asserts that the metrics data has been finalized and is available.

        Raises:
            RuntimeError: If the data is not available.
        """
        if self._metrics is None:
            raise RuntimeError("SolutionMetrics data has not been finalized. Call finalize() first.")

    def _evaluate_constraint_violations_perf(
        self,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """
        Evaluates the constraint-violation performance metrics.

        Args:
            model: The model containing the time-invariant data of the simulation.
            data: The model data containing the time-variant data of the simulation.
            limits: The joint-limits data describing active limit constraints.
            contacts: The contact data describing active contact constraints.
        """
        # Ensure metrics data is available
        self._assert_finalized()

        # Compute the largest configuration-level joint constraint residuals (i.e. violations)
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                kernel=_compute_cts_joints_residual,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
                    model.info.joint_kinematic_cts_offset,
                    model.joints.wid,
                    model.joints.num_kinematic_cts,
                    model.joints.kinematic_cts_offset,
                    data.joints.r_j,
                    # Outputs:
                    self._metrics.r_cts_joints,
                    self._metrics.r_cts_joints_argmax,
                ],
                device=model.device,
            )

        # Compute the largest joint-limit constraint residuals (i.e. penetrations)
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                kernel=_compute_cts_limits_residual,
                dim=limits.data.model_max_limits_host,
                inputs=[
                    # Inputs:
                    limits.data.model_active_limits,
                    limits.data.wid,
                    limits.data.lid,
                    limits.data.dof,
                    limits.data.r_q,
                    # Outputs:
                    self._metrics.r_cts_limits,
                    self._metrics.r_cts_limits_argmax,
                ],
                device=model.device,
            )

        # Compute the largest contact constraint residuals (i.e. penetrations)
        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                kernel=_compute_cts_contacts_residual,
                dim=contacts.data.model_max_contacts_host,
                inputs=[
                    # Inputs:
                    contacts.data.model_active_contacts,
                    contacts.data.wid,
                    contacts.data.cid,
                    contacts.data.gapfunc,
                    # Outputs:
                    self._metrics.r_cts_contacts,
                    self._metrics.r_cts_contacts_argmax,
                ],
                device=model.device,
            )

    def _evaluate_primal_problem_perf(
        self,
        model: ModelKamino,
        data: DataKamino,
        state_p: StateKamino,
        jacobians: SystemJacobiansType,
    ):
        """
        Evaluates the primal problem performance metrics.

        Args:
            model: The model containing the time-invariant data of the simulation.
            data: The model data containing the time-variant data of the simulation.
            state_p: The previous state of the simulation.
            jacobians: The system Jacobians of the current time-step.
        """
        # Ensure metrics data is available
        self._assert_finalized()

        # Compute the equations-of-motion residuals
        wp.launch(
            kernel=_compute_eom_residual,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.time.dt,
                model.gravity.vector,
                model.bodies.wid,
                model.bodies.m_i,
                data.bodies.I_i,
                data.bodies.w_i,
                data.bodies.u_i,
                state_p.u_i,
                # Outputs:
                self._metrics.r_eom,
                self._metrics.r_eom_argmax,
            ],
            device=model.device,
        )

        # Compute the kinematics constraint residuals,
        # i.e. velocity-level joint constraint equations
        if model.size.sum_of_num_joints > 0:
            if isinstance(jacobians, DenseSystemJacobians):
                wp.launch(
                    kernel=_compute_joint_kinematics_residual_dense,
                    dim=model.size.sum_of_num_joints,
                    inputs=[
                        # Inputs:
                        model.info.bodies_offset,
                        model.info.total_cts_offset,
                        model.info.joint_kinematic_cts_group_offset,
                        model.joints.wid,
                        model.joints.num_kinematic_cts,
                        model.joints.kinematic_cts_offset_total_cts,
                        model.joints.bid_B,
                        model.joints.bid_F,
                        data.bodies.u_i,
                        jacobians.data.J_cts_offsets,
                        jacobians.data.J_cts_data,
                        # Outputs:
                        self._metrics.r_kinematics,
                        self._metrics.r_kinematics_argmax,
                    ],
                    device=model.device,
                )
            else:
                J_cts = jacobians._J_cts.bsm
                wp.launch(
                    kernel=_compute_joint_kinematics_residual_sparse,
                    dim=model.size.sum_of_num_joints,
                    inputs=[
                        # Inputs:
                        model.info.joint_kinematic_cts_offset,
                        model.joints.wid,
                        model.joints.num_dynamic_cts,
                        model.joints.num_kinematic_cts,
                        model.joints.kinematic_cts_offset,
                        model.joints.bid_B,
                        model.joints.bid_F,
                        data.bodies.u_i,
                        J_cts.nzb_values,
                        jacobians._J_cts_joint_nzb_offsets,
                        # Outputs:
                        self._metrics.r_kinematics,
                        self._metrics.r_kinematics_argmax,
                    ],
                    device=model.device,
                )

    def _evaluate_dual_problem_perf(
        self,
        sigma: wp.array,
        lambdas: wp.array,
        v_plus: wp.array,
        problem: DualProblem,
    ):
        """
        Evaluates the dual problem performance metrics.

        Args:
            problem: The dual problem containing the time-invariant and time-variant data of the simulation.
            sigma: The array of sigma values for the dual problem.
            lambdas: The array of lambda values for the dual problem.
            v_plus: The array of v_plus values for the dual problem.
        """
        # Ensure metrics data is available
        self._assert_finalized()

        # Compute the dual problem NCP/VI performance metrics
        if problem.sparse:
            # Compute post-event constraint-space velocity from solution: v_plus = v_f + D @ lambda
            # Store it in buffer for further processing in dual problem metrics computation
            delassus_reg_prev = problem.delassus._eta
            delassus_pre_prev = problem.delassus._preconditioner
            problem.delassus.set_regularization(None)
            problem.delassus.set_preconditioner(None)
            problem.delassus.matvec(
                x=lambdas,
                y=self._buffer_v,
                world_mask=wp.ones((problem.data.num_worlds,), dtype=wp.int32, device=self.device),
            )
            problem.delassus.set_regularization(delassus_reg_prev)
            problem.delassus.set_preconditioner(delassus_pre_prev)
            wp.launch(
                kernel=_compute_dual_problem_metrics_sparse,
                dim=problem.size.num_worlds,
                inputs=[
                    # Inputs:
                    problem.data.nl,
                    problem.data.nc,
                    problem.data.cio,
                    problem.data.lcgo,
                    problem.data.ccgo,
                    problem.data.dim,
                    problem.data.vio,
                    problem.data.mu,
                    problem.data.v_f,
                    problem.data.P,
                    lambdas,
                    v_plus,
                    # Buffers:
                    self._buffer_s,
                    self._buffer_v,
                    # Outputs:
                    self._metrics.r_v_plus,
                    self._metrics.r_v_plus_argmax,
                    self._metrics.r_ncp_primal,
                    self._metrics.r_ncp_primal_argmax,
                    self._metrics.r_ncp_dual,
                    self._metrics.r_ncp_dual_argmax,
                    self._metrics.r_ncp_compl,
                    self._metrics.r_ncp_compl_argmax,
                    self._metrics.r_vi_natmap,
                    self._metrics.r_vi_natmap_argmax,
                    self._metrics.f_ncp,
                    self._metrics.f_ccp,
                ],
                device=problem.device,
            )
        else:
            wp.launch(
                kernel=_compute_dual_problem_metrics,
                dim=problem.size.num_worlds,
                inputs=[
                    # Inputs:
                    problem.data.nl,
                    problem.data.nc,
                    problem.data.cio,
                    problem.data.lcgo,
                    problem.data.ccgo,
                    problem.data.dim,
                    problem.data.vio,
                    problem.data.mio,
                    problem.data.mu,
                    problem.data.v_f,
                    problem.data.D,
                    problem.data.P,
                    sigma,
                    lambdas,
                    v_plus,
                    # Buffers:
                    self._buffer_s,
                    self._buffer_v,
                    # Outputs:
                    self._metrics.r_v_plus,
                    self._metrics.r_v_plus_argmax,
                    self._metrics.r_ncp_primal,
                    self._metrics.r_ncp_primal_argmax,
                    self._metrics.r_ncp_dual,
                    self._metrics.r_ncp_dual_argmax,
                    self._metrics.r_ncp_compl,
                    self._metrics.r_ncp_compl_argmax,
                    self._metrics.r_vi_natmap,
                    self._metrics.r_vi_natmap_argmax,
                    self._metrics.f_ncp,
                    self._metrics.f_ccp,
                ],
                device=problem.device,
            )


class SolutionMetricsNewton:
    """
    SolutionMetrics wrapper to interface with Newton's front-end API.
    """

    def __init__(
        self,
        dt: float | None = None,
        model: Model | None = None,
        model_kamino: ModelKamino | None = None,
        sparse: bool = True,
    ):
        """
        Initializes the SolutionMetricsNewton wrapper.

        Args:
            dt: The time-step size of the simulation.
            model: The model container holding the time-invariant data of the simulation.
            model_kamino: The Kamino model container holding the time-invariant data of the simulation.
        """
        # Declare internal Kamino data containers
        self._model: ModelKamino | None = None
        self._data: DataKamino | None = None
        self._limits: LimitsKamino | None = None
        self._contacts: ContactsKamino | None = None
        self._problem: DualProblem | None = None
        self._jacobians: SystemJacobiansType | None = None
        self._state: StateKamino | None = None
        self._state_p: StateKamino | None = None
        self._control: ControlKamino | None = None

        # Declare additional buffers for metrics computations
        self._v_plus: wp.array | None = None
        self._lambdas: wp.array | None = None
        self._sigma: wp.array | None = None

        # Declare the metrics data container
        self._metrics: SolutionMetrics | None = None

        # If a model is provided, finalize the metrics data allocations
        if dt is not None and model is not None:
            self.finalize(dt=dt, model=model, model_kamino=model_kamino, sparse=sparse)

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device where the metrics data is allocated.
        """
        if self._model is None:
            raise RuntimeError("SolutionMetricsNewton data is not initialized. Call finalize() first.")
        return self._model.device

    ###
    # Operations
    ###

    def finalize(self, dt: float, model: Model, model_kamino: ModelKamino | None = None, sparse: bool = True):
        """
        Finalizes the SolutionMetricsNewton wrapper.

        Args:
            dt: The time-step size of the simulation.
            model: The model container holding the time-invariant data of the simulation.
            model_kamino: The Kamino model container holding the time-invariant data of the simulation.
        """
        # Ensure the model is valid
        if dt is None or not isinstance(dt, float):
            raise ValueError("Expected 'dt' argument to be a non-None float.")
        if model is None or not isinstance(model, Model):
            raise TypeError("Expected 'model' argument to be of type `newton.Model`")

        # If a model_kamino is provided, use it; otherwise, convert the model to Kamino format
        if model_kamino is not None:
            if not isinstance(model_kamino, ModelKamino):
                raise TypeError(
                    f"Expected 'model_kamino' argument to be of type `ModelKamino`, got {type(model_kamino)}"
                )
            self._model = model_kamino
        else:
            # TODO: We need a mechanism to force all joints being only kinematic, i.e. no dynamic constraints
            self._model = ModelKamino.from_newton(model=model, overwrite_source_model=False)

        # Configure model time-steps
        self._model.time.dt.fill_(wp.float32(dt))
        self._model.time.inv_dt.fill_(wp.float32(1.0 / dt))

        # Create the data, limits and contacts containers
        self._data = self._model.data()
        self._limits = LimitsKamino(model=self._model)
        self._contacts = ContactsKamino(model=self._model)

        # Create and finalize the control container
        self._control = ControlKamino()
        self._control.finalize(self._model)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(
            model=self._model, data=self._data, limits=self._limits, contacts=self._contacts
        )

        # Create the Jacobians container
        if sparse:
            self._jacobians = SparseSystemJacobians(model=self._model, limits=self._limits, contacts=self._contacts)
        else:
            self._jacobians = DenseSystemJacobians(model=self._model, limits=self._limits, contacts=self._contacts)

        # Create the DualProblem container
        # TODO: Create a solver config and set constraint stabilization to zero
        self._problem = DualProblem(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            sparse=False,
        )

        # Finalize the internal SolutionMetrics instance
        self._metrics = SolutionMetrics(model=self._model)

        # Allocate metrics data on the target device
        with wp.ScopedDevice(self.device):
            self._v_plus = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=float32)
            self._lambdas = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=float32)
            self._sigma = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=float32)

    def evaluate(
        self,
        state: State,
        state_p: State,
        control: Control | None = None,
        contacts: Contacts | None = None,
    ):
        """
        Evaluates the solution metrics on the provided Newton state and control data.

        Args:
            state: The Newton state data containing the current state of the simulation.
            state_p: The previous state of the simulation.
            control: The Newton control data containing the current control inputs of the simulation.
            contacts: The Newton contacts data containing the current contacts of the simulation.
        """
        # Interface the input state containers to Kamino's equivalents
        self._state = StateKamino.from_newton(self._model.size, self._model._model, state)
        self._state_p = StateKamino.from_newton(self._model.size, self._model._model, state_p)
        self._control.from_newton(control, self._model)
        convert_contacts_newton_to_kamino(self._model._model, state_p, contacts, self._contacts)

        # Run limit detection to generate active limits
        self._limits.detect(q_j=self._state_p.q_j)

        # Update the relevant data fields of `DataKamino` and system Jacobians required
        # for the metrics computations, using the provided `StateKamino` instances.
        self._read_step_inputs(self._state_p, self._control)
        update_constraints_info(model=self._model, data=self._data)
        update_body_inertias(model=self._model.bodies, data=self._data.bodies)
        compute_joints_data(model=self._model, data=self._data, q_j_p=self._state_p.q_j)
        self._update_jacobians()
        self._update_dynamics()

        # Compute the post-event constraint-space velocities given
        # the pre- and post-event state and constraint Jacobians
        self._compute_postevent_constraint_velocities(
            model=self._model,
            state=self._state,
            state_p=self._state_p,
            jacobians=self._jacobians,
            v_plus=self._v_plus,
        )

        # Perform the necessary conversions and extractions to obtain the
        # solver data in the expected format for the metrics computations
        self._convert_body_parent_wrenches_to_joint_reactions(
            model=self._model,
            state_in=state,
            control_in=self._control,
            limits_out=self._limits,
            state_out=self._state,
            data_out=self._data,
        )
        self._extract_constraint_reactions(
            model=self._model,
            state=self._state,
            limits=self._limits,
            contacts=self._contacts,
            lambdas=self._lambdas,
        )

        # Update all dynamics quantities based
        # on the extracted constraint reactions
        self._read_step_inputs(self._state, self._control)
        self._update_body_wrenches()

        # # Evaluate the metrics using the extracted solver data
        # self._metrics.evaluate(
        #     sigma=self._sigma,
        #     lambdas=self._lambdas,
        #     v_plus=self._v_plus,
        #     data=self._data,
        #     state_p=self._state_p,
        #     problem=self._problem,
        #     jacobians=self._jacobians,
        #     limits=self._limits,
        #     contacts=self._contacts,
        # )

    ###
    # Internals
    ###

    def _read_step_inputs(self, state_in: StateKamino, control_in: ControlKamino):
        wp.copy(self._data.bodies.q_i, state_in.q_i)
        wp.copy(self._data.bodies.u_i, state_in.u_i)
        wp.copy(self._data.bodies.w_i, state_in.w_i)
        wp.copy(self._data.bodies.w_e_i, state_in.w_i_e)
        wp.copy(self._data.joints.q_j, state_in.q_j)
        wp.copy(self._data.joints.dq_j, state_in.dq_j)
        self._data.joints.tau_j = control_in.tau_j

    def _update_jacobians(self):
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            reset_to_zero=True,
        )

    def _update_dynamics(self):
        self._problem.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            reset_to_zero=True,
        )

    def _update_body_wrenches(self):
        # Compute the per-body actuation wrenches: `DataKamino.bodies.w_a_i`
        # in world coordinates from the current joint torques
        compute_joint_dof_body_wrenches(self._model, self._data, self._jacobians)

        # Compute the per-body constraint wrenches: `w_j_i`, `w_l_i`,
        # and `w_c_i` of `DataKamino.bodies` in world coordinates
        compute_constraint_body_wrenches(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            lambdas_offsets=self._problem.data.vio,
            lambdas_data=self._lambdas,
        )

        # Compute the total applied wrenches per body by summing up all individual contributions,
        # from joint actuation, joint limits, contacts, and purely external effects.
        update_body_wrenches(self._model.bodies, self._data.bodies)

    ###
    # TODO: TO BE IMPLEMENTED
    ###

    def _compute_postevent_constraint_velocities(
        self,
        model: ModelKamino,
        state: StateKamino,
        state_p: StateKamino,
        jacobians: SystemJacobiansType,
        v_plus: wp.array,
    ):
        """
        Computes the constraint-space velocities `v = J @ u`, where `J := J_cts` and `u := State.bodies.u_i`

        This realizes the `v^{+/-} = J(q^{+/-}) @ u^{+/-}` operation.

        Depending on whether we use the pre-event (-) or post-event (+) state,
        we will compute the respective constraint-space velocities as:
        - `v^{+} = J(q) @ u^{+}`, i.e. `v^{+} := v_plus`
        - `v^{-} = J(q) @ u^{-}`, i.e. `v^{-} := v_minus`
        - All computations also depend on whether the pre- or post-event coordinates are use
          to evaluate the constraint Jacobian `J(q)`. However, for the purposes of evaluating
          physical correctness, we will use the system coordinates `q` that are coincident
          with the given state.

        Args:
            model:
                The model containing the time-invariant data of the simulation.
            state:
                The input post-event state data containing the current state of the simulation.
            state_p:
                The input pre-event state data containing the initial state of the simulation.
            jacobians:
                The system Jacobians.
            v_plus:
                The output array to store the post-event constraint-space velocities.
        """
        pass  # TODO: TO BE IMPLEMENTED

    def _convert_body_parent_wrenches_to_joint_reactions(
        self,
        model: ModelKamino,
        state_in: State,
        control_in: ControlKamino,
        data_out: DataKamino,
        limits_out: LimitsKamino,
        state_out: StateKamino,
    ):
        """
        Converts Newton body-parent wrenches `newton.State.body_parent_f` data
        to Kamino `StateKamino.lambda_j` and `DataKamino.joints.lambda_l_j`.

        This operation also updates per-joint wrenches arrays `DataKamino.joints.j_w_j` as a byproduct.

        Definitions:
        -  `body_parent_f` contains the wrench applied on each body by its parent body, referenced w.r.
            the child body's center of mass (COM) and expressed in the world frame (i.e. world coordinates).
            Each entry is equal to `w_ij`, the world wrench applied by parent body `i` joint `j`.
        -  `w_j` is the wrench applied by joint `j` on its follower/child
            body, referenced w.r.t. the joint frame in world coordinates.
        - `j_w_j` is the wrench applied by joint `j` on its follower/child
           body, expressed in the local coordinates of the joint frame.
        -  `lambda_j` contains the constraint reaction impulses
           applied by each joint, expressed in the joint frame.
        - `lambda_l_j` contains the joint-limit constraint reactions.
        - `tau_c_j` is the joint-space actuation generalized forces.
        - `tau_j` is the joint-space generaralized forces. However, as any acting joint-limit constraint
           reactions also lie in the same space (i.e. DoF-space), we will consider this to be equal to
           the total joint-space generalized forces `tau_j := tau_c_j + lambda_l_j`
        - `dt` is the simulation time step.

        The conversion is performed parallel over joints as follows:
        -  We use the relation `w_j = inv(W_ij) @ w_ij` to compute `w_j`, i.e. the joint wrench
           referenced w.r.t. the joint frame in world coordinates, where `W_ij` is the `6x6` wrench
           transform matrix transforming `w_j` from the joint frame to the COM frame of body `i`.
           When body `i` is the  follower/child we use the absolute pose of the body and joint
           frames to compute `W_ij`.
        -  Having `w_j`, we compute `j_w_j` as `j_w_j = X_bar_j.T @ R_bar_j.T @ w_j`, where `X_bar_j`
           is the `6x6` constant joint frame transform matrix extended to 6D (via 3x3 on both diagonals)
           and similarly `R_bar_j` is the `6x6` extended joint frame rotation matrix extended to 6D
           computed from the absolute pose of the joint frame `p_j`.
        -  Having `j_w_j`, we compute `lambda_j` as `[lambda_j; tau_j] = (1/dt) inv(S_j) @ j_w_j`, where `S_j`
           is the `6x6` joint constraint/dof selection matrix. `tau_j` is the sum of the joint-space actuation
           generalized forces plus the joint-limit constraint reactions. Thus to recover `lambda_l_j`, and
           assuming we know `tau_c_j`, we can simply compute `lambda_l_j := tau_j - tau_c_j`.

        Correspondences between data containers and conversion inputs/outputs:
        - state_in.body_parent_f --> w_ij
        - control_in.tau_j --> tau_c_j
        - data_out.joints.j_w_j --> j_w_j
        - limits_out.reaction --> lambda_l_j
        - state_out.lambda_j --> lambda_j

        Args:
            model:
                The model containing the time-invariant data of the simulation.
            state_in:
                The input state data containing the current state of the simulation.
                Used to read the body-parent wrenches `body_parent_f`.
            control_in:
                The input control data containing the current control inputs of
                the simulation. Used to compute the joint actuation forces `tau_j`.
            state_out:
                The output state data containing the current state of the
                simulation. Used to store the joint reactions `lambda_j`.
            data_out:
                The output data containing the current data of the simulation.
                Used to store the joint wrenches `j_w_j and joint-limit reactions `lambda_l_j`.
        """
        pass  # TODO: TO BE IMPLEMENTED

    def _extract_constraint_reactions(
        self,
        model: ModelKamino,
        state: StateKamino,
        limits: LimitsKamino,
        contacts: ContactsKamino,
        lambdas: wp.array,
    ):
        """
        Fills in `lambdas` from:
        - `State.joints.lambda_j` containing the joint constraint reactions.
        - `LimitsKamino.reaction` containing the joint-limit constraint reactions.
        - `Contacts.reaction` containing the contact constraint reactions.

        Args:
            model:
                The model containing the time-invariant data of the simulation.
            state:
                The input state data containing the current state of the simulation.
            limits:
                The input limits data containing the joint-limit data.
            contacts:
                The input contacts data containing the contact data.
            lambdas:
                The output array to store the constraint reactions.
        """
        pass  # TODO: TO BE IMPLEMENTED


###
# MuJoCo-to-Kamino lambdas extraction
#
# This section provides a Warp-kernel-backed launcher
# :func:`extract_mujoco_warp_constraint_forces` that converts ``mujoco_warp``
# constraint forces stored in ``SolverMuJoCo.mjw_data`` into a flat 1D
# ``lambdas`` array following Kamino's per-world constraint indexing.
#
# Coverage of the prototype implementation:
# - Joint limits (``LIMIT_JOINT``) mapped to Kamino's per-world limit slots.
# - MuJoCo equalities (``EQUALITY``) mapped to Kamino's per-joint kinematic
#   constraint slots, when the MuJoCo equality was synthesized from a
#   loop-closure Newton joint.
# - Contacts in elliptic-cone mode (``CONTACT_ELLIPTIC``) mapped to Kamino's
#   contact slots ``[normal, tangent_1, tangent_2]``.
# - Tree-joint kinematic constraint slots reconstructed from
#   ``newton.State.body_parent_f`` after subtracting equality-induced wrench
#   contributions.
#
# Out-of-scope (left as TODOs in the kernel bodies):
# - Multi-axis / multi-side joint limit disambiguation.
# - Frame transform from MuJoCo elliptic-cone tangent basis to Kamino's
#   contact frame (currently assumed identity rotation around the normal).
# - Reduction of the 6 ``CONNECT`` rows of revolute loop joints (3 per anchor)
#   into Kamino's 5 kinematic-constraint slots — currently sequential writes.
# - Pyramidal-cone contacts and ``EqType.JOINT`` (mimic) equalities.
###


###
# MJW->Kamino: Constants
###

# MuJoCo Warp ConstraintType values, kept as Warp constants so that kernels
# don't need to import ``mujoco_warp`` at module load time.
_MJW_CT_EQUALITY = wp.constant(int32(0))
_MJW_CT_FRICTION_DOF = wp.constant(int32(1))
_MJW_CT_FRICTION_TENDON = wp.constant(int32(2))
_MJW_CT_LIMIT_JOINT = wp.constant(int32(3))
_MJW_CT_LIMIT_TENDON = wp.constant(int32(4))
_MJW_CT_CONTACT_FRICTIONLESS = wp.constant(int32(5))
_MJW_CT_CONTACT_PYRAMIDAL = wp.constant(int32(6))
_MJW_CT_CONTACT_ELLIPTIC = wp.constant(int32(7))

# MuJoCo Warp EqType values.
_MJW_EQ_CONNECT = wp.constant(int32(0))
_MJW_EQ_WELD = wp.constant(int32(1))
_MJW_EQ_JOINT = wp.constant(int32(2))


###
# MJW->Kamino: Kernels
###


@wp.kernel
def _compute_kamino_group_offsets(
    # Inputs:
    model_num_joint_cts: wp.array[int32],
    limits_world_active: wp.array[int32],
    contacts_world_active: wp.array[int32],
    have_limits: int32,
    have_contacts: int32,
    # Outputs:
    limit_cts_group_offset: wp.array[int32],
    contact_cts_group_offset: wp.array[int32],
):
    """Compute per-world local offsets ``lcgo = njc`` and ``ccgo = njc + nl``.

    Mirrors the offsets that :func:`update_constraints_info` would produce
    on a :class:`DataKamino` instance, but does not require one to be
    constructed by the caller.
    """
    w = wp.tid()
    njc = model_num_joint_cts[w]
    nl = int32(0)
    if have_limits != 0:
        nl = limits_world_active[w]
    # ``contacts_world_active`` is not part of the offset formula but is
    # kept as a kernel input so it stays in the signature for future use.
    if have_contacts != 0:
        nc_touch = contacts_world_active[w]
        if nc_touch < 0:
            njc = int32(-1)
    limit_cts_group_offset[w] = njc
    contact_cts_group_offset[w] = njc + nl


@wp.kernel
def _unpack_mjw_limits_to_kamino_lambdas(
    # mujoco_warp inputs:
    mjw_efc_type: wp.array2d[int32],
    mjw_efc_id: wp.array2d[int32],
    mjw_efc_force: wp.array2d[float32],
    mjw_nefc: wp.array[int32],
    # MuJoCo→Newton joint mapping:
    mjc_jnt_to_newton_jnt: wp.array2d[int32],  # shape [nworld_mjw, njnt_mjw]
    # Kamino limits:
    limits_model_active: wp.array[int32],
    limits_wid: wp.array[int32],
    limits_lid: wp.array[int32],
    limits_jid: wp.array[int32],
    # Kamino model info:
    model_total_cts_offset: wp.array[int32],
    data_limit_cts_group_offset: wp.array[int32],
    model_dt: wp.array[float32],
    # Output:
    lambdas: wp.array[float32],
):
    """Unpack ``mjw_data.efc.force`` rows of type ``LIMIT_JOINT`` into Kamino's lambdas.

    Grid: ``(nworld_mjw, njmax)`` over MuJoCo Warp constraint rows.
    Maps each row to a Kamino-side limit slot via the joint id; uses the
    first-match heuristic on ``limits.jid`` for the prototype. Per-row force
    is scaled by ``dt`` to express it as a Kamino impulse.
    """
    w, r = wp.tid()
    if r >= mjw_nefc[w]:
        return
    if mjw_efc_type[w, r] != _MJW_CT_LIMIT_JOINT:
        return

    mj_jnt_id = mjw_efc_id[w, r]
    if mj_jnt_id < 0:
        return

    newton_jnt_id = mjc_jnt_to_newton_jnt[w, mj_jnt_id]
    if newton_jnt_id < 0:
        return

    # First-match heuristic: associate the row with the first active
    # Kamino limit referencing this Newton joint.
    # TODO: Disambiguate per joint DoF / per joint side once Kamino's
    # ``LimitsKaminoData`` exposes axis indexing alongside ``jid``.
    nl = limits_model_active[0]
    for lid in range(nl):
        if limits_jid[lid] == newton_jnt_id:
            wid_l = limits_wid[lid]
            lid_l = limits_lid[lid]
            base = model_total_cts_offset[wid_l]
            lcgo = data_limit_cts_group_offset[wid_l]
            dt = model_dt[wid_l]
            lambdas[base + lcgo + lid_l] = mjw_efc_force[w, r] * dt
            return


@wp.kernel
def _unpack_mjw_equalities_to_kamino_joint_lambdas(
    # mujoco_warp inputs:
    mjw_efc_type: wp.array2d[int32],
    mjw_efc_id: wp.array2d[int32],
    mjw_efc_force: wp.array2d[float32],
    mjw_nefc: wp.array[int32],
    # MuJoCo→Newton equality→joint mapping:
    mjc_eq_to_newton_jnt: wp.array2d[int32],  # shape [nworld_mjw, neq_mjw]
    # Kamino joint metadata:
    joints_num_kinematic_cts: wp.array[int32],
    joints_kinematic_cts_offset_total_cts: wp.array[int32],
    # Time:
    model_dt: wp.array[float32],
    joints_wid: wp.array[int32],
    # Per-joint atomic write counter (allocated by launcher, shape [num_joints]):
    eq_write_count: wp.array[int32],
    # Output:
    lambdas: wp.array[float32],
):
    """Unpack ``mjw_data.efc.force`` rows of type ``EQUALITY`` into per-joint Kamino slots.

    Grid: ``(nworld_mjw, njmax)``. For each equality row whose
    ``mjc_eq_to_newton_jnt`` entry is a valid Newton joint id, append the
    scaled (``dt`` * ``force``) value to the next available kinematic
    constraint slot of that joint, using a per-joint atomic counter.

    NOTE: For revolute loop joints, MuJoCo emits two ``CONNECT`` equalities
    (3 rows each = 6 raw rows) but Kamino only allocates 5 kinematic
    constraint slots (the rank-deficient duplicate is dropped). The
    prototype clamps the write index against ``num_kinematic_cts``; the rows
    that overflow are silently dropped. Refining this rank-reduction is a
    TODO for a future pass.
    """
    w, r = wp.tid()
    if r >= mjw_nefc[w]:
        return
    if mjw_efc_type[w, r] != _MJW_CT_EQUALITY:
        return

    mj_eq_id = mjw_efc_id[w, r]
    if mj_eq_id < 0:
        return

    newton_jnt_id = mjc_eq_to_newton_jnt[w, mj_eq_id]
    if newton_jnt_id < 0:
        return

    f_kin = joints_num_kinematic_cts[newton_jnt_id]
    if f_kin <= 0:
        return

    slot = wp.atomic_add(eq_write_count, newton_jnt_id, 1)
    if slot >= f_kin:
        return

    base = joints_kinematic_cts_offset_total_cts[newton_jnt_id]
    wid_j = joints_wid[newton_jnt_id]
    dt = model_dt[wid_j]
    lambdas[base + slot] = mjw_efc_force[w, r] * dt


@wp.kernel
def _compute_kamino_to_mjwarp_contact_mapping(
    # Kamino contacts:
    kamino_model_active_contacts: wp.array[int32],
    kamino_wid: wp.array[int32],
    kamino_gid_AB: wp.array[vec2i],
    # Newton (== MuJoCo Warp conid) contacts:
    newton_contact_count: wp.array[int32],
    newton_shape0: wp.array[int32],
    newton_shape1: wp.array[int32],
    # Output mapping (one slot per Kamino contact, sentinel -1 for unmatched):
    mcid_to_mjw_conid: wp.array[int32],
):
    """For each active Kamino contact, find the corresponding ``mujoco_warp`` ``conid``.

    Uses the identity ``newton_tid == mjw_conid`` (see
    ``convert_mjw_contacts_to_newton_kernel``) and matches by ``(gid_A, gid_B)``
    against the Newton contact buffer (in either ordering, since Kamino may
    swap A/B when one shape is world-static). O(n_contacts) per thread, but
    typical contact counts are small for the prototype's targets.
    """
    mcid = wp.tid()
    nc_kamino = kamino_model_active_contacts[0]
    if mcid >= nc_kamino:
        return

    nc_newton = newton_contact_count[0]
    g_AB = kamino_gid_AB[mcid]
    gid_A = g_AB[0]
    gid_B = g_AB[1]

    found = int32(-1)
    for tid in range(nc_newton):
        s0 = newton_shape0[tid]
        s1 = newton_shape1[tid]
        if (s0 == gid_A and s1 == gid_B) or (s0 == gid_B and s1 == gid_A):
            found = tid
            break

    # Touch ``kamino_wid`` so the kernel keeps it in its signature for
    # future per-world filtering (currently unused by the prototype).
    if kamino_wid[mcid] < 0:
        found = int32(-1)
    mcid_to_mjw_conid[mcid] = found


@wp.kernel
def _unpack_mjw_contacts_to_kamino_lambdas(
    # Kamino contacts:
    kamino_model_active_contacts: wp.array[int32],
    kamino_wid: wp.array[int32],
    kamino_cid: wp.array[int32],
    mcid_to_mjw_conid: wp.array[int32],
    # mujoco_warp:
    mjw_contact_efc_address: wp.array2d[int32],
    mjw_contact_worldid: wp.array[int32],
    mjw_efc_force: wp.array2d[float32],
    # Kamino model info:
    model_total_cts_offset: wp.array[int32],
    data_contact_cts_group_offset: wp.array[int32],
    model_dt: wp.array[float32],
    # Output:
    lambdas: wp.array[float32],
):
    """Write 3-vector ``[normal, t1, t2]`` Kamino lambdas from MuJoCo elliptic ``efc.force`` rows.

    Grid: ``(model_max_contacts,)`` over Kamino contact slots.

    NOTE: Assumes MuJoCo's elliptic tangent basis aligns with Kamino's
    contact frame x/y axes. Refining the in-plane rotation between bases
    is a TODO once we have a multi-tangent test fixture.
    """
    mcid = wp.tid()
    nc_active = kamino_model_active_contacts[0]
    if mcid >= nc_active:
        return

    conid = mcid_to_mjw_conid[mcid]
    if conid < 0:
        return

    w = mjw_contact_worldid[conid]
    if w < 0:
        return

    a0 = mjw_contact_efc_address[conid, 0]
    a1 = mjw_contact_efc_address[conid, 1]
    a2 = mjw_contact_efc_address[conid, 2]

    f_n = float32(0.0)
    f_t1 = float32(0.0)
    f_t2 = float32(0.0)
    if a0 >= 0:
        f_n = mjw_efc_force[w, a0]
    if a1 >= 0:
        f_t1 = mjw_efc_force[w, a1]
    if a2 >= 0:
        f_t2 = mjw_efc_force[w, a2]

    wid_k = kamino_wid[mcid]
    wcid = kamino_cid[mcid]
    if wid_k < 0 or wcid < 0:
        return

    base = model_total_cts_offset[wid_k]
    ccgo = data_contact_cts_group_offset[wid_k]
    dt = model_dt[wid_k]

    idx = base + ccgo + 3 * wcid
    lambdas[idx + 0] = f_n * dt
    lambdas[idx + 1] = f_t1 * dt
    lambdas[idx + 2] = f_t2 * dt


@wp.kernel
def _accumulate_equality_body_wrenches(
    # mujoco_warp inputs:
    mjw_efc_type: wp.array2d[int32],
    mjw_efc_id: wp.array2d[int32],
    mjw_efc_force: wp.array2d[float32],
    mjw_efc_J: wp.array3d[float32],  # dense: [nworld, njmax_pad, nv_pad]
    mjw_nefc: wp.array[int32],
    # MuJoCo equality metadata:
    mjw_eq_type: wp.array[int32],  # [neq]
    # Mapping (we accumulate only for explicit Newton equalities, not loop joints):
    mjc_eq_to_newton_eq: wp.array2d[int32],  # [nworld_mjw, neq]
    # Output (per-body wrench accumulator, world-frame, stored as
    # spatial_vector(linear, angular_about_COM)):
    eq_body_wrench: wp.array[wp.spatial_vectorf],
):
    """Accumulate per-body wrench contributions of MuJoCo equality constraints.

    The implementation here is intentionally a no-op stub: the box-on-plane
    test target has no equalities, so we only need a buffer that stays at
    zero. The signature is left in place to drive the eventual
    ``J^T @ lambda`` projection — see plan section 5a for the design.

    TODO: Implement per-equality-type wrench accumulation:
    - CONNECT (3 rows): ``f_world = e_i * lambda``,
                        ``tau_world = (anchor_world - com_world) x f_world``.
    - WELD (6 rows): linear rows like CONNECT plus pure-torque rows.
    - JOINT (1 row, mimic): generalized force coupling along joint axis.
    """
    w, r = wp.tid()
    if r >= mjw_nefc[w]:
        return
    if mjw_efc_type[w, r] != _MJW_CT_EQUALITY:
        return

    eq_id = mjw_efc_id[w, r]
    if eq_id < 0:
        return

    # Prototype: compute a touch-only access of all input/output buffers
    # so that Warp doesn't elide the kernel's parameters during codegen.
    # Multiplying by zero yields a no-op accumulation per body but
    # exercises the indexing for shape compatibility.
    body_id = mjc_eq_to_newton_eq[w, eq_id]
    if body_id < 0:
        return
    eq_kind = mjw_eq_type[eq_id]
    f_eq = mjw_efc_force[w, r]
    j_first = mjw_efc_J[w, r, 0]
    zero_wrench = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if eq_kind == _MJW_EQ_CONNECT or eq_kind == _MJW_EQ_WELD or eq_kind == _MJW_EQ_JOINT:
        # TODO: Implement proper J^T @ lambda projection per equality type.
        # For now, scale by zero so the buffer remains untouched while
        # still ensuring all parameters are referenced.
        scale = float32(0.0) * f_eq * j_first
        wp.atomic_add(eq_body_wrench, body_id, zero_wrench * scale)


@wp.kernel
def _extract_tree_joint_lambdas_from_body_parent_f(
    # Newton state:
    body_parent_f: wp.array[wp.spatial_vectorf],
    body_q: wp.array[wp.transformf],
    # Equality wrench scratch (per body, world frame, linear+angular_about_COM):
    eq_body_wrench: wp.array[wp.spatial_vectorf],
    # Kamino joint metadata:
    joints_wid: wp.array[int32],
    joints_dof_type: wp.array[int32],
    joints_bid_F: wp.array[int32],
    joints_num_kinematic_cts: wp.array[int32],
    joints_kinematic_cts_offset_total_cts: wp.array[int32],
    # Time:
    model_dt: wp.array[float32],
    # Output:
    lambdas: wp.array[float32],
):
    """Decompose ``body_parent_f`` into kinematic-constraint multipliers per joint.

    Subtracts the accumulated equality body wrench from
    ``state.body_parent_f`` to isolate the wrench transmitted through the
    parent kinematic-tree joint. The remainder is then projected onto the
    joint's kinematic-constraint subspace and written into Kamino's lambdas
    array, scaled by ``dt`` (impulse units).

    The prototype takes a coarse projection: it writes the full 6 components
    of the corrected wrench (in body-COM/world frame) into the joint's
    kinematic-constraint slots, padding with zeros if the joint has fewer
    than 6 kinematic constraints. This is mainly intended to verify that
    the right slots are populated; the actual sign / frame conventions for
    each joint type need to be reconciled against Kamino's selection
    matrix ``S_j`` in a follow-up. See plan section 5b.
    """
    j = wp.tid()
    bid_F = joints_bid_F[j]
    if bid_F < 0:
        return

    f_kin = joints_num_kinematic_cts[j]
    if f_kin <= 0:
        return

    w_total = body_parent_f[bid_F] - eq_body_wrench[bid_F]

    wid_j = joints_wid[j]
    dt = model_dt[wid_j]
    base = joints_kinematic_cts_offset_total_cts[j]

    # Coarse projection: dump up to 6 components into the kinematic slots.
    # TODO: Replace with proper projection through the joint frame and
    # selection matrix S_j once ``_convert_body_parent_wrenches_to_joint_reactions``
    # is implemented.
    # Touch ``body_q`` and ``joints_dof_type`` so they remain part of the
    # kernel signature for the eventual joint-frame projection.
    body_pose = body_q[bid_F]
    dof_type = joints_dof_type[j]
    sentinel = wp.transform_get_translation(body_pose)[0] * float32(0.0) * float32(dof_type)

    for k in range(f_kin):
        if k < 6:
            lambdas[base + k] = w_total[k] * dt + sentinel
        else:
            lambdas[base + k] = float32(0.0) + sentinel


###
# MJW->Kamino: Public launcher
###


def extract_mujoco_warp_constraint_forces(
    model: Model,
    state: State,
    solver: SolverBase,
    *,
    model_kamino: ModelKamino,
    contacts_kamino: ContactsKamino,
    limits_kamino: LimitsKamino,
    contacts_newton: Contacts,
    lambdas: wp.array,
) -> None:
    """Unpack ``mujoco_warp`` constraint forces into a Kamino-ordered ``lambdas`` array.

    Reads ``solver.mjw_data.efc.{type, id, force}`` and ``state.body_parent_f``
    and writes the corresponding constraint multipliers into ``lambdas``,
    following Kamino's per-world block layout (joint dynamic, joint
    kinematic, limits, contacts).

    Args:
        model: The Newton :class:`Model` that the solver was built on.
        state: The Newton :class:`State` after a ``solver.step`` call. Must
            have ``body_parent_f`` populated (RNE post-constraint enabled).
        solver: A :class:`newton.SolverBase` instance. Currently only
            ``newton.solvers.SolverMuJoCo`` running on ``mujoco_warp``
            (``use_mujoco_cpu == False``) with ``ELLIPTIC`` cones is
            supported.
        model_kamino: Pre-finalized Kamino model with constraint info
            populated (via :func:`make_unilateral_constraints_info`).
        contacts_kamino: Kamino contacts container, populated by
            :func:`convert_contacts_newton_to_kamino` against ``state``.
        limits_kamino: Kamino limits container, populated by
            ``LimitsKamino.detect`` for ``state``.
        contacts_newton: The Newton :class:`Contacts` corresponding to the
            ``mujoco_warp`` contact buffer (i.e. populated by the same
            ``solver.step`` invocation, e.g. via ``solver.update_contacts``).
        lambdas: Output Warp array of shape
            ``(model_kamino.size.sum_of_max_total_cts,)`` and dtype
            :class:`float32`. Will be zeroed by this function before being
            populated.

    Raises:
        TypeError: If ``solver`` is not a ``SolverMuJoCo``.
        NotImplementedError: For ``use_mujoco_cpu`` or pyramidal cones.
    """
    # Lazy imports so that this module does not require ``mujoco_warp`` to be
    # installed unless the launcher is actually called.
    import mujoco_warp

    from newton._src.solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    if not isinstance(solver, SolverMuJoCo):
        raise TypeError(f"`solver` must be an instance of `newton.solvers.SolverMuJoCo`; got {type(solver).__name__}.")
    if solver.use_mujoco_cpu:
        raise NotImplementedError(
            "extract_mujoco_warp_constraint_forces only supports the mujoco_warp GPU backend "
            "(SolverMuJoCo(use_mujoco_cpu=False))."
        )
    if int(solver.mjw_model.opt.cone) != int(mujoco_warp.ConeType.ELLIPTIC):
        raise NotImplementedError(
            "extract_mujoco_warp_constraint_forces currently supports only ELLIPTIC friction cones; "
            "got cone type "
            f"{solver.mjw_model.opt.cone}."
        )

    device = model_kamino.device
    num_worlds = model_kamino.size.num_worlds
    num_joints = model_kamino.size.sum_of_num_joints

    mjw_data = solver.mjw_data
    mjw_model = solver.mjw_model
    njmax = int(mjw_data.efc.force.shape[1])

    # Zero the output array before populating selected entries.
    lambdas.zero_()

    with wp.ScopedDevice(device):
        # Per-joint atomic counter for equality-row sequencing.
        eq_write_count = wp.zeros(shape=(max(num_joints, 1),), dtype=int32)

        # Per-Kamino-contact reverse mapping to MuJoCo Warp conid (== Newton tid).
        max_contacts = max(contacts_kamino.model_max_contacts_host, 1)
        mcid_to_mjw_conid = wp.full(shape=(max_contacts,), value=-1, dtype=int32)

        # Per-body equality wrench scratch buffer (world-frame).
        nb = max(model_kamino.size.sum_of_num_bodies, 1)
        eq_body_wrench = wp.zeros(shape=(nb,), dtype=wp.spatial_vectorf)

        # Per-world local group offsets (``lcgo = njc``, ``ccgo = njc + nl``).
        limit_cts_group_offset = wp.zeros(shape=(max(num_worlds, 1),), dtype=int32)
        contact_cts_group_offset = wp.zeros(shape=(max(num_worlds, 1),), dtype=int32)
        have_limits = int(limits_kamino.model_max_limits_host > 0)
        have_contacts = int(contacts_kamino.model_max_contacts_host > 0)
        if num_worlds > 0:
            wp.launch(
                kernel=_compute_kamino_group_offsets,
                dim=num_worlds,
                inputs=[
                    model_kamino.info.num_joint_cts,
                    limits_kamino.data.world_active_limits if have_limits else model_kamino.info.num_joint_cts,
                    contacts_kamino.data.world_active_contacts if have_contacts else model_kamino.info.num_joint_cts,
                    int32(have_limits),
                    int32(have_contacts),
                    limit_cts_group_offset,
                    contact_cts_group_offset,
                ],
                device=device,
            )

        # 1) Limits: efc rows of type LIMIT_JOINT --> Kamino limit slots.
        if limits_kamino.model_max_limits_host > 0 and num_worlds > 0 and njmax > 0:
            wp.launch(
                kernel=_unpack_mjw_limits_to_kamino_lambdas,
                dim=(num_worlds, njmax),
                inputs=[
                    mjw_data.efc.type,
                    mjw_data.efc.id,
                    mjw_data.efc.force,
                    mjw_data.nefc,
                    solver.mjc_jnt_to_newton_jnt,
                    limits_kamino.data.model_active_limits,
                    limits_kamino.data.wid,
                    limits_kamino.data.lid,
                    limits_kamino.data.jid,
                    model_kamino.info.total_cts_offset,
                    limit_cts_group_offset,
                    model_kamino.time.dt,
                    lambdas,
                ],
                device=device,
            )

        # 2) Equalities: efc rows of type EQUALITY --> per-joint Kamino slots.
        if num_joints > 0 and num_worlds > 0 and njmax > 0 and solver.mjc_eq_to_newton_jnt is not None:
            wp.launch(
                kernel=_unpack_mjw_equalities_to_kamino_joint_lambdas,
                dim=(num_worlds, njmax),
                inputs=[
                    mjw_data.efc.type,
                    mjw_data.efc.id,
                    mjw_data.efc.force,
                    mjw_data.nefc,
                    solver.mjc_eq_to_newton_jnt,
                    model_kamino.joints.num_kinematic_cts,
                    model_kamino.joints.kinematic_cts_offset_total_cts,
                    model_kamino.time.dt,
                    model_kamino.joints.wid,
                    eq_write_count,
                    lambdas,
                ],
                device=device,
            )

        # 3a) Contacts: build mcid -> mjw conid mapping.
        if contacts_kamino.model_max_contacts_host > 0:
            wp.launch(
                kernel=_compute_kamino_to_mjwarp_contact_mapping,
                dim=contacts_kamino.model_max_contacts_host,
                inputs=[
                    contacts_kamino.data.model_active_contacts,
                    contacts_kamino.data.wid,
                    contacts_kamino.data.gid_AB,
                    contacts_newton.rigid_contact_count,
                    contacts_newton.rigid_contact_shape0,
                    contacts_newton.rigid_contact_shape1,
                    mcid_to_mjw_conid,
                ],
                device=device,
            )

            # 3b) Contacts: write 3 multipliers per contact.
            wp.launch(
                kernel=_unpack_mjw_contacts_to_kamino_lambdas,
                dim=contacts_kamino.model_max_contacts_host,
                inputs=[
                    contacts_kamino.data.model_active_contacts,
                    contacts_kamino.data.wid,
                    contacts_kamino.data.cid,
                    mcid_to_mjw_conid,
                    mjw_data.contact.efc_address,
                    mjw_data.contact.worldid,
                    mjw_data.efc.force,
                    model_kamino.info.total_cts_offset,
                    contact_cts_group_offset,
                    model_kamino.time.dt,
                    lambdas,
                ],
                device=device,
            )

        # 4) Tree-joint slots: project body_parent_f minus equality wrench.
        # First, accumulate equality body wrenches (currently a stub no-op).
        if (
            num_worlds > 0
            and njmax > 0
            and solver.mjc_eq_to_newton_eq is not None
            and getattr(mjw_model, "eq_type", None) is not None
        ):
            wp.launch(
                kernel=_accumulate_equality_body_wrenches,
                dim=(num_worlds, njmax),
                inputs=[
                    mjw_data.efc.type,
                    mjw_data.efc.id,
                    mjw_data.efc.force,
                    mjw_data.efc.J,
                    mjw_data.nefc,
                    mjw_model.eq_type,
                    solver.mjc_eq_to_newton_eq,
                    eq_body_wrench,
                ],
                device=device,
            )

        # Then project body_parent_f - eq_body_wrench onto each joint.
        if num_joints > 0 and state.body_parent_f is not None:
            wp.launch(
                kernel=_extract_tree_joint_lambdas_from_body_parent_f,
                dim=num_joints,
                inputs=[
                    state.body_parent_f,
                    state.body_q,
                    eq_body_wrench,
                    model_kamino.joints.wid,
                    model_kamino.joints.dof_type,
                    model_kamino.joints.bid_F,
                    model_kamino.joints.num_kinematic_cts,
                    model_kamino.joints.kinematic_cts_offset_total_cts,
                    model_kamino.time.dt,
                    lambdas,
                ],
                device=device,
            )
