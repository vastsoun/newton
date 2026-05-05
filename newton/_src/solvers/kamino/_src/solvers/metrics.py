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
from ..core.bodies import update_body_inertias, update_body_wrenches
from ..core.control import ControlKamino
from ..core.data import DataKamino
from ..core.math import screw, screw_angular, screw_linear
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..core.types import float32, int32, int64, mat33f, uint32, vec2f, vec3f, vec4f, vec6f
from ..dynamics.dual import DualProblem
from ..dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from ..geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from ..geometry.keying import build_pair_key2
from ..kinematics.constraints import (
    make_unilateral_constraints_info,
    unpack_constraint_solutions,
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
        self._problem = DualProblem(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            # solver=ConjugateResidualSolver if sparse else LLTBlockedSolver,
            sparse=sparse,
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

        # Update the relevant data fields in `DataKamino` required for the
        # metrics computations, using the provided `StateKamino` instance.
        self._update_solver_data_prestep(self._model, self._state_p, self._data)

        # Run limit detection to generate active limits
        self._limits.detect(q_j=self._state_p.q_j)

        # Perform the necessary conversions and extractions to obtain the
        # solver data in the expected format for the metrics computations
        self._convert_body_parent_wrenches_to_joint_reactions(
            model=self._model,
            state_in=state,
            control_in=self._control,
            state_out=self._state,
            data_out=self._data,
        )
        self._compute_constraint_velocities(
            model=self._model,
            state=self._state,
            jacobians=self._jacobians,
            v_plus=self._v_plus,
        )
        self._extract_constraint_reactions(
            model=self._model,
            state=self._state,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            lambdas=self._lambdas,
        )

        self._update_constraints(contacts=self._contacts)

        # --------------------------------------------------------

        # Evaluate the metrics using the extracted solver data
        self._metrics.evaluate(
            sigma=self._sigma,
            lambdas=self._lambdas,
            v_plus=self._v_plus,
            data=self._data,
            state_p=self._state_p,
            problem=self._problem,
            jacobians=self._jacobians,
            limits=self._limits,
            contacts=self._contacts,
        )

    ###
    # Internals
    ###

    def _read_step_inputs(self, state_in: StateKamino, control_in: ControlKamino):
        wp.copy(self._data.bodies.q_i, state_in.q_i)
        wp.copy(self._data.bodies.u_i, state_in.u_i)
        wp.copy(self._data.bodies.w_i, state_in.w_i)
        wp.copy(self._data.bodies.w_e_i, state_in.w_i_e)
        wp.copy(self._data.joints.q_j, state_in.q_j)
        wp.copy(self._data.joints.q_j_p, state_in.q_j_p)
        wp.copy(self._data.joints.dq_j, state_in.dq_j)
        wp.copy(self._data.joints.lambda_j, state_in.lambda_j)
        self._data.joints.tau_j = control_in.tau_j
        self._data.joints.q_j_ref = control_in.q_j_ref
        self._data.joints.dq_j_ref = control_in.dq_j_ref
        self._data.joints.tau_j_ref = control_in.tau_j_ref

    def _update_joints_data(self, q_j_p: wp.array | None = None):
        if q_j_p is not None:
            _q_j_p = q_j_p
        else:
            wp.copy(self._data.joints.q_j_p, self._data.joints.q_j)
            _q_j_p = self._data.joints.q_j_p
        compute_joints_data(
            model=self._model,
            data=self._data,
            q_j_p=_q_j_p,
        )

    def _update_intermediates(self, state_in: StateKamino):
        self._update_joints_data(q_j_p=state_in.q_j_p)
        update_body_inertias(model=self._model.bodies, data=self._data.bodies)

    def _update_constraint_info(self):
        update_constraints_info(model=self._model, data=self._data)

    def _update_jacobians(self, contacts: ContactsKamino | None = None):
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            reset_to_zero=True,
        )

    def _update_dynamics(self, contacts: ContactsKamino | None = None):
        self._problem.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            jacobians=self._jacobians,
            reset_to_zero=True,
        )

    def _update_actuation_wrenches(self):
        compute_joint_dof_body_wrenches(self._model, self._data, self._jacobians)

    def _update_wrenches(self):
        update_body_wrenches(self._model.bodies, self._data.bodies)

    def _update_solver_data_prestep(self, state_in: StateKamino, control_in: ControlKamino):
        """
        Updates the relevant data fields in `DataKamino` required for the
        metrics computations, using the provided `StateKamino` instance.

        Args:
            model:
                The model containing the time-invariant data of the simulation.
            state_in:
                The input state data containing the current state of the simulation.
            data_out:
                The solver data to be updated.
        """
        self._read_step_inputs(state_in=state_in, control_in=control_in)
        self._update_intermediates(state_in=state_in)
        self._update_constraint_info()
        self._update_jacobians(contacts=self._contacts)
        self._update_actuation_wrenches()
        self._update_dynamics(contacts=self._contacts)
        self._update_wrenches()

    ###
    # TODO
    ###

    def _convert_body_parent_wrenches_to_joint_reactions(
        self,
        model: ModelKamino,
        state_in: State,
        control_in: ControlKamino,
        state_out: StateKamino,
        data_out: DataKamino | None = None,
    ):
        """
        Converts Newton body-parent wrenches `newton.State.body_parent_f` data to Kamino `StateKamino.lambda_j`.

        This operation also updates per-joint wrenches array `DataKamino.joints.j_w_j` as a byproduct, if provided.

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
        -  Having `j_w_j`, we compute `lambda_j` as `[lambda_j; tau_j] = inv(S_j) @ j_w_j`, where `S_j`
           is the `6x6` joint constraint/dof selection matrix. `tau_j` is the joint actuation force and
           can be discarded, as we only need the constraint reactions `lambda_j`.

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
                Optional, used to store the joint wrenches `j_w_j.
        """
        pass

    def _compute_constraint_velocities(
        self,
        model: ModelKamino,
        state: StateKamino,
        jacobians: SystemJacobiansType,
        v: wp.array,
    ):
        """
        Computes `v = J @ u`, where `J := J_cts` and `u := State.bodies.u_i`
        """
        pass

    def _extract_constraint_reactions(
        self,
        model: ModelKamino,
        state: StateKamino,
        limits: LimitsKamino,
        contacts: ContactsKamino,
        jacobians: SystemJacobiansType,
        lambdas: wp.array,
    ):
        """
        Fills in `lambdas` from `State.joints.lambda_j`, `Limits.reaction` and `Contacts.reaction`
        """
        pass

    def _update_constraints(self, contacts: ContactsKamino | None = None):
        compute_constraint_body_wrenches(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            jacobians=self._jacobians,
            lambdas_offsets=self._problem.data.vio,
            lambdas_data=self._lambdas,
        )
        unpack_constraint_solutions(
            lambdas=self._lambdas,
            v_plus=self._v_plus,
            model=self._model,
            data=self._data,
        )
