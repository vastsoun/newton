# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

from functools import cache

import warp as wp

from ....sim import Contacts, Model, State
from .._src.core.bodies import convert_body_origin_to_com
from .._src.core.conversions import convert_joints
from .._src.core.joints import JointDoFType, JointsModel
from .._src.core.model import ModelKaminoInfo
from .._src.core.size import SizeKamino
from .._src.core.types import vec6f
from .._src.kinematics.joints import (
    compute_joint_pose_and_relative_motion,
    convert_angular_vel_to_universal_joint_intermediary_frame,
    get_joint_constraint_angular_residual_function,
)
from .._src.utils import logger as msg

###
# Module interface
###

__all__ = [
    "ConstraintMetrics",
    "JointData",
    "PhysicsMetrics",
    "compute_contact_constraint_metrics",
    "compute_contact_velocities",
    "compute_joint_constraint_metrics",
    "compute_per_world_contact_constraint_summary",
    "compute_per_world_joint_constraint_summary",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class JointData:
    """TODO"""

    def __init__(self, model: Model):
        """
        Initializes the container with the given model.

        Args:
            model: The model containing the time-invariant data of the simulation.
        """
        self.q_j = wp.array(shape=(model.joint_coord_count,), dtype=wp.float32)
        """The joint coordinates."""

        self.dq_j = wp.array(shape=(model.joint_dof_count,), dtype=wp.float32)
        """The joint DoF velocities."""

        self.r_j = wp.array(shape=(model.joint_constraint_count,), dtype=wp.float32)
        """The joint constraint residual."""

        self.dr_j = wp.array(shape=(model.joint_constraint_count,), dtype=wp.float32)
        """The joint constraint residual time-derivative."""

    def clear(self):
        """
        Clears the joints data to zeros.
        """
        self.q_j.zero_()
        self.dq_j.zero_()
        self.r_j.zero_()
        self.dr_j.zero_()


class ConstraintMetrics:
    """
    A generic container to hold Constrained Rigid Body Dynamics (CRBD) and  Nonlinear
    Complementarity Problem (NCP) physical modelling metrics as arrays of residuals.

    For each residual array ``r_*``, a companion ``r_*_argmax`` array of the same
    shape stores per-entity argmax indices. In per-contact (or per-joint) instances
    these companion arrays are unused and stay ``-1``; in per-world summary
    instances they record the contact (or joint) index that achieved the per-world
    maximum, populated by the per-world reduction kernels via ``wp.atomic_exch``.
    """

    def __init__(self, size: int):
        """
        Initializes the container with the given size representing the number of constraint entities.

        Each constraint entity can be a joint (i.e. equality constraints),
        a joint limit, or a contact (i.e. inequality constraints).

        Args:
            size: The number of constraint entities.
        """
        self.r_cts_penetration = wp.array(shape=(size,), dtype=wp.float32)
        """The configuration-level constraint violation residual."""

        self.r_cts_velocity = wp.array(shape=(size,), dtype=wp.float32)
        """The velocity-level constraint violation residual."""

        self.r_ncp_primal = wp.array(shape=(size,), dtype=wp.float32)
        """The NCP primal residual."""

        self.r_ncp_dual = wp.array(shape=(size,), dtype=wp.float32)
        """The NCP dual residual."""

        self.r_ncp_compl = wp.array(shape=(size,), dtype=wp.float32)
        """The NCP complementarity residual."""

        self.r_vi_natmap = wp.array(shape=(size,), dtype=wp.float32)
        """The Variational Inequality (VI) natural-map residual."""

        self.r_cts_penetration_argmax = wp.full(shape=(size,), value=-1, dtype=wp.int32)
        """Argmax companion for :attr:`r_cts_penetration` (entity index achieving the max)."""

        self.r_cts_velocity_argmax = wp.full(shape=(size,), value=-1, dtype=wp.int32)
        """Argmax companion for :attr:`r_cts_velocity` (entity index achieving the max)."""

        self.r_ncp_primal_argmax = wp.full(shape=(size,), value=-1, dtype=wp.int32)
        """Argmax companion for :attr:`r_ncp_primal` (entity index achieving the max)."""

        self.r_ncp_dual_argmax = wp.full(shape=(size,), value=-1, dtype=wp.int32)
        """Argmax companion for :attr:`r_ncp_dual` (entity index achieving the max)."""

        self.r_ncp_compl_argmax = wp.full(shape=(size,), value=-1, dtype=wp.int32)
        """Argmax companion for :attr:`r_ncp_compl` (entity index achieving the max)."""

        self.r_vi_natmap_argmax = wp.full(shape=(size,), value=-1, dtype=wp.int32)
        """Argmax companion for :attr:`r_vi_natmap` (entity index achieving the max)."""

    def clear(self):
        """
        Clears the metrics container.

        Residual values are zeroed and argmax indices are reset to ``-1``.
        """
        self.r_cts_penetration.zero_()
        self.r_cts_velocity.zero_()
        self.r_ncp_primal.zero_()
        self.r_ncp_dual.zero_()
        self.r_ncp_compl.zero_()
        self.r_vi_natmap.zero_()
        self.r_cts_penetration_argmax.fill_(-1)
        self.r_cts_velocity_argmax.fill_(-1)
        self.r_ncp_primal_argmax.fill_(-1)
        self.r_ncp_dual_argmax.fill_(-1)
        self.r_ncp_compl_argmax.fill_(-1)
        self.r_vi_natmap_argmax.fill_(-1)


class PhysicsMetrics:
    """
    A container to hold CRBD-NCP physics modelling performance metrics for a given model.
    """

    def __init__(self, model: Model | None = None):
        """
        Initializes the container with the given model.

        Args:
            model: The model containing the time-invariant data of the simulation.
        """
        # Declare the constraint metrics containers
        self.joints: ConstraintMetrics | None = None
        """Constraint metrics over all joint kinematic constraints (size ``model.joint_constraint_count``)."""

        self.contacts: ConstraintMetrics | None = None
        """Constraint metrics over all active contacts."""

        self.per_world_joints_summary: ConstraintMetrics | None = None
        """Per-world max+argmax summary of the per-joint constraint metrics."""

        self.per_world_contacts_summary: ConstraintMetrics | None = None
        """Per-world max+argmax summary of the per-contact constraint metrics."""

        # Cached Kamino-side joint model used by :func:`compute_joint_constraint_metrics`.
        # Built once at init via the existing Newton->Kamino conversions; provides the
        # COM-baked ``B_r_Bj`` / ``F_r_Fj`` offsets, the ``X_Bj`` / ``X_Fj`` joint orientation matrices,
        # the per-joint ``dof_type`` and ``kinematic_cts_offset`` arrays required by the
        # residual kernel.
        self._kamino_joints_model: JointsModel | None = None
        self._kamino_body_q_com: wp.array | None = None

        # Initialize the constraint metrics containers if a model is provided
        if model is not None:
            if model.joint_count > 0:
                self.joints = ConstraintMetrics(size=model.joint_constraint_count)
                self.per_world_joints_summary = ConstraintMetrics(size=model.world_count)
                self._init_kamino_joint_context(model)
            else:
                msg.warning("No joints in the model, skipping joint constraint metrics.")
            if model.rigid_contact_max > 0:
                self.contacts = ConstraintMetrics(size=model.rigid_contact_max)
                self.per_world_contacts_summary = ConstraintMetrics(size=model.world_count)
            else:
                msg.warning("No contacts in the model, skipping contact constraint metrics.")

    def _init_kamino_joint_context(self, model: Model) -> None:
        """
        Build a Kamino :class:`JointsModel` for the Newton model and allocate the
        working buffer required by :func:`compute_joint_constraint_metrics`.
        """
        self._kamino_joints_model = convert_joints(
            model,
            SizeKamino(),
            ModelKaminoInfo(),
        )

        # Working buffer for the body-origin -> COM transform applied at every
        # metric evaluation (Newton stores body_q at the body origin, while
        # Kamino's joint formulas expect body poses centered at the COM).
        self._kamino_body_q_com = wp.empty(shape=(model.body_count,), dtype=wp.transformf, device=model.device)

    def clear(self):
        """
        Clears the metrics container.
        """
        if self.joints is not None:
            self.joints.clear()
        if self.contacts is not None:
            self.contacts.clear()
        if self.per_world_joints_summary is not None:
            self.per_world_joints_summary.clear()
        if self.per_world_contacts_summary is not None:
            self.per_world_contacts_summary.clear()


###
# Constants
###

UNIT_X = wp.constant(wp.vec3f(1.0, 0.0, 0.0))
"""3D unit vector for the X axis."""

UNIT_Y = wp.constant(wp.vec3f(0.0, 1.0, 0.0))
"""3D unit vector for the Y axis."""

UNIT_Z = wp.constant(wp.vec3f(0.0, 0.0, 1.0))
"""3D unit vector for the Z axis."""

COS_PI_6 = wp.constant(0.8660254037844387)
"""Convenience constant for cos(PI / 6)."""


###
# Functions
###


@wp.func
def make_contact_frame_znorm(n: wp.vec3f) -> wp.mat33f:
    """
    Makes a 3x3 rotation matrix that represents the contact frame
    whose z-axis is aligned with the given unit normal vector n.

    Args:
        n: The 3D unit normal vector of the contact frame.

    Returns:
        A 3x3 rotation matrix representing the contact frame.
    """
    n = wp.normalize(n)
    e = wp.where(wp.abs(wp.dot(n, UNIT_X)) < COS_PI_6, UNIT_X, UNIT_Y)
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return wp.mat33f(t.x, o.x, n.x, t.y, o.y, n.y, t.z, o.z, n.z)


@wp.func
def make_contact_frame_znorm_quat(n: wp.vec3f) -> wp.quatf:
    """
    Makes a quaternion that represents the contact frame
    whose z-axis is aligned with the given unit normal vector n.

    The associated rotation matrix has columns ``[t, o, n]`` so it maps a
    vector expressed in contact-frame coordinates into the world frame.
    To convert a world-frame vector into contact-frame coordinates, use the
    inverse rotation (``wp.quat_rotate_inv``).

    Args:
        n: The 3D unit normal vector of the contact frame.

    Returns:
        A quaternion representing the contact frame.
    """
    R = make_contact_frame_znorm(n)
    return wp.quat_from_matrix(R)


@wp.func
def project_to_contact_tangential_plane(x: wp.vec3f, n: wp.vec3f) -> wp.vec2f:
    """
    Projects a 3D vector x onto a plane defined by a unit normal n
    and extracts its 2D coordinates based on a generated local basis.

    Args:
        x: The 3D vector to project (e.g., [x, y, z]).
        n: The 3D unit normal vector of the plane.

    Returns:
        A 2D vector [u, v] representing the projected coordinates.
    """
    # Ensure n is a unit vector to avoid scaling issues
    n = wp.normalize(n)

    # 1. Choose an arbitrary 'up' vector not parallel to n
    # If n is highly aligned with the X-axis, use the Y-axis. Otherwise, use the X-axis.
    if wp.abs(n.x) > 0.9:
        v_arb = wp.vec3f(0.0, 1.0, 0.0)
    else:
        v_arb = wp.vec3f(1.0, 0.0, 0.0)

    # 2. Compute the first local basis vector (t1) on the plane
    t1 = wp.normalize(wp.cross(v_arb, n))

    # 3. Compute the second local basis vector (t2) on the plane
    t2 = wp.normalize(wp.cross(n, t1))

    # 4. Extract the 2D coordinates via dot product
    u = wp.dot(x, t1)
    v = wp.dot(x, t2)

    # Return the projected 2D coordinates
    return wp.vec2f(u, v)


@wp.func
def project_to_coulomb_cone(x: wp.vec3f, mu: wp.float32, epsilon: wp.float32 = 0.0) -> wp.vec3f:
    """
    Projects a 3D vector `x` onto an isotropic Coulomb friction cone defined by the friction coefficient `mu`.

    Args:
        x: The input vector to be projected.
        mu: The friction coefficient defining the aperture of the cone.
        epsilon: A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        The vector projected onto the Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if mu * xt_norm > -xn + epsilon:
        if xt_norm < mu * xn + epsilon:
            y = x
        else:
            ys = (mu * xt_norm + xn) / (mu * mu + 1.0)
            yts = mu * ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = ys
    return y


@wp.func
def project_to_coulomb_dual_cone(x: wp.vec3f, mu: wp.float32, epsilon: wp.float32 = 0.0) -> wp.vec3f:
    """
    Projects a 3D vector `x` onto the dual of an isotropic Coulomb
    friction cone defined by the friction coefficient `mu`.

    Args:
        x: The input vector to be projected.
        mu: The friction coefficient defining the aperture of the cone.
        epsilon: A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        wp.vec3f: The vector projected onto the dual Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if xt_norm > -mu * xn + epsilon:
        if mu * xt_norm < xn + epsilon:
            y = x
        else:
            ys = (xt_norm + mu * xn) / (mu * mu + 1.0)
            yts = ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = mu * ys
    return y


@wp.func
def infnorm3(x: wp.vec3f) -> wp.float32:
    """
    Returns the infinity-norm (i.e. maximum absolute value) of a 3D vector.

    Args:
        x: The input 3D vector.

    Returns:
        The infinity-norm of the input vector.
    """
    return wp.max(wp.abs(x))


@wp.func
def resolve_contact_world(
    shape_world: wp.array[wp.int32],
    sid_0: wp.int32,
    sid_1: wp.int32,
) -> wp.int32:
    """Returns the world index of a contact pair.

    Falls back to the partner shape when one side is global (``shape_world < 0``).
    Returns ``-1`` only when both shapes are global, in which case the caller
    should silently skip the contact.
    """
    wid = shape_world[sid_0]
    if wid < int(0):
        wid = shape_world[sid_1]
    return wid


@wp.func
def atomic_max_with_argmax(
    metric: wp.array[wp.float32],
    argmax: wp.array[wp.int32],
    wid: wp.int32,
    value: wp.float32,
    cid: wp.int32,
):
    """Updates the per-world max and argmax for a single residual.

    Mirrors the pattern used by ``_src/solvers/metrics.py``: the max is exact
    by virtue of ``wp.atomic_max``, while the argmax is best-effort under
    contention. In practice the inequality ``value >= previous_max`` is rarely
    false right after an atomic_max returns a larger value from another thread,
    and the argmax converges to the correct contact index once all threads
    finish.
    """
    previous_max = wp.atomic_max(metric, wid, value)
    if value >= previous_max:
        wp.atomic_exch(argmax, wid, cid)


###
# Kernels
###


# TODO: FIX THIS: needs body_q_mins and body_qd_plus
@wp.kernel
def _compute_contact_velocities(
    # Constants:
    rigid_contact_max: int,
    # Inputs:
    shape_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3f],
    body_q: wp.array[wp.transformf],
    body_qd: wp.array[wp.spatial_vectorf],
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    # Outputs:
    contact_velocity: wp.array[wp.spatial_vectorf],
):
    """
    Computes the relative spatial twist of body1 with respect to body0 for each
    contact pair, referenced at body0's COM in world frame.

    The output is a spatial vector ``(v, omega)`` where ``v`` is the linear component of
    body1's twist measured at body0's COM expressed in world coordinates, and ``omega``
    is the relative angular velocity ``omega_body1 - omega_body0`` (reference-point
    invariant). The contact-point relative velocity is recovered as:
    ``v + omega x (r_contact - r_com_body0_world)``.

    Contact-frame residuals (handled by ``_compute_contact_constraint_metrics``)
    instead recompute the relative velocity at the actual contact point.
    """
    # Retrieve the contact index from the thread grid
    cid = wp.tid()

    # Retrieve the active contact count
    num_active = contact_count[0]

    # Skip if the contact index is greater than
    # the active or the maximum contact count
    if cid >= wp.min(num_active, rigid_contact_max):
        return

    # Retrieve the shape and body indices for this contact
    sid_0 = contact_shape0[cid]
    sid_1 = contact_shape1[cid]
    bid_0 = shape_body[sid_0]
    bid_1 = shape_body[sid_1]

    # Retrieve the body transforms for this contact.
    # The body-local COM offset defaults to zero for static bodies so that
    # the world-frame COM reduces to the identity transform's origin.
    r_body_to_com_0 = wp.vec3f(0.0)
    u_body_0 = wp.spatial_vectorf(0.0)
    X_body_0 = wp.transform_identity(dtype=wp.float32)
    if bid_0 >= 0:
        r_body_to_com_0 = body_com[bid_0]
        u_body_0 = body_qd[bid_0]
        X_body_0 = body_q[bid_0]
    r_body_to_com_1 = wp.vec3f(0.0)
    u_body_1 = wp.spatial_vectorf(0.0)
    X_body_1 = wp.transform_identity(dtype=wp.float32)
    if bid_1 >= 0:
        r_body_to_com_1 = body_com[bid_1]
        u_body_1 = body_qd[bid_1]
        X_body_1 = body_q[bid_1]

    # Compute the world-frame COM positions of both bodies. Newton stores
    # `body_qd` linear components at the COM, so the moment-arm shift for the
    # relative twist must be measured between COMs in world coordinates.
    # Fixes B2: previously this kernel used `wp.transform_get_translation(X_body_i)`
    # (body-frame origin in world), which mishandles any body whose `body_com`
    # offset is non-zero (e.g. free-jointed assemblies with off-COM body frames).
    r_com_0_world = wp.transform_point(X_body_0, r_body_to_com_0)
    r_com_1_world = wp.transform_point(X_body_1, r_body_to_com_1)

    # Extract linear and angular parts
    v_body_1 = wp.spatial_top(u_body_1)
    omega_body_1 = wp.spatial_bottom(u_body_1)

    # Compute the spatial twist of body1 with respect to body0 COM reference point
    v_1_ref_to_0 = v_body_1 + wp.cross(omega_body_1, r_com_0_world - r_com_1_world)
    u_1_ref_to_0 = wp.spatial_vector(v_1_ref_to_0, omega_body_1)

    # Compute the relative spatial twist of body1 with
    # respect to body0 at the body0 COM reference point
    u_01_ref_to_0 = u_1_ref_to_0 - u_body_0

    # Store the relative twist of the contacting
    # bodies as the contact spatial twist
    contact_velocity[cid] = u_01_ref_to_0


@wp.kernel
def _compute_contact_constraint_metrics(
    # Constants:
    dt: wp.float32,
    rigid_contact_max: wp.int32,
    # Inputs:
    shape_body: wp.array[wp.int32],
    shape_margin: wp.array[wp.float32],
    shape_material_mu: wp.array[wp.float32],
    shape_material_restitution: wp.array[wp.float32],
    body_com: wp.array[wp.vec3f],
    body_q_minus: wp.array[wp.transformf],
    body_qd_minus: wp.array[wp.spatial_vectorf],
    # body_q_plus: wp.array[wp.transformf],
    body_qd_plus: wp.array[wp.spatial_vectorf],
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_offset0: wp.array[wp.vec3f],
    contact_offset1: wp.array[wp.vec3f],
    contact_point0: wp.array[wp.vec3f],
    contact_point1: wp.array[wp.vec3f],
    contact_normal: wp.array[wp.vec3f],
    contact_force: wp.array[wp.spatial_vectorf],
    # Outputs:
    r_contact_cts_penetration: wp.array[wp.float32],
    r_contact_cts_velocity: wp.array[wp.float32],
    r_contact_ncp_primal: wp.array[wp.float32],
    r_contact_ncp_dual: wp.array[wp.float32],
    r_contact_ncp_compl: wp.array[wp.float32],
    r_contact_vi_natmap: wp.array[wp.float32],
):
    # Retrieve the contact index from the thread grid
    cid = wp.tid()

    # Retrieve the active contact count
    num_active = wp.min(contact_count[0], rigid_contact_max)

    # Skip if the contact index is greater than
    # the active or the maximum contact count
    if cid >= num_active:
        return

    # ---------------------------------------------------------
    # Contact properties
    # ---------------------------------------------------------

    # Retrieve the shape and body indices for this contact
    sid_0 = contact_shape0[cid]
    sid_1 = contact_shape1[cid]
    bid_0 = shape_body[sid_0]
    bid_1 = shape_body[sid_1]

    # Retrieve the shape-specific properties for this contact
    margin_0 = shape_margin[sid_0]
    margin_1 = shape_margin[sid_1]

    # Retrieve the properties for this contact
    r_bc_0_offset = contact_offset0[cid]
    r_bc_1_offset = contact_offset1[cid]
    r_bc_0_local = contact_point0[cid]
    r_bc_1_local = contact_point1[cid]
    n_01 = wp.normalize(contact_normal[cid])
    # wp.printf("[%d] n_01: [%.9f, %.9f, %.9f]\n", cid, n_01[0], n_01[1], n_01[2])

    # Retrieve the material properties for this contact
    mu_0 = shape_material_mu[sid_0]
    mu_1 = shape_material_mu[sid_1]
    mu_01 = 0.5 * (mu_0 + mu_1)
    restitution_0 = shape_material_restitution[sid_0]
    restitution_1 = shape_material_restitution[sid_1]
    restitution_01 = 0.5 * (restitution_0 + restitution_1)

    # Make the contact frame quaternion. The associated rotation matrix has
    # columns ``[t, o, n]``, i.e. it maps contact-frame coordinates to the
    # world frame; the inverse rotation is needed below to express world-frame
    # vectors in contact-frame components.
    q_contact = make_contact_frame_znorm_quat(n_01)
    # wp.printf("[%d] q_contact: [%.9f, %.9f, %.9f, %.9f]\n", cid, q_contact[0], q_contact[1], q_contact[2], q_contact[3])

    # Retrieve the body transforms for this contact
    r_body_to_com_0 = wp.vec3f(0.0)
    u_body_0_minus = wp.spatial_vectorf(0.0)
    u_body_0_plus = wp.spatial_vectorf(0.0)
    X_body_0_minus = wp.transform_identity(dtype=wp.float32)
    # X_body_0_plus = wp.transform_identity(dtype=wp.float32)
    if bid_0 >= 0:
        r_body_to_com_0 = body_com[bid_0]
        u_body_0_minus = body_qd_minus[bid_0]
        u_body_0_plus = body_qd_plus[bid_0]
        X_body_0_minus = body_q_minus[bid_0]
        # X_body_0_plus = body_q_plus[bid_0]
    r_body_to_com_1 = wp.vec3f(0.0)
    u_body_1_minus = wp.spatial_vectorf(0.0)
    u_body_1_plus = wp.spatial_vectorf(0.0)
    X_body_1_minus = wp.transform_identity(dtype=wp.float32)
    # X_body_1_plus = wp.transform_identity(dtype=wp.float32)
    if bid_1 >= 0:
        r_body_to_com_1 = body_com[bid_1]
        u_body_1_minus = body_qd_minus[bid_1]
        u_body_1_plus = body_qd_plus[bid_1]
        X_body_1_minus = body_q_minus[bid_1]
        # X_body_1_plus = body_q_plus[bid_1]

    # Retrieve the spatial contact force, i.e. wrench, applied by body1 onto
    # body0, referenced to the COM of body0, expressed in world frame
    w_10_ref_to_0 = contact_force[cid]

    # ---------------------------------------------------------
    # Contact kinematics and dynamics
    # ---------------------------------------------------------

    # Extract linear and angular parts of the associated bodies' motion
    v_body_com_0_minus = wp.spatial_top(u_body_0_minus)
    v_body_com_1_minus = wp.spatial_top(u_body_1_minus)
    omega_body_0_minus = wp.spatial_bottom(u_body_0_minus)
    omega_body_1_minus = wp.spatial_bottom(u_body_1_minus)
    v_body_com_0_plus = wp.spatial_top(u_body_0_plus)
    v_body_com_1_plus = wp.spatial_top(u_body_1_plus)
    omega_body_0_plus = wp.spatial_bottom(u_body_0_plus)
    omega_body_1_plus = wp.spatial_bottom(u_body_1_plus)
    # wp.printf("[%d] v_body_com_0_minus: [%.9f, %.9f, %.9f]\n", cid, v_body_com_0_minus[0], v_body_com_0_minus[1], v_body_com_0_minus[2])
    # wp.printf("[%d] v_body_com_1_minus: [%.9f, %.9f, %.9f]\n", cid, v_body_com_1_minus[0], v_body_com_1_minus[1], v_body_com_1_minus[2])
    # wp.printf("[%d] omega_body_0_minus: [%.9f, %.9f, %.9f]\n", cid, omega_body_0_minus[0], omega_body_0_minus[1], omega_body_0_minus[2])
    # wp.printf("[%d] omega_body_1_minus: [%.9f, %.9f, %.9f]\n", cid, omega_body_1_minus[0], omega_body_1_minus[1], omega_body_1_minus[2])
    # wp.printf("[%d] v_body_com_0_plus: [%.9f, %.9f, %.9f]\n", cid, v_body_com_0_plus[0], v_body_com_0_plus[1], v_body_com_0_plus[2])
    # wp.printf("[%d] v_body_com_1_plus: [%.9f, %.9f, %.9f]\n", cid, v_body_com_1_plus[0], v_body_com_1_plus[1], v_body_com_1_plus[2])
    # wp.printf("[%d] omega_body_0_plus: [%.9f, %.9f, %.9f]\n", cid, omega_body_0_plus[0], omega_body_0_plus[1], omega_body_0_plus[2])
    # wp.printf("[%d] omega_body_1_plus: [%.9f, %.9f, %.9f]\n", cid, omega_body_1_plus[0], omega_body_1_plus[1], omega_body_1_plus[2])

    # Compute the contact points in world frame
    r_c_0_minus = wp.transform_point(X_body_0_minus, r_bc_0_local + r_bc_0_offset)
    r_c_1_minus = wp.transform_point(X_body_1_minus, r_bc_1_local + r_bc_1_offset)
    # r_c_0_plus = wp.transform_point(X_body_0_plus, r_bc_0_local + r_bc_0_offset)
    # r_c_1_plus = wp.transform_point(X_body_1_plus, r_bc_1_local + r_bc_1_offset)
    # wp.printf("[%d] r_c_0_minus: [%.9f, %.9f, %.9f]\n", cid, r_c_0_minus[0], r_c_0_minus[1], r_c_0_minus[2])
    # wp.printf("[%d] r_c_1_minus: [%.9f, %.9f, %.9f]\n", cid, r_c_1_minus[0], r_c_1_minus[1], r_c_1_minus[2])
    # wp.printf("[%d] r_c_0_plus: [%.9f, %.9f, %.9f]\n", cid, r_c_0_plus[0], r_c_0_plus[1], r_c_0_plus[2])
    # wp.printf("[%d] r_c_1_plus: [%.9f, %.9f, %.9f]\n", cid, r_c_1_plus[0], r_c_1_plus[1], r_c_1_plus[2])

    dr_c_01_minus = r_c_1_minus - r_c_0_minus
    # dr_c_01_plus = r_c_1_plus - r_c_0_plus
    # wp.printf("[%d] dr_c_01_minus: [%.9f, %.9f, %.9f]\n", cid, dr_c_01_minus[0], dr_c_01_minus[1], dr_c_01_minus[2])
    # wp.printf("[%d] dr_c_01_plus: [%.9f, %.9f, %.9f]\n", cid, dr_c_01_plus[0], dr_c_01_plus[1], dr_c_01_plus[2])

    # Compute the world-frame body COM positions used below as the reference
    # points for the per-body twists stored in ``body_qd``.
    r_com_0_world_minus = wp.transform_point(X_body_0_minus, r_body_to_com_0)
    r_com_1_world_minus = wp.transform_point(X_body_1_minus, r_body_to_com_1)
    # r_com_0_world_plus = wp.transform_point(X_body_0_plus, r_body_to_com_0)
    # r_com_1_world_plus = wp.transform_point(X_body_1_plus, r_body_to_com_1)
    # wp.printf("[%d] r_com_0_world_minus: [%.9f, %.9f, %.9f]\n", cid, r_com_0_world_minus[0], r_com_0_world_minus[1], r_com_0_world_minus[2])
    # wp.printf("[%d] r_com_1_world_minus: [%.9f, %.9f, %.9f]\n", cid, r_com_1_world_minus[0], r_com_1_world_minus[1], r_com_1_world_minus[2])
    # wp.printf("[%d] r_com_0_world_plus: [%.9f, %.9f, %.9f]\n", cid, r_com_0_world_plus[0], r_com_0_world_plus[1], r_com_0_world_plus[2])
    # wp.printf("[%d] r_com_1_world_plus: [%.9f, %.9f, %.9f]\n", cid, r_com_1_world_plus[0], r_com_1_world_plus[1], r_com_1_world_plus[2])

    # Reconstruct signed contact distance
    d_01_minus = wp.dot(dr_c_01_minus, n_01) - (margin_0 + margin_1)
    # d_01_plus = wp.dot(dr_c_01_plus, n_01) - (margin_0 + margin_1)
    # wp.printf("[%d] d_01_minus: %.18f\n", cid, d_01_minus)
    # wp.printf("[%d] d_01_plus: %.18f\n", cid, d_01_plus)

    # # Skip if the contact distance is positive, i.e. no penetration
    # # TODO: FIX LOGIC WRT TO MARGINS AND GAP
    # if d_01_minus > 0.0:
    #     return

    # Compute the velocity of the contact on each body.
    v_c_0_minus = v_body_com_0_minus + wp.cross(omega_body_0_minus, r_c_0_minus - r_com_0_world_minus)
    v_c_1_minus = v_body_com_1_minus + wp.cross(omega_body_1_minus, r_c_1_minus - r_com_1_world_minus)
    v_c_0_plus = v_body_com_0_plus + wp.cross(omega_body_0_plus, r_c_0_minus - r_com_0_world_minus)
    v_c_1_plus = v_body_com_1_plus + wp.cross(omega_body_1_plus, r_c_1_minus - r_com_1_world_minus)
    # wp.printf("[%d] v_c_0_minus: [%.9f, %.9f, %.9f]\n", cid, v_c_0_minus[0], v_c_0_minus[1], v_c_0_minus[2])
    # wp.printf("[%d] v_c_1_minus: [%.9f, %.9f, %.9f]\n", cid, v_c_1_minus[0], v_c_1_minus[1], v_c_1_minus[2])
    # wp.printf("[%d] v_c_0_plus: [%.9f, %.9f, %.9f]\n", cid, v_c_0_plus[0], v_c_0_plus[1], v_c_0_plus[2])
    # wp.printf("[%d] v_c_1_plus: [%.9f, %.9f, %.9f]\n", cid, v_c_1_plus[0], v_c_1_plus[1], v_c_1_plus[2])

    # Compute the relative velocity of the contact on each body
    v_01_minus = v_c_1_minus - v_c_0_minus
    v_01_plus = v_c_1_plus - v_c_0_plus
    # wp.printf("[%d] v_01_minus: [%.9f, %.9f, %.9f]\n", cid, v_01_minus[0], v_01_minus[1], v_01_minus[2])
    # wp.printf("[%d] v_01_plus: [%.9f, %.9f, %.9f]\n", cid, v_01_plus[0], v_01_plus[1], v_01_plus[2])
    # TODO (torsional friction): omega_01 = omega_body_1 - omega_body_0

    # Invert the signs to represent the force applied by body0
    # onto body1 and decompose it into linear and angular parts
    f_01 = -wp.spatial_top(w_10_ref_to_0)
    # TODO (torsional friction): tau_01_ref_to_0 = -wp.spatial_bottom(w_10_ref_to_0)
    # wp.printf("[%d] f_01: [%.9f, %.9f, %.9f]\n", cid, f_01[0], f_01[1], f_01[2])

    # Rotate the linear velocity and force into the contact frame.
    v_c_minus = wp.quat_rotate_inv(q_contact, v_01_minus)
    v_c_plus = wp.quat_rotate_inv(q_contact, v_01_plus)
    f_c = wp.quat_rotate_inv(q_contact, f_01)
    # wp.printf("[%d] v_c_minus: [%.9f, %.9f, %.9f]\n", cid, v_c_minus[0], v_c_minus[1], v_c_minus[2])
    # wp.printf("[%d] v_c_plus: [%.9f, %.9f, %.9f]\n", cid, v_c_plus[0], v_c_plus[1], v_c_plus[2])
    # wp.printf("[%d] f_c: [%.9f, %.9f, %.9f]\n", cid, f_c[0], f_c[1], f_c[2])

    # Convert the contact force to an impulse
    f_c *= dt
    # wp.printf("[%d] f_c (impulse): [%.9f, %.9f, %.9f]\n", cid, f_c[0], f_c[1], f_c[2])

    # Compute the De Saxce correction
    s = wp.vec3f(0.0, 0.0, mu_01 * wp.sqrt(v_c_plus.x * v_c_plus.x + v_c_plus.y * v_c_plus.y))
    # wp.printf("[%d] s: [%.9f, %.9f, %.9f]\n", cid, s[0], s[1], s[2])

    # Compute the augmented contact velocity
    v_c_plus_aug = v_c_plus + s
    # wp.printf("[%d] v_c_plus_aug: [%.9f, %.9f, %.9f]\n", cid, v_c_plus_aug[0], v_c_plus_aug[1], v_c_plus_aug[2])

    # ---------------------------------------------------------
    # Contact residuals
    # ---------------------------------------------------------

    # Predict the contact distance after the step
    # d_01_correction = dt * v_c_plus.z
    # wp.printf("[%d] d_01_correction: %.18f\n", cid, d_01_correction)
    # d_01_plus_predicted = d_01_minus + d_01_correction
    # wp.printf("[%d] d_01_plus_predicted: %.18f\n", cid, d_01_plus_predicted)

    # Compute contact status over the state transition of the step
    # contact_active = d_01_minus < 1e-5
    # wp.printf("[%d] contact_active: %d\n", cid, contact_active)
    # contact_active_before = d_01_minus < 0.0
    # wp.printf("[%d] contact_active_before: %d\n", cid, contact_active_before)
    # contact_active_after = d_01_plus < 0.0
    # wp.printf("[%d] contact_active_after: %d\n", cid, contact_active_after)
    # contact_sign_changed = (d_01_plus * d_01_minus) < 0.0
    # wp.printf("[%d] contact_sign_changed: %d\n", cid, contact_sign_changed)

    # # TODO
    # eps = 1e-6
    # contact_just_closed = v_c_minus.z < 0.0
    # contact_remained_closed = d_01_plus < 0.0
    # contact_will_not_open = (wp.min(0.0, d_01_minus) + dt * v_c_plus.z) < eps
    # contact_is_breaking = dt * wp.min(0.0, v_c_plus.z) > 1e-3
    # contact_will_persist = (contact_remained_closed or contact_will_not_open) and not contact_is_breaking
    # compute_metrics = contact_will_persist and not contact_just_closed
    # wp.printf("[%d] contact_just_closed: %d\n", cid, contact_just_closed)
    # wp.printf("[%d] contact_remained_closed: %d\n", cid, contact_remained_closed)
    # wp.printf("[%d] contact_will_not_open: %d\n", cid, contact_will_not_open)
    # wp.printf("[%d] contact_is_breaking: %d\n", cid, contact_is_breaking)
    # wp.printf("[%d] contact_will_persist: %d\n", cid, contact_will_persist)
    # wp.printf("[%d] compute_metrics: %d\n", cid, compute_metrics)

    # TODO
    # eps = 1e-6
    # contact_remained_closed = d_01_plus < 0.0
    # wp.printf("[%d] contact_remained_closed: %d\n", cid, contact_remained_closed)

    # 1. If sum is negative, then the contact is deepening -> check metrics
    # 2. If sum is positive, then the contact is opening -> no metrics
    # contact_traversal = v_c_minus.z + v_c_plus.z  # * dt
    # wp.printf("[%d] contact_traversal: %.18f\n", cid, contact_traversal)

    # contact_is_stable = wp.abs(contact_traversal) < 1e-4
    # wp.printf("[%d] contact_is_stable: %d\n", cid, contact_is_stable)

    contact_restitution = restitution_01 * v_c_minus.z + v_c_plus.z  # * dt
    # wp.printf("[%d] contact_restitution: %.18f\n", cid, contact_restitution)

    contact_is_restitutive = wp.abs(contact_restitution) < 1e-4 and wp.abs(v_c_plus.z) > 1e-3
    # wp.printf("[%d] v_c_plus.z: %.12f\n", cid, v_c_plus.z)
    # wp.printf("[%d] contact_is_restitutive: %d\n", cid, contact_is_restitutive)

    # compute_metrics = True
    compute_metrics = not contact_is_restitutive
    # wp.printf("[%d] compute_metrics: %d\n", cid, compute_metrics)

    # Compute the contact penetration
    r_cts_penetration = wp.abs(wp.min(0.0, d_01_minus))
    # wp.printf("[%d] r_cts_penetration: %.12f\n", cid, r_cts_penetration)

    # Compute the velocity-level constraint violation.
    # The unilateral velocity constraint is ``v_n >= 0`` (separating
    # is admissible). The violation magnitude is ``max(0, -v_n)`` so
    # that admissible separation does not get reported as a residual.
    r_cts_velocity = wp.max(0.0, -wp.dot(v_01_plus, n_01))
    # r_cts_velocity = wp.abs(v_c_plus.z)
    # wp.printf("[%d] r_cts_velocity: %.12f\n", cid, r_cts_velocity)

    # Compute the contact NCP primal
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_primal(lambda) = || lambda - P_K(lambda) ||_inf`, where `P_K()` is the
    # Euclidean projection, i.e. proximal operator, onto K, and `lambda` is the
    # vector of all constraint reactions (i.e. Lagrange multipliers).
    r_ncp_primal = infnorm3(f_c - project_to_coulomb_cone(f_c, mu_01))
    # wp.printf("[%d] r_ncp_primal: %.12f\n", cid, r_ncp_primal)

    # Compute the contact NCP dual
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_dual(v_hat^+) = || v_hat^+ - P_K*(v_hat^+) ||_inf`, where `P_K*()` is
    # the Euclidean projection, i.e. proximal operator, onto K*, and `v_hat^+` is
    # the so-called augmented constraint-space velocity. The latter is defined as
    # `v_hat^+ = v^+ + Gamma(v^+)`, where `v^+ := v_f D @ lambda` is the post-event
    # constraint-space velocity, and `Gamma(v^+)` is the De Saxce correction term.
    # proj_dual = project_to_coulomb_dual_cone(v_c_plus_aug, mu_01)
    # error_proj_dual = v_c_plus_aug - proj_dual
    # wp.printf("[%d] s: [%.9f, %.9f, %.9f]\n", cid, s[0], s[1], s[2])
    # wp.printf("[%d] v_c_plus: [%.9f, %.9f, %.9f]\n", cid, v_c_plus[0], v_c_plus[1], v_c_plus[2])
    # wp.printf("[%d] v_c_plus_aug: [%.9f, %.9f, %.9f]\n", cid, v_c_plus_aug[0], v_c_plus_aug[1], v_c_plus_aug[2])
    # wp.printf("[%d] proj_dual: [%.9f, %.9f, %.9f]\n", cid, proj_dual[0], proj_dual[1], proj_dual[2])
    # wp.printf("[%d] error_proj_dual: [%.9f, %.9f, %.9f]\n", cid, error_proj_dual[0], error_proj_dual[1], error_proj_dual[2])
    # r_ncp_dual = infnorm3(error_proj_dual)
    r_ncp_dual = infnorm3(v_c_plus_aug - project_to_coulomb_dual_cone(v_c_plus_aug, mu_01))
    # wp.printf("[%d] r_ncp_dual: %.12f\n", cid, r_ncp_dual)

    # Compute the contact NCP complementarity
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_compl(lambda) = || lambda.T @ v_hat^+ ||_inf`,
    # where `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    # and `v_hat^+` is the augmented constraint-space velocity defined above.
    # r_ncp_compl = wp.abs(wp.dot(f_c, v_c_plus_aug))
    r_ncp_compl = wp.where(compute_metrics, wp.abs(wp.dot(f_c, v_c_plus_aug)), 0.0)
    # wp.printf("[%d] r_ncp_compl: %.12f\n", cid, r_ncp_compl)

    # Compute the contact VI natural-map
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_vi_natmapv_hat(lambda) = || lambda - P_K(lambda - v_hat^+(lambda)) ||_inf`,
    # where `P_K()` is the Euclidean projection, i.e. proximal operator, onto K,
    # `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    # and `v_hat^+(lambda)` is the augmented constraint-space velocity defined above.
    # r_vi_natmap = infnorm3(f_c - project_to_coulomb_cone(f_c - v_c_plus_aug, mu_01))
    r_vi_natmap = wp.where(compute_metrics, infnorm3(f_c - project_to_coulomb_cone(f_c - v_c_plus_aug, mu_01)), 0.0)
    # wp.printf("[%d] r_vi_natmap: %.12f\n\n", cid, r_vi_natmap)

    # Store the contact residuals
    r_contact_cts_penetration[cid] = r_cts_penetration
    r_contact_cts_velocity[cid] = r_cts_velocity
    r_contact_ncp_primal[cid] = r_ncp_primal
    r_contact_ncp_dual[cid] = r_ncp_dual
    r_contact_ncp_compl[cid] = r_ncp_compl
    r_contact_vi_natmap[cid] = r_vi_natmap


@wp.kernel
def _compute_per_world_contact_metrics_summary(
    # Constants:
    rigid_contact_max: wp.int32,
    # Inputs:
    shape_world: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_r_cts_penetration: wp.array[wp.float32],
    contact_r_cts_velocity: wp.array[wp.float32],
    contact_r_ncp_primal: wp.array[wp.float32],
    contact_r_ncp_dual: wp.array[wp.float32],
    contact_r_ncp_compl: wp.array[wp.float32],
    contact_r_vi_natmap: wp.array[wp.float32],
    # Outputs:
    world_r_cts_penetration: wp.array[wp.float32],
    world_r_cts_penetration_argmax: wp.array[wp.int32],
    world_r_cts_velocity: wp.array[wp.float32],
    world_r_cts_velocity_argmax: wp.array[wp.int32],
    world_r_ncp_primal: wp.array[wp.float32],
    world_r_ncp_primal_argmax: wp.array[wp.int32],
    world_r_ncp_dual: wp.array[wp.float32],
    world_r_ncp_dual_argmax: wp.array[wp.int32],
    world_r_ncp_compl: wp.array[wp.float32],
    world_r_ncp_compl_argmax: wp.array[wp.int32],
    world_r_vi_natmap: wp.array[wp.float32],
    world_r_vi_natmap_argmax: wp.array[wp.int32],
):
    """
    Performs a per-world max+argmax reduction over the per-contact residual arrays.

    One thread per contact slot. Slots beyond the active contact count (or the
    per-contact capacity) early-exit. The world index is resolved from
    ``shape_world`` with a fallback to the partner shape so that one-sided
    static contacts (one global shape) still contribute to the correct world.
    Contacts whose two shapes are both global are silently skipped.

    The reduction uses ``wp.atomic_max`` for the max and ``wp.atomic_exch`` for
    the argmax (best-effort under contention; see :func:`atomic_max_with_argmax`).
    """
    # Retrieve the contact index from the thread grid
    cid = wp.tid()

    # Retrieve the active contact count
    num_active = wp.min(contact_count[0], rigid_contact_max)

    # Skip if the contact index is greater than
    # the active or the maximum contact count
    if cid >= num_active:
        return

    # Retrieve the shape and body indices for this contact
    sid_0 = contact_shape0[cid]
    sid_1 = contact_shape1[cid]

    # Resolve the world index for this contact
    wid = resolve_contact_world(shape_world, sid_0, sid_1)

    # Skip if the world index is less than 0, i.e. both shapes are global
    if wid < int(0):
        return

    # Retrieve the contact residuals for this contact
    r_cts_penetration = contact_r_cts_penetration[cid]
    r_cts_velocity = contact_r_cts_velocity[cid]
    r_ncp_primal = contact_r_ncp_primal[cid]
    r_ncp_dual = contact_r_ncp_dual[cid]
    r_ncp_compl = contact_r_ncp_compl[cid]
    r_vi_natmap = contact_r_vi_natmap[cid]

    # Update the per-world max and argmax for the contact residuals
    atomic_max_with_argmax(world_r_cts_penetration, world_r_cts_penetration_argmax, wid, r_cts_penetration, cid)
    atomic_max_with_argmax(world_r_cts_velocity, world_r_cts_velocity_argmax, wid, r_cts_velocity, cid)
    atomic_max_with_argmax(world_r_ncp_primal, world_r_ncp_primal_argmax, wid, r_ncp_primal, cid)
    atomic_max_with_argmax(world_r_ncp_dual, world_r_ncp_dual_argmax, wid, r_ncp_dual, cid)
    atomic_max_with_argmax(world_r_ncp_compl, world_r_ncp_compl_argmax, wid, r_ncp_compl, cid)
    atomic_max_with_argmax(world_r_vi_natmap, world_r_vi_natmap_argmax, wid, r_vi_natmap, cid)


@wp.kernel
def _compute_per_world_joint_metrics_summary(
    joint_world: wp.array[wp.int32],
    joint_kinematic_cts_offset: wp.array[wp.int32],
    joint_r_cts_penetration: wp.array[wp.float32],
    joint_r_cts_velocity: wp.array[wp.float32],
    world_r_cts_penetration: wp.array[wp.float32],
    world_r_cts_penetration_argmax: wp.array[wp.int32],
    world_r_cts_velocity: wp.array[wp.float32],
    world_r_cts_velocity_argmax: wp.array[wp.int32],
):
    """
    Performs a per-world max+argmax reduction over joint kinematic constraint residuals.

    One thread per joint. Each joint contributes the maximum absolute position-
    and velocity-level residual among its kinematic constraint dimensions
    (indexed via ``kinematic_cts_offset``). The argmax stores the joint index.
    """
    jid = wp.tid()

    wid = joint_world[jid]
    offset = joint_kinematic_cts_offset[jid]
    num_cts = joint_kinematic_cts_offset[jid + 1] - offset

    if num_cts <= 0:
        return

    max_pen = joint_r_cts_penetration[offset]
    max_vel = joint_r_cts_velocity[offset]
    for k in range(1, num_cts):
        idx = offset + k
        val_pen = joint_r_cts_penetration[idx]
        if val_pen > max_pen:
            max_pen = val_pen
        val_vel = joint_r_cts_velocity[idx]
        if val_vel > max_vel:
            max_vel = val_vel

    atomic_max_with_argmax(world_r_cts_penetration, world_r_cts_penetration_argmax, wid, max_pen, jid)
    atomic_max_with_argmax(world_r_cts_velocity, world_r_cts_velocity_argmax, wid, max_vel, jid)


@cache
def make_typed_write_joint_residuals_abs(dof_type: JointDoFType):
    """
    Generate a Warp function that writes the absolute position- and velocity-level
    joint constraint residuals for a specific :class:`JointDoFType`.

    The returned function is a stripped-down counterpart of Kamino's
    ``make_typed_write_joint_data``: it only writes the kinematic constraint
    residuals (``r_cts_penetration`` and ``r_cts_velocity``) and skips the joint
    DoF / generalized coordinate bookkeeping that the full solver pipeline needs.
    """
    cts_axes = dof_type.cts_axes
    num_cts = dof_type.num_cts

    @wp.func
    def _write_typed_joint_residuals_abs(
        kinematic_cts_offset: wp.int32,
        j_r_j: wp.vec3f,
        j_q_j: wp.quatf,
        j_u_j: vec6f,
        r_cts_penetration: wp.array[wp.float32],
        r_cts_velocity: wp.array[wp.float32],
    ):
        # Universal joints need an additional intermediary-body-frame projection
        # of the angular velocity before extracting the constraint axes (mirrors
        # the same step in ``make_typed_write_joint_data``).
        if wp.static(dof_type == JointDoFType.UNIVERSAL):
            j_u_j = convert_angular_vel_to_universal_joint_intermediary_frame(j_q_j, j_u_j)

        if wp.static(num_cts > 0):
            j_theta_j = wp.static(get_joint_constraint_angular_residual_function(dof_type))(j_q_j)
            j_p_j = wp.spatial_vectorf(*j_r_j, *j_theta_j)
            for k in range(num_cts):
                r_cts_penetration[kinematic_cts_offset + k] = wp.abs(j_p_j[cts_axes[k]])
                r_cts_velocity[kinematic_cts_offset + k] = wp.abs(j_u_j[cts_axes[k]])

    return _write_typed_joint_residuals_abs


@cache
def make_write_joint_residuals_abs():
    """
    Generate a Warp dispatch function that routes to the per-DoF-type residual
    writer based on the runtime ``dof_type`` value.
    """

    @wp.func
    def _write_joint_residuals_abs(
        dof_type: wp.int32,
        kinematic_cts_offset: wp.int32,
        j_r_j: wp.vec3f,
        j_q_j: wp.quatf,
        j_u_j: vec6f,
        r_cts_penetration: wp.array[wp.float32],
        r_cts_velocity: wp.array[wp.float32],
    ):
        if dof_type == wp.int32(JointDoFType.FREE.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.FREE))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )
        elif dof_type == wp.int32(JointDoFType.REVOLUTE.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.REVOLUTE))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )
        elif dof_type == wp.int32(JointDoFType.PRISMATIC.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.PRISMATIC))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )
        elif dof_type == wp.int32(JointDoFType.CYLINDRICAL.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.CYLINDRICAL))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )
        elif dof_type == wp.int32(JointDoFType.UNIVERSAL.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.UNIVERSAL))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )
        elif dof_type == wp.int32(JointDoFType.SPHERICAL.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.SPHERICAL))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )
        elif dof_type == wp.int32(JointDoFType.GIMBAL.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.GIMBAL))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )
        elif dof_type == wp.int32(JointDoFType.CARTESIAN.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.CARTESIAN))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )
        elif dof_type == wp.int32(JointDoFType.FIXED.value):
            wp.static(make_typed_write_joint_residuals_abs(JointDoFType.FIXED))(
                kinematic_cts_offset, j_r_j, j_q_j, j_u_j, r_cts_penetration, r_cts_velocity
            )

    return _write_joint_residuals_abs


@cache
def make_compute_joint_constraint_residuals_kernel():
    """
    Generate the kernel that computes per-joint absolute position- and
    velocity-level constraint residuals and writes them into the flat
    ``r_cts_penetration`` / ``r_cts_velocity`` buffers.
    """

    @wp.kernel
    def _compute_joint_constraint_residuals(
        # Kamino JointsModel arrays
        joint_dof_type: wp.array[wp.int32],
        joint_kinematic_cts_offset: wp.array[wp.int32],
        joint_bid_B: wp.array[wp.int32],
        joint_bid_F: wp.array[wp.int32],
        joint_B_r_Bj: wp.array[wp.vec3f],
        joint_F_r_Fj: wp.array[wp.vec3f],
        joint_X_Bj: wp.array[wp.mat33f],
        joint_X_Fj: wp.array[wp.mat33f],
        # Preprocessed Newton state
        body_q_com: wp.array[wp.transformf],  # COM-frame body poses (from state_minus)
        body_qd: wp.array[vec6f],  # COM-frame spatial velocities (from state_plus)
        # Outputs
        r_cts_penetration: wp.array[wp.float32],
        r_cts_velocity: wp.array[wp.float32],
    ):
        jid = wp.tid()

        dof_type = joint_dof_type[jid]
        bid_B = joint_bid_B[jid]
        bid_F = joint_bid_F[jid]
        B_r_Bj = joint_B_r_Bj[jid]
        F_r_Fj = joint_F_r_Fj[jid]
        X_Bj = joint_X_Bj[jid]
        X_Fj = joint_X_Fj[jid]
        kinematic_cts_offset = joint_kinematic_cts_offset[jid]

        # If the base body is the world (bid=-1) use identity pose / zero twist;
        # otherwise pull the body's COM-frame pose and spatial velocity.
        T_B_j = wp.transform_identity(dtype=wp.float32)
        u_B_j = vec6f(0.0)
        if bid_B > -1:
            T_B_j = body_q_com[bid_B]
            u_B_j = body_qd[bid_B]

        T_F_j = body_q_com[bid_F]
        u_F_j = body_qd[bid_F]

        # Compute the joint-local relative pose / twist between the two attached frames.
        _p_j, j_r_j, j_q_j, j_u_j = compute_joint_pose_and_relative_motion(
            T_B_j, T_F_j, u_B_j, u_F_j, B_r_Bj, F_r_Fj, X_Bj, X_Fj
        )

        wp.static(make_write_joint_residuals_abs())(
            dof_type,
            kinematic_cts_offset,
            j_r_j,
            j_q_j,
            j_u_j,
            r_cts_penetration,
            r_cts_velocity,
        )

    return _compute_joint_constraint_residuals


###
# Launchers
###


def compute_contact_velocities(
    model: Model,
    state: State,
    contacts: Contacts,
):
    """
    Computes the contact velocities for the given model and state.
    """
    # Ensure that the contact velocity extended attribute is present
    if contacts.velocity is None:
        raise ValueError("Contact velocity extended attribute is not present.")

    # Ensure all containers are on the same device
    if model.device != contacts.device:
        raise ValueError(
            f"Model and contacts must be on the same device but are on {model.device} and {contacts.device}."
        )
    if model.device != state.device:
        raise ValueError(f"Model and state must be on the same device but are on {model.device} and {state.device}.")

    # Launch the kernel to compute the contact velocities
    wp.launch(
        kernel=_compute_contact_velocities,
        dim=contacts.rigid_contact_max,
        inputs=[
            wp.int32(contacts.rigid_contact_max),
            model.shape_body,
            model.body_com,
            state.body_q,
            state.body_qd,
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
        ],
        outputs=[contacts.velocity],
        device=model.device,
    )


def compute_contact_constraint_metrics(
    model: Model,
    state_minus: State,
    state_plus: State,
    contacts: Contacts,
    metrics: PhysicsMetrics,
    dt: float,
):
    """
    Computes the contact constraint residuals for the given model and state.
    """
    # Ensure all containers are on the same device
    if model.device != contacts.device:
        raise ValueError(
            f"Model and contacts must be on the same device but are on {model.device} and {contacts.device}."
        )
    if model.device != state_minus.device:
        raise ValueError(
            f"Model and state_minus must be on the same device but are on {model.device} and {state_minus.device}."
        )
    if model.device != state_plus.device:
        raise ValueError(
            f"Model and state_plus must be on the same device but are on {model.device} and {state_plus.device}."
        )

    # Ensure the contacts force array is present
    if contacts.force is None:
        raise ValueError("Contacts force array is not allocated. Please request it using `request_contact_attributes`.")

    # Ensure the metrics container is present
    if metrics.contacts is None:
        raise ValueError(
            "Metrics container does not contain a `contacts` attribute. Ensure `model.rigid_contact_max` is set."
        )

    # Clear the metrics container prior to computing the contact constraint
    # residuals to avoid accumulating residuals from previous computations.
    metrics.contacts.clear()

    # Launch the kernel to compute the contact constraint residuals
    wp.launch(
        kernel=_compute_contact_constraint_metrics,
        dim=contacts.rigid_contact_max,
        inputs=[
            wp.float32(dt),
            wp.int32(contacts.rigid_contact_max),
            model.shape_body,
            model.shape_margin,
            model.shape_material_mu,
            model.shape_material_restitution,
            model.body_com,
            state_minus.body_q,
            state_minus.body_qd,
            # state_plus.body_q,
            state_plus.body_qd,
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_offset0,
            contacts.rigid_contact_offset1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.force,
        ],
        outputs=[
            metrics.contacts.r_cts_penetration,
            metrics.contacts.r_cts_velocity,
            metrics.contacts.r_ncp_primal,
            metrics.contacts.r_ncp_dual,
            metrics.contacts.r_ncp_compl,
            metrics.contacts.r_vi_natmap,
        ],
        device=model.device,
    )


def compute_per_world_contact_constraint_summary(
    model: Model,
    contacts: Contacts,
    metrics: PhysicsMetrics,
):
    """
    Reduces the per-contact constraint residuals in ``metrics.contacts`` into
    per-world max and argmax values stored in ``metrics.per_world_contacts_summary``.

    One thread is launched per contact slot. The world index is resolved from
    ``model.shape_world`` with a fallback to the partner shape so that one-sided
    static contacts (where one shape is global) still contribute to the correct
    world; contacts whose two shapes are both global are silently skipped.

    Args:
        model: The Newton model providing ``shape_world`` and
            ``rigid_contact_max`` configuration.
        contacts: The Newton contacts container providing the active contact
            count and shape pair arrays.
        metrics: The ``PhysicsMetrics`` container providing the per-contact
            residual arrays (in ``metrics.contacts``) and the per-world summary
            output arrays (in ``metrics.per_world_contacts_summary``).

    Raises:
        ValueError: If ``metrics.contacts`` or ``metrics.per_world_contacts_summary``
            is missing, or if any of the supplied containers live on a
            different device than ``model``.
    """
    if model.device != contacts.device:
        raise ValueError(
            f"Model and contacts must be on the same device but are on {model.device} and {contacts.device}."
        )
    if metrics.contacts is None:
        raise ValueError(
            "Metrics container does not contain a `contacts` attribute. Ensure `model.rigid_contact_max` is set."
        )
    if metrics.per_world_contacts_summary is None:
        raise ValueError(
            "Metrics container does not contain a `per_world_contacts_summary` attribute. "
            "Ensure `model.world_count > 0` and `model.rigid_contact_max` is set."
        )

    # Reset per-world max buffers to zero and argmax buffers to -1 so the
    # reduction starts from a known baseline. Residual values are non-negative
    # so zeroing the max preserves the global lower bound; argmax of -1
    # signals "no contact contributed to this world".
    summary = metrics.per_world_contacts_summary
    summary.clear()

    # Launch the kernel to compute the per-world contact constraint summary
    wp.launch(
        kernel=_compute_per_world_contact_metrics_summary,
        dim=contacts.rigid_contact_max,
        inputs=[
            wp.int32(contacts.rigid_contact_max),
            model.shape_world,
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            metrics.contacts.r_cts_penetration,
            metrics.contacts.r_cts_velocity,
            metrics.contacts.r_ncp_primal,
            metrics.contacts.r_ncp_dual,
            metrics.contacts.r_ncp_compl,
            metrics.contacts.r_vi_natmap,
        ],
        outputs=[
            summary.r_cts_penetration,
            summary.r_cts_penetration_argmax,
            summary.r_cts_velocity,
            summary.r_cts_velocity_argmax,
            summary.r_ncp_primal,
            summary.r_ncp_primal_argmax,
            summary.r_ncp_dual,
            summary.r_ncp_dual_argmax,
            summary.r_ncp_compl,
            summary.r_ncp_compl_argmax,
            summary.r_vi_natmap,
            summary.r_vi_natmap_argmax,
        ],
        device=model.device,
    )


def compute_per_world_joint_constraint_summary(
    model: Model,
    metrics: PhysicsMetrics,
) -> None:
    """
    Reduces the per-constraint joint kinematic residuals in ``metrics.joints``
    into per-world max and argmax values stored in
    ``metrics.per_world_joints_summary``.

    One thread is launched per joint. For each joint, the maximum position-
    and velocity-level residual over that joint's kinematic constraint block is
    atomically merged into the corresponding world. The argmax companion
    records the joint index that achieved the per-world maximum.

    Args:
        model: The Newton model providing ``joint_world``.
        metrics: The :class:`PhysicsMetrics` container providing the flat
            joint residual arrays (in ``metrics.joints``), the Kamino joint
            context cached at init, and the per-world summary output arrays
            (in ``metrics.per_world_joints_summary``).

    Raises:
        ValueError: If ``metrics.joints``, ``metrics.per_world_joints_summary``,
            or the Kamino joint context was not initialized, or if any supplied
            container lives on a different device than ``model``.
    """
    if metrics.joints is None:
        raise ValueError(
            "Metrics container does not contain a `joints` attribute. Ensure the model has joints "
            "(model.joint_count > 0) when constructing PhysicsMetrics."
        )
    if metrics.per_world_joints_summary is None:
        raise ValueError(
            "Metrics container does not contain a `per_world_joints_summary` attribute. "
            "Ensure `model.world_count > 0` and `model.joint_count > 0`."
        )
    if metrics._kamino_joints_model is None:
        raise ValueError(
            "PhysicsMetrics was not initialized with a Kamino joint context. "
            "Reconstruct PhysicsMetrics from a model with joints."
        )

    joints_model = metrics._kamino_joints_model
    summary = metrics.per_world_joints_summary
    summary.clear()

    wp.launch(
        kernel=_compute_per_world_joint_metrics_summary,
        dim=model.joint_count,
        inputs=[
            model.joint_world,
            joints_model.kinematic_cts_offset,
            metrics.joints.r_cts_penetration,
            metrics.joints.r_cts_velocity,
        ],
        outputs=[
            summary.r_cts_penetration,
            summary.r_cts_penetration_argmax,
            summary.r_cts_velocity,
            summary.r_cts_velocity_argmax,
        ],
        device=model.device,
    )


def compute_joint_constraint_metrics(
    model: Model,
    state_minus: State,
    state_plus: State,
    metrics: PhysicsMetrics,
) -> None:
    """
    Computes per-joint position- and velocity-level kinematic constraint
    residuals and stores them in ``metrics.joints``.

    The Newton joints are evaluated through a Kamino :class:`JointsModel` built
    once at :class:`PhysicsMetrics` initialization (see
    :meth:`PhysicsMetrics._init_kamino_joint_context`). For each joint, the
    relative pose / twist of the two attached frames is computed via
    :func:`compute_joint_pose_and_relative_motion` and the absolute values of
    the kinematic constraint axes are written into the flat
    ``r_cts_penetration`` / ``r_cts_velocity`` arrays at the joint's
    ``kinematic_cts_offset``. The remaining ``ConstraintMetrics`` fields
    (NCP / VI primal / dual / complementarity / natural map residuals) are not
    populated by this routine: joint forces are not generally extractable from
    a Newton ``State``, and the kinematic constraint residual alone already
    captures the dominant numerical signal of interest.

    The position-level residual is evaluated at ``state_minus`` (i.e. the
    pre-step pose) and the velocity-level residual at ``state_plus`` (i.e. the
    post-step velocity), matching Kamino's convention where the velocity
    constraint is enforced on the *outgoing* twist of an integration step.

    Args:
        model: The Newton model.
        state_minus: Pre-step state. Only ``body_q`` is consumed.
        state_plus: Post-step state. Only ``body_qd`` is consumed.
        metrics: The :class:`PhysicsMetrics` container. Must have been
            initialized from a model with ``joint_count > 0``; the Kamino
            joint context cached at init is reused on every call.

    Raises:
        ValueError: If any container lives on a different device than
            ``model``, or if the Kamino joint context was not initialized.
    """
    if model.device != state_minus.device:
        raise ValueError(
            f"Model and state_minus must be on the same device but are on {model.device} and {state_minus.device}."
        )
    if model.device != state_plus.device:
        raise ValueError(
            f"Model and state_plus must be on the same device but are on {model.device} and {state_plus.device}."
        )
    if metrics.joints is None:
        raise ValueError(
            "Metrics container does not contain a `joints` attribute. Ensure the model has joints "
            "(model.joint_count > 0) when constructing PhysicsMetrics."
        )
    if metrics._kamino_joints_model is None or metrics._kamino_body_q_com is None:
        raise ValueError(
            "PhysicsMetrics was not initialized with a Kamino joint context. "
            "Reconstruct PhysicsMetrics from a model with joints, and ensure "
            "joint_X_p.rotation == joint_X_c.rotation on every joint."
        )

    # Reset only the buffers actually written by the kernel. The other
    # ConstraintMetrics fields (NCP/VI residuals) are intentionally left
    # untouched here (they stay at their initial zero values).
    metrics.joints.r_cts_penetration.zero_()
    metrics.joints.r_cts_velocity.zero_()

    # Convert body-origin poses (Newton convention) to COM-centric poses
    # (Kamino convention). state_plus.body_qd is already COM-centric in
    # Newton, so it can be passed straight to the kernel.
    convert_body_origin_to_com(
        body_com=model.body_com,
        body_q=state_minus.body_q,
        body_q_com=metrics._kamino_body_q_com,
    )

    joints_model = metrics._kamino_joints_model
    wp.launch(
        kernel=make_compute_joint_constraint_residuals_kernel(),
        dim=model.joint_count,
        inputs=[
            joints_model.dof_type,
            joints_model.kinematic_cts_offset,
            joints_model.bid_B,
            joints_model.bid_F,
            joints_model.B_r_Bj,
            joints_model.F_r_Fj,
            joints_model.X_Bj,
            joints_model.X_Fj,
            metrics._kamino_body_q_com,
            state_plus.body_qd,
        ],
        outputs=[
            metrics.joints.r_cts_penetration,
            metrics.joints.r_cts_velocity,
        ],
        device=model.device,
    )


###
# Interfaces
###
