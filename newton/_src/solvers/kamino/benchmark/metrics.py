# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import warp as wp

from ....sim import Contacts, Model, State
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
    "compute_per_world_contact_constraint_summary",
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
        """Constraint metrics over all joints."""

        self.contacts: ConstraintMetrics | None = None
        """Constraint metrics over all active contacts."""

        self.per_world_joints_summary: ConstraintMetrics | None = None
        """Per-world max+argmax summary of the per-joint constraint metrics."""

        self.per_world_contacts_summary: ConstraintMetrics | None = None
        """Per-world max+argmax summary of the per-contact constraint metrics."""

        # Initialize the constraint metrics containers if a model is provided
        if model is not None:
            if model.joint_count > 0:
                self.joints = ConstraintMetrics(size=model.joint_count)
                self.per_world_joints_summary = ConstraintMetrics(size=model.world_count)
            else:
                msg.warning("No joints in the model, skipping joint constraint metrics.")
            if model.rigid_contact_max > 0:
                self.contacts = ConstraintMetrics(size=model.rigid_contact_max)
                self.per_world_contacts_summary = ConstraintMetrics(size=model.world_count)
            else:
                msg.warning("No contacts in the model, skipping contact constraint metrics.")

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
        x (vec3f): The input vector to be projected.
        mu (float32): The friction coefficient defining the aperture of the cone.
        epsilon (float32, optional): A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        vec3f: The vector projected onto the Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if mu * xt_norm > -xn + epsilon:
        if xt_norm <= mu * xn + epsilon:
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
        x (vec3f): The input vector to be projected.
        mu (float32): The friction coefficient defining the aperture of the cone.
        epsilon (float32, optional): A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        vec3f: The vector projected onto the dual Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if xt_norm > -mu * xn + epsilon:
        if mu * xt_norm <= xn + epsilon:
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
    body_com: wp.array[wp.vec3f],
    body_q_minun: wp.array[wp.transformf],
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

    # Retrieve the material properties for this contact
    mu_0 = shape_material_mu[sid_0]
    mu_1 = shape_material_mu[sid_1]
    mu_01 = 0.5 * (mu_0 + mu_1)

    # Make the contact frame quaternion. The associated rotation matrix has
    # columns ``[t, o, n]``, i.e. it maps contact-frame coordinates to the
    # world frame; the inverse rotation is needed below to express world-frame
    # vectors in contact-frame components.
    q_contact = make_contact_frame_znorm_quat(n_01)

    # Retrieve the body transforms for this contact
    r_body_to_com_0 = wp.vec3f(0.0)
    u_body_0 = wp.spatial_vectorf(0.0)
    X_body_0 = wp.transform_identity(dtype=wp.float32)
    if bid_0 >= 0:
        r_body_to_com_0 = body_com[bid_0]
        u_body_0 = body_qd_plus[bid_0]
        X_body_0 = body_q_minun[bid_0]
    r_body_to_com_1 = wp.vec3f(0.0)
    u_body_1 = wp.spatial_vectorf(0.0)
    X_body_1 = wp.transform_identity(dtype=wp.float32)
    if bid_1 >= 0:
        r_body_to_com_1 = body_com[bid_1]
        u_body_1 = body_qd_plus[bid_1]
        X_body_1 = body_q_minun[bid_1]

    # Retrieve the spatial contact force, i.e. wrench, applied by body1 onto
    # body0, referenced to the COM of body0, expressed in world frame
    w_10_ref_to_0 = contact_force[cid]

    # ---------------------------------------------------------
    # Contact kinematics and dynamics
    # ---------------------------------------------------------

    # Extract linear and angular parts of the associated bodies' motion
    v_body_com_0 = wp.spatial_top(u_body_0)
    v_body_com_1 = wp.spatial_top(u_body_1)
    omega_body_0 = wp.spatial_bottom(u_body_0)
    omega_body_1 = wp.spatial_bottom(u_body_1)
    wp.printf("[%d] v_body_com_0: [%.9f, %.9f, %.9f]\n", cid, v_body_com_0[0], v_body_com_0[1], v_body_com_0[2])
    wp.printf("[%d] v_body_com_1: [%.9f, %.9f, %.9f]\n", cid, v_body_com_1[0], v_body_com_1[1], v_body_com_1[2])
    wp.printf("[%d] omega_body_0: [%.9f, %.9f, %.9f]\n", cid, omega_body_0[0], omega_body_0[1], omega_body_0[2])
    wp.printf("[%d] omega_body_1: [%.9f, %.9f, %.9f]\n", cid, omega_body_1[0], omega_body_1[1], omega_body_1[2])

    # Compute the contact points in world frame
    r_c_0 = wp.transform_point(X_body_0, r_bc_0_local + r_bc_0_offset)
    r_c_1 = wp.transform_point(X_body_1, r_bc_1_local + r_bc_1_offset)
    wp.printf("[%d] r_c_0: [%.9f, %.9f, %.9f]\n", cid, r_c_0[0], r_c_0[1], r_c_0[2])
    wp.printf("[%d] r_c_1: [%.9f, %.9f, %.9f]\n", cid, r_c_1[0], r_c_1[1], r_c_1[2])

    dr_c_01 = r_c_1 - r_c_0
    wp.printf("[%d] dr_c_01: [%.9f, %.9f, %.9f]\n", cid, dr_c_01[0], dr_c_01[1], dr_c_01[2])

    # Compute the world-frame body COM positions used below as the reference
    # points for the per-body twists stored in ``body_qd``.
    r_com_0_world = wp.transform_point(X_body_0, r_body_to_com_0)
    r_com_1_world = wp.transform_point(X_body_1, r_body_to_com_1)
    wp.printf("[%d] r_com_0_world: [%.9f, %.9f, %.9f]\n", cid, r_com_0_world[0], r_com_0_world[1], r_com_0_world[2])
    wp.printf("[%d] r_com_1_world: [%.9f, %.9f, %.9f]\n", cid, r_com_1_world[0], r_com_1_world[1], r_com_1_world[2])

    # Reconstruct signed contact distance
    d_01 = wp.dot(r_c_1 - r_c_0, n_01) - (margin_0 + margin_1)
    wp.printf("[%d] d_01: %.9f\n", cid, d_01)

    # # Skip if the contact distance is positive, i.e. no penetration
    # # TODO: FIX LOGIC WRT TO MARGINS AND GAP
    # if d_01 > 0.0:
    #     return

    # Compute the velocity of the contact on each body.
    v_c_0 = v_body_com_0 + wp.cross(omega_body_0, r_c_0 - r_com_0_world)
    v_c_1 = v_body_com_1 + wp.cross(omega_body_1, r_c_1 - r_com_1_world)
    wp.printf("[%d] v_c_0: [%.9f, %.9f, %.9f]\n", cid, v_c_0[0], v_c_0[1], v_c_0[2])
    wp.printf("[%d] v_c_1: [%.9f, %.9f, %.9f]\n", cid, v_c_1[0], v_c_1[1], v_c_1[2])

    # Compute the relative velocity of the contact on each body
    v_01 = v_c_1 - v_c_0
    # TODO (torsional friction): omega_01 = omega_body_1 - omega_body_0
    wp.printf("[%d] v_01: [%.9f, %.9f, %.9f]\n", cid, v_01[0], v_01[1], v_01[2])

    # Invert the signs to represent the force applied by body0
    # onto body1 and decompose it into linear and angular parts
    f_01 = -wp.spatial_top(w_10_ref_to_0)
    # TODO (torsional friction): tau_01_ref_to_0 = -wp.spatial_bottom(w_10_ref_to_0)
    wp.printf("[%d] f_01: [%.9f, %.9f, %.9f]\n", cid, f_01[0], f_01[1], f_01[2])

    # Rotate the linear velocity and force into the contact frame.
    v_c = wp.quat_rotate_inv(q_contact, v_01)
    f_c = wp.quat_rotate_inv(q_contact, f_01)
    wp.printf("[%d] v_c: [%.9f, %.9f, %.9f]\n", cid, v_c[0], v_c[1], v_c[2])
    wp.printf("[%d] f_c: [%.9f, %.9f, %.9f]\n", cid, f_c[0], f_c[1], f_c[2])

    # Convert the contact force to an impulse
    f_c *= dt
    wp.printf("[%d] f_c (impulse): [%.9f, %.9f, %.9f]\n", cid, f_c[0], f_c[1], f_c[2])

    # Compute the De Saxce correction
    s = wp.vec3f(0.0, 0.0, mu_01 * wp.length(v_c[0:2]))
    wp.printf("[%d] s: [%.9f, %.9f, %.9f]\n", cid, s[0], s[1], s[2])

    # Compute the augmented contact velocity
    v_c_aug = v_c + s
    wp.printf("[%d] v_c_aug: [%.9f, %.9f, %.9f]\n", cid, v_c_aug[0], v_c_aug[1], v_c_aug[2])

    # ---------------------------------------------------------
    # Contact residuals
    # ---------------------------------------------------------

    # Compute the contact penetration
    r_cts_penetration = wp.abs(d_01)

    # Compute the velocity-level constraint violation.
    # The unilateral velocity constraint is ``v_n >= 0`` (separating
    # is admissible). The violation magnitude is ``max(0, -v_n)`` so
    # that admissible separation does not get reported as a residual.
    r_cts_velocity = wp.max(0.0, -wp.dot(v_01, n_01))

    # Compute the contact NCP primal
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_primal(lambda) = || lambda - P_K(lambda) ||_inf`, where `P_K()` is the
    # Euclidean projection, i.e. proximal operator, onto K, and `lambda` is the
    # vector of all constraint reactions (i.e. Lagrange multipliers).
    r_ncp_primal = infnorm3(f_c - project_to_coulomb_cone(f_c, mu_01))

    # Compute the contact NCP dual
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_dual(v_hat^+) = || v_hat^+ - P_K*(v_hat^+) ||_inf`, where `P_K*()` is
    # the Euclidean projection, i.e. proximal operator, onto K*, and `v_hat^+` is
    # the so-called augmented constraint-space velocity. The latter is defined as
    # `v_hat^+ = v^+ + Gamma(v^+)`, where `v^+ := v_f D @ lambda` is the post-event
    # constraint-space velocity, and `Gamma(v^+)` is the De Saxce correction term.
    r_ncp_dual = infnorm3(v_c_aug - project_to_coulomb_dual_cone(v_c_aug, mu_01))

    # Compute the contact NCP complementarity
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_compl(lambda) = || lambda.T @ v_hat^+ ||_inf`,
    # where `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    # and `v_hat^+` is the augmented constraint-space velocity defined above.
    r_ncp_compl = wp.abs(wp.dot(f_c, v_c_aug))
    wp.printf("[%d] r_ncp_compl: %.9f\n\n", cid, r_ncp_compl)

    # Compute the contact VI natural-map
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_vi_natmapv_hat(lambda) = || lambda - P_K(lambda - v_hat^+(lambda)) ||_inf`,
    # where `P_K()` is the Euclidean projection, i.e. proximal operator, onto K,
    # `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    # and `v_hat^+(lambda)` is the augmented constraint-space velocity defined above.
    r_vi_natmap = infnorm3(f_c - project_to_coulomb_cone(f_c - v_c_aug, mu_01))

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
    metrics.clear()

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
            model.body_com,
            state_minus.body_q,
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


###
# Interfaces
###
