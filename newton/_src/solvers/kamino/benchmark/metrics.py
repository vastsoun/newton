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

    def clear(self):
        """
        Clears the metrics container.
        """
        self.r_cts_penetration.zero_()
        self.r_cts_velocity.zero_()
        self.r_ncp_primal.zero_()
        self.r_ncp_dual.zero_()
        self.r_ncp_compl.zero_()
        self.r_vi_natmap.zero_()


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
        """Constraint metrics over all worlds, summarizing the metrics over all joints and contacts."""

        self.per_world_contacts_summary: ConstraintMetrics | None = None
        """Constraint metrics over all worlds, summarizing the metrics over all joints and contacts."""

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
""" 3D unit vector for the Z axis """

UNIT_Y = wp.constant(wp.vec3f(0.0, 1.0, 0.0))
""" 3D unit vector for the Y axis """

UNIT_Z = wp.constant(wp.vec3f(0.0, 0.0, 1.0))
""" 3D unit vector for the Z axis """

COS_PI_6 = wp.constant(0.8660254037844387)
"""Convenience constant for cos(PI / 6)"""


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
    e = wp.where(wp.abs(wp.dot(n, UNIT_X)) < COS_PI_6, UNIT_Y, UNIT_X)
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return wp.mat33f(t.x, o.x, n.x, t.y, o.y, n.y, t.z, o.z, n.z)


@wp.func
def make_contact_frame_znorm_quat(n: wp.vec3f) -> wp.quatf:
    """
    Makes a quaternion that represents the contact frame
    whose z-axis is aligned with the given unit normal vector n.

    Args:
        n: The 3D unit normal vector of the contact frame.

    Returns:
        A 3x3 rotation matrix representing the contact frame.
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


###
# Kernels
###


@wp.kernel
def _compute_contact_velocities(
    # Constants:
    rigid_contact_max: int,
    # Inputs:
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transformf],
    body_qd: wp.array[wp.spatial_vectorf],
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    # Outputs:
    contact_velocity: wp.array[wp.spatial_vectorf],
):
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

    # Retrieve the body transforms for this contact
    u_body_0 = wp.spatial_vectorf(0.0)
    X_body_0 = wp.transform_identity(dtype=wp.float32)
    if bid_0 >= 0:
        u_body_0 = body_qd[bid_0]
        X_body_0 = body_q[bid_0]
    u_body_1 = wp.spatial_vectorf(0.0)
    X_body_1 = wp.transform_identity(dtype=wp.float32)
    if bid_1 >= 0:
        u_body_1 = body_qd[bid_1]
        X_body_0 = body_q[bid_1]

    # Extract linear and angular parts
    r_body_0 = wp.transform_get_translation(X_body_0)
    r_body_1 = wp.transform_get_translation(X_body_1)
    v_body_1 = wp.spatial_top(u_body_1)
    omega_body_1 = wp.spatial_bottom(u_body_1)

    # Compute the spatial twist of body1 with respect to body0 COM reference point
    v_1_ref_to_0 = v_body_1 + wp.cross(omega_body_1, r_body_1 - r_body_0)
    u_1_ref_to_0 = wp.spatial_vector(v_1_ref_to_0, omega_body_1)

    # Compute the relative spatial twist of body1 with
    # respect to body0 at the body0 COM reference point
    u_01_ref_to_0 = u_1_ref_to_0 - u_body_0

    # Store the relative twist of the contacting
    # bodies as the contact spatial twist
    contact_velocity[cid] = u_01_ref_to_0


@wp.kernel
def _compute_contact_constraint_metrics(
    # Inputs:
    rigid_contact_max: wp.int32,
    shape_body: wp.array[wp.int32],
    shape_margin: wp.array[wp.float32],
    shape_material_mu: wp.array[wp.float32],
    body_com: wp.array[wp.vec3f],
    body_q: wp.array[wp.transformf],
    body_qd: wp.array[wp.spatial_vectorf],
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
    n_01 = contact_normal[cid]

    # Retrieve the material properties for this contact
    mu_0 = shape_material_mu[sid_0]
    mu_1 = shape_material_mu[sid_1]
    mu_01 = 0.5 * (mu_0 + mu_1)

    # Make the contact frame quaternion
    q_contact = make_contact_frame_znorm_quat(n_01)

    # Retrieve the body transforms for this contact
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

    # Retrieve the spatial contact force, i.e. wrench, applied by body1 onto
    # body0, referenced to the COM of body0, expressed in world frame
    w_10_ref_to_0 = contact_force[cid]

    # ---------------------------------------------------------
    # Contact kinematics and dynamics
    # ---------------------------------------------------------

    # Extract linear and angular parts of the associated bodies' motion
    q_body_0 = wp.transform_get_rotation(X_body_0)
    q_body_1 = wp.transform_get_rotation(X_body_1)
    v_body_com_0 = wp.spatial_top(u_body_0)
    v_body_com_1 = wp.spatial_top(u_body_1)
    omega_body_0 = wp.spatial_bottom(u_body_0)
    omega_body_1 = wp.spatial_bottom(u_body_1)

    # Convert the local contact points to world frame
    r_bc_0 = wp.quat_rotate(q_body_0, r_bc_0_local + r_bc_0_offset)
    r_bc_1 = wp.quat_rotate(q_body_1, r_bc_1_local + r_bc_1_offset)

    # Compute the contact points in world frame
    r_c_0 = wp.transform_point(X_body_0, r_bc_0_local + r_bc_0_offset)
    r_c_1 = wp.transform_point(X_body_1, r_bc_1_local + r_bc_1_offset)

    # Reconstruct signed contact distance
    d_01 = wp.dot(r_c_1 - r_c_0, n_01) - (margin_0 + margin_1)

    # Skip if the contact distance is positive, i.e. no penetration
    if d_01 > 0.0:
        return

    # Compute the velocity of the contact on each body
    v_c_0 = v_body_com_0 + wp.cross(omega_body_0, r_bc_0 - r_body_to_com_0)
    v_c_1 = v_body_com_1 + wp.cross(omega_body_1, r_bc_1 - r_body_to_com_1)

    # Compute the relative velocity of the contact on each body
    v_01 = v_c_1 - v_c_0
    # TODO (torsional friction): omega_01 = omega_body_1 - omega_body_0

    # Invert the signs to represent the force applied by body0
    # onto body1 and decompose it into linear and angular parts
    f_01 = -wp.spatial_top(w_10_ref_to_0)
    # TODO (torsional friction): tau_01_ref_to_0 = -wp.spatial_bottom(w_10_ref_to_0)

    # Rotate the linear velocity and force into the contact frame
    v_c = wp.quat_rotate(q_contact, v_01)
    f_c = wp.quat_rotate(q_contact, f_01)

    # Compute the De Saxce correction
    s = wp.vec3f(0.0, 0.0, mu_01 * wp.length(v_c[0:2]))

    # Compute the augmented contact velocity
    v_c_aug = v_c + s

    # ---------------------------------------------------------
    # Contact residuals
    # ---------------------------------------------------------

    # Compute the contact penetration
    r_cts_penetration = wp.abs(d_01)

    # Compute the contact velocity
    r_cts_velocity = wp.abs(wp.dot(v_01, n_01))

    # Compute the contact NCP primal
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_primal(lambda) = || lambda - P_K(lambda) ||_inf`, where `P_K()` is the
    # Euclidean projection, i.e. proximal operator, onto K, and `lambda` is the
    # vector of all constraint reactions (i.e. Lagrange multipliers).
    r_ncp_primal = wp.norm_l1(f_c - project_to_coulomb_cone(f_c, mu_01))

    # Compute the contact NCP dual
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_dual(v_hat^+) = || v_hat^+ - P_K*(v_hat^+) ||_inf`, where `P_K*()` is
    # the Euclidean projection, i.e. proximal operator, onto K*, and `v_hat^+` is
    # the so-called augmented constraint-space velocity. The latter is defined as
    # `v_hat^+ = v^+ + Gamma(v^+)`, where `v^+ := v_f D @ lambda` is the post-event
    # constraint-space velocity, and `Gamma(v^+)` is the De Saxce correction term.
    r_ncp_dual = wp.norm_l1(v_c_aug - project_to_coulomb_dual_cone(v_c_aug, mu_01))

    # Compute the contact NCP complementarity
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_ncp_compl(lambda) = || lambda.T @ v_hat^+ ||_inf`,
    # where `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    # and `v_hat^+` is the augmented constraint-space velocity defined above.
    r_ncp_compl = wp.abs(wp.dot(f_c, v_c_aug))

    # Compute the contact VI natural-map
    # Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    # `r_vi_natmapv_hat(lambda) = || lambda - P_K*(lambda - v_hat^+(lambda)) ||_inf`,
    # where `P_K*()` is the Euclidean projection, i.e. proximal operator, onto K*,
    # `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    # and `v_hat^+(lambda)` is the augmented constraint-space velocity defined above.
    r_vi_natmap = wp.norm_l1(f_c - project_to_coulomb_cone(f_c - v_c_aug, mu_01))

    # Store the contact residuals
    r_contact_cts_penetration[cid] = r_cts_penetration
    r_contact_cts_velocity[cid] = r_cts_velocity
    r_contact_ncp_primal[cid] = r_ncp_primal
    r_contact_ncp_dual[cid] = r_ncp_dual
    r_contact_ncp_compl[cid] = r_ncp_compl
    r_contact_vi_natmap[cid] = r_vi_natmap


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
    state: State,
    contacts: Contacts,
    metrics: PhysicsMetrics,
):
    """
    Computes the contact constraint residuals for the given model and state.
    """
    # Ensure all containers are on the same device
    if model.device != contacts.device:
        raise ValueError(
            f"Model and contacts must be on the same device but are on {model.device} and {contacts.device}."
        )
    if model.device != state.device:
        raise ValueError(f"Model and state must be on the same device but are on {model.device} and {state.device}.")

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
            wp.int32(contacts.rigid_contact_max),
            model.shape_body,
            model.shape_margin,
            model.shape_material_mu,
            model.body_com,
            state.body_q,
            state.body_qd,
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


###
# Interfaces
###
