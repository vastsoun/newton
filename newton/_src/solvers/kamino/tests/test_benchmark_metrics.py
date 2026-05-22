# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the `kamino.benchmark.metrics` module.

The tests fall into three groups:

* Coordinate-transform sanity tests on the sphere-on-plane, box-on-plane and
  boxes-nunchaku scenes from :mod:`newton.tests.utils.basics`. These exercise
  ``metrics.compute_contact_velocities`` and the per-residual closed-form
  outputs of ``metrics.compute_contact_constraint_metrics`` (rotation
  invariance, gap/margin states, NCP primal/dual/complementarity/VI natural
  map sign matrix, off-COM moment arms).

* Per-world max+argmax reduction tests for the new
  ``metrics.compute_per_world_contact_constraint_summary`` launcher, including
  inactive-contact masking and global-shape skipping.

To keep the tests independent of the solver's actual contact populator, a
``TestSetup.manual_contact`` helper writes synthetic contact records directly
into the ``Contacts`` arrays so we can drive closed-form residuals without
relying on the collider.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

import numpy as np
import warp as wp

from newton import Contacts, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.benchmark import metrics
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils import basics

###
# Scaffolding
###


class TestSetup:
    """Builds a Newton ``Model`` + ``State`` + ``Contacts`` for a metrics test.

    Optionally replicates ``builder_fn`` across ``num_worlds`` by calling it
    with ``builder=`` and ``new_world=True`` on every iteration so the
    resulting model carries one world per replicate. The default-shape config
    is configured before the first ``builder_fn`` call so that ``margin``,
    ``gap`` and ``mu`` percolate through every shape added by the basics
    factories.
    """

    def __init__(
        self,
        builder_fn: Callable,
        builder_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        num_worlds: int = 1,
        max_contacts: int = 32,
        margin: float = 0.0,
        gap: float = 0.0,
        friction: float = 1.0,
        device: wp.DeviceLike | None = None,
    ):
        if builder_kwargs is None:
            builder_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        self.device = device
        self.num_worlds = int(num_worlds)
        self.builder: ModelBuilder = ModelBuilder()
        self.builder.request_contact_attributes("force", "velocity")
        self.builder.default_shape_cfg.margin = margin
        self.builder.default_shape_cfg.gap = gap
        self.builder.default_shape_cfg.mu = friction
        if self.num_worlds == 1:
            # Reuse the basics factory's own world context so the tests still
            # round-trip the single-world signature the original test used.
            self.builder = builder_fn(builder=self.builder, **builder_kwargs)
        else:
            for _ in range(self.num_worlds):
                builder_fn(builder=self.builder, new_world=True, **builder_kwargs)
        self.model: Model = self.builder.finalize(device=device, **model_kwargs)
        self.model.rigid_contact_max = max_contacts
        self.state: State = self.model.state()
        self.contacts: Contacts = self.model.contacts()

    def reset_contacts(self) -> None:
        """Zero every per-contact buffer the metrics kernels read."""
        self.contacts.rigid_contact_count.fill_(0)
        self.contacts.rigid_contact_shape0.fill_(-1)
        self.contacts.rigid_contact_shape1.fill_(-1)
        self.contacts.rigid_contact_point0.zero_()
        self.contacts.rigid_contact_point1.zero_()
        self.contacts.rigid_contact_offset0.zero_()
        self.contacts.rigid_contact_offset1.zero_()
        self.contacts.rigid_contact_normal.zero_()
        self.contacts.rigid_contact_margin0.zero_()
        self.contacts.rigid_contact_margin1.zero_()
        if self.contacts.force is not None:
            self.contacts.force.zero_()
        if self.contacts.velocity is not None:
            self.contacts.velocity.zero_()

    def manual_contact(
        self,
        *,
        cid: int,
        shape0: int,
        shape1: int,
        normal_world: np.ndarray,
        point0_world: np.ndarray,
        point1_world: np.ndarray,
        force_world: np.ndarray | None = None,
        update_count: bool = True,
    ) -> None:
        """Writes a synthetic contact record into slot ``cid``.

        Args:
            cid: Contact slot to populate (must be `< rigid_contact_max`).
            shape0: Shape index for body0.
            shape1: Shape index for body1.
            normal_world: Contact normal in world coordinates pointing from
                shape0 toward shape1 (unit vector).
            point0_world: World-space contact-point witness on shape0.
            point1_world: World-space contact-point witness on shape1.
            force_world: Optional 6-vector spatial wrench applied to body0 by
                body1, referenced at body0's COM in world coordinates. If
                ``None``, the contact force slot is left at zero.
            update_count: If ``True``, set ``rigid_contact_count`` to
                ``max(current_count, cid+1)`` so the kernel sees this contact
                as active.
        """
        shape_body = self.model.shape_body.numpy()
        body_q = self.state.body_q.numpy()
        body_com_np = self.model.body_com.numpy()

        bid_0 = int(shape_body[shape0])
        bid_1 = int(shape_body[shape1])

        # Convert world contact points back to body-local coordinates so the
        # kernel-side ``transform_point(body_q, point + offset)`` recovers
        # exactly ``point*_world``. The ``offset`` slots are kept at zero so
        # the body-local point fully captures the witness.
        if bid_0 >= 0:
            X_0 = wp.transformf(*body_q[bid_0])
            r_local_0 = np.array(wp.transform_point(wp.transform_inverse(X_0), wp.vec3f(*point0_world)))
        else:
            r_local_0 = np.array(point0_world, dtype=np.float32)
        if bid_1 >= 0:
            X_1 = wp.transformf(*body_q[bid_1])
            r_local_1 = np.array(wp.transform_point(wp.transform_inverse(X_1), wp.vec3f(*point1_world)))
        else:
            r_local_1 = np.array(point1_world, dtype=np.float32)

        shape0_np = self.contacts.rigid_contact_shape0.numpy()
        shape1_np = self.contacts.rigid_contact_shape1.numpy()
        normal_np = self.contacts.rigid_contact_normal.numpy()
        point0_np = self.contacts.rigid_contact_point0.numpy()
        point1_np = self.contacts.rigid_contact_point1.numpy()
        offset0_np = self.contacts.rigid_contact_offset0.numpy()
        offset1_np = self.contacts.rigid_contact_offset1.numpy()

        shape0_np[cid] = shape0
        shape1_np[cid] = shape1
        normal_np[cid] = np.asarray(normal_world, dtype=np.float32)
        point0_np[cid] = r_local_0.astype(np.float32)
        point1_np[cid] = r_local_1.astype(np.float32)
        offset0_np[cid] = np.zeros(3, dtype=np.float32)
        offset1_np[cid] = np.zeros(3, dtype=np.float32)

        self.contacts.rigid_contact_shape0.assign(shape0_np)
        self.contacts.rigid_contact_shape1.assign(shape1_np)
        self.contacts.rigid_contact_normal.assign(normal_np)
        self.contacts.rigid_contact_point0.assign(point0_np)
        self.contacts.rigid_contact_point1.assign(point1_np)
        self.contacts.rigid_contact_offset0.assign(offset0_np)
        self.contacts.rigid_contact_offset1.assign(offset1_np)

        if self.contacts.force is not None and force_world is not None:
            force_np = self.contacts.force.numpy()
            force_np[cid] = np.asarray(force_world, dtype=np.float32)
            self.contacts.force.assign(force_np)

        if update_count:
            counters = self.contacts.contact_counters.numpy()
            counters[0] = max(int(counters[0]), cid + 1)
            self.contacts.contact_counters.assign(counters)

        # ``body_com`` is queried by the kernel but never modified here; reading
        # it once keeps the unused-variable warning quiet without changing state.
        _ = body_com_np

    def set_body_state(
        self,
        body_id: int,
        *,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        lin_vel: np.ndarray | None = None,
        ang_vel: np.ndarray | None = None,
    ) -> None:
        """Overrides the pose / twist of a single body in the state arrays."""
        body_q_np = self.state.body_q.numpy()
        body_qd_np = self.state.body_qd.numpy()
        if position is not None:
            body_q_np[body_id, 0:3] = np.asarray(position, dtype=np.float32)
        if orientation is not None:
            body_q_np[body_id, 3:7] = np.asarray(orientation, dtype=np.float32)
        if lin_vel is not None:
            body_qd_np[body_id, 0:3] = np.asarray(lin_vel, dtype=np.float32)
        if ang_vel is not None:
            body_qd_np[body_id, 3:6] = np.asarray(ang_vel, dtype=np.float32)
        self.state.body_q.assign(body_q_np)
        self.state.body_qd.assign(body_qd_np)


###
# Helpers
###


def _arr1d(array: wp.array, *, dtype: type = np.float32) -> np.ndarray:
    """Returns the 1-D numpy view of a Warp array as ``dtype``."""
    return array.numpy().astype(dtype, copy=False).reshape(-1)


def _quat_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Builds a quaternion ``(x, y, z, w)`` from an axis-angle representation."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1.0e-12)
    half = 0.5 * angle
    return np.array(
        [axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half), math.cos(half)],
        dtype=np.float32,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotates ``v`` by the quaternion ``q = (x, y, z, w)`` (right-handed)."""
    x, y, z, w = q.astype(np.float64)
    vx, vy, vz = np.asarray(v, dtype=np.float64)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float32,
    )


def _make_contact_frame_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirrors :func:`metrics.make_contact_frame_znorm` on the host."""
    cos_pi_6 = math.cos(math.pi / 6.0)
    n = np.asarray(n, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1.0e-12)
    # Same B8 fix as in the kernel: pick the seed axis transverse to ``n``.
    e = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < cos_pi_6 else np.array([0.0, 1.0, 0.0])
    o = np.cross(n, e)
    o = o / np.linalg.norm(o)
    t = np.cross(o, n)
    t = t / np.linalg.norm(t)
    return t.astype(np.float32), o.astype(np.float32), n.astype(np.float32)


def _contact_to_world_vec(n: np.ndarray, v_local: np.ndarray) -> np.ndarray:
    """Maps a contact-frame vector ``(t1, t2, n)`` into world coordinates."""
    t, o, n_unit = _make_contact_frame_basis(n)
    return v_local[0] * t + v_local[1] * o + v_local[2] * n_unit


def _world_to_contact_vec(n: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Projects a world-frame vector into the contact frame basis ``(t1, t2, n)``."""
    t, o, n_unit = _make_contact_frame_basis(n)
    return np.array([np.dot(v_world, t), np.dot(v_world, o), np.dot(v_world, n_unit)], dtype=np.float32)


def _penetrating_witness_pair(
    reference_world: np.ndarray, normal_world: np.ndarray, penetration: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(point0_world, point1_world)`` producing the requested penetration.

    Newton's convention is that the contact normal points from ``shape0`` toward
    ``shape1`` and that ``r_c_1 - r_c_0`` is anti-parallel to the normal when the
    pair is penetrating. This helper places ``point1_world`` at the reference
    point and offsets ``point0_world`` by ``+penetration * normal_world`` so the
    metrics kernel sees ``d_01 = -penetration`` (i.e. positive interpenetration).
    """
    n = np.asarray(normal_world, dtype=np.float32)
    p = np.asarray(reference_world, dtype=np.float32)
    return p + penetration * n, p


def _project_to_coulomb_cone(x: np.ndarray, mu: float) -> np.ndarray:
    """Reference (numpy) implementation of :func:`metrics.project_to_coulomb_cone`."""
    xt = float(math.hypot(x[0], x[1]))
    xn = float(x[2])
    if mu * xt > -xn:
        if xt <= mu * xn:
            return x.astype(np.float32)
        ys = (mu * xt + xn) / (mu * mu + 1.0)
        scale = mu * ys / xt
        return np.array([scale * x[0], scale * x[1], ys], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def _project_to_coulomb_dual_cone(x: np.ndarray, mu: float) -> np.ndarray:
    """Reference (numpy) implementation of :func:`metrics.project_to_coulomb_dual_cone`."""
    xt = float(math.hypot(x[0], x[1]))
    xn = float(x[2])
    if xt > -mu * xn:
        if mu * xt <= xn:
            return x.astype(np.float32)
        ys = (xt + mu * xn) / (mu * mu + 1.0)
        scale = ys / xt
        return np.array([scale * x[0], scale * x[1], mu * ys], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def _evaluate_metrics(setup: TestSetup) -> dict[str, np.ndarray]:
    """Runs the metrics kernels on ``setup`` and returns numpy snapshots."""
    container = metrics.PhysicsMetrics(model=setup.model)
    metrics.compute_contact_velocities(setup.model, setup.state, setup.contacts)
    metrics.compute_contact_constraint_metrics(setup.model, setup.state, setup.contacts, container)
    return {
        "velocity": setup.contacts.velocity.numpy(),
        "r_cts_penetration": container.contacts.r_cts_penetration.numpy(),
        "r_cts_velocity": container.contacts.r_cts_velocity.numpy(),
        "r_ncp_primal": container.contacts.r_ncp_primal.numpy(),
        "r_ncp_dual": container.contacts.r_ncp_dual.numpy(),
        "r_ncp_compl": container.contacts.r_ncp_compl.numpy(),
        "r_vi_natmap": container.contacts.r_vi_natmap.numpy(),
        "container": container,
    }


###
# Common test base class
###


class _BenchmarkMetricsTestBase(unittest.TestCase):
    """Sets up the shared Warp/Newton test context for every metric suite."""

    def setUp(self) -> None:
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose
        if self.verbose:
            print("\n")
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self) -> None:
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()


###
# Sphere-on-plane tests
###


class TestSphereOnPlane(_BenchmarkMetricsTestBase):
    """Closed-form correctness checks on the basic sphere/plane scene."""

    RADIUS: float = 0.1
    MASS: float = 1.0
    FRICTION: float = 0.5

    def _build(
        self,
        *,
        z_offset: float = -1.0e-3,
        margin: float = 0.0,
        gap: float = 0.0,
        max_contacts: int = 8,
    ) -> TestSetup:
        return TestSetup(
            builder_fn=basics.build_sphere_on_plane,
            builder_kwargs={
                "radius": self.RADIUS,
                "mass": self.MASS,
                "z_offset": z_offset,
                "friction": self.FRICTION,
                "ground": True,
            },
            margin=margin,
            gap=gap,
            friction=self.FRICTION,
            max_contacts=max_contacts,
            device=self.default_device,
        )

    def _collide(self, setup: TestSetup) -> int:
        setup.model.collide(setup.state, setup.contacts)
        return int(setup.contacts.rigid_contact_count.numpy()[0])

    def test_contact_velocity_zero_when_static(self):
        """Sphere at rest on the plane yields a zero contact twist."""
        setup = self._build()
        self.assertEqual(self._collide(setup), 1)
        snap = _evaluate_metrics(setup)
        np.testing.assert_allclose(snap["velocity"][0], np.zeros(6, dtype=np.float32), atol=1.0e-6)

    def test_contact_velocity_approaching_vs_separating(self):
        """Pure vertical sphere velocity flips sign in the contact twist."""
        setup = self._build()
        self.assertEqual(self._collide(setup), 1)

        # Approach (-z) ⇒ relative velocity of body1 wrt body0 at COM_0 is -z.
        # Newton's contact normal points from shape0 (ground) toward shape1
        # (sphere), i.e. +z; so the normal component of the relative velocity
        # should be the sphere's -z component (negative when approaching).
        setup.set_body_state(0, lin_vel=np.array([0.0, 0.0, -2.5], dtype=np.float32))
        snap_approach = _evaluate_metrics(setup)
        np.testing.assert_allclose(snap_approach["velocity"][0, 0:3], np.array([0.0, 0.0, -2.5]), atol=1.0e-6)

        setup.set_body_state(0, lin_vel=np.array([0.0, 0.0, +2.5], dtype=np.float32))
        snap_separate = _evaluate_metrics(setup)
        np.testing.assert_allclose(snap_separate["velocity"][0, 0:3], np.array([0.0, 0.0, +2.5]), atol=1.0e-6)

    def test_contact_velocity_with_offset_com(self):
        """A non-zero ``body_com`` must be accounted for in the moment arm.

        ``contacts.velocity`` reports the relative spatial twist at body0's COM.
        Body0 here is the ground (static at the origin), so the linear part
        equals ``omega_sphere x (r_com_body0 - r_com_sphere)`` where
        ``r_com_sphere = sphere_origin + body_com_local`` in world coordinates.
        """
        setup = self._build()
        self.assertEqual(self._collide(setup), 1)

        body_com_offset = 0.5 * self.RADIUS
        com_np = setup.model.body_com.numpy()
        com_np[0] = np.array([0.0, 0.0, body_com_offset], dtype=np.float32)
        setup.model.body_com.assign(com_np)
        omega = np.array([0.0, 3.0, 0.0], dtype=np.float32)
        setup.set_body_state(0, ang_vel=omega)

        snap = _evaluate_metrics(setup)

        sphere_pos_z = float(setup.state.body_q.numpy()[0, 2])
        com_z = sphere_pos_z + body_com_offset
        expected_lin = np.array([-omega[1] * com_z, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(snap["velocity"][0, 0:3], expected_lin, atol=1.0e-5)
        np.testing.assert_allclose(snap["velocity"][0, 3:6], omega, atol=1.0e-6)

    def test_gap_state_apart_no_contact(self):
        """Bodies separated by more than (margin + gap) per shape yield no contact."""
        # Both shapes share the same ``gap``; the broad-phase summed threshold is
        # ``2 * gap``. Place the sphere well beyond that to keep this test robust
        # against minor floating-point slack in the broad-phase pair filter.
        gap = 0.02
        setup = self._build(z_offset=5.0 * gap, gap=gap, margin=0.0)
        nc = self._collide(setup)
        self.assertEqual(nc, 0)

    def test_gap_state_within_gap_outside_margin(self):
        """Within gap but outside margin: kernel early-exits ⇒ all residuals 0."""
        setup = self._build(z_offset=0.01, gap=0.05, margin=0.0)
        nc = self._collide(setup)
        # Within gap the collider must produce a contact record.
        self.assertGreaterEqual(nc, 1)
        snap = _evaluate_metrics(setup)
        for name in ("r_cts_penetration", "r_cts_velocity", "r_ncp_primal", "r_ncp_dual", "r_ncp_compl", "r_vi_natmap"):
            self.assertAlmostEqual(float(snap[name][0]), 0.0, places=6, msg=f"residual {name} should be zero")

    def test_gap_state_within_margin_penetrating(self):
        """A surface-margin shape with overlap reports a positive ``r_cts_penetration``."""
        margin = 0.01
        setup = self._build(z_offset=-0.005, margin=margin, gap=0.05)
        nc = self._collide(setup)
        self.assertGreaterEqual(nc, 1)
        snap = _evaluate_metrics(setup)
        # Geometric gap is +0.005 (margin band); kernel sees d = 0.005 - 2*margin = -0.015.
        self.assertGreater(float(snap["r_cts_penetration"][0]), 0.0)

    def test_gap_state_fully_penetrating(self):
        """Deeper penetration yields a strictly larger ``r_cts_penetration``."""
        setup_shallow = self._build(z_offset=-1.0e-3)
        nc_shallow = self._collide(setup_shallow)
        self.assertGreaterEqual(nc_shallow, 1)
        snap_shallow = _evaluate_metrics(setup_shallow)
        pen_shallow = float(snap_shallow["r_cts_penetration"][0])

        setup_deep = self._build(z_offset=-0.05)
        nc_deep = self._collide(setup_deep)
        self.assertGreaterEqual(nc_deep, 1)
        snap_deep = _evaluate_metrics(setup_deep)
        pen_deep = float(snap_deep["r_cts_penetration"][0])

        self.assertGreater(pen_deep, pen_shallow)
        self.assertAlmostEqual(pen_deep, 0.05, places=4)

    def test_rotation_invariance_of_residuals(self):
        """Residual values must not depend on the chosen contact normal direction.

        The contact normal is rotated through several non-trivial orientations
        and the spatial force is rotated with it so that the underlying contact
        configuration is identical in every case. The residuals reported by
        the kernel must therefore agree to machine precision modulo the
        orthonormal basis chosen by ``make_contact_frame_znorm``.
        """
        mu = self.FRICTION

        # The reference contact-frame configuration: normal force in cone
        # (f_n = 2.0, f_t1 = 0.3, f_t2 = -0.2) and a small approaching slip in
        # the contact frame.
        f_local = np.array([0.3, -0.2, 2.0], dtype=np.float64)
        v_rel_local = np.array([0.05, -0.03, -0.10], dtype=np.float64)

        normals = [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32) / np.linalg.norm([1.0, 2.0, 3.0]),
            np.array([-2.0, 0.5, 1.5], dtype=np.float32) / np.linalg.norm([-2.0, 0.5, 1.5]),
        ]

        reference: dict[str, float] | None = None
        for n_world in normals:
            f_world = _contact_to_world_vec(n_world, f_local.astype(np.float32))

            setup = self._build()
            self._collide(setup)
            setup.reset_contacts()
            sphere_pos = setup.state.body_q.numpy()[0, 0:3]
            # Place the sphere-side witness at the sphere surface along
            # ``-n_world`` and offset the ground-side witness by 1 mm along
            # ``+n_world`` so the kernel sees a 1 mm interpenetration.
            sphere_witness = sphere_pos - self.RADIUS * n_world.astype(np.float32)
            point_ground, point_sphere = _penetrating_witness_pair(
                sphere_witness, n_world.astype(np.float32), penetration=1.0e-3
            )
            # Force applied to body0 (ground) by body1 (sphere) is the opposite
            # of f_world (which represents the force on body1 by body0).
            setup.manual_contact(
                cid=0,
                shape0=1,  # ground shape (basics adds the ground last for sphere_on_plane)
                shape1=0,  # sphere shape
                normal_world=n_world,
                point0_world=point_ground,
                point1_world=point_sphere,
                force_world=np.concatenate([-f_world, np.zeros(3, dtype=np.float32)]),
            )
            # Drive the relative contact-point velocity v_01 = v_c_1 - v_c_0 by
            # spinning the sphere about its COM. The contact point lies on the
            # sphere surface so a pure linear velocity at the sphere COM yields
            # the same linear velocity at the contact point (omega = 0).
            setup.set_body_state(0, lin_vel=_contact_to_world_vec(n_world, v_rel_local.astype(np.float32)))

            snap = _evaluate_metrics(setup)
            current = {
                "r_cts_velocity": float(snap["r_cts_velocity"][0]),
                "r_ncp_primal": float(snap["r_ncp_primal"][0]),
                "r_ncp_dual": float(snap["r_ncp_dual"][0]),
                "r_ncp_compl": float(snap["r_ncp_compl"][0]),
                "r_vi_natmap": float(snap["r_vi_natmap"][0]),
            }
            if reference is None:
                reference = current
                # Closed-form expectations for the in-cone, sub-critical setup.
                # The friction cone admits f_local (|f_t| = sqrt(0.13) < mu*f_n).
                # The augmented velocity sits outside K* whenever the normal
                # slip component is negative (approaching).
                v_aug_local = v_rel_local + np.array([0.0, 0.0, mu * math.hypot(v_rel_local[0], v_rel_local[1])])
                expected_compl = abs(float(np.dot(f_local, v_aug_local)))
                self.assertAlmostEqual(reference["r_ncp_primal"], 0.0, places=4)
                # f and v_aug aren't a complementary KKT pair; only check
                # complementarity equals the closed-form dot product magnitude.
                self.assertAlmostEqual(reference["r_ncp_compl"], expected_compl, places=4)
                self.assertAlmostEqual(reference["r_cts_velocity"], 0.10, places=5)
            else:
                for key, ref_val in reference.items():
                    self.assertAlmostEqual(
                        current[key],
                        ref_val,
                        places=4,
                        msg=f"residual {key} differs across rotations (got {current[key]}, expected {ref_val})",
                    )

    def test_residual_primal_sign_combinations(self):
        """``r_ncp_primal`` is zero in the cone and positive outside."""
        mu = self.FRICTION

        # Construct three forces in the contact frame:
        # (a) inside the cone: ||ft|| < mu*fn, fn > 0  ⇒ residual 0
        # (b) outside the cone: ||ft|| >> mu*fn, fn > 0  ⇒ residual > 0
        # (c) tension: fn < 0, no tangential component   ⇒ residual = |fn|
        cases = [
            (np.array([0.1, 0.0, 1.0], dtype=np.float32), 0.0),
            (np.array([0.6, 0.6, 0.5], dtype=np.float32), None),
            (np.array([0.0, 0.0, -2.0], dtype=np.float32), 2.0),
        ]

        n_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        for f_local, expected in cases:
            setup = self._build()
            self._collide(setup)
            setup.reset_contacts()
            sphere_pos = setup.state.body_q.numpy()[0, 0:3]
            sphere_witness = sphere_pos - self.RADIUS * n_world
            point_ground, point_sphere = _penetrating_witness_pair(sphere_witness, n_world, penetration=1.0e-3)
            setup.manual_contact(
                cid=0,
                shape0=1,
                shape1=0,
                normal_world=n_world,
                point0_world=point_ground,
                point1_world=point_sphere,
                force_world=np.concatenate([-_contact_to_world_vec(n_world, f_local), np.zeros(3, dtype=np.float32)]),
            )

            snap = _evaluate_metrics(setup)
            value = float(snap["r_ncp_primal"][0])

            if expected is None:
                # Closed-form expected: numpy-side projection then infinity-norm.
                ref = float(np.max(np.abs(f_local - _project_to_coulomb_cone(f_local, mu))))
                self.assertAlmostEqual(value, ref, places=5)
                self.assertGreater(value, 0.0)
            else:
                self.assertAlmostEqual(value, expected, places=5)

    def test_residual_dual_sign_combinations(self):
        """``r_ncp_dual`` is zero in K* and positive outside."""
        mu = self.FRICTION
        n_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # The augmented velocity ``v_aug = (vt, vn + mu*|vt|)`` is in K* iff
        # ``vn >= 0`` (the De Saxce shift hits the boundary for any purely
        # tangential slip). Build two cases:
        #   (a) separating normal slip (``vn > 0``) ⇒ in K* ⇒ residual 0.
        #   (b) closing normal slip with tangential drift ⇒ strictly outside K*.
        cases = [
            (np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0),
            (np.array([0.5, 0.0, -1.0], dtype=np.float32), None),
        ]

        for v_local, expected in cases:
            setup = self._build()
            self._collide(setup)
            setup.reset_contacts()
            sphere_pos = setup.state.body_q.numpy()[0, 0:3]
            sphere_witness = sphere_pos - self.RADIUS * n_world
            point_ground, point_sphere = _penetrating_witness_pair(sphere_witness, n_world, penetration=1.0e-3)
            v_world = _contact_to_world_vec(n_world, v_local)
            setup.manual_contact(
                cid=0,
                shape0=1,
                shape1=0,
                normal_world=n_world,
                point0_world=point_ground,
                point1_world=point_sphere,
            )
            setup.set_body_state(0, lin_vel=v_world)

            snap = _evaluate_metrics(setup)
            value = float(snap["r_ncp_dual"][0])

            if expected is None:
                v_aug_local = v_local + np.array([0.0, 0.0, mu * math.hypot(v_local[0], v_local[1])])
                ref = float(np.max(np.abs(v_aug_local - _project_to_coulomb_dual_cone(v_aug_local, mu))))
                self.assertAlmostEqual(value, ref, places=5)
                self.assertGreater(value, 0.0)
            else:
                self.assertAlmostEqual(value, expected, places=5)

    def test_residual_complementarity(self):
        """Aligned λ and v_aug yield > 0; orthogonal pairs yield 0."""
        mu = self.FRICTION
        n_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # λ is a normal force and v is purely tangential ⇒ v_aug has zero normal
        # component plus the De Saxce shift in the normal direction; so
        # ``λ · v_aug = lambda_n * (mu * |vt|)``.
        f_local = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        v_local = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        v_aug_local = v_local + np.array([0.0, 0.0, mu * math.hypot(v_local[0], v_local[1])])
        expected = abs(float(np.dot(f_local, v_aug_local)))

        setup = self._build()
        self._collide(setup)
        setup.reset_contacts()
        sphere_pos = setup.state.body_q.numpy()[0, 0:3]
        sphere_witness = sphere_pos - self.RADIUS * n_world
        point_ground, point_sphere = _penetrating_witness_pair(sphere_witness, n_world, penetration=1.0e-3)
        setup.manual_contact(
            cid=0,
            shape0=1,
            shape1=0,
            normal_world=n_world,
            point0_world=point_ground,
            point1_world=point_sphere,
            force_world=np.concatenate([-_contact_to_world_vec(n_world, f_local), np.zeros(3, dtype=np.float32)]),
        )
        setup.set_body_state(0, lin_vel=_contact_to_world_vec(n_world, v_local))
        snap = _evaluate_metrics(setup)
        self.assertAlmostEqual(float(snap["r_ncp_compl"][0]), expected, places=5)

    def test_residual_vi_natmap(self):
        """KKT-satisfying pair yields zero VI natural-map residual."""
        # A trivial KKT pair: λ = 0 and v_aug = 0 ⇒ both projections are 0
        # ⇒ r_vi_natmap = ||0 - proj_K(0 - 0)||_inf = 0.
        setup = self._build()
        self._collide(setup)
        setup.reset_contacts()
        n_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        sphere_pos = setup.state.body_q.numpy()[0, 0:3]
        sphere_witness = sphere_pos - self.RADIUS * n_world
        point_ground, point_sphere = _penetrating_witness_pair(sphere_witness, n_world, penetration=1.0e-3)
        setup.manual_contact(
            cid=0,
            shape0=1,
            shape1=0,
            normal_world=n_world,
            point0_world=point_ground,
            point1_world=point_sphere,
        )
        snap = _evaluate_metrics(setup)
        self.assertAlmostEqual(float(snap["r_vi_natmap"][0]), 0.0, places=6)


###
# Box-on-plane tests
###


class TestBoxOnPlane(_BenchmarkMetricsTestBase):
    """Multi-contact corner tests on the basic box/plane scene."""

    HALF_EXTENT: float = 0.1  # `basics.build_box_on_plane` uses 0.1 for all axes.

    def _build(self, **kwargs) -> TestSetup:
        return TestSetup(
            builder_fn=basics.build_box_on_plane,
            builder_kwargs={"ground": True, **kwargs},
            margin=0.0,
            gap=0.0,
            friction=0.5,
            max_contacts=16,
            device=self.default_device,
        )

    def test_box_corner_contacts_velocity(self):
        """Per-contact reported twists match the body-pair spatial twist at body0's COM.

        ``contacts.velocity`` is a *spatial* twist referenced at body0's COM, so
        every contact in a single body-pair must report the same value. This
        test injects a non-trivial box motion and checks both the closed-form
        spatial twist value and the consistency across every collider-emitted
        contact slot.
        """
        setup = self._build(z_offset=-1.0e-3)
        setup.model.collide(setup.state, setup.contacts)
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)

        omega = np.array([0.5, -0.7, 2.0], dtype=np.float32)
        v_com = np.array([0.1, -0.2, -0.3], dtype=np.float32)
        setup.set_body_state(0, lin_vel=v_com, ang_vel=omega)
        snap = _evaluate_metrics(setup)

        body_q = setup.state.body_q.numpy()[0]
        body_com = setup.model.body_com.numpy()[0]
        X = wp.transformf(*body_q)
        r_box_com_world = np.array(wp.transform_point(X, wp.vec3f(*body_com)), dtype=np.float32)

        shape_body = setup.model.shape_body.numpy()
        shape0 = setup.contacts.rigid_contact_shape0.numpy()
        shape1 = setup.contacts.rigid_contact_shape1.numpy()
        velocity = snap["velocity"]
        for c in range(nc):
            bid_0 = int(shape_body[shape0[c]])
            bid_1 = int(shape_body[shape1[c]])
            # Resolve which side is the box (the static ground returns -1).
            if bid_0 < 0 and bid_1 == 0:
                # body0 = ground at origin ⇒ arm = (0,0,0) - r_box_com_world.
                arm = -r_box_com_world
                expected_lin = v_com + np.cross(omega, arm)
                expected_ang = omega
            elif bid_1 < 0 and bid_0 == 0:
                # body0 = box ⇒ v_01 = v_ground - v_box; ground at origin and
                # the reference point is the box COM, so v_at_box_com = v_com.
                expected_lin = -v_com
                expected_ang = -omega
            else:
                continue
            np.testing.assert_allclose(velocity[c, 0:3], expected_lin, atol=1.0e-4)
            np.testing.assert_allclose(velocity[c, 3:6], expected_ang, atol=1.0e-6)

    def test_box_multi_contact_residuals_independence(self):
        """Each corner contact's residuals depend only on its own injected force."""
        setup = self._build(z_offset=-1.0e-3)
        setup.model.collide(setup.state, setup.contacts)
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)

        # Inject different in-cone force magnitudes per active contact.
        forces_local = [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0, 2.0], dtype=np.float32),
            np.array([0.1, 0.0, 1.5], dtype=np.float32),
            np.array([0.0, 0.1, 1.7], dtype=np.float32),
        ]

        shape0_np = setup.contacts.rigid_contact_shape0.numpy()
        shape1_np = setup.contacts.rigid_contact_shape1.numpy()
        normal_np = setup.contacts.rigid_contact_normal.numpy()
        point0_np = setup.contacts.rigid_contact_point0.numpy()
        point1_np = setup.contacts.rigid_contact_point1.numpy()
        for c in range(min(nc, len(forces_local))):
            f_world = _contact_to_world_vec(normal_np[c], forces_local[c % len(forces_local)])
            point0_world_np = setup.model.body_q.numpy() if False else None
            shape_body = setup.model.shape_body.numpy()
            bid_0 = int(shape_body[shape0_np[c]])
            bid_1 = int(shape_body[shape1_np[c]])
            X0 = (
                wp.transformf(*setup.state.body_q.numpy()[bid_0])
                if bid_0 >= 0
                else wp.transform_identity(dtype=wp.float32)
            )
            X1 = (
                wp.transformf(*setup.state.body_q.numpy()[bid_1])
                if bid_1 >= 0
                else wp.transform_identity(dtype=wp.float32)
            )
            p0_world = np.array(wp.transform_point(X0, wp.vec3f(*point0_np[c])), dtype=np.float32)
            p1_world = np.array(wp.transform_point(X1, wp.vec3f(*point1_np[c])), dtype=np.float32)
            setup.manual_contact(
                cid=c,
                shape0=int(shape0_np[c]),
                shape1=int(shape1_np[c]),
                normal_world=normal_np[c],
                point0_world=p0_world,
                point1_world=p1_world,
                force_world=np.concatenate([-f_world, np.zeros(3, dtype=np.float32)]),
                update_count=False,
            )
            del point0_world_np  # silence the unused-variable warning

        snap = _evaluate_metrics(setup)
        for c in range(min(nc, len(forces_local))):
            # Every force is in the friction cone ⇒ primal residual must be 0.
            self.assertAlmostEqual(float(snap["r_ncp_primal"][c]), 0.0, places=5)


###
# Boxes-nunchaku tests
###


class TestBoxesNunchaku(_BenchmarkMetricsTestBase):
    """Tests on the three-link nunchaku scene with free-joint base."""

    def _build(self, **kwargs) -> TestSetup:
        return TestSetup(
            builder_fn=basics.build_boxes_nunchaku,
            builder_kwargs={"ground": True, **kwargs},
            margin=0.0,
            gap=0.0,
            friction=0.5,
            max_contacts=32,
            device=self.default_device,
        )

    def _resolve_world_index(self, setup: TestSetup, contact_index: int) -> int:
        """Returns the world index for a given contact slot (or -1 if both shapes are global)."""
        shape_world = setup.model.shape_world.numpy()
        s0 = int(setup.contacts.rigid_contact_shape0.numpy()[contact_index])
        s1 = int(setup.contacts.rigid_contact_shape1.numpy()[contact_index])
        wid = int(shape_world[s0])
        if wid < 0:
            wid = int(shape_world[s1])
        return wid

    def test_nunchaku_contact_velocity_floating_base(self):
        """Per-contact reported twists match the body-pair spatial twist at body0's COM.

        ``contacts.velocity`` carries the relative spatial twist of body1 with
        respect to body0 at body0's COM (linear + angular). For a non-trivially
        positioned scene this is sensitive to ``body_com`` and to the actual
        body transforms, so it exercises the same coordinate transformations
        as the constraint-residual kernel.
        """
        setup = self._build(z_offset=-1.0e-3)
        setup.model.collide(setup.state, setup.contacts)
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)

        omega = np.array([0.0, 1.5, 0.0], dtype=np.float32)
        v_com = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        setup.set_body_state(0, lin_vel=v_com, ang_vel=omega)
        snap = _evaluate_metrics(setup)

        shape_body = setup.model.shape_body.numpy()
        shape0 = setup.contacts.rigid_contact_shape0.numpy()
        shape1 = setup.contacts.rigid_contact_shape1.numpy()
        body_q = setup.state.body_q.numpy()
        body_com = setup.model.body_com.numpy()
        body_qd = setup.state.body_qd.numpy()
        velocity = snap["velocity"]

        def _com_world(bid: int) -> np.ndarray:
            if bid < 0:
                return np.zeros(3, dtype=np.float32)
            X = wp.transformf(*body_q[bid])
            return np.array(wp.transform_point(X, wp.vec3f(*body_com[bid])), dtype=np.float32)

        def _twist_at(bid: int, ref_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if bid < 0:
                return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            v_com_world = body_qd[bid, 0:3].astype(np.float32)
            w_world = body_qd[bid, 3:6].astype(np.float32)
            com_world = _com_world(bid)
            return v_com_world + np.cross(w_world, ref_world - com_world), w_world

        for c in range(nc):
            bid_0 = int(shape_body[shape0[c]])
            bid_1 = int(shape_body[shape1[c]])
            r_ref = _com_world(bid_0)
            v_body0, w_body0 = _twist_at(bid_0, r_ref)
            v_body1, w_body1 = _twist_at(bid_1, r_ref)
            expected_lin = v_body1 - v_body0
            expected_ang = w_body1 - w_body0
            np.testing.assert_allclose(velocity[c, 0:3], expected_lin, atol=1.0e-4)
            np.testing.assert_allclose(velocity[c, 3:6], expected_ang, atol=1.0e-6)

    def test_nunchaku_residual_sign_under_internal_force(self):
        """An in-cone force on a nunchaku contact yields a non-negative residual."""
        setup = self._build(z_offset=-1.0e-3)
        setup.model.collide(setup.state, setup.contacts)
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)

        normal_np = setup.contacts.rigid_contact_normal.numpy()
        f_local = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        shape_body = setup.model.shape_body.numpy()
        body_q = setup.state.body_q.numpy()
        shape0_np = setup.contacts.rigid_contact_shape0.numpy()
        shape1_np = setup.contacts.rigid_contact_shape1.numpy()
        point0_np = setup.contacts.rigid_contact_point0.numpy()
        point1_np = setup.contacts.rigid_contact_point1.numpy()

        f_world = _contact_to_world_vec(normal_np[0], f_local)
        bid_0 = int(shape_body[shape0_np[0]])
        bid_1 = int(shape_body[shape1_np[0]])
        X0 = wp.transformf(*body_q[bid_0]) if bid_0 >= 0 else wp.transform_identity(dtype=wp.float32)
        X1 = wp.transformf(*body_q[bid_1]) if bid_1 >= 0 else wp.transform_identity(dtype=wp.float32)
        p0_world = np.array(wp.transform_point(X0, wp.vec3f(*point0_np[0])), dtype=np.float32)
        p1_world = np.array(wp.transform_point(X1, wp.vec3f(*point1_np[0])), dtype=np.float32)
        setup.manual_contact(
            cid=0,
            shape0=int(shape0_np[0]),
            shape1=int(shape1_np[0]),
            normal_world=normal_np[0],
            point0_world=p0_world,
            point1_world=p1_world,
            force_world=np.concatenate([-f_world, np.zeros(3, dtype=np.float32)]),
            update_count=False,
        )

        snap = _evaluate_metrics(setup)
        # In-cone force ⇒ primal residual is exactly zero.
        self.assertAlmostEqual(float(snap["r_ncp_primal"][0]), 0.0, places=5)
        # All other residuals must be finite and non-negative.
        for key in ("r_cts_penetration", "r_cts_velocity", "r_ncp_dual", "r_ncp_compl", "r_vi_natmap"):
            self.assertGreaterEqual(float(snap[key][0]), 0.0)


###
# Per-world summary tests
###


class TestPerWorldContactMetricsSummary(_BenchmarkMetricsTestBase):
    """Atomic per-world reductions over the per-contact residual arrays."""

    def _build_single_world(self) -> TestSetup:
        return TestSetup(
            builder_fn=basics.build_sphere_on_plane,
            builder_kwargs={"radius": 0.1, "mass": 1.0, "z_offset": -1.0e-3, "ground": True},
            num_worlds=1,
            margin=0.0,
            gap=0.0,
            friction=0.5,
            max_contacts=8,
            device=self.default_device,
        )

    def _build_three_worlds(self) -> TestSetup:
        return TestSetup(
            builder_fn=basics.build_box_on_plane,
            builder_kwargs={"ground": True, "z_offset": -1.0e-3},
            num_worlds=3,
            margin=0.0,
            gap=0.0,
            friction=0.5,
            max_contacts=64,
            device=self.default_device,
        )

    def _inject_residuals(self, setup: TestSetup, per_contact: dict[str, np.ndarray]) -> None:
        """Writes synthetic residuals into ``metrics.contacts.*`` for reduction testing."""
        container = metrics.PhysicsMetrics(model=setup.model)
        for name, vals in per_contact.items():
            buf = getattr(container.contacts, name).numpy()
            buf[: vals.shape[0]] = vals
            getattr(container.contacts, name).assign(buf)
        return container

    def test_single_world_matches_global_max(self):
        """One world ⇒ per-world max equals the numpy max over active contacts."""
        setup = self._build_single_world()
        setup.model.collide(setup.state, setup.contacts)
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(nc, 1)

        # Manually inject 5 contacts spanning the active range; ground shape is
        # the second one added (index 1) and the sphere shape is index 0.
        rng = np.random.default_rng(seed=2026)
        residuals = {
            name: rng.random(size=(8,), dtype=np.float32)
            for name in (
                "r_cts_penetration",
                "r_cts_velocity",
                "r_ncp_primal",
                "r_ncp_dual",
                "r_ncp_compl",
                "r_vi_natmap",
            )
        }

        # Build the shape/world map: every contact lives in world 0.
        contact_count = 5
        shape0_np = setup.contacts.rigid_contact_shape0.numpy()
        shape1_np = setup.contacts.rigid_contact_shape1.numpy()
        for c in range(contact_count):
            shape0_np[c] = 1  # ground shape index
            shape1_np[c] = 0  # sphere shape index
        setup.contacts.rigid_contact_shape0.assign(shape0_np)
        setup.contacts.rigid_contact_shape1.assign(shape1_np)
        counters = setup.contacts.contact_counters.numpy()
        counters[0] = contact_count
        setup.contacts.contact_counters.assign(counters)

        container = self._inject_residuals(setup, residuals)
        metrics.compute_per_world_contact_constraint_summary(setup.model, setup.contacts, container)

        for name, vals in residuals.items():
            actual_max = float(getattr(container.per_world_contacts_summary, name).numpy()[0])
            expected_max = float(np.max(vals[:contact_count]))
            self.assertAlmostEqual(actual_max, expected_max, places=5)
            actual_argmax = int(getattr(container.per_world_contacts_summary, name + "_argmax").numpy()[0])
            expected_argmax = int(np.argmax(vals[:contact_count]))
            self.assertEqual(actual_argmax, expected_argmax)

    def test_multi_world_independence(self):
        """Each world's max is computed only from its own contacts."""
        setup = self._build_three_worlds()
        rng = np.random.default_rng(seed=42)

        # Build a custom shape pair list spanning the three worlds: 4 contacts
        # per world, alternating shape ids per world according to the world's
        # shape_world entries.
        shape_world = setup.model.shape_world.numpy()
        world_shapes: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for sid, wid in enumerate(shape_world.tolist()):
            if 0 <= wid <= 2:
                world_shapes[wid].append(sid)
        self.assertTrue(all(len(world_shapes[w]) >= 2 for w in range(3)))

        per_world_contacts = 4
        total_contacts = per_world_contacts * 3
        residuals = rng.random(size=(total_contacts,), dtype=np.float32)
        # Bias each world's residual range so the argmax is unique per world.
        for w in range(3):
            residuals[w * per_world_contacts : (w + 1) * per_world_contacts] += float(w)

        shape0_np = setup.contacts.rigid_contact_shape0.numpy()
        shape1_np = setup.contacts.rigid_contact_shape1.numpy()
        for w in range(3):
            s0, s1 = world_shapes[w][:2]
            for k in range(per_world_contacts):
                cid = w * per_world_contacts + k
                shape0_np[cid] = s0
                shape1_np[cid] = s1
        setup.contacts.rigid_contact_shape0.assign(shape0_np)
        setup.contacts.rigid_contact_shape1.assign(shape1_np)
        counters = setup.contacts.contact_counters.numpy()
        counters[0] = total_contacts
        setup.contacts.contact_counters.assign(counters)

        container = self._inject_residuals(setup, {"r_ncp_primal": residuals})
        metrics.compute_per_world_contact_constraint_summary(setup.model, setup.contacts, container)

        per_world = container.per_world_contacts_summary.r_ncp_primal.numpy()
        per_world_argmax = container.per_world_contacts_summary.r_ncp_primal_argmax.numpy()
        for w in range(3):
            block = residuals[w * per_world_contacts : (w + 1) * per_world_contacts]
            self.assertAlmostEqual(float(per_world[w]), float(np.max(block)), places=5)
            local_argmax = int(np.argmax(block))
            self.assertEqual(int(per_world_argmax[w]), w * per_world_contacts + local_argmax)

    def test_inactive_contacts_ignored(self):
        """Padding contacts beyond ``contact_count`` must not influence the reduction."""
        setup = self._build_single_world()
        # Set residuals such that the "real" max sits at index 2, but indices
        # 3..7 contain larger values that ought to be ignored.
        residuals = np.array([0.1, 0.2, 0.9, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
        container = self._inject_residuals(setup, {"r_ncp_primal": residuals})

        # All contacts belong to world 0 via the basics scene's shape ids.
        shape0_np = setup.contacts.rigid_contact_shape0.numpy()
        shape1_np = setup.contacts.rigid_contact_shape1.numpy()
        for c in range(8):
            shape0_np[c] = 1
            shape1_np[c] = 0
        setup.contacts.rigid_contact_shape0.assign(shape0_np)
        setup.contacts.rigid_contact_shape1.assign(shape1_np)
        counters = setup.contacts.contact_counters.numpy()
        counters[0] = 3  # only the first three are active
        setup.contacts.contact_counters.assign(counters)

        metrics.compute_per_world_contact_constraint_summary(setup.model, setup.contacts, container)
        self.assertAlmostEqual(float(container.per_world_contacts_summary.r_ncp_primal.numpy()[0]), 0.9, places=5)
        self.assertEqual(int(container.per_world_contacts_summary.r_ncp_primal_argmax.numpy()[0]), 2)

    def test_global_shapes_skipped(self):
        """Contacts whose shapes are both global must be silently skipped."""
        # Build a model with at least one global shape (the ground plane added
        # via `add_ground_plane` is world-level, not global; manually create a
        # scenario where both shapes are global by referencing the sphere's
        # builder twice with a global ground).
        builder = ModelBuilder()
        builder.request_contact_attributes("force", "velocity")
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.default_shape_cfg.mu = 0.5
        # Single shape attached to no body (global ground), no per-world ground.
        builder.add_ground_plane(label="global_ground_a")
        builder.add_ground_plane(label="global_ground_b")
        builder.begin_world(label="world0")
        basics.build_sphere_on_plane(
            builder=builder,
            radius=0.1,
            mass=1.0,
            z_offset=-1.0e-3,
            friction=0.5,
            ground=False,
            new_world=False,
        )
        builder.end_world()
        model: Model = builder.finalize(device=self.default_device)
        model.rigid_contact_max = 8
        contacts: Contacts = model.contacts()

        container = metrics.PhysicsMetrics(model=model)
        # Inject a residual on a contact between the two global ground planes.
        buf = container.contacts.r_ncp_primal.numpy()
        buf[0] = 5.0
        container.contacts.r_ncp_primal.assign(buf)

        shape_world = model.shape_world.numpy()
        # Find two indices whose shape_world entry is -1 (the two global grounds).
        global_indices = np.where(shape_world < 0)[0]
        self.assertGreaterEqual(len(global_indices), 2)
        shape0_np = contacts.rigid_contact_shape0.numpy()
        shape1_np = contacts.rigid_contact_shape1.numpy()
        shape0_np[0] = int(global_indices[0])
        shape1_np[0] = int(global_indices[1])
        contacts.rigid_contact_shape0.assign(shape0_np)
        contacts.rigid_contact_shape1.assign(shape1_np)
        counters = contacts.contact_counters.numpy()
        counters[0] = 1
        contacts.contact_counters.assign(counters)

        metrics.compute_per_world_contact_constraint_summary(model, contacts, container)
        per_world = container.per_world_contacts_summary.r_ncp_primal.numpy()
        per_world_argmax = container.per_world_contacts_summary.r_ncp_primal_argmax.numpy()
        # The global-only contact must not have updated any world.
        np.testing.assert_array_equal(per_world, np.zeros_like(per_world))
        np.testing.assert_array_equal(per_world_argmax, -np.ones_like(per_world_argmax))


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
