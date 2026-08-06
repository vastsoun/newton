# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MuJoCo-side metric extractors in
:mod:`newton._src.solvers.kamino._src.metrics.mujoco`.

The fourbar asset (``boxes_fourbar.usda``) authors ``joint_4`` with
``physics:excludeFromArticulation = true``, so MuJoCo encodes it as a pair of
``mjEQ_CONNECT`` equalities -- the case the extractors are built to recover.
"""

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.metrics.mujoco import (
    extract_joint_wrenches_solvermujoco,
    populate_joint_parent_f_from_mjw_connect_equalities,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context

_LOOP_CLOSURE_JOINT_LABEL = "/boxes_fourbar/Joints/joint_4"
_ROOT_BODY_LABEL = "/boxes_fourbar/RigidBodies/body_1"
_NEXT_BODY_LABEL = "/boxes_fourbar/RigidBodies/body_2"
_PINNED_FIXED_JOINT_LABEL = "/boxes_fourbar/Joints/joint_world"
_JOINT_1_LABEL = "/boxes_fourbar/Joints/joint_1"

_SIM_DT = 1.0 / 200.0


def _build_fourbar(*, pin_root: bool):
    """Build a single-world fourbar model + ``SolverMuJoCo``. If ``pin_root``,
    ``body_1`` is fixed to the world via an inserted ``PhysicsFixedJoint``."""
    articulation = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(articulation)
    articulation.default_shape_cfg.ke = 2.0e3
    articulation.default_shape_cfg.kd = 1.0e2
    articulation.default_shape_cfg.kf = 1.0e3
    articulation.default_shape_cfg.mu = 0.75

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(newton.examples.get_asset("boxes_fourbar.usda"))
    if pin_root:
        fixed = UsdPhysics.FixedJoint.Define(stage, _PINNED_FIXED_JOINT_LABEL)
        fixed.CreateBody1Rel().SetTargets([_ROOT_BODY_LABEL])
    articulation.add_usd(stage, enable_self_collisions=False, hide_collision_shapes=False)

    if not pin_root:
        # Lift the auto-inserted free joint so the linkage spawns above the ground.
        articulation.joint_q[:3] = [0.0, 0.0, 0.1]
        if len(articulation.joint_q) > 6:
            articulation.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_world(articulation)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    builder.request_state_attributes("joint_parent_f", "body_parent_f")

    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        cone="elliptic",
        impratio=100,
        iterations=100,
        ls_iterations=50,
        nconmax=64,
        njmax=128,
        use_mujoco_contacts=True,
    )
    return model, solver


def _step_once(model, solver):
    """Run one solver step from the model's initial joint state. Returns
    ``state_1`` with ``joint_parent_f`` zeroed (caller contract for the
    extractors)."""
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = newton.Contacts(solver.get_max_contact_count(), 0)

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    state_0.clear_forces()
    solver.step(state_0, state_1, control, contacts, _SIM_DT)
    state_1.joint_parent_f.zero_()
    return state_1


class TestMetricsMujoco(unittest.TestCase):
    """Tests for the MuJoCo CONNECT extractor and the unified ``extract_joint_wrenches_solvermujoco``."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _label_idx(self, labels, label):
        self.assertIn(label, labels, msg=f"Label {label!r} not in {labels}")
        return list(labels).index(label)

    def test_00_finalize_smoke(self):
        """Loop closure survives ``excludeFromArticulation`` and ``joint_parent_f``
        is allocated when requested."""
        model, _ = _build_fourbar(pin_root=False)
        state = model.state()
        self.assertGreater(model.joint_count, 0)
        self.assertIsNotNone(state.joint_parent_f)
        self.assertEqual(state.joint_parent_f.shape, (model.joint_count,))
        self._label_idx(model.joint_label, _LOOP_CLOSURE_JOINT_LABEL)

    def test_01_populate_joint_parent_f_writes_only_loop_closure(self):
        """The CONNECT-only extractor writes a non-zero wrench at the loop closure
        and leaves every other joint at the caller-zeroed baseline."""
        model, solver = _build_fourbar(pin_root=False)
        state_1 = _step_once(model, solver)

        populate_joint_parent_f_from_mjw_connect_equalities(solver, model, state_1)
        wp.synchronize()

        jpf = state_1.joint_parent_f.numpy()
        loop_idx = self._label_idx(model.joint_label, _LOOP_CLOSURE_JOINT_LABEL)

        self.assertEqual(jpf.shape, (model.joint_count, 6))
        self.assertGreater(np.linalg.norm(jpf[loop_idx]), 0.0)
        mask = np.ones(model.joint_count, dtype=bool)
        mask[loop_idx] = False
        np.testing.assert_array_equal(jpf[mask], np.zeros_like(jpf[mask]))

    def test_02_force_balance_pinned_root_mixed_path(self):
        """Linear force balance on the pinned ``body_1`` using the CONNECT-only
        extractor for the loop closure + ``body_parent_f`` for everything else::

            F_g + body_parent_f[body_1] - body_parent_f[body_2] + joint_parent_f[loop] == 0

        where ``-body_parent_f[body_2]`` is the 3rd-law reaction from joint_1
        (body_1 is its leader)."""
        model, solver = _build_fourbar(pin_root=True)
        state_1 = _step_once(model, solver)

        populate_joint_parent_f_from_mjw_connect_equalities(solver, model, state_1)
        wp.synchronize()

        body_idx = self._label_idx(model.body_label, _ROOT_BODY_LABEL)
        next_idx = self._label_idx(model.body_label, _NEXT_BODY_LABEL)
        loop_idx = self._label_idx(model.joint_label, _LOOP_CLOSURE_JOINT_LABEL)

        bpf = state_1.body_parent_f.numpy()
        jpf = state_1.joint_parent_f.numpy()
        f_grav = float(model.body_mass.numpy()[body_idx]) * model.gravity.numpy()[0]
        f_pin = bpf[body_idx, :3]
        f_joint1 = -bpf[next_idx, :3]
        f_loop = jpf[loop_idx, :3]
        residual = (f_grav + f_pin + f_joint1 + f_loop).astype(np.float64)

        self.assertGreater(np.linalg.norm(f_loop), 0.0)
        self.assertLess(np.linalg.norm(residual), 1e-2, msg=f"residual = {residual}")

    def test_03_extract_joint_wrenches_tree_joints_match_body_parent_f(self):
        """Each tree joint's slot in ``joint_parent_f`` equals
        ``body_parent_f[joint_child]`` after :func:`extract_joint_wrenches_solvermujoco`."""
        model, solver = _build_fourbar(pin_root=True)
        state_1 = _step_once(model, solver)

        extract_joint_wrenches_solvermujoco(solver, model, state_1)
        wp.synchronize()

        bpf = state_1.body_parent_f.numpy()
        jpf = state_1.joint_parent_f.numpy()
        joint_child = model.joint_child.numpy()
        joint_articulation = model.joint_articulation.numpy()

        tree = np.nonzero(joint_articulation != -1)[0]
        self.assertGreater(tree.size, 0, msg="No tree joints found.")
        for jnt in tree:
            np.testing.assert_array_equal(jpf[jnt], bpf[joint_child[jnt]])

    def test_04_extract_joint_wrenches_loop_closure_matches_connect_extractor(self):
        """The loop closure's slot from :func:`extract_joint_wrenches_solvermujoco` is
        bit-identical to running the CONNECT-only extractor alone."""
        model, solver = _build_fourbar(pin_root=True)
        state_1 = _step_once(model, solver)

        populate_joint_parent_f_from_mjw_connect_equalities(solver, model, state_1)
        wp.synchronize()
        jpf_connect = state_1.joint_parent_f.numpy().copy()

        state_1.joint_parent_f.zero_()
        extract_joint_wrenches_solvermujoco(solver, model, state_1)
        wp.synchronize()
        jpf_unified = state_1.joint_parent_f.numpy()

        loop_idx = self._label_idx(model.joint_label, _LOOP_CLOSURE_JOINT_LABEL)
        np.testing.assert_array_equal(jpf_unified[loop_idx], jpf_connect[loop_idx])
        self.assertGreater(np.linalg.norm(jpf_unified[loop_idx]), 0.0)

    def test_05_extract_joint_wrenches_force_balance_pinned_root(self):
        """Same force balance as ``test_02`` but sourced entirely from
        ``joint_parent_f`` -- once :func:`extract_joint_wrenches_solvermujoco` has run,
        ``body_parent_f`` is redundant for closing the body-1 free-body diagram::

            F_g + joint_parent_f[pin] - joint_parent_f[joint_1] + joint_parent_f[loop] == 0
        """
        model, solver = _build_fourbar(pin_root=True)
        state_1 = _step_once(model, solver)

        extract_joint_wrenches_solvermujoco(solver, model, state_1)
        wp.synchronize()

        body_idx = self._label_idx(model.body_label, _ROOT_BODY_LABEL)
        pin_idx = self._label_idx(model.joint_label, _PINNED_FIXED_JOINT_LABEL)
        joint_1_idx = self._label_idx(model.joint_label, _JOINT_1_LABEL)
        loop_idx = self._label_idx(model.joint_label, _LOOP_CLOSURE_JOINT_LABEL)

        jpf = state_1.joint_parent_f.numpy()
        f_grav = float(model.body_mass.numpy()[body_idx]) * model.gravity.numpy()[0]
        f_pin = jpf[pin_idx, :3]
        f_joint1 = -jpf[joint_1_idx, :3]
        f_loop = jpf[loop_idx, :3]
        residual = (f_grav + f_pin + f_joint1 + f_loop).astype(np.float64)

        self.assertGreater(np.linalg.norm(f_loop), 0.0)
        self.assertLess(np.linalg.norm(residual), 1e-2, msg=f"residual = {residual}")

    def test_06_loop_closure_wrench_zero_along_joint_axis(self):
        """Shift the loop-closure wrench from the child COM to the joint anchor,
        then assert its projection onto the joint axis is zero -- the axial
        DOF is free on a revolute, and the loop closure is unactuated.

        Note: the axis-perpendicular components of the moment at the anchor
        are *not* zero in general. MuJoCo encodes the hinge as two CONNECT
        equalities offset along the axis, so the combined line of action sits
        at the midpoint of those two anchors -- which does not coincide with
        Newton's ``joint_X_c.p``. The resulting perpendicular moment is real
        (body_1 actually feels it) but its magnitude is a function of MuJoCo's
        compilation choices, not joint physics, so we don't assert on it."""
        model, solver = _build_fourbar(pin_root=True)
        state_1 = _step_once(model, solver)

        populate_joint_parent_f_from_mjw_connect_equalities(solver, model, state_1)
        wp.synchronize()

        loop_idx = self._label_idx(model.joint_label, _LOOP_CLOSURE_JOINT_LABEL)
        child = int(model.joint_child.numpy()[loop_idx])

        jpf = state_1.joint_parent_f.numpy()[loop_idx]
        f = jpf[:3].astype(np.float64)
        tau_com = jpf[3:].astype(np.float64)

        # Body-frame rotation helper (Newton quat convention: (x, y, z, w)).
        body_q = state_1.body_q.numpy()[child].astype(np.float64)
        joint_q = model.joint_X_c.numpy()[loop_idx, 3:].astype(np.float64)

        def _rotate(quat, v):
            qx, qy, qz, qw = quat
            u = np.array([qx, qy, qz])
            return 2.0 * np.dot(u, v) * u + (qw * qw - np.dot(u, u)) * v + 2.0 * qw * np.cross(u, v)

        # Shift moment from child COM to joint anchor.
        body_com_local = model.body_com.numpy()[child].astype(np.float64)
        anchor_local = model.joint_X_c.numpy()[loop_idx, :3].astype(np.float64)
        lever_world = _rotate(body_q[3:], anchor_local - body_com_local)
        tau_anchor = tau_com - np.cross(lever_world, f)

        # USD authored physics:axis = "Y" for joint_4 -> axis is body-local Y
        # transformed through joint_X_c.q, then through the child body's pose.
        axis_world = _rotate(body_q[3:], _rotate(joint_q, np.array([0.0, 1.0, 0.0])))

        self.assertGreater(np.linalg.norm(f), 0.0)
        axial_moment = float(np.dot(tau_anchor, axis_world))
        self.assertLess(
            abs(axial_moment),
            1e-3,
            msg=(
                f"Expected zero axial moment at loop-closure anchor (free DOF, "
                f"unactuated); got {axial_moment:.6g}. "
                f"tau_anchor={tau_anchor}, axis_world={axis_world}."
            ),
        )


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
