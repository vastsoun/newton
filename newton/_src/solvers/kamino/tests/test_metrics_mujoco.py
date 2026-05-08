# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the MuJoCo-side metric extractors in
:mod:`newton._src.solvers.kamino._src.metrics.mujoco`.

The fourbar asset (``boxes_fourbar_loop.usda``) authors ``joint_4`` with
``physics:excludeFromArticulation = true`` so MuJoCo encodes it as a pair of
``mjEQ_CONNECT`` equalities, which is what
:func:`populate_joint_parent_f_from_mjw_connect_equalities` is built to recover.
"""

import unittest

import numpy as np
import warp as wp
from pxr import Usd, UsdPhysics

import newton
import newton.examples
from newton._src.solvers.kamino._src.metrics.mujoco import (
    populate_joint_parent_f_from_mjw_connect_equalities,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Module-level constants
###

_cuda_available = wp.is_cuda_available()

# Path of the loop-closing joint in ``boxes_fourbar_loop.usda``. ``add_usd``
# imports it under the asset's default-prim namespace.
_LOOP_CLOSER_JOINT_LABEL = "/boxes_fourbar/Joints/joint_4"
_ROOT_BODY_LABEL = "/boxes_fourbar/RigidBodies/body_1"
_PINNED_FIXED_JOINT_LABEL = "/boxes_fourbar/Joints/joint_world"


###
# Tests
###


@unittest.skipUnless(_cuda_available, "SolverMuJoCo Warp backend requires CUDA")
class TestMetricsMujocoConnect(unittest.TestCase):
    """End-to-end check of the ``CONNECT`` -> ``joint_parent_f`` pipeline."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _open_fourbar_stage(self, *, pin_root: bool) -> Usd.Stage:
        """Open ``boxes_fourbar_loop.usda`` in-memory; optionally insert a
        ``PhysicsFixedJoint`` from world to ``body_1`` so the linkage's root
        body is grounded (no auto-inserted floating-base joint)."""
        asset_file = newton.examples.get_asset("boxes_fourbar_loop.usda")
        stage = Usd.Stage.Open(asset_file)
        if stage is None:
            self.fail(f"Failed to open USD stage: {asset_file}")
        if pin_root:
            fixed = UsdPhysics.FixedJoint.Define(stage, _PINNED_FIXED_JOINT_LABEL)
            # Leaving body0 empty makes this a fixed-to-world joint.
            fixed.CreateBody1Rel().SetTargets([_ROOT_BODY_LABEL])
        return stage

    def _build_fourbar(self, *, pin_root: bool = False):
        """Construct a single-world fourbar model + ``SolverMuJoCo`` mirroring
        ``newton/examples/robot/example_fourbar.py``. If ``pin_root`` is True,
        ``body_1`` is fixed to the world (no auto-inserted free joint)."""
        articulation = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(articulation)
        articulation.default_shape_cfg.ke = 2.0e3
        articulation.default_shape_cfg.kd = 1.0e2
        articulation.default_shape_cfg.kf = 1.0e3
        articulation.default_shape_cfg.mu = 0.75

        stage = self._open_fourbar_stage(pin_root=pin_root)
        articulation.add_usd(
            stage,
            enable_self_collisions=False,
            hide_collision_shapes=False,
        )

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

    def _loop_closer_joint_index(self, model: newton.Model) -> int:
        """Resolve the loop-closer joint's index from ``model.joint_label``."""
        labels = list(model.joint_label)
        if _LOOP_CLOSER_JOINT_LABEL not in labels:
            self.fail(
                f"Expected joint label {_LOOP_CLOSER_JOINT_LABEL!r} not found in model. Available labels: {labels}"
            )
        return labels.index(_LOOP_CLOSER_JOINT_LABEL)

    def test_00_finalize_smoke(self):
        """Construct the model + solver and verify the loop closer is in the
        joint set (i.e. the USD ``excludeFromArticulation`` flag did not strip it)."""
        model, _ = self._build_fourbar()
        self.assertGreater(model.joint_count, 0)
        # Assert allocation of the requested extended-state attribute.
        state = model.state()
        self.assertIsNotNone(state.joint_parent_f)
        self.assertEqual(state.joint_parent_f.shape, (model.joint_count,))
        # Loop closer must be present as a Newton joint (it just isn't in MuJoCo's
        # articulation tree -- it lives as an EQ_CONNECT instead).
        self._loop_closer_joint_index(model)

    def test_01_populate_joint_parent_f_fourbar(self):
        """Step the fourbar once and verify the launcher writes a non-zero
        wrench into ``state.joint_parent_f`` at the loop-closer joint, and
        leaves every other joint at zero."""
        model, solver = self._build_fourbar()

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = newton.Contacts(solver.get_max_contact_count(), 0)

        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        # One sub-step is enough to produce non-trivial CONNECT reactions.
        sim_dt = 1.0 / 200.0
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, sim_dt)

        # Caller contract: zero ``joint_parent_f`` before invocation.
        state_1.joint_parent_f.zero_()
        populate_joint_parent_f_from_mjw_connect_equalities(solver, model, state_1)
        wp.synchronize()

        jpf = state_1.joint_parent_f.numpy()
        self.assertEqual(jpf.shape, (model.joint_count, 6))

        loop_idx = self._loop_closer_joint_index(model)

        # Loop closer must carry a non-trivial reaction wrench.
        loop_wrench = jpf[loop_idx]
        self.assertGreater(
            np.linalg.norm(loop_wrench),
            0.0,
            msg=f"Expected non-zero wrench at loop-closer joint {loop_idx}, got {loop_wrench}.",
        )

        # The launcher must touch only CONNECT-equality joints. Every other
        # joint in the model is part of MuJoCo's articulation tree and must
        # remain at the (caller-zeroed) baseline.
        mask = np.ones(model.joint_count, dtype=bool)
        mask[loop_idx] = False
        non_loop = jpf[mask]
        np.testing.assert_array_equal(
            non_loop,
            np.zeros_like(non_loop),
            err_msg="Non-loop-closer joints unexpectedly received a wrench from the launcher.",
        )

    def test_02_force_balance_pinned_root(self):
        """Pin ``body_1`` to the world via a fixed joint and verify linear
        force balance on the pinned body.

        With ``body_1`` immobilized (zero velocity, zero acceleration) Newton's
        2nd law on the linear axes reduces to::

            F_gravity_body_1
            + body_parent_f[body_1].linear   # from the pin (fixed-to-world joint)
            + joint_parent_f[loop_idx].linear  # from the loop closer (CONNECT)
            == 0

        because ``joint_4``'s child in ``boxes_fourbar_loop.usda`` is ``body_1``,
        so the launcher's output is the wrench *applied on body_1* by the loop
        closer. We verify the linear residual only -- torques are off-axis from
        the joint anchor by the COM lever arm and are not part of this check.
        """
        model, solver = self._build_fourbar(pin_root=True)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = newton.Contacts(solver.get_max_contact_count(), 0)

        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        sim_dt = 1.0 / 200.0
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, sim_dt)

        # Caller contract: zero ``joint_parent_f`` before invocation. ``body_parent_f``
        # is populated by ``solver.step`` (via the MuJoCo body-parent kernel).
        state_1.joint_parent_f.zero_()
        populate_joint_parent_f_from_mjw_connect_equalities(solver, model, state_1)
        wp.synchronize()

        body_labels = list(model.body_label)
        if _ROOT_BODY_LABEL not in body_labels:
            self.fail(f"Expected body label {_ROOT_BODY_LABEL!r} not found in model. Available labels: {body_labels}")
        root_body_idx = body_labels.index(_ROOT_BODY_LABEL)
        loop_idx = self._loop_closer_joint_index(model)

        # Sanity: the loop closer's child must be body_1 -- otherwise the
        # ``joint_parent_f`` entry isn't on the pinned body and the equation
        # below doesn't apply.
        joint_child = model.joint_child.numpy()
        self.assertEqual(
            int(joint_child[loop_idx]),
            root_body_idx,
            f"Expected loop-closer joint to have body_1 as child; "
            f"got child={joint_child[loop_idx]}, expected {root_body_idx}.",
        )

        body_parent_f = state_1.body_parent_f.numpy()
        joint_parent_f = state_1.joint_parent_f.numpy()
        gravity = model.gravity[0]  # one world; shape (3,)
        body_mass = float(model.body_mass.numpy()[root_body_idx])

        # Linear-only residual.
        f_grav = body_mass * np.asarray(gravity, dtype=np.float64)
        f_pin = body_parent_f[root_body_idx, :3].astype(np.float64)
        f_loop = joint_parent_f[loop_idx, :3].astype(np.float64)
        residual = f_grav + f_pin + f_loop

        # Sanity: the loop closer must actually carry a non-zero linear force
        # (otherwise we'd trivially satisfy the residual check just by
        # body_parent_f cancelling gravity, without exercising the launcher).
        self.assertGreater(
            np.linalg.norm(f_loop),
            0.0,
            msg=f"Expected non-zero loop-closer linear force; got {f_loop}.",
        )

        # MuJoCo's solver is run with iterations=100 and impratio=100; one
        # implicit step from rest leaves a tiny residual, dominated by the
        # solver's constraint tolerance.
        self.assertLess(
            np.linalg.norm(residual),
            1e-2,
            msg=(
                f"Linear force balance on pinned body_1 violated.\n"
                f"  f_gravity = {f_grav}\n"
                f"  body_parent_f.linear = {f_pin}\n"
                f"  joint_parent_f.linear = {f_loop}\n"
                f"  residual = {residual}, |residual| = {np.linalg.norm(residual)}"
            ),
        )


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
