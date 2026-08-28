# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Kamino intrinsic-Euler D6 joints."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
from newton._src.sim import Contacts, JointType
from newton._src.solvers.kamino._src.core.conversions import (
    StructuralUpdateViolation,
    validate_model_structural_updates,
)
from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.kinematics.joints import (
    compute_joints_data,
    intrinsic_3d_reciprocal_axes,
    intrinsic_3d_transported_axes,
    map_intrinsic_3d_angular_velocity_to_rates,
    select_intrinsic_3d_coords,
)
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton.solvers import SolverKamino, SolverMuJoCo
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.dynamic_d6 import run
from newton.tests.kamino.utils.extract import extract_cts_jacobians, extract_dofs_jacobians

_RH_AXES = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
_LH_AXES = (newton.Axis.X, newton.Axis.Z, newton.Axis.Y)


@wp.kernel
def _evaluate_gimbal_chart(
    coords: wp.array[wp.vec3f],
    reference: wp.array[wp.vec3f],
    omega: wp.array[wp.vec3f],
    effort: wp.array[wp.vec3f],
    third_axis_sign: wp.float32,
    selected: wp.array[wp.vec3f],
    basis_product: wp.array[wp.mat33f],
    power: wp.array[wp.vec2f],
):
    """Evaluate chart selection and reciprocal-basis identities."""
    q = coords[0]
    axis_0 = wp.vec3f(1.0, 0.0, 0.0)
    axis_1 = wp.vec3f(0.0, 1.0, 0.0)
    axis_2 = wp.vec3f(0.0, 0.0, third_axis_sign)
    axes = intrinsic_3d_transported_axes(q, axis_0, axis_1, axis_2)
    rotation = (
        wp.quat_from_axis_angle(wp.vec3f(axes[:, 2]), q[2])
        * wp.quat_from_axis_angle(wp.vec3f(axes[:, 1]), q[1])
        * wp.quat_from_axis_angle(wp.vec3f(axes[:, 0]), q[0])
    )
    selected[0] = select_intrinsic_3d_coords(rotation, reference[0], axis_0, axis_1, axis_2)
    reciprocal = intrinsic_3d_reciprocal_axes(selected[0], axis_0, axis_1, axis_2)
    basis_product[0] = wp.transpose(reciprocal) @ intrinsic_3d_transported_axes(selected[0], axis_0, axis_1, axis_2)
    rates = map_intrinsic_3d_angular_velocity_to_rates(selected[0], omega[0], axis_0, axis_1, axis_2)
    power[0] = wp.vec2f(wp.dot(effort[0], rates), wp.dot(reciprocal @ effort[0], omega[0]))


@wp.kernel
def _evaluate_gimbal_reciprocal(
    coords: wp.array[wp.vec3f],
    reciprocal: wp.array[wp.mat33f],
):
    """Evaluate the singularity-safe reciprocal basis."""
    reciprocal[0] = intrinsic_3d_reciprocal_axes(
        coords[0],
        wp.vec3f(1.0, 0.0, 0.0),
        wp.vec3f(0.0, 1.0, 0.0),
        wp.vec3f(0.0, 0.0, 1.0),
    )


def _build_rotational_d6(
    axes: tuple[newton.Axis, newton.Axis, newton.Axis],
    device: wp.DeviceLike,
    *,
    target_ke: float = 0.0,
    armature: float = 0.0,
):
    """Build a minimal articulated three-axis D6 fixture."""
    builder = newton.ModelBuilder()
    parent = builder.add_link(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    child = builder.add_link(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    root = builder.add_joint_fixed(-1, parent)
    d6 = builder.add_joint_d6(
        parent,
        child,
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=axis, target_ke=target_ke, armature=armature) for axis in axes
        ],
    )
    builder.add_articulation([root, d6])
    return builder.finalize(device=device), d6


@dataclass(frozen=True)
class _Fixture:
    """Model and D6 layout metadata for one conformance fixture."""

    model: newton.Model
    q_start: int
    qd_start: int
    target_q_start: int


@dataclass(frozen=True)
class _Probe:
    """Raw coordinate, velocity, and effort data from a solver rollout."""

    q: np.ndarray
    qd: np.ndarray
    effort: np.ndarray
    coord_count: int
    dof_count: int


def _build_fixture(
    fixed_base: bool,
    axes: tuple[newton.Axis, newton.Axis, newton.Axis],
    device: wp.DeviceLike,
    *,
    stiffness: float = 0.0,
    drive_damping: float = 0.0,
    armature: float = 0.0,
    passive_damping: float = 0.0,
    lower: float | np.ndarray = -newton.MAXVAL,
    upper: float | np.ndarray = newton.MAXVAL,
) -> _Fixture:
    """Build a collision-free articulated rotational D6."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    base = builder.add_link(mass=2.0, inertia=wp.mat33(0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0), label="base")
    link = builder.add_link(mass=1.0, inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4), label="link")
    root = (
        builder.add_joint_fixed(parent=-1, child=base, label="root")
        if fixed_base
        else builder.add_joint_free(parent=-1, child=base, label="root")
    )
    lower_values = np.broadcast_to(lower, 3)
    upper_values = np.broadcast_to(upper, 3)
    configs = [
        newton.ModelBuilder.JointDofConfig(
            axis=axis,
            target_pos=0.0,
            target_vel=0.0,
            target_ke=stiffness,
            target_kd=drive_damping,
            damping=passive_damping,
            armature=armature,
            limit_lower=float(lower_values[i]),
            limit_upper=float(upper_values[i]),
            limit_ke=1.0e4,
            limit_kd=100.0,
        )
        for i, axis in enumerate(axes)
    ]
    d6 = builder.add_joint_d6(base, link, angular_axes=configs, label="d6")
    builder.add_articulation([root, d6], label="d6")
    model = builder.finalize(device=device)
    return _Fixture(
        model,
        int(model.joint_q_start.numpy()[d6]),
        int(model.joint_qd_start.numpy()[d6]),
        int(model.joint_target_q_start.numpy()[d6]),
    )


def _make_solver(backend: str, model: newton.Model) -> SolverKamino | SolverMuJoCo:
    """Create a configured collision-free conformance solver."""
    if backend == "kamino":
        config = SolverKamino.Config(
            integrator="euler",
            use_collision_detector=False,
            use_fk_solver=False,
            sparse_jacobian=True,
        )
        config.constraints.alpha = 0.0
        config.constraints.beta = 0.1
        config.padmm.max_iterations = 200
        config.padmm.primal_tolerance = 1.0e-6
        config.padmm.dual_tolerance = 1.0e-6
        config.padmm.compl_tolerance = 1.0e-6
        return SolverKamino(model, config)
    if backend == "mjwarp":
        return SolverMuJoCo(
            model, disable_contacts=True, integrator="implicitfast", iterations=100, use_mujoco_contacts=False
        )
    raise ValueError(f"Unsupported conformance backend: {backend}")


def _run(
    backend: str,
    fixed_base: bool,
    axes: tuple[newton.Axis, newton.Axis, newton.Axis],
    device: wp.DeviceLike,
    *,
    q: np.ndarray | None = None,
    qd: np.ndarray | None = None,
    effort: np.ndarray | None = None,
    position_target: np.ndarray | None = None,
    velocity_target: np.ndarray | None = None,
    steps: int = 1,
    record_trajectory: bool = True,
    **fixture_kwargs,
) -> _Probe:
    """Run a D6 rollout and optionally retain each coordinate sample."""
    fixture = _build_fixture(fixed_base, axes, device, **fixture_kwargs)
    model = fixture.model
    state_in, state_out, control = model.state(), model.state(), model.control()
    if q is not None:
        values = state_in.joint_q.numpy()
        values[fixture.q_start : fixture.q_start + 3] = q
        state_in.joint_q.assign(values)
    if qd is not None:
        values = state_in.joint_qd.numpy()
        values[fixture.qd_start : fixture.qd_start + 3] = qd
        state_in.joint_qd.assign(values)
    if effort is not None:
        values = np.zeros(fixture.model.joint_dof_count, dtype=np.float32)
        values[fixture.qd_start : fixture.qd_start + 3] = effort
        control.joint_f.assign(values)
    if position_target is not None:
        values = control.joint_target_q.numpy()
        values[fixture.target_q_start : fixture.target_q_start + 3] = position_target
        control.joint_target_q.assign(values)
    if velocity_target is not None:
        values = control.joint_target_qd.numpy()
        values[fixture.qd_start : fixture.qd_start + 3] = velocity_target
        control.joint_target_qd.assign(values)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    solver = _make_solver(backend, model)
    contacts = Contacts(rigid_contact_max=0, soft_contact_max=0, device=model.device) if backend == "mjwarp" else None
    q_slice = slice(fixture.q_start, fixture.q_start + 3)
    qd_slice = slice(fixture.qd_start, fixture.qd_start + 3)
    positions = [state_in.joint_q.numpy()[q_slice].copy()] if record_trajectory else []
    velocities = [state_in.joint_qd.numpy()[qd_slice].copy()] if record_trajectory else []
    for _ in range(steps):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        state_in, state_out = state_out, state_in
        if record_trajectory:
            positions.append(state_in.joint_q.numpy()[q_slice].copy())
            velocities.append(state_in.joint_qd.numpy()[qd_slice].copy())
    if not record_trajectory:
        positions.append(state_in.joint_q.numpy()[q_slice].copy())
        velocities.append(state_in.joint_qd.numpy()[qd_slice].copy())
    return _Probe(
        np.stack(positions),
        np.stack(velocities),
        control.joint_f.numpy()[fixture.qd_start : fixture.qd_start + 3].copy(),
        model.joint_coord_count,
        model.joint_dof_count,
    )


def _build_generic_d6(
    n_linear: int,
    angular_axes: tuple[newton.Axis, ...],
    *,
    limits: tuple[float, float] = (-np.inf, np.inf),
    target_ke: float = 0.0,
):
    """Build a minimal D6 fixture with the requested dimensions."""
    builder = newton.ModelBuilder()
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    parent = builder.add_link(mass=1.0, inertia=inertia)
    child = builder.add_link(mass=1.0, inertia=inertia)
    root = builder.add_joint_fixed(-1, parent)
    linear_basis = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
    lower, upper = limits
    d6 = builder.add_joint_d6(
        parent,
        child,
        linear_axes=[
            newton.ModelBuilder.JointDofConfig(
                axis=linear_basis[i], limit_lower=lower, limit_upper=upper, target_ke=target_ke
            )
            for i in range(n_linear)
        ],
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=axis, limit_lower=lower, limit_upper=upper, target_ke=target_ke)
            for axis in angular_axes
        ],
    )
    builder.add_articulation([root, d6])
    return builder.finalize(device="cpu"), d6


class TestGimbal(unittest.TestCase):
    """Verify the rotational D6 representation."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_rotational_d6_converts_to_generic_d6_via_solver(self):
        """Convert a three-angular-axis D6 joint to generic D6 through SolverKamino."""
        model, d6 = _build_rotational_d6(_RH_AXES, self.default_device)
        solver = SolverKamino(model)
        self.assertEqual(solver._model_kamino.joints.dof_type.numpy()[d6], JointDoFType.D6)

    def test_left_handed_rotational_d6_remains_generic_via_solver(self):
        """Keep an X-Z-Y D6 layout generic through SolverKamino."""
        model, d6 = _build_rotational_d6(_LH_AXES, self.default_device)
        solver = SolverKamino(model)
        self.assertEqual(solver._model_kamino.joints.dof_type.numpy()[d6], JointDoFType.D6)

    def test_rotational_d6_converts_to_generic_d6(self):
        """Convert a three-angular-axis D6 joint without specializing it."""
        model, d6 = _build_rotational_d6((newton.Axis.X, newton.Axis.Y, newton.Axis.Z), self.default_device)
        model_kamino = ModelKamino.from_newton(model)
        self.assertEqual(model_kamino.joints.dof_type.numpy()[d6], JointDoFType.D6)
        np.testing.assert_array_equal(model_kamino.joints.dof_dim.numpy()[d6], [0, 3])
        np.testing.assert_array_equal(model_kamino.joints.dof_axes.numpy(), model.joint_axis.numpy())
        self.assertEqual(model_kamino.joints.num_coords.numpy()[d6], 3)
        self.assertEqual(model_kamino.joints.num_dofs.numpy()[d6], 3)
        self.assertEqual(model_kamino.joints.num_kinematic_cts.numpy()[d6], 3)

    def test_left_handed_rotational_d6_remains_generic(self):
        """Keep a left-handed authored axis sequence as generic D6 metadata."""
        model, d6 = _build_rotational_d6((newton.Axis.X, newton.Axis.Z, newton.Axis.Y), self.default_device)
        model_kamino = ModelKamino.from_newton(model)
        self.assertEqual(model_kamino.joints.dof_type.numpy()[d6], JointDoFType.D6)
        np.testing.assert_array_equal(model_kamino.joints.dof_axes.numpy(), model.joint_axis.numpy())

    def test_gimbal_rejects_nonorthogonal_axes(self):
        """Reject a gimbal whose axes are not an orthonormal basis."""
        model, d6 = _build_rotational_d6(_RH_AXES, self.default_device)
        assert model.joint_qd_start is not None
        assert model.joint_axis is not None
        qd_start = model.joint_qd_start.numpy()[d6]
        axes = model.joint_axis.numpy()
        axes[qd_start + 1] = [1.0, 0.0, 0.0]
        model.joint_axis.assign(axes)

        with self.assertRaisesRegex(ValueError, "orthogonal within each multi-axis group"):
            ModelKamino.from_newton(model)

    def test_universal_rejects_nonorthogonal_axes(self):
        """Reject a universal joint whose axes are not perpendicular."""
        builder = newton.ModelBuilder()
        parent = builder.add_link(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        child = builder.add_link(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        root = builder.add_joint_fixed(-1, parent)
        universal = builder.add_joint_d6(
            parent,
            child,
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
            ],
        )
        builder.add_articulation([root, universal])
        model = builder.finalize(device=self.default_device)
        assert model.joint_qd_start is not None
        assert model.joint_axis is not None
        qd_start = model.joint_qd_start.numpy()[universal]
        axes = model.joint_axis.numpy()
        axes[qd_start + 1] = [1.0, 0.0, 0.0]
        model.joint_axis.assign(axes)

        with self.assertRaisesRegex(ValueError, "orthogonal within each multi-axis group"):
            ModelKamino.from_newton(model)

    def test_from_newton_classifies_rotational_d6_as_generic(self):
        """Classify rotational D6 joints independently of axis handedness."""
        limits = np.zeros(3, dtype=np.float32)
        right_handed = JointDoFType.from_newton(JointType.D6, 3, 3, (0, 3), limits, limits, np.eye(3, dtype=np.float32))
        left_handed = JointDoFType.from_newton(
            JointType.D6,
            3,
            3,
            (0, 3),
            limits,
            limits,
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(right_handed, JointDoFType.D6)
        self.assertEqual(left_handed, JointDoFType.D6)

    def test_generic_d6_extracts_all_dimension_layouts(self):
        """Recover FK coordinates, rates, and zero residuals for all D6 dimensions."""
        angular_basis = (newton.Axis.Z, newton.Axis.X, newton.Axis.Y)
        for n_linear in range(4):
            for n_angular in range(4):
                with self.subTest(dof_dim=(n_linear, n_angular)):
                    model, d6 = _build_generic_d6(n_linear, angular_basis[:n_angular])
                    state = model.state()
                    q_start = int(model.joint_q_start.numpy()[d6])
                    qd_start = int(model.joint_qd_start.numpy()[d6])
                    count = n_linear + n_angular
                    expected_q = np.array([0.15, -0.2, 0.25, 0.3, -0.35, 0.2][:count], dtype=np.float32)
                    expected_qd = np.array([0.4, -0.3, 0.2, -0.25, 0.35, -0.15][:count], dtype=np.float32)
                    joint_q = state.joint_q.numpy()
                    joint_qd = state.joint_qd.numpy()
                    joint_q[q_start : q_start + count] = expected_q
                    joint_qd[qd_start : qd_start + count] = expected_qd
                    state.joint_q.assign(joint_q)
                    state.joint_qd.assign(joint_qd)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                    model_kamino = ModelKamino.from_newton(model)
                    data = model_kamino.data()
                    data.bodies.q_i.assign(state.body_q)
                    data.bodies.u_i.assign(state.body_qd)
                    q_previous = wp.array(joint_q, dtype=wp.float32, device="cpu")
                    compute_joints_data(model_kamino, data, q_previous)

                    coords_offset = int(model_kamino.joints.coords_offset.numpy()[d6])
                    dofs_offset = int(model_kamino.joints.dofs_offset.numpy()[d6])
                    cts_offset = int(model_kamino.joints.kinematic_cts_offset.numpy()[d6])
                    num_cts = 6 - count
                    self.assertEqual(model_kamino.joints.num_kinematic_cts.numpy()[d6], num_cts)
                    np.testing.assert_allclose(
                        data.joints.q_j.numpy()[coords_offset : coords_offset + count],
                        expected_q,
                        atol=1.0e-5,
                    )
                    np.testing.assert_allclose(
                        data.joints.dq_j.numpy()[dofs_offset : dofs_offset + count],
                        expected_qd,
                        atol=1.0e-5,
                    )
                    np.testing.assert_allclose(
                        data.joints.r_j.numpy()[cts_offset : cts_offset + num_cts],
                        0.0,
                        atol=1.0e-5,
                    )

    def test_generic_two_axis_d6_detects_constrained_rotation(self):
        """Detect rotation outside a two-axis D6 angular motion subspace."""
        model, d6 = _build_generic_d6(0, (newton.Axis.X, newton.Axis.Y))
        model_kamino = ModelKamino.from_newton(model)
        data = model_kamino.data()
        body_q = model.body_q.numpy()
        child = int(model.joint_child.numpy()[d6])
        body_q[child] = wp.transform(
            wp.vec3(0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5),
        )
        body_u = np.zeros((model.body_count, 6), dtype=np.float32)
        body_u[child, 5] = 0.2
        data.bodies.q_i.assign(body_q)
        data.bodies.u_i.assign(body_u)
        compute_joints_data(model_kamino, data, model_kamino.joints.q_j_0)

        cts_offset = int(model_kamino.joints.kinematic_cts_offset.numpy()[d6])
        angular_residual = data.joints.r_j.numpy()[cts_offset + 3]
        angular_velocity = data.joints.dr_j.numpy()[cts_offset + 3]
        np.testing.assert_allclose(angular_residual, np.sin(0.5), atol=1.0e-6)
        np.testing.assert_allclose(angular_velocity, 0.2 * np.cos(0.5), atol=1.0e-6)

    def test_generic_d6_extracts_left_handed_three_axis_rotation(self):
        """Recover authored coordinates and rates for a left-handed three-axis D6."""
        model, d6 = _build_generic_d6(0, _LH_AXES)
        state = model.state()
        q_start = int(model.joint_q_start.numpy()[d6])
        qd_start = int(model.joint_qd_start.numpy()[d6])
        expected_q = np.array([0.4, -0.3, 0.2], dtype=np.float32)
        expected_qd = np.array([-0.2, 0.15, 0.3], dtype=np.float32)
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        q[q_start : q_start + 3] = expected_q
        qd[qd_start : qd_start + 3] = expected_qd
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        model_kamino = ModelKamino.from_newton(model)
        data = model_kamino.data()
        data.bodies.q_i.assign(state.body_q)
        data.bodies.u_i.assign(state.body_qd)
        compute_joints_data(model_kamino, data, wp.array(q, dtype=wp.float32, device="cpu"))
        coords_offset = int(model_kamino.joints.coords_offset.numpy()[d6])
        dofs_offset = int(model_kamino.joints.dofs_offset.numpy()[d6])
        np.testing.assert_allclose(data.joints.q_j.numpy()[coords_offset : coords_offset + 3], expected_q, atol=1.0e-5)
        np.testing.assert_allclose(data.joints.dq_j.numpy()[dofs_offset : dofs_offset + 3], expected_qd, atol=1.0e-5)

    def test_generic_d6_model_change_rejects_invalid_axes(self):
        """Detect invalid generic D6 axes during model-change validation."""
        model, d6 = _build_generic_d6(2, (newton.Axis.Z, newton.Axis.X))
        model_kamino = ModelKamino.from_newton(model)
        qd_start = int(model.joint_qd_start.numpy()[d6])
        axes = model.joint_axis.numpy()
        axes[qd_start + 1] = axes[qd_start]
        model.joint_axis.assign(axes)
        violations = wp.empty(len(StructuralUpdateViolation), dtype=wp.int32, device="cpu")
        sentinel = validate_model_structural_updates(
            model,
            model_kamino.joints,
            wp.zeros(model.joint_count, dtype=wp.int32, device="cpu"),
            wp.zeros(model.joint_count, dtype=wp.int32, device="cpu"),
            violations,
            check_dof=False,
            check_actuation=False,
            check_axes=True,
            check_inertial=False,
        )
        self.assertEqual(
            violations.numpy()[StructuralUpdateViolation.NONORTHONORMAL_AXES],
            d6,
        )
        self.assertGreater(sentinel, d6)

    def test_chart_selects_nearest_equivalent_branch(self):
        """Select the equivalent Euler branch nearest the authored reference."""
        for third_axis_sign in (1.0, -1.0):
            coords = np.array([[0.4, 0.7, third_axis_sign * -0.5]], dtype=np.float32)
            alternative = np.array(
                [[coords[0, 0] + np.pi, np.pi - coords[0, 1], coords[0, 2] + third_axis_sign * np.pi]],
                dtype=np.float32,
            )
            for reference, expected in ((coords, coords), (alternative, alternative)):
                selected = wp.empty(1, dtype=wp.vec3f, device="cpu")
                product = wp.empty(1, dtype=wp.mat33f, device="cpu")
                power = wp.empty(1, dtype=wp.vec2f, device="cpu")
                wp.launch(
                    _evaluate_gimbal_chart,
                    dim=1,
                    inputs=[
                        wp.array(coords, dtype=wp.vec3f, device="cpu"),
                        wp.array(reference, dtype=wp.vec3f, device="cpu"),
                        wp.array([[0.2, -0.4, 0.3]], dtype=wp.vec3f, device="cpu"),
                        wp.array([[0.7, -0.6, 0.5]], dtype=wp.vec3f, device="cpu"),
                        third_axis_sign,
                    ],
                    outputs=[selected, product, power],
                    device="cpu",
                )
                np.testing.assert_allclose(selected.numpy(), expected, atol=1.0e-5)

    def test_reciprocal_basis_preserves_power(self):
        """Preserve dual-basis identity and generalized power away from singularity."""
        selected = wp.empty(1, dtype=wp.vec3f, device="cpu")
        product = wp.empty(1, dtype=wp.mat33f, device="cpu")
        power = wp.empty(1, dtype=wp.vec2f, device="cpu")
        coords = np.array([[0.8, np.pi / 2.0 - 0.02, 0.6]], dtype=np.float32)
        wp.launch(
            _evaluate_gimbal_chart,
            dim=1,
            inputs=[
                wp.array(coords, dtype=wp.vec3f, device="cpu"),
                wp.array(coords, dtype=wp.vec3f, device="cpu"),
                wp.array([[0.2, -0.4, 0.3]], dtype=wp.vec3f, device="cpu"),
                wp.array([[0.7, -0.6, 0.5]], dtype=wp.vec3f, device="cpu"),
                1.0,
            ],
            outputs=[selected, product, power],
            device="cpu",
        )
        np.testing.assert_allclose(product.numpy()[0], np.eye(3), atol=1.0e-5)
        np.testing.assert_allclose(power.numpy()[0, 0], power.numpy()[0, 1], atol=1.0e-5)

    def test_reciprocal_damping_is_continuous(self):
        """Avoid a discontinuity where Euler reciprocal damping begins."""
        determinants = np.array([1.001e-2, 0.999e-2, 0.0], dtype=np.float32)
        reciprocal = []
        for determinant in determinants:
            output = wp.empty(1, dtype=wp.mat33f, device="cpu")
            coords = wp.array([[0.2, np.arccos(determinant), -0.3]], dtype=wp.vec3f, device="cpu")
            wp.launch(
                _evaluate_gimbal_reciprocal,
                dim=1,
                inputs=[coords],
                outputs=[output],
                device="cpu",
            )
            reciprocal.append(output.numpy()[0])

        relative_jump = np.linalg.norm(reciprocal[1] - reciprocal[0]) / np.linalg.norm(reciprocal[0])
        self.assertLess(relative_jump, 5.0e-3)
        self.assertTrue(np.all(np.isfinite(reciprocal[2])))

    def test_built_gimbal_jacobians_map_body_twists_to_rates(self):
        """Map body twists to authored rates through dense and sparse gimbal Jacobians."""
        q_expected = np.array([0.9, -0.7, 0.5], dtype=np.float32)
        qd_expected = np.array([0.4, -0.3, 0.2], dtype=np.float32)
        for axes in (_RH_AXES, _LH_AXES):
            for sparse_jacobian in (False, True):
                with self.subTest(axes=axes, sparse_jacobian=sparse_jacobian):
                    model, d6 = _build_rotational_d6(axes, self.default_device, armature=0.5)
                    solver = SolverKamino(
                        model,
                        SolverKamino.Config(use_collision_detector=False, sparse_jacobian=sparse_jacobian),
                    )
                    state = model.state()
                    q_start = model.joint_q_start.numpy()[d6]
                    qd_start = model.joint_qd_start.numpy()[d6]
                    joint_q = state.joint_q.numpy()
                    joint_qd = state.joint_qd.numpy()
                    joint_q[q_start : q_start + 3] = q_expected
                    joint_qd[qd_start : qd_start + 3] = qd_expected
                    state.joint_q.assign(joint_q)
                    state.joint_qd.assign(joint_qd)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                    model_kamino = solver._model_kamino
                    solver_kamino = solver._solver_kamino
                    data = solver_kamino._data
                    data.bodies.q_i.assign(state.body_q)
                    data.bodies.u_i.assign(state.body_qd)
                    solver_kamino._update_joints_data(q_j_p=state.joint_q)
                    solver_kamino._update_jacobians()
                    jacobians = solver_kamino._jacobians
                    self.assertEqual(model_kamino.joints.num_dynamic_cts.numpy()[d6], 3)
                    dofs_offset = int(model_kamino.joints.dofs_offset.numpy()[d6])
                    np.testing.assert_allclose(
                        data.joints.m_j.numpy()[dofs_offset : dofs_offset + 3],
                        0.5,
                        atol=1.0e-6,
                    )
                    j_dofs = extract_dofs_jacobians(model_kamino, jacobians)[0]
                    j_cts = extract_cts_jacobians(model_kamino, None, None, jacobians)[0]
                    body_twist = data.bodies.u_i.numpy().reshape(-1)

                    np.testing.assert_allclose(j_dofs @ body_twist, qd_expected, atol=1.0e-5)
                    np.testing.assert_allclose(j_cts[:3] @ body_twist, qd_expected, atol=1.0e-5)
                    np.testing.assert_allclose(j_cts[:3], j_dofs, atol=1.0e-6)

    def test_generic_d6_jacobians_match_rates_and_storage(self):
        """Match D6 rates, power, and dense/sparse rows across representative layouts."""
        layouts = (
            (1, (newton.Axis.Z,)),
            (1, (newton.Axis.Y, newton.Axis.Z)),
            (2, (newton.Axis.Z,)),
            (0, _RH_AXES),
            (0, _LH_AXES),
            (3, _RH_AXES),
        )
        for n_linear, angular_axes in layouts:
            with self.subTest(n_linear=n_linear, angular_axes=angular_axes):
                model, d6 = _build_generic_d6(n_linear, angular_axes)
                state = model.state()
                count = n_linear + len(angular_axes)
                q_expected = np.array([0.2, -0.25, 0.15, 0.4, -0.35, 0.3][:count], dtype=np.float32)
                qd_expected = np.array([0.3, -0.2, 0.25, -0.4, 0.35, -0.15][:count], dtype=np.float32)
                q_start = int(model.joint_q_start.numpy()[d6])
                qd_start = int(model.joint_qd_start.numpy()[d6])
                joint_q = state.joint_q.numpy()
                joint_qd = state.joint_qd.numpy()
                joint_q[q_start : q_start + count] = q_expected
                joint_qd[qd_start : qd_start + count] = qd_expected
                state.joint_q.assign(joint_q)
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                dof_rows = []
                constraint_rows = []
                for sparse_jacobian in (False, True):
                    solver = SolverKamino(
                        model,
                        SolverKamino.Config(use_collision_detector=False, sparse_jacobian=sparse_jacobian),
                    )
                    solver_kamino = solver._solver_kamino
                    data = solver_kamino._data
                    data.bodies.q_i.assign(state.body_q)
                    data.bodies.u_i.assign(state.body_qd)
                    solver_kamino._update_joints_data(q_j_p=state.joint_q)
                    solver_kamino._update_jacobians()
                    j_dofs = extract_dofs_jacobians(solver_kamino._model, solver_kamino._jacobians)[0]
                    j_cts = extract_cts_jacobians(solver_kamino._model, None, None, solver_kamino._jacobians)[0]
                    body_twist = data.bodies.u_i.numpy().reshape(-1)
                    rates = j_dofs @ body_twist
                    self.assertTrue(np.all(np.isfinite(j_dofs)))
                    np.testing.assert_allclose(rates, qd_expected, atol=2.0e-4)
                    np.testing.assert_allclose(j_cts @ body_twist, 0.0, atol=2.0e-4)

                    effort = np.linspace(0.4, 0.4 + 0.1 * (count - 1), count, dtype=np.float32)
                    np.testing.assert_allclose(
                        np.dot(effort, rates),
                        np.dot(j_dofs.T @ effort, body_twist),
                        atol=1.0e-5,
                    )
                    dof_rows.append(j_dofs)
                    constraint_rows.append(j_cts)
                np.testing.assert_allclose(dof_rows[0], dof_rows[1], atol=1.0e-6)
                np.testing.assert_allclose(constraint_rows[0], constraint_rows[1], atol=1.0e-6)

    def test_generic_d6_jacobian_is_finite_at_euler_singularity(self):
        """Keep D6 rates and transpose effort mapping finite at Euler rank loss."""
        model, d6 = _build_generic_d6(0, _RH_AXES)
        state = model.state()
        q_start = int(model.joint_q_start.numpy()[d6])
        qd_start = int(model.joint_qd_start.numpy()[d6])
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        joint_q[q_start : q_start + 3] = [0.3, np.pi / 2.0, -0.2]
        joint_qd[qd_start : qd_start + 3] = [0.4, -0.3, 0.2]
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        solver = SolverKamino(model, SolverKamino.Config(use_collision_detector=False))
        solver_kamino = solver._solver_kamino
        data = solver_kamino._data
        data.bodies.q_i.assign(state.body_q)
        data.bodies.u_i.assign(state.body_qd)
        solver_kamino._update_joints_data(q_j_p=state.joint_q)
        solver_kamino._update_jacobians()
        j_dofs = extract_dofs_jacobians(solver_kamino._model, solver_kamino._jacobians)[0]
        body_twist = data.bodies.u_i.numpy().reshape(-1)
        rates = j_dofs @ body_twist
        effort = np.array([0.7, -0.4, 0.2], dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(j_dofs)))
        self.assertTrue(np.all(np.isfinite(rates)))
        np.testing.assert_allclose(
            np.dot(effort, rates),
            np.dot(j_dofs.T @ effort, body_twist),
            atol=1.0e-5,
        )

    def test_generic_d6_limit_detection_uses_direct_coordinate_order(self):
        """Detect mixed D6 limits directly in linear-first coordinate order on CPU."""
        model, d6 = _build_generic_d6(1, (newton.Axis.Z,), limits=(-0.2, 0.3))
        model_kamino = ModelKamino.from_newton(model)
        data = model_kamino.data()
        coords_offset = int(model_kamino.joints.coords_offset.numpy()[d6])
        dofs_offset = int(model_kamino.joints.dofs_offset.numpy()[d6])
        q = data.joints.q_j.numpy()
        q[coords_offset : coords_offset + 2] = [-0.4, 0.5]
        data.joints.q_j.assign(q)

        limits = LimitsKamino(model=model_kamino)
        limits.detect(data.joints.q_j)
        self.assertEqual(int(limits.model_active_limits.numpy()[0]), 2)
        np.testing.assert_array_equal(np.sort(limits.dof.numpy()[:2]), [dofs_offset, dofs_offset + 1])
        np.testing.assert_allclose(limits.r_q.numpy()[:2], -0.2, atol=1.0e-6)

    def test_fk_reset_preserves_left_handed_coordinates_and_rates(self):
        """Reset an FK-enabled solver with authored left-handed D6 state."""
        model, d6 = _build_rotational_d6(_LH_AXES, self.default_device, target_ke=1.0)
        solver = SolverKamino(model, SolverKamino.Config(use_fk_solver=True))
        state = model.state()
        q_start = model.joint_q_start.numpy()[d6]
        qd_start = model.joint_qd_start.numpy()[d6]
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        q[q_start : q_start + 3] = [0.2, -0.3, 0.4]
        qd[qd_start : qd_start + 3] = [-0.1, 0.15, -0.2]
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        solver.reset(
            state,
            config=SolverKamino.ResetConfig(
                body_poses=SolverKamino.ResetConfig.FromJointQ(state.joint_q),
                body_velocities=SolverKamino.ResetConfig.FromJointU(state.joint_qd),
            ),
        )
        np.testing.assert_allclose(state.joint_q.numpy()[q_start : q_start + 3], q[q_start : q_start + 3], atol=1.0e-5)
        np.testing.assert_allclose(
            state.joint_qd.numpy()[qd_start : qd_start + 3], qd[qd_start : qd_start + 3], atol=1.0e-5
        )

    def test_limits(self):
        """Drive each D6 coordinate to its own position limit."""
        lower = np.array([-0.15, -0.25, -0.35], dtype=np.float32)
        upper = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        target = np.array([0.6, -0.6, 0.6], dtype=np.float32)
        expected = np.where(target > 0.0, upper, lower)
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    probe = _run(
                        "kamino",
                        fixed_base,
                        axes,
                        self.default_device,
                        position_target=target,
                        stiffness=100.0,
                        drive_damping=15.0,
                        lower=lower,
                        upper=upper,
                        steps=50,
                        record_trajectory=False,
                    )
                    np.testing.assert_allclose(probe.q[-1], expected, atol=1.0e-2, rtol=0.0)

    def test_fk_reset_preserves_mixed_generic_d6_layouts(self):
        """Round-trip mixed D6 coordinates and rates through the Kamino FK solver."""
        layouts = ((1, (newton.Axis.Y, newton.Axis.Z)), (2, (newton.Axis.Z,)), (3, _RH_AXES))
        for n_linear, angular_axes in layouts:
            with self.subTest(n_linear=n_linear, angular_axes=angular_axes):
                model, d6 = _build_generic_d6(n_linear, angular_axes, target_ke=1.0)
                solver = SolverKamino(model, SolverKamino.Config(use_fk_solver=True))
                state = model.state()
                count = n_linear + len(angular_axes)
                q_start = int(model.joint_q_start.numpy()[d6])
                qd_start = int(model.joint_qd_start.numpy()[d6])
                expected_q = np.array([0.15, -0.2, 0.25, 0.3, -0.35, 0.2][:count], dtype=np.float32)
                expected_qd = np.array([0.4, -0.3, 0.2, -0.25, 0.35, -0.15][:count], dtype=np.float32)
                q = state.joint_q.numpy()
                qd = state.joint_qd.numpy()
                q[q_start : q_start + count] = expected_q
                qd[qd_start : qd_start + count] = expected_qd
                state.joint_q.assign(q)
                state.joint_qd.assign(qd)
                solver.reset(
                    state,
                    config=SolverKamino.ResetConfig(
                        body_poses=SolverKamino.ResetConfig.FromJointQ(state.joint_q),
                        body_velocities=SolverKamino.ResetConfig.FromJointU(state.joint_qd),
                    ),
                )
                np.testing.assert_allclose(state.joint_q.numpy()[q_start : q_start + count], expected_q, atol=1.0e-5)
                np.testing.assert_allclose(
                    state.joint_qd.numpy()[qd_start : qd_start + count], expected_qd, atol=1.0e-5
                )

    def test_notify_refreshes_d6_axes_and_rejects_dimensions(self):
        """Refresh D6 axes in place and reject structural dimension edits."""
        model, d6 = _build_generic_d6(1, (newton.Axis.Y, newton.Axis.Z), target_ke=1.0)
        solver = SolverKamino(model, SolverKamino.Config(use_fk_solver=True))
        qd_start = int(model.joint_qd_start.numpy()[d6])
        axes = model.joint_axis.numpy()
        axes[qd_start + 1] = [1.0, 0.0, 0.0]
        model.joint_axis.assign(axes)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_allclose(
            solver._solver_kamino._solver_fk.joints_dof_axes.numpy()[d6, :3], axes[qd_start : qd_start + 3]
        )

        dimensions = model.joint_dof_dim.numpy()
        dimensions[d6] = [2, 1]
        model.joint_dof_dim.assign(dimensions)
        with self.assertRaisesRegex(RuntimeError, "D6 dimensions"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)


@unittest.skipUnless(wp.get_cuda_device_count(), "requires CUDA device")
class TestGimbalMJWarp(unittest.TestCase):
    """Compare Kamino and MJWarp through the public Newton state layout."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _assert_pair(
        self, fixed_base: bool, axes, *, scenario: str, rtol: float = 1.0e-2, atol: float = 1.0e-2, **kwargs
    ):
        """Run and compare both solvers for one D6 scenario."""
        mjwarp = _run("mjwarp", fixed_base, axes, self.default_device, **kwargs)
        kamino = _run("kamino", fixed_base, axes, self.default_device, **kwargs)
        np.testing.assert_allclose(mjwarp.q, kamino.q, rtol=rtol, atol=atol, err_msg=f"{scenario}: q")
        np.testing.assert_allclose(mjwarp.qd, kamino.qd, rtol=rtol, atol=atol, err_msg=f"{scenario}: qd")
        return mjwarp, kamino

    def test_all_dimension_layouts_step_with_dynamics(self):
        """Step every D6 dimension layout with limits, drives, damping, and effort."""
        linear_basis = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        angular_basis = (newton.Axis.Z, newton.Axis.X, newton.Axis.Y)
        for n_linear in range(4):
            for n_angular in range(4):
                with self.subTest(dof_dim=(n_linear, n_angular)):
                    count = n_linear + n_angular
                    q = np.linspace(0.05, 0.05 * count, count, dtype=np.float32)
                    qd = np.linspace(-0.1, 0.1, count, dtype=np.float32)
                    effort = np.linspace(0.2, 0.2 + 0.05 * max(0, count - 1), count, dtype=np.float32)
                    target = np.linspace(-0.15, 0.15, count, dtype=np.float32)
                    probe = run(
                        "kamino",
                        f"d6_{n_linear}_{n_angular}",
                        True,
                        linear_basis[:n_linear],
                        angular_basis[:n_angular],
                        q=q,
                        qd=qd,
                        effort=effort,
                        position_target=target,
                        stiffness=10.0,
                        drive_damping=1.0,
                        armature=0.5,
                        passive_damping=0.2,
                        lower=-1.0,
                        upper=1.0,
                        steps=2,
                    )
                    self.assertEqual(probe.coord_count, count)
                    self.assertEqual(probe.dof_count, count)
                    self.assertTrue(np.all(np.isfinite(probe.q)))
                    self.assertTrue(np.all(np.isfinite(probe.qd)))

    def test_layout_and_state_writers(self):
        """Match D6 layout and independent coordinate/rate writes."""
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base, scenario="layout"):
                    mjwarp = _run("mjwarp", fixed_base, axes, self.default_device, steps=0)
                    kamino = _run("kamino", fixed_base, axes, self.default_device, steps=0)
                    root_coords, root_dofs = (0, 0) if fixed_base else (7, 6)
                    self.assertEqual(mjwarp.coord_count, root_coords + 3)
                    self.assertEqual(mjwarp.dof_count, root_dofs + 3)
                    self.assertEqual(kamino.coord_count, root_coords + 3)
                    self.assertEqual(kamino.dof_count, root_dofs + 3)
                for axis in range(3):
                    q = np.zeros(3, dtype=np.float32)
                    qd = np.zeros(3, dtype=np.float32)
                    q[axis] = -0.1 if axis == 1 else 0.1
                    qd[axis] = -0.2 if axis == 2 else 0.2
                    with self.subTest(axes=axes, fixed_base=fixed_base, axis=axis):
                        self._assert_pair(
                            fixed_base, axes, scenario="state writers", q=q, qd=qd, steps=1, rtol=1.0e-5, atol=1.0e-5
                        )

    def test_effort_trajectories(self):
        """Match direct generalized-effort rollouts."""
        kwargs = {
            "q": np.array([0.9, -0.7, 0.5], dtype=np.float32),
            "effort": np.array([1.0, -0.75, 0.5], dtype=np.float32),
            "steps": 10,
        }
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    mjwarp, kamino = self._assert_pair(fixed_base, axes, scenario="effort", **kwargs)
                    np.testing.assert_array_equal(mjwarp.effort, kwargs["effort"])
                    np.testing.assert_array_equal(kamino.effort, kwargs["effort"])

    def test_effort_with_armature_trajectories(self):
        """Match implicit effort-with-armature rollouts."""
        kwargs = {
            "q": np.array([0.9, -0.7, 0.5], dtype=np.float32),
            "effort": np.array([1.0, -0.75, 0.5], dtype=np.float32),
            "armature": 0.5,
            "steps": 10,
        }
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    mjwarp, kamino = self._assert_pair(fixed_base, axes, scenario="effort with armature", **kwargs)
                    np.testing.assert_array_equal(mjwarp.effort, kwargs["effort"])
                    np.testing.assert_array_equal(kamino.effort, kwargs["effort"])

    def test_implicit_pd_trajectories(self):
        """Match implicit PD rollouts."""
        kwargs = {
            "position_target": np.array([0.12, -0.08, 0.05], dtype=np.float32),
            "stiffness": 80.0,
            "drive_damping": 12.0,
            "steps": 20,
        }
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    self._assert_pair(
                        fixed_base,
                        axes,
                        scenario="implicit-pd",
                        # MJWarp implicitfast omits Kamino's dt**2 * kp effective-inertia term.
                        # This intentional discretization difference requires a higher tolerance.
                        rtol=6.0e-2,
                        atol=5.0e-3,
                        **kwargs,
                    )

    def test_unwrapped_pd_targets(self):
        """Match one-step PD motion toward unwrapped targets beyond 2*pi."""
        kwargs = {
            "q": np.array([0.2, -0.5, 0.3], dtype=np.float32),
            "position_target": np.array([2.0 * np.pi + 0.4, -2.0 * np.pi - 0.3, 2.0 * np.pi + 0.2], dtype=np.float32),
            "stiffness": 5.0,
            "drive_damping": 12.0,
            "steps": 1,
        }
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    self._assert_pair(
                        fixed_base,
                        axes,
                        scenario="unwrapped-pd-target",
                        rtol=2.5e-2,
                        atol=5.0e-3,
                        **kwargs,
                    )

    def test_passive_damping(self):
        """Match passive-damping D6 behavior."""
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                for axis in range(3):
                    velocity = np.zeros(3, dtype=np.float32)
                    velocity[axis] = 0.5
                    with self.subTest(axes=axes, fixed_base=fixed_base, axis=axis):
                        baseline = _run("kamino", fixed_base, axes, self.default_device, qd=velocity)
                        damped = _run("kamino", fixed_base, axes, self.default_device, qd=velocity, passive_damping=2.0)
                        self.assertLess(abs(damped.qd[-1, axis]), abs(baseline.qd[-1, axis]))
                with self.subTest(axes=axes, fixed_base=fixed_base, scenario="damping-match"):
                    self._assert_pair(
                        fixed_base,
                        axes,
                        scenario="passive damping",
                        q=np.array([0.9, -0.7, 0.5], dtype=np.float32),
                        qd=np.array([0.5, -0.4, 0.3], dtype=np.float32),
                        passive_damping=2.0,
                        steps=20,
                    )


if __name__ == "__main__":
    setup_tests()
    unittest.main()
