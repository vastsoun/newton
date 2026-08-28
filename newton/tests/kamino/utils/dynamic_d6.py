# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared rollout helpers for dynamic D6 joint conformance tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
from newton import Contacts
from newton.solvers import SolverKamino, SolverMuJoCo

DEVICE = "cuda:0"
DT = 1.0 / 240.0
BASE_INERTIA = wp.mat33(0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0)
LINK_INERTIA = wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)


@dataclass(frozen=True)
class Fixture:
    """Store a D6 model and its state-layout metadata."""

    model: newton.Model
    joint: int
    q_start: int
    qd_start: int
    target_q_start: int
    dof_count: int


@dataclass(frozen=True)
class Probe:
    """Store D6 coordinate and velocity trajectories."""

    q: np.ndarray
    qd: np.ndarray
    effort: np.ndarray
    coord_count: int
    dof_count: int


def build_fixture(
    name: str,
    fixed_base: bool,
    linear_axes: tuple[newton.Axis, ...],
    angular_axes: tuple[newton.Axis, ...],
    *,
    stiffness: float | np.ndarray = 0.0,
    drive_damping: float | np.ndarray = 0.0,
    armature: float | np.ndarray = 0.0,
    passive_damping: float | np.ndarray = 0.0,
    lower: float | np.ndarray = -newton.MAXVAL,
    upper: float | np.ndarray = newton.MAXVAL,
) -> Fixture:
    """Build a collision-free articulated D6 joint."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    base = builder.add_link(mass=2.0, inertia=BASE_INERTIA, label="base")
    link = builder.add_link(mass=1.0, inertia=LINK_INERTIA, label="link")
    root = (
        builder.add_joint_fixed(parent=-1, child=base, label="root")
        if fixed_base
        else builder.add_joint_free(parent=-1, child=base, label="root")
    )

    axes = linear_axes + angular_axes
    dof_count = len(axes)

    def values(value: float | np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(value, dtype=np.float32), (dof_count,))

    stiffness_values = values(stiffness)
    drive_damping_values = values(drive_damping)
    armature_values = values(armature)
    passive_damping_values = values(passive_damping)
    lower_values = values(lower)
    upper_values = values(upper)

    def config(axis: newton.Axis, dof: int) -> newton.ModelBuilder.JointDofConfig:
        return newton.ModelBuilder.JointDofConfig(
            axis=axis,
            target_pos=0.0,
            target_vel=0.0,
            target_ke=float(stiffness_values[dof]),
            target_kd=float(drive_damping_values[dof]),
            damping=float(passive_damping_values[dof]),
            armature=float(armature_values[dof]),
            limit_lower=float(lower_values[dof]),
            limit_upper=float(upper_values[dof]),
            limit_ke=1.0e4,
            limit_kd=100.0,
        )

    joint = builder.add_joint_d6(
        base,
        link,
        linear_axes=[config(axis, dof) for dof, axis in enumerate(linear_axes)],
        angular_axes=[config(axis, len(linear_axes) + dof) for dof, axis in enumerate(angular_axes)],
        label=name,
    )
    builder.add_articulation([root, joint], label=name)
    model = builder.finalize(device=DEVICE)
    assert model.joint_q_start is not None
    assert model.joint_qd_start is not None
    assert model.joint_target_q_start is not None
    return Fixture(
        model,
        joint,
        int(model.joint_q_start.numpy()[joint]),
        int(model.joint_qd_start.numpy()[joint]),
        int(model.joint_target_q_start.numpy()[joint]),
        dof_count,
    )


def make_solver(backend: str, model: newton.Model, sparse_jacobian: bool) -> SolverKamino | SolverMuJoCo:
    """Create a collision-free conformance solver."""
    if backend == "kamino":
        config = SolverKamino.Config(
            integrator="euler",
            use_collision_detector=False,
            use_fk_solver=False,
            sparse_jacobian=sparse_jacobian,
        )
        assert config.constraints is not None
        assert config.padmm is not None
        config.constraints.alpha = 0.0
        config.constraints.beta = 0.1
        config.padmm.max_iterations = 200
        config.padmm.primal_tolerance = 1.0e-6
        config.padmm.dual_tolerance = 1.0e-6
        config.padmm.compl_tolerance = 1.0e-6
        return SolverKamino(model, config)
    if backend == "mjwarp":
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            integrator="implicitfast",
            iterations=100,
            use_mujoco_contacts=False,
        )
    raise ValueError(f"Unsupported conformance backend: {backend}")


def run(
    backend: str,
    name: str,
    fixed_base: bool,
    linear_axes: tuple[newton.Axis, ...],
    angular_axes: tuple[newton.Axis, ...],
    *,
    q: np.ndarray | None = None,
    qd: np.ndarray | None = None,
    effort: np.ndarray | None = None,
    position_target: np.ndarray | None = None,
    velocity_target: np.ndarray | None = None,
    sparse_jacobian: bool = True,
    steps: int = 1,
    **fixture_kwargs,
) -> Probe:
    """Run a D6 rollout and retain its raw trajectory."""
    fixture = build_fixture(name, fixed_base, linear_axes, angular_axes, **fixture_kwargs)
    state_in, state_out, control = fixture.model.state(), fixture.model.state(), fixture.model.control()
    assert state_in.joint_q is not None
    assert state_in.joint_qd is not None
    assert control.joint_f is not None
    assert control.joint_target_q is not None
    assert control.joint_target_qd is not None
    q_slice = slice(fixture.q_start, fixture.q_start + fixture.dof_count)
    qd_slice = slice(fixture.qd_start, fixture.qd_start + fixture.dof_count)
    target_q_slice = slice(fixture.target_q_start, fixture.target_q_start + fixture.dof_count)

    if q is not None:
        values = state_in.joint_q.numpy()
        values[q_slice] = q
        state_in.joint_q.assign(values)
    if qd is not None:
        values = state_in.joint_qd.numpy()
        values[qd_slice] = qd
        state_in.joint_qd.assign(values)
    if effort is not None:
        values = np.zeros(fixture.model.joint_dof_count, dtype=np.float32)
        values[qd_slice] = effort
        control.joint_f.assign(values)
    if position_target is not None:
        values = control.joint_target_q.numpy()
        values[target_q_slice] = position_target
        control.joint_target_q.assign(values)
    if velocity_target is not None:
        values = control.joint_target_qd.numpy()
        values[qd_slice] = velocity_target
        control.joint_target_qd.assign(values)

    newton.eval_fk(fixture.model, state_in.joint_q, state_in.joint_qd, state_in)
    solver = make_solver(backend, fixture.model, sparse_jacobian)
    contacts = Contacts(rigid_contact_max=0, soft_contact_max=0, device=DEVICE) if backend == "mjwarp" else None
    positions = [state_in.joint_q.numpy()[q_slice].copy()]
    velocities = [state_in.joint_qd.numpy()[qd_slice].copy()]
    for _ in range(steps):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, DT)
        state_in, state_out = state_out, state_in
        assert state_in.joint_q is not None
        assert state_in.joint_qd is not None
        positions.append(state_in.joint_q.numpy()[q_slice].copy())
        velocities.append(state_in.joint_qd.numpy()[qd_slice].copy())

    return Probe(
        np.stack(positions),
        np.stack(velocities),
        control.joint_f.numpy()[qd_slice].copy(),
        fixture.model.joint_coord_count,
        fixture.model.joint_dof_count,
    )
