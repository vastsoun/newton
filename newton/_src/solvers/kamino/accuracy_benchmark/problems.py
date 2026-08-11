# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared infrastructure for the accuracy-benchmark paper experiments.

Provides the :class:`ProblemRun` / :class:`ExampleSpec` containers plus the
scaffolding every per-example script under ``examples/paper/`` builds on:
default solver configs, shared model/scene builders, reset callbacks, the
:class:`PhysicsMetrics` + :class:`PhysicsMetricsLogger` attachment, the pelvis-
push :attr:`SetupRunner.force_cb` factory, and a few USD / actuator tuning
helpers whose logic is generic enough to be reused across scenes.

Per-scene code (asset paths, custom preprocessing, per-solver factories, the
``build_*_run`` function) lives in each ``example_benchmark_robot_*.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import warp as wp
from pxr import Sdf, Usd  # noqa TID253

import newton
from newton import solvers
from newton._src.core import Axis
from newton._src.sim import ModelBuilder
from newton._src.solvers.kamino.accuracy_benchmark.logging import PhysicsMetricsLogger
from newton._src.solvers.kamino.accuracy_benchmark.metrics import PhysicsMetrics
from newton._src.solvers.kamino.accuracy_benchmark.setup import SolverSetup
from newton._src.solvers.kamino.utils import SolverKaminoLogger

###
# Module interface
###

__all__ = [
    "ExampleSpec",
    "ProblemRun",
    "add_ground_defaults",
    "apply_default_builder_cfg",
    "attach_kamino_aux_logger",
    "attach_physics_metrics",
    "build_benchmark_model",
    "exclude_from_articulation",
    "flip_joint",
    "get_prim",
    "inflate_body_inertia",
    "kamino_default_config",
    "make_fk_reset_cb",
    "make_kamino_reset_cb",
    "make_paper_setup",
    "make_pelvis_push_cb",
    "mujoco_default_kwargs",
    "scale_pd_gains",
    "set_kamino_joint_armature_damping",
    "set_mujoco_passive_damping",
    "xpbd_default_kwargs",
]


class ProblemRun(NamedTuple):
    """Everything :class:`SetupRunner` needs to drive one paper problem.

    Attributes:
        setups: Mapping of solver name to its :class:`SolverSetup`. ``"kamino"``
            is always present and used as the runner leader.
        force_cb: Optional ``force_cb(state, contacts, sim_time)`` passed to
            :class:`SetupRunner`; ``None`` when the problem is not excited by
            an external body force (e.g. Iron Man).
        camera: Optional ``(position, pitch, yaw)`` triple; ``None`` leaves the
            viewer's default camera in place.
    """

    setups: dict[str, SolverSetup]
    force_cb: Callable | None
    camera: tuple[wp.vec3, float, float] | None


class ExampleSpec(NamedTuple):
    """Batch-runner-facing descriptor for one paper example.

    Each ``example_benchmark_robot_*.py`` exposes a module-level ``SPEC`` of
    this type so :mod:`example_benchmark_paper_all` (and the accuracy-benchmark
    dispatcher) can iterate the four examples uniformly.
    """

    build_fn: Callable[..., ProblemRun]
    """Callable returning a fully-wired :class:`ProblemRun`."""

    build_kwargs: dict
    """Static extra kwargs merged into ``build_fn(dt=..., max_log_frames=..., **build_kwargs)``."""

    sim_stop_time: float
    """Total simulated wall-time (seconds) for the example."""

    problem_name: str
    """Output stem passed to :meth:`SetupRunner.test_final`."""


###
# Shared defaults
###

# Friction / restitution used by every paper robot scene.
FRICTION: float = 0.7
RESTITUTION: float = 0.0


def kamino_default_config() -> solvers.SolverKamino.Config:
    """Base Kamino config for benchmark problems.

    Callers copy this and tweak specific fields (e.g. ``linear_solver_type``,
    ``padmm.max_iterations``, ``constraints.gamma``).
    """
    cfg = solvers.SolverKamino.Config()
    cfg.sparse_jacobian = False
    cfg.sparse_dynamics = False
    cfg.constraints.alpha = 0.01
    cfg.constraints.beta = 0.01
    cfg.constraints.gamma = 0.01
    cfg.constraints.delta = 1e-4
    cfg.dynamics.linear_solver_type = "LLTB"
    cfg.dynamics.preconditioning = True
    cfg.padmm.use_acceleration = True
    cfg.padmm.warmstart_mode = "none"
    cfg.padmm.primal_tolerance = 1e-6
    cfg.padmm.dual_tolerance = 1e-6
    cfg.padmm.compl_tolerance = 1e-6
    cfg.padmm.max_iterations = 500
    cfg.padmm.eta = 1e-5
    cfg.padmm.rho_0 = 0.02
    # The Kamino-internal ``SolutionMetrics`` are not consumed by the paper CSV
    # (we only cross-compare contact-side residuals across solvers, which are
    # populated by ``PhysicsMetrics`` from the Newton state). Leaving them off
    # skips per-step primal / dual / EoM evaluations inside every SolverKamino
    # step. Flip to ``True`` locally when debugging PADMM convergence.
    cfg.compute_solution_metrics = False
    return cfg


def mujoco_default_kwargs() -> dict:
    """Default MuJoCo kwargs for benchmark problems (fresh dict every call)."""
    return {
        "cone": "elliptic",
        "impratio": 1.0,
        "iterations": 100,
        "ls_iterations": 50,
        "tolerance": 1e-8,
        "ls_tolerance": 1e-6,
        "njmax": 512,
        "use_mujoco_contacts": False,
    }


def xpbd_default_kwargs() -> dict:
    """Default XPBD kwargs for benchmark problems (fresh dict every call).

    ``iterations=2`` is the ``SolverXPBD`` default and is used intentionally so
    the cross-solver comparison uses each solver at its documented default cost
    profile.
    """
    return {
        "iterations": 2,
        "soft_body_relaxation": 0.9,
        "soft_contact_relaxation": 0.9,
        "joint_linear_relaxation": 0.7,
        "joint_angular_relaxation": 0.4,
        "joint_linear_compliance": 0.0,
        "joint_angular_compliance": 0.0,
        "rigid_contact_relaxation": 0.8,
        "rigid_contact_con_weighting": True,
        "angular_damping": 0.0,
        "enable_restitution": True,
    }


###
# Shared build helpers
###


def apply_default_builder_cfg(builder: ModelBuilder, friction: float, restitution: float) -> None:
    """Set joint / shape defaults shared by every benchmark paper scene.

    Tight limits, small margin/gap, contact-material stiffness / damping /
    friction. Applied before ``add_usd`` so every added shape/joint inherits
    them.
    """
    builder.default_joint_cfg = ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5, armature=1e-3
    )
    builder.default_shape_cfg.margin = 1e-4
    builder.default_shape_cfg.gap = 1e-3
    builder.default_shape_cfg.ke = 2.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution


def add_ground_defaults(builder: ModelBuilder, friction: float, restitution: float) -> None:
    """Loosen the shape defaults for the ground plane and add it.

    Only used by scenes with a floating base; keeping the ground softer than
    the robot's contact material avoids over-stiff normal-force ringing at the
    foot contacts.
    """
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.add_ground_plane()


def build_benchmark_model(
    solver_type: type[solvers.SolverBase],
    scene_builder_fn: Callable,
    *,
    rigid_contact_max: int,
    scene_kwargs: dict | None = None,
):
    """Build a (builder, model) pair for a benchmark paper scene.

    Registers the solver's custom attributes on the inner scene builder, wraps
    the scene in a world with the extended ``contacts.force`` attribute (read
    by :class:`PhysicsMetrics` when computing per-contact NCP residuals) and
    finalizes the model. Also flips :data:`newton.use_coord_layout_targets`
    so ``control.joint_target_q`` is coord-sized (mandatory for floating-base
    FREE joints).
    """
    newton.use_coord_layout_targets = True
    scene_kwargs = scene_kwargs or {}
    scene_builder = ModelBuilder(up_axis=Axis.Z)
    solver_type.register_custom_attributes(scene_builder)
    scene_builder_fn(builder=scene_builder, **scene_kwargs)

    builder = ModelBuilder(up_axis=Axis.Z)
    builder.request_contact_attributes("force")
    builder.add_world(scene_builder)

    model = builder.finalize(skip_validation_joints=True)
    model.rigid_contact_max = int(rigid_contact_max)
    return builder, model


def make_fk_reset_cb(model: newton.Model) -> Callable:
    """Reset callback used by non-Kamino solvers: reload joint_q / joint_qd + FK."""

    def reset_cb(state_out):
        wp.copy(state_out.joint_q, model.joint_q)
        wp.copy(state_out.joint_qd, model.joint_qd)
        newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

    return reset_cb


def make_kamino_reset_cb(solver: solvers.SolverKamino) -> Callable:
    """Reset callback used by Kamino: delegate to its solver-native reset op."""

    def reset_cb(state_out):
        solver.reset(state=state_out)

    return reset_cb


def attach_physics_metrics(setup: SolverSetup, model: newton.Model, dt: float, max_log_frames: int) -> None:
    """Allocate a bounded ``PhysicsMetrics`` + ``PhysicsMetricsLogger`` on the setup."""
    setup.physics_metrics = PhysicsMetrics(model=model)
    setup.physics_metrics_logger = PhysicsMetricsLogger(
        metrics=setup.physics_metrics,
        max_frames=max_log_frames,
        mode=PhysicsMetricsLogger.Mode.BOUNDED,
        decimation=1,
        dt=dt,
    )


def attach_kamino_aux_logger(setup: SolverSetup, dt: float, max_log_frames: int) -> None:
    """Attach a :class:`SolverKaminoLogger` capturing PADMM iteration diagnostics."""
    setup.aux_logger = SolverKaminoLogger(
        solver=setup.solver,
        max_frames=max_log_frames,
        mode=SolverKaminoLogger.Mode.BOUNDED,
        dt=dt,
        with_iterate_residuals_info=True,
        with_acceleration_info=True,
    )


def make_paper_setup(
    *,
    name: str,
    builder: ModelBuilder,
    model: newton.Model,
    solver: solvers.SolverBase,
    dt: float,
    rigid_contact_max: int,
    max_log_frames: int,
    reset_cb: Callable,
) -> SolverSetup:
    """Build a non-standalone :class:`SolverSetup` with the paper logger stack.

    The paper comparison relies on :class:`PhysicsMetrics` (populated from the
    Newton state, works uniformly across Kamino / MuJoCo / XPBD). Callers wire
    the Kamino :class:`SolverKaminoLogger` (PADMM diagnostics) separately.
    """
    setup = SolverSetup(
        name=name,
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        standalone=False,
    )
    setup.reset_cb = reset_cb
    attach_physics_metrics(setup, model, dt, max_log_frames)
    return setup


###
# External-force helpers (pelvis push)
###


@wp.kernel
def _apply_pelvis_push(
    body_index: wp.int32,
    sim_time: wp.float32,
    force_scale: wp.float32,
    force_start_time: wp.float32,
    force_stop_time: wp.float32,
    state_body_f: wp.array[wp.spatial_vectorf],
):
    """Push ``body_index`` along +X while ``sim_time`` is inside the active window."""
    if sim_time > force_start_time and sim_time < force_stop_time:
        state_body_f[body_index] = wp.spatial_vectorf(force_scale, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        state_body_f[body_index] = wp.spatial_vectorf()


def make_pelvis_push_cb(
    setup: SolverSetup,
    body_label: str,
    *,
    force_scale: float,
    force_start_time: float,
    force_stop_time: float,
) -> Callable:
    """Return a ``force_cb(state, contacts, sim_time)`` closure that pushes a labeled body along +X.

    Resolves ``body_label`` to a body index against ``setup.model`` and returns a
    closure conforming to :class:`SetupRunner`'s ``force_cb`` signature. The
    activation window is checked against ``sim_time`` supplied by the runner
    (rather than an internal counter), so the closure is stateless and
    invocation-count independent — this matters in ``independent=True`` mode
    where the runner fans ``force_cb`` out over every setup per sub-step.
    """
    body_index = int(setup.model.body_label.index(body_label))

    def force_cb(state, contacts, sim_time):
        del contacts
        wp.launch(
            kernel=_apply_pelvis_push,
            dim=1,
            inputs=[
                wp.int32(body_index),
                wp.float32(sim_time),
                wp.float32(force_scale),
                wp.float32(force_start_time),
                wp.float32(force_stop_time),
            ],
            outputs=[state.body_f],
            device=setup.model.device,
        )

    return force_cb


###
# Reusable USD-prep utilities
###


def get_prim(stage: Usd.Stage, path: str):
    """Fetch a prim by path or raise if missing / invalid."""
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Expected prim at {path}")
    return prim


def _swap_attr_pair(prim, name_a: str, name_b: str) -> None:
    a = prim.GetAttribute(name_a)
    b = prim.GetAttribute(name_b)
    va, vb = a.Get(), b.Get()
    a.Set(vb)
    b.Set(va)


def flip_joint(stage: Usd.Stage, joint_path: str) -> None:
    """Swap ``body0``/``body1`` (and matching ``localPos``/``localRot``) on a USD joint.

    Used to normalize inconsistent parent/child conventions on hand-authored
    USD assets before ``add_usd`` infers the articulation tree.
    """
    joint = get_prim(stage, joint_path)
    body0 = joint.GetRelationship("physics:body0")
    body1 = joint.GetRelationship("physics:body1")
    t0, t1 = list(body0.GetTargets()), list(body1.GetTargets())
    body0.SetTargets(t1)
    body1.SetTargets(t0)
    _swap_attr_pair(joint, "physics:localPos0", "physics:localPos1")
    _swap_attr_pair(joint, "physics:localRot0", "physics:localRot1")


def exclude_from_articulation(stage: Usd.Stage, joint_path: str) -> None:
    """Mark a USD joint as excluded from the articulation tree.

    MuJoCo picks these up as ``mjEQ_CONNECT`` equality constraints (loop
    closures); other solvers get no tree edge for them.
    """
    attr = get_prim(stage, joint_path).CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool)
    attr.Set(True)


###
# Reusable per-solver tuning helpers
###


def scale_pd_gains(builder: ModelBuilder, kp_scale: float, kd_scale: float) -> None:
    """Multiply the USD-authored kp/kd on every actuated DoF by the given scales.

    No-op when both scales are ``1.0``. Skips DoFs whose target mode is
    :attr:`newton.JointTargetMode.NONE`.
    """
    if kp_scale == 1.0 and kd_scale == 1.0:
        return
    none_mode = int(newton.JointTargetMode.NONE)
    for dof_i, mode in enumerate(builder.joint_target_mode):
        if mode != none_mode:
            builder.joint_target_ke[dof_i] *= kp_scale
            builder.joint_target_kd[dof_i] *= kd_scale


def inflate_body_inertia(builder: ModelBuilder, body_armature: float) -> None:
    """Add ``body_armature`` * I to every body's inertia tensor.

    XPBD ignores per-joint armature, so mass-ratio-heavy articulations
    (e.g. light parallel-rod bodies feeding into much heavier legs) can be
    stabilized by diagonally inflating each body's inertia.
    """
    for body in range(builder.body_count):
        inertia_np = np.asarray(builder.body_inertia[body], dtype=np.float32).reshape(3, 3)
        inertia_np += np.eye(3, dtype=np.float32) * body_armature
        builder.body_inertia[body] = wp.mat33(inertia_np)


def set_mujoco_passive_damping(model: newton.Model, damping: float) -> None:
    """Set MuJoCo passive damping on every non-base DoF of every world.

    No-op when ``damping <= 0``. Skips the leading 6 DoFs per world (floating-
    base FREE joint) so the base doesn't get dragged against the world.
    """
    if damping <= 0.0:
        return
    pd = model.mujoco.dof_passive_damping.numpy()
    n_dof_per_world = pd.size // model.world_count
    pd.reshape(model.world_count, n_dof_per_world)[:, 6:] = damping
    model.mujoco.dof_passive_damping.assign(pd)


def set_kamino_joint_armature_damping(solver: solvers.SolverKamino, armature: float, damping: float) -> None:
    """Overwrite ``a_j`` (armature) and ``b_j`` (damping) on every actuated Kamino DoF.

    Kamino's joint model exposes per-DoF armature and viscous damping as
    ``JointsModel.a_j`` / ``JointsModel.b_j``. Handy when a paper run needs
    values not directly authored in the USD, since Kamino reads from its own
    ``JointsModel`` copy at solve time.
    """
    none_mode = int(newton.JointTargetMode.NONE)
    act_dof_indices = np.where(solver.model.joint_target_mode.numpy() != none_mode)
    a_j_np = solver._model_kamino.joints.a_j.numpy().copy()
    a_j_np[act_dof_indices] = armature
    solver._model_kamino.joints.a_j.assign(a_j_np)
    b_j_np = solver._model_kamino.joints.b_j.numpy().copy()
    b_j_np[act_dof_indices] = damping
    solver._model_kamino.joints.b_j.assign(b_j_np)
