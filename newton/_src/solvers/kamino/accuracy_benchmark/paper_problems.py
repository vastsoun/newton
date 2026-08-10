# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-(problem, solver) factories for the accuracy-benchmark paper experiments.

Each articulated-robot problem (Iron Man, DR Legs, BDX, Olaf) uses the same
Kamino / MuJoCo / XPBD scaffolding as the validation runs in :mod:`problems`
but attaches a :class:`PhysicsMetrics` + :class:`PhysicsMetricsLogger` pair
so :class:`SetupRunner.test_final` can render cross-solver residual plots and
tables. The Kamino setup additionally wires a :class:`SolverKaminoLogger` for
per-step PADMM diagnostics.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import warp as wp
from pxr import Sdf, Usd, UsdPhysics  # noqa TID253

import newton
from newton import solvers
from newton._src.core import Axis
from newton._src.sim import ModelBuilder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark.assets import resolve_asset
from newton._src.solvers.kamino.accuracy_benchmark.logging import PhysicsMetricsLogger
from newton._src.solvers.kamino.accuracy_benchmark.metrics import PhysicsMetrics
from newton._src.solvers.kamino.accuracy_benchmark.problems import ProblemRun
from newton._src.solvers.kamino.accuracy_benchmark.setup import SolverSetup
from newton._src.solvers.kamino.utils import SolverKaminoLogger

###
# Module interface
###

__all__ = [
    "build_bdx_run",
    "build_dr_legs_run",
    "build_ironman_run",
    "build_olaf_run",
    "make_pelvis_push_cb",
]


###
# Constants
###

# Shared friction / restitution used by every paper robot scene.
_FRICTION: float = 0.7
_RESTITUTION: float = 0.0


# Base Kamino config for articulated systems. Callers tweak specific fields
# (e.g. ``linear_solver_type``, ``padmm.max_iterations``) after copying.
def _kamino_articulated_config() -> solvers.SolverKamino.Config:
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
    cfg.compute_solution_metrics = True
    return cfg


# Default MuJoCo config for articulated systems.
_MUJOCO_KWARGS_ARTICULATED = {
    "cone": "elliptic",
    "impratio": 1.0,
    "iterations": 100,
    "ls_iterations": 50,
    "tolerance": 1e-8,
    "ls_tolerance": 1e-6,
    "njmax": 512,
    "use_mujoco_contacts": False,
}


# Default XPBD kwargs for articulated systems. ``iterations=2`` is the SolverXPBD
# default and is used intentionally here so the cross-solver comparison uses each
# solver at its documented default cost profile.
_XPBD_KWARGS_ARTICULATED = {
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


def _apply_articulated_defaults(builder: ModelBuilder, friction: float, restitution: float) -> None:
    """Set the joint / shape defaults shared by every articulated paper scene.

    These match the values previously duplicated across the paper scripts:
    tight limits, small margin/gap, contact-material stiffness / damping /
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


def _add_ground_defaults(builder: ModelBuilder, friction: float, restitution: float) -> None:
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


def _build_articulated_model(
    solver_type: type[solvers.SolverBase],
    scene_builder_fn: Callable,
    *,
    rigid_contact_max: int,
    scene_kwargs: dict | None = None,
):
    """Build a (builder, model) pair for an articulated paper scene.

    Mirrors :func:`problems._build_problem_model` for the paper flow: registers
    the solver's custom attributes on the inner scene builder, wraps the scene
    in a world with the extended ``body_parent_f`` / ``force`` attributes and
    finalizes the model. Also flips :data:`newton.use_coord_layout_targets` so
    ``control.joint_target_q`` is coord-sized (mandatory for floating-base
    FREE joints).
    """
    newton.use_coord_layout_targets = True
    scene_kwargs = scene_kwargs or {}
    scene_builder = ModelBuilder(up_axis=Axis.Z)
    solver_type.register_custom_attributes(scene_builder)
    scene_builder_fn(builder=scene_builder, **scene_kwargs)

    builder = ModelBuilder(up_axis=Axis.Z)
    builder.request_state_attributes("body_parent_f")
    builder.request_contact_attributes("force")
    builder.add_world(scene_builder)

    model = builder.finalize(skip_validation_joints=True)
    model.rigid_contact_max = int(rigid_contact_max)
    return builder, model


def _make_fk_reset_cb(model: newton.Model) -> Callable:
    """Reset callback used by non-Kamino solvers: reload joint_q / joint_qd + FK."""

    def reset_cb(state_out):
        wp.copy(state_out.joint_q, model.joint_q)
        wp.copy(state_out.joint_qd, model.joint_qd)
        newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

    return reset_cb


def _make_kamino_reset_cb(solver: solvers.SolverKamino) -> Callable:
    """Reset callback used by Kamino: delegate to its solver-native reset op."""

    def reset_cb(state_out):
        solver.reset(state=state_out)

    return reset_cb


def _attach_physics_metrics(setup: SolverSetup, model: newton.Model, dt: float, max_log_frames: int) -> None:
    """Allocate a bounded ``PhysicsMetrics`` + ``PhysicsMetricsLogger`` on the setup."""
    setup.physics_metrics = PhysicsMetrics(model=model)
    setup.physics_metrics_logger = PhysicsMetricsLogger(
        metrics=setup.physics_metrics,
        max_frames=max_log_frames,
        mode=PhysicsMetricsLogger.Mode.BOUNDED,
        decimation=1,
        dt=dt,
    )


def _attach_kamino_aux_logger(setup: SolverSetup, dt: float, max_log_frames: int) -> None:
    """Attach a :class:`SolverKaminoLogger` capturing PADMM iteration diagnostics."""
    setup.aux_logger = SolverKaminoLogger(
        solver=setup.solver,
        max_frames=max_log_frames,
        mode=SolverKaminoLogger.Mode.BOUNDED,
        dt=dt,
        with_iterate_residuals_info=True,
        with_acceleration_info=True,
    )


def _make_paper_setup(
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
    Newton state, works uniformly across Kamino / MuJoCo / XPBD); the
    :class:`SolutionMetricsNewton` front-end is disabled here because its
    Kamino-side conversion path is not maintained in sync with the current
    geometry / model APIs. Callers wire the Kamino :class:`SolverKaminoLogger`
    (PADMM diagnostics) separately.
    """
    setup = SolverSetup(
        name=name,
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        standalone=False,
        with_solution_metrics=False,
    )
    setup.reset_cb = reset_cb
    _attach_physics_metrics(setup, model, dt, max_log_frames)
    return setup


###
# External-force helpers (pelvis push)
###


@wp.kernel
def _apply_pelvis_push(
    body_index: wp.int32,
    dt: wp.float32,
    force_scale: wp.float32,
    force_start_time: wp.float32,
    force_stop_time: wp.float32,
    time: wp.array[wp.float32],
    state_body_f: wp.array[wp.spatial_vectorf],
):
    """Push ``body_index`` along +X while ``time`` is inside the active window."""
    t = time[0]
    if t > force_start_time and t < force_stop_time:
        state_body_f[body_index] = wp.spatial_vectorf(force_scale, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        state_body_f[body_index] = wp.spatial_vectorf()
    time[0] += dt


def make_pelvis_push_cb(
    setup: SolverSetup,
    body_label: str,
    *,
    force_scale: float,
    force_start_time: float,
    force_stop_time: float,
) -> Callable:
    """Return a ``force_cb(state, contacts)`` closure that pushes a labeled body along +X.

    Resolves ``body_label`` to a body index against ``setup.model``, allocates a
    device-side one-element ``time`` counter that the kernel advances per call,
    and returns a closure conforming to :class:`SetupRunner`'s ``force_cb``
    signature.
    """
    body_index = int(setup.model.body_label.index(body_label))
    time_arr = wp.zeros(shape=(1,), dtype=wp.float32)

    def force_cb(state, contacts):
        del contacts
        wp.launch(
            kernel=_apply_pelvis_push,
            dim=1,
            inputs=[
                wp.int32(body_index),
                wp.float32(setup.dt),
                wp.float32(force_scale),
                wp.float32(force_start_time),
                wp.float32(force_stop_time),
                time_arr,
            ],
            outputs=[state.body_f],
            device=setup.model.device,
        )

    return force_cb


###
# Iron Man
###

_IRONMAN_ASSET_RELPATH = "usda/iron_man_fixed_hands_no_shell/iron_man_fixed_hands_no_shell_articulation.usda"


def _scene_ironman(builder: ModelBuilder) -> None:
    """Populate ``builder`` with the fixed-base Iron Man articulation.

    No ``floating=True`` and no ground plane: gravity acts on the articulated
    parts while the base stays clamped to the world.
    """
    _apply_articulated_defaults(builder, _FRICTION, _RESTITUTION)
    builder.add_usd(
        resolve_asset(_IRONMAN_ASSET_RELPATH),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )


def make_setup_ironman_kamino(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverKamino, _scene_ironman, rigid_contact_max=rigid_contact_max)
    cfg = _kamino_articulated_config()
    solver = solvers.SolverKamino(model=model, config=cfg)
    setup = _make_paper_setup(
        name="kamino",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_kamino_reset_cb(solver),
    )
    _attach_kamino_aux_logger(setup, dt, max_log_frames)
    return setup


def make_setup_ironman_mujoco(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverMuJoCo, _scene_ironman, rigid_contact_max=rigid_contact_max)
    solver = solvers.SolverMuJoCo(model, nconmax=rigid_contact_max, **_MUJOCO_KWARGS_ARTICULATED)
    return _make_paper_setup(
        name="mujoco",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model),
    )


def make_setup_ironman_xpbd(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverXPBD, _scene_ironman, rigid_contact_max=rigid_contact_max)
    solver = solvers.SolverXPBD(model, **_XPBD_KWARGS_ARTICULATED)
    return _make_paper_setup(
        name="xpbd",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model),
    )


def build_ironman_run(*, dt: float, max_log_frames: int, rigid_contact_max: int = 128) -> ProblemRun:
    """Build the fixed-base Iron Man :class:`ProblemRun` (Kamino / MuJoCo / XPBD)."""
    kwargs = {"dt": dt, "max_log_frames": max_log_frames, "rigid_contact_max": rigid_contact_max}
    setups = {
        "kamino": make_setup_ironman_kamino(**kwargs),
        "mujoco": make_setup_ironman_mujoco(**kwargs),
        "xpbd": make_setup_ironman_xpbd(**kwargs),
    }
    return ProblemRun(
        setups=setups,
        force_cb=None,
        camera=(wp.vec3(5.0, 5.0, 1.0), -5.0, 180.0 + 48.0),
    )


###
# Olaf
###

_OLAF_ASSET_RELPATH = "usda/Olaf/olaf_articulated.usda"
_OLAF_FORCED_BODY_LABEL = "/Olaf/RigidBodies/pelvis"
_OLAF_FORCE_SCALE = 20.0
_OLAF_FORCE_WINDOW = (1.0, 2.0)
_OLAF_START_Z = 0.5


def _scene_olaf(builder: ModelBuilder) -> None:
    """Populate ``builder`` with a floating-base Olaf on a ground plane."""
    _apply_articulated_defaults(builder, _FRICTION, _RESTITUTION)
    builder.add_usd(
        resolve_asset(_OLAF_ASSET_RELPATH),
        xform=wp.transform(wp.vec3(0.0, 0.0, _OLAF_START_Z), wp.quatf(0.0, 0.0, 0.0, 1.0)),
        floating=True,
        collapse_fixed_joints=False,
        enable_self_collisions=True,
        hide_collision_shapes=True,
    )
    _add_ground_defaults(builder, _FRICTION, _RESTITUTION)


def make_setup_olaf_kamino(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverKamino, _scene_olaf, rigid_contact_max=rigid_contact_max)
    cfg = _kamino_articulated_config()
    cfg.padmm.max_iterations = 400
    solver = solvers.SolverKamino(model=model, config=cfg)
    setup = _make_paper_setup(
        name="kamino",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_kamino_reset_cb(solver),
    )
    _attach_kamino_aux_logger(setup, dt, max_log_frames)
    return setup


def make_setup_olaf_mujoco(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverMuJoCo, _scene_olaf, rigid_contact_max=rigid_contact_max)
    solver = solvers.SolverMuJoCo(model, nconmax=rigid_contact_max, **_MUJOCO_KWARGS_ARTICULATED)
    return _make_paper_setup(
        name="mujoco",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model),
    )


def make_setup_olaf_xpbd(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverXPBD, _scene_olaf, rigid_contact_max=rigid_contact_max)
    # Small joint compliance keeps the light Olaf articulation stable under
    # XPBD's default 2 iterations.
    xpbd_kwargs = {**_XPBD_KWARGS_ARTICULATED, "joint_linear_compliance": 1e-6, "joint_angular_compliance": 1e-6}
    solver = solvers.SolverXPBD(model, **xpbd_kwargs)
    return _make_paper_setup(
        name="xpbd",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model),
    )


def build_olaf_run(*, dt: float, max_log_frames: int, rigid_contact_max: int = 128) -> ProblemRun:
    """Build the floating-base Olaf :class:`ProblemRun` (with a pelvis push)."""
    kwargs = {"dt": dt, "max_log_frames": max_log_frames, "rigid_contact_max": rigid_contact_max}
    setups = {
        "kamino": make_setup_olaf_kamino(**kwargs),
        "mujoco": make_setup_olaf_mujoco(**kwargs),
        "xpbd": make_setup_olaf_xpbd(**kwargs),
    }
    force_cb = make_pelvis_push_cb(
        setups["kamino"],
        body_label=_OLAF_FORCED_BODY_LABEL,
        force_scale=_OLAF_FORCE_SCALE,
        force_start_time=_OLAF_FORCE_WINDOW[0],
        force_stop_time=_OLAF_FORCE_WINDOW[1],
    )
    return ProblemRun(
        setups=setups,
        force_cb=force_cb,
        camera=(wp.vec3(5.0, 5.0, 1.0), -5.0, 180.0 + 48.0),
    )


###
# BDX
###

_BDX_ASSET_RELPATH = "usda/bdx/bipedal.usda"
_BDX_FORCED_BODY_LABEL = "/BD_9002_001209/PELVIS"
_BDX_FORCE_SCALE = 120.0
_BDX_FORCE_WINDOW = (1.0, 3.0)
_BDX_START_Z = 0.5


def _scene_bdx(builder: ModelBuilder) -> None:
    """Populate ``builder`` with a floating-base BDX bipedal on a ground plane."""
    _apply_articulated_defaults(builder, _FRICTION, _RESTITUTION)
    builder.add_usd(
        resolve_asset(_BDX_ASSET_RELPATH),
        xform=wp.transform(wp.vec3(0.0, 0.0, _BDX_START_Z), wp.quatf(0.0, 0.0, 0.0, 1.0)),
        floating=True,
        collapse_fixed_joints=False,
        enable_self_collisions=True,
        hide_collision_shapes=True,
    )
    _add_ground_defaults(builder, _FRICTION, _RESTITUTION)


def make_setup_bdx_kamino(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverKamino, _scene_bdx, rigid_contact_max=rigid_contact_max)
    cfg = _kamino_articulated_config()
    # BDX uses the reverse-Cuthill-McKee ordering for the LLT-blocked solver,
    # tighter iteration budget, and drops the front-end solution metrics pass
    # (comparison relies on PhysicsMetrics + PADMM aux logger, not the
    # Kamino-internal SolutionMetrics).
    cfg.dynamics.linear_solver_type = "LLTBRCM"
    cfg.padmm.max_iterations = 200
    cfg.compute_solution_metrics = False
    solver = solvers.SolverKamino(model=model, config=cfg)
    setup = _make_paper_setup(
        name="kamino",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_kamino_reset_cb(solver),
    )
    # BDX intentionally skips the Kamino aux logger (matches the original
    # paper script; PADMM diagnostics not needed for BDX write-up).
    return setup


def make_setup_bdx_mujoco(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverMuJoCo, _scene_bdx, rigid_contact_max=rigid_contact_max)
    solver = solvers.SolverMuJoCo(model, nconmax=rigid_contact_max, **_MUJOCO_KWARGS_ARTICULATED)
    return _make_paper_setup(
        name="mujoco",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model),
    )


def make_setup_bdx_xpbd(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = _build_articulated_model(solvers.SolverXPBD, _scene_bdx, rigid_contact_max=rigid_contact_max)
    solver = solvers.SolverXPBD(model, **_XPBD_KWARGS_ARTICULATED)
    return _make_paper_setup(
        name="xpbd",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model),
    )


def build_bdx_run(*, dt: float, max_log_frames: int, rigid_contact_max: int = 128) -> ProblemRun:
    """Build the floating-base BDX :class:`ProblemRun` (with a pelvis push)."""
    kwargs = {"dt": dt, "max_log_frames": max_log_frames, "rigid_contact_max": rigid_contact_max}
    setups = {
        "kamino": make_setup_bdx_kamino(**kwargs),
        "mujoco": make_setup_bdx_mujoco(**kwargs),
        "xpbd": make_setup_bdx_xpbd(**kwargs),
    }
    force_cb = make_pelvis_push_cb(
        setups["kamino"],
        body_label=_BDX_FORCED_BODY_LABEL,
        force_scale=_BDX_FORCE_SCALE,
        force_start_time=_BDX_FORCE_WINDOW[0],
        force_stop_time=_BDX_FORCE_WINDOW[1],
    )
    return ProblemRun(
        setups=setups,
        force_cb=force_cb,
        camera=(wp.vec3(5.0, 5.0, 1.0), -5.0, 180.0 + 48.0),
    )


###
# DR Legs
###

_DR_LEGS_FORCED_BODY_LABEL = "/DR_Legs/RigidBodies/pelvis"
_DR_LEGS_FORCE_SCALE = 20.0
_DR_LEGS_FORCE_WINDOW = (1.0, 2.0)
_DR_LEGS_START_Z = 0.5

# USD joints whose body0/body1 (and local-pose attrs) are swapped in the stage
# before ``add_usd`` so every hinge shares a body0=parent convention.
_DR_LEGS_FLIPPED_JOINTS = (
    "/DR_Legs/Joints/j1_l_i",
    "/DR_Legs/Joints/j2_l_i",
    "/DR_Legs/Joints/j3_l_i",
    "/DR_Legs/Joints/j4_l_i",
    "/DR_Legs/Joints/j6_l_i",
    "/DR_Legs/Joints/j6_r_i",
    "/DR_Legs/Joints/j9_l_i",
    "/DR_Legs/Joints/j9_l_o",
    "/DR_Legs/Joints/j9_r_i",
    "/DR_Legs/Joints/j9_r_o",
)

# Joints excluded from the articulation tree; MuJoCo encodes them as
# ``mjEQ_CONNECT`` loop closures (two outer foot closers + four parallel-rod
# closers).
_DR_LEGS_LOOP_CLOSURE_JOINTS = (
    "/DR_Legs/Joints/j6_l_o",
    "/DR_Legs/Joints/j6_r_o",
    "/DR_Legs/Joints/j8_l_i",
    "/DR_Legs/Joints/j8_l_o",
    "/DR_Legs/Joints/j8_r_i",
    "/DR_Legs/Joints/j8_r_o",
)

# Animation channel -> joint path. The bundled 12-column .npy file follows
# this order. Channel signs are corrected for the flip applied above.
_DR_LEGS_ANIMATION_JOINT_PATHS = (
    "/DR_Legs/Joints/j1_l_i",
    "/DR_Legs/Joints/j2_l_i",
    "/DR_Legs/Joints/j6_l_i",
    "/DR_Legs/Joints/j7_l_i",
    "/DR_Legs/Joints/j2_l_o",
    "/DR_Legs/Joints/j7_l_o",
    "/DR_Legs/Joints/j1_r_i",
    "/DR_Legs/Joints/j2_r_i",
    "/DR_Legs/Joints/j6_r_i",
    "/DR_Legs/Joints/j7_r_i",
    "/DR_Legs/Joints/j2_r_o",
    "/DR_Legs/Joints/j7_r_o",
)
_DR_LEGS_ANIMATION_CHANNEL_SIGN = np.array([-1, -1, -1, +1, +1, +1, +1, +1, -1, +1, +1, +1], dtype=np.float32)


def _get_prim(stage: Usd.Stage, path: str):
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


def _flip_joint(stage: Usd.Stage, joint_path: str) -> None:
    joint = _get_prim(stage, joint_path)
    body0 = joint.GetRelationship("physics:body0")
    body1 = joint.GetRelationship("physics:body1")
    t0, t1 = list(body0.GetTargets()), list(body1.GetTargets())
    body0.SetTargets(t1)
    body1.SetTargets(t0)
    _swap_attr_pair(joint, "physics:localPos0", "physics:localPos1")
    _swap_attr_pair(joint, "physics:localRot0", "physics:localRot1")


def _exclude_from_articulation(stage: Usd.Stage, joint_path: str) -> None:
    attr = _get_prim(stage, joint_path).CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool)
    attr.Set(True)


def _get_dr_legs_usd_stage() -> Usd.Stage:
    """Fetch the DR Legs USD asset and apply the in-memory preprocessing.

    The upstream USD has inconsistent body0/body1 orientation on several
    joints and encodes six loop-closure joints as regular articulation
    joints. We flip the former (so add_usd() infers a consistent parent/child
    tree) and exclude the latter (so MuJoCo picks them up as ``mjEQ_CONNECT``
    equality constraints rather than tree edges).
    """
    asset_path = newton.utils.download_asset("disneyresearch")
    asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")
    stage = Usd.Stage.Open(asset_file)
    if stage is None:
        raise RuntimeError(f"Failed to open dr_legs USD stage: {asset_file}")
    UsdPhysics.ArticulationRootAPI.Apply(_get_prim(stage, "/DR_Legs/RigidBodies/pelvis"))
    for jp in _DR_LEGS_FLIPPED_JOINTS:
        _flip_joint(stage, jp)
    for jp in _DR_LEGS_LOOP_CLOSURE_JOINTS:
        _exclude_from_articulation(stage, jp)
    return stage


def _scene_dr_legs(builder: ModelBuilder, usd_stage: Usd.Stage) -> None:
    """Populate ``builder`` with a floating-base DR Legs on a ground plane."""
    _apply_articulated_defaults(builder, _FRICTION, _RESTITUTION)
    builder.add_usd(
        usd_stage,
        xform=wp.transform(wp.vec3(0.0, 0.0, _DR_LEGS_START_Z), wp.quatf(0.0, 0.0, 0.0, 1.0)),
        floating=True,
        collapse_fixed_joints=False,
        enable_self_collisions=True,
        hide_collision_shapes=True,
    )
    _add_ground_defaults(builder, _FRICTION, _RESTITUTION)


def _scale_pd_gains(builder: ModelBuilder, kp_scale: float, kd_scale: float) -> None:
    """Multiply the USD-authored kp/kd on every actuated DoF by the given scales."""
    if kp_scale == 1.0 and kd_scale == 1.0:
        return
    none_mode = int(newton.JointTargetMode.NONE)
    for dof_i, mode in enumerate(builder.joint_target_mode):
        if mode != none_mode:
            builder.joint_target_ke[dof_i] *= kp_scale
            builder.joint_target_kd[dof_i] *= kd_scale


def _set_xpbd_pd_gains(builder: ModelBuilder, kp_scale: float, kd_scale: float) -> None:
    """Overwrite (rather than scale) the PD gains for the XPBD DR Legs setup.

    XPBD's position-drive is much stiffer than MuJoCo/Kamino at the same
    numeric gain, so the paper uses a fixed absolute gain of 500 * scale
    (proportional) and 200 * scale (derivative) on every actuated DoF.
    """
    none_mode = int(newton.JointTargetMode.NONE)
    for dof_i, mode in enumerate(builder.joint_target_mode):
        if mode != none_mode:
            builder.joint_target_ke[dof_i] = 500.0 * kp_scale
            builder.joint_target_kd[dof_i] = 200.0 * kd_scale


def _inflate_body_inertia(builder: ModelBuilder, body_armature: float) -> None:
    """Add ``body_armature`` * I to every body's inertia tensor.

    XPBD ignores per-joint armature, so light parallel-rod bodies (~6 g) feeding
    into much heavier legs (~600 g) explode at the loop-closure constraints.
    Diagonally inflating each body's inertia regularizes the mass ratio.
    """
    for body in range(builder.body_count):
        inertia_np = np.asarray(builder.body_inertia[body], dtype=np.float32).reshape(3, 3)
        inertia_np += np.eye(3, dtype=np.float32) * body_armature
        builder.body_inertia[body] = wp.mat33(inertia_np)


def _set_mujoco_passive_damping(model: newton.Model, damping: float) -> None:
    """Set passive damping on every non-base DoF of every world."""
    if damping <= 0.0:
        return
    pd = model.mujoco.dof_passive_damping.numpy()
    n_dof_per_world = pd.size // model.world_count
    # Skip the leading 6 DoFs per world (floating-base FREE joint); damping
    # those would drag the base against the world.
    pd.reshape(model.world_count, n_dof_per_world)[:, 6:] = damping
    model.mujoco.dof_passive_damping.assign(pd)


def _set_kamino_joint_armature_damping(solver: solvers.SolverKamino, armature: float, damping: float) -> None:
    """Overwrite ``a_j`` (armature) and ``b_j`` (damping) on every actuated DoF.

    Kamino's joint model exposes per-DoF armature and viscous damping as
    ``JointsModel.a_j`` / ``JointsModel.b_j``. The paper DR Legs run tunes
    these to values not directly authored in the USD; done here rather than
    on the Newton builder because Kamino reads from its own ``JointsModel``
    copy at solve time.
    """
    none_mode = int(newton.JointTargetMode.NONE)
    act_dof_indices = np.where(solver.model.joint_target_mode.numpy() != none_mode)
    a_j_np = solver._model_kamino.joints.a_j.numpy().copy()
    a_j_np[act_dof_indices] = armature
    solver._model_kamino.joints.a_j.assign(a_j_np)
    b_j_np = solver._model_kamino.joints.b_j.numpy().copy()
    b_j_np[act_dof_indices] = damping
    solver._model_kamino.joints.b_j.assign(b_j_np)


class _DrLegsAnimation:
    """Coord-layout PD-target driver for the DR Legs gait.

    Reads the bundled 100 Hz .npy animation buffer, resolves the 12 animated
    joint paths against ``model.joint_label`` into coord-space indices via
    ``model.joint_q_start``, and applies the sign correction that matches the
    joints flipped by :func:`_flip_joint`. Every ``__call__(control, sim_time)``
    writes the current frame's targets into ``control.joint_target_q``.

    Requires :data:`newton.use_coord_layout_targets = True` (the paper
    accuracy-benchmark scaffolding sets this in :func:`_build_articulated_model`).
    """

    def __init__(
        self,
        model: newton.Model,
        control: newton.Control,
        *,
        animation_dt: float,
        animation_speed: float,
    ):
        asset_path = newton.utils.download_asset("disneyresearch")
        anim_file = str(asset_path / "dr_legs/animation" / "dr_legs_animation_100fps.npy")
        anim = np.load(anim_file).astype(np.float32)
        if anim.shape[1] != len(_DR_LEGS_ANIMATION_JOINT_PATHS):
            raise RuntimeError(
                f"animation has {anim.shape[1]} channels, expected {len(_DR_LEGS_ANIMATION_JOINT_PATHS)}"
            )
        joint_label = list(model.joint_label)
        joint_q_start = model.joint_q_start.numpy()
        try:
            channel_coords = np.array(
                [joint_q_start[joint_label.index(path)] for path in _DR_LEGS_ANIMATION_JOINT_PATHS],
                dtype=np.int64,
            )
        except ValueError as e:
            raise RuntimeError(f"animation joint not found in model.joint_label: {e}") from e
        n_coord_per_world = model.joint_coord_count // model.world_count
        world_offsets = np.arange(model.world_count, dtype=np.int64) * n_coord_per_world
        # 2-D fancy-index assignment broadcasts a (12,) RHS across worlds.
        self._indices = channel_coords[None, :] + world_offsets[:, None]
        self._data = anim * _DR_LEGS_ANIMATION_CHANNEL_SIGN[None, :]
        self._target_q_host = control.joint_target_q.numpy()
        self._animation_dt = float(animation_dt)
        self._animation_speed = float(animation_speed)

    def __call__(self, control: newton.Control, sim_time: float) -> None:
        n_frames = self._data.shape[0]
        frame = min(int(sim_time * self._animation_speed / self._animation_dt), n_frames - 1)
        self._target_q_host[self._indices] = self._data[frame]
        control.joint_target_q.assign(self._target_q_host)


def make_setup_dr_legs_kamino(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    usd_stage: Usd.Stage,
    animation: bool,
    animation_gain_scale: float,
    animation_kd_scale: float,
    animation_passive_damping: float,
    joint_armature: float,
) -> SolverSetup:
    def scene(builder: ModelBuilder) -> None:
        # SolverKamino path shares the MuJoCo custom attributes: DR Legs mixes
        # tree joints and MuJoCo-only equality constraints, and both solvers
        # need to see the loop-closure metadata to reproduce the same physics.
        solvers.SolverMuJoCo.register_custom_attributes(builder)
        _scene_dr_legs(builder, usd_stage)
        kp_scale = animation_gain_scale if animation else 1.0
        kd_scale = animation_kd_scale if animation else 1.0
        _scale_pd_gains(builder, kp_scale, kd_scale)

    builder, model = _build_articulated_model(solvers.SolverKamino, scene, rigid_contact_max=rigid_contact_max)
    cfg = _kamino_articulated_config()
    cfg.constraints.gamma = 0.1
    solver = solvers.SolverKamino(model=model, config=cfg)
    _set_kamino_joint_armature_damping(solver, armature=joint_armature, damping=animation_passive_damping)
    setup = _make_paper_setup(
        name="kamino",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_kamino_reset_cb(solver),
    )
    _attach_kamino_aux_logger(setup, dt, max_log_frames)
    return setup


def make_setup_dr_legs_mujoco(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    usd_stage: Usd.Stage,
    animation: bool,
    animation_gain_scale: float,
    animation_kd_scale: float,
    animation_passive_damping: float,
) -> SolverSetup:
    def scene(builder: ModelBuilder) -> None:
        _scene_dr_legs(builder, usd_stage)
        kp_scale = animation_gain_scale if animation else 1.0
        kd_scale = animation_kd_scale if animation else 1.0
        _scale_pd_gains(builder, kp_scale, kd_scale)

    builder, model = _build_articulated_model(solvers.SolverMuJoCo, scene, rigid_contact_max=rigid_contact_max)
    if animation:
        _set_mujoco_passive_damping(model, animation_passive_damping)
    solver = solvers.SolverMuJoCo(model, nconmax=rigid_contact_max, **_MUJOCO_KWARGS_ARTICULATED)
    return _make_paper_setup(
        name="mujoco",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model),
    )


def make_setup_dr_legs_xpbd(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    usd_stage: Usd.Stage,
    animation: bool,
    animation_gain_scale: float,
    animation_kd_scale: float,
    xpbd_body_armature: float,
) -> SolverSetup:
    def scene(builder: ModelBuilder) -> None:
        _scene_dr_legs(builder, usd_stage)
        kp_scale = animation_gain_scale if animation else 1.0
        kd_scale = animation_kd_scale if animation else 1.0
        _set_xpbd_pd_gains(builder, kp_scale, kd_scale)
        _inflate_body_inertia(builder, xpbd_body_armature)

    builder, model = _build_articulated_model(solvers.SolverXPBD, scene, rigid_contact_max=rigid_contact_max)
    solver = solvers.SolverXPBD(model, **_XPBD_KWARGS_ARTICULATED)
    return _make_paper_setup(
        name="xpbd",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model),
    )


def build_dr_legs_run(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int = 128,
    animation: bool = False,
    animation_dt: float = 0.01,
    animation_speed: float = 0.5,
    animation_gain_scale: float = 1.2,
    animation_kd_scale: float = 1.0,
    animation_passive_damping: float = 0.5,
    joint_armature: float = 0.001,
    xpbd_body_armature: float = 0.05,
) -> ProblemRun:
    """Build the floating-base DR Legs :class:`ProblemRun`.

    Optionally drives 12 USD-actuated joints from the bundled 100 Hz animation
    (``animation=True``). The animation is applied by a
    :class:`SetupRunner`-level ``control_cb`` closure so all setups share the
    same coord-space PD targets.
    """
    usd_stage = _get_dr_legs_usd_stage()

    def _run_msg():
        # DR Legs' XPBD path is tuned separately from the others; log the
        # applied scales once so the paper runs are self-describing.
        msg.notif("DR Legs: animation=%s speed=%s dt=%s", animation, animation_speed, animation_dt)
        msg.notif(
            "DR Legs: gain_scale=%s kd_scale=%s passive_damping=%s",
            animation_gain_scale,
            animation_kd_scale,
            animation_passive_damping,
        )

    _run_msg()

    kwargs_common = {
        "dt": dt,
        "max_log_frames": max_log_frames,
        "rigid_contact_max": rigid_contact_max,
        "usd_stage": usd_stage,
        "animation": animation,
        "animation_gain_scale": animation_gain_scale,
        "animation_kd_scale": animation_kd_scale,
    }
    setups = {
        "kamino": make_setup_dr_legs_kamino(
            **kwargs_common,
            animation_passive_damping=animation_passive_damping,
            joint_armature=joint_armature,
        ),
        "mujoco": make_setup_dr_legs_mujoco(
            **kwargs_common,
            animation_passive_damping=animation_passive_damping,
        ),
        "xpbd": make_setup_dr_legs_xpbd(
            **kwargs_common,
            xpbd_body_armature=xpbd_body_armature,
        ),
    }

    force_cb = make_pelvis_push_cb(
        setups["kamino"],
        body_label=_DR_LEGS_FORCED_BODY_LABEL,
        force_scale=_DR_LEGS_FORCE_SCALE,
        force_start_time=_DR_LEGS_FORCE_WINDOW[0],
        force_stop_time=_DR_LEGS_FORCE_WINDOW[1],
    )

    return ProblemRun(
        setups=setups,
        force_cb=force_cb,
        camera=(wp.vec3(5.0, 5.0, 1.0), -5.0, 180.0 + 48.0),
    )


def make_dr_legs_animation_cb(
    setups: dict[str, SolverSetup],
    *,
    animation_dt: float = 0.01,
    animation_speed: float = 0.5,
) -> Callable:
    """Return a ``control_cb(control, sim_time)`` closure driving the DR Legs gait.

    Uses the leader setup's model/control to resolve joint indices; the runner
    reuses these buffers for every follower via ``_copy_control`` inside
    :meth:`SolverSetup.step`.
    """
    # Any setup will do — they all share the same joint layout. We take
    # kamino as canonical because it's the leader in the paper runs.
    leader = setups["kamino"]
    animation = _DrLegsAnimation(
        model=leader.model,
        control=leader.control,
        animation_dt=animation_dt,
        animation_speed=animation_speed,
    )
    return functools.partial(_call_animation, animation=animation)


def _call_animation(*, control, sim_time, animation: _DrLegsAnimation) -> None:
    """Adapter matching :class:`SetupRunner`'s ``control_cb`` signature."""
    animation(control, sim_time)
