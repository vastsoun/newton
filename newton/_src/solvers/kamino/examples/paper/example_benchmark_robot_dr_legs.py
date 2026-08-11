# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-benchmark entry point: floating-base Disney Research legs on a ground plane.

Optionally drives 12 USD-actuated joints from the bundled 100 Hz animation
buffer (``--animation``). USD preprocessing (body0/body1 flips and loop-closure
markings) plus per-solver PD gain scaling and inertia regularization all live
in this file so the shared :mod:`accuracy_benchmark.problems` module only
hosts helpers reusable across scenes.

The default configuration is headless data collection (progress bar + csv / pdf
export + console summary). Pass ``--no-headless`` to open a viewer.

Run with::

    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_robot_dr_legs
"""

from __future__ import annotations

import argparse
import functools
import math
from collections.abc import Callable

import numpy as np
import warp as wp
from pxr import Usd, UsdPhysics  # noqa TID253

import newton
import newton.examples
from newton import solvers
from newton._src.sim import ModelBuilder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark import MODE_TIED_REFERENCE, SetupRunner
from newton._src.solvers.kamino.accuracy_benchmark.problems import (
    FRICTION,
    RESTITUTION,
    ExampleSpec,
    ProblemRun,
    ReferenceLeader,
    add_ground_defaults,
    apply_default_builder_cfg,
    attach_kamino_aux_logger,
    build_benchmark_model,
    exclude_from_articulation,
    flip_joint,
    get_prim,
    inflate_body_inertia,
    kamino_default_config,
    make_fk_reset_cb,
    make_kamino_reset_cb,
    make_paper_setup,
    make_pelvis_push_cb,
    mujoco_default_kwargs,
    scale_pd_gains,
    set_kamino_joint_armature_damping,
    set_mujoco_passive_damping,
    xpbd_default_kwargs,
)
from newton._src.solvers.kamino.accuracy_benchmark.setup import SolverSetup


def _kamino_config_dr_legs() -> solvers.SolverKamino.Config:
    """Kamino config shared by DR Legs coarse-follower and reference-leader instances."""
    cfg = kamino_default_config()
    cfg.constraints.gamma = 0.1
    return cfg


SIM_DT: float = 0.001
VIZ_FPS: int = 50
SIM_STOP_TIME: float = 4.0

_FORCED_BODY_LABEL = "/DR_Legs/RigidBodies/pelvis"
_FORCE_SCALE = 20.0
_FORCE_WINDOW = (1.0, 2.0)
_START_Z = 0.5

# USD joints whose body0/body1 (and local-pose attrs) are swapped in the stage
# before ``add_usd`` so every hinge shares a body0=parent convention.
_FLIPPED_JOINTS = (
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
_LOOP_CLOSURE_JOINTS = (
    "/DR_Legs/Joints/j6_l_o",
    "/DR_Legs/Joints/j6_r_o",
    "/DR_Legs/Joints/j8_l_i",
    "/DR_Legs/Joints/j8_l_o",
    "/DR_Legs/Joints/j8_r_i",
    "/DR_Legs/Joints/j8_r_o",
)

# Animation channel -> joint path. The bundled 12-column .npy file follows
# this order. Channel signs are corrected for the flip applied above.
_ANIMATION_JOINT_PATHS = (
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
_ANIMATION_CHANNEL_SIGN = np.array([-1, -1, -1, +1, +1, +1, +1, +1, -1, +1, +1, +1], dtype=np.float32)


###
# Scene / USD preprocessing
###


def _get_usd_stage() -> Usd.Stage:
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
    UsdPhysics.ArticulationRootAPI.Apply(get_prim(stage, "/DR_Legs/RigidBodies/pelvis"))
    for jp in _FLIPPED_JOINTS:
        flip_joint(stage, jp)
    for jp in _LOOP_CLOSURE_JOINTS:
        exclude_from_articulation(stage, jp)
    return stage


def _scene_dr_legs(builder: ModelBuilder, usd_stage: Usd.Stage) -> None:
    """Populate ``builder`` with a floating-base DR Legs on a ground plane."""
    apply_default_builder_cfg(builder, FRICTION, RESTITUTION)
    builder.add_usd(
        usd_stage,
        xform=wp.transform(wp.vec3(0.0, 0.0, _START_Z), wp.quatf(0.0, 0.0, 0.0, 1.0)),
        floating=True,
        collapse_fixed_joints=False,
        enable_self_collisions=True,
        hide_collision_shapes=True,
    )
    add_ground_defaults(builder, FRICTION, RESTITUTION)


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


###
# Animation
###


class _DrLegsAnimation:
    """Coord-layout PD-target driver for the DR Legs gait.

    Reads the bundled 100 Hz .npy animation buffer, resolves the 12 animated
    joint paths against ``model.joint_label`` into coord-space indices via
    ``model.joint_q_start``, and applies the sign correction that matches the
    joints flipped by :func:`flip_joint`. Every ``__call__(control, sim_time)``
    writes the current frame's targets into ``control.joint_target_q``.

    Requires :data:`newton.use_coord_layout_targets = True` (the paper
    accuracy-benchmark scaffolding sets this in :func:`build_benchmark_model`).
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
        if anim.shape[1] != len(_ANIMATION_JOINT_PATHS):
            raise RuntimeError(f"animation has {anim.shape[1]} channels, expected {len(_ANIMATION_JOINT_PATHS)}")
        joint_label = list(model.joint_label)
        joint_q_start = model.joint_q_start.numpy()
        try:
            channel_coords = np.array(
                [joint_q_start[joint_label.index(path)] for path in _ANIMATION_JOINT_PATHS],
                dtype=np.int64,
            )
        except ValueError as e:
            raise RuntimeError(f"animation joint not found in model.joint_label: {e}") from e
        n_coord_per_world = model.joint_coord_count // model.world_count
        world_offsets = np.arange(model.world_count, dtype=np.int64) * n_coord_per_world
        # 2-D fancy-index assignment broadcasts a (12,) RHS across worlds.
        self._indices = channel_coords[None, :] + world_offsets[:, None]
        self._data = anim * _ANIMATION_CHANNEL_SIGN[None, :]
        self._target_q_host = control.joint_target_q.numpy()
        self._animation_dt = float(animation_dt)
        self._animation_speed = float(animation_speed)

    def __call__(self, control: newton.Control, sim_time: float) -> None:
        n_frames = self._data.shape[0]
        frame = min(int(sim_time * self._animation_speed / self._animation_dt), n_frames - 1)
        self._target_q_host[self._indices] = self._data[frame]
        control.joint_target_q.assign(self._target_q_host)


def _call_animation(*, control, sim_time, animation: _DrLegsAnimation) -> None:
    """Adapter matching :class:`SetupRunner`'s ``control_cb`` signature."""
    animation(control, sim_time)


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


###
# Setup factories
###


def make_setup_kamino(
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
        scale_pd_gains(builder, kp_scale, kd_scale)

    builder, model = build_benchmark_model(solvers.SolverKamino, scene, rigid_contact_max=rigid_contact_max)
    solver = solvers.SolverKamino(model=model, config=_kamino_config_dr_legs())
    set_kamino_joint_armature_damping(solver, armature=joint_armature, damping=animation_passive_damping)
    setup = make_paper_setup(
        name="kamino",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=make_kamino_reset_cb(solver),
    )
    attach_kamino_aux_logger(setup, dt, max_log_frames)
    return setup


def make_setup_mujoco(
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
        scale_pd_gains(builder, kp_scale, kd_scale)

    builder, model = build_benchmark_model(solvers.SolverMuJoCo, scene, rigid_contact_max=rigid_contact_max)
    if animation:
        set_mujoco_passive_damping(model, animation_passive_damping)
    solver = solvers.SolverMuJoCo(model, nconmax=rigid_contact_max, **mujoco_default_kwargs())
    return make_paper_setup(
        name="mujoco",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=make_fk_reset_cb(model),
    )


def make_setup_xpbd(
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
        inflate_body_inertia(builder, xpbd_body_armature)

    builder, model = build_benchmark_model(solvers.SolverXPBD, scene, rigid_contact_max=rigid_contact_max)
    solver = solvers.SolverXPBD(model, **xpbd_default_kwargs())
    return make_paper_setup(
        name="xpbd",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=make_fk_reset_cb(model),
    )


def make_reference_leader(
    *,
    solver_name: str,
    dt: float,
    rigid_contact_max: int,
    usd_stage: Usd.Stage,
    animation: bool,
    animation_gain_scale: float,
    animation_kd_scale: float,
    animation_passive_damping: float,
    joint_armature: float,
    xpbd_body_armature: float,
) -> ReferenceLeader:
    """Build a fine-dt :class:`ReferenceLeader` on the DR Legs scene.

    Every per-solver knob available for the coarse follower is honored so the
    fine leader and its matching coarse follower agree apart from step size.
    """
    kp_scale = animation_gain_scale if animation else 1.0
    kd_scale = animation_kd_scale if animation else 1.0

    if solver_name == "kamino":

        def scene(builder: ModelBuilder) -> None:
            solvers.SolverMuJoCo.register_custom_attributes(builder)
            _scene_dr_legs(builder, usd_stage)
            scale_pd_gains(builder, kp_scale, kd_scale)

        _, model = build_benchmark_model(solvers.SolverKamino, scene, rigid_contact_max=rigid_contact_max)
        solver = solvers.SolverKamino(model=model, config=_kamino_config_dr_legs())
        set_kamino_joint_armature_damping(solver, armature=joint_armature, damping=animation_passive_damping)
        reset_cb = make_kamino_reset_cb(solver)
    elif solver_name == "mujoco":

        def scene(builder: ModelBuilder) -> None:
            _scene_dr_legs(builder, usd_stage)
            scale_pd_gains(builder, kp_scale, kd_scale)

        _, model = build_benchmark_model(solvers.SolverMuJoCo, scene, rigid_contact_max=rigid_contact_max)
        if animation:
            set_mujoco_passive_damping(model, animation_passive_damping)
        solver = solvers.SolverMuJoCo(model, nconmax=rigid_contact_max, **mujoco_default_kwargs())
        reset_cb = make_fk_reset_cb(model)
    elif solver_name == "xpbd":

        def scene(builder: ModelBuilder) -> None:
            _scene_dr_legs(builder, usd_stage)
            _set_xpbd_pd_gains(builder, kp_scale, kd_scale)
            inflate_body_inertia(builder, xpbd_body_armature)

        _, model = build_benchmark_model(solvers.SolverXPBD, scene, rigid_contact_max=rigid_contact_max)
        solver = solvers.SolverXPBD(model, **xpbd_default_kwargs())
        reset_cb = make_fk_reset_cb(model)
    else:
        raise ValueError(f"unknown solver_name {solver_name!r}, expected kamino / mujoco / xpbd")

    return ReferenceLeader(name=f"{solver_name}_ref", model=model, solver=solver, dt=dt, reset_cb=reset_cb)


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
    reference_leader_solver: str | None = None,
    reference_leader_fine_substeps: int | None = None,
) -> ProblemRun:
    """Build the floating-base DR Legs :class:`ProblemRun`.

    Optionally drives 12 USD-actuated joints from the bundled 100 Hz animation
    (``animation=True``). The animation is applied by a
    :class:`SetupRunner`-level ``control_cb`` closure so all setups share the
    same coord-space PD targets.

    See :func:`example_benchmark_robot_ironman.build_ironman_run` for the
    ``reference_leader_*`` kwarg semantics.
    """
    usd_stage = _get_usd_stage()

    # DR Legs' XPBD path is tuned separately from the others; log the applied
    # scales once so the paper runs are self-describing.
    msg.notif("DR Legs: animation=%s speed=%s dt=%s", animation, animation_speed, animation_dt)
    msg.notif(
        "DR Legs: gain_scale=%s kd_scale=%s passive_damping=%s",
        animation_gain_scale,
        animation_kd_scale,
        animation_passive_damping,
    )

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
        "kamino": make_setup_kamino(
            **kwargs_common,
            animation_passive_damping=animation_passive_damping,
            joint_armature=joint_armature,
        ),
        "mujoco": make_setup_mujoco(
            **kwargs_common,
            animation_passive_damping=animation_passive_damping,
        ),
        "xpbd": make_setup_xpbd(
            **kwargs_common,
            xpbd_body_armature=xpbd_body_armature,
        ),
    }

    force_cb = make_pelvis_push_cb(
        setups["kamino"],
        body_label=_FORCED_BODY_LABEL,
        force_scale=_FORCE_SCALE,
        force_start_time=_FORCE_WINDOW[0],
        force_stop_time=_FORCE_WINDOW[1],
    )

    reference_leader = None
    if reference_leader_solver is not None:
        if reference_leader_fine_substeps is None or reference_leader_fine_substeps < 1:
            raise ValueError("reference_leader_solver requires reference_leader_fine_substeps >= 1")
        reference_leader = make_reference_leader(
            solver_name=reference_leader_solver,
            dt=dt / reference_leader_fine_substeps,
            rigid_contact_max=rigid_contact_max,
            usd_stage=usd_stage,
            animation=animation,
            animation_gain_scale=animation_gain_scale,
            animation_kd_scale=animation_kd_scale,
            animation_passive_damping=animation_passive_damping,
            joint_armature=joint_armature,
            xpbd_body_armature=xpbd_body_armature,
        )

    return ProblemRun(
        setups=setups,
        force_cb=force_cb,
        camera=(wp.vec3(5.0, 5.0, 1.0), -5.0, 180.0 + 48.0),
        reference_leader=reference_leader,
    )


SPEC = ExampleSpec(
    build_fn=build_dr_legs_run,
    build_kwargs={},
    sim_stop_time=SIM_STOP_TIME,
    problem_name="benchmark_robot_dr_legs",
)


###
# Standalone entry point
###


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument("--leader", type=str, default="kamino", help="Leader solver name (tied / tied_reference mode).")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("tied", "independent", "tied_reference"),
        default="independent",
        help="Runner comparison mode.",
    )
    parser.add_argument(
        "--reference-leader",
        type=str,
        choices=("kamino", "mujoco", "xpbd"),
        default="kamino",
        help="Solver used as the fine-dt reference leader (tied_reference mode only).",
    )
    parser.add_argument(
        "--fine-substeps",
        type=int,
        default=10,
        help="Fine-dt reference-leader substep count per coarse sub-step (tied_reference mode only).",
    )
    parser.add_argument(
        "--use-external-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the pelvis-push external force during the configured time window.",
    )
    parser.add_argument(
        "--animation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drive the 12 USD-actuated joints from the bundled 100 Hz .npy file.",
    )
    parser.add_argument("--animation-dt", type=float, default=0.01, help="Animation time step (s).")
    parser.add_argument(
        "--animation-speed",
        type=float,
        default=0.5,
        help=(
            "Animation playback rate; 1.0 plays the gait at the authored 100 Hz. Defaults to 0.5"
            " because the open-loop gait is reactively unstable for pure XPBD position drives."
        ),
    )
    parser.add_argument("--animation-gain-scale", type=float, default=1.2, help="Multiplier on USD-authored kp.")
    parser.add_argument(
        "--animation-kd-scale",
        type=float,
        default=1.0,
        help="Optional separate multiplier on USD kd (defaults to following --animation-gain-scale).",
    )
    parser.add_argument(
        "--animation-passive-damping",
        type=float,
        default=0.5,
        help="Passive joint damping (N.m.s/rad) applied to every non-base DoF.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(headless=True, num_frames=None)
    return parser


if __name__ == "__main__":
    msg.set_log_level(msg.LogLevel.INFO)
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)

    parser = create_parser()
    viewer, args = newton.examples.init(parser)

    frame_dt = 1.0 / VIZ_FPS
    sim_substeps = max(1, round(frame_dt / SIM_DT))
    sim_dt = frame_dt / sim_substeps
    num_frames = args.num_frames if args.num_frames is not None else max(1, math.ceil(SIM_STOP_TIME * VIZ_FPS))
    max_log_frames = num_frames * sim_substeps
    msg.notif(
        "DR Legs: num_frames=%s, sim_substeps=%s, sim_dt=%.6fs, total_sim_time=%.3fs, mode=%s",
        num_frames,
        sim_substeps,
        sim_dt,
        num_frames * frame_dt,
        args.mode,
    )

    reference_leader_solver = args.reference_leader if args.mode == MODE_TIED_REFERENCE else None
    reference_leader_fine_substeps = args.fine_substeps if args.mode == MODE_TIED_REFERENCE else None
    run = build_dr_legs_run(
        dt=sim_dt,
        max_log_frames=max_log_frames,
        animation=args.animation,
        animation_dt=args.animation_dt,
        animation_speed=args.animation_speed,
        animation_gain_scale=args.animation_gain_scale,
        animation_kd_scale=args.animation_kd_scale,
        animation_passive_damping=args.animation_passive_damping,
        reference_leader_solver=reference_leader_solver,
        reference_leader_fine_substeps=reference_leader_fine_substeps,
    )
    animation_cb = None
    if args.animation:
        animation_cb = make_dr_legs_animation_cb(
            run.setups,
            animation_dt=args.animation_dt,
            animation_speed=args.animation_speed,
        )

    runner = SetupRunner(
        setups=run.setups,
        leader=args.leader,
        viewer=viewer if not args.headless else None,
        force_cb=run.force_cb if args.use_external_force else None,
        control_cb=animation_cb,
        fps=VIZ_FPS,
        sim_substeps=sim_substeps,
        verbose=args.verbose,
        mode=args.mode,
        reference_leader=run.reference_leader,
        fine_substeps_per_coarse=args.fine_substeps,
    )

    if run.camera is not None and hasattr(viewer, "set_camera"):
        viewer.set_camera(*run.camera)

    if args.headless:
        runner.run_headless(num_frames=num_frames)
    else:
        if args.viewer == "gl":
            viewer._paused = True
        newton.examples.run(runner, args)

    runner.test_final(problem_name=SPEC.problem_name)
