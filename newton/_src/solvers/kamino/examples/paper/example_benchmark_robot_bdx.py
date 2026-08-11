# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-benchmark entry point: floating-base BDX bipedal on a ground plane.

Applies a time-windowed pelvis push and compares Kamino / MuJoCo / XPBD
residuals. BDX uses the reverse-Cuthill-McKee ordering for the LLT-blocked
linear solver and skips the Kamino aux logger (PADMM diagnostics not reported
in the paper).

The default configuration is headless data collection (progress bar + csv / pdf
export + console summary). Pass ``--no-headless`` to open a viewer.

Run with::

    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_robot_bdx
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp
from pxr import Usd, UsdPhysics  # noqa TID253

import newton
import newton.examples
from newton import solvers
from newton._src.sim import ModelBuilder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark import MODE_TIED_REFERENCE, SetupRunner
from newton._src.solvers.kamino.accuracy_benchmark.assets import resolve_asset
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
    get_prim,
    kamino_default_config,
    make_fk_reset_cb,
    make_kamino_reset_cb,
    make_paper_setup,
    make_pelvis_push_cb,
    mujoco_default_kwargs,
    xpbd_default_kwargs,
)
from newton._src.solvers.kamino.accuracy_benchmark.setup import SolverSetup


def _kamino_config_bdx() -> solvers.SolverKamino.Config:
    """Kamino config shared by BDX coarse-follower and reference-leader instances."""
    cfg = kamino_default_config()
    # BDX uses the reverse-Cuthill-McKee ordering for the LLT-blocked solver
    # and a tighter iteration budget.
    cfg.dynamics.linear_solver_type = "LLTBRCM"
    cfg.padmm.max_iterations = 200
    return cfg


SIM_DT: float = 0.001
VIZ_FPS: int = 50
SIM_STOP_TIME: float = 5.0

_ASSET_RELPATH = "usda/bdx/bipedal.usda"
_ARTICULATION_ROOT_PATH = "/BD_9002_001209/PELVIS"
_FORCED_BODY_LABEL = _ARTICULATION_ROOT_PATH
_FORCE_SCALE = 120.0
_FORCE_WINDOW = (1.0, 3.0)
_START_Z = 0.5


###
# Scene / setup factories
###


def _get_usd_stage() -> Usd.Stage:
    """Open the BDX USD and mark the pelvis as the articulation root.

    Upstream ``bipedal.usda`` has no ``PhysicsArticulationRootAPI`` authored,
    so ``add_usd`` would import every body as a free-flying root connected by
    orphan joints. Applying the API here promotes the whole tree to a single
    articulation rooted at the pelvis. The USD is already a joint tree (no
    loop closures) so no ``excludeFromArticulation`` markers are needed.
    """
    stage = Usd.Stage.Open(resolve_asset(_ASSET_RELPATH))
    if stage is None:
        raise RuntimeError(f"Failed to open BDX USD stage: {_ASSET_RELPATH}")
    UsdPhysics.ArticulationRootAPI.Apply(get_prim(stage, _ARTICULATION_ROOT_PATH))
    return stage


def _scene_bdx(builder: ModelBuilder, usd_stage: Usd.Stage) -> None:
    """Populate ``builder`` with a floating-base BDX bipedal on a ground plane."""
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


def make_setup_kamino(*, dt: float, max_log_frames: int, rigid_contact_max: int, usd_stage: Usd.Stage) -> SolverSetup:
    builder, model = build_benchmark_model(
        solvers.SolverKamino,
        _scene_bdx,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs={"usd_stage": usd_stage},
    )
    solver = solvers.SolverKamino(model=model, config=_kamino_config_bdx())
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


def make_setup_mujoco(*, dt: float, max_log_frames: int, rigid_contact_max: int, usd_stage: Usd.Stage) -> SolverSetup:
    builder, model = build_benchmark_model(
        solvers.SolverMuJoCo,
        _scene_bdx,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs={"usd_stage": usd_stage},
    )
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


def make_setup_xpbd(*, dt: float, max_log_frames: int, rigid_contact_max: int, usd_stage: Usd.Stage) -> SolverSetup:
    builder, model = build_benchmark_model(
        solvers.SolverXPBD,
        _scene_bdx,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs={"usd_stage": usd_stage},
    )
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
    max_log_frames: int | None = None,
    log_dt: float | None = None,
) -> ReferenceLeader:
    """Build a fine-dt :class:`ReferenceLeader` on the BDX scene.

    When both ``max_log_frames`` and ``log_dt`` are provided, attaches a
    :class:`PhysicsMetricsLogger` so the reference's fine-dt single-step
    residuals overlay the followers' in the CSV/PDF outputs.
    """
    scene_kwargs = {"usd_stage": usd_stage}
    if solver_name == "kamino":
        _, model = build_benchmark_model(
            solvers.SolverKamino, _scene_bdx, rigid_contact_max=rigid_contact_max, scene_kwargs=scene_kwargs
        )
        solver = solvers.SolverKamino(model=model, config=_kamino_config_bdx())
        reset_cb = make_kamino_reset_cb(solver)
    elif solver_name == "mujoco":
        _, model = build_benchmark_model(
            solvers.SolverMuJoCo, _scene_bdx, rigid_contact_max=rigid_contact_max, scene_kwargs=scene_kwargs
        )
        solver = solvers.SolverMuJoCo(model, nconmax=rigid_contact_max, **mujoco_default_kwargs())
        reset_cb = make_fk_reset_cb(model)
    elif solver_name == "xpbd":
        _, model = build_benchmark_model(
            solvers.SolverXPBD, _scene_bdx, rigid_contact_max=rigid_contact_max, scene_kwargs=scene_kwargs
        )
        solver = solvers.SolverXPBD(model, **xpbd_default_kwargs())
        reset_cb = make_fk_reset_cb(model)
    else:
        raise ValueError(f"unknown solver_name {solver_name!r}, expected kamino / mujoco / xpbd")
    leader = ReferenceLeader(name=f"{solver_name}_ref", model=model, solver=solver, dt=dt, reset_cb=reset_cb)
    if max_log_frames is not None and log_dt is not None:
        leader.attach_metrics(max_log_frames=max_log_frames, log_dt=log_dt)
    return leader


def build_bdx_run(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int = 128,
    reference_leader_solver: str | None = None,
    reference_leader_fine_substeps: int | None = None,
) -> ProblemRun:
    """Build the floating-base BDX :class:`ProblemRun` (with a pelvis push).

    See :func:`example_benchmark_robot_ironman.build_ironman_run` for the
    ``reference_leader_*`` kwarg semantics.
    """
    usd_stage = _get_usd_stage()
    kwargs = {
        "dt": dt,
        "max_log_frames": max_log_frames,
        "rigid_contact_max": rigid_contact_max,
        "usd_stage": usd_stage,
    }
    setups = {
        "kamino": make_setup_kamino(**kwargs),
        "mujoco": make_setup_mujoco(**kwargs),
        "xpbd": make_setup_xpbd(**kwargs),
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
            max_log_frames=max_log_frames,
            log_dt=dt,
        )
    return ProblemRun(
        setups=setups,
        force_cb=force_cb,
        camera=(wp.vec3(5.0, 5.0, 1.0), -5.0, 180.0 + 48.0),
        reference_leader=reference_leader,
    )


SPEC = ExampleSpec(
    build_fn=build_bdx_run,
    build_kwargs={},
    sim_stop_time=SIM_STOP_TIME,
    problem_name="benchmark_robot_bdx",
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
        "BDX: num_frames=%s, sim_substeps=%s, sim_dt=%.6fs, total_sim_time=%.3fs, mode=%s",
        num_frames,
        sim_substeps,
        sim_dt,
        num_frames * frame_dt,
        args.mode,
    )

    reference_leader_solver = args.reference_leader if args.mode == MODE_TIED_REFERENCE else None
    reference_leader_fine_substeps = args.fine_substeps if args.mode == MODE_TIED_REFERENCE else None
    run = build_bdx_run(
        dt=sim_dt,
        max_log_frames=max_log_frames,
        reference_leader_solver=reference_leader_solver,
        reference_leader_fine_substeps=reference_leader_fine_substeps,
    )

    runner = SetupRunner(
        setups=run.setups,
        leader=args.leader,
        viewer=viewer if not args.headless else None,
        force_cb=run.force_cb if args.use_external_force else None,
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
