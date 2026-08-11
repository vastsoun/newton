# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-benchmark entry point: fixed-base Iron Man articulation.

Builds Kamino / MuJoCo / XPBD :class:`SolverSetup` triplets for the fixed-base
Iron Man articulation and drives them through the shared :class:`SetupRunner`.
Every setup carries a :class:`PhysicsMetricsLogger` (cross-solver residuals)
and the Kamino setup additionally carries a :class:`SolverKaminoLogger` (PADMM
diagnostics); both are surfaced by :meth:`SetupRunner.test_final`.

The default configuration is headless data collection (progress bar +
csv / pdf export + console summary). Pass ``--no-headless`` (and typically
``--viewer=gl``) to open a viewer for visual inspection of the scene.

Run with::

    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_robot_ironman
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import solvers
from newton._src.sim import ModelBuilder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark import SetupRunner
from newton._src.solvers.kamino.accuracy_benchmark.assets import resolve_asset
from newton._src.solvers.kamino.accuracy_benchmark.problems import (
    FRICTION,
    RESTITUTION,
    ExampleSpec,
    ProblemRun,
    apply_default_builder_cfg,
    attach_kamino_aux_logger,
    build_benchmark_model,
    kamino_default_config,
    make_fk_reset_cb,
    make_kamino_reset_cb,
    make_paper_setup,
    mujoco_default_kwargs,
    xpbd_default_kwargs,
)
from newton._src.solvers.kamino.accuracy_benchmark.setup import SolverSetup

SIM_DT: float = 0.001
VIZ_FPS: int = 50
SIM_STOP_TIME: float = 5.0

_ASSET_RELPATH = "usda/iron_man_fixed_hands_no_shell/iron_man_fixed_hands_no_shell_articulation.usda"


###
# Scene / setup factories
###


def _scene_ironman(builder: ModelBuilder) -> None:
    """Populate ``builder`` with the fixed-base Iron Man articulation.

    No ``floating=True`` and no ground plane: gravity acts on the articulated
    parts while the base stays clamped to the world.
    """
    apply_default_builder_cfg(builder, FRICTION, RESTITUTION)
    builder.add_usd(
        resolve_asset(_ASSET_RELPATH),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )


def make_setup_kamino(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = build_benchmark_model(solvers.SolverKamino, _scene_ironman, rigid_contact_max=rigid_contact_max)
    cfg = kamino_default_config()
    solver = solvers.SolverKamino(model=model, config=cfg)
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


def make_setup_mujoco(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = build_benchmark_model(solvers.SolverMuJoCo, _scene_ironman, rigid_contact_max=rigid_contact_max)
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


def make_setup_xpbd(*, dt: float, max_log_frames: int, rigid_contact_max: int) -> SolverSetup:
    builder, model = build_benchmark_model(solvers.SolverXPBD, _scene_ironman, rigid_contact_max=rigid_contact_max)
    xpbd_kwargs = xpbd_default_kwargs()
    # Iron Man is unusually stiff; the paper uses 200 XPBD iterations to make
    # its residuals comparable to the other solvers rather than the default 2.
    xpbd_kwargs["iterations"] = 200
    solver = solvers.SolverXPBD(model, **xpbd_kwargs)
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


def build_ironman_run(*, dt: float, max_log_frames: int, rigid_contact_max: int = 128) -> ProblemRun:
    """Build the fixed-base Iron Man :class:`ProblemRun` (Kamino / MuJoCo / XPBD)."""
    kwargs = {"dt": dt, "max_log_frames": max_log_frames, "rigid_contact_max": rigid_contact_max}
    setups = {
        "kamino": make_setup_kamino(**kwargs),
        "mujoco": make_setup_mujoco(**kwargs),
        "xpbd": make_setup_xpbd(**kwargs),
    }
    return ProblemRun(
        setups=setups,
        force_cb=None,
        camera=(wp.vec3(5.0, 5.0, 1.0), -5.0, 180.0 + 48.0),
    )


SPEC = ExampleSpec(
    build_fn=build_ironman_run,
    build_kwargs={},
    sim_stop_time=SIM_STOP_TIME,
    problem_name="benchmark_robot_ironman",
)


###
# Standalone entry point
###


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument("--leader", type=str, default="kamino", help="Leader solver name (tied mode only).")
    parser.add_argument(
        "--independent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run each solver on its own trajectory (position-level residual accumulation). "
            "Pass --no-independent to run all solvers tied to the leader (single-step accuracy)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-sub-step diagnostics for every setup (noisy; pair with a small --num-frames).",
    )
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
    # ``num_frames`` counts display frames (each = ``sim_substeps`` physics sub-steps at ``sim_dt``).
    # ``--num-frames`` on the CLI wins; otherwise round ``SIM_STOP_TIME`` up to the nearest frame,
    # with a floor of 1 so short SIM_STOP_TIME values still execute at least one full frame.
    num_frames = args.num_frames if args.num_frames is not None else max(1, math.ceil(SIM_STOP_TIME * VIZ_FPS))
    max_log_frames = num_frames * sim_substeps
    msg.notif(
        "Ironman: num_frames=%s, sim_substeps=%s, sim_dt=%.6fs, total_sim_time=%.3fs",
        num_frames,
        sim_substeps,
        sim_dt,
        num_frames * frame_dt,
    )

    run = build_ironman_run(dt=sim_dt, max_log_frames=max_log_frames)

    runner = SetupRunner(
        setups=run.setups,
        leader=args.leader,
        viewer=viewer if not args.headless else None,
        force_cb=run.force_cb,
        fps=VIZ_FPS,
        sim_substeps=sim_substeps,
        verbose=args.verbose,
        independent=args.independent,
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
