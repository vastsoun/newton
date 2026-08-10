# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-benchmark entry point: fixed-base Iron Man articulation.

Wires the shared :mod:`accuracy_benchmark` scaffolding to a :func:`build_ironman_run`
:class:`ProblemRun`. Every setup carries a :class:`PhysicsMetricsLogger`
(cross-solver residuals) and the Kamino setup additionally carries a
:class:`SolverKaminoLogger` (PADMM diagnostics); both are surfaced by
:meth:`SetupRunner.test_final`.

The default configuration is headless data collection (progress bar +
csv / pdf export + console summary). Pass ``--no-headless`` (and typically
``--viewer=gl``) to open a viewer for visual inspection of the scene.

Run with::

    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_robot_ironman
"""

import argparse
import math

import numpy as np

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark import SetupRunner
from newton._src.solvers.kamino.accuracy_benchmark.problems import build_ironman_run

SIM_DT: float = 0.001
VIZ_FPS: int = 50
SIM_STOP_TIME: float = 5.0


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

    runner.test_final(problem_name="benchmark_robot_ironman")
