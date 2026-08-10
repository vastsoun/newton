# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-benchmark entry point: box-on-plane with a time-windowed sliding force.

Reuses the existing :func:`build_box_on_plane_run` factory from
:mod:`accuracy_benchmark.problems` (also used by :mod:`accuracy_benchmark.__main__`).
Distinct from that entry point in that the scaffolding also allocates a
:class:`PhysicsMetrics` + :class:`PhysicsMetricsLogger` pair per setup for
cross-solver contact-residual comparison.

The default configuration is headless data collection (progress bar + csv / pdf
export + console summary). Pass ``--no-headless`` to open a viewer.

Run with::

    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_basic_box_on_plane
"""

import argparse
import math

import numpy as np

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark import SetupRunner
from newton._src.solvers.kamino.accuracy_benchmark.problems import build_box_on_plane_run

SIM_DT: float = 0.001
VIZ_FPS: int = 50
SIM_STOP_TIME: float = 3.0


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
        "--use-external-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the sliding external force during the configured time window.",
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
        "Box-on-plane: num_frames=%s, sim_substeps=%s, sim_dt=%.6fs, total_sim_time=%.3fs",
        num_frames,
        sim_substeps,
        sim_dt,
        num_frames * frame_dt,
    )

    run = build_box_on_plane_run(
        dt=sim_dt,
        max_log_frames=max_log_frames,
        use_external_force=args.use_external_force,
    )

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

    runner.test_final(problem_name="benchmark_basic_box_on_plane")
