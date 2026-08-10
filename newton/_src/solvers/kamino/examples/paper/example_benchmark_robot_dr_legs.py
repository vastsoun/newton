# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-benchmark entry point: floating-base Disney Research legs on a ground plane.

Optionally drives 12 USD-actuated joints from the bundled 100 Hz animation
buffer (``--animation``). USD preprocessing (body0/body1 flips and loop-closure
markings) plus per-solver PD gain scaling and inertia regularization all live
in :mod:`paper_problems`.

The default configuration is headless data collection (progress bar + csv / pdf
export + console summary). Pass ``--no-headless`` to open a viewer.

Run with::

    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_robot_dr_legs
"""

import argparse
import math

import numpy as np

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark import SetupRunner
from newton._src.solvers.kamino.accuracy_benchmark.paper_problems import (
    build_dr_legs_run,
    make_dr_legs_animation_cb,
)

SIM_DT: float = 0.001
VIZ_FPS: int = 50
SIM_STOP_TIME: float = 4.0


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
        "DR Legs: num_frames=%s, sim_substeps=%s, sim_dt=%.6fs, total_sim_time=%.3fs",
        num_frames,
        sim_substeps,
        sim_dt,
        num_frames * frame_dt,
    )

    run = build_dr_legs_run(
        dt=sim_dt,
        max_log_frames=max_log_frames,
        animation=args.animation,
        animation_dt=args.animation_dt,
        animation_speed=args.animation_speed,
        animation_gain_scale=args.animation_gain_scale,
        animation_kd_scale=args.animation_kd_scale,
        animation_passive_damping=args.animation_passive_damping,
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

    runner.test_final(problem_name="benchmark_robot_dr_legs")
