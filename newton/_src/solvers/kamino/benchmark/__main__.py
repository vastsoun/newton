# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino benchmark entry point. Dispatches a ``--problem`` selection to one of the
:class:`ProblemRun` factories in :mod:`problems` and hands the result to
:class:`SetupRunner` (which is the example passed to :func:`newton.examples.run`).

Run with: python -m newton._src.solvers.kamino.benchmark [--problem box_on_plane | fourbar]
"""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.benchmark.problems import (
    build_box_on_plane_run,
    build_fourbar_run,
)
from newton._src.solvers.kamino.benchmark.setup import SetupRunner


def create_parser():
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--problem",
        choices=("box_on_plane", "fourbar"),
        default="box_on_plane",
        help="Which benchmark problem to run.",
    )
    parser.add_argument(
        "--use-external-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a time-dependent external force to the box. Only meaningful for --problem box_on_plane.",
    )
    return parser


if __name__ == "__main__":
    msg.set_log_level(msg.LogLevel.INFO)
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)

    parser = create_parser()
    viewer, args = newton.examples.init(parser)

    fps = 50
    sim_substeps = 20
    sim_dt = (1.0 / fps) / sim_substeps
    max_log_frames = args.num_frames * sim_substeps

    if args.problem == "box_on_plane":
        run = build_box_on_plane_run(
            dt=sim_dt,
            max_log_frames=max_log_frames,
            use_external_force=args.use_external_force,
        )
    elif args.problem == "fourbar":
        run = build_fourbar_run(dt=sim_dt, max_log_frames=max_log_frames)

    runner = SetupRunner(
        setups=run.setups,
        leader="kamino",
        viewer=viewer,
        force_cb=run.force_cb,
        fps=fps,
        sim_substeps=sim_substeps,
    )

    # ``_paused`` only makes sense for ViewerGL; a headless/null run would freeze with no way to unpause.
    if args.viewer == "gl" and not args.headless:
        viewer._paused = True
    if run.camera is not None and hasattr(viewer, "set_camera"):
        viewer.set_camera(*run.camera)

    newton.examples.run(runner, args)
    runner.test_final(problem_name=args.problem)
