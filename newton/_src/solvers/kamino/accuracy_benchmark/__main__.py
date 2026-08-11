# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino accuracy-benchmark entry point.

Dispatches a ``--problem`` selection to one of the per-example ``build_*_run``
factories hosted in the ``examples/paper/`` scripts and hands the result to
:class:`SetupRunner` (the example object passed to :func:`newton.examples.run`).

Run with: python -m newton._src.solvers.kamino.accuracy_benchmark [--problem <name>]

Supported problems: ``ironman`` / ``olaf`` / ``bdx`` / ``dr_legs``. All four
depend on USD assets under the path returned by
:func:`assets.paper_assets_root` (override via
``$NEWTON_KAMINO_PAPER_ASSETS_ROOT``).
"""

import argparse

import numpy as np

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark.setup import MODE_TIED_REFERENCE, SetupRunner
from newton._src.solvers.kamino.examples.paper.example_benchmark_robot_bdx import build_bdx_run
from newton._src.solvers.kamino.examples.paper.example_benchmark_robot_dr_legs import (
    build_dr_legs_run,
    make_dr_legs_animation_cb,
)
from newton._src.solvers.kamino.examples.paper.example_benchmark_robot_ironman import build_ironman_run
from newton._src.solvers.kamino.examples.paper.example_benchmark_robot_olaf import build_olaf_run

_PAPER_PROBLEMS = ("ironman", "olaf", "bdx", "dr_legs")


def create_parser():
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--problem",
        choices=_PAPER_PROBLEMS,
        default="ironman",
        help="Which paper problem to run.",
    )
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
        help="Apply a time-dependent external force. Ignored by ``ironman``.",
    )
    parser.add_argument(
        "--animation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drive the DR Legs gait from the bundled 100 Hz .npy file. Ignored by other problems.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Log per-sub-step state and contact diagnostics for every setup. "
            "Very noisy with the default sim_substeps=20; pair with a small --num-frames."
        ),
    )
    parser.set_defaults(headless=True)
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

    tied_ref = args.mode == MODE_TIED_REFERENCE
    ref_kwargs = (
        {"reference_leader_solver": args.reference_leader, "reference_leader_fine_substeps": args.fine_substeps}
        if tied_ref
        else {}
    )

    animation_cb = None
    if args.problem == "ironman":
        run = build_ironman_run(dt=sim_dt, max_log_frames=max_log_frames, **ref_kwargs)
    elif args.problem == "olaf":
        run = build_olaf_run(dt=sim_dt, max_log_frames=max_log_frames, **ref_kwargs)
    elif args.problem == "bdx":
        run = build_bdx_run(dt=sim_dt, max_log_frames=max_log_frames, **ref_kwargs)
    elif args.problem == "dr_legs":
        run = build_dr_legs_run(
            dt=sim_dt,
            max_log_frames=max_log_frames,
            animation=args.animation,
            **ref_kwargs,
        )
        if args.animation:
            animation_cb = make_dr_legs_animation_cb(run.setups)

    runner = SetupRunner(
        setups=run.setups,
        leader="kamino",
        viewer=viewer if not args.headless else None,
        force_cb=run.force_cb if args.use_external_force else None,
        control_cb=animation_cb,
        fps=fps,
        sim_substeps=sim_substeps,
        verbose=args.verbose,
        mode=args.mode,
        reference_leader=run.reference_leader,
        fine_substeps_per_coarse=args.fine_substeps,
    )

    if run.camera is not None and hasattr(viewer, "set_camera"):
        viewer.set_camera(*run.camera)

    if args.headless:
        runner.run_headless(num_frames=args.num_frames)
    else:
        # ``_paused`` only makes sense for ViewerGL; a headless/null run would freeze with no way to unpause.
        if args.viewer == "gl":
            viewer._paused = True
        newton.examples.run(runner, args)

    runner.test_final(problem_name=args.problem)
