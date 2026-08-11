# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batch runner: all four paper problems, both comparison modes.

Runs Iron Man / DR Legs / BDX / Olaf back-to-back in *tied* and *independent*
mode and writes the per-run CSV / PDF exports into a mode-segregated directory
tree so a downstream table-builder can pick per-metric results from whichever
mode is more appropriate.

Output layout (``--output-dir`` defaults to ``<this_file>/output/paper_all``)::

    <output-dir>/
      tied/
        ironman/  benchmark_robot_ironman_physics_metrics{,_logscale}.pdf
                  benchmark_robot_ironman_physics_metrics_table.csv
                  benchmark_robot_ironman_kamino_aux.pdf   (if kamino aux logger)
        dr_legs/  ...
        bdx/      ...
        olaf/     ...
      independent/
        ironman/  ...
        dr_legs/  ...
        bdx/      ...
        olaf/     ...

Toggle individual runs via ``EXAMPLES_ENABLED`` / ``MODES_ENABLED`` below (edit
the file to overwrite only the outputs you need after tweaking one example),
or override on the CLI::

    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_paper_all
    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_paper_all --examples ironman
    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_paper_all --modes tied
    python -m newton._src.solvers.kamino.examples.paper.example_benchmark_paper_all --examples bdx olaf --modes independent

Each ``(mode, example)`` pair is built and run in a single Python process; the
model / solver / runner are dropped and ``gc.collect()`` is called between
iterations so long batches don't accumulate GPU allocations.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time

import numpy as np

from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.accuracy_benchmark import SetupRunner
from newton._src.solvers.kamino.examples.paper.example_benchmark_robot_bdx import SPEC as BDX_SPEC
from newton._src.solvers.kamino.examples.paper.example_benchmark_robot_dr_legs import SPEC as DR_LEGS_SPEC
from newton._src.solvers.kamino.examples.paper.example_benchmark_robot_ironman import SPEC as IRONMAN_SPEC
from newton._src.solvers.kamino.examples.paper.example_benchmark_robot_olaf import SPEC as OLAF_SPEC

###
# Configuration
###

SIM_DT: float = 0.001
VIZ_FPS: int = 50

# Toggle any (example, mode) pair by flipping the value below to ``False``.
# The CLI ``--examples`` / ``--modes`` flags further restrict this selection.
EXAMPLES_ENABLED: dict[str, bool] = {
    "ironman": True,
    "dr_legs": True,
    "bdx": True,
    "olaf": True,
}
MODES_ENABLED: dict[str, bool] = {
    "tied": True,
    "independent": True,
}


# Per-example specs come from the individual ``example_benchmark_robot_*.py``
# scripts, so ``sim_stop_time`` and ``problem_name`` live in a single place.
_EXAMPLE_SPECS = {
    "ironman": IRONMAN_SPEC,
    "dr_legs": DR_LEGS_SPEC,
    "bdx": BDX_SPEC,
    "olaf": OLAF_SPEC,
}


###
# Runner
###


def _resolve_toggles(defaults: dict[str, bool], override: list[str] | None) -> list[str]:
    """Turn ``(defaults, override)`` into an ordered list of enabled names.

    Preserves the order of ``defaults`` so both the file toggles and the CLI
    subset are printed / iterated deterministically. Raises on unknown names.
    """
    if override is None:
        return [name for name, enabled in defaults.items() if enabled]
    unknown = [name for name in override if name not in defaults]
    if unknown:
        raise SystemExit(f"unknown selection: {unknown}. Known: {list(defaults)}")
    selected = set(override)
    return [name for name in defaults if name in selected]


def _run_one(
    name: str,
    mode: str,
    base_output_dir: str,
    *,
    num_frames_override: int | None = None,
) -> None:
    """Build and drive one ``(example, mode)`` pair, writing artifacts to disk.

    Fresh :class:`ProblemRun` / :class:`SetupRunner` each call — the previous
    pair's builders/models are dropped by the caller before we're invoked, so
    warp / kamino allocations do not accumulate across the batch.
    """
    spec = _EXAMPLE_SPECS[name]
    independent = mode == "independent"

    frame_dt = 1.0 / VIZ_FPS
    sim_substeps = max(1, round(frame_dt / SIM_DT))
    sim_dt = frame_dt / sim_substeps
    num_frames = (
        num_frames_override if num_frames_override is not None else max(1, math.ceil(spec.sim_stop_time * VIZ_FPS))
    )
    max_log_frames = num_frames * sim_substeps

    msg.notif(
        "[%s / %s] num_frames=%s sim_substeps=%s sim_dt=%.6fs total_sim_time=%.3fs",
        name,
        mode,
        num_frames,
        sim_substeps,
        sim_dt,
        num_frames * frame_dt,
    )

    run = spec.build_fn(dt=sim_dt, max_log_frames=max_log_frames, **spec.build_kwargs)

    runner = SetupRunner(
        setups=run.setups,
        leader="kamino",
        viewer=None,
        force_cb=run.force_cb,
        fps=VIZ_FPS,
        sim_substeps=sim_substeps,
        verbose=False,
        independent=independent,
    )
    runner.run_headless(num_frames=num_frames)

    out_dir = os.path.join(base_output_dir, mode, name)
    runner.test_final(problem_name=spec.problem_name, output_path=out_dir)
    msg.notif("[%s / %s] wrote outputs to %s", name, mode, out_dir)


def _default_output_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "paper_all")


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every paper problem in both tied and independent mode, writing artifacts under one root."
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=None,
        help=f"Subset of examples to run (default: honor EXAMPLES_ENABLED). Names: {list(_EXAMPLE_SPECS)}",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        choices=list(MODES_ENABLED),
        help="Subset of modes to run (default: honor MODES_ENABLED).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory. Defaults to <this_file>/output/paper_all.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Override the per-example frame count (useful for smoke tests). Applies to every run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    msg.set_log_level(msg.LogLevel.INFO)
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)

    args = _create_parser().parse_args(argv)
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir()
    examples = _resolve_toggles(EXAMPLES_ENABLED, args.examples)
    modes = _resolve_toggles(MODES_ENABLED, args.modes)
    if not examples or not modes:
        raise SystemExit(f"nothing to run: examples={examples}, modes={modes}")

    msg.notif("Batch plan: modes=%s x examples=%s -> %s", modes, examples, output_dir)
    total = len(modes) * len(examples)
    started = time.time()

    for i, mode in enumerate(modes):
        for j, name in enumerate(examples):
            index = i * len(examples) + j
            msg.notif("=== [%d/%d] %s / %s ===", index + 1, total, name, mode)
            _run_one(name, mode, output_dir, num_frames_override=args.num_frames)
            # Drop everything the run allocated before starting the next pair.
            gc.collect()

    msg.notif("Batch done: %d runs in %.1fs -> %s", total, time.time() - started, output_dir)


if __name__ == "__main__":
    main()
