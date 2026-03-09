# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import os

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.benchmark.configs import make_benchmark_configs
from newton._src.solvers.kamino._src.utils.benchmark.metrics import BenchmarkMetrics, CodeInfo
from newton._src.solvers.kamino._src.utils.benchmark.problems import (
    BenchmarkProblemNameToConfigFn,
    make_benchmark_problems,
)
from newton._src.solvers.kamino._src.utils.benchmark.render import (
    render_problem_dimensions_table,
    render_solver_configs_table,
)
from newton._src.solvers.kamino._src.utils.benchmark.runner import run_single_benchmark
from newton._src.solvers.kamino._src.utils.device import get_device_spec_info
from newton._src.solvers.kamino._src.utils.sim import Simulator

###
# Functions
###


def parse_benchmark_arguments():
    parser = argparse.ArgumentParser(description="Solver performance benchmark")

    # Warp runtime arguments
    parser.add_argument("--device", type=str, help="Define the Warp device to operate on, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set to `True` to enable CUDA graph capture (only available on CUDA devices). Defaults to `True`.",
    )
    parser.add_argument(
        "--clear-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set to `True` to clear Warp's kernel and LTO caches before execution. Defaults to `False`.",
    )

    # Data collection arguments
    parser.add_argument(
        "--max-num-worlds",
        type=int,
        default=200,
        help="Sets the maximum number of parallel simulation worlds to run. Defaults to `8192`.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=3,
        help="Sets the number of points to sample on the profiling curve. Defaults to `20`.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=BenchmarkProblemNameToConfigFn.keys(),
        default="dr_legs",
        help="Name of the problem to run on. Defaults to `dr_legs`",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=[
            "Dense Jacobian LLT accurate",
            "Dense Jacobian LLT fast",
            "Sparse Jacobian LLT accurate",
            "Sparse Jacobian LLT fast",
            "Sparse Delassus CR accurate",
            "Sparse Delassus CR fast",
        ],
        default="Sparse Delassus CR fast",
        help="Name of the solver config to run on. Defaults to `Sparse Delassus CR fast`",
    )

    # Simulator arguments
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Sets the number of simulation steps to execute. Defaults to `100`.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Sets the simulation time step. Defaults to `0.001`.",
    )
    parser.add_argument(
        "--gravity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enables/disables gravity in the simulation. Defaults to `True`.",
    )
    parser.add_argument(
        "--ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enables/disables ground geometry in the simulation. Defaults to `True`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sets the random seed for the simulation. Defaults to `0`.",
    )

    args = parser.parse_args()
    args.viewer = False

    return args


def run_throughput_profiling(args: argparse.Namespace):
    # Check arguments
    max_num_worlds = args.max_num_worlds
    assert max_num_worlds > 1
    num_points = args.num_points
    assert num_points > 1

    # Print the git commit hash and repository info to the
    # console for traceability and reproducibility of benchmark runs
    codeinfo = CodeInfo()
    msg.notif(f"Benchmark will run with the following repository:\n{codeinfo}\n")

    # Set device if specified, otherwise use Warp's default
    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    # Print device specification info to console for reference
    spec_info = get_device_spec_info(device)
    msg.notif("[Device]: %s", spec_info)
    print(f"Mempool release threshold: {wp.get_mempool_release_threshold()}")

    # Determine if CUDA graphs should be used for execution
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    use_cuda_graph = can_use_cuda_graph and args.cuda_graph
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")
    msg.info(f"using_cuda_graph: {use_cuda_graph}")

    # Get the problem and configuration to profile the throughput for
    problem_name = args.problem
    problem = make_benchmark_problems(
        names=[problem_name],
        gravity=args.gravity,
        ground=args.ground,
    )[problem_name]
    configs_set = make_benchmark_configs(include_default=False)
    config_name = args.config
    config = configs_set[config_name]

    # Define and create the output directory for the benchmark results
    DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data"))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_OUTPUT_PATH = f"{DATA_DIR_PATH}/throughput_{problem_name}_{config_name}_{timestamp}"
    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)

    # Print problem and config
    msg.notif("Profiling throughput for problem '%s' with config '%s'", problem_name, config_name)
    render_solver_configs_table(configs={config_name: config}, groups=["sparse", "linear", "padmm"], to_console=True)
    render_solver_configs_table(
        configs={config_name: config},
        path=os.path.join(RUN_OUTPUT_PATH, "solver_configs.txt"),
        groups=["cts", "sparse", "linear", "padmm", "warmstart"],
        to_console=False,
    )

    # Unpack problem
    builder_fn, control, camera = problem

    # Construct simulator configuration
    sim_config = Simulator.Config(dt=args.dt, solver=config)
    sim_config.solver.enable_fk_solver = False

    # Collect timings and memory usage for various number of worlds
    num_worlds_min = 1
    for point_id in range(num_points):
        # Determine number of worlds
        num_worlds = int(
            num_worlds_min + (num_points - point_id - 1) * (max_num_worlds - num_worlds_min) / (num_points - 1)
        )
        msg.notif("Running with %d worlds", num_worlds)

        # Create subfolder
        subfolder_path = os.path.join(RUN_OUTPUT_PATH, f"{num_worlds} worlds")
        os.makedirs(subfolder_path, exist_ok=True)

        # Initialize metrics to store performance data
        metrics = BenchmarkMetrics(
            problems=[problem_name],
            configs={config_name: config},
            num_steps=args.num_steps,
            step_metrics=False,
            solver_metrics=False,
            physics_metrics=False,
        )

        # Run data collection
        sim_config.collision_detector.max_contacts = 10 * num_worlds
        run_single_benchmark(
            problem_idx=0,
            config_idx=0,
            metrics=metrics,
            args=args,
            builder=builder_fn(num_worlds),
            config=sim_config,
            control=control,
            camera=camera,
            device=device,
            use_cuda_graph=use_cuda_graph,
            print_device_info=True,
        )

        # Finalize and export metrics
        metrics.compute_stats()
        msg.info("Saving benchmark data to HDF5...")
        RUN_HDF5_OUTPUT_PATH = f"{subfolder_path}/metrics.hdf5"
        metrics.save_to_hdf5(path=RUN_HDF5_OUTPUT_PATH)
        msg.info("Done.")

        # Print/export results
        benchmark_output(metrics, subfolder_path)

    # Print table with problem dimensions
    render_problem_dimensions_table(metrics._problem_dims, to_console=True)
    render_problem_dimensions_table(
        metrics._problem_dims,
        path=os.path.join(RUN_OUTPUT_PATH, "problem_dimensions.txt"),
        to_console=False,
    )


def benchmark_output(metrics: BenchmarkMetrics, export_dir: str | None):
    # Compute statistics for the collected benchmark
    # data to prepare for plotting and analysis
    metrics.compute_stats()

    # Print the total performance summary as a formatted table to the console:
    # - The columns span the problems, with a sub-column for each
    #   metric (e.g. total time, total FPS, memory used)
    # - The rows span the solver configurations
    total_metrics_table_path = None
    if export_dir is not None:
        total_metrics_table_path = os.path.join(export_dir, "total_metrics.txt")
    metrics.render_total_metrics_table(path=total_metrics_table_path)

    # For each problem, export a table summarizing the step-time for each solver configuration:
    # - A sub-column for each statistic (mean, std, min, max)
    # - The rows span the solver configurations
    if metrics.step_time is not None:
        step_time_summary_path = None
        if export_dir is not None:
            step_time_summary_path = os.path.join(export_dir, "step_time")
        metrics.render_step_time_table(path=step_time_summary_path)

    # For each problem, export a table summarizing the PADMM metrics for each solver configuration:
    # - The columns span the metrics (e.g. step time, padmm.*, physics.*),
    #   with a sub-column for each statistic (mean, std, min, max)
    # - The rows span the solver configurations
    if metrics.solver_metrics is not None:
        padmm_metrics_summary_path = None
        padmm_metrics_plots_path = None
        if export_dir is not None:
            padmm_metrics_summary_path = os.path.join(export_dir, "padmm_metrics")
            padmm_metrics_plots_path = os.path.join(export_dir, "padmm_metrics")
        metrics.render_padmm_metrics_table(path=padmm_metrics_summary_path)
        metrics.render_padmm_metrics_plots(path=padmm_metrics_plots_path)

    # For each problem, export a table summarizing the PADMM metrics for each solver configuration:
    # - The columns span the metrics (e.g. step time, padmm.*, physics.*),
    #   with a sub-column for each statistic (mean, std, min, max)
    # - The rows span the solver configurations
    if metrics.physics_metrics is not None:
        physics_metrics_summary_path = None
        physics_metrics_plots_path = None
        if export_dir is not None:
            physics_metrics_summary_path = os.path.join(export_dir, "physics_metrics")
            physics_metrics_plots_path = os.path.join(export_dir, "physics_metrics")
        metrics.render_physics_metrics_table(path=physics_metrics_summary_path)
        metrics.render_physics_metrics_plots(path=physics_metrics_plots_path)


###
# Main function
###

if __name__ == "__main__":
    # Load benchmark-specific program arguments
    args = parse_benchmark_arguments()

    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation

    # Clear warp cache if requested
    if args.clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # TODO: Make optional
    # Set the verbosity of the global message logger
    msg.set_log_level(msg.LogLevel.INFO)

    # Run throughput profiling
    run_throughput_profiling(args)
