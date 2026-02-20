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

from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.benchmark.configs import make_benchmark_configs
from newton._src.solvers.kamino.utils.benchmark.metrics import BenchmarkMetrics, CodeInfo
from newton._src.solvers.kamino.utils.benchmark.problems import SUPPORTED_PROBLEM_NAMES, make_benchmark_problems
from newton._src.solvers.kamino.utils.benchmark.runner import run_single_benchmark
from newton._src.solvers.kamino.utils.device import get_device_spec_info
from newton._src.solvers.kamino.utils.sim import SimulatorSettings

###
# DESIGN:
#
#   OUTLINE:
#   1. Supported program arguments
#   2. Generating solver configurations (potentially multiple)
#   3. Pretty-print configurations to console and file
#   4. Execute benchmark runner and collect metrics
#   5. Print results, save to file, and optionally plot
#
#   METRICS:
#   - [x] Memory usage
#   - [x] Total runtime + per-step runtime w/ statistics (mean, std, min, max)
#   - [x] Solver performance metrics (Optional, because of reduced throughput)
#   - [x] Number of PADMM iterations to converge
#   - [x] Final PADMM residuals (primal, dual, compl)
#   - [x] Physical accuracy metrics (e.g. constraint violation, energy drift, etc.)
#
#   ARGUMENTS:
#   - [x] Device selection (e.g. "cuda:0", "cpu")
#   - [x] Number of parallel worlds to simulate
#   - [x] Number of steps to simulate
#   - [x] Gravity on/off
#   - [x] Ground plane on/off
#   - [x] Performance metrics on/off (since it can reduce throughput)
#   - [x] Problem sets (boxes_fourbar, DR Legs, ANYmal, humanoid, etc.)
#
#   CONFIGURATIONS:
#   - Linear solver type (e.g. dense/LLTB, sparse/CG, sparse/CR)
#   - Linear solver max iterations (0 for no limit, only for sparse/CG+CR)
#   - PADMM iterations
#   - PADMM tolerances (primal, dual, compl)
#   - PADMM acceleration on/off
#   - Warm-starting mode (none, contacts, containers)
#   - PADMM initial rho and eta
#   - PADMM rho update strategy (e.g. fixed, balanced)
#
#   FUNCTIONALITY:
#  - [x] Random actuation for each problem (optional)
#  - [x] Separate config generation from execution to allow for easier hyperparameter sweeps and ablations
#  - [x] Store git commit and/or diff for reproducibility
#  - [] Define default configs in appropriate file for reference
#
###

###
# Constants
###

SUPPORTED_BENCHMARK_RUN_MODES = ["total", "perstep", "solver", "accuracy"]
"""
A list of supported benchmark run modes that determine the level of metrics collected during execution.

- "total": Only collects total runtime and final memory usage metrics.
- "perstep": Collects detailed timing metrics for each simulation step to compute throughput statistics.
- "solver": Collects solver performance metrics such as PADMM iterations and residuals.
- "accuracy": Collects solver performance metrics that can be used to evaluate the physical accuracy of the simulation.
"""

# TODO
# SUPPORTED_BENCHMARK_PLOT_MODES = ["console", "perstep", "solver", "accuracy"]
# """
# A list of supported benchmark plot modes that determine the level of metrics collected during execution.

# - "total": Only collects total runtime and final memory usage metrics.
# - "perstep": Collects detailed timing metrics for each simulation step to compute throughput statistics.
# - "solver": Collects solver performance metrics such as PADMM iterations and residuals.
# - "accuracy": Collects solver performance metrics that can be used to evaluate the physical accuracy of the simulation.
# """

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

    # World configuration arguments
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=1,
        help="Sets the number of parallel simulation worlds to run. Defaults to `1`.",
    )
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

    # Benchmark execution arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=SUPPORTED_BENCHMARK_RUN_MODES,
        default="accuracy",
        help="Defines the benchmark mode to run. Defaults to 'total'.\n{SUPPORTED_BENCHMARK_MODES}",
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=SUPPORTED_PROBLEM_NAMES,
        default="fourbar",
        help=f"Defines a single benchmark problem to run. Defaults to 'fourbar'.\nSupported: {SUPPORTED_PROBLEM_NAMES}",
    )
    parser.add_argument(
        "--problem-set",
        nargs="+",
        default=["fourbar", "dr_legs"],
        help="Defines the benchmark problem(s) to run. If unspecified, the default `fourbar` problem will be used.",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set to `True` to run with the simulation viewer. Defaults to `False`.",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set to `True` to run `newton.example.run` tests. Defaults to `False`.",
    )

    # Benchmark plotting arguments (TODO)
    # parser.add_argument(
    #     "--plot-mode",
    #     type=str,
    #     choices=SUPPORTED_BENCHMARK_PLOT_MODES,
    #     default="accuracy",
    #     help="Defines the benchmark mode to run. Defaults to 'total'.\n{SUPPORTED_BENCHMARK_PLOT_MODES}",
    # )

    return parser.parse_args()


def benchmark_run(args: argparse.Namespace):
    """
    Executes the benchmark data generation with the provided arguments.

    This function performs the following steps:
    1. Parses the benchmark arguments to determine the configuration of the run.
    2. Sets the Warp device and determines if CUDA graphs can be used.
    3. Prints device specification info to the console for reference.
    4. Determines the level of metrics to collect based on the specified benchmark mode.
    5. Generates the problem set based on the provided problem names and arguments.
    6. Constructs the `BenchmarkMetrics` object to store collected data.
    7. Iterates over all problem names and settings, executing the benchmark for each combination.
    8. Computes final statistics for the collected benchmark results.
    9. Saves the collected benchmark data to an HDF5 file for later analysis and plotting.
    10. Optionally generates plots from the collected benchmark data.

    Args:
        args: An `argparse.Namespace` object containing the parsed benchmark arguments.
    """

    # First print the benchmark configuration to the console for reference
    msg.notif(f"Running benchmark in mode: {args.mode}")

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

    # Determine if CUDA graphs should be used for execution
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    use_cuda_graph = can_use_cuda_graph and args.cuda_graph
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")
    msg.info(f"using_cuda_graph: {use_cuda_graph}")

    # Determine the metrics to collect based on the benchmark mode
    if args.mode == "total":
        collect_step_metrics = False
        collect_solver_metrics = False
        collect_physics_metrics = False
    elif args.mode == "perstep":
        collect_step_metrics = True
        collect_solver_metrics = False
        collect_physics_metrics = False
    elif args.mode == "solver":
        collect_step_metrics = True
        collect_solver_metrics = True
        collect_physics_metrics = False
    elif args.mode == "accuracy":
        collect_step_metrics = True
        collect_solver_metrics = True
        collect_physics_metrics = True
    else:
        raise ValueError(f"Unsupported benchmark mode '{args.mode}'. Supported modes: {SUPPORTED_BENCHMARK_RUN_MODES}")
    msg.info(f"collect_step_metrics: {collect_step_metrics}")
    msg.info(f"collect_solver_metrics: {collect_solver_metrics}")
    msg.info(f"collect_physics_metrics: {collect_physics_metrics}")

    # Determine the problem set from
    # the single and list arguments
    if len(args.problem_set) == 0:
        problem_names = [args.problem]
    else:
        problem_names = args.problem_set
    msg.notif(f"problem_names: {problem_names}")

    # Generate a set of solver configurations to benchmark over
    configs_set = make_benchmark_configs()
    config_names = list(configs_set.keys())
    msg.notif(f"config_names: {config_names}")

    # Generate the problem set based on the
    # provided problem names and arguments
    problem_set = make_benchmark_problems(
        names=problem_names,
        num_worlds=args.num_worlds,
        gravity=args.gravity,
        ground=args.ground,
    )

    # Construct and initialize the metrics
    # object to store benchmark data
    metrics = BenchmarkMetrics(
        problem_names=problem_names,
        config_names=config_names,
        num_steps=args.num_steps,
        step_metrics=collect_step_metrics,
        solver_metrics=collect_solver_metrics,
        physics_metrics=collect_physics_metrics,
    )

    # Iterator over all problem names and settings and run benchmarks for each
    for problem_name, problem_config in problem_set.items():
        for config_name, configs in configs_set.items():
            msg.notif("Running benchmark for problem '%s' with simulation configs '%s'", problem_name, config_name)

            # Retrieve problem and config indices
            problem_idx = metrics._problem_names.index(problem_name)
            config_idx = metrics._config_names.index(config_name)

            # Unpack problem configurations
            builder, control, camera = problem_config

            # Construct simulator configurations based on the solver
            # configurations for the current benchmark configuration
            sim_configs = SimulatorSettings(dt=args.dt, solver=configs)

            # Execute the benchmark for the current problem and settings
            run_single_benchmark(
                problem_idx=problem_idx,
                config_idx=config_idx,
                metrics=metrics,
                args=args,
                builder=builder,
                configs=sim_configs,
                control=control,
                camera=camera,
                device=device,
                use_cuda_graph=use_cuda_graph,
                print_device_info=True,
            )

    # Compute final statistics for the benchmark results
    metrics.compute_stats()

    # Define and create the output directory for the benchmark results
    DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data"))
    RUN_OUTPUT_NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_OUTPUT_PATH = f"{DATA_DIR_PATH}/{RUN_OUTPUT_NAME}"
    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)

    # Export the collected benchmark data to an HDF5 file for later analysis and plotting
    msg.info("Saving benchmark data to HDF5...")
    RUN_HDF5_OUTPUT_PATH = f"{RUN_OUTPUT_PATH}/metrics.hdf5"
    metrics.save_to_hdf5(path=RUN_HDF5_OUTPUT_PATH)
    msg.info("Done.")

    # TODO: Add option for direct plotting after benchmark execution
    # # Plot logged data after the viewer is closed
    # if args.logging:
    # RUN_PLOT_OUTPUT_PATH = f"{DATA_DIR_PATH}/metrics.hdf5"
    # os.makedirs(RUN_PLOT_OUTPUT_PATH, exist_ok=True)
    #     OUTPUT_PLOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.problems)
    #     os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    #     example.plot(path=OUTPUT_PLOT_PATH, show=args.show_plots)


def benchmark_plot(args: argparse.Namespace):
    # TODO
    pass


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

    # TODO: Add switching between run and plot modes
    # Execute the benchmark run with the provided arguments
    benchmark_run(args)

    # TODO: RETRIEVE AND SAVE GIT COMMIT HASH
    # TODO: LOAD AT SPECIFIED PATH
    # TODO: LOAD LAST CREATED HDF5 IN OUTPUT DIRECTORY
