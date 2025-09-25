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

"""A script to generate benchmark data and performance profiles from an existing HDF5 dataset."""

import os

import admm_linalg_benchmark as bm
import h5py
import numpy as np

import newton._src.solvers.kamino.utils.linalg as linalg
import newton._src.solvers.kamino.utils.logger as msg

###
# Constants
###

# Problem type to load; set to None to load all types
# PROBLEM_TYPE = None
# PROBLEM_TYPE = "Primitive"
# PROBLEM_TYPE = "Robotics"
PROBLEM_TYPE = "Animatronics"

# Problem name to load; set to None to load all problems
# PROBLEM_NAME = None
# PROBLEM_NAME = "box_on_plane"
# PROBLEM_NAME = "boxes_hinged"
# PROBLEM_NAME = "boxes_nunchaku"
# PROBLEM_NAME = "fourbar_free"
PROBLEM_NAME = "walker"
# PROBLEM_NAME = "a1000"

# Sample category to load; set to None to load all categories
# PROBLEM_CATEGORY = None
# PROBLEM_CATEGORY = "IndependentJoints"
# PROBLEM_CATEGORY = "RedundantJoints"
# PROBLEM_CATEGORY = "SingleContact"
# PROBLEM_CATEGORY = "SparseContacts"
# PROBLEM_CATEGORY = "DenseContacts"
PROBLEM_CATEGORY = "DenseConstraints"

# Sample index to load; set to None to load all samples
# PROBLEM_SAMPLE = None
PROBLEM_SAMPLE = 0

# Maximum number of samples to load; set to None to load all samples
# MAX_PROBLEM_SAMPLES = None
MAX_PROBLEM_SAMPLES = 400

# List of keys to exclude when searching for problems
EXCLUDE = ["Unconstrained"]

# Retrieve the path to the data directory
DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# Set the path to the generated HDF5 dataset file
HDF5_DATASET_PATH = f"{DATA_DIR_PATH}/hdf5/simdata.hdf5"

# Set the output path for this run
BENCHMARK_OUTPUT_PATH = f"{DATA_DIR_PATH}/benchmarks/{PROBLEM_TYPE}_{PROBLEM_NAME}_{PROBLEM_CATEGORY}"

# Set path for generated plots
SAMPLE_OUTPUT_PATH = f"{BENCHMARK_OUTPUT_PATH}/sample"

# Set path for generated metrics
DATASET_OUTPUT_PATH = f"{BENCHMARK_OUTPUT_PATH}/dataset"

# Set path for generated plots
PERFPROF_OUTPUT_PATH = f"{BENCHMARK_OUTPUT_PATH}/dataset/perfprof"

###
# Main function
###

if __name__ == "__main__":
    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation
    msg.set_log_level(msg.LogLevel.INFO)

    # Construct and configure the data containers
    msg.info("Loading HDF5 data containers...")
    datafile = h5py.File(HDF5_DATASET_PATH, "r")

    # Select the numpy data type for computations
    # np_dtype = np.float64
    np_dtype = np.float32

    # CONFIGURATIONS
    sample: bool = True
    dataset: bool = False
    profiles: bool = False

    # Revise the root output path to replace 'None' with 'all'
    BENCHMARK_OUTPUT_PATH = BENCHMARK_OUTPUT_PATH.replace("None", "all")
    SAMPLE_OUTPUT_PATH = SAMPLE_OUTPUT_PATH.replace("None", "all")
    DATASET_OUTPUT_PATH = DATASET_OUTPUT_PATH.replace("None", "all")
    PERFPROF_OUTPUT_PATH = PERFPROF_OUTPUT_PATH.replace("None", "all")
    os.makedirs(BENCHMARK_OUTPUT_PATH, exist_ok=True)
    os.makedirs(SAMPLE_OUTPUT_PATH, exist_ok=True)
    os.makedirs(DATASET_OUTPUT_PATH, exist_ok=True)
    os.makedirs(PERFPROF_OUTPUT_PATH, exist_ok=True)

    ###
    # Solver set-up
    ###

    # Create and configure the ADMM solver
    admm_0 = linalg.ADMMSolver(
        dtype=np_dtype,
        primal_tolerance=1e-6,
        dual_tolerance=1e-6,
        compl_tolerance=1e-6,
        iter_tolerance=1e-6,
        diverge_tolerance=1e-1,
        eta=1e-3,
        rho=1.0,
        omega=1.0,
        maxiter=1000,
    )

    # Generate a list of solver variants to benchmark
    solvers = bm.make_solvers(admm_0)

    ###
    # Run on sample problem
    ###

    if sample:
        # Retrieve target data frames
        fpath = bm.build_frame_path(PROBLEM_TYPE, PROBLEM_NAME, PROBLEM_CATEGORY, PROBLEM_SAMPLE)
        msg.info(f"Retrieving data frame at path '{fpath}'...")
        problem_paths = bm.find_problem_paths(datafile=datafile, scope=fpath, exclude=EXCLUDE)
        msg.info(f"Found {len(problem_paths)} paths containing DualProblem data.")
        dataframe = datafile[problem_paths[0]]

        # Load the problem data into a container
        msg.info(f"Loading problem data from '{dataframe.name}'...")
        problem = bm.make_benchmark_problem(
            name=fpath,
            problem=bm.load_problem_data(dataframe=dataframe, dtype=np_dtype),
            ensure_symmetric=False,
            save_matrix_info=True,
            save_symmetry_info=True,
            path=SAMPLE_OUTPUT_PATH + "/systems",
        )

        # Print problem info
        info_str = str(problem.crbd.info)
        msg.info("Problem info:\n%s", info_str)
        print(info_str, file=open(os.path.join(SAMPLE_OUTPUT_PATH, "info.txt"), "w"))

        # Initialize a list to store benchmark metrics
        metrics: list[bm.BenchmarkMetrics] = []

        # Iterate over all solver variants
        msg.info(f"Solving problem '{problem.name}'...")
        for admm, methods in solvers:
            sname = bm.get_solver_typename(admm.schur_solver)
            spath = SAMPLE_OUTPUT_PATH + f"/solvers/{sname}"
            metrics.extend(bm.solve_benchmark_problem(problem, admm, methods, spath))

        # Print all collected metrics
        solvers_str = ""
        for m in metrics:
            mstr = str(m)
            msg.info(f"\n{mstr}\n")
            solvers_str += f"{mstr}\n\n"
        print(solvers_str, file=open(os.path.join(SAMPLE_OUTPUT_PATH, "summary.txt"), "w"))

    ###
    # Run on benchmark dataset
    ###

    if dataset:
        # Create output directories
        os.makedirs(DATASET_OUTPUT_PATH, exist_ok=True)

        # Find and print all DualProblem paths
        search_scope = bm.build_frame_path(PROBLEM_TYPE, PROBLEM_NAME, PROBLEM_CATEGORY)
        msg.info(f"Searching for DualProblem paths in scope '{search_scope}'...")
        problem_paths = bm.find_problem_paths(datafile=datafile, scope=search_scope, exclude=EXCLUDE)
        msg.info(f"Found {len(problem_paths)} paths containing DualProblem data.")

        # Limit the number of samples if requested
        if MAX_PROBLEM_SAMPLES is not None and len(problem_paths) > MAX_PROBLEM_SAMPLES:
            problem_paths = problem_paths[:MAX_PROBLEM_SAMPLES]
            msg.info(f"Limiting to the first {MAX_PROBLEM_SAMPLES} samples.")

        # Iterate over all found DualProblem paths
        metrics: list[bm.BenchmarkMetrics] = []
        msg.info("Iterating over all found DualProblem paths...")
        for path in problem_paths:
            problem = bm.make_benchmark_problem(
                name=path,
                problem=bm.load_problem_data(dataframe=datafile[path], dtype=np_dtype),
                ensure_symmetric=False,
                save_matrix_info=False,
                save_symmetry_info=False,
            )
            for admm, methods in solvers:
                metrics.extend(bm.solve_benchmark_problem(problem, admm, methods, None))

        # Extract performance data from the collected benchmark metrics
        msg.info("Extracting ADMM solver performance data from collected benchmark metrics...")
        perfdata = bm.make_benchmark_performance_data(metrics, problems=problem_paths, output_path=DATASET_OUTPUT_PATH)

        # Extract performance data from the collected benchmark metrics
        msg.info("Extracting linear system solver performance data from collected benchmark metrics...")
        perfdata_linsys = bm.make_benchmark_linsys_performance_data(
            metrics, problems=problem_paths, output_path=DATASET_OUTPUT_PATH
        )

        # Print a coarse summary of solver success rates
        summary_str = bm.make_summary_table(perfdata)
        msg.info("SUMMARY:\n\n%s", summary_str)
        print(summary_str, file=open(os.path.join(DATASET_OUTPUT_PATH, "summary.txt"), "w"))

        # Generate performance profiles and rankings if requested
        if profiles:
            # Create output directories
            os.makedirs(PERFPROF_OUTPUT_PATH + "/admm", exist_ok=True)
            os.makedirs(PERFPROF_OUTPUT_PATH + "/linsys", exist_ok=True)

            # Compute performance profiles for selected metrics
            msg.info("Computing performance profiles...")
            excluded = ["success", "converged", "error"]
            profiles = bm.make_performance_profiles(
                perfdata, exclude=excluded, success_key="success", show=False, path=PERFPROF_OUTPUT_PATH + "/admm"
            )

            # Compute rankings for each metric
            msg.info("Computing performance profile rankings for each metric...")
            rankings = bm.make_perfprof_rankings(profiles)

            # Render rankings table
            rankings_str = bm.make_rankings_table(perfdata["solvers"], rankings)
            msg.info("RANKINGS:\n%s", rankings_str)
            print(rankings_str, file=open(os.path.join(DATASET_OUTPUT_PATH, "rankings.txt"), "w"))

            # Compute performance profiles for selected metrics
            msg.info("Computing linear-system solver performance profiles...")
            profiles_linsys = bm.make_performance_profiles(
                perfdata_linsys, show=False, path=PERFPROF_OUTPUT_PATH + "/linsys"
            )

            # Compute rankings for each metric
            msg.info("Computing linear-system solver performance profile rankings for each metric...")
            rankings_linsys = bm.make_perfprof_rankings(profiles_linsys)

            # Render rankings table
            rankings_linsys_str = bm.make_rankings_table(perfdata_linsys["solvers"], rankings_linsys)
            msg.info("RANKINGS:\n%s", rankings_linsys_str)
            print(rankings_linsys_str, file=open(os.path.join(DATASET_OUTPUT_PATH, "rankings_linsys.txt"), "w"))

    # Close the HDF5 data file
    datafile.close()
    msg.info("Done.")
