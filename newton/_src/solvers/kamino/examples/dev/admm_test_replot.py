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

"""Utility script to re-generate performance profiles and rankings from an existing benchmark dataset."""

import os

import admm_linalg_benchmark as bm
import numpy as np

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

# Sample category to load; set to None to load all categories
# PROBLEM_CATEGORY = None
# PROBLEM_CATEGORY = "IndependentJoints"
# PROBLEM_CATEGORY = "RedundantJoints"
PROBLEM_CATEGORY = "SingleContact"
# PROBLEM_CATEGORY = "SparseContacts"
# PROBLEM_CATEGORY = "DenseContacts"
# PROBLEM_CATEGORY = "DenseConstraints"


# Retrieve the path to the data directory
DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# Set the output path for this run
PROFDATA_INPUT_PATH = f"{DATA_DIR_PATH}/benchmarks/{PROBLEM_TYPE}_{PROBLEM_NAME}_{PROBLEM_CATEGORY}/dataset"

# Set the output path for this run
PROFDATA_OUTPUT_PATH = f"{DATA_DIR_PATH}/replot/{PROBLEM_TYPE}_{PROBLEM_NAME}_{PROBLEM_CATEGORY}"

# Set path for generated plots
PERFPROF_OUTPUT_PATH = f"{PROFDATA_OUTPUT_PATH}/perfprof"


###
# Main function
###

if __name__ == "__main__":
    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=12, threshold=10000, suppress=True)  # Suppress scientific notation
    msg.set_log_level(msg.LogLevel.INFO)

    # Revise the root output path to replace 'None' with 'all'
    PROFDATA_INPUT_PATH = PROFDATA_INPUT_PATH.replace("None", "all")
    PROFDATA_OUTPUT_PATH = PROFDATA_OUTPUT_PATH.replace("None", "all")
    PERFPROF_OUTPUT_PATH = PERFPROF_OUTPUT_PATH.replace("None", "all")
    os.makedirs(PERFPROF_OUTPUT_PATH, exist_ok=True)

    ###
    # Load the benchmark output datasets
    ###

    perfdata_all = np.load(os.path.join(PROFDATA_INPUT_PATH, "perfdata.npy"), allow_pickle=True).item()
    perfdata_linsys_all = np.load(os.path.join(PROFDATA_INPUT_PATH, "perfdata_linsys.npy"), allow_pickle=True).item()

    ###
    # Filter in/out specific solvers
    ###

    perfdata = bm.filter_perfdata_admm(
        perfdata_in=perfdata_all,
        exclude_methods=[],
        keep_methods=["KKT", "Schur-Primal", "Schur-Dual-Prec"],
        exclude_linsys=[],
        keep_linsys=[],
        exclude_metrics=["total_time", "iteration_time", "iterations"],
        keep_metrics=[],
    )

    perfdata_linsys = bm.filter_perfdata_linsys(
        perfdata_in=perfdata_linsys_all,
        exclude_solvers=[],
        keep_solvers=[],
        exclude_metrics=[],
        keep_metrics=[],
    )

    ###
    # Generate performance profiles and rankings for ADMM solvers
    ###

    # Create output directory
    os.makedirs(PERFPROF_OUTPUT_PATH + "/admm", exist_ok=True)

    # Compute performance profiles for selected metrics
    msg.info("Computing performance profiles...")
    excluded = ["success", "converged", "error", "primal_error_rel", "dual_error_rel", "kkt_error_rel"]
    profiles = bm.make_performance_profiles(
        perfdata, exclude=excluded, success_key="success", show=False, path=PERFPROF_OUTPUT_PATH + "/admm"
    )

    # Compute rankings for each metric
    msg.info("Computing performance profile rankings for each metric...")
    rankings = bm.make_perfprof_rankings(profiles)

    # Compute overall rankings (average over all metrics)
    msg.info("Computing overall performance profile rankings for linear-system solvers...")
    rankings["total"] = bm.make_total_perfprof_rankings(
        rankings=rankings,
        keep_metrics=["dual_residual_abs", "primal_error_abs", "dual_error_abs", "kkt_error_abs"],
    )

    # Render rankings table
    rankings_tbl_str = bm.make_rankings_table(perfdata["solvers"], rankings)
    msg.info("RANKINGS (TABLE):\n%s", rankings_tbl_str)
    print(rankings_tbl_str, file=open(os.path.join(PROFDATA_OUTPUT_PATH, "rankings_table.txt"), "w"))

    # Render rankings lists
    rankings_lst_str = bm.make_rho1_rankings_list(perfdata["solvers"], rankings)
    msg.info("RANKINGS (LISTS):\n\n%s", rankings_lst_str)
    print(rankings_lst_str, file=open(os.path.join(PROFDATA_OUTPUT_PATH, "rankings_list.txt"), "w"))

    ###
    # Generate performance profiles and rankings for linear-system solvers
    ###

    # Create output directory
    os.makedirs(PERFPROF_OUTPUT_PATH + "/linsys", exist_ok=True)

    # Compute performance profiles for selected metrics
    msg.info("Computing linear-system solver performance profiles...")
    excluded = [
        "compute_error_rel_min",
        "compute_error_rel_max",
        "compute_error_rel_mean",
        "solve_error_rel_min",
        "solve_error_rel_max",
        "solve_error_rel_mean",
    ]
    profiles_linsys = bm.make_performance_profiles(
        perfdata_linsys, exclude=excluded, show=False, path=PERFPROF_OUTPUT_PATH + "/linsys"
    )

    # Compute rankings for each metric
    msg.info("Computing linear-system solver performance profile rankings for each metric...")
    rankings_linsys = bm.make_perfprof_rankings(profiles_linsys)

    # Compute overall rankings (average over all metrics)
    msg.info("Computing overall performance profile rankings for linear-system solvers...")
    rankings_linsys["total"] = bm.make_total_perfprof_rankings(
        rankings=rankings_linsys,
        keep_metrics=["compute_error_abs_min", "compute_error_abs_max", "solve_error_abs_min", "solve_error_abs_max"],
    )

    # Render rankings table
    rankings_linsys_tbl_str = bm.make_rankings_table(perfdata_linsys["solvers"], rankings_linsys)
    msg.info("RANKINGS (TABLE):\n%s", rankings_linsys_tbl_str)
    print(rankings_linsys_tbl_str, file=open(os.path.join(PROFDATA_OUTPUT_PATH, "rankings_linsys.txt"), "w"))

    # Render rankings lists
    rankings_linsys_lst_str = bm.make_rho1_rankings_list(perfdata_linsys["solvers"], rankings_linsys)
    msg.info("RANKINGS (LISTS):\n\n%s", rankings_linsys_lst_str)
    print(rankings_linsys_lst_str, file=open(os.path.join(PROFDATA_OUTPUT_PATH, "rankings_linsys_list.txt"), "w"))

    # Close the HDF5 data file
    msg.info("Done.")
