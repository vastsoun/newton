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

import numpy as np

###
# Module interface
###

__all__ = [
    "BenchmarkMetrics",
    "SolverMetrics",
    "StatsBinary",
    "StatsFloat",
    "StatsInteger",
]


###
# Types - Statistics
###


class StatsFloat:
    def __init__(self, data: np.ndarray):
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError("StatsFloat requires a floating-point data array.")

        # Declare statistics arrays
        self.median: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.mean: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.std: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.min: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.max: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)

        # Compute float stats of each problem (i.e. along axis=1)
        self.median[:] = np.median(data, axis=1)
        self.mean[:] = np.mean(data, axis=1)
        self.std[:] = np.std(data, axis=1)
        self.min[:] = np.min(data, axis=1)
        self.max[:] = np.max(data, axis=1)


class StatsInteger:
    def __init__(self, data: np.ndarray, num_bins: int = 20):
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("StatsInteger requires an integer data array.")

        # Declare statistics arrays
        self.median: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.mean: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.std: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.min: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.max: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)

        # Compute integer stats of each problem (i.e. along axis=1)
        self.median[:] = np.median(data.astype(np.float32), axis=1)
        self.mean[:] = np.mean(data.astype(np.float32), axis=1)
        self.std[:] = np.std(data.astype(np.float32), axis=1)
        self.min[:] = np.min(data.astype(np.float32), axis=1)
        self.max[:] = np.max(data.astype(np.float32), axis=1)

        # Generate values bins for each problem (i.e. along axis=1) for histogram plotting
        self.hist, self.binedges = np.histogram(data, bins=np.arange(self.min.min(), self.max.max(), num_bins), axis=1)


class StatsBinary:
    def __init__(self, data: np.ndarray):
        if not np.issubdtype(data.dtype, np.integer) or not np.array_equal(data, data.astype(bool)):
            raise ValueError("StatsBinary requires a binary (boolean) data array.")

        # Declare Binary statistics arrays
        self.count_zeros: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)
        self.count_ones: np.ndarray = np.zeros((data.shape[0],), dtype=np.float32)

        # Compute binary stats of each problem (i.e. along axis=1)
        self.count_zeros[:] = np.sum(data == 0, axis=1)
        self.count_ones[:] = np.sum(data == 1, axis=1)


###
# Types - Metrics
###


class SolverMetrics:
    def __init__(self, num_runs: int, num_steps: int):
        # Solver-specific metrics
        self.padmm_success: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.int32)
        self.padmm_iters: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.int32)
        self.padmm_r_p: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.float32)
        self.padmm_r_d: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.float32)
        self.padmm_r_c: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.float32)

        # Linear solver metrics (placeholders for now)
        self.linear_solver_iters: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.float32)
        self.linear_residuals: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.float32)

        # Stats (computed after data collection)
        self.padmm_success_stats: StatsBinary | None = None
        self.padmm_iters_stats: StatsInteger | None = None
        self.padmm_r_p_stats: StatsFloat | None = None
        self.padmm_r_d_stats: StatsFloat | None = None
        self.padmm_r_c_stats: StatsFloat | None = None
        self.linear_solver_iters_stats: StatsInteger | None = None
        self.linear_residuals_stats: StatsFloat | None = None

    def compute_stats(self):
        self.padmm_success_stats = StatsBinary(self.padmm_success)
        self.padmm_iters_stats = StatsInteger(self.padmm_iters)
        self.padmm_r_p_stats = StatsFloat(self.padmm_r_p)
        self.padmm_r_d_stats = StatsFloat(self.padmm_r_d)
        self.padmm_r_c_stats = StatsFloat(self.padmm_r_c)
        self.linear_solver_iters_stats = StatsInteger(self.linear_solver_iters)
        self.linear_residuals_stats = StatsFloat(self.linear_residuals)


class BenchmarkMetrics:
    def __init__(self, run_names: list[str], num_steps: int, with_solver_metrics: bool = False):
        # Cache run names and counts
        self.run_names: list[str] = run_names
        self.num_runs = len(self.run_names)

        # Per-step metrics
        self.step_time: np.ndarray = np.zeros((self.num_runs, num_steps), dtype=np.float32)
        self.total_time: np.ndarray = np.zeros((self.num_runs,), dtype=np.float32)

        # One-time metrics
        self.gpu_memory_used: int = 0
        self.gpu_memory_peak: int = 0

        # Stats (computed after data collection)
        self.step_time_stats: StatsFloat | None = None

        # Optional solver-specific metrics
        self.solver_metrics: SolverMetrics | None = None
        if with_solver_metrics:
            self.solver_metrics = SolverMetrics(self.num_runs, num_steps)

    def compute_stats(self):
        self.step_time_stats = StatsFloat(self.step_time)
        if self.solver_metrics:
            self.solver_metrics.compute_stats()
