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

import h5py
import numpy as np

from .....core.types import override
from ...solver_kamino import SolverKamino

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
    def __init__(self, data: np.ndarray, name: str | None = None):
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError("StatsFloat requires a floating-point data array.")
        self.name: str | None = name

        # Declare statistics arrays
        self.median: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.mean: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.std: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.min: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.max: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)

        # Compute float stats of each problem (i.e. along axis=2)
        self.median[:, :] = np.median(data, axis=2)
        self.mean[:, :] = np.mean(data, axis=2)
        self.std[:, :] = np.std(data, axis=2)
        self.min[:, :] = np.min(data, axis=2)
        self.max[:, :] = np.max(data, axis=2)

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the StatsFloat."""
        return (
            f"StatsFloat[{self.name or '-'}](\n"
            f"median:\n{self.median},\n"
            f"mean:\n{self.mean},\n"
            f"std:\n{self.std},\n"
            f"min:\n{self.min},\n"
            f"max:\n{self.max}\n"
        )


class StatsInteger:
    def __init__(self, data: np.ndarray, num_bins: int = 20, name: str | None = None):
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("StatsInteger requires an integer data array.")
        self.name: str | None = name

        # Declare statistics arrays
        self.median: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.mean: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.std: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.min: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.max: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)

        # Compute integer stats of each problem (i.e. along axis=2)
        self.median[:, :] = np.median(data.astype(np.float32), axis=2)
        self.mean[:, :] = np.mean(data.astype(np.float32), axis=2)
        self.std[:, :] = np.std(data.astype(np.float32), axis=2)
        self.min[:, :] = np.min(data.astype(np.float32), axis=2)
        self.max[:, :] = np.max(data.astype(np.float32), axis=2)

        # Generate values bins for each problem (i.e. along axis=2) for histogram plotting
        # self.hist, self.binedges = np.histogram(data, bins=np.arange(self.min.min(), self.max.max(), num_bins), axis=2)

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the StatsInteger."""
        return (
            f"StatsInteger[{self.name or '-'}](\n"
            f"median:\n{self.median},\n"
            f"mean:\n{self.mean},\n"
            f"std:\n{self.std},\n"
            f"min:\n{self.min},\n"
            f"max:\n{self.max}\n"
        )


class StatsBinary:
    def __init__(self, data: np.ndarray, name: str | None = None):
        if not np.issubdtype(data.dtype, np.integer) or not np.array_equal(data, data.astype(bool)):
            raise ValueError("StatsBinary requires a binary (boolean) data array.")
        self.name: str | None = name

        # Declare Binary statistics arrays
        self.count_zeros: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.count_ones: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)

        # Compute binary stats of each problem (i.e. along axis=2)
        self.count_zeros[:, :] = np.sum(data == 0, axis=2)
        self.count_ones[:, :] = np.sum(data == 1, axis=2)

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the StatsBinary."""
        return f"StatsBinary[{self.name or '-'}](\ncount_zeros:\n{self.count_zeros},\ncount_ones:\n{self.count_ones}\n"


###
# Types - Metrics
###


class SolverMetrics:
    def __init__(self, num_problems: int, num_configs: int, num_steps: int):
        # Solver-specific metrics
        self.padmm_converged: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.int32)
        self.padmm_iters: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.int32)
        self.padmm_r_p: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.padmm_r_d: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.padmm_r_c: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)

        # Linear solver metrics (placeholders for now)
        self.linear_solver_iters: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.linear_solver_r_error: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)

        # Stats (computed after data collection)
        self.padmm_success_stats: StatsBinary | None = None
        self.padmm_iters_stats: StatsInteger | None = None
        self.padmm_r_p_stats: StatsFloat | None = None
        self.padmm_r_d_stats: StatsFloat | None = None
        self.padmm_r_c_stats: StatsFloat | None = None
        self.linear_solver_iters_stats: StatsInteger | None = None
        self.linear_solver_r_error_stats: StatsFloat | None = None

    def compute_stats(self):
        self.padmm_success_stats = StatsBinary(self.padmm_converged, name="padmm_converged")
        self.padmm_iters_stats = StatsInteger(self.padmm_iters, name="padmm_iters")
        self.padmm_r_p_stats = StatsFloat(self.padmm_r_p, name="padmm_r_p")
        self.padmm_r_d_stats = StatsFloat(self.padmm_r_d, name="padmm_r_d")
        self.padmm_r_c_stats = StatsFloat(self.padmm_r_c, name="padmm_r_c")
        # TODO: self.linear_solver_iters_stats = StatsInteger(self.linear_solver_iters, name="linear_solver_iters")
        # TODO: self.linear_solver_r_error_stats = StatsFloat(self.linear_solver_r_error, name="linear_solver_r_error")


class BenchmarkMetrics:
    def __init__(
        self,
        problem_names: list[str],
        config_names: list[str],
        num_steps: int,
        step_metrics: bool = False,
        solver_metrics: bool = False,
        physics_metrics: bool = False,
    ):
        # Cache run names and counts
        self._problem_names: list[str] = problem_names
        self._config_names: list[str] = config_names
        self._num_steps: int = num_steps

        # One-time metrics
        self.memory_used: np.ndarray = np.zeros((self.num_problems, self.num_configs), dtype=np.float32)
        self.total_time: np.ndarray = np.zeros((self.num_problems, self.num_configs), dtype=np.float32)
        self.total_fps: np.ndarray = np.zeros((self.num_problems, self.num_configs), dtype=np.float32)

        # Per-step metrics
        self.step_time: np.ndarray | None = None
        self.step_time_stats: StatsFloat | None = None
        if step_metrics:
            self.step_time = np.zeros((self.num_problems, self.num_configs, num_steps), dtype=np.float32)

        # Optional solver-specific metrics
        self.solver_metrics: SolverMetrics | None = None
        if solver_metrics:
            self.solver_metrics = SolverMetrics(self.num_problems, self.num_configs, num_steps)

        # Optional physics-specific metrics (placeholders for now)
        self.physics_metrics = None  # TODO
        if physics_metrics:
            self.physics_metrics = None  # TODO

    @property
    def num_problems(self) -> int:
        return len(self._problem_names)

    @property
    def num_configs(self) -> int:
        return len(self._config_names)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def record_step(
        self,
        problem_idx: int,
        config_idx: int,
        step_idx: int,
        step_time: float,
        solver: SolverKamino | None = None,
    ):
        self.step_time[problem_idx, config_idx, step_idx] = step_time
        if self.solver_metrics is not None and solver is not None:
            # Extract PADMM solver status info - this is multiworld
            solver_status = solver._solver_fd.data.status.numpy()[0]  # TODO
            self.solver_metrics.padmm_converged[problem_idx, config_idx, step_idx] = int(solver_status[0])
            self.solver_metrics.padmm_iters[problem_idx, config_idx, step_idx] = int(solver_status[1])
            self.solver_metrics.padmm_r_p[problem_idx, config_idx, step_idx] = float(solver_status[2])
            self.solver_metrics.padmm_r_d[problem_idx, config_idx, step_idx] = float(solver_status[3])
            self.solver_metrics.padmm_r_c[problem_idx, config_idx, step_idx] = float(solver_status[4])

    def record_final(
        self,
        problem_idx: int,
        config_idx: int,
        total_steps: int,
        total_time: float,
        memory_used: float,
    ):
        self.memory_used[problem_idx, config_idx] = memory_used
        self.total_time[problem_idx, config_idx] = total_time
        self.total_fps[problem_idx, config_idx] = float(total_steps) / total_time if total_time > 0.0 else 0.0

    def compute_stats(self):
        if self.step_time is not None:
            self.step_time_stats = StatsFloat(self.step_time, name="step_time")
        if self.solver_metrics is not None:
            self.solver_metrics.compute_stats()
        if self.physics_metrics is not None:
            self.physics_metrics.compute_stats()  # TODO

    def save_to_hdf5(self, path: str):
        with h5py.File(path, "w") as datafile:
            # Store raw data arrays in the HDF5 file
            datafile["Raw/perstep/step_time"] = self.step_time
            datafile["Raw/perstep/padmm/converged"] = self.solver_metrics.padmm_converged
            datafile["Raw/perstep/padmm/iterations"] = self.solver_metrics.padmm_iters
            datafile["Raw/perstep/padmm/r_p"] = self.solver_metrics.padmm_r_p
            datafile["Raw/perstep/padmm/r_d"] = self.solver_metrics.padmm_r_d
            datafile["Raw/perstep/padmm/r_c"] = self.solver_metrics.padmm_r_c

            # Store per-problem and per-config metrics in the HDF5 file with appropriate naming
            for p in range(self.num_problems):
                for c in range(self.num_configs):
                    problem_name = self._problem_names[p]
                    config_name = self._config_names[c]
                    run_name = f"{problem_name}/{config_name}"
                    datafile[f"Runs/{run_name}/total/memory_used"] = self.memory_used[p, c]
                    datafile[f"Runs/{run_name}/total/total_time"] = self.total_time[p, c]
                    datafile[f"Runs/{run_name}/perstep/step_time"] = self.step_time[p, c, :]
                    datafile[f"Runs/{run_name}/perstep/padmm/converged"] = self.solver_metrics.padmm_converged[p, c, :]
                    datafile[f"Runs/{run_name}/perstep/padmm/iterations"] = self.solver_metrics.padmm_iters[p, c, :]
                    datafile[f"Runs/{run_name}/perstep/padmm/r_p"] = self.solver_metrics.padmm_r_p[p, c, :]
                    datafile[f"Runs/{run_name}/perstep/padmm/r_d"] = self.solver_metrics.padmm_r_d[p, c, :]
                    datafile[f"Runs/{run_name}/perstep/padmm/r_c"] = self.solver_metrics.padmm_r_c[p, c, :]
