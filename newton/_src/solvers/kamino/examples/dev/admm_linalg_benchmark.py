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

"""A benchmarking framework for comparing implementations of ADMM and linear system solvers."""

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

import newton._src.solvers.kamino.utils.linalg as linalg
import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.utils.io import hdf5
from newton._src.solvers.kamino.utils.profiles import PerformanceProfile
from newton._src.solvers.kamino.utils.sparse import sparseview

###
# Module interface
###


__all__ = [
    "BenchmarkMetrics",
    "BenchmarkProblem",
    "BenchmarkSolution",
    "ConstrainedDynamicsInfo",
    "ConstrainedDynamicsMetrics",
    "ConstrainedDynamicsProblem",
    "ConstrainedDynamicsSolution",
    "LinearSolverMetrics",
    "LinearSystemProblem",
    "LinearSystemSolution",
    "SolutionInfo",
    "SolutionMethods",
    "SolutionMetrics",
    "build_frame_path",
    "clip_below",
    "find_problem_paths",
    "get_solver_typename",
    "linsys_error_inf",
    "linsys_error_l2",
    "linsys_error_rel_inf",
    "linsys_error_rel_l2",
    "linsys_residual",
    "load_problem_data",
    "make_benchmark_linsys_performance_data",
    "make_benchmark_metrics",
    "make_benchmark_performance_data",
    "make_benchmark_problem",
    "make_benchmark_solution",
    "make_dual_system",
    "make_kkt_system",
    "make_performance_profiles",
    "make_perfprof_rankings",
    "make_rankings_table",
    "make_solvers",
    "make_summary_table",
    "primal_dynamics_error_inf",
    "primal_dynamics_error_inf_rel",
    "primal_dynamics_error_l2",
    "primal_dynamics_error_l2_rel",
    "primal_dynamics_residual",
    "solve_benchmark_problem",
    "symmetry_info",
]


###
# Linear System Containers
###


@dataclass
class LinearSystemProblem:
    A: np.ndarray | None = None
    b: np.ndarray | None = None


@dataclass
class LinearSystemSolution:
    x: np.ndarray | None = None


@dataclass
class LinearSolverMetrics:
    compute_error_abs: float = np.inf
    compute_error_rel: float = np.inf
    solve_error_abs: float = np.inf
    solve_error_rel: float = np.inf
    iterations: int = 0
    converged: bool = False
    solved: bool = False


###
# Dynamics Problem Containers
###


@dataclass
class ConstrainedDynamicsInfo:
    # Problem dimensions
    nbd: int = 0
    ncts: int = 0
    nvars: int = 0
    # Problem properties
    jacobian_rank: int = 0
    mass_ratio: float = 0.0
    constraint_density: float = 0.0
    # Derived properties
    jacobian_rank_ratio: float = 0.0
    props_J: linalg.RectangularMatrixProperties | None = None
    props_M: linalg.SquareSymmetricMatrixProperties | None = None
    props_D: linalg.SquareSymmetricMatrixProperties | None = None

    def __str__(self) -> str:
        return (
            "\nDIMENSIONS:-----------------------------------------------\n"
            f"  nbd: {self.nbd}\n"
            f" ncts: {self.ncts}\n"
            f"nvars: {self.nvars}\n"
            "\nPROPERTIES:-----------------------------------------------\n"
            f"      Jacobian rank: {self.jacobian_rank}\n"
            f"         Mass ratio: {self.mass_ratio:.7g}\n"
            f" Constraint density: {self.constraint_density:.7g}\n"
            f"Jacobian rank ratio: {self.jacobian_rank_ratio:.7g}\n"
            "\nMATRIX PROPERTIES:----------------------------------------\n\n"
            f"Jacobian properties:\n{self.props_J}\n"
            f"Mass matrix properties:\n{self.props_M}\n"
            f"Delassus matrix properties:\n{self.props_D}\n"
        )


@dataclass
class ConstrainedDynamicsProblem:
    # System info
    info: ConstrainedDynamicsInfo | None = None
    # Primal forward dynamics
    M: np.ndarray | None = None
    invM: np.ndarray | None = None
    J: np.ndarray | None = None
    h: np.ndarray | None = None
    u_minus: np.ndarray | None = None
    v_star: np.ndarray | None = None
    # Dual forward dynamics
    D: np.ndarray | None = None
    v_f: np.ndarray | None = None


@dataclass
class ConstrainedDynamicsSolution:
    # Solution variables
    lambdas: np.ndarray | None = None
    u_plus: np.ndarray | None = None
    v_plus: np.ndarray | None = None


@dataclass
class ConstrainedDynamicsMetrics:
    # Solution norms
    norm_lambdas: float = np.inf
    norm_v_plus: float = np.inf
    norm_u_plus: float = np.inf
    norm_ux: float = np.inf
    # Primal system error
    primal_error_abs: float = np.inf
    primal_error_rel: float = np.inf
    # Dual system error
    dual_error_abs: float = np.inf
    dual_error_rel: float = np.inf
    # KKT system errors
    kkt_error_abs: float = np.inf
    kkt_error_rel: float = np.inf


###
# Solver Containers
###


@dataclass
class SolutionMethods:
    kkt: bool = True
    schur_primal: bool = True
    schur_dual: bool = True
    schur_dual_prec: bool = True


@dataclass
class SolutionInfo:
    dtype: np.dtype | None = None
    solver_name: str | None = None
    method_name: str | None = None
    linear_solver: str | None = None


@dataclass
class SolutionMetrics(ConstrainedDynamicsMetrics):
    # Overall solver status
    error: bool = False
    success: bool = False
    converged: bool = False
    iterations: int = 0
    # Computation timings
    total_time: float = np.inf
    iteration_time: float = np.inf
    # Residuals of the last ADMM iteration
    primal_residual_abs: float = np.inf
    dual_residual_abs: float = np.inf
    compl_residual_abs: float = np.inf
    iter_residual_abs: float = np.inf
    # Performance of the last linear system solver
    linsys_min: LinearSolverMetrics = field(default_factory=LinearSolverMetrics)
    linsys_max: LinearSolverMetrics = field(default_factory=LinearSolverMetrics)
    linsys_mean: LinearSolverMetrics = field(default_factory=LinearSolverMetrics)


###
# Benchmark Containers
###


@dataclass
class BenchmarkProblem:
    name: str | None = None
    crbd: ConstrainedDynamicsProblem | None = None
    kkt: LinearSystemProblem | None = None
    dual: LinearSystemProblem | None = None


@dataclass
class BenchmarkSolution:
    info: SolutionInfo | None = None
    crbd: ConstrainedDynamicsSolution | None = None
    kkt: LinearSystemSolution | None = None
    dual: LinearSystemSolution | None = None


@dataclass
class BenchmarkMetrics:
    pname: str | None = None
    info: SolutionInfo | None = None
    data: SolutionMetrics | None = None

    @staticmethod
    def _metrics() -> list[str]:
        """Ordered metric columns to display per solver group."""
        return [
            "error",
            "success",
            "converged",
            "iterations",
            "total_time",
            "iteration_time",
            "dual_residual_abs",
            "primal_error_abs",
            "primal_error_rel",
            "dual_error_abs",
            "dual_error_rel",
            "kkt_error_abs",
            "kkt_error_rel",
            "norm_lambdas",
            "norm_u_plus",
        ]

    @staticmethod
    def _metrics_linsys_compute() -> list[str]:
        """Ordered metric columns to display per solver group."""
        return [
            "compute_error_abs_min",
            "compute_error_abs_max",
            "compute_error_abs_mean",
            "compute_error_rel_min",
            "compute_error_rel_max",
            "compute_error_rel_mean",
        ]

    @staticmethod
    def _metrics_linsys_solve() -> list[str]:
        """Ordered metric columns to display per solver group."""
        return [
            "solve_error_abs_min",
            "solve_error_abs_max",
            "solve_error_abs_mean",
            "solve_error_rel_min",
            "solve_error_rel_max",
            "solve_error_rel_mean",
        ]

    @staticmethod
    def _metrics_linsys_all() -> list[str]:
        """Ordered metric columns to display per solver group."""
        return BenchmarkMetrics._metrics_linsys_compute() + BenchmarkMetrics._metrics_linsys_solve()

    @staticmethod
    def fmt(v: Any, dtyp: np.dtype) -> str:
        if isinstance(v, float | np.floating | np.float32 | np.float64):
            # Format using dtype precision if available; fall back to 6 significant digits
            try:
                if int(abs(v)) >= 100:
                    return f"{v:.{max(1, min(np.finfo(np.dtype(dtyp)).precision + 1, 4))}e}"
                else:
                    return f"{v:.{max(1, min(np.finfo(np.dtype(dtyp)).precision + 1, 12))}g}"
            except Exception:
                return f"{v:.6e}"
        return str(v)

    def _solverid(self, separator: str = " / ") -> str:
        """Concatenate info fields to a final solver name."""
        if self.info is None:
            return "Unknown"
        fields = (self.info.solver_name, self.info.method_name, self.info.linear_solver, f"{self.info.dtype}")
        parts = [p for p in fields if p]
        return separator.join(parts) if parts else "Unknown"

    def _values(self) -> dict[str, str]:
        """Map of metric name -> stringified value for this instance."""
        if self.info is None:
            raise ValueError("BenchmarkMetrics info is not set.")
        if self.data is None:
            raise ValueError("BenchmarkMetrics data is not set.")
        return {
            "error": self.fmt(self.data.error, self.info.dtype),
            "success": self.fmt(self.data.success, self.info.dtype),
            "converged": self.fmt(self.data.converged, self.info.dtype),
            "iterations": self.fmt(self.data.iterations, self.info.dtype),
            "total_time": self.fmt(self.data.total_time, self.info.dtype),
            "iteration_time": self.fmt(self.data.iteration_time, self.info.dtype),
            "dual_residual_abs": self.fmt(self.data.dual_residual_abs, self.info.dtype),
            "primal_error_abs": self.fmt(self.data.primal_error_abs, self.info.dtype),
            "primal_error_rel": self.fmt(self.data.primal_error_rel, self.info.dtype),
            "dual_error_abs": self.fmt(self.data.dual_error_abs, self.info.dtype),
            "dual_error_rel": self.fmt(self.data.dual_error_rel, self.info.dtype),
            "kkt_error_abs": self.fmt(self.data.kkt_error_abs, self.info.dtype),
            "kkt_error_rel": self.fmt(self.data.kkt_error_rel, self.info.dtype),
            "norm_lambdas": self.fmt(self.data.norm_lambdas, self.info.dtype),
            "norm_u_plus": self.fmt(self.data.norm_u_plus, self.info.dtype),
        }

    def _values_linsys_compute(self) -> dict[str, str]:
        """Map of metric name -> stringified value for this instance."""
        if self.info is None:
            raise ValueError("BenchmarkMetrics info is not set.")
        if self.data is None:
            raise ValueError("BenchmarkMetrics data is not set.")
        return {
            "error": self.fmt(self.data.error, self.info.dtype),
            "success": self.fmt(self.data.success, self.info.dtype),
            "converged": self.fmt(self.data.converged, self.info.dtype),
            "compute_error_abs_min": self.fmt(self.data.linsys_min.compute_error_abs, self.info.dtype),
            "compute_error_abs_max": self.fmt(self.data.linsys_max.compute_error_abs, self.info.dtype),
            "compute_error_abs_mean": self.fmt(self.data.linsys_mean.compute_error_abs, self.info.dtype),
            "compute_error_rel_min": self.fmt(self.data.linsys_min.compute_error_rel, self.info.dtype),
            "compute_error_rel_max": self.fmt(self.data.linsys_max.compute_error_rel, self.info.dtype),
            "compute_error_rel_mean": self.fmt(self.data.linsys_mean.compute_error_rel, self.info.dtype),
        }

    def _values_linsys_solve(self) -> dict[str, str]:
        """Map of metric name -> stringified value for this instance."""
        if self.info is None:
            raise ValueError("BenchmarkMetrics info is not set.")
        if self.data is None:
            raise ValueError("BenchmarkMetrics data is not set.")
        return {
            "error": self.fmt(self.data.error, self.info.dtype),
            "success": self.fmt(self.data.success, self.info.dtype),
            "converged": self.fmt(self.data.converged, self.info.dtype),
            "solve_error_abs_min": self.fmt(self.data.linsys_min.solve_error_abs, self.info.dtype),
            "solve_error_abs_max": self.fmt(self.data.linsys_max.solve_error_abs, self.info.dtype),
            "solve_error_abs_mean": self.fmt(self.data.linsys_mean.solve_error_abs, self.info.dtype),
            "solve_error_rel_min": self.fmt(self.data.linsys_min.solve_error_rel, self.info.dtype),
            "solve_error_rel_max": self.fmt(self.data.linsys_max.solve_error_rel, self.info.dtype),
            "solve_error_rel_mean": self.fmt(self.data.linsys_mean.solve_error_rel, self.info.dtype),
        }

    def _values_linsys_all(self) -> dict[str, str]:
        """Map of metric name -> stringified value for this instance."""
        if self.info is None:
            raise ValueError("BenchmarkMetrics info is not set.")
        if self.data is None:
            raise ValueError("BenchmarkMetrics data is not set.")
        return {
            "error": self.fmt(self.data.error, self.info.dtype),
            "success": self.fmt(self.data.success, self.info.dtype),
            "converged": self.fmt(self.data.converged, self.info.dtype),
            "compute_error_abs_min": self.fmt(self.data.linsys_min.compute_error_abs, self.info.dtype),
            "compute_error_abs_max": self.fmt(self.data.linsys_max.compute_error_abs, self.info.dtype),
            "compute_error_abs_mean": self.fmt(self.data.linsys_mean.compute_error_abs, self.info.dtype),
            "compute_error_rel_min": self.fmt(self.data.linsys_min.compute_error_rel, self.info.dtype),
            "compute_error_rel_max": self.fmt(self.data.linsys_max.compute_error_rel, self.info.dtype),
            "compute_error_rel_mean": self.fmt(self.data.linsys_mean.compute_error_rel, self.info.dtype),
            "solve_error_abs_min": self.fmt(self.data.linsys_min.solve_error_abs, self.info.dtype),
            "solve_error_abs_max": self.fmt(self.data.linsys_max.solve_error_abs, self.info.dtype),
            "solve_error_abs_mean": self.fmt(self.data.linsys_mean.solve_error_abs, self.info.dtype),
            "solve_error_rel_min": self.fmt(self.data.linsys_min.solve_error_rel, self.info.dtype),
            "solve_error_rel_max": self.fmt(self.data.linsys_max.solve_error_rel, self.info.dtype),
            "solve_error_rel_mean": self.fmt(self.data.linsys_mean.solve_error_rel, self.info.dtype),
        }

    def to_table(self) -> str:
        """Render a single instance as a one-row table string."""

        # Prepare title, columns, and rows
        title = self._solverid() + " @ " + (self.pname if self.pname else "Unknown")
        columns = self._metrics()
        rows = [[self._values()[c] for c in columns]]

        # Compute column widths
        widths: list[int] = []
        for j, name in enumerate(columns):
            col_vals = [rows[0][j]]
            width = max(len(name), *(len(v) for v in col_vals)) if col_vals else len(name)
            widths.append(width)

        # Build header and separator
        header = " | ".join(name.center(w) for name, w in zip(columns, widths, strict=False))
        rule = "-+-".join("-" * w for w in widths)

        # Compute total width for title centering
        total_width = sum(widths) + 3 * (len(widths) - 1)
        title = title.center(total_width)

        # Build and return data row
        line = " | ".join(val.rjust(w) for val, w in zip(rows[0], widths, strict=False))
        return "\n".join([title, rule, header, rule, line, rule])

    def __str__(self) -> str:
        """Pretty print a single instance as a one-row table."""
        return self.to_table()


###
# Functions
###


def linsys_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return A @ x - b


def linsys_error_inf(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return np.max(np.abs(linsys_residual(A, b, x)))


def linsys_error_l2(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return np.linalg.norm(linsys_residual(A, b, x))


def linsys_error_rel_inf(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    eps = np.finfo(A.dtype).eps
    norm_x = np.linalg.norm(x, ord=np.inf)
    norm_b = np.linalg.norm(b, ord=np.inf)
    norm_A = np.linalg.norm(A, ord=np.inf)
    denom = max(norm_b, norm_A * norm_x) + eps
    r = linsys_residual(A, b, x)
    return np.linalg.norm(r, ord=np.inf) / denom


def linsys_error_rel_l2(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    eps = np.finfo(A.dtype).eps
    norm_x = np.linalg.norm(x)
    norm_b = np.linalg.norm(b)
    norm_A = np.linalg.norm(A)
    denom = max(norm_b, norm_A * norm_x) + eps
    r = linsys_residual(A, b, x)
    return np.linalg.norm(r) / denom


def primal_dynamics_residual(
    problem: ConstrainedDynamicsProblem,
    solution: ConstrainedDynamicsSolution,
) -> np.ndarray:
    return problem.M @ (solution.u_plus - problem.u_minus) - problem.J.T @ solution.lambdas - problem.h


def primal_dynamics_error_inf(
    problem: ConstrainedDynamicsProblem,
    solution: ConstrainedDynamicsSolution,
) -> float:
    return np.max(np.abs(primal_dynamics_residual(problem, solution)))


def primal_dynamics_error_l2(
    problem: ConstrainedDynamicsProblem,
    solution: ConstrainedDynamicsSolution,
) -> float:
    return np.max(np.abs(primal_dynamics_residual(problem, solution)))


def primal_dynamics_error_inf_rel(
    problem: ConstrainedDynamicsProblem,
    solution: ConstrainedDynamicsSolution,
) -> float:
    eps = np.finfo(problem.M.dtype).eps
    norm_M = np.linalg.norm(problem.M, ord=np.inf)
    norm_u_plus = np.linalg.norm(solution.u_plus, ord=np.inf)
    norm_u_minus = np.linalg.norm(problem.u_minus, ord=np.inf)
    norm_h = np.linalg.norm(problem.h, ord=np.inf)
    norm_J = np.linalg.norm(problem.J, ord=np.inf)
    norm_lambdas = np.linalg.norm(solution.lambdas, ord=np.inf)
    denom = max(norm_h, norm_J * norm_lambdas, norm_M * max(norm_u_plus, norm_u_minus), eps)
    r_norm = primal_dynamics_error_inf(problem, solution)
    return r_norm / denom


def primal_dynamics_error_l2_rel(
    problem: ConstrainedDynamicsProblem,
    solution: ConstrainedDynamicsSolution,
) -> float:
    eps = np.finfo(problem.M.dtype).eps
    norm_M = np.linalg.norm(problem.M)
    norm_u_plus = np.linalg.norm(solution.u_plus)
    norm_u_minus = np.linalg.norm(problem.u_minus)
    norm_h = np.linalg.norm(problem.h)
    norm_J = np.linalg.norm(problem.J)
    norm_lambdas = np.linalg.norm(solution.lambdas)
    denom = max(norm_h, norm_J * norm_lambdas, norm_M * max(norm_u_plus, norm_u_minus), eps)
    r = primal_dynamics_residual(problem, solution)
    r_norm = np.linalg.norm(r)
    return r_norm / denom


def get_solver_typename(solver: Any) -> str:
    name = type(solver).__name__
    if name.endswith("Solver"):
        name = name[: -len("Solver")]
    return name


def build_frame_path(
    problem_type: str | None = None,
    problem_name: str | None = None,
    problem_category: str | None = None,
    problem_sample: int | None = None,
) -> str:
    """
    Build hierarchical path: TYPE[/NAME[/CATEGORY[/SAMPLE]]].
    - If problem_type is None, return "".
    - If problem_name is None, skip category and problem_sample.
    - If problem_category is None, skip problem_sample.
    """
    if problem_type is None:
        return "/"
    parts = [problem_type]
    if problem_name is not None:
        parts.append(problem_name)
        if problem_category is not None:
            parts.append(problem_category)
            if problem_sample is not None:
                parts.append(str(problem_sample))
    return "/".join(parts)


def find_problem_paths(datafile: h5py.File, scope: str | None, exclude: list[str] | None) -> list[str]:
    """Recursively find all paths ending with '/DualProblem' in an HDF5 file."""

    # Initialize an empty list to store found paths
    paths: list[str] = []

    # Define a visitor function for HDF5 traversal
    def _visitor(name: str, obj):
        split = name.rsplit("/")
        if exclude is not None:
            for ex in exclude:
                if ex in split:
                    return
        # Append the path if it contains 'DualProblem' as the last component
        if split[-1] == "DualProblem":
            # Accept both groups and datasets named 'DualProblem'
            if isinstance(obj, h5py.Group | h5py.Dataset):
                paths.append(obj.parent.name)  # absolute path

    # Traverse the HDF5 file structure, optionally within a specific scope
    if scope is not None:
        if scope in datafile:
            datafile[scope].visititems(_visitor)
        else:
            msg.error("Scope '%s' not found in the HDF5 file.", scope)
    else:
        datafile.visititems(_visitor)

    # Post-process paths based on scope
    paths.sort()

    # Return the list of found and filtered paths
    return paths


def load_problem_data(dataframe: h5py.Group, dtype: type = np.float64) -> ConstrainedDynamicsProblem:
    """Load a DualProblem dataset from an HDF5 group into a ConstrainedDynamicsProblem container."""

    # Load the DualProblem data into the HDF5 data-frame container
    dp_data = hdf5.DualProblemData()
    dp_data.load(dataset=dataframe["DualProblem"], dtype=dtype)

    # Load system info data into the HDF5 data-frame container
    si_data = hdf5.SystemInfoData()
    si_data.load(dataset=dataframe["info"], dtype=dtype)

    # Construct the problem info container
    info = ConstrainedDynamicsInfo(
        nbd=dp_data.info.nbd,
        ncts=dp_data.info.nd,
        nvars=dp_data.info.nbd + dp_data.info.nd,
        jacobian_rank=si_data.jacobian_rank,
        mass_ratio=si_data.mass_ratio,
        constraint_density=si_data.constraint_density,
        jacobian_rank_ratio=si_data.jacobian_rank / dp_data.info.nd if dp_data.info.nd > 0 else np.inf,
        props_J=linalg.RectangularMatrixProperties(dp_data.J),
        props_M=linalg.SquareSymmetricMatrixProperties(dp_data.M),
        props_D=linalg.SquareSymmetricMatrixProperties(dp_data.D),
    )

    # Construct the problem quantities from the HDF5 data
    M = dp_data.M
    invM = dp_data.invM
    J = dp_data.J
    h = dtype(dp_data.dt) * dp_data.h
    u_minus = dp_data.u_minus
    v_star = dp_data.v_i + dp_data.v_b
    D = dp_data.D
    v_f = dp_data.v_f - v_star

    # Pack quantities into the problem container
    return ConstrainedDynamicsProblem(
        info=info,
        M=M,
        invM=invM,
        J=J,
        h=h,
        u_minus=u_minus,
        v_star=v_star,
        D=D,
        v_f=v_f,
    )


def make_kkt_system(
    problem: ConstrainedDynamicsProblem,
    ensure_symmetric: bool = False,
    save_matrix_info: bool = False,
    save_symmetry_info: bool = False,
    path: str | None = None,
) -> LinearSystemProblem:
    """Assemble the KKT system from a ConstrainedDynamicsProblem."""
    if problem.M is None or problem.J is None or problem.h is None or problem.u_minus is None or problem.v_star is None:
        raise ValueError("Incomplete ConstrainedDynamicsProblem data.")

    nbd = problem.M.shape[0]
    ncts = problem.J.shape[0]
    kdim = nbd + ncts
    msg.debug("nbd: %d", nbd)
    msg.debug("ncts: %d", ncts)
    msg.debug("kdim: %d", kdim)

    # Assemble the KKT matrix
    K = np.zeros((kdim, kdim), dtype=problem.M.dtype)
    K[:nbd, :nbd] = problem.M
    K[:nbd, nbd:] = problem.J.T
    K[nbd:, :nbd] = problem.J
    msg.debug("K: norm=%s, shape=%s, dtype=%s", np.linalg.norm(K), K.shape, K.dtype)

    # Assemble the KKT rhs
    k = np.zeros((kdim,), dtype=problem.M.dtype)
    k[:nbd] = problem.M @ problem.u_minus + problem.h
    k[nbd:] = -problem.v_star
    msg.debug("k: norm=%s, shape=%s, dtype=%s", np.linalg.norm(k), k.shape, k.dtype)

    # Optionally ensure symmetry of the KKT matrix
    if ensure_symmetric:
        K = K.dtype.type(0.5) * (K + K.T)

    # Optionally compute matrix properties
    if save_matrix_info:
        properties_K = linalg.SquareSymmetricMatrixProperties(K)
        properties_K_str = str(properties_K)
        msg.debug("K properties: %s", properties_K_str)
        print(properties_K_str, file=open(os.path.join(path, "K_properties.txt"), "w"))

    # Optionally render the KKT matrix and symmetry error info as images
    if save_symmetry_info:
        symmetry_info(K, name="K", title="KKT", eps=np.finfo(problem.M.dtype).eps, path=path)

    # Pack quantities into the linear system container
    return LinearSystemProblem(A=K, b=k)


def make_dual_system(
    problem: ConstrainedDynamicsProblem,
    ensure_symmetric: bool = False,
    save_matrix_info: bool = False,
    save_symmetry_info: bool = False,
    path: str | None = None,
) -> LinearSystemProblem:
    """Assemble the dual system from a ConstrainedDynamicsProblem."""
    if problem.D is None or problem.v_f is None:
        raise ValueError("Incomplete ConstrainedDynamicsProblem data.")

    # Assemble the dual system matrix and rhs
    D = problem.D
    d = -(problem.v_f + problem.v_star)
    msg.debug("D: norm=%s, shape=%s, dtype=%s", np.linalg.norm(D), D.shape, D.dtype)
    msg.debug("d: norm=%s, shape=%s, dtype=%s", np.linalg.norm(d), d.shape, d.dtype)

    # Optionally ensure symmetry of the dual system matrix
    if ensure_symmetric:
        D = D.dtype.type(0.5) * (D + D.T)

    # Optionally compute matrix properties
    if save_matrix_info:
        properties_D = linalg.SquareSymmetricMatrixProperties(D)
        properties_D_str = str(properties_D)
        msg.debug("D properties: %s", properties_D_str)
        print(properties_D_str, file=open(os.path.join(path, "D_properties.txt"), "w"))

    # Optionally render the primal schur complement matrix and symmetry error info as images
    if save_symmetry_info:
        symmetry_info(D, name="D", title="Delassus Matrix", eps=np.finfo(problem.D.dtype).eps, path=path)

    # Pack quantities into the linear system container
    return LinearSystemProblem(A=D, b=d)


def make_benchmark_problem(
    name: str,
    problem: ConstrainedDynamicsProblem,
    ensure_symmetric: bool = False,
    save_matrix_info: bool = False,
    save_symmetry_info: bool = False,
    path: str | None = None,
) -> BenchmarkProblem:
    """Create a BenchmarkProblem from a ConstrainedDynamicsProblem."""
    # If an output path is given, ensure it exists
    if path:
        os.makedirs(path, exist_ok=True)
    # Create the KKT system
    kkt = make_kkt_system(
        problem,
        ensure_symmetric=ensure_symmetric,
        save_matrix_info=save_matrix_info,
        save_symmetry_info=save_symmetry_info,
        path=path,
    )
    # Create the dual system
    dual = make_dual_system(
        problem,
        ensure_symmetric=ensure_symmetric,
        save_matrix_info=save_matrix_info,
        save_symmetry_info=save_symmetry_info,
        path=path,
    )
    # Return the benchmark problem container
    return BenchmarkProblem(name=name, crbd=problem, kkt=kkt, dual=dual)


def make_benchmark_solution(admm: linalg.ADMMSolver, info: SolutionInfo) -> BenchmarkSolution:
    """Create a BenchmarkSolution from an ADMMSolver."""
    return BenchmarkSolution(
        crbd=ConstrainedDynamicsSolution(
            lambdas=admm.lambdas.copy(), u_plus=admm.u_plus.copy(), v_plus=admm.v_plus.copy()
        ),
        kkt=LinearSystemSolution(x=np.concatenate((admm.u_plus, -admm.lambdas))),
        dual=LinearSystemSolution(x=admm.lambdas.copy()),
        info=info,
    )


def make_benchmark_metrics(
    time: float,
    info: linalg.ADMMInfo,
    status: linalg.ADMMStatus,
    problem: BenchmarkProblem,
    solution: BenchmarkSolution,
) -> BenchmarkMetrics:
    # Create a new metrics container
    metrics = BenchmarkMetrics(pname=problem.name, info=solution.info, data=SolutionMetrics())

    # First set wether there was an error
    metrics.data.error = status.result == linalg.ADMMResult.ERROR

    # Set the linear system solver metrics
    # NOTE: These are capture regardless of whether the solve was successful or not
    metrics.data.linsys_min.compute_error_abs = np.min(info.r_linsys_compute_abs)
    metrics.data.linsys_min.compute_error_rel = np.min(info.r_linsys_compute_rel)
    metrics.data.linsys_min.solve_error_abs = np.min(info.r_linsys_solve_abs)
    metrics.data.linsys_min.solve_error_rel = np.min(info.r_linsys_solve_rel)
    metrics.data.linsys_max.compute_error_abs = np.max(info.r_linsys_compute_abs)
    metrics.data.linsys_max.compute_error_rel = np.max(info.r_linsys_compute_rel)
    metrics.data.linsys_max.solve_error_abs = np.max(info.r_linsys_solve_abs)
    metrics.data.linsys_max.solve_error_rel = np.max(info.r_linsys_solve_rel)
    # NOTE: Only compute mean linsys performance if there was no error,
    # because if there was, then the solver had diverged and the mean errors
    # would be meaningless (usually very large values).
    if not metrics.data.error:
        metrics.data.linsys_mean.compute_error_abs = np.mean(info.r_linsys_compute_abs)
        metrics.data.linsys_mean.compute_error_rel = np.mean(info.r_linsys_compute_rel)
        metrics.data.linsys_mean.solve_error_abs = np.mean(info.r_linsys_solve_abs)
        metrics.data.linsys_mean.solve_error_rel = np.mean(info.r_linsys_solve_rel)

    # If there was an error, return early so that all other metrics remain at infinity
    if metrics.data.error:
        return metrics

    # Set the solution metrics
    metrics.data.success = status.result <= linalg.ADMMResult.MAXITER
    metrics.data.converged = status.converged
    metrics.data.iterations = status.iterations
    metrics.data.total_time = time
    metrics.data.iteration_time = time / status.iterations if status.iterations > 0 else np.inf
    metrics.data.primal_residual_abs = status.r_p
    metrics.data.dual_residual_abs = status.r_d
    metrics.data.compl_residual_abs = status.r_c
    metrics.data.iter_residual_abs = status.r_i

    # Compute true constraint-space velocity
    v_plus = problem.dual.A @ solution.crbd.lambdas - problem.dual.b

    # Compute solution norms
    metrics.data.norm_lambdas = np.linalg.norm(solution.crbd.lambdas, ord=np.inf)
    metrics.data.norm_u_plus = np.linalg.norm(solution.crbd.u_plus, ord=np.inf)
    metrics.data.norm_v_plus = np.linalg.norm(v_plus, ord=np.inf)
    metrics.data.norm_ux = np.linalg.norm(solution.kkt.x, ord=np.inf)

    # Compute the CRBD performance metrics
    metrics.data.primal_error_abs = primal_dynamics_error_inf(problem.crbd, solution.crbd)
    metrics.data.primal_error_rel = primal_dynamics_error_inf_rel(problem.crbd, solution.crbd)
    metrics.data.dual_error_abs = linsys_error_inf(problem.dual.A, problem.dual.b, solution.dual.x)
    metrics.data.dual_error_rel = linsys_error_rel_inf(problem.dual.A, problem.dual.b, solution.dual.x)
    metrics.data.kkt_error_abs = linsys_error_inf(problem.kkt.A, problem.kkt.b, solution.kkt.x)
    metrics.data.kkt_error_rel = linsys_error_rel_inf(problem.kkt.A, problem.kkt.b, solution.kkt.x)

    # Return the populated metrics container
    return metrics


def solve_benchmark_problem(
    problem: BenchmarkProblem, admm: linalg.ADMMSolver, methods: SolutionMethods, save_info_path: str | None = None
) -> list[BenchmarkMetrics]:
    """Solve a ConstrainedDynamicsProblem using an ADMM solver."""

    # Initialize solver metrics list
    metrics = []

    # Extract solver info
    solver_name = get_solver_typename(admm)

    # Create directory for saving info if specified
    if save_info_path is not None:
        os.makedirs(save_info_path, exist_ok=True)

    # Solve as KKT system
    if methods.kkt:
        start_kkt = time.perf_counter()
        status_kkt = admm.solve_kkt(
            M=problem.crbd.M,
            J=problem.crbd.J,
            h=problem.crbd.h,
            u_minus=problem.crbd.u_minus,
            v_star=problem.crbd.v_star,
        )
        time_kkt = time.perf_counter() - start_kkt
        msg.debug(f"ADMM.solve_kkt took {time_kkt:.6f} seconds")

        # Generate the benchmark solution and metrics for the KKT solve
        kkt_info = SolutionInfo(
            dtype=admm.dtype,
            solver_name=solver_name,
            method_name="KKT",
            linear_solver=get_solver_typename(admm.kkt_solver),
        )
        metrics.append(
            make_benchmark_metrics(time_kkt, admm.info, status_kkt, problem, make_benchmark_solution(admm, kkt_info))
        )

        # Optionally save convergence plots
        if save_info_path is not None:
            admm.save_info(path=save_info_path, suffix="_kkt")

    # Solve as primal Schur-complement system
    if methods.schur_primal:
        start_schur_prim = time.perf_counter()
        status_schur_prim = admm.solve_schur_primal(
            M=problem.crbd.M,
            J=problem.crbd.J,
            h=problem.crbd.h,
            u_minus=problem.crbd.u_minus,
            v_star=problem.crbd.v_star,
        )
        time_schur_prim = time.perf_counter() - start_schur_prim
        msg.debug(f"ADMM.solve_schur_prim took {time_schur_prim:.6f} seconds")

        # Generate the benchmark solution and metrics for the primal Schur-complement solve
        schur_prim_info = SolutionInfo(
            dtype=admm.dtype,
            solver_name=solver_name,
            method_name="Schur-Primal",
            linear_solver=get_solver_typename(admm.schur_solver),
        )
        metrics.append(
            make_benchmark_metrics(
                time_schur_prim, admm.info, status_schur_prim, problem, make_benchmark_solution(admm, schur_prim_info)
            )
        )

        # Optionally save convergence plots
        if save_info_path is not None:
            admm.save_info(path=save_info_path, suffix="_schur_prim")

    # Solve as dual Schur-complement system w/o preconditioning
    if methods.schur_dual:
        start_schur_dual = time.perf_counter()
        status_schur_dual = admm.solve_schur_dual(
            D=problem.crbd.D,
            v_f=problem.crbd.v_f,
            v_star=problem.crbd.v_star,
            u_minus=problem.crbd.u_minus,
            invM=problem.crbd.invM,
            J=problem.crbd.J,
            h=problem.crbd.h,
            use_preconditioning=False,
        )
        time_schur_dual = time.perf_counter() - start_schur_dual
        msg.debug(f"ADMM.solve_schur_dual took {time_schur_dual:.6f} seconds")

        # Generate the benchmark solution and metrics for the dual Schur-complement solve
        schur_dual_info = SolutionInfo(
            dtype=admm.dtype,
            solver_name=solver_name,
            method_name="Schur-Dual",
            linear_solver=get_solver_typename(admm.schur_solver),
        )
        metrics.append(
            make_benchmark_metrics(
                time_schur_dual, admm.info, status_schur_dual, problem, make_benchmark_solution(admm, schur_dual_info)
            )
        )

        # Optionally save convergence plots
        if save_info_path is not None:
            admm.save_info(path=save_info_path, suffix="_schur_dual")

    # Solve as dual Schur-complement system with preconditioning
    if methods.schur_dual_prec:
        start_schur_dual_prec = time.perf_counter()
        status_schur_dual_prec = admm.solve_schur_dual(
            D=problem.crbd.D,
            v_f=problem.crbd.v_f,
            v_star=problem.crbd.v_star,
            u_minus=problem.crbd.u_minus,
            invM=problem.crbd.invM,
            J=problem.crbd.J,
            h=problem.crbd.h,
            use_preconditioning=True,
        )
        time_schur_dual_prec = time.perf_counter() - start_schur_dual_prec
        msg.debug(f"ADMM.solve_schur_dual (prec) took {time_schur_dual_prec:.6f} seconds")

        # Generate the benchmark solution and metrics for the dual Schur-complement solve with preconditioning
        schur_dual_prec_info = SolutionInfo(
            dtype=admm.dtype,
            solver_name=solver_name,
            method_name="Schur-Dual-Prec",
            linear_solver=get_solver_typename(admm.schur_solver),
        )
        metrics.append(
            make_benchmark_metrics(
                time_schur_dual_prec,
                admm.info,
                status_schur_dual_prec,
                problem,
                make_benchmark_solution(admm, schur_dual_prec_info),
            )
        )

        # Optionally save convergence plots
        if save_info_path is not None:
            admm.save_info(path=save_info_path, suffix="_schur_dual_prec")

    # Return all metrics for this problem
    return metrics


def make_solvers(admm: linalg.ADMMSolver) -> list[tuple[linalg.ADMMSolver, SolutionMethods]]:
    variants: list[tuple[linalg.ADMMSolver, SolutionMethods]] = []

    # admm_np = copy.deepcopy(admm)
    # methods_np = SolutionMethods()
    # admm_np.kkt_solver = linalg.NumPySolver()
    # admm_np.schur_solver = linalg.NumPySolver()
    # variants.append((admm_np, methods_np))

    # admm_sp = copy.deepcopy(admm)
    # methods_sp = SolutionMethods()
    # admm_sp.kkt_solver = linalg.SciPySolver()
    # admm_sp.schur_solver = linalg.SciPySolver()
    # variants.append((admm_sp, methods_sp))

    # admm_llt_np = copy.deepcopy(admm)
    # methods_llt_np = SolutionMethods(kkt=False)
    # admm_llt_np.schur_solver = linalg.LLTNumPySolver()
    # variants.append((admm_llt_np, methods_llt_np))

    # admm_llt_sp = copy.deepcopy(admm)
    # methods_llt_sp = SolutionMethods(kkt=False)
    # admm_llt_sp.schur_solver = linalg.LLTSciPySolver()
    # variants.append((admm_llt_sp, methods_llt_sp))

    # admm_ldlt_sp = copy.deepcopy(admm)
    # methods_ldlt_sp = SolutionMethods()
    # admm_ldlt_sp.kkt_solver = linalg.LDLTSciPySolver()
    # admm_ldlt_sp.schur_solver = linalg.LDLTSciPySolver()
    # variants.append((admm_ldlt_sp, methods_ldlt_sp))

    # admm_lu_sp = copy.deepcopy(admm)
    # methods_lu_sp = SolutionMethods()
    # admm_lu_sp.kkt_solver = linalg.LUSciPySolver()
    # admm_lu_sp.schur_solver = linalg.LUSciPySolver()
    # variants.append((admm_lu_sp, methods_lu_sp))

    # admm_cg = copy.deepcopy(admm)
    # methods_cg = SolutionMethods(kkt=False)
    # admm_cg.schur_solver = linalg.ConjugateGradientSolver(atol=1e-7, rtol=1e-7, epsilon=1e-7, max_iterations=1000)
    # variants.append((admm_cg, methods_cg))

    # admm_minres = copy.deepcopy(admm)
    # methods_minres = SolutionMethods()
    # admm_minres.kkt_solver = linalg.MinimumResidualSolver(atol=1e-6, rtol=1e-6, epsilon=1e-6, max_iterations=1000)
    # admm_minres.schur_solver = linalg.MinimumResidualSolver(atol=1e-6, rtol=1e-6, epsilon=1e-6, max_iterations=1000)
    # variants.append((admm_minres, methods_minres))

    admm_llt_std = copy.deepcopy(admm)
    methods_llt_std = SolutionMethods(kkt=False)
    admm_llt_std.schur_solver = linalg.LLTStdSolver()
    variants.append((admm_llt_std, methods_llt_std))

    admm_ldlt_nopiv = copy.deepcopy(admm)
    methods_ldlt_nopiv = SolutionMethods()
    admm_ldlt_nopiv.kkt_solver = linalg.LDLTNoPivotSolver()
    admm_ldlt_nopiv.schur_solver = linalg.LDLTNoPivotSolver()
    variants.append((admm_ldlt_nopiv, methods_ldlt_nopiv))

    admm_ldlt_blocked = copy.deepcopy(admm)
    methods_ldlt_blocked = SolutionMethods()
    admm_ldlt_blocked.kkt_solver = linalg.LDLTBlockedSolver()
    admm_ldlt_blocked.schur_solver = linalg.LDLTBlockedSolver()
    variants.append((admm_ldlt_blocked, methods_ldlt_blocked))

    admm_ldlt_eigen3 = copy.deepcopy(admm)
    methods_ldlt_eigen3 = SolutionMethods()
    admm_ldlt_eigen3.kkt_solver = linalg.LDLTEigen3Solver()
    admm_ldlt_eigen3.schur_solver = linalg.LDLTEigen3Solver()
    variants.append((admm_ldlt_eigen3, methods_ldlt_eigen3))

    admm_lu_nopiv = copy.deepcopy(admm)
    methods_lu_nopiv = SolutionMethods()
    admm_lu_nopiv.kkt_solver = linalg.LUNoPivotSolver()
    admm_lu_nopiv.schur_solver = linalg.LUNoPivotSolver()
    variants.append((admm_lu_nopiv, methods_lu_nopiv))

    return variants


def make_benchmark_performance_data(
    metrics: list[BenchmarkMetrics], problems: list[str], output_path: str
) -> dict[str, Any]:
    # Iterate over all collected metrics and collect a list of unique solver IDs
    solvers = set()
    for m in metrics:
        solvers.add(m._solverid())
    solvers = sorted(solvers)

    # Extract the list of performance metric names
    # NOTE: this excludes the "norm_lambdas" and "norm_u_plus" entries
    metric_names = BenchmarkMetrics._metrics()
    metric_names.remove("norm_lambdas")
    metric_names.remove("norm_u_plus")

    # Create a dictionary of 2D arrays for each performance metric
    num_solvers = len(solvers)
    num_problems = len(problems)
    perfdata: dict[str, np.ndarray] = {
        metric: np.empty((num_solvers, num_problems), dtype=float) for metric in metric_names
    }

    # Populate the metric data arrays
    for m in metrics:
        s = solvers.index(m._solverid())
        p = problems.index(m.pname) if m.pname in problems else -1
        if p < 0:
            msg.error("Problem name '%s' not found in problem paths.", m.pname)
            continue
        perfdata["error"][s, p] = 1.0 if m.data.error else 0.0
        perfdata["success"][s, p] = 1.0 if m.data.success else 0.0
        perfdata["converged"][s, p] = 1.0 if m.data.converged else 0.0
        perfdata["iterations"][s, p] = float(m.data.iterations)
        perfdata["total_time"][s, p] = float(m.data.total_time)
        perfdata["iteration_time"][s, p] = float(m.data.iteration_time)
        perfdata["dual_residual_abs"][s, p] = float(m.data.dual_residual_abs)
        perfdata["primal_error_abs"][s, p] = float(m.data.primal_error_abs)
        perfdata["primal_error_rel"][s, p] = float(m.data.primal_error_rel)
        perfdata["dual_error_abs"][s, p] = float(m.data.dual_error_abs)
        perfdata["dual_error_rel"][s, p] = float(m.data.dual_error_rel)
        perfdata["kkt_error_abs"][s, p] = float(m.data.kkt_error_abs)
        perfdata["kkt_error_rel"][s, p] = float(m.data.kkt_error_rel)

    # Append solver names and problem names to the solutions dictionary
    perfdata["solvers"] = list(solvers)
    perfdata["problems"] = problems

    # Save the metric data arrays as binary a NumPy file
    np.save(os.path.join(output_path, "perfdata.npy"), perfdata)

    # Return the dictionary of performance data per solver and problem
    return perfdata


def make_benchmark_linsys_performance_data(
    metrics: list[BenchmarkMetrics], problems: list[str], output_path: str
) -> dict[str, Any]:
    # Iterate over all collected metrics and collect a list of unique solver IDs
    linsys_solvers = set()
    methods_per_solver: dict[str, set[str]] = {}
    for m in metrics:
        linsys_solvers.add(m.info.linear_solver)
        if m.info.linear_solver not in methods_per_solver:
            methods_per_solver[m.info.linear_solver] = set()
        methods_per_solver[m.info.linear_solver].add(m.info.method_name)
    linsys_solvers = sorted(linsys_solvers)

    # Extract a list of solution methods that are common between all solvers
    methods = set.intersection(*(methods_per_solver[solver] for solver in linsys_solvers))
    methods = sorted(methods)

    # For each problem, add each method as a separate entry
    linsys_problems = []
    for p in problems:
        for m in methods:
            linsys_problems.append(p + "/" + m)

    # Extract the list of performance metric names
    # NOTE: this excludes the "norm_lambdas" and "norm_u_plus" entries
    metric_names = BenchmarkMetrics._metrics_linsys_all()

    # Create a dictionary of 2D arrays for each performance metric
    num_linsys_solvers = len(linsys_solvers)
    num_linsys_problems = len(linsys_problems)
    perfdata: dict[str, np.ndarray] = {
        metric: np.empty((num_linsys_solvers, num_linsys_problems), dtype=float) for metric in metric_names
    }

    # Populate the metric data arrays
    for m in metrics:
        if m.info.method_name not in methods:
            continue
        s = linsys_solvers.index(m.info.linear_solver)
        pname = m.pname + "/" + m.info.method_name
        p = linsys_problems.index(pname) if pname in linsys_problems else -1
        if p < 0:
            msg.error("Problem name '%s' not found in problem paths.", pname)
            continue
        perfdata["compute_error_abs_min"][s, p] = float(m.data.linsys_min.compute_error_abs)
        perfdata["compute_error_abs_max"][s, p] = float(m.data.linsys_max.compute_error_abs)
        perfdata["compute_error_abs_mean"][s, p] = float(m.data.linsys_mean.compute_error_abs)
        perfdata["compute_error_rel_min"][s, p] = float(m.data.linsys_min.compute_error_rel)
        perfdata["compute_error_rel_max"][s, p] = float(m.data.linsys_max.compute_error_rel)
        perfdata["compute_error_rel_mean"][s, p] = float(m.data.linsys_mean.compute_error_rel)
        perfdata["solve_error_abs_min"][s, p] = float(m.data.linsys_min.solve_error_abs)
        perfdata["solve_error_abs_max"][s, p] = float(m.data.linsys_max.solve_error_abs)
        perfdata["solve_error_abs_mean"][s, p] = float(m.data.linsys_mean.solve_error_abs)
        perfdata["solve_error_rel_min"][s, p] = float(m.data.linsys_min.solve_error_rel)
        perfdata["solve_error_rel_max"][s, p] = float(m.data.linsys_max.solve_error_rel)
        perfdata["solve_error_rel_mean"][s, p] = float(m.data.linsys_mean.solve_error_rel)

    # Append solver names and problem names to the solutions dictionary
    perfdata["solvers"] = list(linsys_solvers)
    perfdata["problems"] = linsys_problems

    # Save the metric data arrays as binary a NumPy file
    np.save(os.path.join(output_path, "perfdata_linsys.npy"), perfdata)

    # Return the populated metrics dictionary
    return perfdata


def filter_perfdata_admm(
    perfdata_in: dict,
    exclude_methods: list[str] | None = None,
    keep_methods: list[str] | None = None,
    exclude_linsys: list[str] | None = None,
    keep_linsys: list[str] | None = None,
    exclude_metrics: list[str] | None = None,
    keep_metrics: list[str] | None = None,
) -> dict:
    # Initialize lists if None
    if exclude_methods is None:
        exclude_methods = []
    if keep_methods is None:
        keep_methods = []
    if exclude_linsys is None:
        exclude_linsys = []
    if keep_linsys is None:
        keep_linsys = []
    if exclude_metrics is None:
        exclude_metrics = []
    if keep_metrics is None:
        keep_metrics = []

    # Find filtered solvers and their corresponding indices
    solvers = []
    solver_indices = []
    for solver in perfdata_in["solvers"]:
        # First check if the solver contains any excluded substrings
        has_excluded = False
        for ex in exclude_methods + exclude_linsys:
            if ex in solver:
                has_excluded = True
                break
        if has_excluded:
            continue
        # Then check if the solver contains any of the kept method substrings
        has_kept_method = False if keep_methods else True
        for kp in keep_methods:
            if kp in solver:
                has_kept_method = True
                break
        if not has_kept_method:
            continue
        # Finally check if the solver contains any of the kept linsys substrings
        has_kept_linsys = False if keep_linsys else True
        for kp in keep_linsys:
            if kp in solver:
                has_kept_linsys = True
                break
        if not has_kept_linsys:
            continue
        # If all checks passed, keep this solver
        solvers.append(solver)
        solver_indices.append(perfdata_in["solvers"].index(solver))

    # Find filtered metrics
    metric_names = list(perfdata_in.keys())
    metric_names.remove("solvers")
    metric_names.remove("problems")
    # First remove excluded metrics
    for ex in exclude_metrics:
        if ex in metric_names:
            metric_names.remove(ex)
    # Then keep only specified metrics
    for m in metric_names:
        if keep_metrics and m not in keep_metrics:
            metric_names.remove(m)

    # Construct the filtered performance data dictionary
    perfdata: dict[str, np.ndarray] = {
        metric: np.empty((len(solvers), len(perfdata_in["problems"])), dtype=float) for metric in metric_names
    }
    for metric in metric_names:
        for i, sid in enumerate(solver_indices):
            perfdata[metric][i, :] = perfdata_in[metric][sid, :]
    perfdata["solvers"] = solvers
    perfdata["problems"] = perfdata_in["problems"]

    # Return the filtered performance data
    return perfdata


def filter_perfdata_linsys(
    perfdata_in: dict,
    exclude_solvers: list[str] | None = None,
    keep_solvers: list[str] | None = None,
    exclude_metrics: list[str] | None = None,
    keep_metrics: list[str] | None = None,
) -> dict:
    # Initialize lists if None
    if exclude_solvers is None:
        exclude_solvers = []
    if keep_solvers is None:
        keep_solvers = []
    if exclude_metrics is None:
        exclude_metrics = []
    if keep_metrics is None:
        keep_metrics = []

    # Find filtered solvers and their corresponding indices
    solvers = []
    solver_indices = []
    for solver in perfdata_in["solvers"]:
        # First check if the solver contains any excluded substrings
        has_excluded = False
        for ex in exclude_solvers:
            if ex in solver:
                has_excluded = True
                break
        if has_excluded:
            continue
        # Then check if the solver contains any of the kept method substrings
        has_kept_method = False if keep_solvers else True
        for kp in keep_solvers:
            if kp in solver:
                has_kept_method = True
                break
        if not has_kept_method:
            continue
        # If all checks passed, keep this solver
        solvers.append(solver)
        solver_indices.append(perfdata_in["solvers"].index(solver))

    # Find filtered metrics
    metric_names = list(perfdata_in.keys())
    metric_names.remove("solvers")
    metric_names.remove("problems")
    # First remove excluded metrics
    for ex in exclude_metrics:
        if ex in metric_names:
            metric_names.remove(ex)
    # Then keep only specified metrics
    for m in metric_names:
        if keep_metrics and m not in keep_metrics:
            metric_names.remove(m)

    # Construct the filtered performance data dictionary
    perfdata: dict[str, np.ndarray] = {
        metric: np.empty((len(solvers), len(perfdata_in["problems"])), dtype=float) for metric in metric_names
    }
    for metric in metric_names:
        for i, sid in enumerate(solver_indices):
            perfdata[metric][i, :] = perfdata_in[metric][sid, :]
    perfdata["solvers"] = solvers
    perfdata["problems"] = perfdata_in["problems"]

    # Return the filtered performance data
    return perfdata


def make_performance_profiles(
    perfdata: dict[str, Any],
    success_key: str | None = None,
    exclude: list[str] | None = None,
    show: bool = False,
    path: str | None = None,
) -> dict[str, PerformanceProfile]:
    # Initialize profiles and rankings containers
    profiles: dict[str, PerformanceProfile] = {}

    # Extract the dictionary keys and remove the excluded ones
    metric_keys = list(perfdata.keys())
    if exclude is not None:
        for key in exclude:
            metric_keys.remove(key)
    metric_keys.remove("solvers")
    metric_keys.remove("problems")

    # Compute performance profiles for selected metrics and store them in the profiles dictionary
    success = None
    if success_key is not None and success_key in metric_keys:
        success = perfdata[success_key]
    for key in metric_keys:
        profiles[key] = PerformanceProfile(data=perfdata[key], success=success, taumax=np.inf)

    # Create output directory if it doesn't exist
    if path is not None:
        os.makedirs(path, exist_ok=True)

    # Render performance profiles to files
    for key, entry in profiles.items():
        title = " ".join(part.capitalize() for part in key.split("_"))
        entry.plot(perfdata["solvers"], title=title, show=show, path=os.path.join(path, f"{key}.png"))

    # Return the dictionary of performance profiles per metric
    return profiles


def make_perfprof_rankings(profiles: dict[str, PerformanceProfile]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rankings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    keys = list(profiles.keys())
    for key in keys:
        rankings[key] = profiles[key].rankings()
    return rankings


def compute_total_perfprof_rankings(
    rankings: dict[str, tuple[np.ndarray, np.ndarray]],
    exclude_metrics: list[str] | None = None,
    keep_metrics: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Extract the dictionary keys and remove the excluded ones and those not in the kept ones
    metrics = list(rankings.keys())
    if exclude_metrics is not None:
        for key in list(metrics):
            if key in exclude_metrics:
                metrics.remove(key)
    if keep_metrics is not None:
        for key in list(metrics):
            if key not in keep_metrics:
                metrics.remove(key)
    # Compute total rankings across all metrics
    rho1_total_rankings = np.zeros_like(rankings[metrics[0]][0]).astype(float)
    tau1_total_rankings = np.zeros_like(rankings[metrics[0]][1]).astype(float)
    for m in metrics:
        r0, r1 = rankings[m]
        rho1_total_rankings += r0
        tau1_total_rankings += r1
    rho1_total_rankings /= float(len(metrics))
    tau1_total_rankings /= float(len(metrics))
    tau1_total_rankings = np.round(rho1_total_rankings).astype(int)
    tau1_total_rankings = np.round(tau1_total_rankings).astype(int)
    # Return the dictionary of performance profile rankings
    return (rho1_total_rankings, tau1_total_rankings)


def make_summary_table(perfdata: dict[str, Any]) -> str:
    # Extract solvers and performance data
    solvers: list[str] = perfdata["solvers"]
    solved: np.ndarray = perfdata["success"]
    converged: np.ndarray = perfdata["converged"]
    errors: np.ndarray = perfdata["error"]

    # Ensure input shapes are correct
    if solved.ndim != 2 or converged.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional.")
    if solved.shape != converged.shape:
        raise ValueError("Input arrays must have the same shape.")
    if solved.shape[0] != len(solvers):
        raise ValueError("Number of solvers must match the number of rows in input arrays.")

    # Compute total number of problems and solvers
    num_problems = solved.shape[1]

    # Compute a summary of the number of errors encountered by each solver
    errors = np.sum(errors, axis=1)
    errors_pc = 100.0 * errors / float(num_problems)

    # Compute a summary of the number of problems solved by each solver
    solved = np.sum(solved, axis=1)
    solved_pc = 100.0 * solved / float(num_problems)

    # Compute a summary of the number of problems in which each solver converged
    converged = np.sum(converged, axis=1)
    converged_pc = 100.0 * converged / float(num_problems)

    # Prepare title, columns, and rows
    columns = [
        "Solver",
        "(#) Solved",
        "(%) Solved",
        "(#) Converged",
        "(%) Converged",
        "(#) Errors",
        "(%) Errors",
    ]
    nprobs = str(int(num_problems))
    rows: list[list[str]] = []
    for i, sid in enumerate(solvers):
        rows.append(
            [
                sid,
                str(int(solved[i])),
                f"{solved_pc[i]:.1f}%",
                str(int(converged[i])),
                f"{converged_pc[i]:.1f}%",
                str(int(errors[i])),
                f"{errors_pc[i]:.1f}%",
            ]
        )

    # Compute column widths
    widths = [len(c) for c in columns]
    for r in rows:
        for j, val in enumerate(r):
            widths[j] = max(widths[j], len(val))

    # Build title, header, separator, and data rows
    title = f"Solver Performance Summary: {nprobs} problems"
    total_width = sum(widths) + 3 * (len(columns) - 1)
    title = title.center(total_width)
    header = " | ".join(columns[j].center(widths[j]) for j in range(len(columns)))
    rule = "-+-".join("-" * widths[j] for j in range(len(columns)))
    lines = [title, rule, header, rule]
    for r in rows:
        lines.append(" | ".join(val.rjust(widths[j]) for j, val in enumerate(r)))
    lines.append(rule)

    # Return the formatted table as a string
    return "\n".join(lines)


def make_rankings_table(solvers: list[str], rankings: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
    # Determine metric order
    keys = list(rankings.keys())

    # Helper formatter
    def fmt(v: Any) -> str:
        try:
            if isinstance(v, np.integer | int):
                return str(int(v))
            if isinstance(v, np.floating | float):
                if abs(float(v) - round(float(v))) < 1e-12:
                    return str(int(round(float(v))))
                return f"{float(v):.4g}"
            return str(v)
        except Exception:
            return str(v)

    # Build rows
    rows: list[list[str]] = []
    for i, sid in enumerate(solvers):
        row: list[str] = [sid]
        for m in keys:
            r0, r1 = rankings[m]
            v0 = r0[i] if i < len(r0) else np.nan
            v1 = r1[i] if i < len(r1) else np.nan
            row.append(fmt(v0))
            row.append(fmt(v1))
        rows.append(row)

    # Build submetrics
    submetrics = ["tau=1", "rho=1"]
    submetrics_cell = submetrics[0] + "  |  " + submetrics[1]
    submetrics_cell_width = len(submetrics_cell)
    max_metric_width = max(len(m) for m in keys)  # w/o space padding
    max_solver_width = max(len(s) for s in solvers)  # w/o space padding

    # Compute column widths
    column_width = max(submetrics_cell_width, max_metric_width)
    subcolumn_width = max(max(len(sm) for sm in submetrics) + 2, column_width // 2 - 1)

    # Build headers columns
    header_columns = ["Solvers", *keys]
    headers_widths = [max_solver_width] + [column_width for _ in range(len(keys))]
    subheaders_widths = [max_solver_width] + [subcolumn_width for _ in range(2 * len(keys))]
    total_width = sum(headers_widths) + 3 * (len(header_columns) - 1)

    # Build header: 'Solvers' plus each metric name spanning two data columns
    header_cells = ["Solvers".center(max_solver_width)]
    for m in keys:
        header_cells.append(m.center(column_width))
    header = " | ".join(header_cells)

    # Build subheaders:
    subheader_cells = [" ".center(max_solver_width)]
    for _ in keys:
        subheader_cells.append(submetrics_cell.center(column_width))
    subheader = " | ".join(subheader_cells)

    # Build rules
    rule_header = "-+-".join("-" * w for w in headers_widths)
    rule_header = "-+-".join("-" * w for w in headers_widths)
    rule_middle = max_solver_width * " " + " |-" + "-+-".join("-" * w for w in headers_widths[1:])
    rule_body = "-+-".join("-" * w for w in subheaders_widths)

    # Build table string
    title = "Solver Rankings"
    lines = [
        title.center(total_width),
        rule_header,
        header,
        rule_middle,
        subheader,
        rule_body,
    ]
    for row in rows:
        lines.append(" | ".join(row[j].rjust(subheaders_widths[j]) for j in range(len(subheaders_widths))))
    lines.append(rule_body)
    return "\n".join(lines)


###
# Utilities
###


def clip_below(A: np.ndarray, min: float = 0.0) -> np.ndarray:
    A_clip = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.abs(A[i, j]) < min:
                A_clip[i, j] = 0.0
            else:
                A_clip[i, j] = A[i, j]
    return A_clip


def symmetry_info(A: np.ndarray, name: str = "A", title: str = "A", path: str | None = None, eps: float = 1e-12):
    # Compute the error matrix between the kamino Delassus matrix and its transpose
    A_sym_err = A - A.T

    # Print error statistics
    print_error_stats(f"{name}_sym", A, A.T, n=A.size, show_errors=False)

    # Clip small errors to zero for visualization
    A_sym_err_clip = clip_below(A_sym_err, min=eps)

    # Visualize the error matrix as an image
    os.makedirs(path, exist_ok=True)
    sparseview(A, title=title, path=os.path.join(path, f"{name}.png"))
    sparseview(A_sym_err, title=f"{title} Symmetry Error", path=os.path.join(path, f"{name}_sym_err.png"))
    sparseview(
        A_sym_err_clip,
        title=f"{title} Symmetry Error (Clipped)",
        path=os.path.join(path, f"{name}_sym_err_clip.png"),
    )
