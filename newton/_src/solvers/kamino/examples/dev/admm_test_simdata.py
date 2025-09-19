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

"""TODO"""

import copy
import os
import time
from dataclasses import dataclass
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
    error: bool = False
    success: bool = False
    converged: bool = False
    iterations: int = 0
    total_time: float = np.inf
    iteration_time: float = np.inf
    primal_residual_inf: float = np.inf
    dual_residual_inf: float = np.inf
    compl_residual_inf: float = np.inf
    iter_residual_inf: float = np.inf


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
            "primal_residual_inf",
            "dual_residual_inf",
            "compl_residual_inf",
            "iter_residual_inf",
            "primal_error_abs",
            "primal_error_rel",
            "dual_error_abs",
            "dual_error_rel",
            "kkt_error_abs",
            "kkt_error_rel",
            "norm_lambdas",
            "norm_u_plus",
        ]

    def _solverid(self) -> str:
        """Concatenate info fields to a final solver name."""
        if self.info is None:
            return "Unknown"
        fields = (self.info.solver_name, self.info.method_name, self.info.linear_solver, f"{self.info.dtype}")
        parts = [p for p in fields if p]
        return " / ".join(parts) if parts else "Unknown"

    def _values(self) -> dict[str, str]:
        """Map of metric name -> stringified value for this instance."""
        d = self.data or SolutionMetrics()

        def fmt(v: Any) -> str:
            if isinstance(v, float | np.floating | np.float32 | np.float64):
                # Format using dtype precision if available; fall back to 6 significant digits
                try:
                    if int(abs(v)) >= 100:
                        return f"{v:.{max(1, min(np.finfo(np.dtype(self.info.dtype)).precision + 1, 4))}e}"
                    else:
                        return f"{v:.{max(1, min(np.finfo(np.dtype(self.info.dtype)).precision + 1, 12))}g}"
                except Exception:
                    return f"{v:.6e}"
            return str(v)

        return {
            "error": fmt(d.error),
            "success": fmt(d.success),
            "converged": fmt(d.converged),
            "iterations": fmt(d.iterations),
            "total_time": fmt(d.total_time),
            "iteration_time": fmt(d.iteration_time),
            "primal_residual_inf": fmt(d.primal_residual_inf),
            "dual_residual_inf": fmt(d.dual_residual_inf),
            "compl_residual_inf": fmt(d.compl_residual_inf),
            "iter_residual_inf": fmt(d.iter_residual_inf),
            "primal_error_abs": fmt(d.primal_error_abs),
            "primal_error_rel": fmt(d.primal_error_rel),
            "dual_error_abs": fmt(d.dual_error_abs),
            "dual_error_rel": fmt(d.dual_error_rel),
            "kkt_error_abs": fmt(d.kkt_error_abs),
            "kkt_error_rel": fmt(d.kkt_error_rel),
            "norm_lambdas": fmt(d.norm_lambdas),
            "norm_u_plus": fmt(d.norm_u_plus),
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
    r = primal_dynamics_residual(problem, solution)
    r_norm = np.linalg.norm(r, ord=np.inf)
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
        msg.debug("K properties: %s", properties_K)

    # Optionally render the KKT matrix and symmetry error info as images
    if save_symmetry_info:
        sparseview(K, title="KKT Matrix", path=os.path.join(PLOT_OUTPUT_PATH, "K.png"))
        symmetry_info(K, name="K", title="KKT", eps=np.finfo(problem.M.dtype).eps)

    # Pack quantities into the linear system container
    return LinearSystemProblem(A=K, b=k)


def make_dual_system(
    problem: ConstrainedDynamicsProblem,
    ensure_symmetric: bool = False,
    save_matrix_info: bool = False,
    save_symmetry_info: bool = False,
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
        msg.debug("D properties: %s", properties_D)

    # Optionally render the primal schur complement matrix and symmetry error info as images
    if save_symmetry_info:
        sparseview(D, title="Delassus", path=os.path.join(PLOT_OUTPUT_PATH, "D.png"))
        symmetry_info(D, name="D", title="Dual Schur-complement", eps=np.finfo(problem.D.dtype).eps)

    # Pack quantities into the linear system container
    return LinearSystemProblem(A=D, b=d)


def make_benchmark_problem(
    name: str,
    problem: ConstrainedDynamicsProblem,
    ensure_symmetric: bool = False,
    save_matrix_info: bool = False,
    save_symmetry_info: bool = False,
) -> BenchmarkProblem:
    """Create a BenchmarkProblem from a ConstrainedDynamicsProblem."""
    kkt = make_kkt_system(
        problem,
        ensure_symmetric=ensure_symmetric,
        save_matrix_info=save_matrix_info,
        save_symmetry_info=save_symmetry_info,
    )
    dual = make_dual_system(
        problem,
        ensure_symmetric=ensure_symmetric,
        save_matrix_info=save_matrix_info,
        save_symmetry_info=save_symmetry_info,
    )
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
    status: linalg.ADMMStatus,
    problem: BenchmarkProblem,
    solution: BenchmarkSolution,
) -> BenchmarkMetrics:
    # Create a new metrics container
    metrics = BenchmarkMetrics(pname=problem.name, info=solution.info, data=SolutionMetrics())

    # First set wether there was an error
    metrics.data.error = status.result == linalg.ADMMResult.ERROR

    # If there was an error, return early so that all other metrics remain at infinity
    if metrics.data.error:
        return metrics

    # Set the solution metrics
    metrics.data.success = status.result <= linalg.ADMMResult.MAXITER
    metrics.data.converged = status.converged
    metrics.data.iterations = status.iterations
    metrics.data.total_time = time
    metrics.data.iteration_time = time / status.iterations if status.iterations > 0 else np.inf
    metrics.data.primal_residual_inf = status.r_p
    metrics.data.dual_residual_inf = status.r_d
    metrics.data.compl_residual_inf = status.r_c
    metrics.data.iter_residual_inf = status.r_i

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
    problem: BenchmarkProblem, admm: linalg.ADMMSolver, methods: SolutionMethods, save_info: bool = False
) -> list[BenchmarkMetrics]:
    """Solve a ConstrainedDynamicsProblem using an ADMM solver."""

    # Initialize solver metrics list
    metrics = []

    # Extract solver info
    solver_name = get_solver_typename(admm)

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
        solution_kkt = make_benchmark_solution(admm, kkt_info)
        metrics.append(make_benchmark_metrics(time_kkt, status_kkt, problem, solution_kkt))

        # Optionally save convergence plots
        if save_info:
            admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_kkt")

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
        solution_schur_prim = make_benchmark_solution(admm, schur_prim_info)
        metrics.append(make_benchmark_metrics(time_schur_prim, status_schur_prim, problem, solution_schur_prim))

        # Optionally save convergence plots
        if save_info:
            admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_schur_prim")

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
        solution_schur_dual = make_benchmark_solution(admm, schur_dual_info)
        metrics.append(make_benchmark_metrics(time_schur_dual, status_schur_dual, problem, solution_schur_dual))

        # Optionally save convergence plots
        if save_info:
            admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_schur_dual")

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
        solution_schur_dual_prec = make_benchmark_solution(admm, schur_dual_prec_info)
        metrics.append(
            make_benchmark_metrics(time_schur_dual_prec, status_schur_dual_prec, problem, solution_schur_dual_prec)
        )

        # Optionally save convergence plots
        if save_info:
            admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_schur_dual_prec")

    # Return all metrics for this problem
    return metrics


def make_solvers(admm: linalg.ADMMSolver) -> list[tuple[linalg.ADMMSolver, SolutionMethods]]:
    variants: list[tuple[linalg.ADMMSolver, SolutionMethods]] = []

    admm_np = copy.deepcopy(admm)
    methods_np = SolutionMethods()
    admm_np.kkt_solver = linalg.NumPySolver()
    admm_np.schur_solver = linalg.NumPySolver()
    variants.append((admm_np, methods_np))

    admm_sp = copy.deepcopy(admm)
    methods_sp = SolutionMethods()
    admm_sp.kkt_solver = linalg.SciPySolver()
    admm_sp.schur_solver = linalg.SciPySolver()
    variants.append((admm_sp, methods_sp))

    admm_llt_np = copy.deepcopy(admm)
    methods_llt_np = SolutionMethods(kkt=False)
    admm_llt_np.schur_solver = linalg.LLTNumPySolver()
    variants.append((admm_llt_np, methods_llt_np))

    admm_llt_sp = copy.deepcopy(admm)
    methods_llt_sp = SolutionMethods(kkt=False)
    admm_llt_sp.schur_solver = linalg.LLTSciPySolver()
    variants.append((admm_llt_sp, methods_llt_sp))

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

    # admm_llt_std = copy.deepcopy(admm)
    # methods_llt_std = SolutionMethods(kkt=False)
    # admm_llt_std.schur_solver = linalg.LLTStdSolver()
    # variants.append((admm_llt_std, methods_llt_std))

    # admm_ldlt_nopiv = copy.deepcopy(admm)
    # methods_ldlt_nopiv = SolutionMethods()
    # admm_ldlt_nopiv.kkt_solver = linalg.LDLTNoPivotSolver()
    # admm_ldlt_nopiv.schur_solver = linalg.LDLTNoPivotSolver()
    # variants.append((admm_ldlt_nopiv, methods_ldlt_nopiv))

    # admm_ldlt_blocked = copy.deepcopy(admm)
    # methods_ldlt_blocked = SolutionMethods()
    # admm_ldlt_blocked.kkt_solver = linalg.LDLTBlockedSolver()
    # admm_ldlt_blocked.schur_solver = linalg.LDLTBlockedSolver()
    # variants.append((admm_ldlt_blocked, methods_ldlt_blocked))

    # admm_ldlt_eigen3 = copy.deepcopy(admm)
    # methods_ldlt_eigen3 = SolutionMethods()
    # admm_ldlt_eigen3.kkt_solver = linalg.LDLTEigen3Solver()
    # admm_ldlt_eigen3.schur_solver = linalg.LDLTEigen3Solver()
    # variants.append((admm_ldlt_eigen3, methods_ldlt_eigen3))

    # admm_lu_nopiv = copy.deepcopy(admm)
    # methods_lu_nopiv = SolutionMethods()
    # admm_lu_nopiv.kkt_solver = linalg.LUNoPivotSolver()
    # admm_lu_nopiv.schur_solver = linalg.LUNoPivotSolver()
    # variants.append((admm_lu_nopiv, methods_lu_nopiv))

    return variants


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


def symmetry_info(A: np.ndarray, name: str = "A", title: str = "A", eps: float = 1e-12):
    # Compute the error matrix between the kamino Delassus matrix and its transpose
    A_sym_err = A - A.T

    # Print error statistics
    print_error_stats(f"{name}_sym", A, A.T, n=A.size, show_errors=False)

    # Clip small errors to zero for visualization
    A_sym_err_clip = clip_below(A_sym_err, min=eps)

    # Visualize the error matrix as an image
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)
    sparseview(A_sym_err, title=f"{title} Symmetry Error", path=os.path.join(PLOT_OUTPUT_PATH, f"{name}_sym_err.png"))
    sparseview(
        A_sym_err_clip,
        title=f"{title} Symmetry Error (Clipped)",
        path=os.path.join(PLOT_OUTPUT_PATH, f"{name}_sym_err_clip.png"),
    )


def make_summary_table(
    solvers: list[str],
    errors: np.ndarray,
    solved: np.ndarray,
    converged: np.ndarray,
) -> str:
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
# PROBLEM_NAME = "fourbar_free"
PROBLEM_NAME = "walker"

# Sample category to load; set to None to load all categories
# PROBLEM_CATEGORY = None
# PROBLEM_CATEGORY = "IndependentJoints"
# PROBLEM_CATEGORY = "RedundantJoints"
PROBLEM_CATEGORY = "SingleContact"
# PROBLEM_CATEGORY = "SparseContacts"
# PROBLEM_CATEGORY = "DenseConstraints"

# Sample index to load; set to None to load all samples
# PROBLEM_SAMPLE = None
PROBLEM_SAMPLE = 0

# List of keys to exclude when searching for problems
EXCLUDE = ["Unconstrained"]

# Retrieve the path to the data directory
DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# Set the path to the generated HDF5 dataset file
HDF5_DATASET_PATH = f"{DATA_DIR_PATH}/hdf5/simdata.hdf5"

# Set path for generated plots
PLOT_OUTPUT_PATH = f"{DATA_DIR_PATH}/plots/simdata/{PROBLEM_NAME}"

###
# Main function
###

if __name__ == "__main__":
    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation
    msg.set_log_level(msg.LogLevel.INFO)

    # Create output directories
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)

    # Construct and configure the data containers
    msg.info("Loading HDF5 data containers...")
    datafile = h5py.File(HDF5_DATASET_PATH, "r")

    # Select the numpy data type for computations
    # np_dtype = np.float64
    np_dtype = np.float32

    # CONFIGURATIONS
    sample: bool = False
    info: bool = True
    dataset: bool = True
    summary: bool = True
    profiles: bool = True

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
    solvers = make_solvers(admm_0)

    ###
    # Single sample problem
    ###

    if sample:
        # Retrieve target data frames
        fpath = build_frame_path(PROBLEM_TYPE, PROBLEM_NAME, PROBLEM_CATEGORY, PROBLEM_SAMPLE)
        msg.info(f"Retrieving data frame at path '{fpath}'...")
        dataframe = datafile[fpath]

        # Load the problem data into a container
        msg.info(f"Loading problem data from '{dataframe.name}'...")
        problem = make_benchmark_problem(
            name=fpath,
            problem=load_problem_data(dataframe=dataframe, dtype=np_dtype),
            ensure_symmetric=False,
            save_matrix_info=False,
            save_symmetry_info=False,
        )

        # Print problem info
        if info:
            msg.info("Problem info:\n%s", problem.crbd.info)

        # Initialize a list to store benchmark metrics
        metrics: list[BenchmarkMetrics] = []

        # Iterate over all solver variants
        msg.info(f"Solving problem '{problem.name}'...")
        for admm, methods in solvers:
            metrics.extend(solve_benchmark_problem(problem, admm, methods, True))

        # Print all collected metrics
        for m in metrics:
            print(f"\n{m}\n")

    ###
    # Benchmark problems
    ###

    if dataset:
        # Find and print all DualProblem paths
        search_scope = build_frame_path(PROBLEM_TYPE, PROBLEM_NAME, PROBLEM_CATEGORY)
        msg.info(f"Searching for DualProblem paths in scope '{search_scope}'...")
        problem_paths = find_problem_paths(datafile=datafile, scope=search_scope, exclude=EXCLUDE)
        msg.info(f"Found {len(problem_paths)} paths containing DualProblem data.")
        # for path in problem_paths:
        #     print(f"- {path}")

        # Iterate over all found DualProblem paths
        metrics: list[BenchmarkMetrics] = []
        msg.info("Iterating over all found DualProblem paths...")
        for path in problem_paths:
            problem = make_benchmark_problem(
                name=path,
                problem=load_problem_data(dataframe=datafile[path], dtype=np_dtype),
                ensure_symmetric=False,
                save_matrix_info=False,
                save_symmetry_info=False,
            )
            for admm, methods in solvers:
                metrics.extend(solve_benchmark_problem(problem, admm, methods, False))
        # for m in metrics:
        #     print(f"\n{m}\n")

        # Iterate over all collected metrics and collect a list of unique solver IDs
        solver_names = set()
        for m in metrics:
            solver_names.add(m._solverid())
        solver_names = sorted(solver_names)
        msg.info(f"Collected metrics for {len(solver_names)} unique solver ID(s).")
        for s in solver_names:
            print(f"- {s}")

        # Print summary of all collected metrics
        num_solvers = len(solver_names)
        num_problems = len(problem_paths)
        msg.info(f"num_solvers = {num_solvers}")
        msg.info(f"num_problems = {num_problems}")

        ###
        # Performance metrics
        ###

        # Create a dictionary of 2D arrays for each metric
        metric_names = BenchmarkMetrics._metrics()
        solutions: dict[str, np.ndarray] = {
            metric: np.empty((num_solvers, num_problems), dtype=float) for metric in metric_names
        }

        # Populate the metric data arrays
        msg.info("Populating metric data arrays...")
        for metric in metrics:
            s = solver_names.index(metric._solverid())
            p = problem_paths.index(metric.pname) if metric.pname in problem_paths else -1
            if p < 0:
                msg.error("Problem name '%s' not found in problem paths.", metric.pname)
                continue
            solutions["error"][s, p] = 1.0 if metric.data.error else 0.0
            solutions["success"][s, p] = 1.0 if metric.data.success else 0.0
            solutions["converged"][s, p] = 1.0 if metric.data.converged else 0.0
            solutions["iterations"][s, p] = float(metric.data.iterations)
            solutions["total_time"][s, p] = metric.data.total_time
            solutions["iteration_time"][s, p] = metric.data.iteration_time
            solutions["primal_residual_inf"][s, p] = metric.data.primal_residual_inf
            solutions["dual_residual_inf"][s, p] = metric.data.dual_residual_inf
            solutions["compl_residual_inf"][s, p] = metric.data.compl_residual_inf
            solutions["iter_residual_inf"][s, p] = metric.data.iter_residual_inf
            solutions["primal_error_abs"][s, p] = metric.data.primal_error_abs
            solutions["primal_error_rel"][s, p] = metric.data.primal_error_rel
            solutions["dual_error_abs"][s, p] = metric.data.dual_error_abs
            solutions["dual_error_rel"][s, p] = metric.data.dual_error_rel
            solutions["kkt_error_abs"][s, p] = metric.data.kkt_error_abs
            solutions["kkt_error_rel"][s, p] = metric.data.kkt_error_rel

    ###
    # Performance summary
    ###

    if dataset and summary:
        # Print a coarse summary of solver success rates
        msg.info(
            "SUMMARY:\n\n%s",
            make_summary_table(solver_names, solutions["error"], solutions["success"], solutions["converged"]),
        )

    ###
    # Performance profiles
    ###

    if dataset and profiles:
        # Compute performance profiles for selected metrics
        msg.info("Computing performance profiles...")
        successes = solutions["success"]
        pp_dual_residual_inf = PerformanceProfile(data=solutions["dual_residual_inf"], success=successes, taumax=np.inf)
        pp_primal_error_abs = PerformanceProfile(data=solutions["primal_error_abs"], success=successes, taumax=np.inf)
        pp_primal_error_rel = PerformanceProfile(data=solutions["primal_error_rel"], success=successes, taumax=np.inf)
        pp_dual_error_abs = PerformanceProfile(data=solutions["dual_error_abs"], success=successes, taumax=np.inf)
        pp_dual_error_rel = PerformanceProfile(data=solutions["dual_error_rel"], success=successes, taumax=np.inf)
        pp_kkt_error_abs = PerformanceProfile(data=solutions["kkt_error_abs"], success=successes, taumax=np.inf)
        pp_kkt_error_rel = PerformanceProfile(data=solutions["kkt_error_rel"], success=successes, taumax=np.inf)
        pp_iteration_time = PerformanceProfile(data=solutions["iteration_time"], success=successes, taumax=np.inf)
        pp_total_time = PerformanceProfile(data=solutions["total_time"], success=successes, taumax=np.inf)

        # Render performance profiles to files
        msg.info("Rendering performance profiles plots...")
        solvers_list = list(solver_names)
        pp_dual_residual_inf.plot(solvers_list, title="Dual Residual")
        pp_primal_error_abs.plot(solvers_list, title="Primal Absolute Error")
        pp_primal_error_rel.plot(solvers_list, title="Primal Relative Error")
        pp_dual_error_abs.plot(solvers_list, title="Dual Absolute Error")
        pp_dual_error_rel.plot(solvers_list, title="Dual Relative Error")
        pp_kkt_error_abs.plot(solvers_list, title="KKT Absolute Error")
        pp_kkt_error_rel.plot(solvers_list, title="KKT Relative Error")
        pp_iteration_time.plot(solvers_list, title="Iteration Time")
        pp_total_time.plot(solvers_list, title="Total Time")

    # TODO:
    #   - Add collection of linysys performance in each problem to create linsys performance profiles
    #   - Create metric-vs-problem_size plots for each metric

    # Close the HDF5 data file
    datafile.close()
    msg.info("Done.")
