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
    error_abs: float = np.inf
    error_rel: float = np.inf
    solved: bool = False


@dataclass
class DirectSolverMetrics(LinearSolverMetrics):
    factorization_error_abs: float = np.inf
    factorization_error_rel: float = np.inf


@dataclass
class IndirectSolverMetrics(LinearSolverMetrics):
    residual_abs: float = np.inf
    residual_rel: float = np.inf
    iterations: int = 0
    converged: bool = False


###
# Dynamics Problem Containers
###


@dataclass
class ConstrainedDynamicsProblem:
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
class SolutionInfo:
    dtype: np.dtype | None = None
    solver_name: str | None = None
    method_name: str | None = None
    linear_solver: str | None = None


@dataclass
class SolutionMetrics(ConstrainedDynamicsMetrics):
    converged: bool = False
    iterations: int = 0
    total_time: float = np.inf
    iteration_time: float = np.inf
    primal_residual_inf: float = np.inf
    dual_residual_inf: float = np.inf
    compl_residual_inf: float = np.inf


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
            "converged",
            "iterations",
            "total_time",
            "iteration_time",
            "primal_residual_inf",
            "dual_residual_inf",
            "compl_residual_inf",
            "primal_error_abs",
            "primal_error_rel",
            "dual_error_abs",
            "dual_error_rel",
            "kkt_error_abs",
            "kkt_error_rel",
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
            if isinstance(v, float):
                # Format using dtype precision if available; fall back to 6 significant digits
                try:
                    dt = np.dtype(self.info.dtype) if (self.info and self.info.dtype is not None) else None
                    if dt is not None and np.issubdtype(dt, np.floating):
                        sig = int(np.finfo(dt).precision)
                        sig = max(1, min(sig, 12))  # clamp for readability
                        return f"{v:.{sig}g}"
                except Exception:
                    return f"{v:.6g}"
            return str(v)

        return {
            "converged": fmt(d.converged),
            "iterations": fmt(d.iterations),
            "total_time": fmt(d.total_time),
            "iteration_time": fmt(d.iteration_time),
            "primal_residual_inf": fmt(d.primal_residual_inf),
            "dual_residual_inf": fmt(d.dual_residual_inf),
            "compl_residual_inf": fmt(d.compl_residual_inf),
            "primal_error_abs": fmt(d.primal_error_abs),
            "primal_error_rel": fmt(d.primal_error_rel),
            "dual_error_abs": fmt(d.dual_error_abs),
            "dual_error_rel": fmt(d.dual_error_rel),
            "kkt_error_abs": fmt(d.kkt_error_abs),
            "kkt_error_rel": fmt(d.kkt_error_rel),
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
    return problem.M @ (solution.u_plus - problem.u_minus) - problem.J.T @ solution.lambdas + problem.h


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


def find_dualproblem_paths(datafile: h5py.File, scope: str | None) -> list[str]:
    """Recursively find all paths ending with '/DualProblem' in an HDF5 file."""

    # Initialize an empty list to store found paths
    paths: list[str] = []

    # Define a visitor function for HDF5 traversal
    def _visitor(name: str, obj):
        # Ensure the terminal component is exactly 'DualProblem'
        if name.rsplit("/", 1)[-1] == "DualProblem":
            # Accept both groups and datasets named 'DualProblem'
            if isinstance(obj, h5py.Group | h5py.Dataset):
                paths.append(obj.name)  # absolute path

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


def load_dualproblem_data(dataframe: h5py.Group, dtype: type = np.float64) -> ConstrainedDynamicsProblem:
    """Load a DualProblem dataset from an HDF5 group into a ConstrainedDynamicsProblem container."""

    # Load the DualProblem data into the HDF5 data-frame container
    pdata = hdf5.DualProblemData()
    pdata.load(dataset=dataframe, dtype=dtype)

    # Construct the problem quantities from the HDF5 data
    M = pdata.M
    invM = pdata.invM
    J = pdata.J
    h = dtype(pdata.dt) * pdata.h
    u_minus = pdata.u_minus
    v_star = pdata.v_i + pdata.v_b
    D = pdata.D
    v_f = pdata.v_f - v_star

    # Pack quantities into the problem container
    return ConstrainedDynamicsProblem(
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
    msg.info("nbd: %d", nbd)
    msg.info("ncts: %d", ncts)
    msg.info("kdim: %d", kdim)

    # Assemble the KKT matrix
    K = np.zeros((kdim, kdim), dtype=problem.M.dtype)
    K[:nbd, :nbd] = problem.M
    K[:nbd, nbd:] = problem.J.T
    K[nbd:, :nbd] = problem.J
    msg.info("K: norm=%s, shape=%s, dtype=%s", np.linalg.norm(K), K.shape, K.dtype)

    # Assemble the KKT rhs
    k = np.zeros((kdim,), dtype=problem.M.dtype)
    k[:nbd] = problem.M @ problem.u_minus + problem.h
    k[nbd:] = -problem.v_star
    msg.info("k: norm=%s, shape=%s, dtype=%s", np.linalg.norm(k), k.shape, k.dtype)

    # Optionally ensure symmetry of the KKT matrix
    if ensure_symmetric:
        dtype = K.dtype
        K = dtype.type(0.5) * (K + K.T)

    # Optionally compute matrix properties
    if save_matrix_info:
        properties_K = linalg.SquareSymmetricMatrixProperties(K)
        msg.info("K properties: %s", properties_K)

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
    d = -problem.v_f
    msg.info("D: norm=%s, shape=%s, dtype=%s", np.linalg.norm(D), D.shape, D.dtype)
    msg.info("d: norm=%s, shape=%s, dtype=%s", np.linalg.norm(d), d.shape, d.dtype)

    # Optionally ensure symmetry of the dual system matrix
    if ensure_symmetric:
        dtype = D.dtype
        D = dtype.type(0.5) * (D + D.T)

    # Optionally compute matrix properties
    if save_matrix_info:
        properties_D = linalg.SquareSymmetricMatrixProperties(D)
        msg.info("D properties: %s", properties_D)

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
        crbd=ConstrainedDynamicsSolution(lambdas=admm.lambdas, u_plus=admm.u_plus, v_plus=admm.v_plus),
        kkt=LinearSystemSolution(x=np.concatenate((admm.u_plus, -admm.lambdas))),
        dual=LinearSystemSolution(x=admm.lambdas),
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

    # Set the basic solution metrics
    metrics.data.converged = status.converged
    metrics.data.iterations = status.iterations
    metrics.data.total_time = time
    metrics.data.iteration_time = time / status.iterations if status.iterations > 0 else np.inf
    metrics.data.primal_residual_inf = status.r_p
    metrics.data.dual_residual_inf = status.r_d
    metrics.data.compl_residual_inf = status.r_c

    # Compute the CRBD performance metrics
    metrics.data.primal_error_abs = primal_dynamics_error_inf(problem.crbd, solution.crbd)
    metrics.data.primal_error_rel = primal_dynamics_error_inf_rel(problem.crbd, solution.crbd)
    metrics.data.dual_error_abs = linsys_error_inf(problem.dual.A, problem.dual.b, solution.dual.x)
    metrics.data.dual_error_rel = linsys_error_rel_inf(problem.dual.A, problem.dual.b, solution.dual.x)
    metrics.data.kkt_error_abs = linsys_error_inf(problem.kkt.A, problem.kkt.b, solution.kkt.x)
    metrics.data.kkt_error_rel = linsys_error_rel_inf(problem.kkt.A, problem.kkt.b, solution.kkt.x)

    # Return the populated metrics container
    return metrics


def solve_benchmark_problem(problem: BenchmarkProblem, admm: linalg.ADMMSolver) -> list[BenchmarkMetrics]:
    """Solve a ConstrainedDynamicsProblem using an ADMM solver."""

    # Extract solver info
    solver_name = get_solver_typename(admm)
    linear_solver = get_solver_typename(admm.linsys_solver)

    # Solve as KKT system
    start_kkt = time.perf_counter()
    status_kkt = admm.solve_kkt(
        M=problem.crbd.M, J=problem.crbd.J, h=problem.crbd.h, u_minus=problem.crbd.u_minus, v_star=problem.crbd.v_star
    )
    time_kkt = time.perf_counter() - start_kkt
    msg.debug(f"ADMM.solve_kkt took {time_kkt:.6f} seconds")

    # Generate the benchmark solution and metrics for the KKT solve
    kkt_info = SolutionInfo(
        dtype=admm.dtype,
        solver_name=solver_name,
        method_name="KKT",
        linear_solver=linear_solver,
    )
    solution_kkt = make_benchmark_solution(admm, kkt_info)
    metrics_kkt = make_benchmark_metrics(time_kkt, status_kkt, problem, solution_kkt)

    # Solve as primal Schur-complement system
    start_schur_prim = time.perf_counter()
    status_schur_prim = admm.solve_schur_primal(
        M=problem.crbd.M, J=problem.crbd.J, h=problem.crbd.h, u_minus=problem.crbd.u_minus, v_star=problem.crbd.v_star
    )
    time_schur_prim = time.perf_counter() - start_schur_prim
    msg.debug(f"ADMM.solve_schur_prim took {time_schur_prim:.6f} seconds")

    # Generate the benchmark solution and metrics for the primal Schur-complement solve
    schur_prim_info = SolutionInfo(
        dtype=admm.dtype,
        solver_name=solver_name,
        method_name="Schur-Primal",
        linear_solver=linear_solver,
    )
    solution_schur_prim = make_benchmark_solution(admm, schur_prim_info)
    metrics_schur_prim = make_benchmark_metrics(time_schur_prim, status_schur_prim, problem, solution_schur_prim)

    # Solve as dual Schur-complement system w/o preconditioning
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
        linear_solver=linear_solver,
    )
    solution_schur_dual = make_benchmark_solution(admm, schur_dual_info)
    metrics_schur_dual = make_benchmark_metrics(time_schur_dual, status_schur_dual, problem, solution_schur_dual)

    # Solve as dual Schur-complement system with preconditioning
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
        linear_solver=linear_solver,
    )
    solution_schur_dual_prec = make_benchmark_solution(admm, schur_dual_prec_info)
    metrics_schur_dual_prec = make_benchmark_metrics(
        time_schur_dual_prec, status_schur_dual_prec, problem, solution_schur_dual_prec
    )

    # Return all metrics for this problem
    return [metrics_kkt, metrics_schur_prim, metrics_schur_dual, metrics_schur_dual_prec]


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


###
# Constants
###


PROBLEM_TYPE = "Primitive"
# PROBLEM_TYPE = "Robotics"
# PROBLEM_TYPE = "Animatronics"

PROBLEM_NAME = "boxes_hinged"
# PROBLEM_NAME = "fourbar_free"
# PROBLEM_NAME = "walker"

PROBLEM_CATEGORY = "IndependentJoints"
# PROBLEM_CATEGORY = "RedundantJoints"
# PROBLEM_CATEGORY = "DenseConstraints"


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
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation
    msg.set_log_level(msg.LogLevel.INFO)

    # Create output directories
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)

    # Construct and configure the data containers
    msg.info("Loading HDF5 data containers...")
    datafile = h5py.File(HDF5_DATASET_PATH, "r")

    # Select the numpy data type for computations
    np_dtype = np.float64
    # np_dtype = np.float32

    ###
    # Solver set-up
    ###

    # Create and configure the ADMM solver
    admm = linalg.ADMMSolver(
        dtype=np_dtype,
        # primal_tolerance=1e-6,
        # dual_tolerance=1e-6,
        # compl_tolerance=1e-6,
        # eta=1e-5,
        # rho=1.0,
        # omega=1.0,
        # maxiter=200,
        # linsys_atol=None,
        # linsys_rtol=None,
        # linsys_ftol=None,
    )

    # Configure the linear system solver
    # admm.linsys_solver = linalg.NumPySolver()

    ###
    # Single-problem demo
    ###

    # Retrieve target data frames
    SAMPLE = 0
    fpath = f"{PROBLEM_TYPE}/{PROBLEM_NAME}/{PROBLEM_CATEGORY}/{SAMPLE}/DualProblem"
    dataframe = datafile[fpath]

    # Load the problem data into a container
    msg.info(f"Loading problem data from '{dataframe.name}'...")
    problem = make_benchmark_problem(
        name=fpath,
        problem=load_dualproblem_data(dataframe=dataframe, dtype=np_dtype),
        ensure_symmetric=False,
        save_matrix_info=False,
        save_symmetry_info=False,
    )

    # Solve the benchmark problem using the ADMM solver
    metrics = solve_benchmark_problem(problem, admm)
    for m in metrics:
        print(f"\n{m}\n")

    ###
    # Multiple-problems demo
    ###

    # Find and print all DualProblem paths
    search_scope = f"{PROBLEM_TYPE}/{PROBLEM_NAME}/{PROBLEM_CATEGORY}"
    msg.info(f"Searching for DualProblem paths in scope '{search_scope}'...")
    dualproblem_paths = find_dualproblem_paths(datafile=datafile, scope=search_scope)
    msg.info(f"Found {len(dualproblem_paths)} DualProblem path(s).")
    for path in dualproblem_paths:
        print(f"- {path}")

    # Iterate over all found DualProblem paths
    metrics = []
    msg.info("Iterating over all found DualProblem paths...")
    for path in dualproblem_paths:
        problem = make_benchmark_problem(
            name=path,
            problem=load_dualproblem_data(dataframe=dataframe, dtype=np_dtype),
            ensure_symmetric=False,
            save_matrix_info=False,
            save_symmetry_info=False,
        )
        metrics.extend(solve_benchmark_problem(problem, admm))

    # TODO
    for m in metrics:
        print(f"\n{m}\n")

    # TODO:
    #   -  Iterate over metrics and create table/matrix for each metric
    #   -  Create a list of unique solver IDs
    #   -  Add collection of problem properties
    #   -  Compute and print performance profiles for each metric
    #   -  Create metric-vs-problem_size plots for each metric

    # Close the HDF5 data file
    datafile.close()
    msg.info("Done.")
