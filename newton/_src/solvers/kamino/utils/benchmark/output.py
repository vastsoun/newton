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

import os
from collections.abc import Callable

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ...linalg.linear import LinearSolverTypeToName
from ...solver_kamino import SolverKaminoSettings

###
# Metrics Data
###


def render_subcolumn_metrics_table_rich(
    title: str,
    row_header: str,
    col_titles: list[str],
    row_titles: list[str],
    subcol_titles: list[str],
    subcol_data: list[np.ndarray],
    subcol_formats: list[str | Callable] | None = None,
    max_width: int | None = None,
    path: str | None = None,
    to_console: bool = False,
):
    n_metrics = len(subcol_data)
    n_problems = len(col_titles)
    n_solvers = len(row_titles)

    if len(subcol_titles) != n_metrics:
        raise ValueError("subcol_titles length must match number of metric arrays")

    for i, arr in enumerate(subcol_data):
        if arr.shape != (n_problems, n_solvers):
            raise ValueError(f"subcol_data[{i}] has shape {arr.shape}, expected {(n_problems, n_solvers)}")

    if subcol_formats is None:
        subcol_formats = [None] * n_metrics
    if len(subcol_formats) != n_metrics:
        raise ValueError("subcol_formats length must match number of metrics")

    def format_value(x, fmt):
        if callable(fmt):
            return fmt(x)
        if isinstance(fmt, str):
            try:
                return format(x, fmt)
            except Exception:
                return str(x)
        if isinstance(x, (int, np.integer)):
            return f"{int(x)}"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.4g}"
        return str(x)

    table = Table(
        title=title,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        pad_edge=True,
    )

    # Solver column
    table.add_column(row_header, style="bold", no_wrap=True, justify="left")

    # Metric columns: problem shown only on first subcolumn in each block
    # Header is a Text object with justify="left" (works on rich 14.0.0)
    for p_name in col_titles:
        for m_idx, m_name in enumerate(subcol_titles):
            top = p_name if m_idx == 0 else ""

            header = Text(justify="left")
            if top:
                header.append(top, style="bold")
            header.append("\n")
            header.append(m_name, style="dim")

            table.add_column(
                header=header,
                justify="right",  # numeric cells
                no_wrap=True,
            )

    # Data rows
    for s_idx, solver in enumerate(row_titles):
        row = [solver]
        for p_idx in range(n_problems):
            for m_idx in range(n_metrics):
                value = subcol_data[m_idx][p_idx, s_idx]
                row.append(format_value(value, subcol_formats[m_idx]))
        table.add_row(*row)

    # Render the table to the console and/or save to file
    if path is not None:
        path_dir = os.path.dirname(path)
        if path_dir and not os.path.exists(path_dir):
            raise ValueError(
                f"Directory for path '{path}' does not exist. Please create the directory before exporting the table."
            )
        with open(path, "w", encoding="utf-8") as f:
            console = Console(file=f, width=max_width)
            console.print(table, crop=False)
    if to_console:
        console = Console(width=max_width)
        console.rule()
        console.print(table, crop=False)
        console.rule()


###
# Solver Configs
###


def render_solver_configs_table(
    configs: dict[str, SolverKaminoSettings],
    path: str | None = None,
    groups: list[str] | None = None,
    to_console: bool = False,
):
    """
    Renders a rich table summarizing the solver configurations.

    Args:
        configs (dict[str, SolverKaminoSettings]):
            A dictionary mapping configuration names to SolverKaminoSettings objects.
        path (str, optional):
            The file path to save the rendered table as a text file. If None, the table is not saved to a file.
        groups (list[str], optional):
            A list of groups to include in the table. If None, default groups are used.\n
            Supported groups include:
            - "cts": Constraint parameters (alpha, beta, gamma, delta, preconditioning)
            - "sparse": Sparse representation settings (sparse, sparse_jacobian)
            - "linear": Linear solver settings (type, kwargs)
            - "padmm": PADMM settings (max_iterations, primal_tol, dual_tol, etc)
            - "warmstart": Warmstarting settings (mode, contact_method)
        to_console (bool, optional):
            If True, also prints the table to the console.

    Raises:
        ValueError:
            If the configs dictionary is empty or if any of the configuration objects are missing required attributes.
        IOError:
            If there is an error writing the table to the specified file path.
    """

    # Define a helper function to add a group of columns with a shared header
    def add_column_group(
        table: Table,
        group_name: str,
        subcol_headers: list[str],
        justify: str = "left",
        color: str | None = None,
    ) -> None:
        for i, sub in enumerate(subcol_headers):
            header = Text(justify="left")
            if i == 0:
                header.append(group_name, style=f"bold {color}" if color else "bold")
            header.append("\n")
            header.append(sub, style=f"dim {color}" if color else "dim")
            col_justify = "center" if justify == "center" else justify
            table.add_column(header=header, justify=col_justify, no_wrap=True)

    # Initialize the table with appropriate columns and styling
    table = Table(
        title="Solver Configurations Summary",
        show_header=True,
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        pad_edge=True,
    )

    # If no groups are specified, default to showing linear solver and PADMM settings
    if groups is None:
        groups = ["linear", "padmm"]

    # Add the first column for configuration names
    add_column_group(table, "Solver Configuration", ["Name"], color="white", justify="left")

    # Add groups of columns based on the specified groups to include in the table
    if "cts" in groups:
        add_column_group(table, "Constraints", ["alpha", "beta", "gamma", "delta", "precond"], color="green")
    if "sparse" in groups:
        add_column_group(table, "Representation", ["sparse", "sparse_jacobian"], color="yellow")
    if "linear" in groups:
        add_column_group(table, "Linear Solver", ["type", "kwargs"], color="magenta")
    if "padmm" in groups:
        add_column_group(
            table,
            "PADMM",
            [
                "max_iterations",
                "primal_tol",
                "dual_tol",
                "compl_tol",
                "restart_tol",
                "eta",
                "rho_0",
                "rho_min",
                "penalty_update",
                "penalty_freq",
                "accel",
            ],
            color="cyan",
        )
    if "warmstart" in groups:
        add_column_group(table, "Warmstarting", ["mode", "contact_method"], color="blue")

    # Add rows for each configuration
    for name, cfg in configs.items():
        cfg_row = []
        if "cts" in groups:
            cfg_row.extend(
                [
                    f"{cfg.problem.alpha}",
                    f"{cfg.problem.beta}",
                    f"{cfg.problem.gamma}",
                    f"{cfg.problem.delta}",
                    str(cfg.problem.preconditioning),
                ]
            )
        if "sparse" in groups:
            cfg_row.extend([str(cfg.sparse), str(cfg.sparse_jacobian)])
        if "linear" in groups:
            cfg_row.extend([str(LinearSolverTypeToName[cfg.linear_solver_type]), str(cfg.linear_solver_kwargs)])
        if "padmm" in groups:
            cfg_row.extend(
                [
                    str(cfg.padmm.max_iterations),
                    f"{cfg.padmm.primal_tolerance:.0e}",
                    f"{cfg.padmm.dual_tolerance:.0e}",
                    f"{cfg.padmm.compl_tolerance:.0e}",
                    f"{cfg.padmm.restart_tolerance:.0e}",
                    f"{cfg.padmm.eta:.0e}",
                    f"{cfg.padmm.rho_0}",
                    f"{cfg.padmm.rho_min}",
                    str(cfg.padmm.penalty_update_method.name),
                    str(cfg.padmm.penalty_update_freq),
                    str(cfg.use_solver_acceleration),
                ]
            )
        if "warmstart" in groups:
            cfg_row.extend([str(cfg.warmstart_mode.name), str(cfg.contact_warmstart_method.name)])
        table.add_row(name, *cfg_row)

    # Render the table to the console and/or save to file
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            console = Console(file=f, width=500)
            console.print(table, crop=False)
    if to_console:
        console = Console(width=None)
        console.rule()
        console.print(table, crop=False)
        console.rule()
