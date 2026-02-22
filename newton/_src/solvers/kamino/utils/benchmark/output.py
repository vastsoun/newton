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

###
# Functions
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
