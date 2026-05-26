# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a specialized logger that records :class:`PhysicsMetrics` per-world
summary history on the same device as the wrapped metrics container.

The :class:`PhysicsMetricsLogger` class allocates per-frame log buffers for
every per-world summary field exposed by
``PhysicsMetrics.per_world_contacts_summary`` (six float32 residuals and six
companion int32 argmax indices). It supports an optional fixed-size rolling
window or bounded early-exit overflow policy, a configurable sample-decimation
rate, and an optional pinned simulation time step ``dt``. The per-frame
counters and rollover / bounding logic live on the target device, so a single
:meth:`PhysicsMetricsLogger.log` call expands to a fixed sequence of Warp
kernel launches that can be safely captured into a CUDA graph alongside the
solver's ``step`` and the metrics evaluation pipeline.

It also exposes utilities to extract the recorded data as numpy arrays in
chronological order, and to render per-metric matplotlib plots that follow the
equation-subtitled format used by
:class:`newton._src.solvers.kamino._src.metrics.SolutionMetricsLogger`.

Usage
-----

A typical example for using this module is::

    from newton._src.solvers.kamino.benchmark import (
        PhysicsMetrics,
        PhysicsMetricsLogger,
        compute_contact_constraint_metrics,
        compute_contact_velocities,
        compute_per_world_contact_constraint_summary,
    )

    metrics = PhysicsMetrics(model=model)
    logger = PhysicsMetricsLogger(
        metrics=metrics,
        max_frames=1000,
        mode=PhysicsMetricsLogger.Mode.ROLLING,
        decimation=2,
        dt=dt,
    )

    for _ in range(num_steps):
        # ... advance the simulation ...
        compute_contact_velocities(model, state, contacts)
        compute_contact_constraint_metrics(model, state, contacts, metrics)
        compute_per_world_contact_constraint_summary(model, contacts, metrics)
        logger.log()

    np_data = logger.to_numpy()
    logger.plot(path="/tmp/metrics", ext="pdf")
    logger.table(path="/tmp/metrics", to_console=True)
"""

from __future__ import annotations

import csv
import os
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .._src.utils import logger as msg
from .metrics import PhysicsMetrics

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

###
# Module interface
###

__all__ = ["PhysicsMetricsLogger"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

# Names of the scalar (float32) per-world summary fields recorded by the logger.
# Order matches `PhysicsMetrics.per_world_contacts_summary` and the entries are
# used both for buffer allocation and for plotting.
_SCALAR_FIELDS_FLOAT32: tuple[str, ...] = (
    "r_cts_penetration",
    "r_cts_velocity",
    "r_ncp_primal",
    "r_ncp_dual",
    "r_ncp_compl",
    "r_vi_natmap",
)

# Companion argmax fields (one per scalar) storing the contact index that
# attained the per-world maximum for that residual.
_ARGMAX_FIELDS_INT32: tuple[str, ...] = tuple(f + "_argmax" for f in _SCALAR_FIELDS_FLOAT32)

# Human-readable plot titles per scalar metric (without the equation suffix).
_METRIC_TITLES: dict[str, str] = {
    "r_cts_penetration": "Contact Penetration Residual",
    "r_cts_velocity": "Contact Normal-Velocity Residual",
    "r_ncp_primal": "NCP Primal Residual",
    "r_ncp_dual": "NCP Dual Residual",
    "r_ncp_compl": "NCP Complementary Residual",
    "r_vi_natmap": "VI Natural-Map Residual",
}

# LaTeX equations rendered as the second line of each metric's plot title.
# Aligned with `SolutionMetricsLogger._METRIC_EQUATIONS` for the four shared
# NCP/VI metrics so cross-tool plots are visually consistent.
_METRIC_EQUATIONS: dict[str, str] = {
    "r_cts_penetration": r"$\max_k \, | \, d_k \, |$",
    "r_cts_velocity": r"$\max_k \, \max(0, -v_k^T \, n_k)$",
    "r_ncp_primal": r"$\Vert \, \lambda - P_K(\lambda) \, \Vert_\infty $",
    "r_ncp_dual": r"$\Vert \, v_a^+ - P_{K^*}(v_a^+) \, \Vert_\infty $",
    "r_ncp_compl": r"$\Vert \, \lambda^T \, v_a^+ \, \Vert_\infty $",
    "r_vi_natmap": r"$\Vert \, \lambda - P_{K}(\lambda - v_a^+(\lambda)) \, \Vert_\infty $",
}

# Color palette for cross-setup overlay plots, cycled if more than 8 setups.
_OVERLAY_COLORS: tuple[str, ...] = (
    "purple",
    "blue",
    "red",
    "green",
    "yellow",
    "cyan",
    "orange",
    "brown",
)

# Line-style palette cycled alongside ``_OVERLAY_COLORS`` so overlapping curves
# (identical residuals across solvers, common in the contact-penetration plots
# where the metric depends only on the shared pre-step state and contact
# geometry) remain visually distinguishable. Without this, the last-drawn
# solver's solid line completely obscures earlier solvers' identical curves.
_OVERLAY_LINESTYLES: tuple[str, ...] = (
    "-",
    "--",
    ":",
    "-.",
)

# Marker palette cycled alongside ``_OVERLAY_COLORS`` so identical curves
# remain distinguishable at zoom levels where the line styles are too dense
# to be resolved.
_OVERLAY_MARKERS: tuple[str, ...] = (
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "*",
)


###
# Statistics helpers
###


def _format_stat_cell(value: float) -> str:
    """Formats a single statistic value for table / CSV output.

    Uses ``.6g`` formatting which naturally renders ``NaN`` / ``inf`` as
    ``"nan"`` / ``"inf"``, and gives readable precision for both small and
    large residual magnitudes.
    """
    return format(float(value), ".6g")


# Statistic labels (in the column order they appear in the rendered tables).
_STAT_LABELS: tuple[str, ...] = ("max", "mean")


def _compute_summary_stats(
    np_data: dict[str, np.ndarray],
    per_world: bool,
) -> dict[str, tuple[float, float] | np.ndarray]:
    """Computes max/mean over the recorded scalar-residual history.

    Args:
        np_data: The dictionary returned by
            :meth:`PhysicsMetricsLogger.to_numpy`, mapping per-world summary
            field names to arrays of shape ``(num_frames, num_worlds)``.
            Only the scalar fields in :data:`_SCALAR_FIELDS_FLOAT32` are
            consumed; argmax companions are ignored.
        per_world: If ``True``, reduce only along the frame axis so the result
            is per-world. If ``False``, also flatten across worlds so the
            result is a single ``(max, mean)`` pair per field.

    Returns:
        For ``per_world=False``: a dictionary mapping each scalar field name
        to a ``(max, mean)`` pair of Python floats. The values are ``NaN``
        when the recorded buffer is empty.

        For ``per_world=True``: a dictionary mapping each scalar field name to
        a ``float64`` array of shape ``(num_worlds, 2)`` whose last axis
        carries ``(max, mean)`` per world. The array is filled with ``NaN``
        when the recorded buffer is empty.
    """
    stats: dict[str, tuple[float, float] | np.ndarray] = {}
    n_stats = len(_STAT_LABELS)
    for field in _SCALAR_FIELDS_FLOAT32:
        arr = np_data.get(field)
        if arr is None or arr.size == 0:
            if per_world:
                # ``to_numpy()`` preserves the 2-D shape even when empty so the
                # second axis tells us the number of worlds. Fall back to 0
                # worlds otherwise (defensive only).
                num_worlds = int(arr.shape[1]) if arr is not None and arr.ndim == 2 else 0
                stats[field] = np.full((num_worlds, n_stats), np.nan, dtype=np.float64)
            else:
                stats[field] = tuple(float("nan") for _ in range(n_stats))  # type: ignore[assignment]
            continue
        if per_world:
            stats[field] = np.stack([arr.max(axis=0), arr.mean(axis=0)], axis=-1).astype(np.float64)
        else:
            flat = arr.reshape(-1)
            stats[field] = (float(flat.max()), float(flat.mean()))
    return stats


def _build_stats_rows(
    stats_by_logger: list[tuple[str | None, dict[str, tuple[float, float] | np.ndarray]]],
    per_world: bool,
    num_worlds: int,
) -> list[list[str]]:
    """Assembles the formatted data rows shared by the rich and CSV outputs.

    Args:
        stats_by_logger: A list of ``(name, stats)`` pairs as returned by
            :func:`_compute_summary_stats`. ``table()`` passes a singleton
            list; ``table_comparison()`` passes one pair per logger.
        per_world: If ``True``, emit ``6 * num_worlds`` rows ordered by world
            then metric; otherwise emit one row per scalar field.
        num_worlds: Number of worlds carried by each logger. Used only when
            ``per_world`` is ``True`` to determine the row count.

    Returns:
        A list of rows. Every row is a list of pre-formatted string cells in
        the order they appear in the rendered table (leading ``world`` /
        ``metric`` cells followed by the per-logger stat pairs).
    """
    rows: list[list[str]] = []
    n_outer = num_worlds if per_world else 1
    n_stats = len(_STAT_LABELS)
    for w in range(n_outer):
        for field in _SCALAR_FIELDS_FLOAT32:
            row: list[str] = []
            if per_world:
                row.append(f"world_{w}")
            row.append(field)
            for _name, stats in stats_by_logger:
                if per_world:
                    arr = stats[field]
                    for s in range(n_stats):
                        row.append(_format_stat_cell(arr[w, s]))
                else:
                    pair = stats[field]
                    for s in range(n_stats):
                        row.append(_format_stat_cell(pair[s]))
            rows.append(row)
    return rows


def _gradient_color(t: float) -> str:
    """Maps a normalized rank ``t`` in ``[0, 1]`` to a green->yellow->red hex color.

    ``t = 0`` returns light green (best), ``t = 0.5`` returns light yellow,
    ``t = 1`` returns light red (worst). Values outside ``[0, 1]`` are
    clamped. Colors are intentionally pastel so the colored text remains
    legible against both light and dark terminal backgrounds.
    """
    t = max(0.0, min(1.0, float(t)))
    if t <= 0.5:
        s = t * 2.0
        r = round(144 + (255 - 144) * s)
        g = round(238 + (255 - 238) * s)
        b = round(144 + (153 - 144) * s)
    else:
        s = (t - 0.5) * 2.0
        r = 255
        g = round(255 - (255 - 153) * s)
        b = 153
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _rank_values_to_colors(values: list[float]) -> list[str | None]:
    """Maps a list of per-logger values to gradient colors by rank.

    Lower values are "better" (green) and higher values are "worse" (red);
    intermediate values are linearly interpolated through yellow. Non-finite
    (``NaN`` / ``inf``) entries map to ``None`` so they get no styling.

    Args:
        values: One value per logger, in column order.

    Returns:
        A list of the same length as ``values``. Each entry is a hex color
        string ``"#RRGGBB"`` for finite entries, or ``None`` for non-finite
        entries. If fewer than two distinct finite values are present the
        ranking is ill-defined and every finite cell receives the neutral
        mid-gradient color so the user still sees a visual marker.
    """
    finite_vals = [v for v in values if np.isfinite(v)]
    if len(finite_vals) < 2:
        return [None] * len(values)
    vmin = min(finite_vals)
    vmax = max(finite_vals)
    if vmax == vmin:
        neutral = _gradient_color(0.5)
        return [neutral if np.isfinite(v) else None for v in values]
    span = vmax - vmin
    return [_gradient_color((v - vmin) / span) if np.isfinite(v) else None for v in values]


def _compute_ranking_colors(
    stats_by_logger: list[tuple[str | None, dict[str, tuple[float, float] | np.ndarray]]],
    per_world: bool,
    num_worlds: int,
) -> list[list[str | None]]:
    """Returns a per-cell color grid for the comparison table.

    The returned grid has the same shape as the rows produced by
    :func:`_build_stats_rows`: leading ``World`` / ``Metric`` cells get
    ``None``, and each per-logger stat cell is colored by ranking that
    logger's value against the other loggers' values for the same row and
    stat. Lower values (better residuals) are pulled toward green; higher
    values (worse residuals) are pulled toward red.
    """
    n_loggers = len(stats_by_logger)
    n_stats = len(_STAT_LABELS)
    leading = 2 if per_world else 1
    n_outer = num_worlds if per_world else 1
    grid: list[list[str | None]] = []
    for w in range(n_outer):
        for field in _SCALAR_FIELDS_FLOAT32:
            cells: list[str | None] = [None] * leading
            logger_cells: list[str | None] = [None] * (n_loggers * n_stats)
            for s in range(n_stats):
                values: list[float] = []
                for _name, stats in stats_by_logger:
                    if per_world:
                        values.append(float(stats[field][w, s]))
                    else:
                        values.append(float(stats[field][s]))
                colors = _rank_values_to_colors(values)
                for li, c in enumerate(colors):
                    logger_cells[li * n_stats + s] = c
            cells.extend(logger_cells)
            grid.append(cells)
    return grid


def _lazy_import_rich():
    """Imports the ``rich`` symbols needed for table rendering.

    Mirrors the pattern used in
    :mod:`newton._src.solvers.kamino._src.utils.benchmark.render` so that
    ``rich`` remains an opt-in dependency: callers that never request
    console output never trigger this import.

    Raises:
        ImportError: If the ``rich`` package is not installed.
    """
    try:
        from rich import box  # noqa: PLC0415
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The `rich` package is required for console table output. Install it with: pip install rich"
        ) from e
    return box, Console, Table, Text


###
# Kernels
###


@wp.kernel
def _update_log_decision(
    max_frames: wp.int32,
    decimation: wp.int32,
    mode: wp.int32,
    call_count: wp.array[wp.int32],
    frames_total: wp.array[wp.int32],
    decision: wp.array[wp.int32],
):
    """Compute whether the current call should write a frame, and where.

    Args:
        max_frames: The maximum number of frames in the log buffers.
        decimation: The sample-decimation rate.
        mode: ``0`` for rolling, ``1`` for bounded.
        call_count: Single-element array tracking total :meth:`log` invocations.
        frames_total: Single-element array tracking total successful writes.
        decision: Two-element output, ``[should_write, write_idx]``.
    """
    cc = call_count[0]
    ft = frames_total[0]

    should_write = wp.int32(1)
    if (cc % decimation) != wp.int32(0):
        should_write = wp.int32(0)
    if mode == wp.int32(1) and ft >= max_frames:
        should_write = wp.int32(0)

    decision[0] = should_write
    if should_write == wp.int32(1):
        decision[1] = ft % max_frames
    else:
        decision[1] = wp.int32(0)

    call_count[0] = cc + wp.int32(1)


@wp.kernel
def _finalize_log_decision(
    decision: wp.array[wp.int32],
    frames_total: wp.array[wp.int32],
):
    """Increment ``frames_total`` if the current call actually wrote a frame."""
    if decision[0] == wp.int32(1):
        frames_total[0] = frames_total[0] + wp.int32(1)


@wp.kernel
def _write_log_row_float32(
    src: wp.array[wp.float32],
    decision: wp.array[wp.int32],
    dest: wp.array2d[wp.float32],
):
    """Copies one ``(num_worlds,)`` source row into ``dest[decision[1], :]``.

    Short-circuits when ``decision[0] == 0`` so the launch is a no-op on
    decimation-skipped or bounded-mode-overflowed calls.
    """
    wid = wp.tid()
    if decision[0] == wp.int32(0):
        return
    dest[decision[1], wid] = src[wid]


@wp.kernel
def _write_log_row_int32(
    src: wp.array[wp.int32],
    decision: wp.array[wp.int32],
    dest: wp.array2d[wp.int32],
):
    """Copies one ``(num_worlds,)`` source row into ``dest[decision[1], :]``."""
    wid = wp.tid()
    if decision[0] == wp.int32(0):
        return
    dest[decision[1], wid] = src[wid]


###
# Interfaces
###


class PhysicsMetricsLogger:
    """
    Records :class:`PhysicsMetrics` per-world summary history on the metrics' device.

    The logger reads the per-world maximum and argmax buffers exposed by
    ``metrics.per_world_contacts_summary`` (populated by
    :func:`compute_per_world_contact_constraint_summary`) and appends one row
    per :meth:`log` call into a fixed-shape ``(max_frames, num_worlds)``
    storage. Both the per-world max (``float32``) and its companion argmax
    contact index (``int32``) are retained for every residual.

    The buffer-overflow policy is controlled by :class:`Mode`:

    - :attr:`Mode.ROLLING` wraps the write index modulo ``max_frames``, so the
      buffer always holds the most recent ``max_frames`` samples.
    - :attr:`Mode.BOUNDED` stops recording once ``max_frames`` samples have been
      logged; subsequent :meth:`log` calls are no-ops.

    The optional ``decimation`` argument skips intermediate calls so only every
    ``decimation``-th call actually writes a new frame; this is useful when
    :meth:`log` is invoked once per simulation step but a coarser sampling is
    sufficient for analysis.

    Every host-side decision in :meth:`log` (decimation gate, overflow check,
    write-index computation, counter increments) is performed on-device through
    dedicated Warp kernels. This makes a single :meth:`log` invocation a fixed
    sequence of kernel launches whose data dependencies live entirely in
    device memory, and it can be safely included inside :class:`wp.ScopedCapture`
    blocks alongside the per-step solver and metrics kernels.

    Numpy extraction via :meth:`to_numpy` always returns the recorded samples
    in chronological order (oldest first), and :meth:`plot` renders one
    matplotlib figure per scalar metric. Argmax fields are recorded but not
    plotted, mirroring the convention used by
    :class:`SolutionMetricsLogger`.
    """

    class Mode(IntEnum):
        """Buffer overflow behavior for :class:`PhysicsMetricsLogger`."""

        ROLLING = 0
        """Wrap around at ``max_frames``; oldest frames are overwritten."""

        BOUNDED = 1
        """Stop logging once ``max_frames`` samples have been recorded."""

    plt = None
    """Class-level cache for the optional :mod:`matplotlib.pyplot` import."""

    @classmethod
    def _initialize_plt(cls):
        """Imports :mod:`matplotlib.pyplot` lazily and caches it on the class."""
        if cls.plt is None:
            try:
                import matplotlib.pyplot as plt

                cls.plt = plt
            except ImportError:
                return

    def __init__(
        self,
        metrics: PhysicsMetrics,
        max_frames: int,
        mode: Mode = Mode.BOUNDED,
        decimation: int = 1,
        dt: float | None = None,
    ):
        """
        Initializes the physics-metrics logger.

        Args:
            metrics: The :class:`PhysicsMetrics` container to record from. Must
                have been constructed with a non-``None`` ``model`` so that
                ``metrics.per_world_contacts_summary`` is allocated.
            max_frames: The maximum number of frames recorded by the logger.
                Must be a strictly positive integer.
            mode: The buffer-overflow policy. Defaults to :attr:`Mode.BOUNDED`.
            decimation: Sample decimation rate. Only every ``decimation``-th
                :meth:`log` call writes a new frame. Defaults to ``1`` (no
                decimation). Must be a strictly positive integer.
            dt: Optional simulation time step used to scale the time axis on
                plots. If supplied it is pinned for the lifetime of the
                logger. If ``None``, plots fall back to a unit-less
                "Simulation Step" labelling.

        Raises:
            TypeError: If ``metrics`` is not a :class:`PhysicsMetrics`, or if
                ``mode`` is not a :class:`Mode` value.
            ValueError: If ``max_frames`` or ``decimation`` is not a strictly
                positive integer, or if ``metrics.per_world_contacts_summary``
                is unallocated, or if ``dt`` is not a positive number.
        """
        if not isinstance(metrics, PhysicsMetrics):
            raise TypeError(f"Expected 'metrics' to be of type `PhysicsMetrics`, got {type(metrics)}.")
        if metrics.per_world_contacts_summary is None:
            raise ValueError(
                "PhysicsMetricsLogger requires `metrics.per_world_contacts_summary` to be allocated. "
                "Construct `PhysicsMetrics` with a model whose `rigid_contact_max > 0` and "
                "`world_count > 0`."
            )
        if not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError(f"Expected 'max_frames' to be a positive integer, got {max_frames!r}.")
        if not isinstance(decimation, int) or decimation <= 0:
            raise ValueError(f"Expected 'decimation' to be a positive integer, got {decimation!r}.")
        if not isinstance(mode, PhysicsMetricsLogger.Mode):
            raise TypeError("Expected 'mode' to be a `PhysicsMetricsLogger.Mode` value.")
        if dt is not None:
            if not isinstance(dt, (int, float)) or float(dt) <= 0.0:
                raise ValueError(f"Expected 'dt' to be a positive number, got {dt!r}.")

        # Attempt to initialize matplotlib for plotting
        self._initialize_plt()

        # Store the metrics instance and related configurations
        self._metrics: PhysicsMetrics = metrics
        self._max_frames: int = int(max_frames)
        self._mode: PhysicsMetricsLogger.Mode = mode
        self._decimation: int = int(decimation)
        self._dt: float | None = float(dt) if dt is not None else None

        # Resolve the target device and per-world fan-out from the per-world
        # summary container; this is the authoritative source for both since the
        # logger directly reads (and replicates the shape of) those arrays.
        summary = metrics.per_world_contacts_summary
        self._device: wp.DeviceLike = summary.r_cts_penetration.device
        self._num_worlds: int = int(summary.r_cts_penetration.shape[0])

        # Allocate every per-frame log buffer on the metrics' device. The 2-D
        # layout ``(max_frames, num_worlds)`` matches the per-world scalar
        # fan-out of the underlying metrics fields. The internal counters and
        # per-call decision scratch buffer also live on the device so that
        # :meth:`log` expands to a fixed sequence of kernel launches that can
        # be safely captured into a CUDA graph alongside the solver step.
        with wp.ScopedDevice(self._device):
            shape = (self._max_frames, self._num_worlds)
            for field in _SCALAR_FIELDS_FLOAT32:
                setattr(self, f"log_{field}", wp.zeros(shape=shape, dtype=wp.float32))
            for field in _ARGMAX_FIELDS_INT32:
                setattr(self, f"log_{field}", wp.full(shape=shape, value=-1, dtype=wp.int32))

            # Device-side counters and per-call decision scratch buffer.
            # ``_call_count`` tracks every :meth:`log` invocation (used by the
            # decimation gate); ``_frames_total`` tracks the number of writes
            # that actually landed in the buffer (used by the overflow /
            # chronological-ordering logic); ``_decision`` carries the
            # per-call ``[should_write, write_idx]`` pair from
            # :func:`_update_log_decision` to the copy kernels.
            self._call_count: wp.array = wp.zeros(shape=1, dtype=wp.int32)
            self._frames_total: wp.array = wp.zeros(shape=1, dtype=wp.int32)
            self._decision: wp.array = wp.zeros(shape=2, dtype=wp.int32)

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """Returns the device where the log buffers are allocated."""
        return self._device

    @property
    def num_worlds(self) -> int:
        """Returns the number of worlds recorded per frame."""
        return self._num_worlds

    @property
    def max_frames(self) -> int:
        """Returns the maximum number of frames the buffer can hold."""
        return self._max_frames

    @property
    def mode(self) -> Mode:
        """Returns the buffer-overflow policy."""
        return self._mode

    @property
    def decimation(self) -> int:
        """Returns the sample decimation rate."""
        return self._decimation

    @property
    def dt(self) -> float | None:
        """Returns the pinned simulation time step (in seconds), if any."""
        return self._dt

    @property
    def num_total_writes(self) -> int:
        """Returns the cumulative number of writes (including overwritten frames in rolling mode)."""
        return int(self._frames_total.numpy()[0])

    @property
    def num_logged_frames(self) -> int:
        """Returns the number of valid frames currently stored in the buffer."""
        return min(self.num_total_writes, self._max_frames)

    @property
    def num_calls(self) -> int:
        """Returns the cumulative number of :meth:`log` invocations."""
        return int(self._call_count.numpy()[0])

    @property
    def is_full(self) -> bool:
        """Returns whether the buffer has reached ``max_frames`` writes."""
        return self.num_total_writes >= self._max_frames

    ###
    # Operations
    ###

    def reset(self):
        """Resets the logger counters and clears every log buffer."""
        self._call_count.zero_()
        self._frames_total.zero_()
        self._decision.zero_()
        for field in _SCALAR_FIELDS_FLOAT32:
            getattr(self, f"log_{field}").zero_()
        for field in _ARGMAX_FIELDS_INT32:
            getattr(self, f"log_{field}").fill_(-1)

    def log(self):
        """Records the current per-world summary values into the next buffer slot.

        Every invocation expands to the same fixed sequence of Warp kernel
        launches, so the call can be safely captured into a CUDA graph along
        with the solver and metrics evaluation kernels. Decimation skips and
        bounded-mode overflow are enforced on-device by
        :func:`_update_log_decision`, with the per-field copy kernels
        short-circuiting on the resulting ``[should_write, write_idx]`` buffer.
        """
        wp.launch(
            kernel=_update_log_decision,
            dim=1,
            inputs=[
                wp.int32(self._max_frames),
                wp.int32(self._decimation),
                wp.int32(int(self._mode)),
                self._call_count,
                self._frames_total,
                self._decision,
            ],
            device=self._device,
        )

        summary = self._metrics.per_world_contacts_summary
        for field in _SCALAR_FIELDS_FLOAT32:
            wp.launch(
                kernel=_write_log_row_float32,
                dim=self._num_worlds,
                inputs=[getattr(summary, field), self._decision, getattr(self, f"log_{field}")],
                device=self._device,
            )
        for field in _ARGMAX_FIELDS_INT32:
            wp.launch(
                kernel=_write_log_row_int32,
                dim=self._num_worlds,
                inputs=[getattr(summary, field), self._decision, getattr(self, f"log_{field}")],
                device=self._device,
            )

        wp.launch(
            kernel=_finalize_log_decision,
            dim=1,
            inputs=[self._decision, self._frames_total],
            device=self._device,
        )

    ###
    # Numpy extraction
    ###

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Returns the recorded data as numpy arrays in chronological order.

        Each entry of the returned dictionary has shape
        ``(num_logged_frames, num_worlds)`` and is keyed by the corresponding
        per-world summary field name. Both the scalar metric fields and their
        argmax companions are returned.

        In :attr:`Mode.ROLLING` the rows are rotated so that the oldest
        recorded frame is at index ``0`` and the most recent at index
        ``num_logged_frames - 1``.

        Returns:
            A dictionary mapping field name to its recorded values.
        """
        total = self.num_total_writes
        n = min(total, self._max_frames)
        result: dict[str, np.ndarray] = {}
        for field in (*_SCALAR_FIELDS_FLOAT32, *_ARGMAX_FIELDS_INT32):
            buf = getattr(self, f"log_{field}").numpy()
            if n == 0:
                result[field] = buf[:0].copy()
                continue
            if self._mode == PhysicsMetricsLogger.Mode.ROLLING and total > self._max_frames:
                write_idx = total % self._max_frames
                result[field] = np.concatenate([buf[write_idx:], buf[:write_idx]], axis=0)
            else:
                result[field] = buf[:n].copy()
        return result

    def time_axis(self) -> np.ndarray:
        """Returns the per-frame time axis used by the plots.

        When :attr:`dt` is available the returned array is in seconds and
        accounts for the configured ``decimation``; otherwise it falls back
        to a unit-less simulation-step axis (also scaled by ``decimation``).
        """
        n = self.num_logged_frames
        scale = (self._dt if self._dt is not None else 1.0) * float(self._decimation)
        return np.arange(n, dtype=np.float32) * scale

    ###
    # Plotting
    ###

    def plot(
        self,
        filename: str | None = None,
        path: str | None = None,
        show: bool = False,
        ext: str = "pdf",
        log_scale: bool = False,
        log_floor: float = 1e-12,
    ):
        """
        Renders one matplotlib figure per scalar per-world summary metric.

        Each figure follows the equation-subtitled format used by
        :class:`SolutionMetricsLogger.plot`: the title is the human-readable
        metric name with the underlying mathematical definition rendered as a
        LaTeX subtitle. One curve is drawn per world. The argmax companion
        fields are *not* plotted (they are retained in :meth:`to_numpy` for
        downstream diagnostic use).

        Args:
            filename: Optional filename prefix. The final file name is
                ``{filename}_{metric_name}.{ext}`` (or ``{metric_name}.{ext}``
                if no prefix is provided).
            path: If provided, each figure is saved as
                ``{path}/{filename}{_metric_name}.{ext}``. The directory must
                already exist.
            show: If ``True`` the figures are also displayed (blocking).
            ext: The file extension / matplotlib format to save with.
                Defaults to ``"pdf"`` to match the benchmarks output.
            log_scale: If ``True``, the y-axis of each figure is rendered
                with a base-10 logarithmic scale. Residual samples are
                clamped to ``log_floor`` before plotting so that values
                that are exactly zero do not break the log scale or
                trigger matplotlib's ``nonpositive`` warnings.
            log_floor: Strictly positive floor used to clamp samples when
                ``log_scale`` is enabled. Unused otherwise. Defaults to
                ``1e-12``.

        Raises:
            ValueError: If the output directory does not exist, or if
                ``log_scale`` is ``True`` and ``log_floor`` is not a
                strictly positive number.
        """
        if self.plt is None:
            msg.warning("matplotlib is not available, skipping plotting.")
            return
        if self.num_logged_frames == 0:
            msg.warning("No logged frames to plot, skipping plotting.")
            return
        if path is not None and not os.path.isdir(path):
            raise ValueError(f"Plot output directory '{path}' does not exist. Please create it before calling plot().")
        if log_scale and (not isinstance(log_floor, (int, float)) or float(log_floor) <= 0.0):
            raise ValueError(
                f"Expected 'log_floor' to be a strictly positive number when 'log_scale' is True, got {log_floor!r}."
            )
        if filename is None:
            filename = ""
            separator = ""
        else:
            separator = "_"

        floor = float(log_floor)
        time = self.time_axis()
        np_data = self.to_numpy()
        x_label = "Time (s)" if self._dt is not None else "Step"
        for field in _SCALAR_FIELDS_FLOAT32:
            equation = _METRIC_EQUATIONS[field]
            base_title = _METRIC_TITLES[field]
            title = f"{base_title} \n ({equation})"
            fig, ax = self.plt.subplots(1, 1, figsize=(10, 6))
            data = np_data[field]
            for w in range(self._num_worlds):
                y = data[:, w]
                if log_scale:
                    y = np.maximum(y, floor)
                ax.plot(
                    time,
                    y,
                    label=f"world_{w}",
                    marker="o",
                    markersize=4,
                )
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(field)
            if log_scale:
                ax.set_yscale("log")
            ax.grid()
            if self._num_worlds > 1:
                ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            if path is not None:
                fig_path = os.path.join(path, f"{filename}{separator}{field}.{ext}")
                fig.savefig(fig_path, format=ext, dpi=300, bbox_inches="tight")
            if show:
                self.plt.show()
            self.plt.close(fig)

    @classmethod
    def plot_comparison(
        cls,
        loggers: dict[str, PhysicsMetricsLogger],
        filename: str | None = None,
        path: str | None = None,
        show: bool = False,
        grid: bool = False,
        ext: str = "pdf",
        log_scale: bool = False,
        log_floor: float = 1e-12,
    ):
        """
        Renders overlaid :class:`PhysicsMetricsLogger` plots across multiple logger instances.

        Iterates the scalar metric fields recorded by every setup's logger and
        plots them on a shared axis, drawing one curve per world per setup using
        :data:`_OVERLAY_COLORS` cycled by setup index. The figure title and
        LaTeX subtitle follow :meth:`PhysicsMetricsLogger.plot`, so the output
        is visually consistent with the per-logger plots.

        Args:
            loggers: A dictionary of logger instances keyed by name.
            filename: Optional filename prefix (or full name when ``grid`` is
                ``True``). Defaults to ``"metrics"`` for the grid layout and
                to the metric name alone for per-metric layouts.
            path: If provided, each figure is saved as
                ``{path}/{metric_name}.{ext}`` (or
                ``{path}/{filename}.{ext}`` for the grid layout). The
                directory must already exist.
            show: If ``True`` the figures are also displayed (blocking).
            grid: If ``True``, render all metrics in a single 2x3 grid figure
                instead of one figure per metric.
            ext: The file extension / matplotlib format to save with.
                Defaults to ``"pdf"`` to match the benchmarks output.
            log_scale: If ``True``, the y-axis is rendered with a base-10
                logarithmic scale. Samples are clamped to ``log_floor``
                before plotting so that residuals that are exactly zero do
                not break the log scale or trigger matplotlib's
                ``nonpositive`` warnings.
            log_floor: Strictly positive floor used to clamp samples when
                ``log_scale`` is enabled. Unused otherwise. Defaults to
                ``1e-12``.

        Raises:
            ValueError: If any logger is not a :class:`PhysicsMetricsLogger`,
                if the loggers do not share ``num_worlds``, if the output
                directory does not exist, or if ``log_scale`` is ``True``
                and ``log_floor`` is not a strictly positive number.
        """
        if cls.plt is None:
            cls._initialize_plt()
        if cls.plt is None:
            msg.critical("matplotlib is not available, skipping plotting.")
            return

        if not all(isinstance(logger, PhysicsMetricsLogger) for logger in loggers.values()):
            raise ValueError("All loggers must be instances of PhysicsMetricsLogger.")

        if not any(logger.num_logged_frames > 0 for logger in loggers.values()):
            msg.warning("No logged frames to plot, skipping plotting.")
            return

        first_logger = next(iter(loggers.values()))
        if not all(logger.num_worlds == first_logger.num_worlds for logger in loggers.values()):
            raise ValueError("All loggers must have the same number of worlds.")

        if path is not None and not os.path.isdir(path):
            raise ValueError(
                f"Plot output directory '{path}' does not exist. Please create it before calling plot_comparison()."
            )

        if log_scale and (not isinstance(log_floor, (int, float)) or float(log_floor) <= 0.0):
            raise ValueError(
                f"Expected 'log_floor' to be a strictly positive number when 'log_scale' is True, got {log_floor!r}."
            )

        plt = cls.plt
        x_label = "Time (s)" if first_logger.dt is not None else "Step"
        logged_data = [
            (name, logger.num_worlds, logger.time_axis(), logger.to_numpy()) for name, logger in loggers.items()
        ]

        if grid:
            if filename is None:
                filename = "metrics"
            n_rows, n_cols = 2, 3
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
            axes = axes.flatten()
            for i, field in enumerate(_SCALAR_FIELDS_FLOAT32):
                cls._plot_overlay_metric(
                    logged_data,
                    field,
                    x_label,
                    axes[i],
                    log_scale=log_scale,
                    log_floor=float(log_floor),
                )
            for j in range(len(_SCALAR_FIELDS_FLOAT32), len(axes)):
                axes[j].set_visible(False)
            fig.tight_layout()
            if path is not None:
                fig.savefig(os.path.join(path, f"{filename}.{ext}"), format=ext, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
        else:
            if filename is None:
                filename = ""
                separator = ""
            else:
                separator = "_"
            for field in _SCALAR_FIELDS_FLOAT32:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                cls._plot_overlay_metric(
                    logged_data,
                    field,
                    x_label,
                    ax,
                    log_scale=log_scale,
                    log_floor=float(log_floor),
                )
                fig.tight_layout()
                if path is not None:
                    fig.savefig(
                        os.path.join(path, f"{filename}{separator}{field}.{ext}"),
                        format=ext,
                        dpi=300,
                        bbox_inches="tight",
                    )
                if show:
                    plt.show()
                plt.close(fig)

    ###
    # Tables
    ###

    def table(
        self,
        filename: str | None = None,
        path: str | None = None,
        to_console: bool = False,
        per_world: bool = False,
    ) -> None:
        """Summarizes the recorded per-world residuals as a min/max/mean table.

        Reduces the chronological history returned by :meth:`to_numpy` over
        the frame axis (and optionally also the world axis) and emits the
        resulting statistics as a CSV file and/or a :mod:`rich`-rendered
        console table. Only the six scalar fields in
        :data:`_SCALAR_FIELDS_FLOAT32` are summarized; argmax companion
        fields are skipped, mirroring the convention used by :meth:`plot`.

        Args:
            filename: Optional CSV filename (without extension). Defaults to
                ``"metrics_stats"``. Ignored when ``path`` is ``None``.
            path: If provided, the table is written to
                ``{path}/{filename}.csv``. The directory must already exist.
            to_console: If ``True``, the table is also rendered to stdout
                using :mod:`rich`. The ``rich`` package must be installed.
            per_world: If ``True``, statistics are reported per world (one
                row per ``(world, metric)`` pair). If ``False``, statistics
                are aggregated across worlds (one row per metric).

        Raises:
            ValueError: If ``path`` is provided but does not refer to an
                existing directory.
            ImportError: If ``to_console=True`` and the ``rich`` package is
                not installed.
        """
        if path is None and not to_console:
            msg.warning("table() called with no output target (path=None, to_console=False); skipping.")
            return
        if path is not None and not os.path.isdir(path):
            raise ValueError(f"Output directory '{path}' does not exist. Please create it before calling table().")
        if self.num_logged_frames == 0:
            msg.warning("No logged frames to summarize, skipping table generation.")
            return
        if filename is None:
            filename = "metrics_stats"

        np_data = self.to_numpy()
        stats = _compute_summary_stats(np_data, per_world=per_world)
        rows = _build_stats_rows(
            stats_by_logger=[(None, stats)],
            per_world=per_world,
            num_worlds=self._num_worlds,
        )

        if path is not None:
            csv_path = os.path.join(path, f"{filename}.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                header: list[str] = []
                if per_world:
                    header.append("World")
                header.append("Metric")
                header.extend(_STAT_LABELS)
                writer.writerow(header)
                writer.writerows(rows)

        if to_console:
            self._render_stats_table(
                rows=rows,
                logger_names=[],
                per_world=per_world,
                title=("Physics Metrics Statistics (per-world)" if per_world else "Physics Metrics Statistics"),
            )

    @classmethod
    def table_comparison(
        cls,
        loggers: dict[str, PhysicsMetricsLogger],
        filename: str | None = None,
        path: str | None = None,
        to_console: bool = False,
        per_world: bool = False,
        color_rankings: bool = False,
    ) -> None:
        """Summarizes multiple loggers' residuals side-by-side as a stats table.

        For each scalar metric (and optionally each world), computes max and
        mean over each logger's recorded history and renders the per-logger
        pairs as nested sub-columns. The output formats and validation
        mirror :meth:`table` and :meth:`plot_comparison`.

        Args:
            loggers: A dictionary of :class:`PhysicsMetricsLogger` instances
                keyed by display name. All loggers must agree on
                :attr:`num_worlds`.
            filename: Optional CSV filename (without extension). Defaults to
                ``"metrics_stats_comparison"``. Ignored when ``path`` is
                ``None``.
            path: If provided, the table is written to
                ``{path}/{filename}.csv``. The directory must already exist.
            to_console: If ``True``, the table is also rendered to stdout
                using :mod:`rich`. The ``rich`` package must be installed.
            per_world: If ``True``, statistics are reported per world (one
                row per ``(world, metric)`` pair). If ``False``, statistics
                are aggregated across worlds (one row per metric).
            color_rankings: If ``True``, the console-rendered stat cells are
                colored by ranking against the peer loggers for the same
                ``(metric, stat)`` (and ``world`` when ``per_world=True``).
                Lower values (better residuals) are pulled toward light
                green; higher values (worse residuals) toward light red,
                with a yellow midpoint for intermediate ranks. Has no
                effect on the CSV output, which always carries the raw
                numeric values.

        Raises:
            ValueError: If ``loggers`` is empty, any value is not a
                :class:`PhysicsMetricsLogger`, the loggers disagree on
                :attr:`num_worlds`, or ``path`` does not refer to an
                existing directory.
            ImportError: If ``to_console=True`` and the ``rich`` package is
                not installed.
        """
        if not loggers:
            raise ValueError("At least one logger must be provided.")
        if not all(isinstance(logger, PhysicsMetricsLogger) for logger in loggers.values()):
            raise ValueError("All loggers must be instances of PhysicsMetricsLogger.")
        first_logger = next(iter(loggers.values()))
        if not all(logger.num_worlds == first_logger.num_worlds for logger in loggers.values()):
            raise ValueError("All loggers must have the same number of worlds.")
        if path is None and not to_console:
            msg.warning("table_comparison() called with no output target (path=None, to_console=False); skipping.")
            return
        if path is not None and not os.path.isdir(path):
            raise ValueError(
                f"Output directory '{path}' does not exist. Please create it before calling table_comparison()."
            )
        if not any(logger.num_logged_frames > 0 for logger in loggers.values()):
            msg.warning("No logged frames to summarize, skipping table generation.")
            return
        if filename is None:
            filename = "metrics_stats_comparison"

        num_worlds = first_logger.num_worlds
        stats_by_logger: list[tuple[str | None, dict[str, tuple[float, float] | np.ndarray]]] = [
            (name, _compute_summary_stats(logger.to_numpy(), per_world=per_world)) for name, logger in loggers.items()
        ]
        rows = _build_stats_rows(stats_by_logger, per_world=per_world, num_worlds=num_worlds)

        if path is not None:
            csv_path = os.path.join(path, f"{filename}.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # Two-row header: top row carries logger names spanning their
                # (max, mean) sub-columns; the bottom row carries the
                # sub-column labels under each logger group. The leading
                # ``World`` / ``Metric`` cells are placed on the bottom row.
                top_header: list[str] = []
                sub_header: list[str] = []
                if per_world:
                    top_header.append("")
                    sub_header.append("World")
                top_header.append("")
                sub_header.append("Metric")
                for name in loggers.keys():
                    top_header.append(name)
                    top_header.extend("" for _ in range(len(_STAT_LABELS) - 1))
                    sub_header.extend(_STAT_LABELS)
                writer.writerow(top_header)
                writer.writerow(sub_header)
                writer.writerows(rows)

        if to_console:
            cell_styles: list[list[str | None]] | None = None
            if color_rankings:
                cell_styles = _compute_ranking_colors(
                    stats_by_logger=stats_by_logger,
                    per_world=per_world,
                    num_worlds=num_worlds,
                )
            cls._render_stats_table(
                rows=rows,
                logger_names=list(loggers.keys()),
                per_world=per_world,
                title=(
                    "Physics Metrics Statistics Comparison (per-world)"
                    if per_world
                    else "Physics Metrics Statistics Comparison"
                ),
                cell_styles=cell_styles,
            )

    ###
    # Internals
    ###

    @staticmethod
    def _render_stats_table(
        rows: list[list[str]],
        logger_names: list[str],
        per_world: bool,
        title: str,
        cell_styles: list[list[str | None]] | None = None,
    ) -> None:
        """Renders a stats table to stdout via :mod:`rich`.

        When ``logger_names`` is empty the table uses flat ``max/mean``
        columns (the :meth:`table` case). When ``logger_names`` is non-empty
        each logger gets its own ``max/mean`` sub-column group cycling
        through :data:`_OVERLAY_COLORS` so the comparison table stays
        visually aligned with :meth:`plot_comparison`'s color scheme.

        Args:
            rows: Pre-formatted row strings as produced by
                :func:`_build_stats_rows`.
            logger_names: Per-logger display names. Empty for :meth:`table`.
            per_world: Whether the rows include a leading ``World`` cell.
            title: Plain-text title to display above the table.
            cell_styles: Optional matching grid of rich style strings (or
                ``None``) used to color individual cells. When ``None`` (or
                an entry is ``None``) the cell renders unstyled.
        """
        box, Console, Table, Text = _lazy_import_rich()
        table = Table(
            title=title,
            show_header=True,
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            pad_edge=True,
        )
        if per_world:
            table.add_column("World", justify="left", no_wrap=True, style="bold")
        table.add_column("Metric", justify="left", no_wrap=True, style="bold")
        if logger_names:
            for i, name in enumerate(logger_names):
                color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
                for j, sub in enumerate(_STAT_LABELS):
                    header = Text(justify="left")
                    if j == 0:
                        header.append(name, style=f"bold {color}")
                    header.append("\n")
                    header.append(sub, style=f"dim {color}")
                    table.add_column(header=header, justify="right", no_wrap=True)
        else:
            for sub in _STAT_LABELS:
                table.add_column(sub, justify="right", no_wrap=True)
        for r_idx, row in enumerate(rows):
            row_styles = cell_styles[r_idx] if cell_styles is not None else None
            if row_styles is None:
                table.add_row(*row)
            else:
                cells: list[str | Text] = []
                for c_idx, cell in enumerate(row):
                    style = row_styles[c_idx] if c_idx < len(row_styles) else None
                    if style is None:
                        cells.append(cell)
                    else:
                        cells.append(Text(cell, style=style))
                table.add_row(*cells)
        console = Console()
        console.rule()
        console.print(table, crop=False)
        console.rule()

    @staticmethod
    def _plot_overlay_metric(
        data: list[tuple[str, int, np.ndarray, dict[str, np.ndarray]]],
        field: str,
        x_label: str,
        ax: plt.Axes,
        log_scale: bool = False,
        log_floor: float = 1e-12,
    ):
        """
        Draws one overlaid metric panel onto ``ax`` for the given scalar ``field``.

        Each entry of ``data`` is a ``(name, nw, time, np_data)`` tuple where
        ``time`` and ``np_data`` are pre-computed via the logger's
        :meth:`time_axis` and :meth:`to_numpy` methods. One curve is drawn per
        world per logger, cycling through :data:`_OVERLAY_COLORS`,
        :data:`_OVERLAY_LINESTYLES`, and :data:`_OVERLAY_MARKERS`. Cycling
        line styles and markers (combined with sub-unit ``alpha``) keeps
        identical curves from different setups distinguishable: contact-
        penetration residuals in particular depend only on the shared pre-step
        state and contact geometry, so two leader/follower solvers will
        produce bit-identical curves that would otherwise be hidden by the
        last-drawn solid line.

        Args:
            data: Pre-computed ``(name, num_worlds, time, np_data)`` tuples,
                one per logger.
            field: Scalar residual field to plot.
            x_label: Label for the x-axis.
            ax: Matplotlib axes to draw on.
            log_scale: If ``True``, the y-axis is rendered with a base-10
                logarithmic scale and samples are clamped to ``log_floor``
                prior to plotting so that zero residuals do not produce
                ``nonpositive`` matplotlib warnings.
            log_floor: Strictly positive floor used to clamp samples when
                ``log_scale`` is enabled. Unused when ``log_scale`` is
                ``False``.
        """
        for i, (name, nw, time, np_data) in enumerate(data):
            color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
            linestyle = _OVERLAY_LINESTYLES[i % len(_OVERLAY_LINESTYLES)]
            marker = _OVERLAY_MARKERS[i % len(_OVERLAY_MARKERS)]
            for w in range(nw):
                world_label = f" (world_{w})" if nw > 1 else ""
                y = np_data[field][:, w]
                if log_scale:
                    y = np.maximum(y, log_floor)
                ax.plot(
                    time,
                    y,
                    color=color,
                    marker=marker,
                    markersize=3,
                    linestyle=linestyle,
                    alpha=0.7,
                    label=f"{name}{world_label}",
                )
        equation = _METRIC_EQUATIONS[field]
        base_title = _METRIC_TITLES[field]
        ax.set_title(f"{base_title} \n ({equation})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(field)
        if log_scale:
            ax.set_yscale("log")
        ax.grid()
        ax.legend(loc="best", frameon=False)
