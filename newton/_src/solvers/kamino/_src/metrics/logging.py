# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a specialized logger that records :class:`SolutionMetricsNewton`
history on the same device as the wrapped metrics container.

The :class:`SolutionMetricsLogger` class allocates per-frame log buffers for
every field of :class:`SolutionMetricsData`, with an optional fixed-size
rolling window or bounded early-exit overflow policy and a configurable
sample-decimation rate. It also exposes utilities to extract the recorded
data as numpy arrays in chronological order, and to render per-metric
matplotlib plots that follow the equation-subtitled format used by
:func:`render_physics_metrics_plots` in the benchmarks utilities.

Usage
-----

A typical example for using this module is::

    from newton._src.solvers.kamino._src.metrics import (
        SolutionMetricsLogger,
        SolutionMetricsNewton,
    )

    metrics = SolutionMetricsNewton(dt=dt, model=model)
    logger = SolutionMetricsLogger(
        metrics=metrics,
        max_frames=1000,
        mode=SolutionMetricsLogger.Mode.ROLLING,
        decimation=2,
    )

    for _ in range(num_steps):
        # ... advance the simulation, run the solver, evaluate the metrics ...
        metrics.evaluate(state, state_p, control, contacts)
        logger.log()

    np_data = logger.to_numpy()
    logger.plot(path="/tmp/metrics", ext="pdf")
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..core.types import float32, int32, int64
from ..solvers.metrics import SolutionMetrics
from ..utils import logger as msg
from .core import SolutionMetricsNewton

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

###
# Module interface
###

__all__ = ["SolutionMetricsLogger"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

# Names of the scalar (float32) metric fields recorded by the logger,
# in the same order used by `render_physics_metrics_plots`.
_SCALAR_METRIC_FIELDS: tuple[str, ...] = (
    "r_eom",
    "r_kinematics",
    "r_cts_joints",
    "r_cts_limits",
    "r_cts_contacts",
    "r_v_plus",
    "r_ncp_primal",
    "r_ncp_dual",
    "r_ncp_compl",
    "r_vi_natmap",
    "f_ncp",
    "f_ccp",
)

# Names of the argmax fields that pack two indices into a 64-bit key.
_ARGMAX_FIELDS_INT64: tuple[str, ...] = (
    "r_eom_argmax",
    "r_kinematics_argmax",
    "r_cts_joints_argmax",
    "r_cts_limits_argmax",
)

# Names of the argmax fields that store a single 32-bit index.
_ARGMAX_FIELDS_INT32: tuple[str, ...] = (
    "r_cts_contacts_argmax",
    "r_v_plus_argmax",
    "r_ncp_primal_argmax",
    "r_ncp_dual_argmax",
    "r_ncp_compl_argmax",
    "r_vi_natmap_argmax",
)

# Human-readable plot titles per scalar metric (without the equation suffix).
_METRIC_TITLES: dict[str, str] = {
    "r_eom": "Equations-of-Motion Residual",
    "r_kinematics": "Joint Kinematics Constraint Residual",
    "r_cts_joints": "Joints Constraint Residual",
    "r_cts_limits": "Limits Constraint Residual",
    "r_cts_contacts": "Contacts Constraint Residual",
    "r_v_plus": "Post-Event Constraint Velocity Residual",
    "r_ncp_primal": "NCP Primal Residual",
    "r_ncp_dual": "NCP Dual Residual",
    "r_ncp_compl": "NCP Complementary Residual",
    "r_vi_natmap": "VI Natural-Map Residual",
    "f_ncp": "NCP Objective",
    "f_ccp": "CCP Objective",
}

# LaTeX equations rendered as the second line of each metric's plot title.
# Identical to the strings used by `render_physics_metrics_plots` so the two
# tools produce visually consistent output.
_METRIC_EQUATIONS: dict[str, str] = {
    "r_eom": r"$\Vert \, M \, (u^+ - u^-) - dt \, (h + J_a^T \, \tau) - J^T \, \lambda \, \Vert_\infty $",
    "r_kinematics": r"$\Vert \, J_j \cdot u^+ \, \Vert_\infty $",
    "r_cts_joints": r"$\Vert \, f_j(q) \, \Vert_\infty $",
    "r_cts_limits": r"$\Vert \, f_l(q) \, \Vert_\infty $",
    "r_cts_contacts": r"$\Vert \, f_{c,N}(q) \, \Vert_\infty $",
    "r_v_plus": r"$\Vert \, v^+ - D \cdot \lambda - v_f \, \Vert_\infty $",
    "r_ncp_primal": r"$\Vert \, \lambda - P_K(\lambda) \, \Vert_\infty $",
    "r_ncp_dual": r"$\Vert \, v_a^+ - P_{K^*}(v_a^+) \, \Vert_\infty $",
    "r_ncp_compl": r"$\Vert \, \lambda^T \, v_a^+ \, \Vert_\infty $",
    "r_vi_natmap": r"$\Vert \, \lambda - P_{K^*}(\lambda - v_a^+(\lambda)) \, \Vert_\infty $",
    "f_ncp": r"$ 0.5 \, \lambda^T \, D \, \lambda + \lambda^T \, (v_f + s) $",
    "f_ccp": r"$ 0.5 \, \lambda^T \, D \, \lambda + v_f^T \, \lambda $",
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

###
# Kernels
###


@wp.kernel
def _write_log_row_float32(
    src: wp.array[float32],
    write_idx: int32,
    dest: wp.array2d[float32],
):
    """Copies one ``(num_worlds,)`` source row into ``dest[write_idx, :]``."""
    wid = wp.tid()
    dest[write_idx, wid] = src[wid]


@wp.kernel
def _write_log_row_int64(
    src: wp.array[int64],
    write_idx: int32,
    dest: wp.array2d[int64],
):
    """Copies one ``(num_worlds,)`` source row into ``dest[write_idx, :]``."""
    wid = wp.tid()
    dest[write_idx, wid] = src[wid]


@wp.kernel
def _write_log_row_int32(
    src: wp.array[int32],
    write_idx: int32,
    dest: wp.array2d[int32],
):
    """Copies one ``(num_worlds,)`` source row into ``dest[write_idx, :]``."""
    wid = wp.tid()
    dest[write_idx, wid] = src[wid]


###
# Interfaces
###


class SolutionMetricsLogger:
    """
    Records :class:`SolutionMetricsData` history on the metrics' device.

    The logger accepts either a :class:`SolutionMetricsNewton` wrapper or a
    raw :class:`SolutionMetrics` instance. Both expose the same trio of
    public properties (``model``, ``data``, ``device``) that the logger
    relies on, so it works against either backend interchangeably.

    The logger allocates one Warp 2-D buffer per
    :class:`SolutionMetricsData` field of shape
    ``(max_frames, num_worlds)`` on the same device as the wrapped
    metrics container. Each call to :meth:`log` appends the current
    per-world values from ``metrics.data`` into the next slot of the
    rolling window.

    The buffer-overflow policy is controlled by :class:`Mode`:

    - :attr:`Mode.ROLLING` wraps the write index modulo ``max_frames``,
      so the buffer always holds the most recent ``max_frames`` samples.
    - :attr:`Mode.BOUNDED` stops recording once ``max_frames`` samples
      have been logged; subsequent :meth:`log` calls are no-ops.

    The optional ``decimation`` argument skips intermediate calls so only
    every ``decimation``-th call actually writes a new frame; this is
    useful when :meth:`log` is invoked once per simulation step but a
    coarser sampling is sufficient for analysis.

    Numpy extraction via :meth:`to_numpy` always returns the recorded
    samples in chronological order (oldest first), and :meth:`plot`
    renders one matplotlib figure per scalar metric in the format used
    by :func:`render_physics_metrics_plots`.

    The argmax companion fields are recorded too; they retain the packed
    representation produced by the metrics kernels and can be unpacked via
    :meth:`unpack_argmax_key`.
    """

    class Mode(IntEnum):
        """Buffer overflow behavior for :class:`SolutionMetricsLogger`."""

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
        metrics: SolutionMetricsNewton | SolutionMetrics,
        max_frames: int,
        mode: Mode = Mode.BOUNDED,
        decimation: int = 1,
        dt: float | None = None,
    ):
        """
        Initializes the solution-metrics logger.

        Args:
            metrics: The metrics container to record from. Either a
                :class:`SolutionMetricsNewton` wrapper (Newton front-end API)
                or a raw :class:`SolutionMetrics` instance (the back-end
                container used internally by :class:`SolverKaminoImpl`).
                Must have been finalized prior to constructing the logger.
            max_frames: The maximum number of frames recorded by the logger.
                Must be a strictly positive integer.
            mode: The buffer-overflow policy. Defaults to :attr:`Mode.BOUNDED`.
            decimation: Sample decimation rate. Only every ``decimation``-th
                :meth:`log` call writes a new frame. Defaults to ``1`` (no
                decimation). Must be a strictly positive integer.
            dt: Optional simulation time step used to scale the time axis on
                plots. If a positive value is supplied it is pinned for the
                lifetime of the logger. If ``None``, the time step is read
                live from ``metrics.model.time.dt[0]`` on every access (so
                values populated by the solver after logger construction are
                picked up); non-positive or unreadable values fall back to a
                unit-less "Simulation Step" labelling.
        """
        if not isinstance(metrics, (SolutionMetricsNewton, SolutionMetrics)):
            raise TypeError("Expected 'metrics' to be of type `SolutionMetricsNewton` or `SolutionMetrics`.")
        try:
            # `data` raises RuntimeError on either container when not finalized.
            _ = metrics.data
            _ = metrics.model
        except RuntimeError as e:
            raise RuntimeError(
                "SolutionMetricsLogger requires a finalized metrics instance. Call finalize() first."
            ) from e
        if not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError(f"Expected 'max_frames' to be a positive integer, got {max_frames!r}.")
        if not isinstance(decimation, int) or decimation <= 0:
            raise ValueError(f"Expected 'decimation' to be a positive integer, got {decimation!r}.")
        if not isinstance(mode, SolutionMetricsLogger.Mode):
            raise TypeError("Expected 'mode' to be a `SolutionMetricsLogger.Mode` value.")

        # Attempt to initialize matplotlib for plotting
        self._initialize_plt()

        # Store the metrics instance and related configurations
        self._metrics: SolutionMetricsNewton | SolutionMetrics = metrics
        self._max_frames: int = int(max_frames)
        self._mode: SolutionMetricsLogger.Mode = mode
        self._decimation: int = int(decimation)
        self._device: wp.DeviceLike = metrics.device
        self._num_worlds: int = int(metrics.model.size.num_worlds)

        # Resolve the simulation time step. An explicit positive ``dt`` is pinned
        # for the lifetime of the logger; otherwise the value is read live from
        # ``metrics.model.time.dt[0]`` on every access so updates the solver
        # makes after logger construction (e.g. via ``set_uniform_timestep`` in
        # ``SolverKamino.step``) are reflected in the time axis.
        if dt is not None:
            if not isinstance(dt, (int, float)) or float(dt) <= 0.0:
                raise ValueError(f"Expected 'dt' to be a positive number, got {dt!r}.")
            self._dt_override: float | None = float(dt)
        else:
            self._dt_override = None

        # Internal counters: ``_call_count`` tracks every :meth:`log` invocation (used
        # by the decimation gate); ``_frames_total`` tracks the number of writes that
        # actually landed in the buffer (used by the overflow / chronological-ordering
        # logic).
        self._call_count: int = 0
        self._frames_total: int = 0

        # Allocate every per-frame log buffer on the metrics' device. The 2-D layout
        # ``(max_frames, num_worlds)`` matches the per-world scalar fan-out of the
        # underlying metrics fields.
        with wp.ScopedDevice(self._device):
            self.log_r_eom: wp.array | None = None
            self.log_r_eom_argmax: wp.array | None = None
            self.log_r_kinematics: wp.array | None = None
            self.log_r_kinematics_argmax: wp.array | None = None
            self.log_r_cts_joints: wp.array | None = None
            self.log_r_cts_joints_argmax: wp.array | None = None
            self.log_r_cts_limits: wp.array | None = None
            self.log_r_cts_limits_argmax: wp.array | None = None
            self.log_r_cts_contacts: wp.array | None = None
            self.log_r_cts_contacts_argmax: wp.array | None = None
            self.log_r_v_plus: wp.array | None = None
            self.log_r_v_plus_argmax: wp.array | None = None
            self.log_r_ncp_primal: wp.array | None = None
            self.log_r_ncp_primal_argmax: wp.array | None = None
            self.log_r_ncp_dual: wp.array | None = None
            self.log_r_ncp_dual_argmax: wp.array | None = None
            self.log_r_ncp_compl: wp.array | None = None
            self.log_r_ncp_compl_argmax: wp.array | None = None
            self.log_r_vi_natmap: wp.array | None = None
            self.log_r_vi_natmap_argmax: wp.array | None = None
            self.log_f_ncp: wp.array | None = None
            self.log_f_ccp: wp.array | None = None

            shape = (self._max_frames, self._num_worlds)
            for field in _SCALAR_METRIC_FIELDS:
                setattr(self, f"log_{field}", wp.zeros(shape=shape, dtype=float32))
            for field in _ARGMAX_FIELDS_INT64:
                setattr(self, f"log_{field}", wp.full(shape=shape, value=-1, dtype=int64))
            for field in _ARGMAX_FIELDS_INT32:
                setattr(self, f"log_{field}", wp.full(shape=shape, value=-1, dtype=int32))

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
        """Returns the resolved simulation time step, or ``None`` if unavailable.

        If an explicit positive ``dt`` was provided to the constructor it is
        returned as-is; otherwise the value is read live from
        ``metrics.model.time.dt[0]`` so updates made by the solver after the
        logger was constructed are reflected here. Non-positive (or
        unreadable) values yield ``None``.
        """
        return self._resolve_dt()

    def _resolve_dt(self) -> float | None:
        """Resolve the current effective time step.

        See :attr:`dt` for the resolution policy.
        """
        if self._dt_override is not None:
            return self._dt_override
        try:
            value = float(self._metrics.model.time.dt.numpy()[0])
        except Exception:
            return None
        return value if value > 0.0 else None

    @property
    def num_logged_frames(self) -> int:
        """Returns the number of valid frames currently stored in the buffer."""
        return min(self._frames_total, self._max_frames)

    @property
    def num_total_writes(self) -> int:
        """Returns the cumulative number of writes (including those overwritten in rolling mode)."""
        return self._frames_total

    @property
    def is_full(self) -> bool:
        """Returns whether the buffer has reached ``max_frames`` writes."""
        return self._frames_total >= self._max_frames

    ###
    # Operations
    ###

    def reset(self):
        """Resets the logger counters and clears every log buffer."""
        self._call_count = 0
        self._frames_total = 0
        for field in _SCALAR_METRIC_FIELDS:
            getattr(self, f"log_{field}").zero_()
        for field in _ARGMAX_FIELDS_INT64:
            getattr(self, f"log_{field}").fill_(-1)
        for field in _ARGMAX_FIELDS_INT32:
            getattr(self, f"log_{field}").fill_(-1)

    def log(self):
        """Records the current :class:`SolutionMetricsData` values into the next buffer slot.

        Calls that fall on a decimation-skipped phase, or that occur after the
        buffer has filled in :attr:`Mode.BOUNDED`, are silently dropped.
        """
        # Decimation gate: only every ``decimation``-th call writes a new frame.
        if (self._call_count % self._decimation) != 0:
            self._call_count += 1
            return

        # Bounded-mode early-exit once the buffer is full.
        if self._mode == SolutionMetricsLogger.Mode.BOUNDED and self._frames_total >= self._max_frames:
            self._call_count += 1
            return

        write_idx = int32(self._frames_total % self._max_frames)
        data = self._metrics.data

        for field in _SCALAR_METRIC_FIELDS:
            wp.launch(
                kernel=_write_log_row_float32,
                dim=self._num_worlds,
                inputs=[getattr(data, field), write_idx, getattr(self, f"log_{field}")],
                device=self._device,
            )
        for field in _ARGMAX_FIELDS_INT64:
            wp.launch(
                kernel=_write_log_row_int64,
                dim=self._num_worlds,
                inputs=[getattr(data, field), write_idx, getattr(self, f"log_{field}")],
                device=self._device,
            )
        for field in _ARGMAX_FIELDS_INT32:
            wp.launch(
                kernel=_write_log_row_int32,
                dim=self._num_worlds,
                inputs=[getattr(data, field), write_idx, getattr(self, f"log_{field}")],
                device=self._device,
            )

        self._frames_total += 1
        self._call_count += 1

    ###
    # Numpy extraction
    ###

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Returns the recorded data as numpy arrays in chronological order.

        Each entry of the returned dictionary has shape
        ``(num_logged_frames, num_worlds)`` and is keyed by the
        corresponding :class:`SolutionMetricsData` field name. Both the
        scalar metric fields and their argmax companions are returned.

        In :attr:`Mode.ROLLING` the rows are rotated so that the oldest
        recorded frame is at index ``0`` and the most recent at index
        ``num_logged_frames - 1``.

        Returns:
            A dictionary mapping field name to its recorded values.
        """
        n = self.num_logged_frames
        result: dict[str, np.ndarray] = {}
        for field in (*_SCALAR_METRIC_FIELDS, *_ARGMAX_FIELDS_INT64, *_ARGMAX_FIELDS_INT32):
            buf = getattr(self, f"log_{field}").numpy()
            if n == 0:
                result[field] = buf[:0].copy()
                continue
            if self._mode == SolutionMetricsLogger.Mode.ROLLING and self._frames_total > self._max_frames:
                # The buffer wrapped around at least once; rotate so that the oldest
                # recorded frame is at index 0.
                write_idx = self._frames_total % self._max_frames
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
        dt = self._resolve_dt()
        scale = (dt if dt is not None else 1.0) * float(self._decimation)
        return np.arange(n, dtype=np.float32) * scale

    @staticmethod
    def unpack_argmax_key(key: int) -> tuple[int, int]:
        """Unpacks a packed 64-bit argmax key into its two 31-bit halves.

        Args:
            key: The packed key as produced by ``build_pair_key2``.

        Returns:
            A pair ``(index_A, index_B)`` of 31-bit indices.
        """
        index_a = (int(key) >> 32) & 0x7FFFFFFF
        index_b = int(key) & 0x7FFFFFFF
        return int(index_a), int(index_b)

    ###
    # Plotting
    ###

    def plot(
        self,
        path: str | None = None,
        show: bool = False,
        ext: str = "pdf",
    ):
        """
        Renders one matplotlib figure per scalar metric.

        Each figure follows the equation-subtitled format used by
        :func:`render_physics_metrics_plots`: the title is the
        human-readable metric name with the underlying mathematical
        definition rendered as a LaTeX subtitle. One curve is drawn per
        world.

        Args:
            path: If provided, each figure is saved as
                ``{path}/{metric_name}.{ext}``. The directory must already
                exist.
            show: If ``True`` the figures are also displayed (blocking).
            ext: The file extension / matplotlib format to save with.
                Defaults to ``"pdf"`` to match the benchmarks output.
        """
        if self.plt is None:
            msg.warning("matplotlib is not available, skipping plotting.")
            return
        if self.num_logged_frames == 0:
            msg.warning("No logged frames to plot, skipping plotting.")
            return
        if path is not None and not os.path.isdir(path):
            raise ValueError(f"Plot output directory '{path}' does not exist. Please create it before calling plot().")
        time = self.time_axis()
        np_data = self.to_numpy()
        x_label = "Time (s)" if self._resolve_dt() is not None else "Step"
        for field in _SCALAR_METRIC_FIELDS:
            equation = _METRIC_EQUATIONS[field]
            base_title = _METRIC_TITLES[field]
            title = f"{base_title} \n ({equation})"
            fig, ax = self.plt.subplots(1, 1, figsize=(10, 6))
            data = np_data[field]
            for w in range(self._num_worlds):
                ax.plot(
                    time,
                    data[:, w],
                    label=f"world_{w}",
                    marker="o",
                    markersize=4,
                )
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(field)
            ax.grid()
            if self._num_worlds > 1:
                ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            if path is not None:
                fig_path = os.path.join(path, f"{field}.{ext}")
                fig.savefig(fig_path, format=ext, dpi=300, bbox_inches="tight")
            if show:
                self.plt.show()
            self.plt.close(fig)

    @classmethod
    def plot_comparison(
        cls,
        loggers: dict[str, SolutionMetricsLogger],
        path: str | None = None,
        show: bool = False,
        grid: bool = False,
        ext: str = "pdf",
    ):
        """
        Renders overlaid :class:`SolutionMetricsLogger` plots across multiple logger instances.

        Iterates the scalar metric fields recorded by every setup's logger and
        plots them on a shared axis, drawing one curve per world per setup using
        :data:`PLOT_COLORS` cycled by setup index. The figure title and LaTeX
        subtitle follow :meth:`SolutionMetricsLogger.plot`, so the output is
        visually consistent with the per-logger plots.

        Args:
            loggers:
                A dictionary of logger instances keyed by name.
            path:
                If provided, each figure is saved as ``{path}/{metric_name}.{ext}``.
                The directory must already exist.
            show:
                If ``True`` the figures are also displayed (blocking).
            grid:
                If ``True``, render all metrics in a single 3x4
                grid figure instead of one figure per metric.
            ext:
                The file extension / matplotlib format to save with.
                Defaults to ``"pdf"`` to match the benchmarks output.
        """
        # Attempt to initialize matplotlib for plotting
        if cls.plt is None:
            cls._initialize_plt()
        if cls.plt is None:
            msg.critical("matplotlib is not available, skipping plotting.")
            return

        # Ensure all loggers are valid
        if not all(isinstance(logger, SolutionMetricsLogger) for logger in loggers.values()):
            raise ValueError("All loggers must be instances of SolutionMetricsLogger.")

        # Check that at least one logger has logged frames
        if not any(logger.num_logged_frames > 0 for logger in loggers.values()):
            msg.warning("No logged frames to plot, skipping plotting.")
            return

        # Get the first logger
        first_logger = list[SolutionMetricsLogger](loggers.values())[0]

        # Check that all loggers have the same number of worlds
        if not all(logger.num_worlds == first_logger.num_worlds for logger in loggers.values()):
            raise ValueError("All loggers must have the same number of worlds.")

        # Check that the output directory exists
        if path is not None and not os.path.isdir(path):
            raise ValueError(
                f"Plot output directory '{path}' does not exist. Please create it before calling plot_comparison()."
            )

        # Get the matplotlib instance
        plt = cls.plt

        # Get the time axis for the first logger
        x_label = "Time (s)" if first_logger.dt is not None else "Step"

        # Get the numpy data for all loggers
        logged_data = [
            (name, logger.num_worlds, logger.time_axis(), logger.to_numpy()) for name, logger in loggers.items()
        ]

        # Plot the data: If grid is True, plot all metrics in a
        # single 3x4 grid Otherwise, plot one figure per metric
        if grid:
            n_rows, n_cols = 3, 4
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 14))
            axes = axes.flatten()
            for i, field in enumerate(_SCALAR_METRIC_FIELDS):
                cls._plot_overlay_metric(logged_data, field, x_label, axes[i])
            # Hide any unused cells in case _SCALAR_METRIC_FIELDS shrinks below n_rows*n_cols.
            for j in range(len(_SCALAR_METRIC_FIELDS), len(axes)):
                axes[j].set_visible(False)
            fig.tight_layout()
            if path is not None:
                fig.savefig(os.path.join(path, f"metrics_grid.{ext}"), format=ext, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
        else:
            for field in _SCALAR_METRIC_FIELDS:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                cls._plot_overlay_metric(logged_data, field, x_label, ax)
                fig.tight_layout()
                if path is not None:
                    fig.savefig(os.path.join(path, f"{field}.{ext}"), format=ext, dpi=300, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(fig)

    ###
    # Internals
    ###

    @staticmethod
    def _plot_overlay_metric(
        data: list[tuple[str, int, np.ndarray, dict[str, np.ndarray]]],
        field: str,
        x_label: str,
        ax: plt.Axes,
    ):
        """
        Draws one overlaid metric panel onto ``ax`` for the given scalar ``field``.

        Each entry of ``data`` is a ``(name, nw, time, np_data)`` tuple where ``time`` and
        ``np_data`` are pre-computed via the logger's :meth:`SolutionMetricsLogger.time_axis`
        and :meth:`SolutionMetricsLogger.to_numpy` methods. One curve is drawn per world per
        logger, cycling through :data:`_OVERLAY_COLORS`.
        """
        for i, (name, nw, time, np_data) in enumerate(data):
            color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
            for w in range(nw):
                world_label = f" (world_{w})" if nw > 1 else ""
                ax.plot(
                    time,
                    np_data[field][:, w],
                    color=color,
                    marker="o",
                    markersize=3,
                    linestyle="-",
                    # linewidth=1.0,
                    label=f"{name}{world_label}",
                )
        equation = _METRIC_EQUATIONS[field]
        base_title = _METRIC_TITLES[field]
        ax.set_title(f"{base_title} \n ({equation})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(field)
        ax.grid()
        ax.legend(loc="best", frameon=False)
