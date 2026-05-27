# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a specialized logger that records :class:`SolverKamino` PADMM solver
status and active-constraint counts on the same device as the solver.

The :class:`SolverKaminoLogger` class allocates per-frame Warp buffers for each
recorded metric, with an optional fixed-size rolling window or bounded
early-exit overflow policy and a configurable sample-decimation rate. Unlike
:class:`SolutionMetricsLogger`, the per-frame counters and rollover/bounding
logic live on the target device, so a single :meth:`SolverKaminoLogger.log`
call expands to a fixed sequence of Warp kernel launches that can be safely
captured into a CUDA graph alongside :meth:`SolverKamino.step`.

It also exposes utilities to extract the recorded data as numpy arrays in
chronological order, and to render a single matplotlib figure with one
subplot per metric, overlaying one curve per simulated world.

Usage
-----

A typical example for using this module is::

    import newton

    from newton._src.solvers.kamino.utils.logger import SolverKaminoLogger

    solver = newton.solvers.SolverKamino(model=model, config=config)
    logger = SolverKaminoLogger(
        solver=solver,
        max_frames=1000,
        mode=SolverKaminoLogger.Mode.ROLLING,
        decimation=2,
    )

    for _ in range(num_steps):
        solver.step(state_in, state_out, control, contacts, dt)
        logger.log()
        state_in, state_out = state_out, state_in

    np_data = logger.to_numpy()
    logger.plot(filename="kamino_status", path="/tmp", ext="pdf")
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .._src.core.types import float32, int32
from .._src.solvers.padmm.types import PADMMPenalty, PADMMStatus
from .._src.utils import logger as msg

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from ..solver_kamino import SolverKamino

###
# Module interface
###

__all__ = ["SolverKaminoLogger"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

# Names of the float32 scalar metrics always recorded by the logger.
_SCALAR_METRIC_FIELDS_FLOAT32: tuple[str, ...] = (
    "r_p",
    "r_d",
    "r_c",
    "rho",
)

# Names of the int32 scalar metrics always recorded by the logger.
_SCALAR_METRIC_FIELDS_INT32: tuple[str, ...] = (
    "iterations",
    "converged",
    "num_restarts",
    "num_rho_updates",
    "num_limits",
    "num_contacts",
)

# Additional float32 fields recorded when ``with_iterate_residuals_info=True``.
# All three come from :class:`PADMMStatus`.
_SCALAR_METRIC_FIELDS_FLOAT32_ITERATE: tuple[str, ...] = (
    "r_dx",
    "r_dy",
    "r_dz",
)

# Additional float32 fields recorded when ``with_acceleration_info=True``.
# ``r_a`` comes from :class:`PADMMStatus`; ``a`` and ``a_factor`` come from
# :class:`PADMMState` and are only allocated when the underlying PADMM solver
# was constructed with ``use_acceleration=True``.
_SCALAR_METRIC_FIELDS_FLOAT32_ACCEL: tuple[str, ...] = (
    "r_a",
    "a",
    "a_factor",
)

# Default subset of fields shown in :meth:`SolverKaminoLogger.plot`, in display order.
_PLOTTED_METRIC_FIELDS: tuple[str, ...] = (
    "iterations",
    "r_p",
    "r_d",
    "r_c",
    "rho",
    "num_rho_updates",
    "num_limits",
    "num_contacts",
)

# Optional plotted fields for the iterate-residual extension.
_PLOTTED_METRIC_FIELDS_ITERATE: tuple[str, ...] = (
    "r_dx",
    "r_dy",
    "r_dz",
)

# Optional plotted fields for the acceleration-info extension. ``num_restarts``
# is always logged as a basic int32 field, but it is only surfaced on the plot
# when ``with_acceleration_info=True`` because it only makes sense alongside
# the rest of the Nesterov-acceleration diagnostics.
_PLOTTED_METRIC_FIELDS_ACCEL: tuple[str, ...] = (
    "r_a",
    "a",
    "a_factor",
    "num_restarts",
)

# Human-readable plot titles per metric.
_METRIC_TITLES: dict[str, str] = {
    "iterations": "PADMM Iterations",
    "r_p": "PADMM Primal Residual",
    "r_d": "PADMM Dual Residual",
    "r_c": "PADMM Complementarity Residual",
    "rho": "PADMM ALM Penalty (rho)",
    "num_rho_updates": "PADMM Penalty Updates",
    "num_limits": "Active Joint Limits",
    "num_contacts": "Active Contacts",
    "converged": "PADMM Convergence Flag",
    "num_restarts": "PADMM Acceleration Restarts",
    "r_dx": "PADMM Primal Iterate Residual",
    "r_dy": "PADMM Slack Iterate Residual",
    "r_dz": "PADMM Dual Iterate Residual",
    "r_a": "PADMM Combined Primal-Dual Residual (r_a)",
    "a": "Nesterov Acceleration Variable (a)",
    "a_factor": "Nesterov Acceleration Factor",
}


###
# Kernels
###


@wp.kernel
def _update_log_decision(
    max_frames: int32,
    decimation: int32,
    mode: int32,
    call_count: wp.array[int32],
    frames_total: wp.array[int32],
    decision: wp.array[int32],
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

    should_write = int32(1)
    if (cc % decimation) != int32(0):
        should_write = int32(0)
    if mode == int32(1) and ft >= max_frames:
        should_write = int32(0)

    decision[0] = should_write
    if should_write == int32(1):
        decision[1] = ft % max_frames
    else:
        decision[1] = int32(0)

    call_count[0] = cc + int32(1)


@wp.kernel
def _finalize_log_decision(
    decision: wp.array[int32],
    frames_total: wp.array[int32],
):
    """Increment ``frames_total`` if the current call actually wrote a frame.

    Args:
        decision: Two-element decision buffer ``[should_write, write_idx]``.
        frames_total: Single-element array tracking total successful writes.
    """
    if decision[0] == int32(1):
        frames_total[0] = frames_total[0] + int32(1)


@wp.kernel
def _copy_padmm_status_row(
    src: wp.array[PADMMStatus],
    decision: wp.array[int32],
    log_iterations: wp.array2d[int32],
    log_converged: wp.array2d[int32],
    log_num_restarts: wp.array2d[int32],
    log_r_p: wp.array2d[float32],
    log_r_d: wp.array2d[float32],
    log_r_c: wp.array2d[float32],
):
    """Copy one ``(num_worlds,)`` slice of :class:`PADMMStatus` into the logs.

    Args:
        src: The source array of solver status structs (one per world).
        decision: Two-element decision buffer ``[should_write, write_idx]``.
        log_iterations: Target log buffer for ``status.iterations``.
        log_converged: Target log buffer for ``status.converged``.
        log_num_restarts: Target log buffer for ``status.num_restarts``.
        log_r_p: Target log buffer for ``status.r_p``.
        log_r_d: Target log buffer for ``status.r_d``.
        log_r_c: Target log buffer for ``status.r_c``.
    """
    wid = wp.tid()
    if decision[0] == int32(0):
        return
    write_idx = decision[1]
    status = src[wid]
    log_iterations[write_idx, wid] = status.iterations
    log_converged[write_idx, wid] = status.converged
    log_num_restarts[write_idx, wid] = status.num_restarts
    log_r_p[write_idx, wid] = status.r_p
    log_r_d[write_idx, wid] = status.r_d
    log_r_c[write_idx, wid] = status.r_c


@wp.kernel
def _copy_padmm_penalty_row(
    src: wp.array[PADMMPenalty],
    decision: wp.array[int32],
    log_rho: wp.array2d[float32],
    log_num_rho_updates: wp.array2d[int32],
):
    """Copy one ``(num_worlds,)`` slice of :class:`PADMMPenalty` into the logs.

    Args:
        src: The source array of solver penalty structs (one per world).
        decision: Two-element decision buffer ``[should_write, write_idx]``.
        log_rho: Target log buffer for ``penalty.rho``.
        log_num_rho_updates: Target log buffer for ``penalty.num_updates``.
    """
    wid = wp.tid()
    if decision[0] == int32(0):
        return
    write_idx = decision[1]
    penalty = src[wid]
    log_rho[write_idx, wid] = penalty.rho
    log_num_rho_updates[write_idx, wid] = penalty.num_updates


@wp.kernel
def _copy_int32_row(
    src: wp.array[int32],
    decision: wp.array[int32],
    dest: wp.array2d[int32],
):
    """Copy one ``(num_worlds,)`` int32 source row into ``dest[write_idx, :]``.

    Args:
        src: The source per-world int32 array.
        decision: Two-element decision buffer ``[should_write, write_idx]``.
        dest: Target 2-D log buffer of shape ``(max_frames, num_worlds)``.
    """
    wid = wp.tid()
    if decision[0] == int32(0):
        return
    write_idx = decision[1]
    dest[write_idx, wid] = src[wid]


@wp.kernel
def _copy_padmm_iterate_residuals_row(
    src: wp.array[PADMMStatus],
    decision: wp.array[int32],
    log_r_dx: wp.array2d[float32],
    log_r_dy: wp.array2d[float32],
    log_r_dz: wp.array2d[float32],
):
    """Copy the per-iteration ``r_dx``/``r_dy``/``r_dz`` fields of :class:`PADMMStatus`.

    Args:
        src: The source array of solver status structs (one per world).
        decision: Two-element decision buffer ``[should_write, write_idx]``.
        log_r_dx: Target log buffer for ``status.r_dx``.
        log_r_dy: Target log buffer for ``status.r_dy``.
        log_r_dz: Target log buffer for ``status.r_dz``.
    """
    wid = wp.tid()
    if decision[0] == int32(0):
        return
    write_idx = decision[1]
    status = src[wid]
    log_r_dx[write_idx, wid] = status.r_dx
    log_r_dy[write_idx, wid] = status.r_dy
    log_r_dz[write_idx, wid] = status.r_dz


@wp.kernel
def _copy_padmm_acceleration_row(
    src_status: wp.array[PADMMStatus],
    src_a: wp.array[float32],
    src_a_factor: wp.array[float32],
    decision: wp.array[int32],
    log_r_a: wp.array2d[float32],
    log_a: wp.array2d[float32],
    log_a_factor: wp.array2d[float32],
):
    """Copy the Nesterov-acceleration state into the logs.

    Args:
        src_status: The source array of solver status structs (one per world).
        src_a: Per-world Nesterov acceleration variable from :class:`PADMMState`.
        src_a_factor: Per-world Nesterov acceleration factor from :class:`PADMMState`.
        decision: Two-element decision buffer ``[should_write, write_idx]``.
        log_r_a: Target log buffer for ``status.r_a``.
        log_a: Target log buffer for ``state.a``.
        log_a_factor: Target log buffer for ``state.a_factor``.
    """
    wid = wp.tid()
    if decision[0] == int32(0):
        return
    write_idx = decision[1]
    log_r_a[write_idx, wid] = src_status[wid].r_a
    log_a[write_idx, wid] = src_a[wid]
    log_a_factor[write_idx, wid] = src_a_factor[wid]


###
# Interfaces
###


class SolverKaminoLogger:
    """
    Records :class:`SolverKamino` PADMM solver status on the solver's device.

    The logger allocates one Warp 2-D buffer of shape
    ``(max_frames, num_worlds)`` per recorded metric on the same device as the
    underlying :class:`SolverKamino`. Each call to :meth:`log` appends the
    current per-world solver-status values into the next slot of the rolling
    window.

    The buffer-overflow policy is controlled by :class:`Mode`:

    - :attr:`Mode.ROLLING` wraps the write index modulo ``max_frames``, so
      the buffer always holds the most recent ``max_frames`` samples.
    - :attr:`Mode.BOUNDED` stops recording once ``max_frames`` samples have
      been logged; subsequent :meth:`log` calls write nothing.

    The optional ``decimation`` argument skips intermediate calls so only every
    ``decimation``-th call actually writes a new frame.

    Unlike :class:`SolutionMetricsLogger`, every host-side decision in
    :meth:`log` (decimation gate, overflow check, write-index computation,
    counter increments) is performed on-device through dedicated Warp kernels.
    This makes a single :meth:`log` invocation a fixed sequence of kernel
    launches whose data dependencies live entirely in device memory, and it
    can be safely included inside :class:`wp.ScopedCapture` blocks alongside
    :meth:`SolverKamino.step`.

    The recorded metrics are:

    - ``iterations``: PADMM iterations performed during the last solve.
    - ``converged``: ``1`` if the solver converged, ``0`` otherwise.
    - ``num_restarts``: Number of Nesterov acceleration restarts.
    - ``r_p``: Final PADMM primal residual.
    - ``r_d``: Final PADMM dual residual.
    - ``r_c``: Final PADMM complementarity residual.
    - ``rho``: Final ALM penalty parameter.
    - ``num_rho_updates``: Number of penalty updates during the solve.
    - ``num_limits``: Active joint-limit constraints per world.
    - ``num_contacts``: Active contact constraints per world.

    Two opt-in flags add advanced PADMM diagnostics without changing the
    :class:`SolverKamino` front end:

    - ``with_iterate_residuals_info=True`` adds ``r_dx``, ``r_dy``, ``r_dz``
      (the primal/slack/dual iterate residuals) as plotted panels.
    - ``with_acceleration_info=True`` adds ``r_a`` (combined primal-dual
      residual), ``a`` (the per-world Nesterov acceleration variable),
      ``a_factor`` (the per-world Nesterov factor), and ``num_restarts``
      (the per-world acceleration-restart count, which is always logged but
      only surfaced on the figure under this flag). Requires the underlying
      ``PADMMSolver`` to have been built with ``use_acceleration=True``.

    The plot grid auto-sizes at construction time so all enabled metrics
    are laid out on one figure.
    """

    class Mode(IntEnum):
        """Buffer overflow behavior for :class:`SolverKaminoLogger`."""

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
        solver: SolverKamino,
        max_frames: int,
        mode: Mode = Mode.BOUNDED,
        decimation: int = 1,
        dt: float | None = None,
        with_iterate_residuals_info: bool = False,
        with_acceleration_info: bool = False,
    ):
        """
        Initializes the :class:`SolverKamino` status logger.

        Args:
            solver: The :class:`SolverKamino` instance to record from. Must
                already be constructed; its internal Kamino containers
                (``_solver_kamino``, ``_contacts_kamino``) are used to source
                the logged metrics.
            max_frames: The maximum number of frames the log buffers can hold.
                Must be a strictly positive integer.
            mode: The buffer-overflow policy. Defaults to :attr:`Mode.BOUNDED`.
            decimation: Sample decimation rate. Only every ``decimation``-th
                :meth:`log` call writes a new frame. Must be a strictly
                positive integer. Defaults to ``1`` (no decimation).
            dt: Optional simulation time step [s] used to scale the time axis
                on plots. If a positive value is supplied it is pinned for the
                lifetime of the logger. If ``None``, the time step is read
                live from ``solver._solver_kamino._model.time.dt[0]`` on every
                access; non-positive or unreadable values fall back to a
                unit-less ``"Step"`` axis label.
            with_iterate_residuals_info: If ``True`` additionally allocates
                and records the PADMM primal/slack/dual iterate residuals
                ``r_dx``, ``r_dy`` and ``r_dz``. Defaults to ``False``.
            with_acceleration_info: If ``True`` additionally allocates and
                records the Nesterov-acceleration diagnostics ``r_a`` (the
                combined primal-dual residual used for restart checks), ``a``
                (the per-world acceleration variable) and ``a_factor`` (the
                per-world Nesterov factor). Requires the underlying
                ``PADMMSolver`` to have been built with
                ``use_acceleration=True``. Defaults to ``False``.
        """
        # Import here to avoid module-level imports and circular dependencies.
        from ..solver_kamino import SolverKamino  # noqa: PLC0415

        if not isinstance(solver, SolverKamino):
            raise TypeError(f"Expected 'solver' to be a `SolverKamino` instance, got {type(solver)}.")
        if not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError(f"Expected 'max_frames' to be a positive integer, got {max_frames!r}.")
        if not isinstance(decimation, int) or decimation <= 0:
            raise ValueError(f"Expected 'decimation' to be a positive integer, got {decimation!r}.")
        if not isinstance(mode, SolverKaminoLogger.Mode):
            raise TypeError("Expected 'mode' to be a `SolverKaminoLogger.Mode` value.")
        if not isinstance(with_iterate_residuals_info, bool):
            raise TypeError(
                f"Expected 'with_iterate_residuals_info' to be a bool, got {type(with_iterate_residuals_info)}."
            )
        if not isinstance(with_acceleration_info, bool):
            raise TypeError(f"Expected 'with_acceleration_info' to be a bool, got {type(with_acceleration_info)}.")

        # Attempt to initialize matplotlib for plotting
        self._initialize_plt()

        # Cache references to the internal Kamino containers from which solver
        # status, penalty, limits, and contacts data are sourced at log time.
        impl = solver._solver_kamino
        self._solver: SolverKamino = solver
        self._status_src: wp.array = impl.solver_fd.data.status
        self._penalty_src: wp.array = impl.solver_fd.data.penalty
        self._limits = impl._limits
        self._contacts = solver._contacts_kamino

        # When acceleration info is requested, fail fast if the underlying
        # PADMM solver was built without acceleration: the ``a`` / ``a_factor``
        # arrays are only allocated in that mode.
        self._with_iterate_residuals_info: bool = with_iterate_residuals_info
        self._with_acceleration_info: bool = with_acceleration_info
        if self._with_acceleration_info:
            state = impl.solver_fd.data.state
            if state.a is None or state.a_factor is None:
                raise ValueError(
                    "Cannot enable 'with_acceleration_info=True' because the underlying PADMMSolver was "
                    "constructed with 'use_acceleration=False'. Enable acceleration in the solver config "
                    "(e.g. 'solver_config.padmm.use_acceleration = True') or disable this logger option."
                )
            self._state_a_src: wp.array = state.a
            self._state_a_factor_src: wp.array = state.a_factor

        # Build the active scalar / plotted field tuples for this instance.
        # The basic fields are always present; the optional extension fields
        # are appended in stable order so that ``to_numpy``/``reset``/``plot``
        # iterate over a deterministic sequence.
        float32_fields = list(_SCALAR_METRIC_FIELDS_FLOAT32)
        plotted_fields = list(_PLOTTED_METRIC_FIELDS)
        if self._with_iterate_residuals_info:
            float32_fields.extend(_SCALAR_METRIC_FIELDS_FLOAT32_ITERATE)
            plotted_fields.extend(_PLOTTED_METRIC_FIELDS_ITERATE)
        if self._with_acceleration_info:
            float32_fields.extend(_SCALAR_METRIC_FIELDS_FLOAT32_ACCEL)
            plotted_fields.extend(_PLOTTED_METRIC_FIELDS_ACCEL)
        self._scalar_fields_float32: tuple[str, ...] = tuple(float32_fields)
        self._scalar_fields_int32: tuple[str, ...] = tuple(_SCALAR_METRIC_FIELDS_INT32)
        self._scalar_fields_all: tuple[str, ...] = self._scalar_fields_float32 + self._scalar_fields_int32
        self._plotted_fields: tuple[str, ...] = tuple(plotted_fields)

        # Auto-size the plot grid so all enabled metrics fit on one figure.
        # We cap the number of columns at four to match the prior 2x4 layout
        # and to keep panels readable at typical figure widths.
        n_plots = len(self._plotted_fields)
        self._plot_cols: int = min(4, n_plots)
        self._plot_rows: int = (n_plots + self._plot_cols - 1) // self._plot_cols

        # Cache host-side configurations and shapes
        self._max_frames: int = int(max_frames)
        self._mode: SolverKaminoLogger.Mode = mode
        self._decimation: int = int(decimation)
        self._device: wp.DeviceLike = impl.device
        self._num_worlds: int = int(impl._model.size.num_worlds)

        # Resolve the simulation time step. An explicit positive ``dt`` is pinned
        # for the lifetime of the logger; otherwise the value is read live from
        # ``solver._solver_kamino._model.time.dt[0]`` on every access so that
        # updates the solver makes after logger construction are reflected here.
        if dt is not None:
            if not isinstance(dt, (int, float)) or float(dt) <= 0.0:
                raise ValueError(f"Expected 'dt' to be a positive number, got {dt!r}.")
            self._dt_override: float | None = float(dt)
        else:
            self._dt_override = None

        # Capture host-side capacities to gate optional copy launches in
        # contact- or limit-free models without breaking graph capture.
        self._has_limits: bool = self._limits is not None and int(self._limits.model_max_limits_host) > 0
        self._has_contacts: bool = self._contacts is not None and int(self._contacts.model_max_contacts_host) > 0

        # Allocate every per-frame log buffer on the solver device. The 2-D layout
        # ``(max_frames, num_worlds)`` matches the per-world scalar fan-out of the
        # underlying solver fields.
        shape = (self._max_frames, self._num_worlds)
        with wp.ScopedDevice(self._device):
            for field in self._scalar_fields_float32:
                setattr(self, f"log_{field}", wp.zeros(shape=shape, dtype=float32))
            for field in self._scalar_fields_int32:
                setattr(self, f"log_{field}", wp.zeros(shape=shape, dtype=int32))

            # Device-side counters and per-call decision scratch buffer.
            self._call_count: wp.array = wp.zeros(shape=1, dtype=int32)
            self._frames_total: wp.array = wp.zeros(shape=1, dtype=int32)
            self._decision: wp.array = wp.zeros(shape=2, dtype=int32)

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
        """Returns the resolved simulation time step [s], or ``None`` if unavailable.

        If an explicit positive ``dt`` was provided to the constructor it is
        returned as-is; otherwise the value is read live from
        ``solver._solver_kamino._model.time.dt[0]`` so updates made by the
        solver after the logger was constructed are reflected here.
        Non-positive (or unreadable) values yield ``None``.
        """
        return self._resolve_dt()

    @property
    def num_total_writes(self) -> int:
        """Returns the cumulative number of writes (including those overwritten in rolling mode)."""
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
        for field in self._scalar_fields_all:
            getattr(self, f"log_{field}").zero_()

    def log(self):
        """Records the current solver-status values into the next buffer slot.

        Every invocation expands to the same fixed sequence of Warp kernel
        launches, so the call can be safely captured into a CUDA graph along
        with :meth:`SolverKamino.step`. Decimation skips and bounded-mode
        overflow are enforced on-device.
        """
        # Compute the per-call write decision (should_write, write_idx) and
        # increment the call counter on the device.
        wp.launch(
            kernel=_update_log_decision,
            dim=1,
            inputs=[
                int32(self._max_frames),
                int32(self._decimation),
                int32(int(self._mode)),
                self._call_count,
                self._frames_total,
                self._decision,
            ],
            device=self._device,
        )

        # Copy the PADMM solver status fields into their log buffers.
        wp.launch(
            kernel=_copy_padmm_status_row,
            dim=self._num_worlds,
            inputs=[
                self._status_src,
                self._decision,
                self.log_iterations,
                self.log_converged,
                self.log_num_restarts,
                self.log_r_p,
                self.log_r_d,
                self.log_r_c,
            ],
            device=self._device,
        )

        # Copy the PADMM penalty fields into their log buffers.
        wp.launch(
            kernel=_copy_padmm_penalty_row,
            dim=self._num_worlds,
            inputs=[
                self._penalty_src,
                self._decision,
                self.log_rho,
                self.log_num_rho_updates,
            ],
            device=self._device,
        )

        # Copy active-limit counts when the model has limit capacity.
        if self._has_limits:
            wp.launch(
                kernel=_copy_int32_row,
                dim=self._num_worlds,
                inputs=[
                    self._limits.world_active_limits,
                    self._decision,
                    self.log_num_limits,
                ],
                device=self._device,
            )

        # Copy active-contact counts when the model has contact capacity.
        if self._has_contacts:
            wp.launch(
                kernel=_copy_int32_row,
                dim=self._num_worlds,
                inputs=[
                    self._contacts.world_active_contacts,
                    self._decision,
                    self.log_num_contacts,
                ],
                device=self._device,
            )

        # Copy the optional PADMM iterate residual fields when enabled.
        if self._with_iterate_residuals_info:
            wp.launch(
                kernel=_copy_padmm_iterate_residuals_row,
                dim=self._num_worlds,
                inputs=[
                    self._status_src,
                    self._decision,
                    self.log_r_dx,
                    self.log_r_dy,
                    self.log_r_dz,
                ],
                device=self._device,
            )

        # Copy the optional Nesterov acceleration diagnostics when enabled.
        if self._with_acceleration_info:
            wp.launch(
                kernel=_copy_padmm_acceleration_row,
                dim=self._num_worlds,
                inputs=[
                    self._status_src,
                    self._state_a_src,
                    self._state_a_factor_src,
                    self._decision,
                    self.log_r_a,
                    self.log_a,
                    self.log_a_factor,
                ],
                device=self._device,
            )

        # Increment ``frames_total`` on-device if a write was made.
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
        metric name.

        In :attr:`Mode.ROLLING` the rows are rotated so that the oldest
        recorded frame is at index ``0`` and the most recent at index
        ``num_logged_frames - 1``.

        Returns:
            A dictionary mapping field name to its recorded values.
        """
        total = self.num_total_writes
        n = min(total, self._max_frames)
        result: dict[str, np.ndarray] = {}
        for field in self._scalar_fields_all:
            buf = getattr(self, f"log_{field}").numpy()
            if n == 0:
                result[field] = buf[:0].copy()
                continue
            if self._mode == SolverKaminoLogger.Mode.ROLLING and total > self._max_frames:
                # The buffer wrapped around at least once; rotate so that the
                # oldest recorded frame is at index 0.
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
        dt = self._resolve_dt()
        scale = (dt if dt is not None else 1.0) * float(self._decimation)
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
    ):
        """
        Renders a single matplotlib figure with one subplot per metric.

        The figure is laid out on an auto-sized grid sized at construction
        time from the enabled options: the base set of metrics
        (``iterations``, the three residuals, ``rho``, penalty updates,
        active limits, active contacts) is always included, and additional
        rows are added when ``with_iterate_residuals_info`` or
        ``with_acceleration_info`` are enabled. Each subplot draws one
        curve per world against :meth:`time_axis`.

        Args:
            filename: Optional base filename. When ``path`` is also provided,
                the figure is saved as ``{path}/{filename}.{ext}``. If
                ``filename`` is ``None``, defaults to ``"solver_status"``.
            path: Optional output directory. The directory must already exist.
                If ``None``, the figure is not saved.
            show: If ``True`` the figure is also displayed (blocking).
            ext: The file extension / matplotlib format to save with.
                Defaults to ``"pdf"``.
        """
        if self.plt is None:
            msg.warning("matplotlib is not available, skipping plotting.")
            return
        if self.num_logged_frames == 0:
            msg.warning("No logged frames to plot, skipping plotting.")
            return
        if path is not None and not os.path.isdir(path):
            raise ValueError(f"Plot output directory '{path}' does not exist. Please create it before calling plot().")

        plt = self.plt
        if filename is None:
            filename = "solver_status"

        time = self.time_axis()
        np_data = self.to_numpy()
        x_label = "Time (s)" if self._resolve_dt() is not None else "Step"

        n_rows, n_cols = self._plot_rows, self._plot_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()
        for i, field in enumerate(self._plotted_fields):
            ax = axes_flat[i]
            data = np_data[field]
            for w in range(self._num_worlds):
                ax.plot(
                    time,
                    data[:, w],
                    label=f"world_{w}",
                    marker="o",
                    markersize=3,
                    linestyle="-",
                )
            ax.set_title(_METRIC_TITLES.get(field, field))
            ax.set_xlabel(x_label)
            ax.set_ylabel(field)
            ax.grid()
            if self._num_worlds > 1:
                ax.legend(loc="best", frameon=False)
        # Hide any unused cells.
        for j in range(len(self._plotted_fields), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        if path is not None:
            fig.savefig(
                os.path.join(path, f"{filename}.{ext}"),
                format=ext,
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close(fig)

    @classmethod
    def plot_comparison(
        cls,
        loggers: dict[str, SolverKaminoLogger],
        filename: str | None = None,
        path: str | None = None,
        show: bool = False,
        ext: str = "pdf",
    ):
        """
        Renders overlaid :class:`SolverKaminoLogger` plots across multiple logger instances.

        Iterates the plotted metric fields and draws one curve per world per
        setup on a shared axis, cycling through a fixed color palette. The
        figure layout mirrors :meth:`plot`: the grid is auto-sized from the
        first logger's enabled options, and only fields exposed by every
        logger are drawn.

        Args:
            loggers: A dictionary of logger instances keyed by setup name.
            filename: Optional base filename. When ``path`` is also provided,
                the figure is saved as ``{path}/{filename}.{ext}``. If
                ``filename`` is ``None``, defaults to ``"solver_status"``.
            path: Optional output directory. The directory must already exist.
            show: If ``True`` the figure is also displayed (blocking).
            ext: The file extension / matplotlib format to save with.
                Defaults to ``"pdf"``.
        """
        # Attempt to initialize matplotlib for plotting
        if cls.plt is None:
            cls._initialize_plt()
        if cls.plt is None:
            msg.critical("matplotlib is not available, skipping plotting.")
            return

        if not all(isinstance(logger, SolverKaminoLogger) for logger in loggers.values()):
            raise ValueError("All loggers must be instances of SolverKaminoLogger.")

        if not any(logger.num_logged_frames > 0 for logger in loggers.values()):
            msg.warning("No logged frames to plot, skipping plotting.")
            return

        first = next(iter(loggers.values()))
        if not all(logger.num_worlds == first.num_worlds for logger in loggers.values()):
            raise ValueError("All loggers must have the same number of worlds.")

        # Only overlay metrics that every logger has actually recorded so
        # callers can mix loggers with different optional fields enabled.
        common_fields = tuple(
            field
            for field in first._plotted_fields
            if all(field in logger._plotted_fields for logger in loggers.values())
        )

        if path is not None and not os.path.isdir(path):
            raise ValueError(
                f"Plot output directory '{path}' does not exist. Please create it before calling plot_comparison()."
            )

        plt = cls.plt
        if filename is None:
            filename = "solver_status"

        x_label = "Time (s)" if first.dt is not None else "Step"
        logged_data = [
            (name, logger.num_worlds, logger.time_axis(), logger.to_numpy()) for name, logger in loggers.items()
        ]

        n_plots = len(common_fields)
        n_cols = min(4, n_plots) if n_plots > 0 else 1
        n_rows = (n_plots + n_cols - 1) // n_cols if n_plots > 0 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()
        for i, field in enumerate(common_fields):
            cls._plot_overlay_metric(logged_data, field, x_label, axes_flat[i])
        for j in range(len(common_fields), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        if path is not None:
            fig.savefig(
                os.path.join(path, f"{filename}.{ext}"),
                format=ext,
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close(fig)

    ###
    # Internals
    ###

    def _resolve_dt(self) -> float | None:
        """Resolve the current effective time step.

        See :attr:`dt` for the resolution policy.
        """
        if self._dt_override is not None:
            return self._dt_override
        try:
            value = float(self._solver._solver_kamino._model.time.dt.numpy()[0])
        except Exception:
            return None
        return value if value > 0.0 else None

    @staticmethod
    def _plot_overlay_metric(
        data: list[tuple[str, int, np.ndarray, dict[str, np.ndarray]]],
        field: str,
        x_label: str,
        ax: plt.Axes,
    ):
        """Draws one overlaid metric panel onto ``ax`` for the given ``field``.

        Args:
            data: A list of ``(name, nw, time, np_data)`` tuples where
                ``time`` and ``np_data`` are pre-computed via
                :meth:`SolverKaminoLogger.time_axis` and
                :meth:`SolverKaminoLogger.to_numpy`.
            field: The name of the metric to plot.
            x_label: The x-axis label.
            ax: The matplotlib axes to draw on.
        """
        overlay_colors = ("purple", "blue", "red", "green", "orange", "cyan", "brown", "magenta")
        for i, (name, nw, time, np_data) in enumerate(data):
            color = overlay_colors[i % len(overlay_colors)]
            for w in range(nw):
                world_label = f" (world_{w})" if nw > 1 else ""
                ax.plot(
                    time,
                    np_data[field][:, w],
                    color=color,
                    marker="o",
                    markersize=3,
                    linestyle="-",
                    label=f"{name}{world_label}",
                )
        ax.set_title(_METRIC_TITLES.get(field, field))
        ax.set_xlabel(x_label)
        ax.set_ylabel(field)
        ax.grid()
        ax.legend(loc="best", frameon=False)
