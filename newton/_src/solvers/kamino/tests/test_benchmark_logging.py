# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for :class:`PhysicsMetricsLogger`.

The tests build a small two-world model through :mod:`newton.tests.utils.basics`,
construct a :class:`PhysicsMetrics` container, and drive the logger through
``log()`` calls. To keep the tests independent of the solver / collider,
synthetic values are written directly into
``metrics.per_world_contacts_summary`` between successive ``log()`` calls so
each test owns its expected history exactly.

Coverage:

* Constructor argument validation (types, ranges, missing per-world summary).
* Buffer shapes, ``num_worlds``, ``num_logged_frames`` semantics.
* Bounded vs. rolling overflow policies.
* Decimation sub-sampling.
* ``reset()`` clears every buffer + counter.
* CUDA-graph capture of ``log()`` (skipped on CPU).
* ``plot()`` smoke test (writes a non-empty file to disk).
* Argmax companion fields are logged but never plotted.
"""

from __future__ import annotations

import csv
import importlib.util
import unittest

import numpy as np
import warp as wp

from newton import Contacts, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.benchmark import (
    PhysicsMetrics,
    PhysicsMetricsLogger,
)
from newton._src.solvers.kamino.benchmark.logging import (
    _METRIC_EQUATIONS,
    _METRIC_TITLES,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils import basics

_RICH_AVAILABLE = importlib.util.find_spec("rich") is not None

###
# Constants
###

_SCALAR_FIELDS = (
    "r_cts_penetration",
    "r_cts_velocity",
    "r_ncp_primal",
    "r_ncp_dual",
    "r_ncp_compl",
    "r_vi_natmap",
)
_ARGMAX_FIELDS = tuple(f + "_argmax" for f in _SCALAR_FIELDS)


###
# Scaffolding
###


class _LoggerTestSetup:
    """Builds a multi-world :class:`PhysicsMetrics` + :class:`PhysicsMetricsLogger`.

    The model is a heterogeneous two-world scene (sphere-on-plane in world 0
    and box-on-plane in world 1) so the per-world summary buffers carry
    independent residuals per world.
    """

    def __init__(
        self,
        max_frames: int,
        mode: PhysicsMetricsLogger.Mode = PhysicsMetricsLogger.Mode.BOUNDED,
        decimation: int = 1,
        num_worlds: int = 2,
        max_contacts: int = 16,
        dt: float | None = None,
        device: wp.DeviceLike | None = None,
    ):
        self.max_frames = max_frames
        self.num_worlds = int(num_worlds)
        self.dt = dt
        self.device = device

        self.builder: ModelBuilder = ModelBuilder()
        self.builder.request_contact_attributes("force", "velocity")
        for w in range(self.num_worlds):
            if w == 0:
                basics.build_sphere_on_plane(
                    builder=self.builder,
                    radius=0.1,
                    mass=1.0,
                    z_offset=-1.0e-3,
                    friction=0.5,
                    new_world=True,
                    ground=True,
                )
            else:
                basics.build_box_on_plane(
                    builder=self.builder,
                    z_offset=-1.0e-3,
                    friction=0.5,
                    new_world=True,
                    ground=True,
                )

        self.model: Model = self.builder.finalize(device=device)
        self.model.rigid_contact_max = max_contacts
        self.state: State = self.model.state()
        self.contacts: Contacts = self.model.contacts()

        self.metrics: PhysicsMetrics = PhysicsMetrics(model=self.model)
        self.logger: PhysicsMetricsLogger = PhysicsMetricsLogger(
            metrics=self.metrics,
            max_frames=max_frames,
            mode=mode,
            decimation=decimation,
            dt=dt,
        )

    def assign_per_world(self, field: str, values: np.ndarray) -> None:
        """Writes ``values`` (shape ``(num_worlds,)``) into the per-world summary array."""
        arr = getattr(self.metrics.per_world_contacts_summary, field)
        arr.assign(values.astype(arr.numpy().dtype, copy=False))

    def assign_all(
        self,
        *,
        floats: dict[str, np.ndarray] | None = None,
        argmax: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Bulk-assigns float and int per-world summary fields."""
        if floats:
            for field, vals in floats.items():
                self.assign_per_world(field, vals)
        if argmax:
            for field, vals in argmax.items():
                self.assign_per_world(field, vals)


def _populate_uniform_floats(setup: _LoggerTestSetup, scalar: float) -> None:
    """Writes ``scalar`` into every per-world float summary field."""
    for field in _SCALAR_FIELDS:
        setup.assign_per_world(field, np.full(setup.num_worlds, scalar, dtype=np.float32))


def _populate_per_world_floats(setup: _LoggerTestSetup, values: np.ndarray) -> None:
    """Writes ``values`` (shape ``(num_worlds,)``) into every per-world float summary field."""
    arr = values.astype(np.float32, copy=False)
    for field in _SCALAR_FIELDS:
        setup.assign_per_world(field, arr)


###
# Tests
###


class TestPhysicsMetricsLogger(unittest.TestCase):
    def setUp(self) -> None:
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.output_path = test_context.output_path / "test_benchmark_logging"
        self.output_path.mkdir(parents=True, exist_ok=True)
        msg.reset_log_level()

    ###
    # Constructor / configuration
    ###

    def test_constructor_validation(self):
        """Invalid args raise; valid args expose correctly-shaped buffers."""
        setup = _LoggerTestSetup(max_frames=8, num_worlds=2, device=self.default_device)

        # Invalid 'metrics' type
        with self.assertRaises(TypeError):
            PhysicsMetricsLogger(metrics="not_metrics", max_frames=8)
        # Missing per-world summary container
        bare_metrics = PhysicsMetrics()
        with self.assertRaises(ValueError):
            PhysicsMetricsLogger(metrics=bare_metrics, max_frames=8)
        # Invalid 'max_frames'
        with self.assertRaises(ValueError):
            PhysicsMetricsLogger(metrics=setup.metrics, max_frames=0)
        with self.assertRaises(ValueError):
            PhysicsMetricsLogger(metrics=setup.metrics, max_frames=-1)
        # Invalid 'decimation'
        with self.assertRaises(ValueError):
            PhysicsMetricsLogger(metrics=setup.metrics, max_frames=8, decimation=0)
        # Invalid 'mode'
        with self.assertRaises(TypeError):
            PhysicsMetricsLogger(metrics=setup.metrics, max_frames=8, mode="rolling")
        # Invalid 'dt'
        with self.assertRaises(ValueError):
            PhysicsMetricsLogger(metrics=setup.metrics, max_frames=8, dt=-0.1)

        # The actual logger created in setup must expose correct shapes and dtypes.
        self.assertEqual(setup.logger.max_frames, 8)
        self.assertEqual(setup.logger.num_worlds, setup.num_worlds)
        self.assertEqual(setup.logger.mode, PhysicsMetricsLogger.Mode.BOUNDED)
        self.assertEqual(setup.logger.decimation, 1)
        self.assertEqual(setup.logger.num_logged_frames, 0)
        self.assertEqual(setup.logger.num_total_writes, 0)
        self.assertFalse(setup.logger.is_full)

        expected_shape = (setup.logger.max_frames, setup.logger.num_worlds)
        for field in _SCALAR_FIELDS:
            buf = getattr(setup.logger, f"log_{field}")
            self.assertEqual(buf.shape, expected_shape)
            self.assertEqual(buf.dtype, wp.float32)
        for field in _ARGMAX_FIELDS:
            buf = getattr(setup.logger, f"log_{field}")
            self.assertEqual(buf.shape, expected_shape)
            self.assertEqual(buf.dtype, wp.int32)
            np.testing.assert_array_equal(buf.numpy(), -np.ones(expected_shape, dtype=np.int32))

    ###
    # Buffer-overflow semantics
    ###

    def test_bounded_overflow_stops_logging(self):
        """``Mode.BOUNDED`` stops writing once ``max_frames`` is reached."""
        n = 4
        setup = _LoggerTestSetup(max_frames=n, mode=PhysicsMetricsLogger.Mode.BOUNDED, device=self.default_device)

        for k in range(2 * n):
            _populate_uniform_floats(setup, scalar=float(k))
            setup.logger.log()

        self.assertEqual(setup.logger.num_logged_frames, n)
        self.assertEqual(setup.logger.num_total_writes, n)
        self.assertTrue(setup.logger.is_full)
        self.assertEqual(setup.logger.num_calls, 2 * n)

        np_data = setup.logger.to_numpy()
        for field in _SCALAR_FIELDS:
            self.assertEqual(np_data[field].shape, (n, setup.num_worlds))
            # The first n frames should contain values [0..n-1] broadcast across
            # each world; later frames are dropped.
            for k in range(n):
                np.testing.assert_allclose(
                    np_data[field][k],
                    np.full(setup.num_worlds, float(k), dtype=np.float32),
                    atol=0.0,
                )

    def test_rolling_wraparound_returns_chronological_order(self):
        """``Mode.ROLLING`` keeps the last ``max_frames`` samples in chronological order."""
        n = 5
        extra = 3
        total_steps = 2 * n + extra
        setup = _LoggerTestSetup(
            max_frames=n,
            mode=PhysicsMetricsLogger.Mode.ROLLING,
            device=self.default_device,
        )

        for k in range(total_steps):
            _populate_uniform_floats(setup, scalar=float(k))
            setup.logger.log()

        self.assertEqual(setup.logger.num_logged_frames, n)
        self.assertEqual(setup.logger.num_total_writes, total_steps)
        self.assertTrue(setup.logger.is_full)

        np_data = setup.logger.to_numpy()
        for field in _SCALAR_FIELDS:
            expected = np.tile(
                np.arange(total_steps - n, total_steps, dtype=np.float32)[:, None],
                (1, setup.num_worlds),
            )
            np.testing.assert_allclose(np_data[field], expected, atol=0.0)

    def test_decimation_skips_frames(self):
        """``decimation=k`` writes only every k-th call; intermediate calls are no-ops."""
        decimation = 3
        num_calls = 10
        setup = _LoggerTestSetup(
            max_frames=16,
            mode=PhysicsMetricsLogger.Mode.BOUNDED,
            decimation=decimation,
            device=self.default_device,
        )

        argmax_history = np.zeros((num_calls, setup.num_worlds), dtype=np.int32)
        for k in range(num_calls):
            _populate_uniform_floats(setup, scalar=float(k))
            argmax_value = np.array([k * 2, k * 2 + 1], dtype=np.int32)[: setup.num_worlds]
            for field in _ARGMAX_FIELDS:
                setup.assign_per_world(field, argmax_value)
            argmax_history[k] = argmax_value
            setup.logger.log()

        expected_writes = (num_calls + decimation - 1) // decimation
        self.assertEqual(setup.logger.num_logged_frames, expected_writes)
        self.assertEqual(setup.logger.num_total_writes, expected_writes)
        self.assertEqual(setup.logger.num_calls, num_calls)

        np_data = setup.logger.to_numpy()
        expected_floats = np.arange(0, num_calls, decimation, dtype=np.float32)
        for field in _SCALAR_FIELDS:
            np.testing.assert_allclose(
                np_data[field][:, 0],
                expected_floats,
                atol=0.0,
                err_msg=f"decimated logger retained the wrong float history for {field}",
            )
        for field in _ARGMAX_FIELDS:
            np.testing.assert_array_equal(
                np_data[field],
                argmax_history[::decimation],
                err_msg=f"decimated logger retained the wrong argmax history for {field}",
            )

    ###
    # Reset
    ###

    def test_reset_clears_state(self):
        """``reset()`` restores counters to zero and clears every log buffer."""
        setup = _LoggerTestSetup(max_frames=6, device=self.default_device)
        for k in range(3):
            _populate_uniform_floats(setup, scalar=float(k))
            for field in _ARGMAX_FIELDS:
                setup.assign_per_world(field, np.full(setup.num_worlds, k, dtype=np.int32))
            setup.logger.log()
        self.assertEqual(setup.logger.num_logged_frames, 3)

        setup.logger.reset()
        self.assertEqual(setup.logger.num_logged_frames, 0)
        self.assertEqual(setup.logger.num_total_writes, 0)
        self.assertEqual(setup.logger.num_calls, 0)

        for field in _SCALAR_FIELDS:
            buf = getattr(setup.logger, f"log_{field}").numpy()
            np.testing.assert_array_equal(buf, np.zeros_like(buf), err_msg=f"log_{field} not zeroed by reset()")
        for field in _ARGMAX_FIELDS:
            buf = getattr(setup.logger, f"log_{field}").numpy()
            np.testing.assert_array_equal(buf, -np.ones_like(buf), err_msg=f"log_{field} not reset to -1 by reset()")

    ###
    # CUDA-graph capture
    ###

    def test_graph_capture_log_only(self):
        """``logger.log()`` can be captured into a CUDA graph and replayed."""
        if not self.default_device.is_cuda:
            self.skipTest("Graph capture requires a CUDA device.")

        setup = _LoggerTestSetup(max_frames=16, device=self.default_device)
        reference_scalar = 1.25
        reference_argmax = np.array([7, 11], dtype=np.int32)[: setup.num_worlds]
        _populate_uniform_floats(setup, scalar=reference_scalar)
        for field in _ARGMAX_FIELDS:
            setup.assign_per_world(field, reference_argmax)

        replay_count = 7
        with wp.ScopedCapture(device=setup.logger.device) as capture:
            setup.logger.log()
        graph = capture.graph

        for _ in range(replay_count):
            wp.capture_launch(graph)
        wp.synchronize_device(setup.logger.device)

        self.assertEqual(setup.logger.num_logged_frames, replay_count)
        self.assertEqual(setup.logger.num_total_writes, replay_count)
        self.assertEqual(setup.logger.num_calls, replay_count)

        np_data = setup.logger.to_numpy()
        for k in range(replay_count):
            for field in _SCALAR_FIELDS:
                np.testing.assert_allclose(
                    np_data[field][k],
                    np.full(setup.num_worlds, reference_scalar, dtype=np.float32),
                    atol=0.0,
                    err_msg=f"captured logger row {k} has unexpected {field} value",
                )
            for field in _ARGMAX_FIELDS:
                np.testing.assert_array_equal(
                    np_data[field][k],
                    reference_argmax,
                    err_msg=f"captured logger row {k} has unexpected {field} value",
                )

    def test_graph_capture_bounded_overflow(self):
        """Bounded-mode overflow is enforced inside graph capture."""
        if not self.default_device.is_cuda:
            self.skipTest("Graph capture requires a CUDA device.")

        max_frames = 4
        replay_count = 10
        setup = _LoggerTestSetup(
            max_frames=max_frames,
            mode=PhysicsMetricsLogger.Mode.BOUNDED,
            device=self.default_device,
        )
        _populate_uniform_floats(setup, scalar=0.5)

        with wp.ScopedCapture(device=setup.logger.device) as capture:
            setup.logger.log()
        graph = capture.graph

        for _ in range(replay_count):
            wp.capture_launch(graph)
        wp.synchronize_device(setup.logger.device)

        self.assertEqual(setup.logger.num_logged_frames, max_frames)
        self.assertEqual(setup.logger.num_total_writes, max_frames)
        self.assertEqual(setup.logger.num_calls, replay_count)
        self.assertTrue(setup.logger.is_full)

    ###
    # Plotting
    ###

    def test_plot_smoke(self):
        """``plot()`` writes a non-empty figure per scalar metric to disk."""
        if PhysicsMetricsLogger.plt is None:
            PhysicsMetricsLogger._initialize_plt()
        if PhysicsMetricsLogger.plt is None:
            self.skipTest("matplotlib is not available.")

        setup = _LoggerTestSetup(max_frames=12, device=self.default_device, dt=0.005)
        for k in range(5):
            _populate_uniform_floats(setup, scalar=float(k) * 0.1)
            setup.logger.log()

        filename = "test_plot_smoke"
        ext = "pdf"
        # Remove any stale output from a prior run.
        for field in _SCALAR_FIELDS:
            out_path = self.output_path / f"{filename}_{field}.{ext}"
            if out_path.exists():
                out_path.unlink()

        setup.logger.plot(filename=filename, path=str(self.output_path), show=False, ext=ext)

        for field in _SCALAR_FIELDS:
            out_path = self.output_path / f"{filename}_{field}.{ext}"
            self.assertTrue(out_path.is_file(), msg=f"Expected plot output at {out_path}")
            self.assertGreater(out_path.stat().st_size, 0)
        # Argmax fields must NOT produce plots.
        for field in _ARGMAX_FIELDS:
            out_path = self.output_path / f"{filename}_{field}.{ext}"
            self.assertFalse(out_path.exists(), msg=f"argmax plot should not exist at {out_path}")

    def test_plot_comparison_smoke(self):
        """``plot_comparison()`` overlays two loggers' histories on a shared axis."""
        if PhysicsMetricsLogger.plt is None:
            PhysicsMetricsLogger._initialize_plt()
        if PhysicsMetricsLogger.plt is None:
            self.skipTest("matplotlib is not available.")

        setup_a = _LoggerTestSetup(max_frames=6, device=self.default_device, dt=0.005)
        setup_b = _LoggerTestSetup(max_frames=6, device=self.default_device, dt=0.005)
        for k in range(3):
            _populate_uniform_floats(setup_a, scalar=float(k))
            _populate_uniform_floats(setup_b, scalar=float(k) * 2.0)
            setup_a.logger.log()
            setup_b.logger.log()

        filename = "test_plot_comparison_smoke"
        ext = "pdf"
        out_path = self.output_path / f"{filename}.{ext}"
        if out_path.exists():
            out_path.unlink()

        PhysicsMetricsLogger.plot_comparison(
            loggers={"setup_a": setup_a.logger, "setup_b": setup_b.logger},
            filename=filename,
            path=str(self.output_path),
            grid=True,
            show=False,
            ext=ext,
        )

        self.assertTrue(out_path.is_file(), msg=f"Expected grid plot at {out_path}")
        self.assertGreater(out_path.stat().st_size, 0)

    def test_plot_log_scale_smoke(self):
        """``plot(log_scale=True)`` renders on a log y-axis and tolerates zero samples."""
        if PhysicsMetricsLogger.plt is None:
            PhysicsMetricsLogger._initialize_plt()
        if PhysicsMetricsLogger.plt is None:
            self.skipTest("matplotlib is not available.")

        setup = _LoggerTestSetup(max_frames=8, device=self.default_device, dt=0.005)
        # Include at least one all-zero frame to exercise the safe-clamp path.
        _populate_uniform_floats(setup, scalar=0.0)
        setup.logger.log()
        for k in range(1, 5):
            _populate_uniform_floats(setup, scalar=float(k) * 0.1)
            setup.logger.log()

        filename = "test_plot_log_scale_smoke"
        ext = "pdf"
        for field in _SCALAR_FIELDS:
            out_path = self.output_path / f"{filename}_{field}.{ext}"
            if out_path.exists():
                out_path.unlink()

        setup.logger.plot(
            filename=filename,
            path=str(self.output_path),
            show=False,
            ext=ext,
            log_scale=True,
            log_floor=1e-10,
        )

        for field in _SCALAR_FIELDS:
            out_path = self.output_path / f"{filename}_{field}.{ext}"
            self.assertTrue(out_path.is_file(), msg=f"Expected log-scale plot at {out_path}")
            self.assertGreater(out_path.stat().st_size, 0)

    def test_plot_comparison_log_scale_smoke(self):
        """``plot_comparison(log_scale=True)`` overlays loggers on a base-10 log y-axis."""
        if PhysicsMetricsLogger.plt is None:
            PhysicsMetricsLogger._initialize_plt()
        if PhysicsMetricsLogger.plt is None:
            self.skipTest("matplotlib is not available.")

        setup_a = _LoggerTestSetup(max_frames=6, device=self.default_device, dt=0.005)
        setup_b = _LoggerTestSetup(max_frames=6, device=self.default_device, dt=0.005)
        # Mix zero and positive samples across the two setups.
        for k in range(4):
            _populate_uniform_floats(setup_a, scalar=0.0 if k == 0 else float(k))
            _populate_uniform_floats(setup_b, scalar=float(k) * 2.0)
            setup_a.logger.log()
            setup_b.logger.log()

        filename = "test_plot_comparison_log_scale_smoke"
        ext = "pdf"
        out_path = self.output_path / f"{filename}.{ext}"
        if out_path.exists():
            out_path.unlink()

        PhysicsMetricsLogger.plot_comparison(
            loggers={"setup_a": setup_a.logger, "setup_b": setup_b.logger},
            filename=filename,
            path=str(self.output_path),
            grid=True,
            show=False,
            ext=ext,
            log_scale=True,
            log_floor=1e-10,
        )

        self.assertTrue(out_path.is_file(), msg=f"Expected log-scale grid plot at {out_path}")
        self.assertGreater(out_path.stat().st_size, 0)

    def test_plot_log_scale_validation(self):
        """``log_scale=True`` requires a strictly positive ``log_floor``."""
        if PhysicsMetricsLogger.plt is None:
            PhysicsMetricsLogger._initialize_plt()
        if PhysicsMetricsLogger.plt is None:
            self.skipTest("matplotlib is not available.")

        setup = _LoggerTestSetup(max_frames=4, device=self.default_device, dt=0.005)
        _populate_uniform_floats(setup, scalar=0.0)
        setup.logger.log()

        with self.assertRaises(ValueError):
            setup.logger.plot(
                filename="test_plot_log_scale_invalid",
                path=str(self.output_path),
                show=False,
                log_scale=True,
                log_floor=0.0,
            )

        setup_b = _LoggerTestSetup(max_frames=4, device=self.default_device, dt=0.005)
        _populate_uniform_floats(setup_b, scalar=1.0)
        setup_b.logger.log()

        with self.assertRaises(ValueError):
            PhysicsMetricsLogger.plot_comparison(
                loggers={"a": setup.logger, "b": setup_b.logger},
                filename="test_plot_comparison_log_scale_invalid",
                path=str(self.output_path),
                grid=True,
                show=False,
                log_scale=True,
                log_floor=-1.0,
            )

    ###
    # Argmax-companion semantics
    ###

    def test_argmax_logged_but_not_plotted(self):
        """Argmax companion fields are surfaced by ``to_numpy()`` but never plotted."""
        if PhysicsMetricsLogger.plt is None:
            PhysicsMetricsLogger._initialize_plt()

        setup = _LoggerTestSetup(max_frames=4, device=self.default_device)
        rng = np.random.default_rng(seed=0)
        for _ in range(3):
            for field in _SCALAR_FIELDS:
                setup.assign_per_world(field, rng.random(setup.num_worlds, dtype=np.float32))
            for field in _ARGMAX_FIELDS:
                setup.assign_per_world(field, rng.integers(0, 7, size=setup.num_worlds, dtype=np.int32))
            setup.logger.log()

        np_data = setup.logger.to_numpy()
        # The argmax fields must be present, with the expected shape and dtype.
        for field in _ARGMAX_FIELDS:
            self.assertIn(field, np_data)
            self.assertEqual(np_data[field].shape, (3, setup.num_worlds))
            self.assertEqual(np_data[field].dtype, np.int32)
        # Argmax fields are not named in the equation/title dictionaries used by plot().
        for field in _ARGMAX_FIELDS:
            self.assertNotIn(field, _METRIC_EQUATIONS)
            self.assertNotIn(field, _METRIC_TITLES)

    ###
    # Time axis
    ###

    def test_time_axis_with_dt_scales_by_decimation(self):
        """``time_axis()`` scales by both ``dt`` and ``decimation`` when ``dt`` is provided."""
        dt = 0.01
        decimation = 4
        setup = _LoggerTestSetup(
            max_frames=10,
            decimation=decimation,
            dt=dt,
            device=self.default_device,
        )
        for _ in range(8 * decimation):
            _populate_uniform_floats(setup, scalar=0.1)
            setup.logger.log()
        ts = setup.logger.time_axis()
        self.assertEqual(ts.shape[0], setup.logger.num_logged_frames)
        np.testing.assert_allclose(ts, np.arange(ts.shape[0]) * dt * decimation, atol=1.0e-7)

    def test_time_axis_without_dt_falls_back_to_steps(self):
        """``time_axis()`` reports integer step indices when ``dt`` is ``None``."""
        setup = _LoggerTestSetup(
            max_frames=5,
            decimation=2,
            dt=None,
            device=self.default_device,
        )
        for _ in range(8):
            _populate_uniform_floats(setup, scalar=0.1)
            setup.logger.log()
        ts = setup.logger.time_axis()
        np.testing.assert_allclose(ts, np.arange(ts.shape[0]) * 2, atol=0.0)

    ###
    # Tables
    ###

    def _drive_logger(
        self,
        setup: _LoggerTestSetup,
        n_frames: int,
        scale_world_0: float = 1.0,
        offset_world_0: float = 0.0,
        scale_world_1: float = 1.0,
        offset_world_1: float = 0.0,
    ) -> np.ndarray:
        """Populates ``setup`` with per-frame per-world distinct floats.

        Returns the ``(n_frames, num_worlds)`` ground-truth array so the
        test can compute exact ``min/max/mean`` references against the CSV
        output.
        """
        all_values = np.zeros((n_frames, setup.num_worlds), dtype=np.float32)
        for k in range(n_frames):
            if setup.num_worlds == 1:
                vals = np.array([scale_world_0 * k + offset_world_0], dtype=np.float32)
            else:
                vals = np.array(
                    [
                        scale_world_0 * k + offset_world_0,
                        scale_world_1 * k + offset_world_1,
                    ],
                    dtype=np.float32,
                )
            _populate_per_world_floats(setup, vals)
            setup.logger.log()
            all_values[k] = vals
        return all_values

    def test_table_csv_aggregate(self):
        """``table(per_world=False)`` writes one CSV row per metric with global stats."""
        setup = _LoggerTestSetup(max_frames=5, num_worlds=2, device=self.default_device)
        all_values = self._drive_logger(
            setup,
            n_frames=5,
            scale_world_0=1.0,
            offset_world_0=0.0,
            scale_world_1=1.0,
            offset_world_1=10.0,
        )

        filename = "test_table_csv_aggregate"
        csv_path = self.output_path / f"{filename}.csv"
        if csv_path.exists():
            csv_path.unlink()

        setup.logger.table(filename=filename, path=str(self.output_path), per_world=False)

        self.assertTrue(csv_path.is_file(), msg=f"Expected CSV at {csv_path}")
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, ["Metric", "max", "mean"])
        self.assertEqual(len(rows), len(_SCALAR_FIELDS))

        expected_max = float(all_values.max())
        expected_mean = float(all_values.mean())
        observed_fields = {row[0] for row in rows}
        self.assertEqual(observed_fields, set(_SCALAR_FIELDS))
        for row in rows:
            self.assertAlmostEqual(float(row[1]), expected_max, places=5)
            self.assertAlmostEqual(float(row[2]), expected_mean, places=5)

    def test_table_csv_per_world(self):
        """``table(per_world=True)`` writes one CSV row per ``(world, metric)`` with per-world stats."""
        setup = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        all_values = self._drive_logger(
            setup,
            n_frames=4,
            scale_world_0=1.0,
            offset_world_0=0.0,
            scale_world_1=2.0,
            offset_world_1=5.0,
        )

        filename = "test_table_csv_per_world"
        csv_path = self.output_path / f"{filename}.csv"
        if csv_path.exists():
            csv_path.unlink()

        setup.logger.table(filename=filename, path=str(self.output_path), per_world=True)

        self.assertTrue(csv_path.is_file(), msg=f"Expected CSV at {csv_path}")
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, ["World", "Metric", "max", "mean"])
        self.assertEqual(len(rows), len(_SCALAR_FIELDS) * setup.num_worlds)
        for row in rows:
            world_token, field, mx, mean = row
            self.assertIn(field, _SCALAR_FIELDS)
            self.assertTrue(world_token.startswith("world_"))
            w = int(world_token.split("_")[1])
            self.assertAlmostEqual(float(mx), float(all_values[:, w].max()), places=5)
            self.assertAlmostEqual(float(mean), float(all_values[:, w].mean()), places=5)

    def test_table_console_smoke(self):
        """``table(to_console=True)`` renders a rich table to stdout without raising."""
        if not _RICH_AVAILABLE:
            self.skipTest("rich is not available")

        setup = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        self._drive_logger(setup, n_frames=3, scale_world_1=2.0)
        setup.logger.table(to_console=True, per_world=False)
        setup.logger.table(to_console=True, per_world=True)

    def test_table_warns_with_no_frames(self):
        """``table()`` on a brand-new logger warns and writes no CSV file."""
        setup = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        filename = "test_table_warns_with_no_frames"
        csv_path = self.output_path / f"{filename}.csv"
        if csv_path.exists():
            csv_path.unlink()

        setup.logger.table(filename=filename, path=str(self.output_path))

        self.assertFalse(csv_path.exists(), msg=f"CSV should not exist for empty logger at {csv_path}")

    def test_table_warns_when_no_output_target(self):
        """``table()`` with neither ``path`` nor ``to_console`` set must warn and return."""
        setup = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        self._drive_logger(setup, n_frames=3)
        # Should not raise; just warn and return.
        setup.logger.table(path=None, to_console=False)

    def test_table_invalid_path_raises(self):
        """``table(path=<nonexistent>)`` must raise ``ValueError``."""
        setup = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        self._drive_logger(setup, n_frames=2)
        missing = self.output_path / "this_subdir_does_not_exist"
        if missing.exists():
            self.skipTest("expected non-existent path actually exists; skipping")
        with self.assertRaises(ValueError):
            setup.logger.table(filename="anything", path=str(missing))

    def test_table_comparison_csv_aggregate(self):
        """``table_comparison(per_world=False)`` produces a two-row header CSV with both loggers' stats."""
        setup_a = _LoggerTestSetup(max_frames=5, num_worlds=2, device=self.default_device)
        setup_b = _LoggerTestSetup(max_frames=5, num_worlds=2, device=self.default_device)
        all_a = self._drive_logger(setup_a, n_frames=4, scale_world_1=1.0, offset_world_1=1.0)
        all_b = self._drive_logger(
            setup_b,
            n_frames=4,
            scale_world_0=2.0,
            offset_world_0=0.0,
            scale_world_1=3.0,
            offset_world_1=0.0,
        )

        filename = "test_table_comparison_csv_aggregate"
        csv_path = self.output_path / f"{filename}.csv"
        if csv_path.exists():
            csv_path.unlink()

        PhysicsMetricsLogger.table_comparison(
            loggers={"setup_a": setup_a.logger, "setup_b": setup_b.logger},
            filename=filename,
            path=str(self.output_path),
            per_world=False,
        )

        self.assertTrue(csv_path.is_file(), msg=f"Expected CSV at {csv_path}")
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            top_header = next(reader)
            sub_header = next(reader)
            rows = list(reader)

        self.assertEqual(top_header, ["", "setup_a", "", "setup_b", ""])
        self.assertEqual(sub_header, ["Metric", "max", "mean", "max", "mean"])
        self.assertEqual(len(rows), len(_SCALAR_FIELDS))
        for row in rows:
            self.assertIn(row[0], _SCALAR_FIELDS)
            self.assertAlmostEqual(float(row[1]), float(all_a.max()), places=5)
            self.assertAlmostEqual(float(row[2]), float(all_a.mean()), places=5)
            self.assertAlmostEqual(float(row[3]), float(all_b.max()), places=5)
            self.assertAlmostEqual(float(row[4]), float(all_b.mean()), places=5)

    def test_table_comparison_csv_per_world(self):
        """``table_comparison(per_world=True)`` produces per-``(world, metric)`` rows with per-world stats per logger."""
        setup_a = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        setup_b = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        all_a = self._drive_logger(setup_a, n_frames=3, scale_world_1=2.0)
        all_b = self._drive_logger(setup_b, n_frames=3, scale_world_0=2.0, scale_world_1=3.0)

        filename = "test_table_comparison_csv_per_world"
        csv_path = self.output_path / f"{filename}.csv"
        if csv_path.exists():
            csv_path.unlink()

        PhysicsMetricsLogger.table_comparison(
            loggers={"setup_a": setup_a.logger, "setup_b": setup_b.logger},
            filename=filename,
            path=str(self.output_path),
            per_world=True,
            to_console=True,
        )

        self.assertTrue(csv_path.is_file(), msg=f"Expected CSV at {csv_path}")
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            top_header = next(reader)
            sub_header = next(reader)
            rows = list(reader)

        self.assertEqual(top_header, ["", "", "setup_a", "", "setup_b", ""])
        self.assertEqual(sub_header, ["World", "Metric", "max", "mean", "max", "mean"])
        self.assertEqual(len(rows), len(_SCALAR_FIELDS) * setup_a.num_worlds)
        for row in rows:
            world_token = row[0]
            field = row[1]
            self.assertTrue(world_token.startswith("world_"))
            self.assertIn(field, _SCALAR_FIELDS)
            w = int(world_token.split("_")[1])
            self.assertAlmostEqual(float(row[2]), float(all_a[:, w].max()), places=5)
            self.assertAlmostEqual(float(row[3]), float(all_a[:, w].mean()), places=5)
            self.assertAlmostEqual(float(row[4]), float(all_b[:, w].max()), places=5)
            self.assertAlmostEqual(float(row[5]), float(all_b[:, w].mean()), places=5)

    def test_table_comparison_color_rankings_smoke(self):
        """``table_comparison(color_rankings=True)`` renders colored cells to stdout without raising."""
        if not _RICH_AVAILABLE:
            self.skipTest("rich is not available")
        setup_a = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        setup_b = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        self._drive_logger(setup_a, n_frames=3, scale_world_0=1.0, scale_world_1=2.0)
        self._drive_logger(setup_b, n_frames=3, scale_world_0=2.0, scale_world_1=3.0)

        PhysicsMetricsLogger.table_comparison(
            loggers={"setup_a": setup_a.logger, "setup_b": setup_b.logger},
            to_console=True,
            color_rankings=True,
            per_world=False,
        )
        PhysicsMetricsLogger.table_comparison(
            loggers={"setup_a": setup_a.logger, "setup_b": setup_b.logger},
            to_console=True,
            color_rankings=True,
            per_world=True,
        )

    def test_table_comparison_color_rankings_csv_unchanged(self):
        """``color_rankings=True`` must not change the CSV (numeric data only)."""
        setup_a = _LoggerTestSetup(max_frames=4, num_worlds=1, device=self.default_device)
        setup_b = _LoggerTestSetup(max_frames=4, num_worlds=1, device=self.default_device)
        self._drive_logger(setup_a, n_frames=3, scale_world_0=1.0)
        self._drive_logger(setup_b, n_frames=3, scale_world_0=2.0)

        plain = self.output_path / "test_table_comparison_color_rankings_plain.csv"
        colored = self.output_path / "test_table_comparison_color_rankings_colored.csv"
        for p in (plain, colored):
            if p.exists():
                p.unlink()

        PhysicsMetricsLogger.table_comparison(
            loggers={"setup_a": setup_a.logger, "setup_b": setup_b.logger},
            filename=plain.stem,
            path=str(self.output_path),
            color_rankings=False,
        )
        PhysicsMetricsLogger.table_comparison(
            loggers={"setup_a": setup_a.logger, "setup_b": setup_b.logger},
            filename=colored.stem,
            path=str(self.output_path),
            color_rankings=True,
        )
        self.assertEqual(plain.read_text(), colored.read_text())

    def test_table_comparison_raises_on_mismatched_worlds(self):
        """``table_comparison()`` rejects loggers with differing ``num_worlds``."""
        setup_a = _LoggerTestSetup(max_frames=4, num_worlds=2, device=self.default_device)
        setup_b = _LoggerTestSetup(max_frames=4, num_worlds=1, device=self.default_device)
        with self.assertRaises(ValueError):
            PhysicsMetricsLogger.table_comparison(
                loggers={"a": setup_a.logger, "b": setup_b.logger},
                filename="test_table_comparison_raises_on_mismatched_worlds",
                path=str(self.output_path),
            )


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
