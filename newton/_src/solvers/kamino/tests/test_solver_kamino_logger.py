# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino: Tests for the on-device :class:`SolverKaminoLogger`."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils.logger import SolverKaminoLogger
from newton.tests.utils import basics

###
# Scaffolding
###


class _LoggerTestSetup:
    """Builds a multi-world `SolverKamino` + `SolverKaminoLogger` for the tests."""

    def __init__(
        self,
        max_frames: int,
        mode: SolverKaminoLogger.Mode = SolverKaminoLogger.Mode.BOUNDED,
        decimation: int = 1,
        num_worlds: int = 2,
        max_world_contacts: int = 8,
        dt: float = 0.005,
        device: wp.DeviceLike | None = None,
        with_iterate_residuals_info: bool = False,
        with_acceleration_info: bool = False,
    ):
        self.dt = dt
        self.num_worlds = num_worlds

        # Build a homogeneous multi-world box-on-plane scene using Newton's
        # public ModelBuilder so the tests exercise the high-level
        # `SolverKamino` API path that downstream users follow.
        self.builder: ModelBuilder = ModelBuilder()
        for _ in range(num_worlds):
            basics.build_box_on_plane(builder=self.builder, new_world=True)
        self.builder.request_contact_attributes("force")
        self.builder.num_rigid_contacts_per_world = max_world_contacts

        self.model: Model = self.builder.finalize(device=device)
        self.state: State = self.model.state()
        self.state_p: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        # Minimal stable solver config; collision detection runs through Newton
        # so contacts are populated by `model.collide`.
        solver_config = newton.solvers.SolverKamino.Config()
        self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)

        self.logger = SolverKaminoLogger(
            solver=self.solver,
            max_frames=max_frames,
            mode=mode,
            decimation=decimation,
            dt=dt,
            with_iterate_residuals_info=with_iterate_residuals_info,
            with_acceleration_info=with_acceleration_info,
        )

    @property
    def device(self) -> wp.DeviceLike:
        return self.solver.device

    def step(self):
        """Take one simulation step and log it."""
        self.model.collide(self.state_p, self.contacts)
        self.solver.step(
            state_in=self.state_p,
            state_out=self.state,
            control=self.control,
            contacts=self.contacts,
            dt=self.dt,
        )
        self.logger.log()
        self.state, self.state_p = self.state_p, self.state

    def status_snapshot(self) -> np.ndarray:
        """Return the current PADMM status as a numpy structured array."""
        return self.solver._solver_kamino.solver_fd.data.status.numpy().copy()

    def penalty_snapshot(self) -> np.ndarray:
        """Return the current PADMM penalty as a numpy structured array."""
        return self.solver._solver_kamino.solver_fd.data.penalty.numpy().copy()

    def state_a_snapshot(self) -> np.ndarray:
        """Return the current Nesterov acceleration variable per world."""
        return self.solver._solver_kamino.solver_fd.data.state.a.numpy().copy()

    def state_a_factor_snapshot(self) -> np.ndarray:
        """Return the current Nesterov acceleration factor per world."""
        return self.solver._solver_kamino.solver_fd.data.state.a_factor.numpy().copy()


###
# Tests
###


class TestSolverKaminoLogger(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.output_path = test_context.output_path / "test_solver_kamino_logger"
        self.output_path.mkdir(parents=True, exist_ok=True)
        msg.reset_log_level()

    ###
    # Constructor / configuration
    ###

    def test_constructor_validation(self):
        """Invalid args raise; valid args expose correctly-shaped buffers."""
        setup = _LoggerTestSetup(max_frames=8, num_worlds=2, device=self.default_device)

        # Invalid 'solver' type
        with self.assertRaises(TypeError):
            SolverKaminoLogger(solver="not_a_solver", max_frames=8)
        # Invalid 'max_frames'
        with self.assertRaises(ValueError):
            SolverKaminoLogger(solver=setup.solver, max_frames=0)
        with self.assertRaises(ValueError):
            SolverKaminoLogger(solver=setup.solver, max_frames=-1)
        # Invalid 'decimation'
        with self.assertRaises(ValueError):
            SolverKaminoLogger(solver=setup.solver, max_frames=8, decimation=0)
        # Invalid 'mode'
        with self.assertRaises(TypeError):
            SolverKaminoLogger(solver=setup.solver, max_frames=8, mode="rolling")
        # Invalid 'dt'
        with self.assertRaises(ValueError):
            SolverKaminoLogger(solver=setup.solver, max_frames=8, dt=-0.1)

        # The actual logger created in setup must expose correct shapes and dtypes.
        self.assertEqual(setup.logger.max_frames, 8)
        self.assertEqual(setup.logger.num_worlds, setup.num_worlds)
        self.assertEqual(setup.logger.mode, SolverKaminoLogger.Mode.BOUNDED)
        self.assertEqual(setup.logger.decimation, 1)
        self.assertEqual(setup.logger.num_logged_frames, 0)
        self.assertEqual(setup.logger.num_total_writes, 0)
        self.assertFalse(setup.logger.is_full)

        # Float fields are float32, integer fields are int32; all 2-D buffers
        # are shaped `(max_frames, num_worlds)`.
        float_fields = ("r_p", "r_d", "r_c", "rho")
        int_fields = ("iterations", "converged", "num_restarts", "num_rho_updates", "num_limits", "num_contacts")
        expected_shape = (setup.logger.max_frames, setup.logger.num_worlds)
        for field in float_fields:
            buf = getattr(setup.logger, f"log_{field}")
            self.assertEqual(buf.shape, expected_shape)
            self.assertEqual(buf.dtype, wp.float32)
        for field in int_fields:
            buf = getattr(setup.logger, f"log_{field}")
            self.assertEqual(buf.shape, expected_shape)
            self.assertEqual(buf.dtype, wp.int32)

    ###
    # Buffer-overflow semantics
    ###

    def test_bounded_overflow_stops_logging(self):
        """`Mode.BOUNDED` logger stops writing once `max_frames` is reached."""
        n = 4
        setup = _LoggerTestSetup(max_frames=n, mode=SolverKaminoLogger.Mode.BOUNDED, device=self.default_device)

        for _ in range(2 * n):
            setup.step()

        self.assertEqual(setup.logger.num_logged_frames, n)
        self.assertEqual(setup.logger.num_total_writes, n)
        self.assertTrue(setup.logger.is_full)
        self.assertEqual(setup.logger.num_calls, 2 * n)

        np_data = setup.logger.to_numpy()
        self.assertEqual(np_data["iterations"].shape, (n, setup.num_worlds))
        self.assertTrue((np_data["iterations"] >= 0).all())
        self.assertTrue(np.isfinite(np_data["r_p"]).all())

    def test_rolling_wraparound_returns_chronological_order(self):
        """`Mode.ROLLING` keeps the last `max_frames` samples in chronological order."""
        n = 5
        extra = 3
        total_steps = 2 * n + extra
        setup = _LoggerTestSetup(
            max_frames=n,
            mode=SolverKaminoLogger.Mode.ROLLING,
            device=self.default_device,
        )

        # Snapshot the per-step iteration counts so we can verify chronological
        # ordering after the buffer wraps around twice.
        iter_history = np.zeros((total_steps, setup.num_worlds), dtype=np.int32)
        for k in range(total_steps):
            setup.step()
            iter_history[k] = setup.status_snapshot()["iterations"]

        self.assertEqual(setup.logger.num_logged_frames, n)
        self.assertEqual(setup.logger.num_total_writes, total_steps)
        self.assertTrue(setup.logger.is_full)

        # The buffer should hold the most recent `n` samples in chronological order.
        np_data = setup.logger.to_numpy()
        expected = iter_history[total_steps - n :]
        np.testing.assert_array_equal(
            np_data["iterations"],
            expected,
            err_msg="Rolling-mode `iterations` rows are not in chronological order after wraparound.",
        )

    def test_decimation_skips_frames(self):
        """`decimation=k` writes only every k-th call; intermediate calls are no-ops."""
        decimation = 3
        num_calls = 10
        setup = _LoggerTestSetup(
            max_frames=16,
            mode=SolverKaminoLogger.Mode.BOUNDED,
            decimation=decimation,
            device=self.default_device,
        )

        # Record the iteration count seen at each step so we can verify which
        # ones the logger actually retained.
        iter_history = np.zeros((num_calls, setup.num_worlds), dtype=np.int32)
        for k in range(num_calls):
            setup.step()
            iter_history[k] = setup.status_snapshot()["iterations"]

        # With decimation=3 and 10 calls, exactly ceil(10/3) = 4 frames are written
        # (calls 0, 3, 6, 9).
        expected_writes = (num_calls + decimation - 1) // decimation
        self.assertEqual(setup.logger.num_logged_frames, expected_writes)
        self.assertEqual(setup.logger.num_total_writes, expected_writes)
        self.assertEqual(setup.logger.num_calls, num_calls)

        np_data = setup.logger.to_numpy()
        expected = iter_history[::decimation]
        np.testing.assert_array_equal(
            np_data["iterations"],
            expected,
            err_msg="Decimated logger retained the wrong sub-sampled iteration history.",
        )

    ###
    # Reset
    ###

    def test_reset_clears_state(self):
        """`reset()` restores counters to zero and clears every log buffer."""
        setup = _LoggerTestSetup(max_frames=6, device=self.default_device)
        for _ in range(3):
            setup.step()
        self.assertEqual(setup.logger.num_logged_frames, 3)

        setup.logger.reset()
        self.assertEqual(setup.logger.num_logged_frames, 0)
        self.assertEqual(setup.logger.num_total_writes, 0)
        self.assertEqual(setup.logger.num_calls, 0)

        for field in (
            "iterations",
            "converged",
            "num_restarts",
            "num_rho_updates",
            "num_limits",
            "num_contacts",
            "r_p",
            "r_d",
            "r_c",
            "rho",
        ):
            buf = getattr(setup.logger, f"log_{field}").numpy()
            np.testing.assert_array_equal(
                buf,
                np.zeros_like(buf),
                err_msg=f"log_{field} was not zeroed by `reset()`.",
            )

    ###
    # CUDA-graph capture (the key improvement over SolutionMetricsLogger)
    ###

    def test_graph_capture_log_only(self):
        """`logger.log()` can be captured into a CUDA graph and replayed."""
        if not self.default_device.is_cuda:
            self.skipTest("Graph capture requires a CUDA device.")

        setup = _LoggerTestSetup(max_frames=16, device=self.default_device)

        # Take one warm-up step so `solver_fd.data.status` holds valid values
        # before the capture replays them repeatedly.
        setup.step()
        reference_iter = setup.status_snapshot()["iterations"].copy()
        reference_rho = setup.penalty_snapshot()["rho"].copy()

        # Reset the logger; the captured graph should write exactly K frames.
        setup.logger.reset()

        replay_count = 7
        with wp.ScopedCapture(device=setup.device) as capture:
            setup.logger.log()
        graph = capture.graph

        for _ in range(replay_count):
            wp.capture_launch(graph)
        wp.synchronize_device(setup.device)

        self.assertEqual(setup.logger.num_logged_frames, replay_count)
        self.assertEqual(setup.logger.num_total_writes, replay_count)
        self.assertEqual(setup.logger.num_calls, replay_count)

        # Since the solver state was not advanced between replays, every
        # logged row must match the reference snapshot.
        np_data = setup.logger.to_numpy()
        for k in range(replay_count):
            np.testing.assert_array_equal(
                np_data["iterations"][k],
                reference_iter,
                err_msg=f"Captured logger row {k} has unexpected iterations value.",
            )
            np.testing.assert_array_equal(
                np_data["rho"][k],
                reference_rho,
                err_msg=f"Captured logger row {k} has unexpected rho value.",
            )

    def test_graph_capture_bounded_overflow(self):
        """Bounded-mode overflow is enforced inside graph capture."""
        if not self.default_device.is_cuda:
            self.skipTest("Graph capture requires a CUDA device.")

        max_frames = 4
        replay_count = 10
        setup = _LoggerTestSetup(
            max_frames=max_frames,
            mode=SolverKaminoLogger.Mode.BOUNDED,
            device=self.default_device,
        )

        # Populate solver status with one warm-up step, then start fresh.
        setup.step()
        setup.logger.reset()

        with wp.ScopedCapture(device=setup.device) as capture:
            setup.logger.log()
        graph = capture.graph

        for _ in range(replay_count):
            wp.capture_launch(graph)
        wp.synchronize_device(setup.device)

        # The on-device decision kernel must early-exit once max_frames is reached.
        self.assertEqual(setup.logger.num_logged_frames, max_frames)
        self.assertEqual(setup.logger.num_total_writes, max_frames)
        self.assertEqual(setup.logger.num_calls, replay_count)
        self.assertTrue(setup.logger.is_full)

    ###
    # Plotting
    ###

    def test_plot_smoke(self):
        """`plot()` writes a non-empty figure to disk when matplotlib is available."""
        if SolverKaminoLogger.plt is None:
            SolverKaminoLogger._initialize_plt()
        if SolverKaminoLogger.plt is None:
            self.skipTest("matplotlib is not available.")

        setup = _LoggerTestSetup(max_frames=12, device=self.default_device)
        for _ in range(5):
            setup.step()

        filename = "test_plot_smoke"
        ext = "pdf"
        out_path = self.output_path / f"{filename}.{ext}"
        if out_path.exists():
            out_path.unlink()

        setup.logger.plot(filename=filename, path=str(self.output_path), show=False, ext=ext)

        self.assertTrue(out_path.is_file(), msg=f"Expected plot output at {out_path}")
        self.assertGreater(out_path.stat().st_size, 0)

    ###
    # Optional advanced PADMM info
    ###

    def test_iterate_residuals_buffers_off_by_default(self):
        """Without `with_iterate_residuals_info`, iterate-residual buffers are not allocated."""
        setup = _LoggerTestSetup(max_frames=4, device=self.default_device)
        for field in ("r_dx", "r_dy", "r_dz"):
            self.assertFalse(
                hasattr(setup.logger, f"log_{field}"),
                msg=f"log_{field} should NOT be allocated when with_iterate_residuals_info=False.",
            )
        np_data = setup.logger.to_numpy()
        for field in ("r_dx", "r_dy", "r_dz"):
            self.assertNotIn(field, np_data)

    def test_acceleration_buffers_off_by_default(self):
        """Without `with_acceleration_info`, acceleration buffers are not allocated."""
        setup = _LoggerTestSetup(max_frames=4, device=self.default_device)
        for field in ("r_a", "a", "a_factor"):
            self.assertFalse(
                hasattr(setup.logger, f"log_{field}"),
                msg=f"log_{field} should NOT be allocated when with_acceleration_info=False.",
            )
        np_data = setup.logger.to_numpy()
        for field in ("r_a", "a", "a_factor"):
            self.assertNotIn(field, np_data)

    def test_with_iterate_residuals_info_allocates_and_logs(self):
        """`with_iterate_residuals_info=True` allocates `r_dx`/`r_dy`/`r_dz` and logs them."""
        n_steps = 4
        setup = _LoggerTestSetup(
            max_frames=8,
            device=self.default_device,
            with_iterate_residuals_info=True,
        )

        expected_shape = (setup.logger.max_frames, setup.logger.num_worlds)
        for field in ("r_dx", "r_dy", "r_dz"):
            buf = getattr(setup.logger, f"log_{field}")
            self.assertEqual(buf.shape, expected_shape)
            self.assertEqual(buf.dtype, wp.float32)

        # Capture per-step snapshots so we can verify logged rows match.
        history = {field: np.zeros((n_steps, setup.num_worlds), dtype=np.float32) for field in ("r_dx", "r_dy", "r_dz")}
        for k in range(n_steps):
            setup.step()
            snap = setup.status_snapshot()
            for field in ("r_dx", "r_dy", "r_dz"):
                history[field][k] = snap[field]

        np_data = setup.logger.to_numpy()
        for field in ("r_dx", "r_dy", "r_dz"):
            self.assertIn(field, np_data)
            self.assertEqual(np_data[field].shape, (n_steps, setup.num_worlds))
            np.testing.assert_allclose(
                np_data[field],
                history[field],
                err_msg=f"Logged {field} does not match per-step PADMMStatus snapshots.",
            )

    def test_with_acceleration_info_allocates_and_logs(self):
        """`with_acceleration_info=True` allocates `r_a`/`a`/`a_factor` and logs them."""
        n_steps = 4
        setup = _LoggerTestSetup(
            max_frames=8,
            device=self.default_device,
            with_acceleration_info=True,
        )

        expected_shape = (setup.logger.max_frames, setup.logger.num_worlds)
        for field in ("r_a", "a", "a_factor"):
            buf = getattr(setup.logger, f"log_{field}")
            self.assertEqual(buf.shape, expected_shape)
            self.assertEqual(buf.dtype, wp.float32)

        history_r_a = np.zeros((n_steps, setup.num_worlds), dtype=np.float32)
        history_a = np.zeros((n_steps, setup.num_worlds), dtype=np.float32)
        history_a_factor = np.zeros((n_steps, setup.num_worlds), dtype=np.float32)
        for k in range(n_steps):
            setup.step()
            history_r_a[k] = setup.status_snapshot()["r_a"]
            history_a[k] = setup.state_a_snapshot()
            history_a_factor[k] = setup.state_a_factor_snapshot()

        np_data = setup.logger.to_numpy()
        np.testing.assert_allclose(np_data["r_a"], history_r_a, err_msg="Logged r_a does not match snapshots.")
        np.testing.assert_allclose(np_data["a"], history_a, err_msg="Logged a does not match snapshots.")
        np.testing.assert_allclose(
            np_data["a_factor"], history_a_factor, err_msg="Logged a_factor does not match snapshots."
        )

    def test_with_acceleration_info_plots_num_restarts(self):
        """`with_acceleration_info=True` adds `num_restarts` to the plotted metrics."""
        # Basic logger keeps `num_restarts` as a logged-only field (always
        # written to `log_num_restarts` but not surfaced on the figure).
        basic = _LoggerTestSetup(max_frames=4, device=self.default_device)
        self.assertNotIn(
            "num_restarts",
            basic.logger._plotted_fields,
            msg="Basic logger should not plot `num_restarts` (it stays logged-only).",
        )

        # Enabling acceleration info promotes `num_restarts` to a plotted
        # panel alongside `r_a`, `a`, and `a_factor`.
        accel = _LoggerTestSetup(
            max_frames=4,
            device=self.default_device,
            with_acceleration_info=True,
        )
        self.assertIn(
            "num_restarts",
            accel.logger._plotted_fields,
            msg="`with_acceleration_info=True` should add `num_restarts` to the plotted metrics.",
        )
        # And it must appear exactly once.
        self.assertEqual(
            accel.logger._plotted_fields.count("num_restarts"),
            1,
            msg="`num_restarts` should appear exactly once in the plotted fields tuple.",
        )

    def test_reset_clears_extension_buffers(self):
        """`reset()` clears the optional iterate-residual and acceleration buffers."""
        setup = _LoggerTestSetup(
            max_frames=6,
            device=self.default_device,
            with_iterate_residuals_info=True,
            with_acceleration_info=True,
        )
        for _ in range(3):
            setup.step()

        setup.logger.reset()
        self.assertEqual(setup.logger.num_logged_frames, 0)

        for field in ("r_dx", "r_dy", "r_dz", "r_a", "a", "a_factor"):
            buf = getattr(setup.logger, f"log_{field}").numpy()
            np.testing.assert_array_equal(
                buf,
                np.zeros_like(buf),
                err_msg=f"log_{field} was not zeroed by `reset()`.",
            )

    def test_plot_with_all_options(self):
        """`plot()` with both extensions enabled lays out an auto-sized grid."""
        if SolverKaminoLogger.plt is None:
            SolverKaminoLogger._initialize_plt()
        if SolverKaminoLogger.plt is None:
            self.skipTest("matplotlib is not available.")

        setup = _LoggerTestSetup(
            max_frames=12,
            device=self.default_device,
            with_iterate_residuals_info=True,
            with_acceleration_info=True,
        )
        for _ in range(5):
            setup.step()

        filename = "test_plot_with_all_options"
        ext = "pdf"
        out_path = self.output_path / f"{filename}.{ext}"
        if out_path.exists():
            out_path.unlink()

        setup.logger.plot(filename=filename, path=str(self.output_path), show=False, ext=ext)

        self.assertTrue(out_path.is_file(), msg=f"Expected plot output at {out_path}")
        self.assertGreater(out_path.stat().st_size, 0)


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
