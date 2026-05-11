# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import os
import time
import unittest
from collections.abc import Callable

import numpy as np
import warp as wp

import newton
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.metrics import SolutionMetricsLogger, SolutionMetricsNewton
from newton._src.solvers.kamino._src.metrics.logging import (
    _METRIC_EQUATIONS,
    _METRIC_TITLES,
    _SCALAR_METRIC_FIELDS,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.examples import print_progress_bar
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils import basics

###
# Constants
###

# Strict tolerance for quantities that are computed on identical inputs
# across both sides of the comparison (e.g. system Jacobians, dual-problem
# building blocks, post-event constraint-space velocity at active rows).
TOL_STRICT_RTOL = 1e-6
TOL_STRICT_ATOL = 1e-6

# Lambda-independent metric residuals. These depend only on the post-event
# state, the kinematic constraint residuals stored on `data`, and the limit /
# contact gap functions, all of which are populated identically by both flows.
METRIC_FIELDS_LAMBDA_INDEPENDENT = (
    "r_kinematics",
    "r_cts_joints",
    "r_cts_limits",
    "r_cts_contacts",
)

# Metric residuals that consume the extracted lambdas (or quantities
# derived from them, such as the body wrenches in r_eom).
METRIC_FIELDS_LAMBDA_DEPENDENT = (
    "r_eom",
    "r_v_plus",
    "r_ncp_primal",
    "r_ncp_dual",
    "r_ncp_compl",
    "r_vi_natmap",
    "f_ncp",
    "f_ccp",
)

###
# Scaffolding
###


class TestSetup:
    def __init__(
        self,
        builder_fn,
        builder_kwargs: dict | None = None,
        request_state_attributes: tuple[str, ...] = (),
        device: wp.DeviceLike = None,
        max_world_contacts: int = 32,
        max_frames: int = 100,
        dt: float = 0.001,
    ):
        # Cache scalar configurations
        self.max_world_contacts = max_world_contacts
        self.max_frames = max_frames
        self.dt = dt

        # Construct the Newton model description with the requested attributes
        if builder_kwargs is None:
            builder_kwargs = {}
        self.builder: ModelBuilder = builder_fn(**builder_kwargs)
        self.builder.request_contact_attributes("force")
        if request_state_attributes:
            self.builder.request_state_attributes(*request_state_attributes)
        self.builder.num_rigid_contacts_per_world = max_world_contacts

        # Finalise the Newton-side runtime containers
        self.model: Model = self.builder.finalize(skip_validation_joints=True)
        self.state: State = self.model.state()
        self.state_p: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        # Metrics evaluator
        self.metrics = SolutionMetricsNewton(
            model=self.builder.finalize(skip_validation_joints=True),
            dt=self.dt,
            sparse=False,
        )

        # Reference solver with metrics computation enabled
        solver_config = self._make_solver_config_default()
        solver_config.compute_solution_metrics = True
        self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)

        # Metrics loggers
        self.logger_metrics = SolutionMetricsLogger(
            metrics=self.metrics,
            max_frames=self.max_frames,
            mode=SolutionMetricsLogger.Mode.BOUNDED,
            dt=self.dt,
        )
        self.logger_solver = SolutionMetricsLogger(
            metrics=self.solver._solver_kamino.metrics,
            max_frames=self.max_frames,
            mode=SolutionMetricsLogger.Mode.BOUNDED,
            dt=self.dt,
        )

    def step(self):
        self.model.collide(self.state_p, self.contacts)
        self.solver.step(
            state_in=self.state_p,
            state_out=self.state,
            control=self.control,
            contacts=self.contacts,
            dt=self.dt,
        )
        self.solver.update_contacts(self.contacts, self.state)
        self.metrics.evaluate(
            state=self.state,
            state_p=self.state_p,
            control=self.control,
            contacts=self.contacts,
        )
        self.logger_metrics.log()
        self.logger_solver.log()
        self.state, self.state_p = self.state_p, self.state

    @staticmethod
    def _make_solver_config_default() -> newton.solvers.SolverKamino.Config:
        config = newton.solvers.SolverKamino.Config()
        config.constraints = config.constraints or DualProblem.Config().constraints.__class__()
        config.dynamics = config.dynamics or DualProblem.Config().dynamics.__class__()
        config.constraints.alpha = 0.0
        config.constraints.beta = 0.0
        config.constraints.gamma = 0.0
        config.constraints.delta = 0.0
        config.dynamics.preconditioning = False
        return config


###
# Helper Functions
###


def _assert_loggers_match(
    testcase: unittest.TestCase,
    logger_metrics: SolutionMetricsLogger,
    logger_solver: SolutionMetricsLogger,
    fields: tuple[str, ...],
    rtol: float = TOL_STRICT_RTOL,
    atol: float = TOL_STRICT_ATOL,
):
    """
    Cross-check the two loggers' recorded data field-by-field.

    Compares the full ``(num_logged_frames, num_worlds)`` arrays for every
    entry in ``fields``, then iterates step-by-step so the failure message
    surfaces the first divergent step.
    """
    np_metrics = logger_metrics.to_numpy()
    np_solver = logger_solver.to_numpy()

    for field in fields:
        arr_metrics = np_metrics[field]
        arr_solver = np_solver[field]
        testcase.assertEqual(
            arr_metrics.shape,
            arr_solver.shape,
            msg=f"SolutionMetricsData.{field} shape mismatch: {arr_metrics.shape} vs {arr_solver.shape}",
        )
        np.testing.assert_allclose(
            arr_metrics,
            arr_solver,
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"SolutionMetricsData.{field} disagrees between SolutionMetricsNewton "
                f"and the SolverKamino reference over the full trajectory."
            ),
        )
        for step_idx in range(arr_metrics.shape[0]):
            np.testing.assert_allclose(
                arr_metrics[step_idx],
                arr_solver[step_idx],
                rtol=rtol,
                atol=atol,
                err_msg=(
                    f"SolutionMetricsData.{field} disagrees at step {step_idx} between "
                    f"SolutionMetricsNewton and the SolverKamino reference."
                ),
            )


def plot_logger_comparison(
    logger_metrics: SolutionMetricsLogger,
    logger_solver: SolutionMetricsLogger,
    path: str | None = None,
    show: bool = False,
    ext: str = "pdf",
    label_metrics: str = "SolutionMetricsNewton",
    label_solver: str = "SolverKamino",
):
    """Render one matplotlib figure per scalar metric overlaying two loggers.

    Plots ``logger_metrics`` data in blue (marker ``"o"``, solid line) and
    ``logger_solver`` data in red (marker ``"x"``, dashed line) on the same
    axis, drawing one curve per world per source. The figure title and
    LaTeX subtitle follow the same format as :meth:`SolutionMetricsLogger.plot`,
    so the per-metric output is visually consistent with the single-source
    plots produced by the logger itself.

    Args:
        logger_metrics: Logger wrapping the :class:`SolutionMetricsNewton`
            instance under test (rendered in blue).
        logger_solver: Logger wrapping the :class:`SolverKamino`-internal
            :class:`SolutionMetrics` reference (rendered in red).
        path: If provided, saves each figure as ``{path}/{metric_name}.{ext}``.
            The directory must already exist.
        show: If ``True``, displays the figures interactively (blocking).
        ext: File extension / matplotlib format. Defaults to ``"pdf"`` to
            match the benchmarks output.
        label_metrics: Legend label prefix for the wrapper-side data.
        label_solver: Legend label prefix for the solver-side data.
    """
    SolutionMetricsLogger.initialize_plt()
    if SolutionMetricsLogger.plt is None:
        msg.warning("matplotlib is not available, skipping plotting.")
        return
    if logger_metrics.num_logged_frames == 0 or logger_solver.num_logged_frames == 0:
        msg.warning("No logged frames to plot, skipping plotting.")
        return
    if path is not None and not os.path.isdir(path):
        raise ValueError(
            f"Plot output directory '{path}' does not exist. Please create it before calling plot_logger_comparison()."
        )

    plt = SolutionMetricsLogger.plt

    time_metrics = logger_metrics.time_axis()
    time_solver = logger_solver.time_axis()
    x_label = "Time (s)" if logger_metrics.dt is not None else "Simulation Step"

    np_metrics = logger_metrics.to_numpy()
    np_solver = logger_solver.to_numpy()

    nw_metrics = logger_metrics.num_worlds
    nw_solver = logger_solver.num_worlds

    for field in _SCALAR_METRIC_FIELDS:
        equation = _METRIC_EQUATIONS[field]
        base_title = _METRIC_TITLES[field]
        title = f"{base_title} \n ({equation})"

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for w in range(nw_metrics):
            world_label = f" (world_{w})" if nw_metrics > 1 else ""
            ax.plot(
                time_metrics,
                np_metrics[field][:, w],
                color="blue",
                marker="o",
                markersize=3,
                linestyle="-",
                label=f"{label_metrics}{world_label}",
            )
        for w in range(nw_solver):
            world_label = f" (world_{w})" if nw_solver > 1 else ""
            ax.plot(
                time_solver,
                np_solver[field][:, w],
                color="red",
                marker="x",
                markersize=3,
                linestyle="--",
                label=f"{label_solver}{world_label}",
            )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(field)
        ax.grid()
        ax.legend(loc="best", frameon=False)

        fig.tight_layout()

        if path is not None:
            fig_path = os.path.join(path, f"{field}.{ext}")
            fig.savefig(fig_path, format=ext, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)


###
# Stepwise per-builder cross-check tests
###


class TestSolutionMetricsNewton(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = True
        self.progress = True
        self.savefig = True
        self.show = False
        self.output_path = test_context.output_path / "test_metrics"
        if self.savefig:
            self.output_path.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def _run_one(
        self,
        builder_fn: Callable,
        builder_kwargs: dict,
        builder_name: str,
        max_world_contacts: int,
        max_frames: int,
        num_frames: int,
    ):
        """
        Execute the stepwise comparison for one (path, builder) combination.

        Args:
            builder_fn: The builder function to invoke.
            builder_kwargs: Builder keyword arguments.
            builder_name: Builder name used as the plot output subdirectory.
            max_world_contacts: Per-world rigid-contact capacity.
        """
        request_state_attributes = ("body_parent_f", "joint_parent_f")

        setup = TestSetup(
            builder_fn=builder_fn,
            builder_kwargs=builder_kwargs,
            request_state_attributes=request_state_attributes,
            device=self.default_device,
            max_world_contacts=max_world_contacts,
            max_frames=max_frames,
        )

        start_time = time.time()
        for frame in range(num_frames):
            setup.step()
            if self.progress:
                print_progress_bar(frame + 1, num_frames, start_time, prefix="Progress", suffix="")

        self.assertEqual(setup.logger_metrics.num_logged_frames, num_frames)
        self.assertEqual(setup.logger_solver.num_logged_frames, num_frames)

        fields = METRIC_FIELDS_LAMBDA_INDEPENDENT
        fields = fields + METRIC_FIELDS_LAMBDA_DEPENDENT

        _assert_loggers_match(
            self,
            setup.logger_metrics,
            setup.logger_solver,
            fields=fields,
        )

        if self.savefig or self.show:
            path_label = "joint_parent_f"
            plot_dir = self.output_path / path_label / builder_name
            save_path: str | None = None
            if self.savefig:
                plot_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(plot_dir)
            msg.notif(f"Generating overlay plots for {path_label} / {builder_name}...")
            plot_logger_comparison(
                logger_metrics=setup.logger_metrics,
                logger_solver=setup.logger_solver,
                path=save_path,
                show=self.show,
                ext="pdf",
            )

    ###
    # Tests
    ###

    def test_boxes_fourbar(self):
        self._run_one(
            builder_fn=basics.build_boxes_fourbar,
            builder_kwargs={"z_offset": -1e-5, "floatingbase": True},
            builder_name="boxes_fourbar",
            max_world_contacts=32,
            max_frames=500,
            num_frames=100,
        )


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
