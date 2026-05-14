# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import time
import unittest
from collections.abc import Callable

import numpy as np
import warp as wp

import newton
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.metrics import SolutionMetricsLogger, SolutionMetricsNewton
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.examples import print_progress_bar
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils import basics

###
# Constants
###

# Tolerance for *lambda-independent* metrics. These reduce to comparing per-step
# kinematic residuals and constraint gap functions on the post-event state, which
# both evaluation paths populate from identical inputs and through identical
# kernels, so bit-exact agreement is achievable.
TOL_STRICT_RTOL = 1e-7
TOL_STRICT_ATOL = 0.0

# Tolerance for *lambda-dependent* metrics. SolverKamino consumes its internally
# computed constraint reactions directly, whereas SolutionMetricsNewton recovers
# them from the float32 ``joint_parent_f`` snapshot via the inverse joint-wrench
# transform; that roundtrip is bounded below by ~1 ULP per scalar (~1e-9 for
# reactions of magnitude ~1e-2) and is amplified by ``1/dt`` when reconstructing
# body wrenches. The peak observed gap on `boxes_fourbar` is ~3e-6 in
# ``r_v_plus``/``r_ncp_dual``, so an atol/rtol of 1e-5 leaves ~3x headroom while
# still keeping the comparison meaningful.
TOL_LAMBDA_DEP_RTOL = 1e-5
TOL_LAMBDA_DEP_ATOL = 1e-5

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

# All metric fields
METRIC_FIELDS_ALL = METRIC_FIELDS_LAMBDA_INDEPENDENT + METRIC_FIELDS_LAMBDA_DEPENDENT

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
        sparse_jacobian: bool = False,
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

        # Finalise the Newton-side runtime containers
        self.model: Model = self.builder.finalize(skip_validation_joints=True)
        self.model.rigid_contact_max = max_world_contacts
        self.state: State = self.model.state()
        self.state_p: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        # Reference solver with metrics computation enabled
        solver_config = self._make_solver_config_default()
        solver_config.compute_solution_metrics = True
        solver_config.sparse_jacobian = sparse_jacobian
        self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)

        # Metrics evaluator
        metrics_model = self.builder.finalize(skip_validation_joints=True)
        metrics_model.rigid_contact_max = max_world_contacts
        self.metrics = SolutionMetricsNewton(
            model=metrics_model,
            dt=self.dt,
            sparse=sparse_jacobian,
        )

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

    def _run_test_setup(
        self,
        builder_fn: Callable,
        builder_kwargs: dict,
        builder_name: str,
        max_world_contacts: int,
        max_frames: int,
        num_frames: int,
        sparse_jacobian: bool = False,
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
            sparse_jacobian=sparse_jacobian,
        )

        start_time = time.time()
        for frame in range(num_frames):
            setup.step()
            if self.progress:
                print_progress_bar(frame + 1, num_frames, start_time, prefix="Progress", suffix="")

        self.assertEqual(setup.logger_metrics.num_logged_frames, num_frames)
        self.assertEqual(setup.logger_solver.num_logged_frames, num_frames)

        # Lambda-independent metrics: both paths derive these from the same
        # post-event state and constraint data, so we require bit-exact agreement.
        _assert_loggers_match(
            testcase=self,
            logger_metrics=setup.logger_metrics,
            logger_solver=setup.logger_solver,
            fields=METRIC_FIELDS_LAMBDA_INDEPENDENT,
            rtol=TOL_STRICT_RTOL,
            atol=TOL_STRICT_ATOL,
        )

        # Lambda-dependent metrics: SolutionMetricsNewton must reconstruct the
        # constraint reactions from the float32 joint_parent_f snapshot exported
        # by SolverKamino; the roundtrip introduces a ~1 ULP precision floor that
        # the inverse wrench transform and inv(dt) scaling propagate downstream.
        _assert_loggers_match(
            testcase=self,
            logger_metrics=setup.logger_metrics,
            logger_solver=setup.logger_solver,
            fields=METRIC_FIELDS_LAMBDA_DEPENDENT,
            rtol=TOL_LAMBDA_DEP_RTOL,
            atol=TOL_LAMBDA_DEP_ATOL,
        )

        if self.savefig or self.show:
            plot_dir = self.output_path / builder_name
            save_path: str | None = None
            if self.savefig:
                plot_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(plot_dir)
            msg.notif(f"Generating overlay plots for {builder_name}...")
            SolutionMetricsLogger.plot_comparison(
                loggers={"metrics": setup.logger_metrics, "solver": setup.logger_solver},
                path=save_path,
                show=self.show,
                grid=True,
                ext="pdf",
            )

    ###
    # Tests
    ###

    def test_with_joint_parent_f_on_boxes_fourbar(self):
        self._run_test_setup(
            builder_fn=basics.build_boxes_fourbar,
            builder_kwargs={"z_offset": -1e-5, "floatingbase": True, "limits": False},
            builder_name="boxes_fourbar",
            max_world_contacts=32,
            max_frames=500,
            num_frames=500,
            sparse_jacobian=True,
        )


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
