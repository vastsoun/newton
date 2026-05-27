# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `solvers/metrics.py`."""

import unittest
from collections.abc import Iterable, Mapping

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.bodies import update_body_wrenches
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.dynamics.wrenches import compute_constraint_body_wrenches
from newton._src.solvers.kamino._src.integrators.euler import integrate_euler_semi_implicit
from newton._src.solvers.kamino._src.kinematics.jacobians import SparseSystemJacobians, SystemJacobiansType
from newton._src.solvers.kamino._src.models.builders.basics import build_box_on_plane, build_boxes_hinged
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver
from newton._src.solvers.kamino._src.solvers.padmm.types import PADMMData
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_padmm import TestSetup
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_cts_jacobians,
    extract_delassus,
    extract_info_vectors,
    extract_problem_vector,
)

###
# Constants
###

# Per-world residual fields produced by `SolutionMetrics`. Each entry has a
# matching ``<field>_argmax`` array on `SolutionMetricsData` storing the
# argmax constraint index that achieved the residual on that world.
METRIC_RESIDUAL_FIELDS: tuple[str, ...] = (
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
)

# Per-world objective-value fields (no ``_argmax`` counterpart).
METRIC_OBJECTIVE_FIELDS: tuple[str, ...] = ("f_ccp", "f_ncp")

# All metric fields exposed for cross-implementation comparisons (e.g. dense vs sparse).
METRIC_COMPARE_FIELDS: tuple[str, ...] = METRIC_RESIDUAL_FIELDS + METRIC_OBJECTIVE_FIELDS

# Mapping from `compute_metrics_numpy` output keys to `SolutionMetricsData` field names.
_NUMPY_METRIC_FIELD_MAP: dict[str, str] = {
    "r_v_plus": "r_v_plus",
    "f_ccp": "f_ccp",
    "f_ncp": "f_ncp",
    "r_ncp_p": "r_ncp_primal",
    "r_ncp_d": "r_ncp_dual",
    "r_ncp_c": "r_ncp_compl",
    "r_vi_natmap": "r_vi_natmap",
}


###
# Helpers
###


def _integrate_and_sync_state(test: TestSetup) -> None:
    """Integrate one Euler-semi-implicit step and sync ``state.u_i = data.bodies.u_i``.

    The metrics kernel reads ``state.u_i`` to evaluate the reference
    ``v_plus_true = J_cts @ u^+``, so the post-event body twists produced by the
    integrator must be mirrored onto ``state.u_i`` before calling
    :meth:`SolutionMetrics.evaluate`.
    """
    integrate_euler_semi_implicit(model=test.model, data=test.data)
    wp.copy(test.state.u_i, test.data.bodies.u_i)


def _solve_padmm_and_propagate(test: TestSetup, solver: PADMMSolver) -> None:
    """Solve the dual problem with PADMM and propagate the solution to the post-event state.

    Steps:
        1. Cold-start the solver and solve ``test.problem``.
        2. Convert the solver's constraint impulses (lambdas) into per-body
           constraint wrenches and add them into ``data.bodies.w_i``. Without
           this step the integrator's post-event velocity equals the
           unconstrained free velocity ``v_f`` and the metrics would not see
           the impulse contribution ``M^{-1} J_cts^T lambda``.
        3. Run a single Euler-semi-implicit step and sync ``state.u_i`` to the
           post-event body twists (see :func:`_integrate_and_sync_state`).

    This mirrors what ``SolverKaminoImpl._solve_forward_dynamics`` does in
    production.
    """
    solver.reset()
    solver.coldstart()
    solver.solve(problem=test.problem)

    compute_constraint_body_wrenches(
        model=test.model,
        data=test.data,
        limits=test.limits,
        contacts=test.contacts,
        jacobians=test.jacobians,
        lambdas_offsets=test.problem.data.vio,
        lambdas_data=solver.data.solution.lambdas,
    )
    update_body_wrenches(test.model.bodies, test.data.bodies)
    _integrate_and_sync_state(test)


def _evaluate_metrics(
    metrics: SolutionMetrics,
    test: TestSetup,
    lambdas: wp.array,
    v_plus: wp.array,
    *,
    problem: DualProblem | None = None,
    jacobians: SystemJacobiansType | None = None,
) -> None:
    """Call :meth:`SolutionMetrics.evaluate` with `test`'s containers.

    ``problem`` and ``jacobians`` default to ``test.problem`` and ``test.jacobians``;
    pass overrides to evaluate against e.g. an independently-built sparse problem.
    """
    metrics.evaluate(
        lambdas=lambdas,
        v_plus=v_plus,
        state=test.state,
        state_p=test.state_p,
        problem=problem if problem is not None else test.problem,
        jacobians=jacobians if jacobians is not None else test.jacobians,
        limits=test.limits,
        contacts=test.contacts,
    )


def _log_active_constraint_counts(test: TestSetup) -> tuple[int, int]:
    """Log via ``msg.info`` and return the active limit/contact counts on world 0."""
    nl = int(test.limits.model_active_limits.numpy()[0]) if test.limits.model_max_limits_host > 0 else 0
    nc = int(test.contacts.model_active_contacts.numpy()[0]) if test.contacts.model_max_contacts_host > 0 else 0
    msg.info("num active limits: %s", nl)
    msg.info("num active contacts: %s\n", nc)
    return nl, nc


def _log_metric_values(
    metrics: SolutionMetrics,
    fields: Iterable[str] = METRIC_RESIDUAL_FIELDS,
) -> None:
    """Log per-world values and ``_argmax`` indices for each `field` via ``msg.info``."""
    fields = tuple(fields)
    for field in fields:
        msg.info(f"metrics.{field}: %s", getattr(metrics.data, field))
    msg.info("")
    for field in fields:
        msg.info(f"metrics.{field}_argmax: %s", getattr(metrics.data, f"{field}_argmax"))
    msg.info("")


def _compute_max_contact_penetration(test: TestSetup) -> float:
    """Maximum positive contact penetration depth across the active contacts of world 0.

    Reads ``test.contacts.gapfunc[cid][3]`` over the active contact range and
    returns the largest value, which is the residual that ``r_cts_contacts``
    should equal on a converged (or trivially zero) impulse solution.
    """
    nc = int(test.contacts.model_active_contacts.numpy()[0])
    gap = test.contacts.gapfunc.numpy()
    max_pen = 0.0
    for cid in range(nc):
        max_pen = max(max_pen, float(gap[cid][3]))
    return max_pen


def _perturb_array(arr: wp.array, rng: np.random.Generator, scale: float = 0.1) -> None:
    """Add Gaussian noise of the given scale to a Warp ``float32`` array, in place."""
    arr_np = arr.numpy()
    arr_np += scale * rng.standard_normal(arr_np.shape, dtype=np.float32)
    arr.assign(arr_np)


def _build_v_plus_true(test: TestSetup) -> list[np.ndarray]:
    """Build the per-world reference ``v^+ = J_cts @ u^+`` in float64 numpy.

    Mirrors ``SolutionMetrics._eval_reference_quantities``: extracts the dense
    constraint Jacobians (active rows only) per world and contracts them
    against the post-event body twists in ``test.data.bodies.u_i``.
    """
    j_cts_per_world = extract_cts_jacobians(
        model=test.model,
        limits=test.limits,
        contacts=test.contacts,
        jacobians=test.jacobians,
        only_active_cts=True,
    )
    bodies_offset = test.model.info.bodies_offset.numpy()
    u_i_post = test.data.bodies.u_i.numpy()
    v_plus_true: list[np.ndarray] = []
    for world_id in range(test.model.size.num_worlds):
        bio = int(bodies_offset[world_id])
        nb = int(bodies_offset[world_id + 1] - bio)
        u_world = u_i_post[bio : bio + nb].astype(np.float64).reshape(6 * nb)
        v_plus_true.append(j_cts_per_world[world_id].astype(np.float64) @ u_world)
    return v_plus_true


def _assert_per_world_array_size(
    testcase: unittest.TestCase,
    metrics: SolutionMetrics,
    num_worlds: int,
    fields: Iterable[str] = METRIC_RESIDUAL_FIELDS,
) -> None:
    """Assert each ``metrics.data.<field>`` and ``<field>_argmax`` array has size ``num_worlds``."""
    for field in fields:
        arr = getattr(metrics.data, field)
        argmax = getattr(metrics.data, f"{field}_argmax")
        testcase.assertEqual(arr.size, num_worlds, msg=f"metrics.data.{field}.size != {num_worlds}")
        testcase.assertEqual(argmax.size, num_worlds, msg=f"metrics.data.{field}_argmax.size != {num_worlds}")


def _assert_metrics_zero(
    testcase: unittest.TestCase,
    metrics: SolutionMetrics,
    max_contact_penetration: float,
    *,
    rtol: float = 1e-7,
    atol: float = 0.0,
    atol_overrides: Mapping[str, float] | None = None,
) -> None:
    """Assert every :data:`METRIC_RESIDUAL_FIELDS` entry on world 0 is zero (or
    ``max_contact_penetration`` for ``r_cts_contacts``).

    Each field is compared via :func:`numpy.testing.assert_allclose`. The defaults
    ``(rtol=1e-7, atol=0.0)`` reproduce the bare ``assert_allclose(x, 0.0)``
    semantics used by ``test_02``. Pass ``atol=1e-5`` (matching ~5 decimal places)
    for solver-driven tests, and ``atol_overrides`` to relax specific fields
    (e.g. ``{"r_ncp_dual": 1e-4}`` for ``test_04``).
    """
    overrides = atol_overrides or {}
    for field in METRIC_RESIDUAL_FIELDS:
        expected = max_contact_penetration if field == "r_cts_contacts" else 0.0
        actual = float(getattr(metrics.data, field).numpy()[0])
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=overrides.get(field, atol),
            err_msg=f"metrics.data.{field}[0]={actual!r} != expected={expected!r}",
        )


def _assert_metrics_match(
    testcase: unittest.TestCase,
    metrics_a: SolutionMetrics,
    metrics_b: SolutionMetrics,
    fields: Iterable[str] = METRIC_COMPARE_FIELDS,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """Cross-check two :class:`SolutionMetrics` instances field-by-field."""
    for field in fields:
        np.testing.assert_allclose(
            getattr(metrics_a.data, field).numpy(),
            getattr(metrics_b.data, field).numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"metrics field `{field}` does not match between the two SolutionMetrics instances",
        )


def _assert_metrics_match_numpy(
    testcase: unittest.TestCase,
    metrics: SolutionMetrics,
    metrics_np: dict,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """Cross-check `metrics` against the float64 numpy reference produced by :func:`compute_metrics_numpy`."""
    for np_key, field in _NUMPY_METRIC_FIELD_MAP.items():
        np.testing.assert_allclose(
            metrics_np[np_key],
            getattr(metrics.data, field).numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"metrics.data.{field} does not match numpy reference `{np_key}`",
        )


def compute_metrics_numpy(
    problem: DualProblem,
    solver_data: PADMMData,
    v_plus_true: list[np.ndarray],
) -> dict[np.ndarray]:
    """Compute the solver metrics with numpy, using float64.

    Args:
        problem: The dual forward dynamics problem of the current time-step.
        solver_data: The current dual solution and intermediate solver state.
        v_plus_true: Per-world post-event constraint-space velocities ``v^+ = J_cts @ u^+``,
            extracted to one ``(ncts,)`` ``np.ndarray`` per world (active rows only). This
            mirrors the reference quantity that :class:`SolutionMetrics` builds internally
            from ``state.u_i`` and the constraint Jacobians.
    """
    output = {}
    output["r_v_plus"] = []
    output["s"] = []
    output["f_ccp"] = []
    output["f_ncp"] = []
    output["v_aug"] = []
    output["r_ncp_p"] = []
    output["r_ncp_d"] = []
    output["r_ncp_c"] = []
    output["r_vi_natmap"] = []

    D = extract_delassus(problem.delassus, only_active_dims=True)
    num_matrices = len(D)

    lambdas = extract_problem_vector(problem.delassus, solver_data.solution.lambdas.numpy().astype(np.float64), True)
    v_plus_est = extract_problem_vector(problem.delassus, solver_data.solution.v_plus.numpy().astype(np.float64), True)
    v_f = extract_problem_vector(problem.delassus, problem.data.v_f.numpy().astype(np.float64), True)
    P = extract_problem_vector(problem.delassus, problem.data.P.numpy().astype(np.float64), True)

    mu = extract_info_vectors(
        problem.data.cio.numpy(), problem.data.mu.numpy().astype(np.float64), problem.delassus.info.dim.numpy()
    )

    num_joint_cts = problem.data.njc.numpy()
    num_contacts = problem.data.nc.numpy()
    num_limits = problem.data.nl.numpy()
    contact_group_offset = problem.data.ccgo.numpy()
    limit_group_offset = problem.data.lcgo.numpy()

    # Match `compute_ccp_objectiv_product` in `solvers/metrics.py`, which adds a small
    # numerical epsilon to the diagonal preconditioner before dividing.
    eps_ccp = 1.0e-6

    for mat_id in range(num_matrices):
        lambdas_i = lambdas[mat_id]
        v_plus_est_i = v_plus_est[mat_id]
        v_f_i = v_f[mat_id]
        mu_i = mu[mat_id]
        P_i = P[mat_id]

        # Take the post-event constraint-space velocity directly from the externally-provided
        # reference, i.e. `v^+ = J_cts @ u^+`. This matches the new kernel implementation,
        # which computes its reference v_plus the same way (see `SolutionMetrics._eval_reference_quantities`).
        v_plus_true_i = v_plus_true[mat_id].astype(np.float64)
        # Compute the post-event constraint-space velocity error as: r_v_plus = || v_plus_est - v_plus_true ||_inf
        r_v_plus_i = np.max(np.abs(v_plus_est_i - v_plus_true_i))
        output["r_v_plus"].append(r_v_plus_i)

        # Compute the De Saxce correction for each contact as: s = G(v_plus)
        s_i = np.zeros_like(v_plus_true_i)
        for contact_id in range(num_contacts[mat_id]):
            v_idx = contact_group_offset[mat_id] + 3 * contact_id
            s_i[v_idx + 2] = mu_i[contact_id] * np.linalg.norm(v_plus_true_i[v_idx : v_idx + 2])
        output["s"].append(s_i)

        # Compute the CCP optimization objective as: f_ccp = 0.5 * lambda.dot(v_plus + v_f / (P + eps)).
        # `problem.data.v_f` stores the preconditioned free velocity `v_f_p = P @ v_f_unprec`,
        # so dividing by `P` element-wise recovers the unpreconditioned `v_f` that the
        # CCP objective is defined against (matches `compute_ccp_objectiv_product` in the kernel,
        # including its small epsilon term).
        f_ccp_i = 0.5 * lambdas_i.dot(v_plus_true_i + v_f_i / (P_i + eps_ccp))
        output["f_ccp"].append(f_ccp_i)

        # Compute the NCP optimization objective as:  f_ncp = f_ccp + lambda.dot(s)
        f_ncp_i = f_ccp_i + lambdas_i.dot(s_i)
        output["f_ncp"].append(f_ncp_i)

        # Compute the augmented post-event constraint-space velocity as: v_aug = v_plus + s
        v_aug_i = v_plus_true_i + s_i
        output["v_aug"].append(v_aug_i)

        # Compute the NCP primal residual as: r_p := || lambda - proj_K(lambda) ||_inf
        r_ncp_p_i = 0.0
        for limit_id in range(num_limits[mat_id]):
            lcio = limit_group_offset[mat_id] + limit_id
            r_ncp_p_i = np.max(r_ncp_p_i, np.abs(lambdas_i[lcio] - np.max(0.0, lambdas_i[lcio])))

        def project_to_coulomb_cone(x, mu):
            xt_norm = np.linalg.norm(x[:2])
            if mu * xt_norm > -x[2]:
                if xt_norm <= mu * x[2]:
                    return x
                else:
                    ys = (mu * xt_norm + x[2]) / (mu * mu + 1.0)
                    yts = mu * ys / xt_norm
                    return np.array([yts * x[0], yts * x[1], ys])
            return np.zeros(3)

        for contact_id in range(num_contacts[mat_id]):
            ccio = contact_group_offset[mat_id] + 3 * contact_id
            lambda_c = lambdas_i[ccio : ccio + 3] - project_to_coulomb_cone(
                lambdas_i[ccio : ccio + 3], mu_i[contact_id]
            )
            r_ncp_p_i = np.max([r_ncp_p_i, np.max(np.abs(lambda_c))])

        output["r_ncp_p"].append(r_ncp_p_i)

        # Compute the NCP dual residual as: r_d := || v_plus + s - proj_dual_K(v_plus + s)  ||_inf
        r_ncp_d_i = 0.0
        for jid in range(num_joint_cts[mat_id]):
            v_j = v_aug_i[jid]
            r_j = np.abs(v_j)
            r_ncp_d_i = max(r_ncp_d_i, r_j)

        for lid in range(num_limits[mat_id]):
            v_l = float(v_aug_i[limit_group_offset[mat_id] + lid])
            v_l -= np.max(0.0, v_l)
            r_l = np.abs(v_l)
            r_ncp_d_i = max(r_ncp_d_i, r_l)

        def project_to_coulomb_dual_cone(x: np.ndarray, mu: float) -> np.ndarray:
            xn = x[2]
            xt_norm = np.linalg.norm(x[:2])
            y = np.zeros(3)
            if xt_norm > -mu * xn:
                if mu * xt_norm <= xn:
                    y = x
                else:
                    ys = (xt_norm + mu * xn) / (mu * mu + 1.0)
                    yts = ys / xt_norm
                    y[0] = yts * x[0]
                    y[1] = yts * x[1]
                    y[2] = mu * ys
            return y

        for cid in range(num_contacts[mat_id]):
            ccio_c = contact_group_offset[mat_id] + 3 * cid
            mu_c = mu_i[cid]
            v_c = v_aug_i[ccio_c : ccio_c + 3].copy()
            v_c -= project_to_coulomb_dual_cone(v_c, mu_c)
            r_c = np.max(np.abs(v_c))
            r_ncp_d_i = max(r_ncp_d_i, r_c)

        output["r_ncp_d"].append(r_ncp_d_i)

        # Compute the NCP complementarity (lambda _|_ (v_plus + s)) residual as r_c := || lambda.dot(v_plus + s) ||_inf
        r_ncp_c_i = 0.0
        for lid in range(num_limits[mat_id]):
            lcio = limit_group_offset[mat_id] + lid
            v_l = v_aug_i[lcio]
            lambda_l = lambdas_i[lcio]
            r_l = np.abs(v_l * lambda_l)
            r_ncp_c_i = max(r_ncp_c_i, r_l)

        for cid in range(num_contacts[mat_id]):
            ccio = contact_group_offset[mat_id] + 3 * cid
            v_c = v_aug_i[ccio : ccio + 3]
            lambda_c = lambdas_i[ccio : ccio + 3]
            r_c = np.abs(np.dot(v_c, lambda_c))
            r_ncp_c_i = max(r_ncp_c_i, r_c)
        output["r_ncp_c"].append(r_ncp_c_i)

        # Compute the natural-map residuals as: r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf
        r_vi_natmap_i = 0.0
        for lid in range(num_limits[mat_id]):
            lcio = limit_group_offset[mat_id] + lid
            v_l = v_aug_i[lcio]
            lambda_l = lambdas_i[lcio]
            lambda_l -= np.max(0.0, lambda_l - v_l)
            lambda_l = np.abs(lambda_l)
            r_vi_natmap_i = max(r_vi_natmap_i, lambda_l)

        for cid in range(num_contacts[mat_id]):
            ccio = contact_group_offset[mat_id] + 3 * cid
            mu_c = mu_i[cid]
            v_c = v_aug_i[ccio : ccio + 3]
            lambda_c = lambdas_i[ccio : ccio + 3]
            lambda_c -= project_to_coulomb_cone(lambda_c - v_c, mu_c)
            lambda_c = np.abs(lambda_c)
            lambda_c_max = np.max(lambda_c)
            r_vi_natmap_i = max(r_vi_natmap_i, lambda_c_max)

        output["r_vi_natmap"].append(r_vi_natmap_i)

    return output


###
# Tests
###


class TestSolverMetrics(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.seed = 42

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        """
        Test creating a SolutionMetrics instance with default initialization.
        """
        # Creating a default solver metrics evaluator without any model
        # should result in an instance without any memory allocation.
        metrics = SolutionMetrics()
        self.assertIsNone(metrics._device)
        self.assertIsNone(metrics._data)
        self.assertIsNone(metrics._buffer_s)
        self.assertIsNone(metrics._buffer_v)

        # Requesting the solver data container when the
        # solver has not been finalized should raise an
        # error since no allocations have been made.
        self.assertRaises(RuntimeError, lambda: metrics.data)

    def test_01_finalize_default(self):
        """
        Test creating a SolutionMetrics instance with default initialization and then finalizing all memory allocations.
        """
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Creating a default solver metrics evaluator without any model
        # should result in an instance without any memory allocation.
        metrics = SolutionMetrics()
        metrics.finalize(model=test.model, data=test.data)

        # Check that the solver has been properly allocated
        self.assertIsNotNone(metrics._data)
        self.assertIsNotNone(metrics._device)
        self.assertIs(metrics._device, test.model.device)
        self.assertIsNotNone(metrics._buffer_s)
        self.assertIsNotNone(metrics._buffer_v)

        # Check buffer-allocation sizes
        msg.info("num_worlds: %s", test.model.size.num_worlds)
        msg.info("sum_of_max_total_cts: %s", test.model.size.sum_of_max_total_cts)
        msg.info("buffer_s size: %s", metrics._buffer_s.size)
        msg.info("buffer_v size: %s", metrics._buffer_v.size)
        self.assertEqual(metrics._buffer_s.size, test.model.size.sum_of_max_total_cts)
        self.assertEqual(metrics._buffer_v.size, test.model.size.sum_of_max_total_cts)

        # Check per-world residual / argmax allocation sizes
        _assert_per_world_array_size(self, metrics, test.model.size.num_worlds)

    def test_02_evaluate_trivial_solution(self):
        """
        Tests evaluating metrics on an all-zeros trivial solution.
        """
        test = TestSetup(
            builder_fn=build_box_on_plane,
            max_world_contacts=4,
            gravity=False,
            perturb=False,
            device=self.default_device,
        )
        metrics = SolutionMetrics(model=test.model, data=test.data)

        # Define a trivial solution (all zeros)
        with wp.ScopedDevice(test.model.device):
            lambdas = wp.zeros(test.model.size.sum_of_max_total_cts, dtype=wp.float32)
            v_plus = wp.zeros(test.model.size.sum_of_max_total_cts, dtype=wp.float32)

        # Build the test problem and integrate the state over a single time-step
        test.build()
        _integrate_and_sync_state(test)

        nl, nc = _log_active_constraint_counts(test)
        self.assertEqual(nl, 0)
        self.assertEqual(nc, 4)

        # Compute the metrics on the trivial solution
        metrics.reset()
        _evaluate_metrics(metrics, test, lambdas, v_plus)
        _log_metric_values(metrics)

        # Check that all residuals are zero (except r_cts_contacts, which equals
        # the maximum contact penetration on a zero-impulse solution).
        _assert_metrics_zero(self, metrics, _compute_max_contact_penetration(test))

        # Check that all argmax indices match this builder's expected layout.
        np.testing.assert_allclose(metrics.data.r_eom_argmax.numpy()[0], 0)  # only one body
        np.testing.assert_allclose(metrics.data.r_kinematics_argmax.numpy()[0], -1)  # no joints
        np.testing.assert_allclose(metrics.data.r_cts_joints_argmax.numpy()[0], -1)  # no joints
        np.testing.assert_allclose(metrics.data.r_cts_limits_argmax.numpy()[0], -1)  # no limits
        # NOTE: all contacts have the same residual,
        # so the argmax evaluates to the last constraint
        np.testing.assert_allclose(metrics.data.r_v_plus_argmax.numpy()[0], 11)
        # NOTE: all contacts have the same penetration,
        # so the argmax evaluates to the last contact
        np.testing.assert_allclose(metrics.data.r_cts_contacts_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_primal_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_dual_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_compl_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_vi_natmap_argmax.numpy()[0], 3)

    def test_03_evaluate_padmm_solution_box_on_plane(self):
        """
        Tests evaluating metrics on a solution computed with the Proximal-ADMM (PADMM) solver.
        """
        test = TestSetup(
            builder_fn=build_box_on_plane,
            max_world_contacts=4,
            gravity=True,
            perturb=True,
            device=self.default_device,
        )
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)
        metrics = SolutionMetrics(model=test.model, data=test.data)

        test.build()
        _solve_padmm_and_propagate(test, solver)

        metrics.reset()
        _evaluate_metrics(metrics, test, solver.data.solution.lambdas, solver.data.solution.v_plus)

        _log_active_constraint_counts(test)
        _log_metric_values(metrics)

        # Check that all residuals are ~zero (except r_cts_contacts, which equals
        # the maximum contact penetration on a converged solution).
        _assert_metrics_zero(self, metrics, _compute_max_contact_penetration(test), atol=1e-5)

    def test_04_evaluate_padmm_solution_boxes_hinged(self):
        """
        Tests evaluating metrics on a solution computed with the Proximal-ADMM (PADMM) solver.
        """
        test = TestSetup(
            builder_fn=build_boxes_hinged,
            max_world_contacts=8,
            gravity=True,
            perturb=True,
            device=self.default_device,
        )
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)
        metrics = SolutionMetrics(model=test.model, data=test.data)

        test.build()
        _solve_padmm_and_propagate(test, solver)
        _evaluate_metrics(metrics, test, solver.data.solution.lambdas, solver.data.solution.v_plus)

        _log_active_constraint_counts(test)
        _log_metric_values(metrics)

        # `r_ncp_dual` is one decimal place less accurate on this builder but
        # still correct; everything else converges to ~5 decimal places.
        _assert_metrics_zero(
            self,
            metrics,
            _compute_max_contact_penetration(test),
            atol=1e-5,
            atol_overrides={"r_ncp_dual": 1e-4},
        )

    def test_05_validate_metrics_boxes_hinged(self):
        """
        Compares metrics from `SolutionMetrics` with metrics computed by a
        reference routine using float64 numpy arrays, on a perturbed PADMM solution.
        """
        test = TestSetup(
            builder_fn=build_boxes_hinged,
            max_world_contacts=8,
            gravity=True,
            perturb=True,
            device=self.default_device,
            sparse=False,
        )
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)
        metrics = SolutionMetrics(model=test.model, data=test.data)

        test.build()
        _solve_padmm_and_propagate(test, solver)

        # Perturb the solver's solution to obtain non-trivial metrics
        rng = np.random.default_rng(seed=self.seed)
        _perturb_array(solver.data.solution.lambdas, rng, scale=0.1)
        _perturb_array(solver.data.solution.v_plus, rng, scale=0.1)

        # Evaluate metrics on the perturbed solution
        _evaluate_metrics(metrics, test, solver.data.solution.lambdas, solver.data.solution.v_plus)

        rtol = 1e-6
        atol = 1e-6

        # Compute the float64 numpy reference and cross-check against the kernel-side metrics
        v_plus_true_per_world = _build_v_plus_true(test)
        metrics_np = compute_metrics_numpy(test.problem, solver.data, v_plus_true=v_plus_true_per_world)
        for key, value in metrics_np.items():
            msg.info(f"{key}: {value}")
        _assert_metrics_match_numpy(self, metrics, metrics_np, rtol=rtol, atol=atol)

        # Cross-check `s` (stored in `_buffer_s`) and `v_aug` (stored in `_buffer_v`),
        # which are not exposed as scalar metric fields. This is a somewhat hacky way
        # to inspect intermediate quantities computed by the metrics kernel.
        s = extract_problem_vector(test.problem.delassus, metrics._buffer_s.numpy(), True)
        v_aug = extract_problem_vector(test.problem.delassus, metrics._buffer_v.numpy(), True)
        for world_id in range(test.model.size.num_worlds):
            np.testing.assert_allclose(metrics_np["s"][world_id], s[world_id], rtol=rtol, atol=atol)
            np.testing.assert_allclose(metrics_np["v_aug"][world_id], v_aug[world_id], rtol=rtol, atol=atol)

    def test_06_compare_dense_sparse_boxes_hinged(self):
        """
        Compares metrics evaluated on dense and sparse problems on a perturbed
        PADMM solution.
        """
        test = TestSetup(
            builder_fn=build_boxes_hinged,
            max_world_contacts=8,
            gravity=True,
            perturb=True,
            device=self.default_device,
            sparse=False,
        )
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)
        metrics_dense = SolutionMetrics(model=test.model, data=test.data)
        metrics_sparse = SolutionMetrics(model=test.model, data=test.data)

        # Build a sparse counterpart of the dense Jacobians and dual problem.
        jacobians_sparse = SparseSystemJacobians(
            model=test.model,
            limits=test.limits,
            contacts=test.detector.contacts,
        )
        jacobians_sparse.build(
            model=test.model,
            data=test.data,
            limits=test.limits.data,
            contacts=test.detector.contacts.data,
        )
        problem_sparse = DualProblem(
            model=test.model,
            data=test.data,
            limits=test.limits,
            contacts=test.contacts,
            jacobians=jacobians_sparse,
            sparse=True,
        )
        problem_sparse.build(
            model=test.model,
            data=test.data,
            jacobians=jacobians_sparse,
            limits=test.limits,
            contacts=test.detector.contacts,
        )

        # Solve the (dense) test problem and propagate to the post-event state.
        test.build()
        _solve_padmm_and_propagate(test, solver)

        # Bring the sparse problem's regularized Delassus into the same state.
        solver._initialize()
        solver._update_sparse_regularization(problem_sparse)
        problem_sparse.delassus.update()

        # Perturb the solver's solution to obtain non-trivial metrics
        rng = np.random.default_rng(seed=self.seed)
        _perturb_array(solver.data.solution.lambdas, rng, scale=1.0)
        _perturb_array(solver.data.solution.v_plus, rng, scale=1.0)

        # Evaluate metrics for the dense and sparse paths
        _evaluate_metrics(metrics_dense, test, solver.data.solution.lambdas, solver.data.solution.v_plus)
        _evaluate_metrics(
            metrics_sparse,
            test,
            solver.data.solution.lambdas,
            solver.data.solution.v_plus,
            problem=problem_sparse,
            jacobians=jacobians_sparse,
        )

        rtol = 1e-6
        atol = 1e-6

        # Compare Jacobians (dense extraction vs sparse storage)
        j_cts_dense_np = extract_cts_jacobians(
            model=test.model,
            limits=test.limits,
            contacts=test.contacts,
            jacobians=test.jacobians,
            only_active_cts=True,
        )
        j_cts_sparse_np = jacobians_sparse._J_cts.bsm.numpy()
        for j_cts_dense_np_i, j_cts_sparse_np_i in zip(j_cts_dense_np, j_cts_sparse_np, strict=True):
            np.testing.assert_allclose(j_cts_dense_np_i, j_cts_sparse_np_i, rtol=rtol, atol=atol)

        # Compare Delassus matrices (dense vs sparse)
        d_dense_np = extract_delassus(delassus=test.problem.delassus, only_active_dims=True)
        d_sparse_np = extract_delassus(delassus=problem_sparse.delassus, only_active_dims=True)
        for d_dense_np_i, d_sparse_np_i in zip(d_dense_np, d_sparse_np, strict=True):
            np.testing.assert_allclose(d_dense_np_i, d_sparse_np_i, rtol=rtol, atol=atol)

        # Cross-check `v_aug` (stored in `_buffer_v`) and `s` (stored in `_buffer_s`).
        # Somewhat hacky way to inspect intermediate quantities computed by the metrics kernel.
        np.testing.assert_allclose(
            metrics_dense._buffer_v.numpy(), metrics_sparse._buffer_v.numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            metrics_dense._buffer_s.numpy(), metrics_sparse._buffer_s.numpy(), rtol=rtol, atol=atol
        )

        # Cross-check every scalar metric field
        _assert_metrics_match(self, metrics_dense, metrics_sparse, rtol=rtol, atol=atol)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
