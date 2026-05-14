# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`SolutionMetricsNewton`.

The tests exercise :class:`SolutionMetricsNewton` end-to-end against a reference
:class:`SolverKamino` step. Both the metrics wrapper's :class:`DualProblem` and
the test-side solver are configured with all constraint stabilization
parameters set to zero and dual-problem preconditioning disabled, so the
intermediate quantities they produce are bit-comparable.

For each builder in the per-test coverage matrix the test:

#. Steps the solver to populate the Newton ``state.body_parent_f`` (and,
   for the joint-parent-f path, synthesises ``state.joint_parent_f`` from
   ``body_parent_f`` since the solver does not write it directly).
#. Calls :meth:`SolutionMetricsNewton.evaluate` with the relevant Newton
   containers.
#. Manually finalises and runs the wrapped :class:`SolutionMetrics` instance
   on the metrics-side intermediate buffers using the solver's ``sigma``
   (the call inside :meth:`SolutionMetricsNewton.evaluate` is currently
   commented out and the wrapper allocates ``_sigma`` with a layout that is
   incompatible with the dense metrics kernel).
#. Cross-checks the metrics-side
   :class:`DataKamino`, :class:`SystemJacobians`, :class:`DualProblem`,
   active ``v_plus``, active ``lambdas`` and :class:`SolutionMetricsData`
   against the corresponding solver-internal references.

The body-parent-f path can recover joint kinematic-constraint Lagrange
multipliers exactly only at *leaf* non-FREE joints (joints whose follower
body is not the base of any other non-FREE joint). For non-leaf joints the
recovered lambdas mix in contributions from descendant joints and likewise
contaminate every quantity they propagate into (body wrenches, EoM residual,
NCP residuals, etc.). The :func:`_compare_metrics_against_solver` driver
therefore restricts the body-parent-f comparisons to leaf-only joint indices
and the lambda-independent metric fields, while the joint-parent-f path
performs the full comparison.
"""

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.jacobians import (
    DenseSystemJacobians,
    SparseSystemJacobians,
)
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.metrics import SolutionMetricsLogger, SolutionMetricsNewton
from newton._src.solvers.kamino._src.metrics.logging import (
    _METRIC_EQUATIONS,
    _METRIC_TITLES,
    _SCALAR_METRIC_FIELDS,
)
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetrics, SolutionMetricsData
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_cts_jacobians,
    extract_dofs_jacobians,
    extract_problem_vector,
)
from newton.tests.utils import basics

###
# Constants
###

# Strict tolerance for quantities that are computed on identical inputs
# across both sides of the comparison (e.g. system Jacobians, dual-problem
# building blocks, post-event constraint-space velocity at active rows).
TOL_STRICT_RTOL = 1e-5
TOL_STRICT_ATOL = 1e-5

# Solver-converged tolerance for quantities that depend on the iterative
# PADMM solution (the active lambdas, and any scalar metric derived from
# them). The PADMM cold-start defaults are good to a few parts in ``1e-4``
# on the small builders used here.
TOL_SOLVED_RTOL = 1e-5
TOL_SOLVED_ATOL = 1e-5


###
# Helpers
###


def make_joint_parent_f_from_solver_state(solver: newton.solvers.SolverKamino) -> wp.array:
    """Build an exact per-joint ``joint_parent_f`` from the solver's Jacobians and lambdas.

    For each non-FREE joint ``j`` with follower body ``bid_F``, the world-frame
    wrench applied on body ``F`` by joint ``j`` (referenced at body ``F``'s
    centre of mass) is the linear combination of the post-event impulses
    along ``j``'s kinematic-constraint rows::

        joint_parent_f[j] = sum_{r in j's kinematic rows}
                                (J_cts[r, 6*bid_F : 6*bid_F+6]).T * (lambdas[r] / dt)

    Restricting the sum to the rows that belong to joint ``j`` (rather than to
    every kinematic row touching body ``F``) gives an exact per-joint wrench
    even for non-leaf joints in tree articulations and for builders with
    kinematic loops, where the body-parent-f synthesis blends contributions
    from multiple joints onto the shared follower body.

    The ``J_cts`` returned by :func:`extract_cts_jacobians` is built at the
    pre-event geometry that the solver also used to compute ``lambdas``;
    :class:`SolutionMetricsNewton` rebuilds its own Jacobians at the same
    pre-event geometry, so the round-trip through
    ``_compute_joint_wrenches_from_joint_parent_wrenches`` recovers the
    solver's ``lambdas`` to within the PADMM convergence tolerance.

    Args:
        solver: The configured :class:`SolverKamino` whose post-step state,
            Jacobians, and lambdas provide the inputs.

    Returns:
        A ``wp.array`` of ``wp.spatial_vectorf`` with shape ``(num_joints,)``,
        allocated on the same device as the solver-side Kamino model.
    """
    impl = solver._solver_kamino
    model_k = impl._model
    inv_dt = float(model_k.time.inv_dt.numpy()[0])

    # Per-world dense J_cts blocks. Use the *full* layout (active + inactive
    # rows) because each joint addresses its kinematic rows by world-local
    # offset, which is independent of how many limits / contacts are active.
    j_cts_per_world = extract_cts_jacobians(
        model_k,
        impl._limits,
        solver._contacts_kamino,
        impl._jacobians,
        only_active_cts=False,
    )

    bid_F_global = model_k.joints.bid_F.numpy()
    dof_type = model_k.joints.dof_type.numpy()
    wid_per_joint = model_k.joints.wid.numpy()
    kin_offset_joint_cts = model_k.joints.kinematic_cts_offset_joint_cts.numpy()
    num_kin_cts = model_k.joints.num_kinematic_cts.numpy()
    bodies_offset = model_k.info.bodies_offset.numpy()
    total_cts_offset = model_k.info.total_cts_offset.numpy()
    global_lambdas = impl._solver_fd.data.solution.lambdas.numpy()

    free_value = int(JointDoFType.FREE.value)
    num_joints = int(model_k.joints.num_joints)
    joint_parent_f_np = np.zeros((num_joints, 6), dtype=np.float32)

    for j in range(num_joints):
        if int(dof_type[j]) == free_value:
            continue
        n_kin = int(num_kin_cts[j])
        if n_kin == 0:
            continue

        w = int(wid_per_joint[j])
        bid_F_local = int(bid_F_global[j]) - int(bodies_offset[w])
        row_start_local = int(kin_offset_joint_cts[j])
        row_start_global = int(total_cts_offset[w]) + row_start_local

        J_cts_world = j_cts_per_world[w]
        J_block = J_cts_world[
            row_start_local : row_start_local + n_kin,
            6 * bid_F_local : 6 * bid_F_local + 6,
        ]
        lambdas_slice = global_lambdas[row_start_global : row_start_global + n_kin] * inv_dt

        joint_parent_f_np[j] = J_block.T @ lambdas_slice

    return wp.array(joint_parent_f_np, dtype=wp.spatial_vectorf, device=model_k.device)


def get_leaf_joint_lambda_indices(model_kamino: ModelKamino) -> np.ndarray:
    """Return the ``lambda_j`` indices corresponding to leaf non-FREE joints.

    A non-FREE joint ``j`` is a *leaf joint* when its follower body ``bid_F`` is
    not the base body ``bid_B`` of any other non-FREE joint. The convert
    kernels recover these joints' kinematic-constraint Lagrange multipliers
    exactly, because the per-body accumulator only contains contributions from
    joint ``j`` for that body.

    For non-leaf joints the recovered lambdas mix contributions from descendant
    joints (their base-side reactions land on the same follower body), and an
    exact comparison against the solver's reference lambdas is not expected to
    hold along the body-parent-f path.

    Args:
        model_kamino: The Kamino model whose joint topology is queried.

    Returns:
        A 1-D ``numpy.ndarray`` of indices into ``data.joints.lambda_j``.
    """
    bid_F = model_kamino.joints.bid_F.numpy()
    bid_B = model_kamino.joints.bid_B.numpy()
    dof_type = model_kamino.joints.dof_type.numpy()
    kin_offset = model_kamino.joints.kinematic_cts_offset_joint_cts.numpy()
    num_kin_cts = model_kamino.joints.num_kinematic_cts.numpy()
    free_value = int(JointDoFType.FREE.value)

    bases_of_non_free = {int(b) for b, dt in zip(bid_B, dof_type, strict=True) if dt != free_value and b >= 0}

    leaf_indices: list[int] = []
    for jid in range(len(dof_type)):
        if dof_type[jid] == free_value:
            continue
        if int(bid_F[jid]) in bases_of_non_free:
            continue
        start = int(kin_offset[jid])
        leaf_indices.extend(range(start, start + int(num_kin_cts[jid])))
    return np.asarray(leaf_indices, dtype=np.int64)


###
# Builder coverage matrices
###


def _builders_without_loops_for_body_parent_f() -> list[tuple[str, callable, dict, int]]:
    """Builders exercised by ``test_02_evaluate_body_parent_f_path``.

    Each entry is a ``(name, builder_fn, builder_kwargs, max_world_contacts)``
    tuple. ``boxes_fourbar`` is excluded because it has a kinematic loop, which
    Newton's articulation builder cannot expose via ``body_parent_f``.
    """
    return [
        ("box_on_plane", basics.build_box_on_plane, {"z_offset": -1e-5}, 8),
        ("cartpole", basics.build_cartpole, {"z_offset": -1e-5}, 8),
        ("boxes_hinged", basics.build_boxes_hinged, {"z_offset": -1e-5}, 32),
        ("boxes_nunchaku", basics.build_boxes_nunchaku, {"z_offset": -1e-5}, 32),
        ("boxes_nunchaku_vertical", basics.build_boxes_nunchaku_vertical, {"z_offset": -1e-5}, 32),
    ]


def _builders_with_loops_for_joint_parent_f() -> list[tuple[str, callable, dict, int]]:
    """Builders exercised by ``test_03_evaluate_joint_parent_f_path``.

    Adds ``boxes_fourbar`` (closed loop) to the body-parent-f set, with
    ``floatingbase=True`` so Newton can construct a valid articulation tree.
    """
    return [
        *_builders_without_loops_for_body_parent_f(),
        ("boxes_fourbar", basics.build_boxes_fourbar, {"z_offset": -1e-5, "floatingbase": True}, 32),
    ]


###
# Test scaffolding
###


def _make_zero_stab_solver_config() -> newton.solvers.SolverKamino.Config:
    """Build a :class:`SolverKamino.Config` matching :class:`SolutionMetricsNewton`.

    Mirrors the configuration that :meth:`SolutionMetricsNewton.finalize`
    applies to its internal :class:`DualProblem`: all stabilization terms
    zeroed and dual preconditioning disabled, so the metrics-side problem
    matches the solver-side problem field-by-field. The
    ``compute_solution_metrics`` flag is enabled to populate
    ``solver._solver_kamino._metrics`` with the reference
    :class:`SolutionMetricsData`.
    """
    config = newton.solvers.SolverKamino.Config()
    config.constraints = config.constraints or DualProblem.Config().constraints.__class__()
    config.dynamics = config.dynamics or DualProblem.Config().dynamics.__class__()
    config.constraints.alpha = 0.0
    config.constraints.beta = 0.0
    config.constraints.gamma = 0.0
    config.constraints.delta = 0.0
    config.dynamics.preconditioning = False
    config.compute_solution_metrics = True
    return config


class TestSetup:
    """Per-builder Newton-side scaffolding driving a stabilization-zero solver.

    The setup owns:
      * The Newton :class:`Model`, :class:`State` x2, :class:`Control`,
        :class:`Contacts` containers built from ``builder_fn``.
      * A :class:`SolverKamino` configured with all constraint stabilization
        parameters zeroed, preconditioning disabled and
        ``compute_solution_metrics=True``.

    The :class:`SolutionMetricsNewton` instance under test owns its own
    parallel set of Kamino containers (``_model``, ``_data``, ``_limits``,
    ``_contacts``, ``_jacobians``, ``_problem``); this class therefore does
    not duplicate them.
    """

    def __init__(
        self,
        builder_fn,
        builder_kwargs: dict | None = None,
        dt: float = 0.001,
        max_world_contacts: int = 32,
        request_state_attributes: tuple[str, ...] = (),
        device: wp.DeviceLike = None,
    ):
        # Cache scalar configuration
        self.dt = dt
        self.max_world_contacts = max_world_contacts

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

        # Create the reference solver with stabilization zeroed (mirroring
        # SolutionMetricsNewton.finalize) and metrics computation enabled.
        self.solver = newton.solvers.SolverKamino(model=self.model, config=_make_zero_stab_solver_config())

    def step_and_synthesize_joint_parent_f(self):
        """Step the solver and synthesise ``joint_parent_f`` if it was requested.

        :meth:`SolverKamino.step` populates ``state.body_parent_f`` but never
        writes ``state.joint_parent_f``. To exercise the joint-parent-f path
        of :meth:`SolutionMetricsNewton.evaluate` we synthesise the per-joint
        wrench directly from the solver's lambdas and Jacobians via
        :func:`make_joint_parent_f_from_solver_state`, which is exact for
        every non-FREE joint - including non-leaf joints in tree
        articulations and joints in kinematic loops, where the
        ``body_parent_f``-based synthesis blends contributions from multiple
        joints onto the shared follower body.
        """
        self.model.collide(self.state_p, self.contacts)
        self.solver.step(
            state_in=self.state_p,
            state_out=self.state,
            control=self.control,
            contacts=self.contacts,
            dt=self.dt,
        )

        if self.state.joint_parent_f is None:
            return
        joint_parent_f_synth = make_joint_parent_f_from_solver_state(self.solver)
        wp.copy(self.state.joint_parent_f, joint_parent_f_synth)


###
# Comparison helpers
###


def assert_models_equal_but_not_same_malloc(testcase: unittest.TestCase, model_1: Model, model_2: Model):
    """Assert that two :class:`Model` instances are equal but distinct allocations."""
    testcase.assertIsNotNone(model_1)
    testcase.assertIsNotNone(model_2)
    testcase.assertIsInstance(model_1, Model)
    testcase.assertIsInstance(model_2, Model)
    testcase.assertEqual(model_1.device, model_2.device)

    model_attributes = [
        "body_q",
        "body_qd",
        "body_mass",
        "body_inertia",
        "body_inv_mass",
        "body_inv_inertia",
        "body_flags",
        "body_label",
        "body_world",
        "body_world_start",
    ]

    for attribute in model_attributes:
        if not hasattr(model_1, attribute) or not hasattr(model_2, attribute):
            testcase.fail(f"Model attribute '{attribute}' is not found in one of the models.")
        attr_1 = getattr(model_1, attribute)
        attr_2 = getattr(model_2, attribute)
        if isinstance(attr_1, wp.array) and isinstance(attr_2, wp.array):
            np.testing.assert_equal(attr_1.numpy(), attr_2.numpy())
            testcase.assertNotEqual(attr_1.ptr, attr_2.ptr)
        elif isinstance(attr_1, np.ndarray) and isinstance(attr_2, np.ndarray):
            np.testing.assert_equal(attr_1, attr_2)
            testcase.assertNotEqual(attr_1.ptr, attr_2.ptr)
        elif isinstance(attr_1, (list, dict, tuple, set)) and isinstance(attr_2, type(attr_1)):
            testcase.assertEqual(attr_1, attr_2)
        else:
            testcase.fail(f"Model attribute '{attribute}' is not one of the supported model attribute types.")


def assert_kamino_data_allclose(
    testcase: unittest.TestCase,
    data_lhs: DataKamino,
    data_rhs: DataKamino,
    *,
    body_attrs: tuple[str, ...],
    joint_attrs: tuple[str, ...],
    rtol: float = TOL_STRICT_RTOL,
    atol: float = TOL_STRICT_ATOL,
):
    """Assert numerical equivalence of two :class:`DataKamino` instances.

    Args:
        testcase: The active :class:`unittest.TestCase`.
        data_lhs / data_rhs: Containers produced by, respectively, the
            metrics-side and solver-side flows.
        body_attrs: Names of fields under ``data.bodies`` to compare.
        joint_attrs: Names of fields under ``data.joints`` to compare.
        rtol / atol: Tolerances forwarded to :func:`numpy.testing.assert_allclose`.
    """
    testcase.assertIsNotNone(data_lhs)
    testcase.assertIsNotNone(data_rhs)
    testcase.assertIsInstance(data_lhs, DataKamino)
    testcase.assertIsInstance(data_rhs, DataKamino)
    testcase.assertEqual(data_lhs.device, data_rhs.device)

    for container_name, attrs in (("bodies", body_attrs), ("joints", joint_attrs)):
        container_lhs = getattr(data_lhs, container_name)
        container_rhs = getattr(data_rhs, container_name)
        for attr_name in attrs:
            attr_lhs = getattr(container_lhs, attr_name)
            attr_rhs = getattr(container_rhs, attr_name)
            if attr_lhs is None and attr_rhs is None:
                continue
            testcase.assertIsNotNone(
                attr_lhs, msg=f"DataKamino.{container_name}.{attr_name} is None on the metrics side."
            )
            testcase.assertIsNotNone(
                attr_rhs, msg=f"DataKamino.{container_name}.{attr_name} is None on the solver side."
            )
            np.testing.assert_allclose(
                attr_lhs.numpy(),
                attr_rhs.numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"DataKamino.{container_name}.{attr_name} disagrees between metrics and solver.",
            )


def assert_jacobians_allclose(
    testcase: unittest.TestCase,
    jac_lhs: DenseSystemJacobians | SparseSystemJacobians,
    jac_rhs: DenseSystemJacobians | SparseSystemJacobians,
    model_lhs: ModelKamino,
    model_rhs: ModelKamino,
    limits_lhs: LimitsKamino,
    limits_rhs: LimitsKamino,
    contacts_lhs: ContactsKamino,
    contacts_rhs: ContactsKamino,
    *,
    rtol: float = TOL_STRICT_RTOL,
    atol: float = TOL_STRICT_ATOL,
):
    """Compare ``J_cts`` (active rows) and ``J_dofs`` between two Jacobian containers.

    Both sides are expected to be :class:`DenseSystemJacobians` built from the
    same pre-event configuration. The method extracts each per-world block via
    :func:`extract_cts_jacobians` / :func:`extract_dofs_jacobians` and
    compares them with :func:`numpy.testing.assert_allclose`.
    """
    testcase.assertIsInstance(jac_lhs, DenseSystemJacobians)
    testcase.assertIsInstance(jac_rhs, DenseSystemJacobians)

    j_cts_lhs = extract_cts_jacobians(model_lhs, limits_lhs, contacts_lhs, jac_lhs, only_active_cts=True)
    j_cts_rhs = extract_cts_jacobians(model_rhs, limits_rhs, contacts_rhs, jac_rhs, only_active_cts=True)
    j_dofs_lhs = extract_dofs_jacobians(model_lhs, jac_lhs)
    j_dofs_rhs = extract_dofs_jacobians(model_rhs, jac_rhs)

    testcase.assertEqual(len(j_cts_lhs), len(j_cts_rhs))
    testcase.assertEqual(len(j_dofs_lhs), len(j_dofs_rhs))

    for w in range(len(j_cts_lhs)):
        np.testing.assert_allclose(
            j_cts_lhs[w],
            j_cts_rhs[w],
            rtol=rtol,
            atol=atol,
            err_msg=f"J_cts disagrees between metrics and solver in world {w}.",
        )
    for w in range(len(j_dofs_lhs)):
        np.testing.assert_allclose(
            j_dofs_lhs[w],
            j_dofs_rhs[w],
            rtol=rtol,
            atol=atol,
            err_msg=f"J_dofs disagrees between metrics and solver in world {w}.",
        )


def assert_dual_problem_allclose(
    testcase: unittest.TestCase,
    prob_lhs: DualProblem,
    prob_rhs: DualProblem,
    *,
    rtol: float = TOL_STRICT_RTOL,
    atol: float = TOL_STRICT_ATOL,
):
    """Compare the active blocks of ``v_f``, ``P`` and ``mu``.

    The Delassus matrix ``D`` is intentionally skipped: PADMM mutates the
    solver-side ``D`` in-place during the proximal regularization step (it
    adds the per-constraint sigma to the diagonal as part of
    :func:`_update_delassus_proximal_regularization`), so the post-step
    ``solver._problem_fd.delassus.D`` no longer equals the freshly-built
    ``D`` produced by :class:`SolutionMetricsNewton`. The build itself is
    deterministic in ``J_cts`` and ``M^{-1}``, both of which are compared
    independently elsewhere (Jacobians via :func:`assert_jacobians_allclose`
    and the mass matrix via the shared ``model``/``data`` references).

    Uses :func:`extract_problem_vector` to slice out the per-world active
    rows of ``v_f`` / ``P``, and compares ``mu`` directly (it is laid out
    per-contact, not per-constraint).
    """
    testcase.assertIsNotNone(prob_lhs)
    testcase.assertIsNotNone(prob_rhs)

    for vec_name in ("v_f", "P"):
        vec_lhs = getattr(prob_lhs.data, vec_name).numpy()
        vec_rhs = getattr(prob_rhs.data, vec_name).numpy()
        blocks_lhs = extract_problem_vector(prob_lhs.delassus, vec_lhs, only_active_dims=True)
        blocks_rhs = extract_problem_vector(prob_rhs.delassus, vec_rhs, only_active_dims=True)
        for w in range(len(blocks_lhs)):
            np.testing.assert_allclose(
                blocks_lhs[w],
                blocks_rhs[w],
                rtol=rtol,
                atol=atol,
                err_msg=f"DualProblem.{vec_name} disagrees between metrics and solver in world {w}.",
            )

    np.testing.assert_allclose(
        prob_lhs.data.mu.numpy(),
        prob_rhs.data.mu.numpy(),
        rtol=rtol,
        atol=atol,
        err_msg="DualProblem.mu disagrees between metrics and solver.",
    )


def _compute_active_constraint_indices(
    model: ModelKamino,
    data: DataKamino,
    *,
    leaf_only_joint_cts: bool,
) -> np.ndarray:
    """Return the global ``lambdas`` indices of the currently active constraints.

    The returned mask spans:
      * The active joint kinematic constraints. When ``leaf_only_joint_cts``
        is ``True`` only leaf-joint kinematic constraints are included (see
        :func:`get_leaf_joint_lambda_indices`); otherwise all kinematic
        constraints are included. Joint dynamic constraints are skipped
        (the builders used here have none).
      * The active limit constraints in each world.
      * The active contact constraints in each world.

    Args:
        model: The Kamino model providing per-world block offsets.
        data: The Kamino data providing per-world active counts.
        leaf_only_joint_cts: If ``True`` restrict the joint-cts portion of
            the mask to leaf-joint kinematic constraints (see module docstring
            for the rationale).
    """
    total_cts_offset = model.info.total_cts_offset.numpy()
    limit_cts_group_offset = data.info.limit_cts_group_offset.numpy()
    contact_cts_group_offset = data.info.contact_cts_group_offset.numpy()
    num_limit_cts = data.info.num_limit_cts.numpy()
    num_contact_cts = data.info.num_contact_cts.numpy()
    num_worlds = model.size.num_worlds

    # Per-joint kinematic-cts global offsets (used for the leaf-only path)
    joint_kin_offset_total = model.joints.kinematic_cts_offset_total_cts.numpy()
    joint_num_kin_cts = model.joints.num_kinematic_cts.numpy()
    leaf_local_indices = (
        get_leaf_joint_lambda_indices(model)
        if leaf_only_joint_cts
        else np.empty((0,), dtype=np.int64)  # unused on the full-cts path
    )
    leaf_local_set = {int(i) for i in leaf_local_indices.tolist()}

    # joints.kinematic_cts_offset_joint_cts is the offset of each joint's
    # kinematic cts within the *joint_cts* (per-world) layout; convert it to a
    # set of global indices using the per-joint kinematic_cts_offset_total_cts.
    joint_kin_offset_joint_cts = model.joints.kinematic_cts_offset_joint_cts.numpy()

    leaf_global_indices: set[int] = set()
    if leaf_only_joint_cts:
        for jid in range(joint_kin_offset_total.shape[0]):
            local_start = int(joint_kin_offset_joint_cts[jid])
            n_kin = int(joint_num_kin_cts[jid])
            for k in range(n_kin):
                local_idx = local_start + k
                if local_idx in leaf_local_set:
                    leaf_global_indices.add(int(joint_kin_offset_total[jid]) + k)

    active: list[int] = []
    for w in range(num_worlds):
        block_start = int(total_cts_offset[w])
        njc_world = int(limit_cts_group_offset[w])  # joint cts occupy [0, njc) within the world block
        ccgo = int(contact_cts_group_offset[w])
        nlc = int(num_limit_cts[w])
        ncc = int(num_contact_cts[w])

        if leaf_only_joint_cts:
            for k in range(njc_world):
                if (block_start + k) in leaf_global_indices:
                    active.append(block_start + k)
        else:
            active.extend(range(block_start, block_start + njc_world))

        # Active limit constraints
        active.extend(range(block_start + njc_world, block_start + njc_world + nlc))
        # Active contact constraints
        active.extend(range(block_start + ccgo, block_start + ccgo + ncc))

    return np.asarray(active, dtype=np.int64)


def assert_lambdas_active_allclose(
    testcase: unittest.TestCase,
    arr_lhs: wp.array,
    arr_rhs: wp.array,
    active_idx: np.ndarray,
    *,
    rtol: float = TOL_SOLVED_RTOL,
    atol: float = TOL_SOLVED_ATOL,
    err_msg: str = "Active-index slices disagree between metrics and solver.",
):
    """Assert that two flat constraint-space arrays match at active indices.

    Both packed lambdas and ``v_plus`` use the same per-world layout, so this
    helper is shared by the lambda comparison and the active-``v_plus``
    comparison.
    """
    testcase.assertIsNotNone(arr_lhs)
    testcase.assertIsNotNone(arr_rhs)
    if active_idx.size == 0:
        return
    np.testing.assert_allclose(
        arr_lhs.numpy()[active_idx],
        arr_rhs.numpy()[active_idx],
        rtol=rtol,
        atol=atol,
        err_msg=err_msg,
    )


def assert_metrics_data_allclose(
    testcase: unittest.TestCase,
    data_lhs: SolutionMetricsData,
    data_rhs: SolutionMetricsData,
    *,
    fields: tuple[str, ...],
    rtol: float = TOL_SOLVED_RTOL,
    atol: float = TOL_SOLVED_ATOL,
):
    """Compare numeric fields of two :class:`SolutionMetricsData`.

    The companion ``*_argmax`` fields are intentionally **not** compared. The
    metrics kernels populate them via an ``atomic_max`` / ``atomic_exch``
    pattern that is racy when several entries share the (near-)maximal
    residual value: any of the tied indices is a valid winner, and the
    metrics-side and solver-side launches may pick different ones in
    different runs even when the residual value itself agrees to numerical
    precision. Comparing argmax keys exactly therefore produces flaky
    failures while adding no signal beyond what the residual values already
    provide.

    Args:
        testcase: The active :class:`unittest.TestCase`.
        data_lhs / data_rhs: The metrics-side and solver-side metric data.
        fields: Names of numeric fields to compare.
        rtol / atol: Tolerances applied to the numeric residuals.
    """
    for field in fields:
        arr_lhs = getattr(data_lhs, field)
        arr_rhs = getattr(data_rhs, field)
        msg.info("[LHS] SolutionMetricsData.%s: %s", field, arr_lhs.numpy())
        msg.info("[RHS] SolutionMetricsData.%s: %s\n", field, arr_rhs.numpy())
        testcase.assertIsNotNone(arr_lhs, msg=f"SolutionMetricsData.{field} is None on the metrics side.")
        testcase.assertIsNotNone(arr_rhs, msg=f"SolutionMetricsData.{field} is None on the solver side.")
        np.testing.assert_allclose(
            arr_lhs.numpy(),
            arr_rhs.numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"SolutionMetricsData.{field} disagrees between metrics and solver.",
        )


###
# Field-set definitions for `assert_kamino_data_allclose`
###

# DataKamino fields populated identically along both wrench-input paths and
# in both the metrics and solver flows. Body wrenches are excluded because
# they are computed from the (potentially path-dependent) extracted lambdas.
DATA_BODY_ATTRS_PATH_INVARIANT = ("q_i", "u_i", "I_i", "inv_I_i", "w_e_i", "w_a_i")

# DataKamino body fields that depend on the extracted constraint reactions.
# Exact match holds only when the lambdas themselves match exactly, i.e. on
# the joint-parent-f path or on the body-parent-f path for builders whose
# non-FREE joints are all leaves.
DATA_BODY_ATTRS_LAMBDA_DEPENDENT = ("w_j_i", "w_l_i", "w_c_i", "w_i")

# DataKamino joint fields that are aliased / written from the input state and
# control containers. ``q_j_p``, ``p_j``, ``r_j``, ``dr_j`` are skipped because
# the metrics flow and the solver flow capture them at different events.
DATA_JOINT_ATTRS_PATH_INVARIANT = ("q_j", "dq_j", "tau_j")

# DataKamino joint fields that depend on the extracted constraint reactions.
# ``lambda_j`` carries the same per-joint information in the joint-frame basis
# used by the convert kernels and is populated by both flows, so it stands in
# for the joint wrenches in the cross-check.
DATA_JOINT_ATTRS_LAMBDA_DEPENDENT = ("lambda_j",)

# Lambda-independent metric residuals. These depend only on the post-event
# state, the kinematic constraint residuals stored on `data`, and the limit /
# contact gap functions, all of which are populated identically by both
# flows.
METRIC_FIELDS_LAMBDA_INDEPENDENT = (
    "r_kinematics",
    "r_cts_joints",
    "r_cts_limits",
    "r_cts_contacts",
)

# Metric residuals that consume the extracted lambdas (or quantities derived
# from them, such as the body wrenches in r_eom).
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
# Test-side metrics evaluation workaround
###


def _run_test_side_metrics_evaluation(metrics: SolutionMetricsNewton, solver: newton.solvers.SolverKamino):
    """Populate ``metrics._metrics.data`` after :meth:`SolutionMetricsNewton.evaluate`.

    The current :class:`SolutionMetricsNewton` implementation has the call to
    its inner :meth:`SolutionMetrics.evaluate` commented out and allocates
    ``_sigma`` with a layout that is incompatible with the dense metrics
    kernel. To produce the same :class:`SolutionMetricsData` the solver
    computes internally (so the cross-check has something to compare against)
    we manually:

    #. Finalise ``metrics._metrics`` against the metrics-side ``model``/``data``.
    #. Run :meth:`SolutionMetrics.evaluate` on the metrics-side intermediate
       buffers using a zero ``sigma`` (the metrics-side Delassus matrix is
       freshly built and is *not* augmented by PADMM's proximal
       regularisation, so the kernel must not subtract a non-zero
       ``sigma * inv(P) * lambdas`` term when re-evaluating
       ``v_plus = v_f + D @ lambda``).

    This workaround keeps the production wrapper untouched while still letting
    the test cross-check the wrapped metrics output. Once
    :class:`SolutionMetricsNewton` is updated to call its inner
    :meth:`SolutionMetrics.evaluate` directly the workaround can be deleted.

    Note that the comparable solver-side metric is computed with the
    *regularised* ``D`` and the matching non-zero ``sigma`` so that the
    proximal contribution cancels out: both sides therefore evaluate
    ``v_plus = v_f + D_unreg @ lambda`` and the cross-check is well-defined.

    Args:
        metrics: The :class:`SolutionMetricsNewton` instance under test.
        solver: The reference :class:`SolverKamino`. Currently unused, but
            retained for API symmetry with future versions of the workaround
            that will need to pass solver-side state through the call.
    """
    del solver  # see docstring; reserved for future use.

    metrics._metrics.finalize(metrics._model, metrics._data)
    metrics._metrics.reset()
    metrics._metrics.evaluate(
        lambdas=metrics._lambdas,
        v_plus=metrics._v_plus,
        state=metrics._state,
        state_p=metrics._state_p,
        jacobians=metrics._jacobians,
        problem=metrics._problem,
        limits=metrics._limits,
        contacts=metrics._contacts,
    )


###
# Driver: per-builder cross-check
###


def _compare_metrics_against_solver(
    testcase: unittest.TestCase,
    setup: TestSetup,
    metrics: SolutionMetricsNewton,
    *,
    body_parent_f_path: bool,
):
    """Run a single builder's metrics evaluation and cross-check against the solver.

    The driver:

    #. Steps the solver and (if requested) synthesises ``joint_parent_f``.
    #. Sanity-checks that the relevant Newton extended state attribute carries
       a non-trivial wrench (so the test is actually exercising the conversion).
    #. Calls :meth:`SolutionMetricsNewton.evaluate`.
    #. Manually finalises and evaluates ``metrics._metrics`` so the metrics
       data is populated (see :func:`_run_test_side_metrics_evaluation`).
    #. Performs the comparison suite (a)-(e) at tolerances appropriate for the
       wrench-input path (see module docstring).

    Args:
        testcase: The active :class:`unittest.TestCase`.
        setup: The per-builder Newton-side scaffolding.
        metrics: The :class:`SolutionMetricsNewton` instance under test.
        body_parent_f_path: ``True`` when the test is exercising the
            ``state.body_parent_f`` branch (which constrains the comparison to
            leaf-joint indices and lambda-independent metric fields).
    """
    setup.step_and_synthesize_joint_parent_f()

    if body_parent_f_path:
        testcase.assertIsNotNone(setup.state.body_parent_f)
    else:
        testcase.assertIsNotNone(setup.state.joint_parent_f)

    # SolverKamino.step() does not propagate solver-computed contact reactions
    # back to the Newton ``Contacts.force`` array. Call ``update_contacts`` so
    # the metrics-side ``contacts.force`` matches the values used internally
    # by the solver to populate ``DataKamino.bodies.w_c_i``.
    setup.solver.update_contacts(setup.contacts, setup.state)

    metrics.evaluate(
        state=setup.state,
        state_p=setup.state_p,
        control=setup.control,
        contacts=setup.contacts,
    )
    _run_test_side_metrics_evaluation(metrics, setup.solver)

    solver_impl = setup.solver._solver_kamino
    solver_fd_solution = solver_impl._solver_fd.data.solution
    solver_contacts = setup.solver._contacts_kamino

    # (a) DataKamino: bodies & joints
    body_attrs: tuple[str, ...] = DATA_BODY_ATTRS_PATH_INVARIANT
    joint_attrs: tuple[str, ...] = DATA_JOINT_ATTRS_PATH_INVARIANT
    if not body_parent_f_path:
        body_attrs = body_attrs + DATA_BODY_ATTRS_LAMBDA_DEPENDENT
        joint_attrs = joint_attrs + DATA_JOINT_ATTRS_LAMBDA_DEPENDENT
    assert_kamino_data_allclose(
        testcase,
        metrics._data,
        solver_impl._data,
        body_attrs=body_attrs,
        joint_attrs=joint_attrs,
    )

    # (b) Jacobians
    assert_jacobians_allclose(
        testcase,
        metrics._jacobians,
        solver_impl._jacobians,
        metrics._model,
        solver_impl._model,
        metrics._limits,
        solver_impl._limits,
        metrics._contacts,
        solver_contacts,
    )

    # (c) Dual problem
    assert_dual_problem_allclose(testcase, metrics._problem, solver_impl._problem_fd)

    # (d) Active v_plus and lambdas
    active_idx = _compute_active_constraint_indices(
        metrics._model,
        metrics._data,
        leaf_only_joint_cts=body_parent_f_path,
    )
    assert_lambdas_active_allclose(
        testcase,
        metrics._v_plus,
        solver_fd_solution.v_plus,
        active_idx,
        rtol=TOL_STRICT_RTOL,
        atol=TOL_STRICT_ATOL,
        err_msg="Active v_plus disagrees between metrics and solver.",
    )
    assert_lambdas_active_allclose(
        testcase,
        metrics._lambdas,
        solver_fd_solution.lambdas,
        active_idx,
        err_msg="Active lambdas disagree between metrics and solver.",
    )

    # (e) Solution metrics data
    metric_fields: tuple[str, ...] = METRIC_FIELDS_LAMBDA_INDEPENDENT
    if not body_parent_f_path:
        metric_fields = metric_fields + METRIC_FIELDS_LAMBDA_DEPENDENT
    assert_metrics_data_allclose(
        testcase,
        metrics._metrics.data,
        solver_impl._metrics.data,
        fields=metric_fields,
    )


###
# Tests
###


class TestSolverMetricsNewton(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose
        self.seed = 42

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
        """Test creating a SolutionMetrics instance with default initialization."""
        metrics = SolutionMetricsNewton()
        self.assertIsNotNone(metrics)
        self.assertIsNone(metrics._model)
        self.assertIsNone(metrics._data)
        self.assertIsNone(metrics._limits)
        self.assertIsNone(metrics._contacts)
        self.assertIsNone(metrics._problem)
        self.assertIsNone(metrics._jacobians)
        self.assertIsNone(metrics._state)
        self.assertIsNone(metrics._state_p)
        self.assertIsNone(metrics._control)
        self.assertIsNone(metrics._v_plus)
        self.assertIsNone(metrics._lambdas)
        self.assertIsNone(metrics._metrics)

    def test_01_finalize_default(self):
        """Test that ``finalize`` allocates every internal buffer and clones the model."""
        setup = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        metrics = SolutionMetricsNewton(
            dt=setup.dt,
            model=setup.builder.finalize(skip_validation_joints=True),
            sparse=False,
        )

        # All internal allocations must be present after finalize.
        self.assertIsNotNone(metrics._model)
        self.assertIsNotNone(metrics._data)
        self.assertIsNotNone(metrics._limits)
        self.assertIsNotNone(metrics._contacts)
        self.assertIsNotNone(metrics._control)
        self.assertIsNotNone(metrics._jacobians)
        self.assertIsInstance(metrics._jacobians, DenseSystemJacobians)
        self.assertIsNotNone(metrics._problem)
        self.assertIsInstance(metrics._problem, DualProblem)
        self.assertIsNotNone(metrics._v_plus)
        self.assertIsNotNone(metrics._lambdas)
        self.assertIsNotNone(metrics._metrics)
        self.assertIsInstance(metrics._metrics, SolutionMetrics)

        # The Kamino model wrapped by the metrics is structurally equal to the
        # solver-side model but lives in a distinct allocation.
        assert_models_equal_but_not_same_malloc(self, metrics._model._model, setup.model)

    def test_02_evaluate_body_parent_f_path(self):
        """Cross-check :meth:`SolutionMetricsNewton.evaluate` along the body-parent-f path.

        Walks every builder in the body-parent-f coverage matrix (5 entries) and runs
        the comparison suite described in :func:`_compare_metrics_against_solver`.
        Joint-cts comparisons are restricted to leaf indices and lambda-dependent
        metric fields are skipped (see module docstring).
        """
        for name, builder_fn, builder_kwargs, max_world_contacts in _builders_without_loops_for_body_parent_f():
            with self.subTest(builder=name):
                setup = TestSetup(
                    builder_fn=builder_fn,
                    builder_kwargs=builder_kwargs,
                    max_world_contacts=max_world_contacts,
                    device=self.default_device,
                    request_state_attributes=("body_parent_f",),
                )

                # Reuse ``setup.model`` so :class:`SolutionMetricsNewton` infers the
                # same ``rigid_contact_max`` (and therefore ``model_max_contacts_host``)
                # as the solver, which sees the value mutated to ``1000`` by the
                # collision pipeline during ``setup.model.contacts()``. Allocating a
                # fresh model via ``setup.builder.finalize(...)`` would observe the
                # un-mutated builder default and produce mismatched contact buffers.
                metrics = SolutionMetricsNewton(
                    dt=setup.dt,
                    model=setup.model,
                    sparse=False,
                )

                _compare_metrics_against_solver(
                    self,
                    setup,
                    metrics,
                    body_parent_f_path=True,
                )

    def test_03_evaluate_joint_parent_f_path(self):
        """Cross-check :meth:`SolutionMetricsNewton.evaluate` along the joint-parent-f path.

        Walks every builder in the joint-parent-f coverage matrix (6 entries,
        including the ``boxes_fourbar`` closed-loop case) and runs the full
        comparison suite (no leaf-only restriction; lambda-dependent metric
        fields included).
        """
        for name, builder_fn, builder_kwargs, max_world_contacts in _builders_with_loops_for_joint_parent_f():
            with self.subTest(builder=name):
                setup = TestSetup(
                    builder_fn=builder_fn,
                    builder_kwargs=builder_kwargs,
                    max_world_contacts=max_world_contacts,
                    device=self.default_device,
                    request_state_attributes=("body_parent_f", "joint_parent_f"),
                )

                # See the comment in :meth:`test_02_evaluate_body_parent_f_path`:
                # we reuse ``setup.model`` to keep ``rigid_contact_max`` consistent
                # between the metrics-side and solver-side contact allocations.
                metrics = SolutionMetricsNewton(
                    dt=setup.dt,
                    model=setup.model,
                    sparse=False,
                )

                _compare_metrics_against_solver(
                    self,
                    setup,
                    metrics,
                    body_parent_f_path=False,
                )


###
# SolutionMetricsLogger helpers
###


def _make_finalized_metrics(device: wp.DeviceLike) -> tuple[TestSetup, SolutionMetricsNewton]:
    """Build a small finalized :class:`SolutionMetricsNewton` for logger tests.

    The wrapper's inner :class:`SolutionMetrics` is finalised against the
    metrics-side ``model`` / ``data`` so that ``metrics.data`` is a fully
    allocated :class:`SolutionMetricsData` instance whose per-world arrays
    can be filled in deterministically by the tests.
    """
    setup = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=device)
    metrics = SolutionMetricsNewton(dt=setup.dt, model=setup.model, sparse=False)
    # Finalise the inner SolutionMetrics so ``metrics.data`` exposes valid arrays.
    metrics._metrics.finalize(metrics._model, metrics._data)
    return setup, metrics


def _seed_metrics_data(metrics: SolutionMetricsNewton, value: float, argmax_value: int = 0):
    """Populate :attr:`SolutionMetricsNewton.data` with deterministic test values."""
    data = metrics.data
    data.r_eom.fill_(value)
    data.r_kinematics.fill_(value + 1.0)
    data.r_cts_joints.fill_(value + 2.0)
    data.r_cts_limits.fill_(value + 3.0)
    data.r_cts_contacts.fill_(value + 4.0)
    data.r_v_plus.fill_(value + 5.0)
    data.r_ncp_primal.fill_(value + 6.0)
    data.r_ncp_dual.fill_(value + 7.0)
    data.r_ncp_compl.fill_(value + 8.0)
    data.r_vi_natmap.fill_(value + 9.0)
    data.f_ncp.fill_(value + 10.0)
    data.f_ccp.fill_(value + 11.0)
    data.r_eom_argmax.fill_(argmax_value)
    data.r_kinematics_argmax.fill_(argmax_value + 1)
    data.r_cts_joints_argmax.fill_(argmax_value + 2)
    data.r_cts_limits_argmax.fill_(argmax_value + 3)
    data.r_cts_contacts_argmax.fill_(argmax_value + 4)
    data.r_v_plus_argmax.fill_(argmax_value + 5)
    data.r_ncp_primal_argmax.fill_(argmax_value + 6)
    data.r_ncp_dual_argmax.fill_(argmax_value + 7)
    data.r_ncp_compl_argmax.fill_(argmax_value + 8)
    data.r_vi_natmap_argmax.fill_(argmax_value + 9)


_SCALAR_METRIC_FIELDS_FOR_TEST: tuple[str, ...] = (
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


_ARGMAX_METRIC_FIELDS_FOR_TEST: tuple[str, ...] = (
    "r_eom_argmax",
    "r_kinematics_argmax",
    "r_cts_joints_argmax",
    "r_cts_limits_argmax",
    "r_cts_contacts_argmax",
    "r_v_plus_argmax",
    "r_ncp_primal_argmax",
    "r_ncp_dual_argmax",
    "r_ncp_compl_argmax",
    "r_vi_natmap_argmax",
)


def _matplotlib_available() -> bool:
    """Return ``True`` if matplotlib can be imported in this environment."""
    try:
        import matplotlib
        import matplotlib.pyplot  # noqa: F401

        return True
    except ImportError:
        return False


###
# SolutionMetricsLogger tests
###


class TestSolutionMetricsLogger(unittest.TestCase):
    """Unit tests for :class:`SolutionMetricsLogger`.

    These tests exercise the logger's allocation/sizing semantics, the
    bounded vs rolling overflow modes, the decimation gate, and the
    matplotlib export. They populate the wrapped metrics container with
    deterministic per-world values via :func:`_seed_metrics_data` so the
    assertions don't depend on the iterative solver's convergence.

    The plot test mirrors the :mod:`test_solvers_padmm` convention: when
    ``test_context.verbose`` is set, the generated figures are also
    persisted under ``test_context.output_path / "test_solution_metrics_logger"``.
    Set :attr:`show` to ``True`` to additionally display the plots
    interactively (blocks until the windows are closed).
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

        # Toggle these to opt in to detailed test output, persistent plot
        # artifacts, and interactive plot display. Defaults follow the
        # ``test_solvers_padmm.py`` pattern: ``verbose`` and ``savefig``
        # piggyback on ``test_context.verbose`` and ``show`` is off by
        # default so the test runner is never blocked by plot windows.
        self.verbose = test_context.verbose
        self.savefig = test_context.verbose
        self.show = False
        self.output_path = test_context.output_path / "test_solution_metrics_logger"

        # Create the per-test output directory only when we're actually
        # going to save anything to it.
        if self.savefig:
            self.output_path.mkdir(parents=True, exist_ok=True)

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

    def test_logger_make_default(self):
        """``__init__`` allocates every log buffer on the metrics' device."""
        _setup, metrics = _make_finalized_metrics(self.default_device)
        max_frames = 7

        logger = SolutionMetricsLogger(
            metrics=metrics,
            max_frames=max_frames,
            mode=SolutionMetricsLogger.Mode.BOUNDED,
            decimation=1,
        )

        self.assertEqual(logger.max_frames, max_frames)
        self.assertEqual(logger.mode, SolutionMetricsLogger.Mode.BOUNDED)
        self.assertEqual(logger.decimation, 1)
        self.assertEqual(logger.num_logged_frames, 0)
        self.assertEqual(logger.num_total_writes, 0)
        self.assertFalse(logger.is_full)
        self.assertEqual(logger.num_worlds, metrics._model.size.num_worlds)
        self.assertEqual(logger.device, metrics.device)

        expected_shape = (max_frames, metrics._model.size.num_worlds)
        for field in _SCALAR_METRIC_FIELDS_FOR_TEST:
            buf = getattr(logger, f"log_{field}")
            self.assertIsNotNone(buf, msg=f"log_{field} was not allocated")
            self.assertEqual(buf.shape, expected_shape)
            self.assertEqual(buf.dtype, wp.float32)
            self.assertEqual(buf.device, metrics.device)
        for field in _ARGMAX_METRIC_FIELDS_FOR_TEST:
            buf = getattr(logger, f"log_{field}")
            self.assertIsNotNone(buf, msg=f"log_{field} was not allocated")
            self.assertEqual(buf.shape, expected_shape)
            self.assertEqual(buf.device, metrics.device)

    def test_logger_invalid_construction(self):
        """The constructor rejects malformed arguments early."""
        _setup, metrics = _make_finalized_metrics(self.default_device)

        with self.assertRaises(TypeError):
            SolutionMetricsLogger(metrics=object(), max_frames=4)

        with self.assertRaises(ValueError):
            SolutionMetricsLogger(metrics=metrics, max_frames=0)

        with self.assertRaises(ValueError):
            SolutionMetricsLogger(metrics=metrics, max_frames=4, decimation=0)

        with self.assertRaises(ValueError):
            SolutionMetricsLogger(metrics=metrics, max_frames=4, dt=0.0)

        # An un-finalised metrics container is rejected.
        empty = SolutionMetricsNewton()
        with self.assertRaises(RuntimeError):
            SolutionMetricsLogger(metrics=empty, max_frames=4)

    def test_logger_log_records_per_world_data(self):
        """A single :meth:`log` call captures every metric field."""
        _setup, metrics = _make_finalized_metrics(self.default_device)
        logger = SolutionMetricsLogger(metrics=metrics, max_frames=4)

        _seed_metrics_data(metrics, value=1.5, argmax_value=2)
        logger.log()

        self.assertEqual(logger.num_logged_frames, 1)
        self.assertEqual(logger.num_total_writes, 1)

        np_data = logger.to_numpy()
        nw = metrics._model.size.num_worlds

        for offset, field in enumerate(_SCALAR_METRIC_FIELDS_FOR_TEST):
            expected = np.full((1, nw), 1.5 + float(offset), dtype=np.float32)
            np.testing.assert_array_equal(np_data[field], expected)

        for offset, field in enumerate(_ARGMAX_METRIC_FIELDS_FOR_TEST):
            expected = np.full((1, nw), 2 + offset)
            np.testing.assert_array_equal(np_data[field].astype(np.int64), expected.astype(np.int64))

    def test_logger_bounded_early_exit(self):
        """In :attr:`Mode.BOUNDED` extra calls past ``max_frames`` are no-ops."""
        _setup, metrics = _make_finalized_metrics(self.default_device)
        max_frames = 3
        logger = SolutionMetricsLogger(
            metrics=metrics,
            max_frames=max_frames,
            mode=SolutionMetricsLogger.Mode.BOUNDED,
        )

        # Log ``2 * max_frames`` distinct values; only the first ``max_frames`` should land.
        for i in range(2 * max_frames):
            _seed_metrics_data(metrics, value=float(i))
            logger.log()

        self.assertEqual(logger.num_logged_frames, max_frames)
        self.assertEqual(logger.num_total_writes, max_frames)
        self.assertTrue(logger.is_full)

        np_data = logger.to_numpy()
        nw = metrics._model.size.num_worlds
        expected_r_eom = np.array([[0.0] * nw, [1.0] * nw, [2.0] * nw], dtype=np.float32)
        np.testing.assert_array_equal(np_data["r_eom"], expected_r_eom)

    def test_logger_rolling_wrap_around(self):
        """In :attr:`Mode.ROLLING` :meth:`to_numpy` returns the most recent frames in order."""
        _setup, metrics = _make_finalized_metrics(self.default_device)
        max_frames = 4
        logger = SolutionMetricsLogger(
            metrics=metrics,
            max_frames=max_frames,
            mode=SolutionMetricsLogger.Mode.ROLLING,
        )

        # Log ``max_frames + N`` distinct values; the buffer should hold the most
        # recent ``max_frames`` of them, rotated to chronological order.
        n_extra = 3
        total_logs = max_frames + n_extra
        for i in range(total_logs):
            _seed_metrics_data(metrics, value=float(i))
            logger.log()

        self.assertEqual(logger.num_logged_frames, max_frames)
        self.assertEqual(logger.num_total_writes, total_logs)
        self.assertTrue(logger.is_full)

        np_data = logger.to_numpy()
        nw = metrics._model.size.num_worlds
        expected_first_world = np.array(
            [float(i) for i in range(total_logs - max_frames, total_logs)],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(np_data["r_eom"][:, 0], expected_first_world)
        # All worlds should carry the same per-frame value (since fill_ broadcasts).
        for w in range(1, nw):
            np.testing.assert_array_equal(np_data["r_eom"][:, w], expected_first_world)

    def test_logger_decimation_skips_calls(self):
        """``decimation=k`` records only every ``k``-th :meth:`log` call."""
        _setup, metrics = _make_finalized_metrics(self.default_device)
        decimation = 3
        max_frames = 4
        logger = SolutionMetricsLogger(
            metrics=metrics,
            max_frames=max_frames,
            mode=SolutionMetricsLogger.Mode.BOUNDED,
            decimation=decimation,
        )

        # Make ``decimation * max_frames`` calls; exactly ``max_frames`` should land.
        for i in range(decimation * max_frames):
            _seed_metrics_data(metrics, value=float(i))
            logger.log()

        self.assertEqual(logger.num_logged_frames, max_frames)
        self.assertEqual(logger.num_total_writes, max_frames)

        np_data = logger.to_numpy()
        # The recorded values correspond to calls 0, decimation, 2*decimation, ...
        expected_first_world = np.array(
            [float(i * decimation) for i in range(max_frames)],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(np_data["r_eom"][:, 0], expected_first_world)

        # The time axis should account for the decimation factor.
        time_axis = logger.time_axis()
        self.assertEqual(time_axis.shape, (max_frames,))
        scale = (logger.dt if logger.dt is not None else 1.0) * float(decimation)
        np.testing.assert_allclose(time_axis, np.arange(max_frames, dtype=np.float32) * scale)

    def test_logger_reset(self):
        """:meth:`reset` clears counters and zeroes the buffers."""
        _setup, metrics = _make_finalized_metrics(self.default_device)
        logger = SolutionMetricsLogger(metrics=metrics, max_frames=4)

        _seed_metrics_data(metrics, value=5.0)
        logger.log()
        logger.log()
        self.assertEqual(logger.num_logged_frames, 2)

        logger.reset()
        self.assertEqual(logger.num_logged_frames, 0)
        self.assertEqual(logger.num_total_writes, 0)
        self.assertFalse(logger.is_full)

        # Buffers must be zero after reset (scalar fields) and -1 (argmax fields).
        nw = metrics._model.size.num_worlds
        np.testing.assert_array_equal(logger.log_r_eom.numpy(), np.zeros((logger.max_frames, nw), dtype=np.float32))
        np.testing.assert_array_equal(
            logger.log_r_eom_argmax.numpy(), np.full((logger.max_frames, nw), -1, dtype=np.int64)
        )

        # A subsequent log() writes at index 0 again.
        _seed_metrics_data(metrics, value=7.0)
        logger.log()
        self.assertEqual(logger.num_logged_frames, 1)
        np.testing.assert_array_equal(logger.to_numpy()["r_eom"][0], np.full((nw,), 7.0, dtype=np.float32))

    def test_logger_unpack_argmax_key(self):
        """:meth:`unpack_argmax_key` reverses the ``build_pair_key2`` packing."""
        index_a = 0x12345
        index_b = 0xABCDE
        key = (index_a << 32) | index_b
        a, b = SolutionMetricsLogger.unpack_argmax_key(key)
        self.assertEqual(a, index_a)
        self.assertEqual(b, index_b)

    def test_logger_graph_capture(self):
        """`logger.log()` can be captured into a CUDA graph and replayed.

        Verifies the device-side counter / decision pattern: the captured
        graph references on-device counters and decision buffers, so
        replaying it without re-launching from the host must still advance
        ``num_total_writes`` and write deterministic per-world rows into
        the log buffers.
        """
        if not self.default_device.is_cuda:
            self.skipTest("Graph capture requires a CUDA device.")

        _setup, metrics = _make_finalized_metrics(self.default_device)
        logger = SolutionMetricsLogger(metrics=metrics, max_frames=16)

        # Seed the metrics container with deterministic values so every
        # captured-and-replayed row is bit-identical.
        seed_value = 1.5
        seed_argmax = 2
        _seed_metrics_data(metrics, value=seed_value, argmax_value=seed_argmax)

        # Force any pending allocations / kernel JITs onto the device with a
        # warm-up launch outside the captured region.
        logger.log()
        logger.reset()

        replay_count = 7
        with wp.ScopedCapture(device=self.default_device) as capture:
            logger.log()
        graph = capture.graph

        for _ in range(replay_count):
            wp.capture_launch(graph)
        wp.synchronize_device(self.default_device)

        self.assertEqual(logger.num_logged_frames, replay_count)
        self.assertEqual(logger.num_total_writes, replay_count)
        self.assertEqual(logger.num_calls, replay_count)

        np_data = logger.to_numpy()
        nw = metrics._model.size.num_worlds
        for offset, field in enumerate(_SCALAR_METRIC_FIELDS_FOR_TEST):
            expected = np.full((replay_count, nw), seed_value + float(offset), dtype=np.float32)
            np.testing.assert_array_equal(
                np_data[field],
                expected,
                err_msg=f"Captured logger {field} row does not match the seeded value across replays.",
            )
        for offset, field in enumerate(_ARGMAX_METRIC_FIELDS_FOR_TEST):
            expected = np.full((replay_count, nw), seed_argmax + offset)
            np.testing.assert_array_equal(
                np_data[field].astype(np.int64),
                expected.astype(np.int64),
                err_msg=f"Captured logger {field} row does not match the seeded argmax across replays.",
            )

    def test_logger_graph_capture_bounded_overflow(self):
        """Bounded-mode overflow is enforced inside graph capture.

        With ``Mode.BOUNDED`` the on-device decision kernel must early-exit
        once ``max_frames`` writes have landed, even when the captured
        graph is replayed more times than the buffer can hold.
        """
        if not self.default_device.is_cuda:
            self.skipTest("Graph capture requires a CUDA device.")

        _setup, metrics = _make_finalized_metrics(self.default_device)
        max_frames = 4
        replay_count = 10
        logger = SolutionMetricsLogger(
            metrics=metrics,
            max_frames=max_frames,
            mode=SolutionMetricsLogger.Mode.BOUNDED,
        )

        _seed_metrics_data(metrics, value=3.0)

        # Warm up to materialize on-device state, then start fresh.
        logger.log()
        logger.reset()

        with wp.ScopedCapture(device=self.default_device) as capture:
            logger.log()
        graph = capture.graph

        for _ in range(replay_count):
            wp.capture_launch(graph)
        wp.synchronize_device(self.default_device)

        self.assertEqual(logger.num_logged_frames, max_frames)
        self.assertEqual(logger.num_total_writes, max_frames)
        self.assertEqual(logger.num_calls, replay_count)
        self.assertTrue(logger.is_full)

    @unittest.skipUnless(_matplotlib_available(), "matplotlib is required for plot generation")
    def test_logger_plot_writes_per_metric_files(self):
        """:meth:`plot` writes exactly one file per scalar metric.

        The test always validates the file-generation logic via a temporary
        directory (so it leaves no artifacts behind by default). When
        :attr:`savefig` is enabled the plots are additionally persisted as
        PDFs under :attr:`output_path` so they can be inspected after the
        run, and when :attr:`show` is enabled the plots are also displayed
        interactively (note: this blocks the test runner until the plot
        windows are closed).
        """
        _setup, metrics = _make_finalized_metrics(self.default_device)
        logger = SolutionMetricsLogger(metrics=metrics, max_frames=4)

        for i in range(3):
            _seed_metrics_data(metrics, value=float(i))
            logger.log()

        # Always verify the file-generation logic via a temporary directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.plot(path=tmpdir, ext="png")

            for field in _SCALAR_METRIC_FIELDS_FOR_TEST:
                fig_path = os.path.join(tmpdir, f"{field}.png")
                self.assertTrue(os.path.isfile(fig_path), msg=f"expected plot at {fig_path}")
            # No additional files should be produced for argmax fields.
            self.assertEqual(
                len(os.listdir(tmpdir)),
                len(_SCALAR_METRIC_FIELDS_FOR_TEST),
            )

        # Optionally persist the plots under the unit-test output directory
        # and/or display them interactively (see the class docstring for
        # details on the ``savefig`` / ``show`` toggles).
        if self.savefig or self.show:
            msg.notif("Generating solution metrics logger plots...")
            save_path = str(self.output_path) if self.savefig else None
            logger.plot(path=save_path, show=self.show, ext="pdf")
            if self.savefig:
                for field in _SCALAR_METRIC_FIELDS_FOR_TEST:
                    fig_path = self.output_path / f"{field}.pdf"
                    self.assertTrue(fig_path.is_file(), msg=f"expected plot at {fig_path}")


###
# Stepwise per-builder comparison helpers
###


def _resolve_solver_metrics(solver: newton.solvers.SolverKamino) -> SolutionMetrics:
    """Return the :class:`SolverKamino`-internal :class:`SolutionMetrics` reference.

    The reference is populated on every :meth:`SolverKamino.step` call when
    the solver is configured with ``compute_solution_metrics=True`` (which
    :func:`_make_zero_stab_solver_config` enables). Raises :class:`RuntimeError`
    if the solver was constructed without that flag, so the per-step
    comparison cannot proceed against a missing reference.
    """
    impl_metrics = solver._solver_kamino.metrics
    if impl_metrics is None:
        raise RuntimeError(
            "SolverKamino was not configured with `compute_solution_metrics=True`; "
            "the per-step comparison cannot proceed."
        )
    return impl_metrics


def _step_and_log_trajectory(
    setup: TestSetup,
    metrics: SolutionMetricsNewton,
    logger_metrics: SolutionMetricsLogger,
    logger_solver: SolutionMetricsLogger,
    num_steps: int,
):
    """Step the solver ``num_steps`` times and log both metrics each step.

    Each iteration:

    #. Runs Newton-side collision detection at the current ``state_p``.
    #. Steps :class:`SolverKamino`.
    #. If the post-step Newton state advertises a ``joint_parent_f``
       buffer, synthesises an exact per-joint wrench from the solver's
       Jacobians and lambdas via :func:`make_joint_parent_f_from_solver_state`
       and copies it into the state. (The solver itself only writes
       ``body_parent_f`` natively; the synthesised version is exact for
       every non-FREE joint, including non-leaf joints in tree
       articulations and joints in kinematic loops.)
    #. Refreshes ``contacts.force`` so the metrics-side ``contacts.force``
       matches the values the solver consumed internally.
    #. Calls :meth:`SolutionMetricsNewton.evaluate`.
    #. Logs both the wrapper-side and solver-internal metrics.
    #. Swaps ``state``/``state_p`` so the latest state becomes the input
       to the next step.

    Args:
        setup: The per-builder Newton-side scaffolding.
        metrics: The :class:`SolutionMetricsNewton` instance under test.
        logger_metrics: Logger wrapping ``metrics``.
        logger_solver: Logger wrapping the solver-internal :class:`SolutionMetrics`.
        num_steps: Number of simulation steps to execute and log.
    """
    for _ in range(num_steps):
        setup.model.collide(setup.state_p, setup.contacts)
        setup.solver.step(
            state_in=setup.state_p,
            state_out=setup.state,
            control=setup.control,
            contacts=setup.contacts,
            dt=setup.dt,
        )
        if setup.state.joint_parent_f is not None:
            joint_parent_f_synth = make_joint_parent_f_from_solver_state(setup.solver)
            wp.copy(setup.state.joint_parent_f, joint_parent_f_synth)
        setup.solver.update_contacts(setup.contacts, setup.state)
        metrics.evaluate(
            state=setup.state,
            state_p=setup.state_p,
            control=setup.control,
            contacts=setup.contacts,
        )
        logger_metrics.log()
        logger_solver.log()
        setup.state, setup.state_p = setup.state_p, setup.state


def _assert_loggers_match(
    testcase: unittest.TestCase,
    logger_metrics: SolutionMetricsLogger,
    logger_solver: SolutionMetricsLogger,
    *,
    fields: tuple[str, ...],
    rtol: float = TOL_SOLVED_RTOL,
    atol: float = TOL_SOLVED_ATOL,
):
    """Cross-check the two loggers' recorded data field-by-field.

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
    *,
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
    SolutionMetricsLogger._initialize_plt()
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


class TestSolutionMetricsNewtonStepwise(unittest.TestCase):
    """End-to-end stepwise cross-check of :class:`SolutionMetricsNewton`.

    For each builder under test the test:

    #. Builds a :class:`TestSetup` whose state requests both wrench
       attributes (joint-parent-f path) or only ``body_parent_f``
       (body-parent-f path).
    #. Creates a :class:`SolutionMetricsNewton` instance to verify and
       resolves the solver-internal :class:`SolutionMetrics` reference
       via :func:`_resolve_solver_metrics`.
    #. Allocates two :class:`SolutionMetricsLogger` instances - one per
       metrics container.
    #. Steps the solver :attr:`NUM_STEPS` times via
       :func:`_step_and_log_trajectory`, calling
       :meth:`SolutionMetricsNewton.evaluate` after every step and logging
       both metrics containers.
    #. Asserts the recorded :class:`SolutionMetricsData` arrays match
       (full trajectory and per-step), restricting the comparison to
       the lambda-independent fields along the body-parent-f path
       (where lambdas can only be exactly recovered for leaf joints).
    #. Optionally renders an overlay plot per metric when
       :attr:`savefig` / :attr:`show` are enabled - one red line for the
       :class:`SolverKamino` reference and one blue line for the
       :class:`SolutionMetricsNewton` data, on the same axis.

    Each ``test_*`` targets exactly one (path, builder) combination and
    can be invoked individually, e.g.::

        python -m newton._src.solvers.kamino.tests.test_metrics_core \\
            TestSolutionMetricsNewtonStepwise.test_joint_parent_f_box_on_plane
    """

    NUM_STEPS = 200
    """Trajectory length used by every per-(path, builder) test."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

        # Mirror the toggles used by :class:`TestSolutionMetricsLogger`:
        # ``verbose`` and ``savefig`` piggyback on ``test_context.verbose``;
        # ``show`` is off by default so the test runner is never blocked by
        # plot windows.
        self.verbose = True
        self.savefig = True
        self.show = False
        self.output_path = test_context.output_path / "test_solution_metrics_newton_stepwise"

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
        builder_fn,
        builder_kwargs: dict,
        max_world_contacts: int,
        *,
        body_parent_f_path: bool,
        builder_name: str,
    ):
        """Execute the stepwise comparison for one (path, builder) combination.

        Args:
            builder_fn: The builder function to invoke.
            builder_kwargs: Builder keyword arguments.
            max_world_contacts: Per-world rigid-contact capacity.
            body_parent_f_path: ``True`` to drive
                :meth:`SolutionMetricsNewton.evaluate` from
                ``state.body_parent_f`` (lambda-independent fields only);
                ``False`` to use the synthesised ``state.joint_parent_f``
                (full metric-field coverage).
            builder_name: Builder name used as the plot output subdirectory.
        """
        if body_parent_f_path:
            request_state_attributes: tuple[str, ...] = ("body_parent_f",)
        else:
            request_state_attributes = ("body_parent_f", "joint_parent_f")

        setup = TestSetup(
            builder_fn=builder_fn,
            builder_kwargs=builder_kwargs,
            max_world_contacts=max_world_contacts,
            device=self.default_device,
            request_state_attributes=request_state_attributes,
        )

        # Reuse ``setup.model`` so :class:`SolutionMetricsNewton` infers the
        # same ``rigid_contact_max`` (and therefore ``model_max_contacts_host``)
        # as the solver. See the comment in :meth:`test_02_evaluate_body_parent_f_path`
        # for the exhaustive rationale.
        metrics = SolutionMetricsNewton(
            dt=setup.dt,
            model=setup.model,
            sparse=False,
        )

        solver_metrics = _resolve_solver_metrics(setup.solver)

        logger_metrics = SolutionMetricsLogger(
            metrics=metrics,
            max_frames=self.NUM_STEPS,
            mode=SolutionMetricsLogger.Mode.BOUNDED,
        )
        logger_solver = SolutionMetricsLogger(
            metrics=solver_metrics,
            max_frames=self.NUM_STEPS,
            mode=SolutionMetricsLogger.Mode.BOUNDED,
        )

        _step_and_log_trajectory(
            setup=setup,
            metrics=metrics,
            logger_metrics=logger_metrics,
            logger_solver=logger_solver,
            num_steps=self.NUM_STEPS,
        )

        self.assertEqual(logger_metrics.num_logged_frames, self.NUM_STEPS)
        self.assertEqual(logger_solver.num_logged_frames, self.NUM_STEPS)

        fields = METRIC_FIELDS_LAMBDA_INDEPENDENT
        if not body_parent_f_path:
            fields = fields + METRIC_FIELDS_LAMBDA_DEPENDENT

        _assert_loggers_match(
            self,
            logger_metrics,
            logger_solver,
            fields=fields,
        )

        if self.savefig or self.show:
            path_label = "body_parent_f" if body_parent_f_path else "joint_parent_f"
            plot_dir = self.output_path / path_label / builder_name
            save_path: str | None = None
            if self.savefig:
                plot_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(plot_dir)
            msg.notif(f"Generating overlay plots for {path_label} / {builder_name}...")
            plot_logger_comparison(
                logger_metrics=logger_metrics,
                logger_solver=logger_solver,
                path=save_path,
                show=self.show,
                ext="pdf",
            )

    ###
    # body_parent_f path tests (5 builders, lambda-independent fields only)
    ###

    def test_body_parent_f_box_on_plane(self):
        """Body-parent-f path stepwise cross-check on ``box_on_plane``."""
        self._run_one(
            builder_fn=basics.build_box_on_plane,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=8,
            body_parent_f_path=True,
            builder_name="box_on_plane",
        )

    def test_body_parent_f_cartpole(self):
        """Body-parent-f path stepwise cross-check on ``cartpole``."""
        self._run_one(
            builder_fn=basics.build_cartpole,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=8,
            body_parent_f_path=True,
            builder_name="cartpole",
        )

    def test_body_parent_f_boxes_hinged(self):
        """Body-parent-f path stepwise cross-check on ``boxes_hinged``."""
        self._run_one(
            builder_fn=basics.build_boxes_hinged,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            body_parent_f_path=True,
            builder_name="boxes_hinged",
        )

    def test_body_parent_f_boxes_nunchaku(self):
        """Body-parent-f path stepwise cross-check on ``boxes_nunchaku``."""
        self._run_one(
            builder_fn=basics.build_boxes_nunchaku,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            body_parent_f_path=True,
            builder_name="boxes_nunchaku",
        )

    def test_body_parent_f_boxes_nunchaku_vertical(self):
        """Body-parent-f path stepwise cross-check on ``boxes_nunchaku_vertical``."""
        self._run_one(
            builder_fn=basics.build_boxes_nunchaku_vertical,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            body_parent_f_path=True,
            builder_name="boxes_nunchaku_vertical",
        )

    ###
    # joint_parent_f path tests (6 builders, full metric-field comparison)
    ###

    def test_joint_parent_f_box_on_plane(self):
        """Joint-parent-f path stepwise cross-check on ``box_on_plane``."""
        self._run_one(
            builder_fn=basics.build_box_on_plane,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=8,
            body_parent_f_path=False,
            builder_name="box_on_plane",
        )

    def test_joint_parent_f_cartpole(self):
        """Joint-parent-f path stepwise cross-check on ``cartpole``."""
        self._run_one(
            builder_fn=basics.build_cartpole,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=8,
            body_parent_f_path=False,
            builder_name="cartpole",
        )

    def test_joint_parent_f_boxes_hinged(self):
        """Joint-parent-f path stepwise cross-check on ``boxes_hinged``."""
        self._run_one(
            builder_fn=basics.build_boxes_hinged,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            body_parent_f_path=False,
            builder_name="boxes_hinged",
        )

    def test_joint_parent_f_boxes_nunchaku(self):
        """Joint-parent-f path stepwise cross-check on ``boxes_nunchaku``."""
        self._run_one(
            builder_fn=basics.build_boxes_nunchaku,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            body_parent_f_path=False,
            builder_name="boxes_nunchaku",
        )

    def test_joint_parent_f_boxes_nunchaku_vertical(self):
        """Joint-parent-f path stepwise cross-check on ``boxes_nunchaku_vertical``."""
        self._run_one(
            builder_fn=basics.build_boxes_nunchaku_vertical,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            body_parent_f_path=False,
            builder_name="boxes_nunchaku_vertical",
        )

    def test_joint_parent_f_boxes_fourbar(self):
        """Joint-parent-f path stepwise cross-check on ``boxes_fourbar`` (closed-loop)."""
        self._run_one(
            builder_fn=basics.build_boxes_fourbar,
            builder_kwargs={"z_offset": -1e-5, "floatingbase": True},
            max_world_contacts=32,
            body_parent_f_path=False,
            builder_name="boxes_fourbar",
        )


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
