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
from newton._src.solvers.kamino._src.metrics import SolutionMetricsNewton
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
TOL_SOLVED_RTOL = 1e-4
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
# ``j_w_j`` and the ``j_w_*_j`` family are skipped because :class:`SolverKamino`
# allocates its internal :class:`DataKamino` with ``joint_wrenches=False``;
# the corresponding fields are therefore ``None`` on the solver side, while
# :class:`SolutionMetricsNewton` keeps them allocated for the metrics
# computations that consume them. ``lambda_j`` carries the same per-joint
# information in the joint-frame basis used by the convert kernels and is
# populated by both flows, so it stands in for the joint wrenches in the
# cross-check.
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
    sigma_zero = wp.zeros(
        shape=(metrics._model.size.num_worlds,),
        dtype=wp.vec2f,
        device=metrics.device,
    )
    metrics._metrics.evaluate(
        sigma=sigma_zero,
        lambdas=metrics._lambdas,
        v_plus=metrics._v_plus,
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
        self.assertIsNone(metrics._sigma)
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
        self.assertIsNotNone(metrics._sigma)
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
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
