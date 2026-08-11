# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared benchmark scaffolding: per-solver ``SolverSetup`` and multi-solver ``SetupRunner``.

Each :class:`SolverSetup` may attach a :class:`PhysicsMetrics` +
:class:`PhysicsMetricsLogger` pair (cross-solver constraint residuals computed
from the Newton state); this is the metric stack the paper CSV/table pipeline
consumes. It may also attach an ``aux_logger`` (typically
:class:`SolverKaminoLogger`) for solver-specific PADMM diagnostics that live
outside the paper CSV output.
"""

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

from ....sim import Contacts, Control, Model, ModelBuilder, State
from ....solvers import SolverBase
from ....viewer import ViewerBase
from .._src.utils import logger as msg
from ..examples import print_progress_bar
from .logging import PhysicsMetricsLogger
from .metrics import (
    PhysicsMetrics,
    compute_contact_constraint_metrics,
    compute_joint_constraint_metrics,
    compute_per_world_contact_constraint_summary,
    compute_per_world_joint_constraint_summary,
)

# Runner mode names — keep in sync with the CLI ``--mode`` choices exposed by
# the per-example scripts and by ``example_benchmark_paper_all``.
MODE_TIED: str = "tied"
MODE_INDEPENDENT: str = "independent"
MODE_TIED_REFERENCE: str = "tied_reference"
_ALLOWED_MODES: frozenset[str] = frozenset({MODE_TIED, MODE_INDEPENDENT, MODE_TIED_REFERENCE})

###
# Module interface
###

__all__ = [
    "MODE_INDEPENDENT",
    "MODE_TIED",
    "MODE_TIED_REFERENCE",
    "SetupRunner",
    "SolverSetup",
]


###
# Snapshot helpers
###


def _copy_array(dst: wp.array, src: wp.array, name: str) -> None:
    """Copy ``src`` into ``dst`` after asserting ``src.size == dst.size``.

    ``wp.copy`` only raises when the source overruns the destination. When the
    source is smaller, it silently copies a prefix and leaves the destination's
    tail stale. The explicit size check converts that into a named error.
    Skips the copy when ``src is dst`` — the runner's independent mode reuses
    the setup's own state buffers as ``step()`` inputs.
    """
    if dst is src:
        return
    if dst.size != src.size:
        raise ValueError(f"{name!r}: size mismatch (src={src.size}, dst={dst.size})")
    wp.copy(dst, src)


def _copy_optional(dst: object, src: object, attr: str) -> None:
    """Copy an optional ``wp.array`` field, requiring symmetric allocation.

    Raises ``ValueError`` if exactly one side has the array allocated — a silent
    skip would mean one solver's snapshot is missing data the other solver wrote.
    """
    val_dst = getattr(dst, attr, None)
    val_src = getattr(src, attr, None)
    if val_dst is None and val_src is None:
        return
    if val_dst is None:
        raise ValueError(f"destination is missing array for {attr!r} which is present in source")
    if val_src is None:
        raise ValueError(f"source is missing array for {attr!r} which is present in destination")
    _copy_array(val_dst, val_src, attr)


def _copy_state(state_in: State, state_out: State) -> None:
    """Copy every field of ``state_in`` into ``state_out``. No-op when ``state_in is state_out``."""
    if state_in is state_out:
        return
    if state_in.body_count != state_out.body_count:
        raise ValueError(f"states have different body_count: src={state_in.body_count}, dst={state_out.body_count}")
    if state_in.joint_coord_count != state_out.joint_coord_count:
        raise ValueError(
            f"states have different joint_coord_count: src={state_in.joint_coord_count}, dst={state_out.joint_coord_count}"
        )
    if state_in.joint_dof_count != state_out.joint_dof_count:
        raise ValueError(
            f"states have different joint_dof_count: src={state_in.joint_dof_count}, dst={state_out.joint_dof_count}"
        )
    _copy_array(state_out.body_q, state_in.body_q, "body_q")
    _copy_array(state_out.body_qd, state_in.body_qd, "body_qd")
    _copy_array(state_out.body_f, state_in.body_f, "body_f")
    _copy_array(state_out.joint_q, state_in.joint_q, "joint_q")
    _copy_array(state_out.joint_qd, state_in.joint_qd, "joint_qd")
    for attr in ("body_q_prev", "body_qdd", "body_parent_f", "joint_parent_f"):
        _copy_optional(state_out, state_in, attr)


def _copy_control(control_in: Control, control_out: Control) -> None:
    """Copy every field of ``control_in`` into ``control_out``. No-op when the two aliases match."""
    if control_in is control_out:
        return
    for attr in ("joint_f", "joint_target_q", "joint_target_qd", "joint_act"):
        _copy_optional(control_out, control_in, attr)


def _copy_contacts(contacts_in: Contacts, contacts_out: Contacts) -> None:
    """Copy every field of ``contacts_in`` into ``contacts_out``. No-op when the two aliases match."""
    if contacts_in is contacts_out:
        return
    if contacts_in.rigid_contact_max != contacts_out.rigid_contact_max:
        raise ValueError(
            f"contacts have different rigid_contact_max: src={contacts_in.rigid_contact_max}, dst={contacts_out.rigid_contact_max}"
        )
    if contacts_in.soft_contact_max != contacts_out.soft_contact_max:
        raise ValueError(
            f"contacts have different soft_contact_max: src={contacts_in.soft_contact_max}, dst={contacts_out.soft_contact_max}"
        )
    for attr in (
        "contact_counters",
        "contact_generation",
        "rigid_contact_tids",
        "rigid_contact_point_id",
        "rigid_contact_shape0",
        "rigid_contact_shape1",
        "rigid_contact_margin0",
        "rigid_contact_margin1",
        "rigid_contact_offset0",
        "rigid_contact_offset1",
        "rigid_contact_point0",
        "rigid_contact_point1",
        "rigid_contact_normal",
    ):
        _copy_array(getattr(contacts_out, attr), getattr(contacts_in, attr), attr)
    for attr in (
        "rigid_contact_match_index",
        "rigid_contact_stiffness",
        "rigid_contact_damping",
        "rigid_contact_friction",
        "rigid_contact_match_index",
        "rigid_contact_new_indices",
        "rigid_contact_new_count",
        "rigid_contact_broken_indices",
        "rigid_contact_broken_count",
        "force",
    ):
        _copy_optional(contacts_out, contacts_in, attr)


###
# Interfaces
###


class SolverSetup:
    """TODO"""

    def __init__(
        self,
        name: str,
        builder: ModelBuilder,
        model: Model,
        solver: SolverBase,
        dt: float,
        rigid_contact_max: int = 32,
        standalone: bool = False,
        verbose: bool = False,
        reset_cb: Callable | None = None,
        control_cb: Callable | None = None,
    ):
        """
        Constructs a solver benchmark setup for a given problem-solver combination.

        The `builder`, `model` and `solver` are constructed outside of the
        setup, because they may require specialized configurations for each
        problem-solver combination, such as extended or custom attributes.

        Args:
            name: The name of the setup.
            builder: The model builder used to construct the model.
            model: The model to be simulated.
            solver: The solver to be used for the simulation.
            dt: The time step size.
            rigid_contact_max: The maximum number of rigid contacts per world.
            standalone: Whether the setup is standalone or not.
            verbose: When ``True``, ``step()`` logs the pre-step ``state_in``
                and the post-step contact buffer via :mod:`logger` at
                ``info`` level.
            reset_cb: The callback to be called to reset the simulation.
            control_cb: The callback to be called to control the simulation.
        """
        # Required input attributes
        self.name: str = name
        self.builder: ModelBuilder = builder
        self.model: Model = model
        self.solver: SolverBase = solver
        self.dt: float = dt

        # Optional input attributes
        self.standalone: bool = standalone
        self.verbose: bool = verbose
        self.reset_cb: Callable | None = reset_cb
        self.control_cb: Callable | None = control_cb

        # Optional cross-solver physics-constraint metrics computed from the
        # Newton state. Attached by the per-solver factories via
        # :func:`_attach_physics_metrics`; populated in :meth:`step` and
        # surfaced by :meth:`SetupRunner.test_final` when at least one setup
        # in the run carries a populated logger.
        self.physics_metrics: PhysicsMetrics | None = None
        self.physics_metrics_logger: PhysicsMetricsLogger | None = None
        # Optional solver-specific auxiliary logger (e.g. SolverKaminoLogger for
        # PADMM iteration diagnostics). Duck-typed: only ``.log()`` is required
        # in ``step()``; ``SetupRunner.test_final`` also calls ``.plot(...)``
        # when present.
        self.aux_logger: Any | None = None
        self.state_out: State | None = None
        self.state_in: State | None = None
        self.control: Control | None = None
        self.contacts: Contacts | None = None
        self._state_in_snapshot: State | None = None

        # Ensure the model is configured for the expected rigid-contact capacity
        # and allocate the relevant state, control and contacts containers.
        self.model.rigid_contact_max = rigid_contact_max

        # _state_in_snapshot holds a frozen pre-step copy of state_in used as
        # state_p in the physics-metrics evaluator (see :meth:`step`), which
        # the solver may otherwise mutate via state_in during step().
        # Only needed in non-standalone mode.
        self.state_out = self.model.state()
        self.state_in = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        if not self.standalone:
            self._state_in_snapshot = self.model.state()

    ###
    # Operations
    ###

    def reset(self, **kwargs):
        """TODO"""
        if self.reset_cb is not None:
            self.reset_cb(**kwargs)

    def actuate(self, **kwargs):
        """TODO"""
        if self.control_cb is not None:
            self.control_cb(**kwargs)

    def _log_verbose_state(self) -> None:
        """Dump ``state_in`` body / joint arrays at ``info`` level. No-op unless ``verbose``."""
        if not self.verbose:
            return
        msg.info("[%s] state_in.body_f:\n%s", self.name, self.state_in.body_f)
        msg.info("[%s] state_in.body_q:\n%s", self.name, self.state_in.body_q)
        msg.info("[%s] state_in.body_qd:\n%s", self.name, self.state_in.body_qd)
        msg.info("[%s] state_in.joint_q: %s", self.name, self.state_in.joint_q)
        msg.info("[%s] state_in.joint_qd: %s", self.name, self.state_in.joint_qd)

    def _log_verbose_contacts(self) -> None:
        """Dump the active rigid-contact slice of ``self.contacts``. No-op unless ``verbose``."""
        if not self.verbose:
            return
        nc = int(self.contacts.rigid_contact_count.numpy()[0])
        msg.info("[%s] contacts.count: %s", self.name, nc)
        msg.info("[%s] contacts.margin0: %s", self.name, self.contacts.rigid_contact_margin0[:nc])
        msg.info("[%s] contacts.margin1: %s", self.name, self.contacts.rigid_contact_margin1[:nc])
        msg.info("[%s] contacts.point0:\n%s", self.name, self.contacts.rigid_contact_point0[:nc])
        msg.info("[%s] contacts.point1:\n%s", self.name, self.contacts.rigid_contact_point1[:nc])
        msg.info("[%s] contacts.normal:\n%s", self.name, self.contacts.rigid_contact_normal[:nc])
        if self.contacts.force is not None:
            msg.info("[%s] contacts.force:\n%s", self.name, self.contacts.force[:nc])

    def step(
        self,
        state_in: State | None = None,
        control_in: Control | None = None,
        contacts_in: Contacts | None = None,
    ):
        """TODO"""
        if self.standalone:
            # Standalone mode: the setup owns ``self.state_in`` / ``self.control`` /
            # ``self.contacts`` directly; no external inputs and no per-step copy.
            self.actuate(state_in=self.state_in, control_out=self.control)
            self.model.collide(self.state_in, self.contacts)
            state_p = self.state_in
        else:
            if state_in is None or not isinstance(state_in, State):
                raise ValueError("state_in must be a State object")
            if control_in is None or not isinstance(control_in, Control):
                raise ValueError("control_in must be a Control object")
            if contacts_in is None or not isinstance(contacts_in, Contacts):
                raise ValueError("contacts_in must be a Contacts object")

            # Non-standalone mode: snapshot the (typically shared) inputs into
            # the setup's private buffers so the solver step / update_contacts
            # / physics metrics operate on a copy isolated from other setups
            # in the runner. ``_state_in_snapshot`` is the frozen ``state_p``
            # passed to the physics-metrics evaluator.
            _copy_state(state_in, self.state_in)
            _copy_state(state_in, self._state_in_snapshot)
            _copy_control(control_in, self.control)
            _copy_contacts(contacts_in, self.contacts)
            state_p = self._state_in_snapshot

        self._log_verbose_state()

        self.solver.step(
            state_in=self.state_in,
            state_out=self.state_out,
            control=self.control,
            contacts=self.contacts,
            dt=self.dt,
        )

        self.solver.update_contacts(contacts=self.contacts, state=self.state_in)

        self._log_verbose_contacts()

        if self.physics_metrics is not None:
            self._evaluate_physics_metrics(state_p=state_p)

        if self.physics_metrics_logger is not None:
            self.physics_metrics_logger.log()
        if self.aux_logger is not None:
            self.aux_logger.log()

        # Standalone mode advances by swapping state_in/state_out so the next call
        # reads from the new post-step state. Non-standalone setups don't swap;
        # the runner manages the canonical state externally.
        if self.standalone:
            self.state_in, self.state_out = self.state_out, self.state_in

    def _evaluate_physics_metrics(self, state_p: State) -> None:
        """Populate ``physics_metrics`` per-contact and per-joint residuals.

        Contact/joint groups are gated by the corresponding ``PhysicsMetrics``
        sub-container so a scene without articulation (e.g. box-on-plane)
        transparently skips the joint pass.
        """
        pm = self.physics_metrics
        if pm.contacts is not None:
            compute_contact_constraint_metrics(self.model, state_p, self.state_out, self.contacts, pm, self.dt)
            compute_per_world_contact_constraint_summary(self.model, self.contacts, pm)
        if pm.joints is not None:
            compute_joint_constraint_metrics(self.model, state_p, self.state_out, pm)
            compute_per_world_joint_constraint_summary(self.model, pm)

    def render(
        self,
        viewer: ViewerBase,
        state_in: State | None = None,
        state_out: State | None = None,
        contacts_in: Contacts | None = None,
    ):
        """TODO"""
        if not self.standalone and (state_in is None or state_out is None or contacts_in is None):
            raise ValueError("state_in, state_out, and contacts_in must be provided if not standalone")
        if state_in is None:
            state_in = self.state_in
        if state_out is None:
            state_out = self.state_out
        if contacts_in is None:
            contacts_in = self.contacts
        viewer.log_state(state_out)
        viewer.log_contacts(contacts_in, state_in)


class SetupRunner:
    """Drives one or more :class:`SolverSetup` instances in one of three comparison modes.

    Three modes are selected via the ``mode`` string:

    - ``"tied"``: the runner owns a single canonical ``state_in`` / ``control``
      / ``contacts`` triplet, steps every setup from those same inputs each
      sub-step, and propagates the leader's post-step state as the next
      canonical state. Best for **single-step accuracy** metrics (every solver
      sees the same per-step problem).
    - ``"independent"``: each setup keeps its own trajectory in its private
      ``state_in`` / ``state_out`` / ``contacts`` buffers; the runner still
      applies the shared ``force_cb`` / ``control_cb`` inputs to each setup's
      canonical state but does not propagate any leader. Best for
      **position-level residual accumulation** — solvers diverge along their
      own trajectories rather than tracking the leader.
    - ``"tied_reference"``: like ``"tied"``, but the canonical trajectory is
      driven by a dedicated :class:`~newton._src.solvers.kamino.accuracy_benchmark.problems.ReferenceLeader`
      running at ``sim_dt / fine_substeps_per_coarse`` (default 10x tighter).
      Coarse followers solve a single-step problem at the paper's ``sim_dt``
      from the canonical state produced by the fine trajectory; their
      ``state_out`` goes to metrics and is discarded. Decouples the shared
      per-step problem from any single coarse solver's bias — best for
      **cross-solver comparison of single-step accuracy** where the current
      tied mode's leader dependence is a concern.

    In every mode setups must be constructed with ``standalone=False`` — the
    runner handles state cycling.

    Tied / tied_reference modes zero MuJoCo's ``qacc_warmstart`` before every
    coarse-comparison substep so each substep is a true cold-start
    single-step solve (see
    :func:`~newton._src.solvers.kamino.accuracy_benchmark.problems.clear_mujoco_warmstart`);
    the check-and-clear is a no-op for non-MuJoCo solvers.

    Conforms to the ``example`` interface expected by :func:`newton.examples.run`
    so it can be passed directly as the example object. For headless
    data-collection runs, prefer :meth:`run_headless` which enforces
    ``num_frames`` and prints a progress bar.
    """

    def __init__(
        self,
        setups: dict[str, SolverSetup],
        leader: str,
        viewer: ViewerBase | None = None,
        force_cb: Callable | None = None,
        control_cb: Callable | None = None,
        fps: float = 50.0,
        sim_substeps: int = 20,
        verbose: bool = False,
        mode: str = MODE_TIED,
        reference_leader: Any = None,
        fine_substeps_per_coarse: int = 10,
        use_cuda_graph: bool = True,
    ):
        """Construct a runner around a dict of solver setups.

        Args:
            setups: Mapping of name to :class:`SolverSetup`. All setups must be
                non-standalone (``standalone=False``).
            leader: Key of the leader setup. In tied mode drives state
                propagation; in independent / tied_reference mode used only to
                pick the model backing the runner's shared ``control`` buffer
                (for ``control_cb``) and the viewer's ``set_model`` call.
            viewer: Optional viewer. When ``None``, ``render()`` is a no-op and the
                runner skips the default ``viewer.apply_forces`` fallback.
            force_cb: Optional callable ``force_cb(state, contacts, sim_time)``
                applied to ``state_in`` at the start of every sub-step (after
                ``clear_forces``). ``sim_time`` is the substep's physical time
                (``self.sim_time + i * self.sim_dt``); passing it explicitly
                keeps the callback stateless so all three modes are equivalent
                for the leader — in independent mode the runner fans
                ``force_cb`` out over every setup per sub-step, and in
                tied_reference mode both the coarse comparison step and the
                fine trajectory read the correct sub-step time.
                When set, replaces the default ``viewer.apply_forces`` fallback.
            control_cb: Optional callable ``control_cb(control, sim_time)`` invoked
                at the start of every sub-step to update the shared canonical
                :class:`Control`. In independent mode the shared control is
                fanned out to every setup via ``_copy_control``. In
                tied_reference mode the reference leader uses its own dedicated
                :class:`Control` buffer so fine-dt updates don't clobber the
                coarse follower's control between coarse substeps.
            fps: Viewer/render frame rate. ``frame_dt = 1/fps`` is the per-call
                advance of ``self.sim_time``.
            sim_substeps: Number of physics sub-steps per ``step()`` call.
                ``sim_dt = frame_dt / sim_substeps`` is the physical integration
                step size. Each setup's solver should have been constructed with
                this same ``sim_dt``.
            verbose: When ``True``, sets ``setup.verbose = True`` on every setup so
                every sub-step dumps state / contact / metrics diagnostics. Noisy;
                pair with a low ``--num-frames`` value.
            mode: One of ``"tied"`` / ``"independent"`` / ``"tied_reference"``
                — see class docstring.
            reference_leader: Required when ``mode == "tied_reference"``: a
                :class:`~newton._src.solvers.kamino.accuracy_benchmark.problems.ReferenceLeader`
                built by the per-example ``build_*_run`` factory. Ignored in
                other modes.
            fine_substeps_per_coarse: Only used in tied_reference mode. Number
                of fine sub-steps the reference leader advances per coarse
                sub-step; must equal ``sim_dt / reference_leader.dt`` (validated
                at construction time). Defaults to 10.
            use_cuda_graph: When ``True`` (and running on CUDA), capture the
                per-frame substep loop into a CUDA graph on the first call to
                :meth:`step` and replay it on subsequent calls. Automatically
                disabled if ``control_cb`` or ``viewer`` is set (both introduce
                host-side operations that can't be captured), if the device is
                not CUDA, or if ``wp.config.verify_cuda`` is on.
        """
        if leader not in setups:
            raise ValueError(f"leader {leader!r} not in setups: {list(setups)}")
        for name, setup in setups.items():
            if setup.standalone:
                raise ValueError(
                    f"setup {name!r} must be standalone=False when used inside SetupRunner; "
                    f"the runner manages state cycling in every mode"
                )
        if sim_substeps < 1:
            raise ValueError(f"sim_substeps must be >= 1, got {sim_substeps}")
        if mode not in _ALLOWED_MODES:
            raise ValueError(f"mode must be one of {sorted(_ALLOWED_MODES)}, got {mode!r}")
        if mode == MODE_TIED_REFERENCE:
            if reference_leader is None:
                raise ValueError("mode='tied_reference' requires a reference_leader")
            if fine_substeps_per_coarse < 1:
                raise ValueError(f"fine_substeps_per_coarse must be >= 1, got {fine_substeps_per_coarse}")

        if verbose:
            for setup in setups.values():
                setup.verbose = True

        self.setups: dict[str, SolverSetup] = setups
        self.leader_name: str = leader
        self.leader: SolverSetup = setups[leader]
        self.viewer: ViewerBase | None = viewer
        self.force_cb: Callable | None = force_cb
        self.control_cb: Callable | None = control_cb
        self.mode: str = mode
        self.reference_leader: Any = reference_leader if mode == MODE_TIED_REFERENCE else None
        self.fine_substeps_per_coarse: int = fine_substeps_per_coarse

        self.fps: float = fps
        self.sim_substeps: int = sim_substeps
        self.frame_dt: float = 1.0 / fps
        self.sim_dt: float = self.frame_dt / sim_substeps
        self.sim_time: float = 0.0

        # Consistency check: reference leader dt must divide sim_dt exactly (up
        # to floating-point round-off) — otherwise the fine trajectory would
        # not reach the next coarse sub-step boundary in an integer number of
        # substeps.
        if self.reference_leader is not None:
            expected_fine_dt = self.sim_dt / self.fine_substeps_per_coarse
            if not np.isclose(self.reference_leader.dt, expected_fine_dt, rtol=1e-6):
                raise ValueError(
                    f"reference_leader.dt={self.reference_leader.dt} does not match "
                    f"sim_dt / fine_substeps_per_coarse = {self.sim_dt} / "
                    f"{self.fine_substeps_per_coarse} = {expected_fine_dt}"
                )

        # In tied / tied_reference mode, warn-loudly if a follower's Kamino
        # solver has warmstart on: each coarse substep is meant to be a cold
        # single-step problem. MuJoCo's own warmstart is unavoidable at the
        # backend level and is neutralised per-substep via clear_mujoco_warmstart.
        if mode in (MODE_TIED, MODE_TIED_REFERENCE):
            # Deferred import to avoid a circular dependency (problems.py
            # imports SolverSetup from this module).
            from .problems import assert_warmstart_disabled  # noqa: PLC0415

            for name, setup in setups.items():
                assert_warmstart_disabled(setup.solver, context=f"setup {name!r}")
            if self.reference_leader is not None:
                assert_warmstart_disabled(
                    self.reference_leader.solver,
                    context=f"reference_leader {self.reference_leader.name!r}",
                )

        # Shared canonical buffers. In tied mode these hold the live state
        # every setup steps from. In independent mode ``state_in`` / ``contacts``
        # are unused (each setup owns its own) but ``control`` is still used
        # as the ``control_cb`` scratch buffer, fanned out to every setup.
        # In tied_reference mode ``state_in`` is unused (the canonical state
        # lives on the reference leader) but ``control`` and ``contacts`` are
        # still owned by the runner and used per coarse sub-step.
        self.model: Model = self.leader.model
        self.state_in: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        # CUDA-graph capture is only meaningful when the substep loop is a pure
        # sequence of device kernels — ``control_cb`` (host-side ``.assign``, e.g.
        # DR Legs animation) and ``viewer.apply_forces`` (host-side event
        # handling) both introduce non-captureable operations.
        self._use_cuda_graph: bool = bool(
            use_cuda_graph
            and wp.get_device().is_cuda
            and not wp.config.verify_cuda
            and self.control_cb is None
            and self.viewer is None
        )
        self._graph: Any | None = None

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.reset()

    def reset(self) -> None:
        """Re-initialise the canonical state(s) by calling each setup's ``reset_cb``.

        In **tied** mode, followers reset first, then the leader, all writing into
        the shared ``self.state_in``. This matters when a follower's ``reset_cb``
        writes a base pose via forward kinematics (touching ``joint_q`` and the
        derived ``body_q``/``body_qd``) and the leader's ``reset_cb`` applies
        solver-internal initialisation on top — running the leader last prevents
        the follower from overwriting it.

        In **independent** mode, each setup resets its own private ``state_in``
        buffer instead. There is no cross-setup ordering constraint.

        In **tied_reference** mode, every follower resets its private
        ``state_in`` (its buffer is overwritten each sub-step, but we still
        need to reset any solver-internal state), then the reference leader
        resets its own canonical ``state_in``.
        """
        if self.mode == MODE_INDEPENDENT:
            for setup in self.setups.values():
                setup.reset(state_out=setup.state_in)
        elif self.mode == MODE_TIED_REFERENCE:
            for setup in self.setups.values():
                setup.reset(state_out=setup.state_in)
            self.reference_leader.reset()
        else:  # MODE_TIED
            for name, setup in self.setups.items():
                if name == self.leader_name:
                    continue
                setup.reset(state_out=self.state_in)
            self.leader.reset(state_out=self.state_in)

    def step(self) -> None:
        """Run ``sim_substeps`` physics sub-steps and advance ``sim_time`` by ``frame_dt``.

        Dispatches to the mode-specific ``_step_*`` method. When
        ``use_cuda_graph`` is enabled (default on CUDA without ``control_cb`` /
        ``viewer``), the first call captures the substep loop into a graph and
        subsequent calls replay it via :func:`wp.capture_launch`.
        """
        if self._use_cuda_graph:
            if self._graph is None:
                self._graph = self._capture_step_graph()
            if self._graph is not None:
                wp.capture_launch(self._graph)
                self.sim_time += self.frame_dt
                return
        self._step_fn()()
        self.sim_time += self.frame_dt

    def _step_fn(self) -> Callable[[], None]:
        """Return the substep function for the active mode."""
        if self.mode == MODE_INDEPENDENT:
            return self._step_independent
        if self.mode == MODE_TIED_REFERENCE:
            return self._step_tied_reference
        return self._step_tied

    def _capture_step_graph(self):
        """Capture the current mode's substep loop into a reusable CUDA graph.

        Falls back to ``None`` (i.e. the eager path in :meth:`step`) if the
        capture raises — e.g. a solver internally issues a host-side operation
        we can't work around here. The failed capture attempt is logged so the
        first-frame slowdown is diagnosable.
        """
        step_fn = self._step_fn()
        try:
            with wp.ScopedCapture() as capture:
                step_fn()
            msg.notif("CUDA graph captured (%s substeps, mode=%s)", self.sim_substeps, self.mode)
            return capture.graph
        except Exception as exc:
            msg.warning("CUDA graph capture failed (%s); falling back to eager step", exc)
            self._use_cuda_graph = False
            return None

    def _clear_follower_warmstart(self) -> None:
        """Zero MuJoCo's per-step warmstart on every follower for the next coarse step.

        Called in tied / tied_reference modes so every coarse sub-step is a true
        cold-start single-step problem. Kamino has warmstart disabled at config
        level (asserted in ``__init__``) and XPBD has no persistent warmstart,
        so this is a no-op for those solvers.
        """
        # Deferred import: problems.py imports SolverSetup from this module.
        from .problems import clear_mujoco_warmstart  # noqa: PLC0415

        for setup in self.setups.values():
            clear_mujoco_warmstart(setup.solver)

    def _step_tied(self) -> None:
        """Run ``sim_substeps`` physics sub-steps against the shared canonical state.

        Per sub-step:
            1. Clear MuJoCo warmstart so the coarse step is a cold-start solve.
            2. Clear forces on ``state_in``.
            3. Apply external forces (via ``force_cb`` if set, else ``viewer.apply_forces``).
            4. Run collision detection into ``contacts``.
            5. Step every setup from ``(state_in, control, contacts)`` — each writes to
               its own ``state_out``.
            6. Swap ``self.state_in`` with ``self.leader.state_out`` so the next sub-step
               reads from the new post-step state. After the swap, ``state_in`` holds
               the post-step state and ``self.leader.state_out`` holds the pre-step
               state (matching the contact geometry).
        """
        for i in range(self.sim_substeps):
            substep_time = self.sim_time + i * self.sim_dt
            self._clear_follower_warmstart()
            self.state_in.clear_forces()
            if self.control_cb is not None:
                self.control_cb(control=self.control, sim_time=substep_time)
            if self.force_cb is not None:
                self.force_cb(state=self.state_in, contacts=self.contacts, sim_time=substep_time)
            elif self.viewer is not None:
                self.viewer.apply_forces(self.state_in)

            self.model.collide(self.state_in, self.contacts)

            for setup in self.setups.values():
                setup.step(state_in=self.state_in, control_in=self.control, contacts_in=self.contacts)

            self.state_in, self.leader.state_out = self.leader.state_out, self.state_in

    def _step_tied_reference(self) -> None:
        """Run ``sim_substeps`` coarse sub-steps against a fine-dt canonical trajectory.

        Per coarse sub-step at coarse time ``t``:
            1. Take ``canonical = reference_leader.state_in`` (state at ``t``).
            2. Clear its ``body_f`` (leftover from the last fine sub-step) and
               apply the coarse-tick force / control (both sampled at ``t``).
            3. Coarse collide from ``canonical`` into ``self.contacts``.
            4. Clear MuJoCo warmstart on every follower and step each of them
               once from ``(canonical, self.control, self.contacts)`` at coarse
               dt. ``setup.state_out`` feeds ``physics_metrics``; the state is
               otherwise discarded.
            5. Advance the reference leader by ``fine_substeps_per_coarse`` fine
               sub-steps to reach ``t + coarse_dt``. This is where the canonical
               trajectory actually advances.
        """
        for i in range(self.sim_substeps):
            substep_time = self.sim_time + i * self.sim_dt
            canonical = self.reference_leader.state_in
            canonical.clear_forces()
            if self.control_cb is not None:
                self.control_cb(control=self.control, sim_time=substep_time)
            if self.force_cb is not None:
                self.force_cb(state=canonical, contacts=self.contacts, sim_time=substep_time)

            self.model.collide(canonical, self.contacts)

            self._clear_follower_warmstart()
            for setup in self.setups.values():
                setup.step(state_in=canonical, control_in=self.control, contacts_in=self.contacts)

            self.reference_leader.step_n(
                n_substeps=self.fine_substeps_per_coarse,
                sim_time_start=substep_time,
                force_cb=self.force_cb,
                control_cb=self.control_cb,
            )

    def _step_independent(self) -> None:
        """Run ``sim_substeps`` physics sub-steps with each setup on its own trajectory.

        Per sub-step, and per setup:
            1. Clear forces on the setup's private ``state_in``.
            2. Apply external forces (via ``force_cb`` or ``viewer.apply_forces``).
            3. Run collision detection into the setup's private ``contacts``.
            4. Call ``setup.step()`` on its own buffers (self-copies short-circuited).
            5. Swap the setup's ``state_in`` ↔ ``state_out`` so the next sub-step
               reads from the new post-step state.

        ``control_cb`` writes into the runner's shared ``self.control`` once per
        sub-step, then setup.step's internal ``_copy_control`` fans it out to each
        setup — every solver sees the same PD targets even when trajectories diverge.
        """
        for i in range(self.sim_substeps):
            substep_time = self.sim_time + i * self.sim_dt
            if self.control_cb is not None:
                self.control_cb(control=self.control, sim_time=substep_time)
            for setup in self.setups.values():
                state_in = setup.state_in
                state_in.clear_forces()
                if self.force_cb is not None:
                    self.force_cb(state=state_in, contacts=setup.contacts, sim_time=substep_time)
                elif self.viewer is not None:
                    self.viewer.apply_forces(state_in)
                setup.model.collide(state_in, setup.contacts)
                setup.step(state_in=state_in, control_in=self.control, contacts_in=setup.contacts)
                setup.state_in, setup.state_out = setup.state_out, setup.state_in

    def render(self) -> None:
        """Log the canonical post-step state and pre-step contact geometry to the viewer.

        In tied mode uses the shared ``self.state_in`` (post-swap: holds the
        leader's post-step state) together with ``self.leader.contacts``. In
        independent mode uses the leader's private ``state_in`` (post-swap:
        also post-step) so the viewer shows the leader's trajectory. In
        tied_reference mode displays the reference leader's canonical state
        (the fine-dt trajectory driving every follower).
        """
        if self.viewer is None:
            return
        if self.mode == MODE_INDEPENDENT:
            state_display = self.leader.state_in
        elif self.mode == MODE_TIED_REFERENCE:
            state_display = self.reference_leader.state_in
        else:
            state_display = self.state_in
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(state_display)
        self.viewer.log_contacts(self.leader.contacts, self.leader.state_out)
        self.viewer.end_frame()

    def run_headless(self, num_frames: int, progress: bool = True) -> None:
        """Run ``num_frames`` frames without a viewer, optionally printing a progress bar.

        Called from paper scripts when ``args.headless`` — the newton example runner
        doesn't enforce ``args.num_frames`` for the GL viewer's headless mode, so we
        drive the loop directly here. ``wp.synchronize()`` after each ``step`` gives
        the progress bar a stable FPS estimate; the ``test_final`` call still lives
        on the caller side so we don't force an output layout choice.

        Args:
            num_frames: Number of frames to advance (each frame runs
                :attr:`sim_substeps` physics sub-steps).
            progress: When ``True``, refreshes an ASCII progress bar with ETA / FPS.
        """
        msg.notif(
            "Running %s frames (%s sub-steps each, mode=%s, leader=%s, cuda_graph=%s)",
            num_frames,
            self.sim_substeps,
            self.mode,
            self.leader_name,
            self._use_cuda_graph,
        )
        start_time = time.time()
        for i in range(num_frames):
            self.step()
            # Sync so the progress-bar FPS/ETA reflects real device work rather
            # than queued kernel launches. Cheap in the CUDA-graph path (one
            # sync per frame after ``sim_substeps`` fused launches); still
            # negligible for the eager path.
            wp.synchronize()
            if progress:
                print_progress_bar(i + 1, num_frames, start_time, prefix="Progress", suffix="")
        msg.notif("Finished headless run")

    def test_final(self, problem_name: str = "comparison", output_path: str | None = None) -> None:
        """Emit end-of-run comparison plots and tables for every attached logger.

        When at least one setup carries a populated
        :class:`PhysicsMetricsLogger`, emits
        ``<problem_name>_physics_metrics{,_logscale}.pdf`` overlay plots and a
        ``<problem_name>_physics_metrics_table.csv`` (also rendered to console
        with color rankings). Per-setup ``aux_logger.plot(...)`` outputs are
        emitted as ``<problem_name>_<setup_name>_aux.pdf`` when the logger is
        attached.

        Args:
            problem_name: Used as both the filename stem and the default sub-directory
                under ``<this_file>/output/``.
            output_path: Optional override for the output directory. When ``None``,
                defaults to ``<this_file>/output/<problem_name>``.
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", problem_name)
        os.makedirs(output_path, exist_ok=True)

        physics_loggers = {
            name: setup.physics_metrics_logger
            for name, setup in self.setups.items()
            if setup.physics_metrics_logger is not None and setup.physics_metrics_logger.num_logged_frames > 0
        }
        if physics_loggers:
            PhysicsMetricsLogger.plot_comparison(
                physics_loggers, filename=f"{problem_name}_physics_metrics", path=output_path, ext="pdf", grid=True
            )
            PhysicsMetricsLogger.plot_comparison(
                physics_loggers,
                filename=f"{problem_name}_physics_metrics_logscale",
                path=output_path,
                ext="pdf",
                grid=True,
                log_scale=True,
            )
            PhysicsMetricsLogger.table_comparison(
                physics_loggers,
                filename=f"{problem_name}_physics_metrics_table",
                path=output_path,
                to_console=True,
                color_rankings=True,
            )

        for name, setup in self.setups.items():
            if setup.aux_logger is None:
                continue
            plot = getattr(setup.aux_logger, "plot", None)
            if plot is None:
                continue
            plot(path=output_path, ext="pdf", filename=f"{problem_name}_{name}_aux")
