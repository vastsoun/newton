# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared benchmark scaffolding: per-solver ``SolverSetup`` and multi-solver ``SetupRunner``.

Both classes support the two logger flavours used by the accuracy benchmark:

- :class:`SolutionMetricsNewton` + :class:`SolutionMetricsLogger` — Kamino-front-end
  residuals; always instantiated by the setup.
- :class:`PhysicsMetrics` + :class:`PhysicsMetricsLogger` — cross-solver constraint
  residuals computed directly from the Newton state; opt-in via
  ``setup.physics_metrics`` / ``setup.physics_metrics_logger``.

Plus an opt-in ``setup.aux_logger`` slot for solver-specific PADMM diagnostics
(:class:`SolverKaminoLogger`), which are complementary to both logger flavours.
"""

import os
import time
from collections.abc import Callable
from typing import Any

import warp as wp

from ....sim import Contacts, Control, Model, ModelBuilder, State
from ....solvers import SolverBase
from ....viewer import ViewerBase
from .._src.metrics import SolutionMetricsLogger, SolutionMetricsNewton
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

###
# Module interface
###

__all__ = ["SetupRunner", "SolverSetup"]


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
        kwargs_builder: dict | None = None,
        kwargs_logger: dict | None = None,
        with_solution_metrics: bool = True,
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
            verbose: When ``True``, ``step()`` logs the pre-step ``state_in``,
                the post-step contact buffer, and the post-evaluate metrics
                contact data via :mod:`logger` at ``info`` level.
            reset_cb: The callback to be called to reset the simulation.
            control_cb: The callback to be called to control the simulation.
            kwargs_builder: Additional keyword arguments to be passed to the builder.
            kwargs_logger: Additional keyword arguments to be passed to the logger.
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

        # Derived attributes
        self.metrics: SolutionMetricsNewton | None = None
        self.logger: SolutionMetricsLogger | None = None
        # Optional second logger over a solver-internal metrics object (e.g.
        # ``solver._solver_kamino.metrics`` for SolverKamino). Attached by
        # the per-solver factories when the solver exposes such an object;
        # logged in ``step()`` and surfaced by ``SetupRunner.test_final``.
        self.solver_logger: SolutionMetricsLogger | None = None
        # Optional cross-solver physics-constraint metrics computed from the
        # Newton state and populated in ``step()`` when both are attached.
        # See ``PhysicsMetrics`` / ``PhysicsMetricsLogger`` for the underlying
        # buffers; ``SetupRunner.test_final`` renders their comparison output
        # when at least one setup in the run carries a populated logger.
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

        # TODO Add check that the required extended attributes are present in the builder

        # Ensure the model is configured for the expected rigid-contact capacity
        # and allocate the relevant state, control and contacts containers.
        self.model.rigid_contact_max = rigid_contact_max

        # _state_in_snapshot holds a frozen pre-step copy of state_in used as
        # state_p in metrics.evaluate, which the solver may otherwise mutate
        # via state_in during step(). Only needed in non-standalone mode.
        self.state_out = self.model.state()
        self.state_in = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        if not self.standalone:
            self._state_in_snapshot = self.model.state()

        # Finalise the metrics and logger. Opt-in via ``with_solution_metrics``:
        # the ``SolutionMetricsNewton`` front-end wraps Kamino's converters and
        # is not maintained in sync with the geometry / solver APIs — paper
        # setups turn it off and rely on ``PhysicsMetrics`` (see below) for the
        # cross-solver comparison.
        if with_solution_metrics:
            # NOTE: A dedicated builder finalize call is used because ``ModelKamino``
            # currently modifies the source model in-place, which would otherwise
            # break the assumptions of the primary solver's own model.
            if kwargs_builder is None:
                kwargs_builder = {"skip_validation_joints": True}
            metrics_model = self.builder.finalize(**kwargs_builder)
            metrics_model.rigid_contact_max = rigid_contact_max
            self.metrics = SolutionMetricsNewton(
                model=metrics_model,
                dt=self.dt,
                sparse=False,
            )
            if kwargs_logger is None:
                kwargs_logger = {
                    "max_frames": 5000,
                    "mode": SolutionMetricsLogger.Mode.BOUNDED,
                    "dt": self.dt,
                }
            self.logger = SolutionMetricsLogger(metrics=self.metrics, **kwargs_logger)

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

    def _log_verbose_metrics(self) -> None:
        """Dump the active slice of ``metrics._contacts`` (ContactsKamino). No-op unless ``verbose``."""
        if not self.verbose or self.metrics is None:
            return
        # ``_contacts`` is private but stable enough for diagnostic logging;
        # mirrors the sandbox comparison-example access path.
        kcontacts = self.metrics._contacts
        nc = int(kcontacts.model_active_contacts.numpy()[0])
        msg.info("[%s] metrics.contacts.count: %s", self.name, nc)
        msg.info("[%s] metrics.contacts.margins:\n%s", self.name, kcontacts.data.margins[:nc])
        msg.info("[%s] metrics.contacts.position_A:\n%s", self.name, kcontacts.data.position_A[:nc])
        msg.info("[%s] metrics.contacts.position_B:\n%s", self.name, kcontacts.data.position_B[:nc])
        msg.info("[%s] metrics.contacts.gapfunc:\n%s", self.name, kcontacts.data.gapfunc[:nc])
        msg.info("[%s] metrics.contacts.reaction:\n%s", self.name, kcontacts.data.reaction[:nc])

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

            # Non-standalone mode: snapshot the (typically shared) inputs into the
            # setup's private buffers so the solver step / update_contacts /
            # metrics.evaluate operate on a copy that's isolated from other setups
            # in the runner. ``_state_in_snapshot`` is the frozen ``state_p`` for
            # the metrics evaluator.
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

        if self.metrics is not None:
            self.metrics.evaluate(
                state=self.state_out,
                state_p=state_p,
                control=self.control,
                contacts=self.contacts,
            )

        self._log_verbose_metrics()

        if self.physics_metrics is not None:
            self._evaluate_physics_metrics(state_p=state_p)

        if self.logger is not None:
            self.logger.log()
        if self.solver_logger is not None:
            self.solver_logger.log()
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
    """Drives one or more :class:`SolverSetup` instances in one of two comparison modes.

    Two modes are exposed via the ``independent`` flag:

    - **Tied** (``independent=False``): the runner owns a single canonical
      ``state_in`` / ``control`` / ``contacts`` triplet, steps every setup from
      those same inputs each sub-step, and propagates the leader's post-step
      state as the next canonical state. Best for **single-step accuracy**
      metrics (every solver sees the same per-step problem).
    - **Independent** (``independent=True``): each setup keeps its own
      trajectory in its private ``state_in`` / ``state_out`` / ``contacts``
      buffers; the runner still applies the shared ``force_cb`` / ``control_cb``
      inputs to each setup's canonical state but does not propagate any leader.
      Best for **position-level residual accumulation** — solvers diverge along
      their own trajectories rather than tracking the leader.

    In both modes setups must be constructed with ``standalone=False`` — the
    runner handles state cycling for independent mode via an explicit
    ``state_in`` ↔ ``state_out`` swap on each setup.

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
        independent: bool = False,
        use_cuda_graph: bool = True,
    ):
        """Construct a runner around a dict of solver setups.

        Args:
            setups: Mapping of name to :class:`SolverSetup`. All setups must be
                non-standalone (``standalone=False``).
            leader: Key of the leader setup. In tied mode drives state propagation;
                in independent mode used only to pick the model backing the
                runner's shared ``control`` buffer (for ``control_cb``) and the
                viewer's ``set_model`` call.
            viewer: Optional viewer. When ``None``, ``render()`` is a no-op and the
                runner skips the default ``viewer.apply_forces`` fallback.
            force_cb: Optional callable ``force_cb(state, contacts)`` applied to
                ``state_in`` at the start of every sub-step (after ``clear_forces``).
                When set, replaces the default ``viewer.apply_forces`` fallback.
                In independent mode, invoked once per setup with that setup's own
                ``state_in`` / ``contacts``.
            control_cb: Optional callable ``control_cb(control, sim_time)`` invoked
                at the start of every sub-step to update the shared canonical
                :class:`Control`. In independent mode the shared control is fanned
                out to every setup via ``_copy_control``.
            fps: Viewer/render frame rate. ``frame_dt = 1/fps`` is the per-call
                advance of ``self.sim_time``.
            sim_substeps: Number of physics sub-steps per ``step()`` call.
                ``sim_dt = frame_dt / sim_substeps`` is the physical integration
                step size. Each setup's solver should have been constructed with
                this same ``sim_dt``.
            verbose: When ``True``, sets ``setup.verbose = True`` on every setup so
                every sub-step dumps state / contact / metrics diagnostics. Noisy;
                pair with a low ``--num-frames`` value.
            independent: When ``True``, run each setup on its own trajectory (no
                leader propagation). See class docstring for when to pick which
                mode.
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
                    f"the runner manages state cycling in both tied and independent modes"
                )
        if sim_substeps < 1:
            raise ValueError(f"sim_substeps must be >= 1, got {sim_substeps}")

        if verbose:
            for setup in setups.values():
                setup.verbose = True

        self.setups: dict[str, SolverSetup] = setups
        self.leader_name: str = leader
        self.leader: SolverSetup = setups[leader]
        self.viewer: ViewerBase | None = viewer
        self.force_cb: Callable | None = force_cb
        self.control_cb: Callable | None = control_cb
        self.independent: bool = independent

        self.fps: float = fps
        self.sim_substeps: int = sim_substeps
        self.frame_dt: float = 1.0 / fps
        self.sim_dt: float = self.frame_dt / sim_substeps
        self.sim_time: float = 0.0

        # Shared canonical buffers. In tied mode these hold the live state
        # every setup steps from. In independent mode ``state_in`` / ``contacts``
        # are unused (each setup owns its own) but ``control`` is still used
        # as the ``control_cb`` scratch buffer, fanned out to every setup.
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
        """
        if self.independent:
            for setup in self.setups.values():
                setup.reset(state_out=setup.state_in)
        else:
            for name, setup in self.setups.items():
                if name == self.leader_name:
                    continue
                setup.reset(state_out=self.state_in)
            self.leader.reset(state_out=self.state_in)

    def step(self) -> None:
        """Run ``sim_substeps`` physics sub-steps and advance ``sim_time`` by ``frame_dt``.

        Dispatches to :meth:`_step_tied` or :meth:`_step_independent` based on
        ``self.independent``. See class docstring for the difference. When
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
        if self.independent:
            self._step_independent()
        else:
            self._step_tied()
        self.sim_time += self.frame_dt

    def _capture_step_graph(self):
        """Capture the current mode's substep loop into a reusable CUDA graph.

        Falls back to ``None`` (i.e. the eager path in :meth:`step`) if the
        capture raises — e.g. a solver internally issues a host-side operation
        we can't work around here. The failed capture attempt is logged so the
        first-frame slowdown is diagnosable.
        """
        step_fn = self._step_independent if self.independent else self._step_tied
        try:
            with wp.ScopedCapture() as capture:
                step_fn()
            msg.notif("CUDA graph captured (%s substeps)", self.sim_substeps)
            return capture.graph
        except Exception as exc:
            msg.warning("CUDA graph capture failed (%s); falling back to eager step", exc)
            self._use_cuda_graph = False
            return None

    def _step_tied(self) -> None:
        """Run ``sim_substeps`` physics sub-steps against the shared canonical state.

        Per sub-step:
            1. Clear forces on ``state_in``.
            2. Apply external forces (via ``force_cb`` if set, else ``viewer.apply_forces``).
            3. Run collision detection into ``contacts``.
            4. Step every setup from ``(state_in, control, contacts)`` — each writes to
               its own ``state_out``.
            5. Swap ``self.state_in`` with ``self.leader.state_out`` so the next sub-step
               reads from the new post-step state. After the swap, ``state_in`` holds
               the post-step state and ``self.leader.state_out`` holds the pre-step
               state (matching the contact geometry).
        """
        for _ in range(self.sim_substeps):
            self.state_in.clear_forces()
            if self.control_cb is not None:
                self.control_cb(control=self.control, sim_time=self.sim_time)
            if self.force_cb is not None:
                self.force_cb(state=self.state_in, contacts=self.contacts)
            elif self.viewer is not None:
                self.viewer.apply_forces(self.state_in)

            self.model.collide(self.state_in, self.contacts)

            for setup in self.setups.values():
                setup.step(state_in=self.state_in, control_in=self.control, contacts_in=self.contacts)

            self.state_in, self.leader.state_out = self.leader.state_out, self.state_in

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
        for _ in range(self.sim_substeps):
            if self.control_cb is not None:
                self.control_cb(control=self.control, sim_time=self.sim_time)
            for setup in self.setups.values():
                state_in = setup.state_in
                state_in.clear_forces()
                if self.force_cb is not None:
                    self.force_cb(state=state_in, contacts=setup.contacts)
                elif self.viewer is not None:
                    self.viewer.apply_forces(state_in)
                setup.model.collide(state_in, setup.contacts)
                setup.step(state_in=state_in, control_in=self.control, contacts_in=setup.contacts)
                setup.state_in, setup.state_out = setup.state_out, setup.state_in

    def render(self) -> None:
        """Log the canonical post-step state and pre-step contact geometry to the viewer.

        In tied mode uses the shared ``self.state_in`` (post-swap: holds the leader's
        post-step state) together with ``self.leader.contacts``. In independent mode
        uses the leader's private ``state_in`` (post-swap: also post-step) so the
        viewer shows the leader's trajectory; the follower trajectories diverge on
        their own private buffers but are not currently multiplexed in the viewer.
        """
        if self.viewer is None:
            return
        state_display = self.state_in if not self.independent else self.leader.state_in
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
            "independent" if self.independent else "tied",
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

        Always emits ``<problem_name>_solvers.pdf`` comparing the front-end
        :class:`SolutionMetricsNewton` of every setup. For any setup whose
        ``solver_logger`` is populated (i.e. the solver exposes an internal
        metrics object), also emits ``<problem_name>_<setup_name>.pdf``
        comparing that solver's internal metrics against its own front-end
        evaluator.

        When at least one setup carries a populated
        :class:`PhysicsMetricsLogger`, additionally emits
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

        front_end_loggers = {name: setup.logger for name, setup in self.setups.items() if setup.logger is not None}
        if front_end_loggers:
            SolutionMetricsLogger.plot_comparison(
                front_end_loggers, filename=f"{problem_name}_solvers", path=output_path, ext="pdf", grid=True
            )

        for name, setup in self.setups.items():
            if setup.solver_logger is None or setup.logger is None:
                continue
            internal_vs_front_end = {"solver": setup.solver_logger, name: setup.logger}
            SolutionMetricsLogger.plot_comparison(
                internal_vs_front_end,
                filename=f"{problem_name}_{name}",
                path=output_path,
                ext="pdf",
                grid=True,
            )

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
