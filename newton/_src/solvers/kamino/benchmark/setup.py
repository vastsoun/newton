# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import os
from collections.abc import Callable

import warp as wp

from ....sim import Contacts, Control, Model, ModelBuilder, State
from ....solvers import SolverBase
from ....viewer import ViewerBase
from .._src.metrics import SolutionMetricsLogger, SolutionMetricsNewton
from .._src.utils import logger as msg

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
    """
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
    """Copy every field of ``state_in`` into ``state_out``."""
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
    """Copy every field of ``control_in`` into ``control_out``."""
    for attr in ("joint_f", "joint_target_pos", "joint_target_vel", "joint_act"):
        _copy_optional(control_out, control_in, attr)


def _copy_contacts(contacts_in: Contacts, contacts_out: Contacts) -> None:
    """Copy every field of ``contacts_in`` into ``contacts_out``."""
    if contacts_in.rigid_contact_max != contacts_out.rigid_contact_max:
        raise ValueError(
            f"contacts have different rigid_contact_max: src={contacts_in.rigid_contact_max}, dst={contacts_out.rigid_contact_max}"
        )
    if contacts_in.soft_contact_max != contacts_out.soft_contact_max:
        raise ValueError(
            f"contacts have different soft_contact_max: src={contacts_in.soft_contact_max}, dst={contacts_out.soft_contact_max}"
        )
    # ``contact_counters`` is the packed counter array; ``rigid_contact_count`` and
    # ``soft_contact_count`` are 1-element slice views into it and update implicitly.
    _copy_array(contacts_out.contact_counters, contacts_in.contact_counters, "contact_counters")
    for attr in (
        "rigid_contact_tids",
        "rigid_contact_point_id",
        "rigid_contact_shape0",
        "rigid_contact_shape1",
        "rigid_contact_margin0",
        "rigid_contact_margin1",
        "rigid_contact_point0",
        "rigid_contact_point1",
        "rigid_contact_offset0",
        "rigid_contact_offset1",
        "rigid_contact_normal",
    ):
        _copy_array(getattr(contacts_out, attr), getattr(contacts_in, attr), attr)
    _copy_optional(contacts_out, contacts_in, "force")


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

        # Finalise the metrics and logger
        # NOTE: We need to create a new model for the metrics to operate on,
        # because ModelKamino currently modifies the model in-place which can
        # break the assumptions of each solver.
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
        if not self.verbose:
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

        self.metrics.evaluate(
            state=self.state_out,
            state_p=state_p,
            control=self.control,
            contacts=self.contacts,
        )

        self._log_verbose_metrics()

        self.logger.log()
        if self.solver_logger is not None:
            self.solver_logger.log()

        # Standalone mode advances by swapping state_in/state_out so the next call
        # reads from the new post-step state. Non-standalone setups don't swap;
        # the runner manages the canonical state externally.
        if self.standalone:
            self.state_in, self.state_out = self.state_out, self.state_in

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
    """Drives one or more :class:`SolverSetup` instances against a shared canonical state.

    The runner allocates its own canonical ``state_in``, ``control``, and ``contacts``
    buffers from the leader setup's model and steps every setup from those same inputs
    each sub-step. Setups inside the runner must be ``standalone=False`` — the runner is
    the single source of truth for the live state.

    Conforms to the ``example`` interface expected by :func:`newton.examples.run` so it
    can be passed directly as the example object.
    """

    def __init__(
        self,
        setups: dict[str, SolverSetup],
        leader: str,
        viewer: ViewerBase | None = None,
        force_cb: Callable | None = None,
        fps: float = 50.0,
        sim_substeps: int = 20,
        verbose: bool = False,
    ):
        """Construct a runner around a dict of solver setups.

        Args:
            setups: Mapping of name to :class:`SolverSetup`. All setups must be
                non-standalone (``standalone=False``).
            leader: Key of the leader setup whose post-step output is treated as the
                canonical state for the next frame.
            viewer: Optional viewer. When ``None``, ``render()`` is a no-op and the
                runner skips the default ``viewer.apply_forces`` fallback.
            force_cb: Optional callable ``force_cb(state, contacts)`` applied to
                ``state_in`` at the start of every sub-step (after ``clear_forces``).
                When set, replaces the default ``viewer.apply_forces`` fallback.
            fps: Viewer/render frame rate. ``frame_dt = 1/fps`` is the per-call
                advance of ``self.sim_time``.
            sim_substeps: Number of physics sub-steps per ``step()`` call.
                ``sim_dt = frame_dt / sim_substeps`` is the physical integration
                step size. Each setup's solver should have been constructed with
                this same ``sim_dt``.
            verbose: When ``True``, sets ``setup.verbose = True`` on every setup so
                every sub-step dumps state / contact / metrics diagnostics. Noisy;
                pair with a low ``--num-frames`` value.
        """
        if leader not in setups:
            raise ValueError(f"leader {leader!r} not in setups: {list(setups)}")
        for name, setup in setups.items():
            if setup.standalone:
                raise ValueError(
                    f"setup {name!r} must be standalone=False when used inside SetupRunner; "
                    f"the runner owns state_in / control / contacts"
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

        self.fps: float = fps
        self.sim_substeps: int = sim_substeps
        self.frame_dt: float = 1.0 / fps
        self.sim_dt: float = self.frame_dt / sim_substeps
        self.sim_time: float = 0.0

        # Canonical state buffers, allocated from the leader's model
        self.model: Model = self.leader.model
        self.state_in: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.reset()

    def reset(self) -> None:
        """Re-initialise ``state_in`` by calling each setup's ``reset_cb``.

        Followers reset first, then the leader. This matters when a follower's
        ``reset_cb`` writes a base pose via forward kinematics (touching ``joint_q``
        and the derived ``body_q``/``body_qd``) and the leader's ``reset_cb``
        applies solver-internal initialisation on top — running the leader last
        prevents the follower from overwriting it.
        """
        for name, setup in self.setups.items():
            if name == self.leader_name:
                continue
            setup.reset(state_out=self.state_in)
        self.leader.reset(state_out=self.state_in)

    def step(self) -> None:
        """Run ``sim_substeps`` physics sub-steps and advance ``sim_time`` by ``frame_dt``.

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
            if self.force_cb is not None:
                self.force_cb(state=self.state_in, contacts=self.contacts)
            elif self.viewer is not None:
                self.viewer.apply_forces(self.state_in)

            self.model.collide(self.state_in, self.contacts)

            for setup in self.setups.values():
                setup.step(state_in=self.state_in, control_in=self.control, contacts_in=self.contacts)

            self.state_in, self.leader.state_out = self.leader.state_out, self.state_in

        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Log the canonical post-step state and pre-step contact geometry to the viewer.

        Uses ``self.leader.contacts`` (the leader setup's private contacts copy, with
        solver-written forces from ``update_contacts``) rather than ``self.contacts``
        (the runner's pre-collide-only buffer).
        """
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_in)
        self.viewer.log_contacts(self.leader.contacts, self.leader.state_out)
        self.viewer.end_frame()

    def test_final(self, problem_name: str = "comparison", output_path: str | None = None) -> None:
        """Write :meth:`SolutionMetricsLogger.plot_comparison` PDFs for the run.

        Always emits ``<problem_name>_solvers.pdf`` comparing the front-end
        :class:`SolutionMetricsNewton` of every setup. For any setup whose
        ``solver_logger`` is populated (i.e. the solver exposes an internal
        metrics object), also emits ``<problem_name>_<setup_name>.pdf``
        comparing that solver's internal metrics against its own front-end
        evaluator — a useful diagnostic for verifying the two paths agree.

        Args:
            problem_name: Used as both the filename stem and the default sub-directory
                under ``<this_file>/output/``.
            output_path: Optional override for the output directory. When ``None``,
                defaults to ``<this_file>/output/<problem_name>``.
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", problem_name)
        os.makedirs(output_path, exist_ok=True)

        front_end_loggers = {name: setup.logger for name, setup in self.setups.items()}
        SolutionMetricsLogger.plot_comparison(
            front_end_loggers, filename=f"{problem_name}_solvers", path=output_path, ext="pdf", grid=True
        )

        for name, setup in self.setups.items():
            if setup.solver_logger is None:
                continue
            internal_vs_front_end = {"solver": setup.solver_logger, name: setup.logger}
            SolutionMetricsLogger.plot_comparison(
                internal_vs_front_end,
                filename=f"{problem_name}_{name}",
                path=output_path,
                ext="pdf",
                grid=True,
            )
