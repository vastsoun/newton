# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

from collections.abc import Callable

import warp as wp

from ....viewer import ViewerBase
from ....solvers import SolverBase
from ....sim import Model, ModelBuilder, State, Control, Contacts
from .._src.metrics import SolutionMetricsLogger, SolutionMetricsNewton

###
# Module interface
###

__all__ = ["SolverSetup"]


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
        self.reset_cb: Callable | None = reset_cb
        self.control_cb: Callable | None = control_cb

        # Derived attributes
        self.metrics: SolutionMetricsNewton | None = None
        self.logger: SolutionMetricsLogger | None = None
        self.state_out: State | None = None
        self.state_in: State | None = None
        self.control: Control | None = None
        self.contacts: Contacts | None = None

        # TODO Add check that the required extended attritubes are present in the builder

        # Ensure the model is configured for the expected rigid-contact capacity
        # and allocate the relevant state, control and contacts containers.
        self.model.rigid_contact_max = rigid_contact_max

        # Allocate the necessary step data containers depending on
        # whether the solver will operate independently or not.
        self.state_out = self.model.state()
        if self.standalone:
            self.state_in = self.model.state()
            self.control = self.model.control()
            self.contacts = self.model.contacts()

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

    def step(
        self,
        state_in: State | None = None,
        control_in: Control | None = None,
        contacts_in: Contacts | None = None,
    ):
        """TODO"""
        # If the setup is standalone, use the internal state, control and contacts containers
        if self.standalone:
            state_in = self.state_in
            control_in = self.control
            contacts_in = self.contacts
        # Otherwise, ensure required inputs are valid
        else:
            if state_in is None or not isinstance(state_in, State):
                raise ValueError("state_in must be a State object")
            if control_in is None or not isinstance(control_in, Control):
                raise ValueError("control_in must be a Control object")
            if contacts_in is None or not isinstance(contacts_in, Contacts):
                raise ValueError("contacts_in must be a Contacts object")

        # If the setup is standalone, the query the actuation callback to compute
        # control inputs and then run collisiond detection to generate contacts
        # NOTE: When not standaline these will be called outside of the step op.
        if self.standalone:
            self.actuate(state_in=state_in, control_out=control_in)
            self.model.collide(state_in, contacts_in)

        # Step the solver to compute the next state. The result is always stored
        # in`self.state_out` regardless of whether the setup is standalone or not.
        self.solver.step(
            state_in=state_in,
            state_out=self.state_out,
            control=control_in,
            contacts=contacts_in,
            dt=self.dt,
        )

        # Update the contact forces in the contacts container from the solver
        self.solver.update_contacts(contacts=contacts_in, state=state_in)

        # Then evaluate the metrics on the current state transition
        self.metrics.evaluate(
            state=self.state_out,
            state_p=state_in,
            control=control_in,
            contacts=contacts_in,
        )

        # Log the new metrics for this step with the logger
        self.logger.log()

        # If the setup is standalone, swap the input
        # and output states to progress the simulation.
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
    """TODO"""

    def __init__(
        self,
        # Required inputs
        setups: dict[str, SolverSetup],
        leader: str | None = None,
        # Optional inputs
        fps: float = 50,
        dt: float = 0.001,
        num_frames: int = 5000,
        device: wp.DeviceLike = None,
        viewer: ViewerBase | None = None,
    ):
        """TODO"""
        # Cache run-time configurations
        self.fps = fps
        self.dt = dt
        self.num_frames = num_frames
        self.device = device
        self.viewer = viewer

        # Declare internal time-keeping variables
        self.time: float = 0.0
        self.steps: int = 0

        # Cache the setups
        self.setups: dict[str, SolverSetup] = setups

        # Set the leading solver and assign it to the viewer viewer if provided
        self.leader: SolverSetup | None = None
        self.followers: list[SolverSetup] | None = None
        if leader is not None:
            if leader not in setups.keys():
                raise ValueError(f"Leader solver {leader} not found in setups: {setups.keys()}")
            self.leader = setups[leader]
            if viewer is not None:
                viewer.set_model(self.leader.model)
            self.followers = [s for s in setups.values() if s.name != leader]

    def step(self):
        """TODO"""
        # TODO
        if self.leader is not None:
            # Clear and apply forces to the leading solver state
            self.leader.state_in.clear_forces()
            self.leader.state_out.clear_forces()
            self.viewer.apply_forces(self.leader.state_in)

            # Detect collisions
            self.leader.model.collide(self.leader.state_in, self.leader.contacts)

            # Step all solvers on the leading solver initial state
            for solver in self.setups.values():
                solver.step(state_in=self.leader.state_in, control_in=self.leader.control, contacts_in=self.leader.contacts)

            # Swap the leading solver state
            self.leader.state_out, self.leader.state_in = self.leader.state_in, self.leader.state_out

        # TODO: If not leader, step all solvers
        else:
            for solver in self.setups.values():
                solver.step()


    def render(self):
        """TODO"""
        self.viewer.begin_frame(self.time)
        if self.leader is not None:
            self.leader.render(self.viewer)
        self.viewer.end_frame()
