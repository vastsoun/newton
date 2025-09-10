###########################################################################
# KAMINO: High-level Simulation Interface
###########################################################################

from __future__ import annotations

import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.bodies import update_body_inertias, update_body_wrenches
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.control import Control
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.state import State
from newton._src.solvers.kamino.core.time import advance_time
from newton._src.solvers.kamino.dynamics.dual import DualProblem
from newton._src.solvers.kamino.dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.detector import CollisionDetector
from newton._src.solvers.kamino.integrators.euler import integrate_semi_implicit_euler
from newton._src.solvers.kamino.kinematics.constraints import make_unilateral_constraints_info, update_constraints_info
from newton._src.solvers.kamino.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino.kinematics.joints import compute_joints_state
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.linalg.cholesky import SequentialCholeskyFactorizer
from newton._src.solvers.kamino.solvers.padmm import PADMMDualSolver

###
# Module interface
###

__all__ = ["Simulator"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Simulator interface
###


class SimulatorData:
    def __init__(self, model: Model, device: Devicelike = None):
        # First allocate the compact state and control containers for the previous and next steps
        # NOTE: The `next` state is to be understood as the current state, previous is always the past
        self.s_p: State = model.state(device=device)
        self.s_n: State = model.state(device=device)
        self.c_p: Control = model.control(device=device)
        self.c_n: Control = model.control(device=device)

        # Then allocate the extended state container with all necessary internal data
        # NOTE: We are skipping the bodies state as we will capture it by reference to the next state
        self.state: ModelData = model.data(device=device, skip_body_dofs=False)

        # # Finally, initialize the references to the bodies state in the extended state
        # self.state.bodies.q_i = self.s_n.q_i
        # self.state.bodies.u_i = self.s_n.u_i

    def flip(self):
        """
        Flip the current and previous states references.
        """
        # Swap the previous and next states and controls
        self.s_p, self.s_n = self.s_n, self.s_p
        self.c_p, self.c_n = self.c_n, self.c_p
        # Update the bodies state references
        self.state.bodies.q_i = self.s_n.q_i
        self.state.bodies.u_i = self.s_n.u_i

    def forward(self):
        """
        Copies the next state to the previous and model states
        """
        wp.copy(self.state.bodies.q_i, self.s_n.q_i)
        wp.copy(self.state.bodies.u_i, self.s_n.u_i)

    def cache(self):
        """
        Copies the next state to the previous and model states
        """
        wp.copy(self.s_p.q_i, self.s_n.q_i)
        wp.copy(self.s_p.u_i, self.s_n.u_i)


class Simulator:
    def __init__(self, builder: ModelBuilder, dt: float = 0.001, device: Devicelike = None, shadow: bool = False):
        # Host-side time-keeping
        self._time: float = 0.0
        self._max_time: float = 0.0
        self._steps: int = 0
        self._max_steps: int = 0

        # Ensure the time-step is positive
        if dt <= 0.0:
            raise ValueError(f"Invalid time-step: {dt}. Must be a positive value.")

        # Cache the time-step use for the simulation
        self._dt: float = dt

        # Cache the target device use for the simulation
        self._device: Devicelike = device

        # Joint Limits
        self._limits = Limits(builder=builder, device=self._device)

        # Collision Detection
        self._collision_detector = CollisionDetector(builder=builder, device=self._device)

        # Model
        self._model = builder.finalize(device=self._device)

        # Configure model time-steps
        self._model.time.set_timestep(self._dt)

        # Allocate system data on the device
        self._data = SimulatorData(model=self._model, device=self._device)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(
            model=self._model, state=self._data.state, limits=self._limits, contacts=self.contacts, device=self._device
        )

        # Allocate Jacobians data on the device
        self._jacobians = DenseSystemJacobians(
            model=self._model,
            limits=self._limits,
            contacts=self._collision_detector.contacts,
            device=self._device,
        )

        # Allocate the dual problem data on the device
        # TODO: Make the factorizer configurable
        self._dual_problem = DualProblem(
            model=self._model,
            state=self._data.state,
            limits=self._limits,
            contacts=self._collision_detector.contacts,
            factorizer=SequentialCholeskyFactorizer,  # TODO: Make this configurable
            # TODO: settings=None,
            device=self._device,
        )

        # Allocate the dual solver data on the device
        # TODO: Make the solver parameters configurable
        self._dual_solver = PADMMDualSolver(
            model=self._model,
            state=self._data.state,
            limits=self._limits,
            contacts=self._collision_detector.contacts,
            collect_info=True,  # TODO: Make this configurable
            device=self._device,
        )

        # Initialize callbacks
        self._control_cb = None
        self._pre_step_cb = None
        self._mid_step_cb = None
        self._post_step_cb = None

        # Define optional data shadowing  on the CPU
        self._host: SimulatorData | None = None
        if shadow:
            self.sync_host()

        # Initialize the simulation state
        with wp.ScopedDevice(self._device):
            self.reset()

    ###
    # Properties
    ###

    @property
    def time(self) -> float:
        return self._time

    @property
    def max_time(self) -> float:
        return self._max_time

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def model(self) -> Model:
        return self._model

    @property
    def data(self) -> SimulatorData:
        return self._data

    @property
    def model_data(self) -> ModelData:
        return self._data.state

    @property
    def state_previous(self) -> State:
        return self._data.s_p

    @property
    def state(self) -> State:
        return self._data.s_n

    @property
    def control_previous(self) -> Control:
        return self._data.c_p

    @property
    def control(self) -> Control:
        return self._data.c_n

    @property
    def limits(self) -> Limits:
        return self._limits

    @property
    def contacts(self) -> Contacts:
        return self._collision_detector.contacts

    @property
    def collision_detector(self) -> CollisionDetector:
        return self._collision_detector

    @property
    def jacobians(self) -> DenseSystemJacobians:
        return self._jacobians

    @property
    def problem(self) -> DualProblem:
        return self._dual_problem

    @property
    def solver(self) -> PADMMDualSolver:
        return self._dual_solver

    @property
    def host(self) -> SimulatorData | None:
        # return self._host
        return self._data

    ###
    # Callbacks
    ###

    def set_control_callback(self, callback):
        """
        Set a callback to be called at the beggining of the step.
        """
        self._control_cb = callback

    def set_pre_step_callback(self, callback):
        """
        Set a callback to be called before forward dynamics solve.
        """
        self._pre_step_cb = callback

    def set_mid_step_callback(self, callback):
        """
        Set a callback to be called between forward dynamics solver and state integration.
        """
        self._mid_step_cb = callback

    def set_post_step_callback(self, callback):
        """
        Set a callback to be called after state integration.
        """
        self._post_step_cb = callback

    ###
    # Implementatin-specific Operations
    ###

    def _reset_time(self):
        self._time = 0.0
        self._steps = 0
        self._data.state.time.zero()

    def _reset_bodies_state(self):
        # Copy the initial state defined in the model into the previous and next state buffers
        wp.copy(self._data.s_p.q_i, self._model.bodies.q_i_0)
        wp.copy(self._data.s_p.u_i, self._model.bodies.u_i_0)
        wp.copy(self._data.s_n.q_i, self._model.bodies.q_i_0)
        wp.copy(self._data.s_n.u_i, self._model.bodies.u_i_0)

        # Update the model state references to the next state
        self._data.state.bodies.q_i = self._data.s_n.q_i
        self._data.state.bodies.u_i = self._data.s_n.u_i

    def _reset_bodies_wrenches(self):
        self._data.state.bodies.w_i.zero_()
        self._data.state.bodies.w_a_i.zero_()
        self._data.state.bodies.w_j_i.zero_()
        self._data.state.bodies.w_l_i.zero_()
        self._data.state.bodies.w_c_i.zero_()
        self._data.state.bodies.w_e_i.zero_()

    def _reset_joints_state(self):
        compute_joints_state(self._model, self._data.state)

    def _reset_joints_wrenches(self):
        self._data.state.joints.lambda_j.zero_()
        self._data.state.joints.tau_j.zero_()
        self._data.state.joints.j_w_j.zero_()
        self._data.state.joints.j_w_a_j.zero_()
        self._data.state.joints.j_w_l_j.zero_()
        self._data.state.joints.j_w_c_j.zero_()
        self._data.s_p.lambda_j.zero_()
        self._data.s_n.lambda_j.zero_()
        self._data.c_p.tau_j.zero_()
        self._data.c_n.tau_j.zero_()

    def _run_control_callback(self):
        """
        Run the control callback if it has been set.
        """
        if self._control_cb is not None:
            self._control_cb(self)

    def _run_presetp_callback(self):
        """
        Run the pre-step callback if it has been set.
        """
        if self._pre_step_cb is not None:
            self._pre_step_cb(self)

    def _run_midsetp_callback(self):
        """
        Run the mid-step callback if it has been set.
        """
        if self._mid_step_cb is not None:
            self._mid_step_cb(self)

    def _run_poststep_callback(self):
        """
        Executes the post-step callback if it has been set.
        """
        if self._post_step_cb is not None:
            self._post_step_cb(self)

    def _update_actuation_wrences(self):
        # Clear the previous actuation wrenches
        self._data.state.bodies.w_a_i.zero_()

        # TODO: We need to decide wether to compute these directly or first initialize the joint DoF Jacobians
        # Compute the actuation wrenches from the control inputs (taus)
        compute_joint_dof_body_wrenches(self._model, self._data.state, self._jacobians.data)

    def _clear_constraint_wrenches(self):
        # TODO: How to cache these to be used for a systematic warm-start of the constraint solver?
        self._data.state.bodies.w_j_i.zero_()
        self._data.state.bodies.w_l_i.zero_()
        self._data.state.bodies.w_c_i.zero_()

    def _check_limits(self):
        self._limits.detect(self._model, self._data.state)

    def _collide(self):
        self._collision_detector.collide(self._model, self._data.state)

    def _update_constraint_info(self):
        update_constraints_info(model=self._model, state=self._data.state)

    def _forward_intermediate(self):
        update_body_inertias(self._model.bodies, self._data.state.bodies)
        compute_joints_state(self._model, self._data.state)

    def _forward_kinematics(self):
        # Build actuation and constraint Jacobians
        self._jacobians.build(
            model=self._model,
            state=self._data.state,
            limits=self._limits.data,
            contacts=self.contacts.data,
            reset_to_zero=True,
        )

    def _forward_dynamics(self):
        # Construct the dual problem from the current model state and contacts
        self._dual_problem.build(
            model=self._model,
            state=self._data.state,
            limits=self._limits.data,
            contacts=self.contacts.data,
            jacobians=self.jacobians.data,
            reset_to_zero=True,
        )

    def _forward_constraints(self):
        # Solve the dual problem to compute the constraint reactions
        self._dual_solver.solve(problem=self._dual_problem)

        # Unpack the constraint reaction multipliers into body wrenches
        compute_constraint_body_wrenches(
            model=self._model,
            state=self._data.state,
            limits=self._limits.data,
            contacts=self.contacts.data,
            jacobians=self._jacobians.data,
            lambdas_offsets=self._dual_problem.data.vio,
            lambdas_data=self._dual_solver.data.solution.lambdas,
        )

    def _forward_wrenches(self):
        update_body_wrenches(self._model.bodies, self._data.state.bodies)

    def _forward(self):
        # Update intermediate quantities
        self._forward_intermediate()

        # Update the kinematics
        self._forward_kinematics()

        # Update the dynamics
        self._forward_dynamics()

        # Compute constraint reactions
        self._forward_constraints()

        # Post-processing
        self._forward_wrenches()

    def _integrate(self):
        # Integrate the bodies state and update the next state buffer
        integrate_semi_implicit_euler(self._model, self._data.state, self._data.s_n)

        # # Copy the integrated state to the previous and current body states
        self._data.forward()

        # TODO: How can buffer flipping work with CUDA graph capture?
        # # First swap the references to the previous and next state buffers
        # self._data.s_p, self._data.s_n = self._data.s_n, self._data.s_p
        # self._data.c_p, self._data.c_n = self._data.c_n, self._data.c_p
        # # Then integrate the bodies state and update the next state buffer
        # integrate_semi_implicit_euler(self._model, self._data.state, self._data.s_n)
        # # Then update the model state references to latest next state
        # self._data.state.bodies.q_i = self._data.s_n.q_i
        # self._data.state.bodies.u_i = self._data.s_n.u_i

    def _advance_time(self):
        self._steps += 1
        self._time += self._dt
        advance_time(self._model.time, self._data.state.time)

    ###
    # Front-end Operations
    ###

    def reset(self):
        # Reset the time and step count
        self._reset_time()

        # First reset the states of all bodies
        self._reset_bodies_state()
        self._reset_bodies_wrenches()

        # Reset the state of all joints
        self._reset_joints_state()
        self._reset_joints_wrenches()

        # Update the kinematics
        # NOTE: This constructs the system Jacobians, which ensures
        # that control action will be applied before the first step
        self._forward_kinematics()

    def step(self):
        # Copy the integrated state to the previous and current body states
        self._data.cache()

        # Run the control callback if it has been set
        self._run_control_callback()

        # Apply actuation forces to the bodies
        self._update_actuation_wrences()

        # Run limit detection to generate active joint limits
        self._check_limits()

        # Run collision detection to generate for active contacts
        self._collide()

        # Update the constraint state info
        self._update_constraint_info()

        # Clear all mutable constraint wrenches
        self._clear_constraint_wrenches()

        # Run the pre-step callback if it has been set
        self._run_presetp_callback()

        # Compute forward dynamics
        self._forward()

        # Run the mid-step callback if it has been set
        self._run_midsetp_callback()

        # Integrate the state
        self._integrate()

        # Run the post-step callback if it has been set
        self._run_poststep_callback()

        # Update time-keeping
        self._advance_time()

        # Post-processing: update the rigid-body system base and DoF states
        # NOTE: If no base and actuation are present, these calls should do nothing
        # self._update_base()
        # self._update_joints()
        # self._update_metrics()

    def sync_host(self):
        """
        Updates the host-side data with the in-device data.
        """
        # Construct the host data if it does not exist
        if self._host is None:
            self._host = SimulatorData(model=self._model, device="cpu")
        # Update the host data from the device data
        # TODO: Implement the host data update
        # self._host.state = self._data.state
