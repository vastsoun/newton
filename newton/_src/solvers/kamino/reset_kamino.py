# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Defines the :class:`ResetKamino` class, providing a resetting backend for
simulating of constrained multi-body systems for arbitrary mechanical assemblies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import warp as wp

# Kamino imports
from .core.bodies import update_body_inertias
from .core.state import State
from .kinematics.joints import compute_joints_data
from .kinematics.resets import reset_select_worlds_to_initial_state, reset_select_worlds_to_state
from .solver_kamino import SolverKamino
from .solvers.fk import ForwardKinematicsSolver, ForwardKinematicsSolverSettings

###
# Types
###


@dataclass
class ResetKaminoSettings:
    """
    A container to hold configurations for :class:`ResetKamino`.
    """

    solver: ForwardKinematicsSolverSettings = field(default_factory=ForwardKinematicsSolverSettings)
    """Settings for the maximal-coordinate forward kinematics solver of Kamino."""

    def check(self) -> None:
        """Validates relevant solver settings."""
        pass

    def __post_init__(self):
        """Post-initialization to validate settings."""
        self.check()


###
# Interfaces
###


class ResetKamino:
    """
    TODO
    """

    def __init__(
        self,
        solver: SolverKamino,
        settings: ResetKaminoSettings | None = None,
    ):
        """
        Initializes the Kamino physics solver for the given set of multi-body systems
        defined in `model`, and the total contact allocations defined in `contacts`.

        Explicit solver settings may be provided through the `settings` argument. If no
        settings are provided, default settings will be used.

        Args:
            model (Model): The multi-body systems model to simulate.
            contacts (Contacts): The contact data container for the simulation.
            settings (SolverKaminoSettings | None): Optional solver settings.
        """
        # Cache references to the model and data of Kamino
        self._solver_sim: SolverKamino = solver

        # Cache solver settings: If no settings are provided, use defaults
        if settings is None:
            settings = ResetKaminoSettings()
        settings.check()
        self._settings: ResetKaminoSettings = settings

        # Allocate the forward dynamics solver on the device
        self._solver_fk = ForwardKinematicsSolver(model=self._solver_sim._model, settings=self._settings.solver)

        # Initialize callbacks
        self._pre_reset_cb: Callable[[ResetKamino], None] = None
        self._post_reset_cb: Callable[[ResetKamino], None] = None

    ###
    # Properties
    ###

    @property
    def solver_fk(self) -> ForwardKinematicsSolver:
        """
        Returns the forward kinematics solver.
        """
        return self._solver_fk

    ###
    # Configurations - Callbacks
    ###

    def set_pre_reset_callback(self, callback: Callable[[ResetKamino], None]):
        """
        Set a reset callback to be called at the beginning of each call to `reset_*()` methods.
        """
        self._pre_reset_cb = callback

    def set_post_reset_callback(self, callback: Callable[[ResetKamino], None]):
        """
        Set a reset callback to be called at the end of each call to to `reset_*()` methods.
        """
        self._post_reset_cb = callback

    ###
    # Reset API
    ###

    def reset(self, world_mask: wp.array | None = None, reset_constraints: bool = True):
        """
        Resets the simulation to the initial state defined in the model.

        This operation also accepts an optional per-world mask that indicates
        which worlds should be reset as well as a flag to indicate whether
        joint constraint reactions should be reset to zero.

        Args:
            world_mask (wp.array): Array of per-world masks that indicate which worlds should be reset.\n
            reset_constraints (bool): Whether to reset joint constraint reactions to zero.
        """
        # Run the pre-reset callback if it has been set
        self._run_pre_reset_callback()

        # Reset (optionally select) worlds to the initial state of the model
        if world_mask is None:
            self._reset_all_worlds_to_initial_state(reset_constraints)
        else:
            self._reset_select_worlds_to_initial_state(world_mask, reset_constraints)

        # Run the common post-reset processing operations
        self._reset_post_process()

        # Run the post-reset callback if it has been set
        self._run_post_reset_callback()

    def reset_to_state(self, state: State, world_mask: wp.array | None = None, reset_constraints: bool = True):
        """
        Resets the simulation to a fully specified maximal-coordinate state.

        This operation therefore only uses the provided body poses and twists to reset the simulation,
        while the joint states are subsequently computed as part of a forward kinematics update.

        Args:
            state (State): The state container from which the body states will be used to reset the simulation.
            world_mask (wp.array): Array of per-world flags that indicate which worlds should be reset.\n
                For each element 'w' in the array, if 'world_mask[w] != 0' then world 'w' will be reset,
                otherwise it will be left unchanged (i.e. skipped).
            reset_constraints (bool): If True, also copies joint constraint forces
                from the provided state in order to warm-start the constraint solver.
        """
        # Run the pre-reset callback if it has been set
        self._run_pre_reset_callback()

        # Reset (optionally select) worlds to the specified state
        if world_mask is None:
            self._reset_all_worlds_to_state(state, reset_constraints)
        else:
            self._reset_select_worlds_to_state(state, world_mask, reset_constraints)

        # Run the common post-reset processing operations
        self._reset_post_process()

        # Run the post-reset callback if it has been set
        self._run_post_reset_callback()

    def reset_to_actuators_state(
        self, actuators_q: wp.array, actuators_dq: wp.array, world_mask: wp.array | None = None
    ):
        """
        Resets the simulation to a specified state of the actuators (i.e. generalized coordinates and velocities).

        This operation serves as reduced-coordinate-like interface to reset the simulation based on
        the states of only the actuated joints. It computes the corresponding body poses and twists
        of all bodies, as well as the states of passive joints, using an iterative forward kinematics
        solver. The resulting body poses and twists are then used to reset the state of the simulation.

        This method is mostly intended for fixed-base systems, where the base body is connected to the
        world via a unary joint. It can still be useful for floating-base systems, however, if only the
        actuated joints need to be reset. In cases were the base body also needs to be reset, please use
        the `reset_to_base_and_actuators_state()` method instead.

        Args:
            actuators_q (wp.array): Array of actuated joint coordinates.\n
                Expects shape of ``(sum_of_num_actuated_joint_coords,)`` and type :class:`float`.
            actuators_dq (wp.array): Array of actuated joint velocities.\n
                Expects shape of ``(sum_of_num_actuated_joint_dofs,)`` and type :class:`float`.
            world_mask (wp.array): Array of per-world flags that indicate which worlds should be reset.\n
                For each element 'w' in the array, if 'world_mask[w] != 0' then world 'w' will be reset,
                otherwise it will be left unchanged (i.e. skipped).
        """
        # TODO:
        # - Reset time of select worlds --> parallel over worlds
        # - Reset all bodies of select worlds according to the delta transform
        #   between current and target base pose --> parallel over bodies
        #   --> TODO: Do we need to offset all bodies?
        #   --> TODO: Can we just offset the model parameters instead? If yes, which?
        # - Use the ForwardKinematics solver given specified joint coordinates and velocities to compute body states
        # - Compute joint states from body states, parallel over joints
        raise NotImplementedError("Simulator.reset_to_actuators_state() is not yet implemented.")

        # Run the pre-reset callback if it has been set
        self._run_pre_reset_callback()

        # Reset (optionally select) worlds to the specified actuators state
        # TODO

        # Run the common post-reset processing operations
        self._reset_post_process()

        # Run the post-reset callback if it has been set
        self._run_post_reset_callback()

    def reset_to_base_and_actuators_state(
        self,
        base_q: wp.array,
        base_u: wp.array,
        actuators_q: wp.array,
        actuators_dq: wp.array,
        world_mask: wp.array = None,
    ):
        """
        Resets the simulation to a specified state of  base and actuators (i.e. generalized coordinates and velocities).

        This operation serves as reduced-coordinate-like interface to reset the simulation based on
        the states of only the base body and the actuated joints. It computes the corresponding body
        poses and twists of all other bodies, as well as the states of passive joints, using an
        iterative forward kinematics solver. The resulting body poses and twists are then used to
        reset the state of the simulation.

        This method is mostly intended for floating-base systems, where no joint is defined between
        the world and the base body, or if the latter has been assigned a 6-DoF `FREE` joint. It can
        still be useful for fixed-base systems, in cases where it is desired to reset the system to an
        alternate fixture configuration. In this case, however, the `base_u` input will be ignored. If
        the base body does need to be reset, please use the `reset_to_actuators_state()` method instead.

        Args:
            base_q (wp.array): Array of base body poses.\n
                Expects shape of ``(num_worlds,)`` and type :class:`transform`.
            base_u (wp.array): Array of base body twists.\n
                Expects shape of ``(num_worlds,)`` and type :class:`vec6`.
            actuators_q (wp.array): Array of actuated joint coordinates.\n
                Expects shape of ``(sum_of_num_actuated_joint_coords,)`` and type :class:`float`.
            actuators_dq (wp.array): Array of actuated joint velocities.\n
                Expects shape of ``(sum_of_num_actuated_joint_dofs,)`` and type :class:`float`.
            world_mask (wp.array): Array of per-world flags that indicate which worlds should be reset.\n
                For each element 'w' in the array, if 'world_mask[w] != 0' then world 'w' will be reset,
                otherwise it will be left unchanged (i.e. skipped).
        """
        # TODO:
        # - Reset time of select worlds --> parallel over worlds
        # - Reset all bodies of select worlds according to the delta transform
        #   between current and target base pose --> parallel over bodies
        #   --> TODO: Do we need to offset all bodies?
        #   --> TODO: Can we just offset the model parameters instead? If yes, which?
        # - Use the ForwardKinematics solver given specified joint coordinates and velocities to compute body states
        # - Compute joint states from body states, parallel over joints
        raise NotImplementedError("Simulator.reset_to_base_and_actuators_state() is not yet implemented.")

        # Run the pre-reset callback if it has been set
        self._run_pre_reset_callback()

        # Reset (optionally select) worlds to the specified base and actuators state
        # TODO

        # Run the common post-reset processing operations
        self._reset_post_process()

        # Run the post-reset callback if it has been set
        self._run_post_reset_callback()

    ###
    # Internals - Callback Operations
    ###

    def _run_pre_reset_callback(self):
        """
        Runs the pre-reset callback if it has been set.
        """
        if self._pre_reset_cb is not None:
            self._pre_reset_cb(self)

    def _run_post_reset_callback(self):
        """
        Runs the post-reset callback if it has been set.
        """
        if self._post_reset_cb is not None:
            self._post_reset_cb(self)

    ###
    # Internals - Reset Operations
    ###

    def _reset_time_of_all_worlds(self):
        """
        Resets the time and step count of the simulation.
        """
        self._solver_sim._data.time.zero()

    def _reset_bodies_to_model_initial_state(self):
        """
        Resets the state of all bodies to the initial states defined in the model.
        """
        wp.copy(self._solver_sim._data.bodies.q_i, self._solver_sim._model.bodies.q_i_0)
        wp.copy(self._solver_sim._data.bodies.u_i, self._solver_sim._model.bodies.u_i_0)

    def _reset_bodies_data(self):
        """
        Resets all internal solver data of bodies from the current reset state.

        This includes updating the body inertias from the body states, and clearing all body wrenches.
        """
        # Update the in-world-frame body inertias from the body states
        update_body_inertias(model=self._solver_sim._model.bodies, data=self._solver_sim._data.bodies)

        # Clear all body wrenches by setting them to zero
        self._solver_sim._data.bodies.clear_all_wrenches()

    def _reset_bodies_state_and_data_to_initial(self):
        """
        Resets the solver internal data of all bodies to the initial states defined in the model.
        """
        # First set the active body states to the initial states defined in the model
        self._reset_bodies_to_model_initial_state()

        # Then reset all internal body data (i.e. inertias, wrenches etc)
        self._reset_bodies_data()

    def _reset_joints_data(self, reset_constraints: bool = True):
        """
        Resets all internal solver data of joints from the current reset state.

        This includes updating the joint state from the body states,
        and clearing all joint constraints, actuation and wrenches.
        """
        # Then compute the initial joint states based on the body states
        compute_joints_data(
            model=self._solver_sim._model,
            q_j_ref=self._solver_sim._model.joints.q_j_ref,
            data=self._solver_sim._data,
            correction=self._solver_sim._settings.rotation_correction,
        )

        # Finally, clear all joint constraint reactions,
        # actuation forces, and wrenches, setting them to zero
        if reset_constraints:
            self._solver_sim._data.joints.clear_constraint_reactions()
        self._solver_sim._data.joints.clear_actuation_forces()
        self._solver_sim._data.joints.clear_wrenches()

    def _reset_post_process(self):
        """
        Resets solver internal data and calls reset callbacks.

        This is a common operation that must be called after resetting bodies and joints,
        that ensures that all state and control data are synchronized with the internal
        solver state, and that intermediate quantities are updated accordingly.
        """
        # First clear the next-step control inputs so they correctly propagate to the previous-step
        self._solver_sim._data.joints.tau_j.zero_()

        # TODO: World masking for selective resets
        # Reset the forward dynamics solver to clear internal state
        # NOTE: This will cause the solver to perform a cold-start
        # on the first call to `step()`
        self._solver_sim._solver_fd.reset()

    def _reset_all_worlds_to_initial_state(self, reset_constraints: bool = True):
        """
        Resets the simulation to the initial state defined in the model.
        """
        # Reset the time and step count
        self._reset_time_of_all_worlds()

        # First reset the states of all bodies
        self._reset_bodies_state_and_data_to_initial()

        # Then reset the state of all joints
        self._reset_joints_data(reset_constraints)

    def _reset_select_worlds_to_initial_state(self, world_mask: wp.array | None = None, reset_constraints: bool = True):
        """
        Resets the simulation to the initial state defined in the model.

        Args:
            world_mask (wp.array): Array of per-world masks that indicate which worlds should be reset.\n
            reset_constraints (bool): Whether to reset joint constraint reactions to zero.
        """
        # Reset the worlds specified in the `world_mask` array to the given state
        reset_select_worlds_to_initial_state(
            model=self._solver_sim._model,
            data=self._solver_sim._data,
            mask=world_mask,
            reset_constraints=reset_constraints,
        )

    def _reset_all_worlds_to_state(self, state: State, reset_constraints: bool = True):
        """
        Resets all worlds of the simulation to a fully specified state.

        Args:
            state (State): The state container from which the body states will be used to reset the simulation.
            reset_constraints (bool): If True, also copies joint constraint forces
                from the provided state in order to warm-start the constraint solver.
        """
        # Reset the time and step count
        self._reset_time_of_all_worlds()

        # Copy the specified state into the internal solver state for bodies
        wp.copy(self._solver_sim._data.bodies.q_i, state.q_i)
        wp.copy(self._solver_sim._data.bodies.u_i, state.u_i)

        # Then reset all internal body data (i.e. inertias, wrenches etc)
        self._reset_bodies_data()

        # Then reset the state of all joints
        self._reset_joints_data(reset_constraints=reset_constraints)

        # Optionally also copy joint constraint forces
        # NOTE: Used to warm-start the constraint solver
        if not reset_constraints:
            wp.copy(self._solver_sim._data.joints.lambda_j, state.lambda_j)

    def _reset_select_worlds_to_state(
        self, state: State, world_mask: wp.array | None = None, reset_constraints: bool = True
    ):
        """
        Resets the simulation to a specific state.

        Args:
            state (State): The state container from which the body states will be used to reset the simulation.
            world_mask (wp.array): Array of per-world masks that indicate which worlds should be reset.\n
                For each element 'w' in the array, if 'world_mask[w] != 0' then world 'w' will be reset,
                otherwise it will be left unchanged (i.e. skipped).
            reset_constraints (bool): If True, also copies joint constraint forces
                from the provided state in order to warm-start the constraint solver.
        """

        # Reset the worlds specified in the `world_mask` array to the given state
        reset_select_worlds_to_state(
            model=self._solver_sim._model,
            data=self._solver_sim._data,
            state=state,
            mask=world_mask,
            reset_constraints=reset_constraints,
        )
