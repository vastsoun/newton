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
Defines the :class:`SolverKamino` class, providing a physics backend for
simulating constrained multi-body systems for arbitrary mechanical assemblies.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import warp as wp

from ...core.types import override
from ...sim import (
    Contacts,
    Control,
    JointType,
    Model,
    ModelBuilder,
    State,
)
from ..flags import SolverNotifyFlags
from ..solver import SolverBase

if TYPE_CHECKING:
    from ._src.geometry.detector import CollisionDetector
    from ._src.solver_kamino_impl import SolverKaminoImpl

###
# Module interface
###

__all__ = [
    "SolverKamino",
]


###
# Interfaces
###


class SolverKamino(SolverBase):
    """
    A physics solver for simulating constrained multi-body systems for arbitrary mechanical assemblies.

    This solver uses the Proximal-ADMM algorithm to solve the forward dynamics formulated
    as a Nonlinear Complementarity Problem (NCP) over the set of bilateral kinematic joint
    constraints and unilateral constraints that include joint-limits and contacts.

    References:
        - Tsounis, Vassilios, Ruben Grandia, and Moritz Bächer.
          On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops.
          arXiv preprint arXiv:2504.19771 (2025).
          https://doi.org/10.48550/arXiv.2504.19771
        - Carpentier, Justin, Quentin Le Lidec, and Louis Montaut.
          From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach.
          20th edition of the “Robotics: Science and Systems”(RSS) Conference. 2024.
          https://roboticsproceedings.org/rss20/p108.pdf
        - Tasora, A., Mangoni, D., Benatti, S., & Garziera, R. (2021).
          Solving variational inequalities and cone complementarity problems in
          nonsmooth dynamics using the alternating direction method of multipliers.
          International Journal for Numerical Methods in Engineering, 122(16), 4093-4113.
          https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6693

    After constructing :class:`ModelKamino`, :class:`StateKamino`, :class:`ControlKamino` and :class:`ContactsKamino`
    objects, this physics solver may be used to advance the simulation state forward in time.

    Example
    -------

    .. code-block:: python

        contacts = ...
        config = newton.solvers.SolverKamino.Config()
        solver = newton.solvers.SolverKamino(model, contacts, config)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    # Class variables to cache the imported module
    _kamino = None

    # Placeholder for the SolverKamino.Config class
    # which is defined in _src/solver_kamino_impl.py
    Config: type[SolverKaminoImpl.Config] | None = None

    def __init__(
        self,
        model: Model,
        solver_config: SolverKaminoImpl.Config | None = None,
        collision_detector_config: CollisionDetector.Config | None = None,
    ):
        """
        Constructs a Kamino solver for the given model and optional configurations.

        Args:
            model: The Newton model to simulate.
            solver_config: Configuration for the Kamino solver.
                If ``None``, a default configuration is created by parsing the Newton model's
                custom attributes (e.g. from USD) using :meth:`SolverKamino.Config.from_model`.
            collision_detector_config: Configuration for the internal collision detector.
                If ``None``, the default configuration is used.
        """
        # Initialize the base solver
        super().__init__(model=model)

        # Import all Kamino dependencies and cache them
        # as class variables if not already done
        self._import_kamino()

        # Validate that the model does not contain unsupported components
        self._validate_model_compatibility(model)

        # Create a Kamino model from the Newton model
        self._model_kamino = self._kamino.ModelKamino.from_newton(model)

        # Create a collision detector
        self._collision_detector_kamino = self._kamino.CollisionDetector(
            model=self._model_kamino,
            config=collision_detector_config,
        )

        # Capture a reference to the contacts container
        self._contacts_kamino = self._collision_detector_kamino.contacts

        # Create solver config if none is provided. This will also parse any solver-specific
        # attributes imported from USD.
        if solver_config is None:
            solver_config = self.Config.from_model(model)

        # Initialize the internal Kamino solver
        self._solver_kamino = self._kamino.SolverKaminoImpl(
            model=self._model_kamino,
            contacts=self._contacts_kamino,
            config=solver_config,
        )

    def reset(
        self,
        state_out: State,
        world_mask: wp.array | None = None,
        actuator_q: wp.array | None = None,
        actuator_u: wp.array | None = None,
        joint_q: wp.array | None = None,
        joint_u: wp.array | None = None,
        base_q: wp.array | None = None,
        base_u: wp.array | None = None,
    ):
        """
        Resets the simulation state given a combination of desired base body
        and joint states, as well as an optional per-world mask array indicating
        which worlds should be reset. The reset state is written to `state_out`.

        For resets given absolute quantities like base body poses, the
        `state_out` must initially contain the current state of the simulation.

        Args:
            state_out: The output state container to which the reset state data is written.
            world_mask: Optional array of per-world masks indicating which worlds should be reset.\n
                Shape of `(num_worlds,)` and type :class:`wp.int8 | wp.bool`
            actuator_q: Optional array of target actuated joint coordinates.\n
                Shape of `(num_actuated_joint_coords,)` and type :class:`wp.float32`
            actuator_u: Optional array of target actuated joint DoF velocities.\n
                Shape of `(num_actuated_joint_dofs,)` and type :class:`wp.float32`
            joint_q: Optional array of target joint coordinates.\n
                Shape of `(num_joint_coords,)` and type :class:`wp.float32`
            joint_u: Optional array of target joint DoF velocities.\n
                Shape of `(num_joint_dofs,)` and type :class:`wp.float32`
            base_q: Optional array of target base body poses.\n
                Shape of `(num_worlds,)` and type :class:`wp.transformf`
            base_u: Optional array of target base body twists.\n
                Shape of `(num_worlds,)` and type :class:`wp.spatial_vectorf`
        """
        # Convert base pose from body-origin to COM frame
        if base_q is not None:
            base_q_com = wp.zeros_like(base_q)
            self._kamino.convert_base_origin_to_com(
                base_body_index=self._model_kamino.info.base_body_index,
                body_com=self._model_kamino.bodies.i_r_com_i,
                base_q=base_q,
                base_q_com=base_q_com,
            )
            base_q = base_q_com

        # TODO: fix brittle in-place update of arrays after conversion
        state_out_kamino = self._kamino.StateKamino.from_newton(self._model_kamino.size, self.model, state_out)
        self._solver_kamino.reset(
            state_out=state_out_kamino,
            world_mask=world_mask,
            actuator_q=actuator_q,
            actuator_u=actuator_u,
            joint_q=joint_q,
            joint_u=joint_u,
            base_q=base_q,
            base_u=base_u,
        )

        # Convert com-frame poses from Kamino reset to body-origin frame
        self._kamino.convert_body_com_to_origin(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q_com=state_out_kamino.q_i,
            body_q=state_out_kamino.q_i,
            world_mask=world_mask,
            body_wid=self._model_kamino.bodies.wid,
        )

    @override
    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        """
        Simulate the model for a given time step using the given control input.

        When ``contacts`` is not ``None`` (i.e. produced by :meth:`Model.collide`),
        those contacts are converted to Kamino's internal format and used directly,
        bypassing Kamino's own collision detector.  When ``contacts`` is ``None``,
        Kamino's internal collision pipeline runs as a fallback.

        Args:
            state_in: The input state.
            state_out: The output state.
            control: The control input.
                Defaults to `None` which means the control values from the
                :class:`Model` are used.
            contacts: The contact information from Newton's collision
                pipeline, or ``None`` to use Kamino's internal collision detector.
            dt: The time step (typically in seconds).
        """
        # Interface the input state containers to Kamino's equivalents
        # NOTE: These should produce zero-copy views/references
        # to the arrays of the source Newton containers.
        state_in_kamino = self._kamino.StateKamino.from_newton(self._model_kamino.size, self.model, state_in)
        state_out_kamino = self._kamino.StateKamino.from_newton(self._model_kamino.size, self.model, state_out)

        # Handle the control input, defaulting to the model's
        # internal control arrays if None is provided.
        if control is None:
            control = self.model.control(clone_variables=False)
        control_kamino = self._kamino.ControlKamino.from_newton(control)

        # If contacts are provided, use them directly, bypassing Kamino's collision detector
        if contacts is not None:
            self._kamino.convert_contacts_newton_to_kamino(self.model, state_in, contacts, self._contacts_kamino)
            _detector = None
        # Otherwise, use Kamino's internal collision detector to generate contacts
        else:
            _detector = self._collision_detector_kamino

        # Convert Newton body-frame poses to Kamino CoM-frame poses using
        # Kamino's corrected body-com offsets (can differ from Newton model data).
        self._kamino.convert_body_origin_to_com(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q=state_in_kamino.q_i,
            body_q_com=state_in_kamino.q_i,
        )

        # Step the physics solver
        self._solver_kamino.step(
            state_in=state_in_kamino,
            state_out=state_out_kamino,
            control=control_kamino,
            contacts=self._contacts_kamino,
            detector=_detector,
            dt=dt,
        )

        # Convert back from Kamino CoM-frame to Newton body-frame poses using
        # the same corrected body-com offsets as the forward conversion.
        self._kamino.convert_body_com_to_origin(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q_com=state_in_kamino.q_i,
            body_q=state_in_kamino.q_i,
        )
        self._kamino.convert_body_com_to_origin(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q_com=state_out_kamino.q_i,
            body_q=state_out_kamino.q_i,
        )

    @override
    def notify_model_changed(self, flags: int):
        """Propagate Newton model property changes to Kamino's internal ModelKamino.

        Args:
            flags: Bitmask of :class:`SolverNotifyFlags` indicating which properties changed.
        """
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self._update_gravity()

        if flags & SolverNotifyFlags.BODY_PROPERTIES:
            pass  # TODO: convert to CoM-frame if body_q_i_0 is changed at runtime?

        if flags & SolverNotifyFlags.BODY_INERTIAL_PROPERTIES:
            # Kamino's RigidBodiesModel references Newton's arrays directly
            # (m_i, inv_m_i, i_I_i, inv_i_I_i, i_r_com_i), so no copy needed.
            pass

        if flags & SolverNotifyFlags.SHAPE_PROPERTIES:
            pass  # TODO: ???

        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self._update_joint_transforms()

        if flags & SolverNotifyFlags.JOINT_DOF_PROPERTIES:
            # Joint limits (q_j_min, q_j_max, dq_j_max, tau_j_max) are direct
            # references to Newton's arrays, so no copy needed.
            pass

        if flags & SolverNotifyFlags.ACTUATOR_PROPERTIES:
            pass  # TODO: ???

        if flags & SolverNotifyFlags.CONSTRAINT_PROPERTIES:
            pass  # TODO: ???

        unsupported = flags & ~(
            SolverNotifyFlags.MODEL_PROPERTIES
            | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            | SolverNotifyFlags.JOINT_PROPERTIES
            | SolverNotifyFlags.JOINT_DOF_PROPERTIES
        )
        if unsupported:
            self._kamino.msg.warning(
                "SolverKamino.notify_model_changed: flags 0x%x not yet supported",
                unsupported,
            )

    @override
    def update_contacts(self, contacts: Contacts, state: State) -> None:
        """
        Converts Kamino contacts to Newton's Contacts format.

        Args:
            contacts: The Newton Contacts object to populate.
            state: Simulation state providing ``body_q`` for converting
                world-space contact positions to body-local frame.
        """
        # Ensure the containers are not None and of the correct shape
        if contacts is None:
            raise ValueError("contacts cannot be None when calling SolverKamino.update_contacts")
        elif not isinstance(contacts, Contacts):
            raise TypeError(f"contacts must be of type Contacts, got {type(contacts)}")
        if state is None:
            raise ValueError("state cannot be None when calling SolverKamino.update_contacts")
        elif not isinstance(state, State):
            raise TypeError(f"state must be of type State, got {type(state)}")

        # Skip the conversion if contacts have not been allocated
        if self._contacts_kamino is None or self._contacts_kamino._data.model_max_contacts_host == 0:
            return

        # Ensure the output contacts containers has sufficient size to hold the contact data from Kamino
        if self._contacts_kamino._data.model_max_contacts_host > contacts.rigid_contact_max:
            raise ValueError(
                f"Contacts container has insufficient capacity for Kamino contacts: "
                f"model_max_contacts={self._contacts_kamino._data.model_max_contacts_host} > "
                f"contacts.rigid_contact_max={contacts.rigid_contact_max}"
            )

        # If all checks pass, proceed to convert contacts from Kamino to Newton format
        self._kamino.convert_contacts_kamino_to_newton(self.model, state, self._contacts_kamino, contacts)

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """
        Register custom attributes for SolverKamino.

        Args:
            builder: The model builder to register the custom attributes to.
        """
        # Ensure Kamino dependencies are imported so that any custom attributes defined
        if cls._kamino is None:
            cls._import_kamino()

        # State attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_f_total",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.BODY,
                dtype=wp.spatial_vectorf,
                default=wp.spatial_vectorf(0.0),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_q_prev",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.JOINT_COORD,
                dtype=wp.float32,
                default=0.0,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_lambdas",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.float32,
                default=0.0,
            )
        )

        # Register KaminoSceneAPI attributes so the USD importer will store them on the model
        cls.Config.register_custom_attributes(builder)

    ###
    # Internals
    ###

    @classmethod
    def _import_kamino(cls):
        """Import the Kamino dependencies and cache them as class variables."""
        if cls._kamino is None:
            try:
                with warnings.catch_warnings():
                    # Set a filter to make all ImportWarnings "always" appear
                    # This is useful to debug import errors on Windows, for example
                    warnings.simplefilter("always", category=ImportWarning)

                    from . import _src as kamino  # noqa: PLC0415

                    cls._kamino = kamino
                    cls.Config = kamino.SolverKaminoImpl.Config
            except ImportError as e:
                raise ImportError("Kamino backend not found.") from e

    @staticmethod
    def _validate_model_compatibility(model: Model):
        """
        Validates that the model does not contain components unsupported by SolverKamino:
        - particles
        - springs
        - triangles, edges, tetrahedra
        - muscles
        - equality constraints
        - distance, cable, or gimbal joints

        Args:
            model: The Newton model to validate.

        Raises:
            ValueError: If the model contains unsupported components.
        """

        unsupported_features = []
        if model.particle_count > 0:
            unsupported_features.append(f"particles (found {model.particle_count})")
        if model.spring_count > 0:
            unsupported_features.append(f"springs (found {model.spring_count})")
        if model.tri_count > 0:
            unsupported_features.append(f"triangle elements (found {model.tri_count})")
        if model.edge_count > 0:
            unsupported_features.append(f"edge elements (found {model.edge_count})")
        if model.tet_count > 0:
            unsupported_features.append(f"tetrahedral elements (found {model.tet_count})")
        if model.muscle_count > 0:
            unsupported_features.append(f"muscles (found {model.muscle_count})")
        if model.equality_constraint_count > 0:
            unsupported_features.append(f"equality constraints (found {model.equality_constraint_count})")

        # Check for unsupported joint types
        if model.joint_count > 0:
            joint_type_np = model.joint_type.numpy()
            joint_dof_dim_np = model.joint_dof_dim.numpy()
            joint_q_start_np = model.joint_q_start.numpy()
            joint_qd_start_np = model.joint_qd_start.numpy()

            unsupported_joint_types = {}

            for j in range(model.joint_count):
                joint_type = int(joint_type_np[j])
                dof_dim = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
                q_count = int(joint_q_start_np[j + 1] - joint_q_start_np[j])
                qd_count = int(joint_qd_start_np[j + 1] - joint_qd_start_np[j])

                # Check for explicitly unsupported joint types
                if joint_type == JointType.DISTANCE:
                    unsupported_joint_types["DISTANCE"] = unsupported_joint_types.get("DISTANCE", 0) + 1
                elif joint_type == JointType.CABLE:
                    unsupported_joint_types["CABLE"] = unsupported_joint_types.get("CABLE", 0) + 1
                # Check for GIMBAL configuration (3 coords, 3 DoFs, 0 linear/3 angular)
                elif joint_type == JointType.D6 and q_count == 3 and qd_count == 3 and dof_dim == (0, 3):
                    unsupported_joint_types["D6 (GIMBAL)"] = unsupported_joint_types.get("D6 (GIMBAL)", 0) + 1

            if len(unsupported_joint_types) > 0:
                joint_desc = [f"{name} ({count} instances)" for name, count in unsupported_joint_types.items()]
                unsupported_features.append("joint types: " + ", ".join(joint_desc))

        # If any unsupported features were found, raise an error
        if len(unsupported_features) > 0:
            error_msg = "SolverKamino cannot simulate this model due to unsupported features:"
            for feature in unsupported_features:
                error_msg += "\n  - " + feature
            raise ValueError(error_msg)

    def _update_gravity(self):
        """
        Updates Kamino's :class:`GravityModel` from Newton's model.gravity.

        Called when :data:`SolverNotifyFlags.MODEL_PROPERTIES` is raised,
        indicating that ``model.gravity`` may have changed at runtime.
        """
        self._kamino.convert_model_gravity(self.model, self._model_kamino.gravity)

    def _update_joint_transforms(self):
        """
        Re-derive Kamino joint anchors and axes from Newton's joint_X_p / joint_X_c.

        Called when :data:`SolverNotifyFlags.JOINT_PROPERTIES` is raised,
        indicating that ``model.joint_X_p`` or ``model.joint_X_c`` may have
        changed at runtime (e.g. animated root transforms).
        """
        self._kamino.convert_model_joint_transforms(self.model, self._model_kamino.joints)
