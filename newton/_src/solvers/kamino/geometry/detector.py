###########################################################################
# KAMINO: Collision Detector Interface
###########################################################################

from __future__ import annotations

import warp as wp

from typing import List

from warp.context import Devicelike

from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.model import Model
from newton._src.solvers.kamino.core.model import ModelData   # TODO: change to state.State
from newton._src.solvers.kamino.geometry.collisions import Collisions
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.broadphase import nxn_broadphase
from newton._src.solvers.kamino.geometry.primitives import primitive_narrowphase
from newton._src.solvers.kamino.core.geometry import update_collision_geometries_state


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Collision Detector class
###

class CollisionDetector:
    """
    Collision Detection (CD) front-end interface.

    This class is responsible for performing collision detection as well
    as managing the collision containers and their memory allocations.
    """
    def __init__(
            self, builder: ModelBuilder | None = None,
            default_max_contacts: int | None = None,
            device: Devicelike = None
    ):
        # Cache the target device
        self._device = device

        # Declare the collisions and contacts containers
        self.collisions: Collisions | None = None
        self.contacts: Contacts | None = None

        # Declare the maximum number of contacts allocation caches
        self._model_max_contacts: int = 0
        self._world_max_contacts: List[int] = [0]

        # Retrieve the required contact capacity required by the model
        model_max_contacts, world_max_contacts = builder.required_contact_capacity()

        # Allocate the collisions and contacts containers if the model requires them (indicated by >= 0)
        if model_max_contacts >= 0:

            # NOTE #1: collisions are the inputs/outputs of the broad phase
            self.collisions = Collisions(builder=builder, device=device)

            # NOTE #2: contacts are the outputs of the narrow phase
            self.contacts = Contacts(capacity=world_max_contacts, default_max_contacts=default_max_contacts, device=device)

            # Cache the maximum number of contacts allocated for the model
            self._model_max_contacts: int = self.contacts.num_model_max_contacts
            self._world_max_contacts: List[int] = self.contacts.num_world_max_contacts

    @property
    def model_max_contacts(self) -> int:
        """
        The total maximum number of contacts allocated for the model across all worlds.
        """
        return self._model_max_contacts

    @property
    def world_max_contacts(self) -> int:
        """
        The maximum number of contacts allocated for each world in the model.
        """
        return self._world_max_contacts

    def collide(self, model: Model, state: ModelData):  # TODO: change to state.State
        """
        Perform collision detection for the a model with the specific state.
        """
        # Skip this operation if the model does not allocate contacts
        # TODO: change this to check if the model has any cgeoms
        if self._model_max_contacts <= 0:
            return

        # Clear all current collision pairs and contacts
        self.collisions.clear()
        self.contacts.clear()

        # Upate geometries states from the states of the bodies
        update_collision_geometries_state(state.bodies.q_i, model.cgeoms, state.cgeoms)

        # Perform the broad-phase collision detection to generate collision pairs
        nxn_broadphase(model.cgeoms, state.cgeoms, self.collisions.cmodel, self.collisions.cstate)

        # Perform the narrow-phase collision detection to generate active contacts
        primitive_narrowphase(model, state, self.collisions, self.contacts)
