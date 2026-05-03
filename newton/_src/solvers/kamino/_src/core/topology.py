# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines types & containers for constrained multi-body system topologies.

A `topology` represents a collection of bodies connected by joints
that captures their connectivity in the form of a graph. Concretely,
it is defined by two ordered sets of indices: a set of `bodies` and
a set of `joints` that refer to the corresponding entities in the
world. Bodies thus correspond to nodes of the graph and joints to edges.

Within the context of mechanical systems, and robotics in particular, a
topology represents a generalization of an `articulation`, which, is a
specific type of topology where the graph reduces to a topological tree
(i.e. without loops), a.k.a. a kinematic tree [1].
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .....sim import Model
from ..topology import TopologySpanningTree
from .size import SizeKamino
from .types import Descriptor, override

###
# Module interface
###

__all__ = [
    "TopologyDescriptor",
    "TopologyModel",
]


###
# Interfaces
###


@dataclass
class TopologyDescriptor(Descriptor):
    """
    A container to describe a single topology entity in the model builder.
    """

    ###
    # Attributes
    ###

    tree: TopologySpanningTree | None = None
    """Spanning tree defining the topology entity."""

    ###
    # Metadata - to be set by the ModelBuilderKamino instance when added
    ###

    wid: int = -1
    """
    Index of the world to which the topology entity belongs.\n
    Defaults to `-1`, indicating that the topology entity has not yet been added to a world.
    """

    tid: int = -1
    """
    Index of the topology entity w.r.t. its world.\n
    Defaults to `-1`, indicating that the topology entity has not yet been added to a world.
    """

    ###
    # Operations
    ###

    @override
    def __str__(self) -> str:
        """Returns a human-readable string representation of the TopologyDescriptor."""
        return (
            f"TopologyDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"wid: {self.wid},\n"
            f"tid: {self.tid},\n"
            f"tree: {self.tree},\n"
            f")"
        )

    @override
    def __repr__(self) -> str:
        """Returns a human-readable string representation of the TopologyDescriptor object."""
        return self.__str__()


@dataclass
class TopologyModel:
    """
    A container holding the time-invariant data defining topology entities of a simulation model.
    """

    ###
    # Meta-Data
    ###

    num_worlds: int = 0
    """Total number of worlds in the model (host-side)."""

    num_topologies: int = 0
    """Total number of topology entities in the model (host-side)."""

    max_joints_per_topology: int = 0
    """Maximum number of joints in any topology entity (host-side)."""

    max_arc_joints_per_topology: int = 0
    """Maximum number of arc joints in any topology entity (host-side)."""

    max_chord_joints_per_topology: int = 0
    """Maximum number of chord joints in any topology entity (host-side)."""

    max_coords_per_topology: int = 0
    """Maximum number of generalized coordinates in any arc of any topology entity (host-side)."""

    max_arc_coords_per_topology: int = 0
    """Maximum number of generalized coordinates in any arc of any topology entity (host-side)."""

    max_chord_coords_per_topology: int = 0
    """Maximum number of generalized coordinates in any chord of any topology entity (host-side)."""

    max_dofs_per_topology: int = 0
    """Maximum number of generalized degrees of freedom in any topology entity (host-side)."""

    max_arc_dofs_per_topology: int = 0
    """Maximum number of generalized degrees of freedom in any arc of any topology entity (host-side)."""

    max_chord_dofs_per_topology: int = 0
    """Maximum number of generalized degrees of freedom in any chord of any topology entity (host-side)."""

    label: list[str] | None = None
    """
    A list containing the label of each topology entity.\n
    Length of ``num_topologies`` and type :class:`str`.
    NOTE: This attribute aliases :attr:`ModelBuilder.articulation_label`.
    """

    ###
    # Identifiers
    ###

    wid: wp.array[wp.int32] | None = None
    """
    World index of each topology entity.\n
    Shape of ``(num_topologies,)`` and type :class:`int32`.
    NOTE: This attribute aliases :attr:`ModelBuilder.articulation_world`.
    """

    tid: wp.array[wp.int32] | None = None
    """
    Topology index of each topology entity w.r.t. its world (matches :attr:`TopologyDescriptor.tid`).\n
    Shape of ``(num_topologies,)`` and type :class:`int32`.
    """

    ###
    # Counters
    ###

    world_start: wp.array[wp.int32] | None = None
    """
    Start index of the first topology entity per world.\n
    Shape of ``(num_worlds + 2,)`` and type :class:`int32`.

    The entries at indices ``0`` to ``num_worlds - 1`` store the start index
    of the topology entities belonging to that world. The second-last element
    (accessible via index ``-2``) stores the start index of the global topology
    entities (i.e. with world index ``-1``) added to the end of the model, and
    the last element stores the total topology entity count.

    The number of topology entities in a given world ``w`` can be computed as::

        num_topology_entities_in_world = world_start[w + 1] - world_start[w]

    The total number of global topology entities can be computed as::

        num_global_topology_entities = world_start[-1] - world_start[-2] + world_start[0]

    NOTE: This attribute aliases :attr:`ModelBuilder.articulation_world_start`.
    """

    ###
    # Parameterization
    ###

    tree_joints_start: wp.array[wp.int32] | None = None
    """
    Start index of the first joint per topology entity.\n
    Shape of ``(num_topologies + 1,)`` and type :class:`int32`.

    The number of joints in a given topology entity ``t`` can be computed as::

        num_joints_in_topology = tree_joints_start[t + 1] - tree_joints_start[t]

    NOTE: This attribute aliases :attr:`ModelBuilder.articulation_start`.
    """

    arc_joints_start: wp.array[wp.int32] | None = None
    """
    Start index of the first arc joint per topology entity.\n
    Shape of ``(num_topologies + 1,)`` and type :class:`int32`.

    The number of arc joints in a given topology entity ``t`` can be computed as::

        num_arc_joints_in_topology = arc_joints_start[t + 1] - arc_joints_start[t]
    """

    chord_joints_start: wp.array[wp.int32] | None = None
    """
    Start index of the first chord joint per topology entity.\n
    Shape of ``(num_topologies + 1,)`` and type :class:`int32`.

    The number of chord joints in a given topology entity ``t`` can be computed as::

        num_chord_joints_in_topology = chord_joints_start[t + 1] - chord_joints_start[t]
    """

    ###
    # Factories
    ###

    @staticmethod
    def from_descriptors(descriptors: list[list[TopologyDescriptor]]) -> TopologyModel:
        """
        Generates a :class:`TopologyModel` from a nested list of :class:`TopologyDescriptor` objects.

        Args:
            descriptors: A nested list of :class:`TopologyDescriptor` objects, where
                each inner list corresponds to the topology entities of a single world.

        Returns:
            A :class:`TopologyModel` instance containing the data from the provided descriptors.

        Raises:
            NotImplementedError: This factory has not been implemented yet.
        """
        raise NotImplementedError("TopologyModel.from_descriptors is not implemented yet.")

    @staticmethod
    def from_newton(model: Model, size: SizeKamino) -> TopologyModel:
        """
        Generates a :class:`TopologyModel` by parsing the body, joint and articulation arrays of a :class:`Model`.

        Args:
            model: A :class:`Model` instance containing the body, joint, and articulation arrays.
            size: A :class:`SizeKamino` instance containing the size information for the model.
        Returns:
            A :class:`TopologyModel` instance containing the data from the provided model.

        Raises:
            NotImplementedError: This factory has not been implemented yet.
        """
        raise NotImplementedError("TopologyModel.from_newton is not implemented yet.")
