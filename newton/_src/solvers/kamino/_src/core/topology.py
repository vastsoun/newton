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
    Index of the world to which the body belongs.\n
    Defaults to `-1`, indicating that the body has not yet been added to a world.
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

    max_arc_joints_per_topology: int = 0
    """Maximum number of arc joints in any topology entity (host-side)."""

    max_chord_joints_per_topology: int = 0
    """Maximum number of chord joints in any topology entity (host-side)."""

    max_arc_coords_per_topology: int = 0
    """Maximum number of generalized coordinates in any arc of any topology entity (host-side)."""

    max_chord_coords_per_topology: int = 0
    """Maximum number of generalized coordinates in any chord of any topology entity (host-side)."""

    max_arc_dofs_per_topology: int = 0
    """Maximum number of generalized degrees of freedom in any arc of any topology entity (host-side)."""

    max_chord_dofs_per_topology: int = 0
    """Maximum number of generalized degrees of freedom in any chord of any topology entity (host-side)."""

    label: list[str] | None = None
    """
    A list containing the label of each topology entity.\n
    Length of ``num_topologies`` and type :class:`str`.
    """

    ###
    # Identifiers
    ###

    wid: wp.array[wp.int32] | None = None
    """
    World index of each topology entity.\n
    Shape of ``(num_topologies,)`` and type :class:`int32`.
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
    """

    ###
    # Parameterization
    ###

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
    def from_newton(model: Model) -> TopologyModel:
        """
        Generates a :class:`TopologyModel` by parsing the body, joint and articulation arrays of a :class:`Model`.

        Args:
            model: A :class:`Model` instance containing the body, joint, and articulation arrays.

        Returns:
            A :class:`TopologyModel` instance containing the data from the provided model.

        Raises:
            NotImplementedError: This factory has not been implemented yet.
        """
        raise NotImplementedError("TopologyModel.from_newton is not implemented yet.")

    ###
    # Operations
    ###

    # TODO: IMPLEMENT


###
# TODO PROCESS:
#   1. Parse each world's lists of RigidBodyDescriptor and JointDescriptor to generate lists of NodeType and EdgeType, and use them to create a TopologyGraph.
#
#   2. Each TopologyGraph is parsed into its constituent components, i.e. subgraphs, using the modular component generator back-end.
#       2a. Each component is classified as an island or an orphan, and as connected or isolated.
#       2b. For each component, assign a base node/edge if there is a single grounding node/edge, or if multiple grounding edges are present but
#           only one of them is a 6-DoF FREE joint, take that. If a base node/edge is assigned, remove it from the grounding lists. If multiple
#           grounding edges are present and more than one of them are 6-DoF FREE joints, raise an error, as this violates the modelling conventions.
#           If no base node/edge can be assigned, then the component leaves them as `None` to be processed in the next steps.
#
#   3. Create a list of joints (i.e. JointDescriptors), as well as a list of EdgeType, to connect each isolated component to the world.
#      3a. For each isolated orphan, add a 6-DoF FREE joint connecting the orphan node to the world node, and assign it as a base edge.
#      3b. For each isolated island, use the set base selector back-end to select a base node and edge, to add a 6-DoF FREE joint connecting
#          the base node to the world node, and assign that as the component's base edge. The base selector accepts the TopologyComponent as
#          input, as well as the RigidBodyDescriptor and JointDescriptor lists of the world, to inform its selection.
#
#   4. For each graph component, generate a list of SpanningTree candidates using the modular spanning tree generator back-end,
#      taking as inputs the component subgraph, as well as the RigidBodyDescriptor and JointDescriptor lists of the world.
#       4a. If Base nodes/edges are present, then only generate spanning tree candidates using those as root nodes
#       4b. If no base node/edges are present but the component has grounding nodes/edges, use the latter as root nodes for spanning tree candidate generation.
#       4c. Otherwise, try brute-force generation of all possible subtrees.
#       4d. The SpanningTreeTraversal (e.g. DFS, BFS) argument should be supported.
#
#   5. For each component, select a single spanning tree from the list of candidates using the modular spanning tree selection back-end,
#      taking as inputs the list of spanning tree candidates, as well as the RigidBodyDescriptor and JointDescriptor lists of the world.
#
#   6. For each spanning tree, check the parent array against Featherstone's regular numbering rules, and if they are not satisfied, perform
#      index reassignment of the bodies and joints in the TopologyGraph instance accordingly, using the modular index reassignment mechanism.
#
#   7. Generate a TopologyDescriptor from the selected SpanningTree of each component of the TopologyGraph of the world.
#
#   8. Add the TopologyDescriptor to the model builder, and assign it to the corresponding bodies and joints in the world.
#
#   9. Generate the TopologyModel from the list of TopologyDescriptors.
#
#   10. (Optional) Perform index reassignment of the bodies and joints in the graph to optimize for better data locality
#       and satisfaction of Featherstone's regular numbering rules, by updating the TopologyGraph instance accordingly.
#
#
# TODO IMPLEMENTATIONS:
#   1. Implement a simple subgraph component base/edge selector back-end that assigns the base node/edge
#      to the heaviest moving body node as a first example of a TopologyComponent generation back-end.
#
#   2. Implement a minimum depth spanning tree generation back-end as a first example:
#       2a. Implement a brute-force minimum-depth spanning tree generation function that
#           generates all possible minimum-depth spanning trees of the component subgraph.
#       2b. Implement a function that generates all minimum-depth spanning trees starting from a given root node.
#       2c. Implement a function that computes the degree of each node in the component subgraph
#       2d. Implement a function that generates all minimum-depth spanning trees of the component subgraph by:
#           - first checking if a base node/edge is defined for the corresponding component, and use 2b to generate all spanning trees from the base node as root.
#           - if the component doesn't define a base node/edge, but has grounding nodes/edges, generate all minimum-depth spanning trees with those as root nodes using 2b.
#           - if no base or grounding nodes are present the generate all spanning trees setting the the node with the highest degree using 2c.
#           - in case of ties in degree, use the brute-force method of 2a. and generate all possible spanning trees.
#           - this function should accept arguments for:
#               - the spanning tree traversal mode (e.g. DFS, BFS)
#               - override all prioritization rules and just generate all possible spanning trees, if admissible.
#               - direct specification of the root node/edge for spanning tree generation, if admissible, and set that as the base node/edge in the source component.
#               - accept a maximum number of spanning trees to generate, and stop the generation process once that number is reached, if admissible.
#               - whether to prioritize balanced/symmetric trees over unbalanced ones, if admissible.
#
#   3. Implement a simple heuristic spanning tree selection back-end as a first example:
#       - For islands, select the spanning tree by:
#           - ordering based on tree depth, and selecting the one with minimum depth, if there are no ties.
#           - prioritize balanced/symmetric subtrees over unbalanced ones, if admissible.
#           - In case of remaining ties, just select and return the first candidate in the list.
#       - For orphans, select the trivial spanning tree with no edges.
#
#   4. Implement a mechanism (class or function) for body/joint index reassignment to optimize
#      for better data locality and satisfaction of Featherstone's regular numbering rules.
#       4a. It should take as input a TopologyGraph, performing a copy of the original and operate on that
#       4b. The components and trees attributes should be re-ordered according to the number of nodes + number of edges
#       4c. Construct two lists to hold index re-mappings using an appropriate container: dict or list of tuples, etc.
#       4c. It should first reorder nodes/edges to exactly group them according to their component membership
#       4d. Then, it should reorder nodes/edges within each component according to their spanning tree membership, i.e. first the tree arcs, then the chords.
#       4e. Finally, it should reorder nodes/edges within each tree according to the traversal mode of the original tree starting from the base node, if defined, or the node with the highest degree otherwise, and following Featherstone's regular numbering rules, i.e. for each joint edge (i, j), where i is the predecessor body and j is the successor body, it should ensure that min(i, j) < max(i, j) and that the parent of body j is body i.
#       4f. It should generate and return:
#       - a new TopologyGraph with re-ordered components and spanning trees, as well as updated node and edge indices according to the new body/joint ordering.
#       - a list of body node index reassignments, where the entry at index `i` gives the new body index assigned to the body with original index `i`.
#       - a list of joint edge index reassignments, where the entry at index `j` gives the new joint index assigned to the joint with original index `j`.
#
# TODO: CHECKS in UTs:
# - Define tests with a connected island with multiple grounding edges.
#   - In a first valid case, check that all grounding edges are correctly identified, but no base edge is assigned
#   - In a second valid case, set two of the grounding edges as a floating base and ensure that an exception is raised
#
###
