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

See [1] for more details on the topic of topology and its relevance to rigid-body dynamics algorithms.
See [2] for a concise introduction to branch-induced sparsity in rigid-body dynamics.

References:
- [1] Featherstone, Roy. Rigid body dynamics algorithms. Boston, MA: Springer US, 2008.
- [2] Roy Featherstone, Branch-Induced Sparsity in Rigid-Body Dynamics
      https://royfeatherstone.org/talks/BISinRBD.pdf

----
Terminology

- "node":
    A node represents a body in the topology.
    Each node is uniquely identified by its associated body index in the
    range `[0, NB-1]`, where `NB` is the total number of bodies.

- "edge":
    An edge represents a joint in the topology.
    Each edge is uniquely identified by its associated joint index in the
    range `[0, NJ-1]`, where `NJ` is the total number of joints.

- "world node":
    An implicit node representing the fixed body of the world.
    It is denoted with a body index of `-1` by default.

- "component":
    A connected subgraph, i.e. a maximal set of nodes that are mutually reachable through edges.
    A graph may contain multiple components, which are disconnected from each other.

- "connected component":
    A component that is connected to the world node with at least one edge.

- "isolated component":
    A component that is disconnected from the rest of the graph, including the world node.

- "island":
    A component containing at least two body nodes. If connected to the world,
    then it's a "connected island", otherwise it's an "isolated island".

- "orphan":
    A single-node component. If connected to the world, then it's
    a "connected orphan", otherwise it's an "isolated orphan".

- "component grounding node" or just "grounding node":
    A component's node connected to the world node via any type of joint except
    for a 6-DoF FREE joint. Each component can have multiple grounding nodes.

- "component grounding edge" or just "grounding edge":
    An edge connecting one the component's nodes to the world node, and can represent any type
    of joint except for a 6-DoF FREE joint. Each component can have multiple grounding edges.

- "component base node" or just "base node":
    A node of a component that is connected to the world node via any type of
    joint. A graph can have multiple base nodes, but only one per component.

- "component base edge" or just "base edge":
    An edge of a component that connects its base node to the world node via any type
    of joint. A graph can have multiple base edges, but only one per component.

- "spanning tree":
    A spanning tree ``Gt_i`` of a component ``G_i`` is a proper subgraph that includes all its nodes and a
    subset of its edges, such that the subgraph is a topological tree (i.e. it is connected and acyclic). A
    spanning tree is used to represent the primary kinematic structure of the component, while the remaining
    edges (called chords) represent additional constraints that form kinematic loops.

- "tree root node" or just "root node":
    A root node is an alias for the unique base node of a component when
    the latter is, or has a derived, spanning tree (i.e. an articulation).

- "tree root edge" or just "root edge":
    A root edge is an alias for the unique base edge of a component when
    the latter is, or has a derived, spanning tree (i.e. an articulation).

- "arc edge" or just "arc":
    An edge included in the spanning tree of a component.

- "chord edge" or just "chord":
    An edge not included in the spanning tree of a component
    that forms a loop in the original component subgraph.

----
Topological Conventions

Graphs & Components:
- A topology graph is an undirected graph `G` defined by a set of (moving) body nodes and joint edges, where each edge connects a pair of body nodes.
- The graph also includes the implicit world node with index `-1`.
- Each graph can contain multiple component subgraphs ``G_i`` with ``i`` in ``[0, NC-1]``, where ``NC`` is the number of components in the graph.
- A component subgraph ``G_i`` is "connected" if it contains at least one edge connecting any of its body nodes to the world node, and "isolated" otherwise.
- A component subgraph ``G_i`` is an "island" if it contains more than one body node, and an "orphan" if it contains exactly one body node.
- Each component subgraph ``G_i`` can contain multiple edges connecting it to the world node.
- Edges connecting a body node to the world node are exclusively either "grounding" edges or "base" edges, and cannot be both.
- Each component can have multiple grounding edges, but only one base edge, and thus only one base node.
- Grounding edges can correspond to any joint type except for 6-DoF FREE joints.
- Base edges can correspond to any type of joint.
- These conventions allow for modelling mechanisms such as stewart platforms using kinematic trees where the floating end-effector
  body can be assigned as the base node via a 6-DoF FREE base joint, while the joints connecting the mechanism to the floor can
  be assigned as grounding edges. This allows the spanning tree to be rooted at the end-effector body instead of those connected
  to the ground. The latter and their associated grounding edges are then used to define loop joints to constrain the mechanism.

Spanning Trees:
- A spanning tree ``Gt`` of a graph ``G`` is a proper subgraph of ``G`` that includes all its nodes
  and a subset of its edges such that ``Gt`` is a topological tree (i.e. it is connected and acyclic).
- If a graph ``G`` contains ``NC`` components, the each component subgraph ``G_i`` of ``G`` with ``i`` in ``[0, NC-1]``, can have a spanning tree ``Gt_i``
- Each spanning tree ``Gt`` contains ``NB`` moving body nodes, ``NJ`` joint edges, ``NB - 1`` arcs (i.e. tree joints), ``NJ - NB + 1`` chords (i.e. loop joints)
- The importance of the one-chord cycles is that they identify the set of independent kinematic loops
- The number of chords ``NL = NJ - NB`` denotes the number of of independent kinematic loops
- Although the number of chords ``NL`` in ``G`` is fixed, the set of arcs that happen to be chords will vary with the choice of ``Gt``.
- Each spanning tree defines exactly one moving body node that is designated as the local "root body node", or simply "root" of the spanning tree.
- The "root body node" is the unique moving body node that is directly connected to the implicit world node, and
  thus also defines the "base body node" of the component subgraph from which the spanning tree is derived.
- The implicit world node is not included in the spanning tree, but is considered to be the
  parent of the "root body node", thus making it the global root node of the spanning tree.

Numbering Rules for Spanning Trees (Featherstone's "Regular Numbering" Scheme):
- We assume that a graph ``G`` contains ``NB`` moving body nodes (with the fixed-base world node not included in this count),
  and ``NJ`` joint edges that includes the 6-DoF base joint connecting the floating-base root node to the world node.
1. Choose a spanning tree ``Gt`` of ``G``.
2. Assign the number ``0`` to the node representing the fixed base, and define this node to be the global root node of ``Gt``.
3. The remaining nodes are numbered from ``1`` to ``NB`` in any order such that each node has a higher number than its parent in ``Gt``.
4. Number the arcs in ``Gt`` from ``1`` to ``NB`` such that arc ``i`` connects between node ``i`` and its parent.
5. Number all remaining edges, i.e. chords, from ``NB + 1`` to ``NJ`` in any order.
6. Each body gets the same number as its node, and each joint gets the same number as its arc.

Modified Numbering Rules for Spanning Trees in Kamino (Modified Featherstone's "Regular Numbering" Scheme for 0-based indexing):
- We assume that a graph ``G`` contains ``NB`` moving body nodes (with the fixed-base world node not included in this count),
  and ``NJ`` joint edges that includes the 6-DoF base joint connecting the floating-base root node to the world node.
1. Choose a spanning tree ``Gt`` of ``G``.
2. Assign the number ``-1`` to the node representing the fixed base, and define this node to be the global root node of ``Gt``.
3. The remaining nodes are numbered from ``0`` to ``NB - 1`` in any order such that each node has a higher number than its parent in ``Gt``.
4. Number the arcs in ``Gt`` from ``0`` to ``NB - 1`` such that arc ``i`` connects between node ``i`` and its parent.
5. Number all remaining edges, i.e. chords, from ``NB`` to ``NJ - 1`` in any order.
6. Each body gets the same number/index as its node, and each joint gets the same number/index as its arc.

Notes on the modified scheme:
- Because the world node has index ``-1`` rather than ``0``, the parent of body ``0`` (the
  unique root body of each spanning tree) is the world node by convention. In the
  ``parents`` array of :class:`TopologySpanningTree` this is encoded as ``parents[0] = -1``.
- The condition that each node has a higher number than its parent in step 3 thus holds
  for all moving body nodes ``i ≥ 1`` whose parent is also a moving body, and is satisfied
  trivially for the root body ``0`` whose parent is the world node ``-1``.

"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Literal

import warp as wp

from .....sim import JointType, Model
from ..utils import logger as msg
from .bodies import RigidBodyDescriptor
from .joints import JointDescriptor, JointDoFType
from .types import Descriptor, override

###
# Module interface
###

__all__ = [
    "DEFAULT_WORLD_NODE_INDEX",
    "ComponentConnectivity",
    "ComponentType",
    "EdgeType",
    "NodeType",
    "OrientedEdge",
    "SpanningTreeTraversal",
    "TopologyComponent",
    "TopologyComponentBaseSelectorBase",
    "TopologyComponentParser",
    "TopologyComponentParserBase",
    "TopologyDescriptor",
    "TopologyGraph",
    "TopologyGraphVisualizer",
    "TopologyGraphVisualizerBase",
    "TopologyModel",
    "TopologySpanningTree",
    "TopologySpanningTreeGeneratorBase",
    "TopologySpanningTreeSelectorBase",
]

###
# Constants
###

DEFAULT_WORLD_NODE_INDEX: int = -1
"""
The default index for the implicit world node in the topology graph.
This is the conventional index used for the world body in Newton.
"""


###
# Types
###

NodeType = int
"""
Type alias for body nodes in a topology, which
are represented as integer `body_index` values.
"""

EdgeType = tuple[int, int, tuple[NodeType, NodeType]]
"""
Type alias for joint edges in a topology, which are represented as tuples of the
form `(joint_type, joint_index, (predecessor_body_index, successor_body_index))`.
"""

ComponentType = Literal["island", "orphan"]
"""
A string literal defining the subgraph component types, which can be
either "island" (a connected subgraph) or "orphan" (an isolated node).
"""

ComponentConnectivity = Literal["connected", "isolated"]
"""
A string literal defining the component connectivity, which can be either "connected"
(i.e. connected to the world node) or "isolated" (i.e. not connected to the world node).
"""

SpanningTreeTraversal = Literal["dfs", "bfs"]
"""
A string literal defining the supported traversal modes used in spanning-tree generation
algorithms, which can be either "dfs" (depth-first search) or "bfs" (breadth-first search).
"""


@dataclass(frozen=True, slots=True)
class OrientedEdge:
    """
    A joint edge whose endpoint pair has been oriented relative to a chosen
    spanning-tree root, while preserving the original (input) endpoint pair.

    Used by spanning-tree generator backends to disambiguate edge polarity when the parent/child
    relationship implied by a tree differs from the predecessor/successor pair of the original
    input edge. The :attr:`original` pair preserves the input orientation so that the source graph
    can be reconstructed, while the :attr:`oriented` pair carries the parent-to-child polarity used
    by downstream arc/chord representations.
    """

    joint_type: int
    """The type of the joint."""
    joint_index: int
    """The index of the joint."""
    original: tuple[NodeType, NodeType]
    """The original edge as provided in the input subgraph."""
    oriented: tuple[NodeType, NodeType]
    """The oriented edge as used in the returned tree/chord representation."""

    def as_tuple(self) -> EdgeType:
        """Returns the oriented edge as an :data:`EdgeType` tuple of the form ``(joint_type, joint_index, oriented)``."""
        return (self.joint_type, self.joint_index, self.oriented)


@dataclass
class TopologyComponent:
    """
    A container to represent a component of a graph, i.e. a subgraph with its associated
    body nodes and joint edges, as well as meta-data used for spanning tree generation.
    """

    nodes: list[NodeType] | None = None
    """List of body nodes associated with the component."""

    edges: list[EdgeType] | None = None
    """List of joint edges associated with the component."""

    ground_nodes: list[NodeType] | None = None
    """List of body nodes that connect the component to the implicit world node."""

    ground_edges: list[EdgeType] | None = None
    """List of joint edges that connect the component to the implicit world node."""

    base_node: NodeType | None = None
    """The base body node of the component. `None` if the component has no assigned base node."""

    base_edge: EdgeType | None = None
    """The base joint edge of the component. `None` if the component has no assigned base joint."""

    is_connected: bool | None = None
    """
    Indicates whether the component is connected to the world node.

    Set during parsing based on the original graph topology (``True`` if the component
    has any grounding edge or an auto-assigned base edge). Refreshed by the base selector
    module :meth:`TopologyGraph.select_component_bases` when a previously-isolated component
    is assigned a base edge by the base selector module.
    """

    is_island: bool | None = None
    """Indicates whether the component is an "island", if True, or an "orphan", if False."""

    @override
    def __post_init__(self):
        """Post-initialization method to validate component attributes.

        Raises:
            ValueError: If any of the following invariants is violated:

                - A grounding edge represents a 6-DoF FREE joint.
                - ``base_edge`` is set but ``base_node`` is not, or vice versa.
                - ``base_node`` is not contained in ``nodes``.
                - ``base_node`` is not one of the two endpoints of ``base_edge``.
                - ``ground_nodes`` is not the set of non-world endpoints of ``ground_edges``.
                - ``is_island`` does not match ``len(nodes) > 1``.
        """
        # If grounding nodes/edges are provided, check that they are
        # not FREE joints which would violate modelling conventions
        if self.ground_edges is not None:
            for edge in self.ground_edges:
                joint_type, _, _ = edge
                if joint_type == JointType.FREE:
                    raise ValueError(f"Grounding edge `{edge}` cannot be 6-DoF FREE joint.")

        # Base node and base edge must either both be assigned or both be None
        if (self.base_node is None) != (self.base_edge is None):
            raise ValueError(
                f"Component must define both `base_node` and `base_edge`, or neither: "
                f"got base_node={self.base_node!r}, base_edge={self.base_edge!r}."
            )

        # If a base node is assigned, it must be a member of the component's nodes
        # and must be one of the two endpoints of the base edge
        if self.base_node is not None:
            if self.nodes is not None and self.base_node not in self.nodes:
                raise ValueError(
                    f"Component base node `{self.base_node}` is not contained in component nodes `{self.nodes}`."
                )
            if self.base_edge is not None:
                _, _, base_pair = self.base_edge
                if self.base_node not in base_pair:
                    raise ValueError(
                        f"Component base node `{self.base_node}` is not an endpoint of base edge `{self.base_edge}`."
                    )

        # Grounding nodes must be exactly the set of non-world endpoints of grounding edges.
        # We require the world index to disambiguate; a negative value other than `world`
        # would not be inferable here, so we accept any negative endpoint as "world-like"
        # for validation purposes only.
        if self.ground_edges is not None and self.ground_nodes is not None:
            implied_ground_nodes = {n for _, _, pair in self.ground_edges for n in pair if n >= 0}
            if set(self.ground_nodes) != implied_ground_nodes:
                raise ValueError(
                    f"Component `ground_nodes` {sorted(self.ground_nodes)} does not match the non-world "
                    f"endpoints {sorted(implied_ground_nodes)} of `ground_edges`."
                )

        # `is_island` must agree with the actual node count, when both are available
        if self.is_island is not None and self.nodes is not None:
            expected_is_island = len(self.nodes) > 1
            if self.is_island != expected_is_island:
                raise ValueError(
                    f"Component `is_island={self.is_island}` is inconsistent with `len(nodes)={len(self.nodes)}`; "
                    f"expected `is_island={expected_is_island}`."
                )

    @override
    def __str__(self) -> str:
        """Returns a human-readable string representation of the TopologyComponent object."""
        return (
            f"TopologyComponent(\n"
            f"nodes: {self.nodes},\n"
            f"edges: {self.edges},\n"
            f"ground_nodes: {self.ground_nodes},\n"
            f"ground_edges: {self.ground_edges},\n"
            f"base_node: {self.base_node},\n"
            f"base_edge: {self.base_edge},\n"
            f"is_connected: {self.is_connected},\n"
            f"is_island: {self.is_island},\n"
            f")"
        )

    @override
    def __repr__(self) -> str:
        """Returns a human-readable string representation of the TopologyComponent object."""
        return self.__str__()


@dataclass
class TopologySpanningTree:
    """
    Defines a compact container to represent a spanning tree of a component subgraph.

    For each component subgraph `G_i`, a spanning tree `Gt_i` represents a proper
    subgraph of `G_i` that includes all the nodes of `G_i` and a subset of its edges,
    such that `Gt_i` is a tree (i.e. it is connected and acyclic). The edges included
    in the spanning tree are called "arcs", while the remaining ones are called "chords",
    i.e. are edges that form loops in the original component subgraph. A spanning tree
    represents the topology of a multi-body system as a so-called kinematic tree.

    Note that this data structure not only includes information about the spanning tree itself,
    but also a description of the loop-closure constraints in the original component subgraph.

    Note:
        This is a **host-side staging container**. All array-shaped fields (``arcs``,
        ``chords``, ``predecessors``, ``successors``, ``parents``, ``support``,
        ``children``, ``subtree``) are stored as Python lists for ease of construction
        and mutation by the spanning-tree generator and selector backends. They are
        converted to packed :class:`wp.array[wp.int32]` device buffers only when the
        per-world descriptors are aggregated into a :class:`TopologyModel`. Do not pass
        these list fields directly to Warp kernels.
    """

    ###
    # Meta-Data
    ###

    traversal: SpanningTreeTraversal = "dfs"
    """
    The traversal mode used for generating the spanning tree, which can
    be either "dfs" (depth-first search) or "bfs" (breadth-first search).
    """

    depth: int = 0
    """
    The depth of the spanning tree, defined as the maximum number of
    arcs in any path from the root node to a leaf node in the tree.
    """

    directed: bool = False
    """
    Indicates whether the spanning tree is directed, i.e. if the
    arcs have a defined direction from parent to child nodes.
    """

    num_bodies: int = 0
    """
    Total number of moving body nodes (``N_B``) contained in the spanning-tree of the component subgraph.\n
    Note that ``num_bodies`` does not include the implicit world node with index `-1`.
    """

    num_joints: int = 0
    """
    Total number of joint edges (``N_J``) contained in the spanning-tree of the component subgraph.\n
    Note that ``num_joints`` includes the base joint edge connecting the component to the implicit world node with index `-1`.
    """

    num_tree_arcs: int = 0
    """
    Number of joint edges included in the spanning tree, i.e. arcs.\n
    Equals ``num_bodies`` for a spanning tree of a connected component.
    """

    num_tree_chords: int = 0
    """
    Number of joint edges not included in the spanning tree, i.e. chords.\n
    Equals ``num_joints - num_bodies`` indicating the number of independent loops in the component subgraph.
    """

    ###
    # Definition
    ###

    component: TopologyComponent | None = None
    """
    The source component subgraph associated with the spanning tree.\n
    Defaults to `None`, indicating that the component has not been assigned to the spanning tree.
    """

    root: NodeType | None = None
    """
    The root body node of the spanning tree, which is the unique moving body node directly connected to the implicit world node.\n
    Defaults to `None`, indicating that the root node has not been assigned to the spanning tree.
    """

    arcs: list[int] | None = None
    """
    List of joint "arc" edge indices included in the spanning tree, i.e. tree joints.\n
    Shape of `(num_tree_arcs,)` and type :class:`int`.
    Defaults to `None`, indicating that arc edges have not been generated.\n
    """

    chords: list[int] | None = None
    """
    List of joint "chord" edge indices not included in the spanning tree, i.e. loop joints.\n
    Shape of `(num_tree_chords,)` and type :class:`int`.
    Defaults to `None`, indicating that chord edges have not been generated.\n
    """

    ###
    # Parameterization
    ###

    predecessors: list[int] | None = None
    """
    List of per-joint edge predecessor body node indices contained in the spanning tree.\n
    Shape of `(num_joints,)` and type :class:`int`.\n
    Note that the predecessor of the root node is conventionally set to `-1`.\n
    Defaults to `None`, indicating that predecessor nodes have not been generated.
    """

    successors: list[int] | None = None
    """
    List of per-joint edge successor body node indices contained in the spanning tree.\n
    Shape of `(num_joints,)` and type :class:`int`.\n
    Note that the successor of the root node is conventionally set to `-1`.\n
    Defaults to `None`, indicating that successor nodes have not been generated.
    """

    parents: list[int] | None = None
    """
    List of body node indices identifying the parent of each body in the spanning tree.\n

    In the Featherstone conventions using the "regular numbering" scheme, the
    parent array is denoted by `λ` (`lambda`) and satisfies the properties:

    ``λ(i) = min(p(i), s(i)), and λ(i) < i, where 0 ≤ i ≤ NB - 1``.

    Note that the parent of the root node is conventionally set to `-1`.\n

    Shape of `(num_bodies,)` and type :class:`int`.
    """

    support: list[list[int]] | None = None
    """
    List of per-body tree edge indices identifying the support joints of each body in the spanning tree.\n

    In the Featherstone conventions using the "regular numbering" scheme, the support
    set is denoted by `κ` (`kappa`) and for any body ``i`` except the base, ``κ(i)``
    is the set of all tree joint edges on the path between body ``i`` and the base.
    It must also satisfy the property that ``j ∈ κ(i) ⇒ i ∈ v(j)``.

    Shape of `(num_bodies,)` and type :class:`list[int]`.
    """

    children: list[list[int]] | None = None
    """
    List of per-body tree node indices identifying the children bodies of each body in the spanning tree.\n

    In the Featherstone conventions using the "regular numbering" scheme, the children set is denoted by
    `µ` (`mu`) and for any body ``i``, including the base, ``µ(i)`` is the set of children of body ``i``.
    It must also satisfy the property that ``µ(i) ⊆ v(i)``.

    Shape of `(num_bodies,)` and type :class:`list[int]`.
    """

    subtree: list[list[int]] | None = None
    """
    List of per-body tree node indices identifying the subtree body nodes of each body in the spanning tree.\n

    In the Featherstone conventions using the "regular numbering" scheme, the subtree set is denoted by `v` (`nu`)
    and for any body ``i``, including the base, ``v(i)`` is the set of bodies in the subtree starting at body ``i``.
    ``v(i)`` can also be regarded as the set of all bodies supported by joint ``i``.
    It must also satisfy the properties:
    - ``µ(i) ⊆ v(i)``
    - ``j ∈ κ(i) ⇒ i ∈ v(j)``

    Shape of `(num_bodies,)` and type :class:`list[int]`.
    """


###
# Module Interfaces
###


class TopologyComponentParserBase:
    """
    A base class defining an interface for modules that parse the
    nodes and edges of a topology graph into a list of components.
    """

    @abstractmethod
    def parse_components(self, nodes: list[NodeType], edges: list[EdgeType], world: int) -> list[TopologyComponent]:
        """
        Parses the given nodes and edges into a list of components.

        Args:
            nodes:
                List of body node indices in the graph.
            edges:
                List of joint edges in the graph, where each edge is a tuple of the form
                `(joint type, joint_index (predecessor_body_index, child_body_index))`.
            world:
                The index of the implicit world node in the graph.

        Returns:
            A list of `TopologyComponent` objects representing the components of the graph.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TopologyComponentBaseSelectorBase:
    """
    A base class defining an interface for modules that select
    the base node and edge for a given component subgraph.
    """

    @abstractmethod
    def select_base(
        self,
        component: TopologyComponent,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> tuple[NodeType, EdgeType]:
        """Selects the base node and edge for the given component subgraph.

        Args:
            component: The component subgraph to select the base node and edge for.
            bodies: Optional list of body descriptors to aid in base node/edge selection.
            joints: Optional list of joint descriptors to aid in grounding node/edge selection.

        Returns:
            A tuple of the selected base node `NodeType` and edge `EdgeType`.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TopologySpanningTreeGeneratorBase:
    """
    A base class defining an interface for modules that
    generate spanning trees for a given component subgraph.
    """

    @abstractmethod
    def generate_spanning_trees(
        self,
        component: TopologyComponent,
        traversal_mode: SpanningTreeTraversal,
        max_candidates: int | None = None,
        roots: list[NodeType] | None = None,
    ) -> list[TopologySpanningTree]:
        """Generates a spanning tree for the given component subgraph.

        Args:
            component:
                The component subgraph to generate spanning tree candidates for.
            traversal_mode:
                The traversal mode to use for the spanning tree.
            max_candidates:
                Optional integer specifying the maximum number of spanning
                tree candidates to generate for the component.
            roots:
                Optional list of node indices to use as roots for the spanning tree.
                If not provided, the base or grounding nodes(s)/edge(s) will be used
                as roots, if they exist. Otherwise, brute-force enumeration of all
                possible roots may be performed.

        Returns:
            A list of `TopologySpanningTree` objects representing the source component subgraph.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TopologySpanningTreeSelectorBase:
    """
    A base class defining an interface for modules that select
    the best spanning tree from a list of spanning trees.
    """

    @abstractmethod
    def select_spanning_tree(
        self,
        candidates: list[TopologySpanningTree],
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> TopologySpanningTree:
        """Selects the best spanning tree from the given list of spanning trees.

        Args:
            candidates: List of candidate spanning trees to select from.
            bodies: Optional list of body descriptors to aid in spanning tree selection.
            joints: Optional list of joint descriptors to aid in spanning tree selection.

        Returns:
            The selected spanning tree.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TopologyGraphVisualizerBase:
    """
    A base class defining an interface for modules that render a topology
    graph, components and spanning trees for visualization.
    """

    @abstractmethod
    def render_graph(
        self,
        nodes: list[NodeType],
        edges: list[EdgeType],
        components: list[TopologyComponent],
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        joints: list[JointDescriptor] | None = None,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Renders the given topology graph and its constituent components.

        Args:
            nodes: The nodes of the topology graph to visualize.
            edges: The edges of the topology graph to visualize.
            components: The components of the topology graph to visualize.
            world_node: The index of the implicit world node in the graph.
            joints:
                Optional list of joint descriptors used to look up joint names for edge
                labels. When provided, ``joints[joint_index].name`` is included in each
                edge label; when omitted, the label uses the joint index and joint-type
                abbreviation only.
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def render_component_spanning_tree_candidates(
        self,
        component: TopologyComponent,
        candidates: list[TopologySpanningTree],
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Renders the spanning tree candidates for the given topology graph.

        Args:
            component: The component containing the spanning tree candidates to visualize.
            candidates: The list of spanning tree candidates to visualize.
            world_node: The index of the implicit world node in the graph.
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def render_component_spanning_tree(
        self,
        component: TopologyComponent,
        tree: TopologySpanningTree,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Renders the selected spanning tree for the given topology graph.

        Args:
            component: The component containing the selected spanning tree to visualize.
            tree: The selected spanning tree to visualize.
            world_node: The index of the implicit world node in the graph.
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
        """
        raise NotImplementedError("Subclasses must implement this method.")


###
# Interfaces
###


class TopologyGraph:
    """
    A container to represent a topological undirected graph `G`.
    """

    def __init__(
        self,
        # Graph attributes
        nodes: list[NodeType],
        edges: list[EdgeType] | None = None,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        # Pipeline modules
        component_parser: TopologyComponentParserBase | None = None,
        base_selector: TopologyComponentBaseSelectorBase | None = None,
        tree_generator: TopologySpanningTreeGeneratorBase | None = None,
        tree_selector: TopologySpanningTreeSelectorBase | None = None,
        graph_visualizer: TopologyGraphVisualizerBase | None = None,
        # Source model descriptors
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        # Parsing configurations
        tree_traversal_mode: SpanningTreeTraversal = "dfs",
        max_tree_candidates: int = 32,
        autoparse: bool = False,
    ):
        """
        Initializes the TopologyGraph object with the given nodes, edges, and
        optional modules for component parsing and spanning tree generation.

        Args:
            nodes:
                List of body node indices in the graph.
            edges:
                Optional list of joint edges in the graph, where each edge is a tuple of the
                form ``(joint_type, joint_index, (predecessor_body_index, successor_body_index))``.
            world_node:
                The index of the implicit world node in the graph. Must be a negative integer
                that is not contained in ``nodes``. Defaults to :data:`DEFAULT_WORLD_NODE_INDEX`.
            component_parser:
                Optional module to parse the graph nodes and edges into a list of components.
                Defaults to a shipped :class:`TopologyComponentParser` instance when not provided.
            base_selector:
                Optional module to select the base node and edge for component subgraphs whose
                base could not be auto-assigned during parsing. Required only when at least one
                parsed component lacks an auto-assigned base; see :meth:`select_component_bases`.
            tree_generator:
                Optional module to generate spanning tree candidates for each component
                subgraph. Required when calling :meth:`generate_spanning_trees` or :meth:`parse`.
            tree_selector:
                Optional module to select the best spanning tree from a list of candidates.
                Required when calling :meth:`select_spanning_trees` or :meth:`parse`.
            graph_visualizer:
                Optional module to render the topology graph, components, and spanning trees.
                Defaults to a shipped :class:`TopologyGraphVisualizer` instance when not provided.
            bodies:
                Optional list of rigid body descriptors associated with the graph nodes,
                forwarded to the base selector and tree selector modules to inform their
                heuristics. Can also be supplied at :meth:`parse` time.
            joints:
                Optional list of joint descriptors associated with the graph edges, forwarded
                to the base selector and tree selector modules to inform their heuristics. Can
                also be supplied at :meth:`parse` time.
            tree_traversal_mode:
                Default traversal mode used by :meth:`generate_spanning_trees`, one of ``"dfs"``
                or ``"bfs"``. Defaults to ``"dfs"``.
            max_tree_candidates:
                Default upper bound on the number of candidate spanning trees generated per
                component by :meth:`generate_spanning_trees`. Defaults to ``32``.
            autoparse:
                If ``True``, run the full :meth:`parse` pipeline (component parsing, base
                selection, spanning tree generation, and spanning tree selection) immediately
                after construction. Requires ``tree_generator`` and ``tree_selector`` to be
                provided. Defaults to ``False``.

        Raises:
            ValueError:
                If ``nodes`` is ``None``, if any node has an invalid format (non-integer or
                negative), if any edge has an invalid format, if ``world_node`` is not a
                negative integer, or if ``world_node`` is contained in ``nodes``. If
                ``autoparse=True``, also raises if any module required by the full pipeline
                is missing (see :meth:`parse`).
        """
        # Cache the input graph attributes that define the graph contents and structure
        self._nodes: list[NodeType] = nodes
        """
        List of body node indices contained in the graph.

        Each node is uniquely identified by its associated index in the range
        `[0, N-1]`, where `N` is the total number of nodes in the graph.

        `N` excludes the implicit world node with index `-1`, which
        is present in the graph if any node is connected to it.
        """
        self._edges: list[EdgeType] | None = edges
        """
        List of joint indices contained in the graph.

        Each edge is uniquely identified by its associated index in
        the range `[0, M-1]`, where `M` is the total number of edges.
        """
        self._world_node: int = world_node
        """
        The index of the implicit world node in the graph.\n
        Defaults to `-1`, which is the conventional index for the world body in Newton.
        """

        # Cache parsing configurations
        self._tree_traversal_mode: SpanningTreeTraversal = tree_traversal_mode
        """The traversal mode used for generating the spanning trees."""
        self._max_tree_candidates: int = max_tree_candidates
        """ The maximum number of candidate spanning trees to generate for each component of the graph."""

        # Validate the input graph attributes to ensure they are
        # consistent with the expected formats and conventions
        self._validate_inputs()

        # Store input modules for component parsing and spanning tree generation
        self._component_parser: TopologyComponentParserBase | None = component_parser
        """A module to parse the graph nodes and edges into a list of components."""
        self._base_selector: TopologyComponentBaseSelectorBase | None = base_selector
        """A module to select the base node for each component subgraph."""
        self._tree_generator: TopologySpanningTreeGeneratorBase | None = tree_generator
        """A module to generate a spanning tree for each component subgraph."""
        self._tree_selector: TopologySpanningTreeSelectorBase | None = tree_selector
        """A module to select the best spanning tree from the given list of spanning trees."""
        self._graph_visualizer: TopologyGraphVisualizerBase | None = graph_visualizer
        """A module to render the topology graph, components and spanning trees for visualization."""

        # Set default modules where shipped concrete defaults exist. Modules without a
        # default (`base_selector`, `tree_generator`, `tree_selector`) are left as `None`
        # and validated lazily in :meth:`parse` and the per-step methods so that callers
        # who only invoke a subset of the pipeline (e.g. :meth:`parse_components`) do not
        # need to provide every module up front.
        if self._component_parser is None:
            self._component_parser = TopologyComponentParser()
        if self._graph_visualizer is None:
            self._graph_visualizer = TopologyGraphVisualizer()

        # Declare and initialize internal caches for the source model descriptors
        self._bodies: list[RigidBodyDescriptor] | None = bodies
        self._joints: list[JointDescriptor] | None = joints

        # Declare derived attributes
        self._components: list[TopologyComponent] | None = None
        """
        A list of topology graph components, i.e. subgraphs.
        """
        self._candidates: list[list[TopologySpanningTree]] | None = None
        """
        A list of candidate spanning tree subgraphs corresponding to each component of the graph.
        """
        self._trees: list[TopologySpanningTree] | None = None
        """
        A list of selected spanning tree subgraphs corresponding to each component of the graph.
        """

        # If `autoparse` is True, automatically parse the graph nodes
        # and edges into components and generate spanning trees
        if autoparse:
            self.parse()

    ###
    # Properties
    ###

    @property
    def nodes(self) -> list[NodeType]:
        """Returns the list of body node indices contained in the graph."""
        return self._nodes

    @property
    def edges(self) -> list[EdgeType] | None:
        """Returns the list of joint edges contained in the graph."""
        return self._edges

    @property
    def world_node(self) -> int:
        """Returns the index of the implicit world node in the graph."""
        return self._world_node

    @property
    def components(self) -> list[TopologyComponent]:
        """Returns the list of components parsed from the graph."""
        if self._components is None:
            raise ValueError("Graph components have not been parsed yet.")
        return self._components

    @property
    def candidates(self) -> list[list[TopologySpanningTree]]:
        """Returns the list of candidate spanning trees generated for each component of the graph."""
        if self._candidates is None:
            raise ValueError("Candidate spanning trees have not been generated yet.")
        return self._candidates

    @property
    def trees(self) -> list[TopologySpanningTree]:
        """Returns the list of selected spanning trees for each component of the graph."""
        if self._trees is None:
            raise ValueError("Spanning trees have not been selected yet.")
        return self._trees

    ###
    # Operations
    ###

    def parse(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        # TODO: Add option to specify the component base node/edge indices and skip the base selector module
    ) -> None:
        """
        Parses the graph nodes and edges into a list of components, and generates a spanning tree for each component.

        Args:
            bodies: Optional list of rigid body descriptors to aid in component parsing.
            joints: Optional list of joint descriptors to aid in component parsing.

        Raises:
            ValueError:
                If any module required by the full pipeline (component parser, tree generator,
                tree selector) is missing, if graph parsing or spanning tree generation fails,
                or if the graph attributes are invalid. All missing required modules are
                reported in a single error rather than failing partway through the pipeline.
        """
        # Validate up front that every module required by the full pipeline is available,
        # so that the user can fix them in one round instead of failing in step N of M.
        # The base selector is treated as required only if at least one component lacks an
        # auto-assigned base after parsing — see :meth:`select_component_bases`.
        missing = [
            name
            for name, mod in (
                ("component_parser", self._component_parser),
                ("tree_generator", self._tree_generator),
                ("tree_selector", self._tree_selector),
            )
            if mod is None
        ]
        if missing:
            raise ValueError(
                f"Cannot run full topology parsing pipeline: missing required module(s): "
                f"{', '.join(missing)}. Provide them via the `TopologyGraph` constructor."
            )

        # Use the provided body and joint descriptors for parsing if given,
        # otherwise use the cached descriptors from initialization
        _bodies = bodies if bodies is not None else self._bodies
        _joints = joints if joints is not None else self._joints

        # Parse the graph nodes and edges into and run the topology discovery
        # pipeline to generate spanning trees for each component of the graph
        self.parse_components()
        self.select_component_bases(bodies=_bodies, joints=_joints)
        self.generate_spanning_trees()
        self.select_spanning_trees(bodies=_bodies, joints=_joints)

    def parse_components(self) -> list[TopologyComponent]:
        """
        Parses the graph nodes and edges into a list of components using the provided component parser module.

        Returns:
            A list of `TopologyComponent` objects representing the components of the graph.

        Raises:
            ValueError:
                If graph parsing fails, i.e. if the component parser
                returns `None` or if the graph attributes are invalid.
        """
        if self._component_parser is None:
            raise ValueError("No component parser module provided, cannot parse graph components.")
        self._components = self._component_parser.parse_components(
            nodes=self._nodes, edges=self._edges, world=self._world_node
        )
        if self._components is None:
            raise ValueError("Graph component parsing failed.")
        return self._components

    def select_component_bases(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> None:
        """
        Selects the base node and edge for each component of the graph using the provided base selector module.

        The component parser may already auto-assign a base node/edge for components with a
        single grounding edge, or with a single 6-DoF FREE joint among multiple grounding
        edges. This method only invokes the base selector for components that still lack a
        base after parsing, so a missing base selector is only an error when at least one
        component requires one.

        Args:
            bodies: Optional list of body descriptors to aid in base node/edge selection.
            joints: Optional list of joint descriptors to aid in grounding node/edge selection.

        Raises:
            ValueError:
                If components have not been parsed yet, if any component still lacks a base
                node/edge but no base selector module was provided, or if the base selector
                returns `None` for a component.
        """
        # Ensure that the graph components are generated before
        # selecting the base node and edge for each component
        if self._components is None:
            raise ValueError("Graph components must be generated before base node/edge selection.")

        # Determine which components still need a base assignment after parsing
        components_needing_base = [c for c in self._components if c.base_edge is None]
        if not components_needing_base:
            return

        # Components remain that need a base — a base selector module is now required
        if self._base_selector is None:
            raise ValueError(
                f"No base selector module provided, but {len(components_needing_base)} component(s) "
                f"still lack a base node/edge after parsing. Provide a `base_selector` module via "
                f"the `TopologyGraph` constructor."
            )

        # Use the provided body and joint descriptors for parsing if given,
        # otherwise use the cached descriptors from initialization
        _bodies = bodies if bodies is not None else self._bodies
        _joints = joints if joints is not None else self._joints

        # Run base selection for components that need it
        for component in components_needing_base:
            base_node, base_edge = self._base_selector.select_base(component=component, bodies=_bodies, joints=_joints)
            if base_node is None or base_edge is None:
                raise ValueError(f"Base node/edge selection failed for component: {component}")
            # TODO: Should this be done in-place in the base selector module instead of here?
            component.base_node = base_node
            component.base_edge = base_edge
            # The component is now connected to the world via the assigned base edge.
            # `is_connected` was set during parsing based on the *original* graph topology,
            # so it must be refreshed for components that were initially isolated.
            component.is_connected = True

    def generate_spanning_trees(
        self,
        traversal_mode: SpanningTreeTraversal = "dfs",
        max_candidates: int | None = None,
        roots: list[NodeType] | None = None,
    ) -> list[list[TopologySpanningTree]]:
        """
        Generates a spanning tree for each component of the graph using the provided tree generator module.

        Args:
            max_candidates:
                Optional integer specifying the maximum number of spanning
                tree candidates to generate for each component of the graph.
                This overrides the maximum number of candidates specified
                in the object constructor.

        Returns:
            A list of lists of `TopologySpanningTree` objects representing
            the candidate spanning trees for each component of the graph.

        Raises:
            ValueError:
                If spanning tree generation fails, i.e. if the tree generator
                returns `None` or if the graph attributes are invalid.
        """
        # Ensure that the graph components are generated before
        # generating spanning trees for each component of the graph
        if self._components is None:
            raise ValueError("Graph components must be generated before spanning tree generation.")

        # Ensure that a tree generator module is provided
        if self._tree_generator is None:
            raise ValueError("No tree generator module provided, cannot generate spanning trees.")

        # If a maximum number of candidates is provided, use it to limit the number of candidates generated
        if max_candidates is not None:
            if not isinstance(max_candidates, int):
                raise TypeError("Maximum number of spanning tree candidates must be an integer.")
            if max_candidates <= 0:
                raise ValueError("Maximum number of spanning tree candidates must be greater than 0.")
            _max_candidates = max_candidates
        else:
            _max_candidates = self._max_tree_candidates

        # Ensure that the traversal mode is valid. `SpanningTreeTraversal` is a
        # `typing.Literal[...]` alias and is not iterable at runtime, so we
        # validate against an explicit tuple of supported values.
        if traversal_mode not in ("dfs", "bfs"):
            raise ValueError(f"Invalid traversal mode: {traversal_mode!r}")
        _traversal_mode = traversal_mode

        # Proceed with generating spanning tree candidates for each component of the graph using
        # the provided tree generator module, if given, otherwise skip this and rely on
        # the user to generate the spanning trees directly from the components
        candidates = []
        for component in self._components:
            trees = self._tree_generator.generate_spanning_trees(
                component=component,
                traversal_mode=_traversal_mode,
                max_candidates=_max_candidates,
                roots=roots,
            )
            if trees is None:
                raise ValueError(f"Spanning tree generation failed for component: {component}")
            candidates.append(trees)
        if len(candidates) != len(self._components):
            raise ValueError("Spanning tree generation failed for some components.")
        self._candidates = candidates
        return self._candidates

    def select_spanning_trees(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> list[TopologySpanningTree]:
        """
        Selects the best spanning tree for each component of the graph from the
        generated candidate spanning trees using the provided tree selector module.

        Args:
            bodies: Optional list of body descriptors to aid in spanning tree selection.
            joints: Optional list of joint descriptors to aid in spanning tree selection.

        Returns:
            A list of `TopologySpanningTree` objects representing the
            selected spanning tree for each component of the graph.

        Raises:
            ValueError:
                If candidate spanning trees have not been generated yet, if no tree selector
                module was provided, or if the tree selector returns `None` for a component.
        """
        # Ensure that the candidate spanning trees are generated before
        # selecting the best spanning tree for each component of the graph
        if self._candidates is None:
            raise ValueError("Candidate spanning trees must be generated before spanning tree selection.")

        # A tree selector module is required to populate the per-component selected
        # spanning tree list, since there is no shipped default selection heuristic.
        if self._tree_selector is None:
            raise ValueError(
                "No tree selector module provided, cannot select spanning trees. Provide a "
                "`tree_selector` module via the `TopologyGraph` constructor."
            )

        # Use the provided body and joint descriptors for parsing if given,
        # otherwise use the cached descriptors from initialization
        _bodies = bodies if bodies is not None else self._bodies
        _joints = joints if joints is not None else self._joints

        # Run tree selection for every component
        self._trees = []
        for trees in self._candidates:
            tree = self._tree_selector.select_spanning_tree(candidates=trees, bodies=_bodies, joints=_joints)
            if tree is None:
                raise ValueError("Spanning tree selection failed for component.")
            self._trees.append(tree)
        return self._trees

    ###
    # Exports
    ###

    def export_descriptors(self) -> list[TopologyDescriptor]:
        """
        Exports the selected spanning trees for each component of the graph as a list of `TopologyDescriptor` objects.

        Returns:
            A list of `TopologyDescriptor` objects representing the selected spanning trees for each component of the graph.

        Raises:
            ValueError:
                If the selected spanning trees are not generated or if the graph attributes are invalid.
        """
        if self._trees is None:
            raise ValueError("Selected spanning trees must be generated before exporting topology descriptors.")
        descriptors = []
        for i, tree in enumerate(self._trees):
            # Synthesize a stable per-tree name from the tree's root body index when
            # available, falling back to the per-component index otherwise. The
            # `Descriptor` base class requires a non-empty `name` at construction.
            name = f"tree_at_root_{tree.root}" if tree.root is not None else f"tree_{i}"
            descriptors.append(TopologyDescriptor(name=name, tree=tree))
        return descriptors

    ###
    # Visualization
    ###

    def render_graph(
        self,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """
        Renders the graph and its constituent components using the configured graph visualizer module.

        Args:
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
        """
        if self._graph_visualizer is None:
            raise ValueError("No graph visualizer module provided, cannot render graph.")
        if self._components is None:
            raise ValueError("Graph components must be generated before rendering.")
        self._graph_visualizer.render_graph(
            nodes=self._nodes,
            edges=self._edges if self._edges is not None else [],
            components=self._components,
            world_node=self._world_node,
            joints=self._joints,
            figsize=figsize,
            path=path,
            show=show,
        )

    def render_spanning_tree_candidates(
        self,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """
        Renders the candidate spanning trees for each component of the graph using the configured graph visualizer module.

        Args:
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
        """
        if self._graph_visualizer is None:
            raise ValueError("No graph visualizer module provided, cannot render spanning tree candidates.")
        if self._components is None:
            raise ValueError("Graph components must be generated before rendering.")
        if self._candidates is None:
            raise ValueError("Candidate spanning trees must be generated before rendering.")
        for component, candidates in zip(self._components, self._candidates, strict=True):
            self._graph_visualizer.render_component_spanning_tree_candidates(
                component=component,
                candidates=candidates,
                world_node=self._world_node,
                figsize=figsize,
                path=path,
                show=show,
            )

    def render_spanning_trees(
        self,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """
        Renders the selected spanning trees for each component of the graph using the configured graph visualizer module.

        Args:
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
        """
        if self._graph_visualizer is None:
            raise ValueError("No graph visualizer module provided, cannot render spanning trees.")
        if self._components is None:
            raise ValueError("Graph components must be generated before rendering.")
        if self._trees is None:
            raise ValueError("Selected spanning trees must be generated before rendering.")
        for component, tree in zip(self._components, self._trees, strict=True):
            self._graph_visualizer.render_component_spanning_tree(
                component=component, tree=tree, world_node=self._world_node, figsize=figsize, path=path, show=show
            )

    ###
    # Internals
    ###

    @staticmethod
    def _assert_node_valid(node: NodeType) -> None:
        """
        Asserts that the given node is in the correct format.

        Raises:
            ValueError:
                If the node is not an integer representing a body index.
        """
        # First check that the node is an integer representing a body index in the graph
        if not isinstance(node, int):
            raise ValueError(f"Graph node `{node}` is not an integer representing a body index.")
        # Then check that the node index is non-negative, as negative indices are reserved for the implicit world node
        if node < 0:
            raise ValueError(
                f"Graph node `{node}` is a negative integer, which is reserved for the implicit world node."
            )

    @staticmethod
    def _assert_edge_valid(edge: EdgeType) -> None:
        """
        Asserts that the given edge is in the correct format.

        Raises:
            ValueError:
                If the edge is not a tuple of the form ``(type, jid, (pbid, sbid))`` or if any of the elements are not integers.
        """
        # First check that the edge is a tuple with the correct format
        if not isinstance(edge, tuple) or len(edge) != 3:
            raise ValueError(f"Graph edge `{edge}` is not in the correct format (type, jid, (pbid, sbid)).")
        # Then check that the each element of the edge is in the correct format and type
        joint_type, joint_index, body_pair = edge
        if not isinstance(joint_type, int):
            raise ValueError(f"Graph edge `{edge}` has a non-integer joint type.")
        if not isinstance(joint_index, int):
            raise ValueError(f"Graph edge `{edge}` has a non-integer joint index.")
        if not isinstance(body_pair, tuple) or len(body_pair) != 2:
            raise ValueError(f"Graph edge `{edge}` has an invalid body pair format.")
        if not all(isinstance(b, int) for b in body_pair):
            raise ValueError(f"Graph edge `{edge}` has non-integer body indices.")

    @staticmethod
    def _assert_world_node_valid(world_node: int, nodes: list[NodeType]) -> None:
        """
        Asserts that the given world node index is in the correct format.

        Raises:
            ValueError:
                If the world node index is not an integer or if it is a non-negative integer included in the nodes list.
        """
        if not isinstance(world_node, int):
            raise ValueError(f"World index `{world_node}` is not an integer representing the world node index.")
        if world_node >= 0:
            raise ValueError(
                f"World index `{world_node}` is a non-negative integer, but it should be a negative integer representing the implicit world node."
            )
        # Ensure that the world index is not included in the nodes list
        if world_node in nodes:
            raise ValueError(f"World index `{world_node}` should not be included in the nodes list.")

    def _validate_inputs(self):
        """
        Checks that the input graph attributes are valid and
        consistent with the expected formats and conventions.

        Raises:
            ValueError:
                If nodes are not provided, if edges are not in the correct format, if the world
                index is not an integer, or if the world index is included in the nodes list.
        """
        # Ensure that nodes are provided, as they are necessary to define the graph contents
        if self._nodes is None:
            raise ValueError("Nodes must be provided to initialize the graph.")

        # Edges are optional, as a graph can consist of isolated nodes
        # only, but if provided, they must be in the correct format
        if self._edges is None:
            self._edges = []

        # Ensure that nodes are in the correct format
        for node in self._nodes:
            self._assert_node_valid(node)

        # Ensure edges are in the correct format
        for edge in self._edges:
            self._assert_edge_valid(edge)

        # Ensure that the world index is in the correct format
        self._assert_world_node_valid(self._world_node, self._nodes)


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
    # Operations
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
        """
        pass  # TODO: IMPLEMENT

    @staticmethod
    def from_newton(model: Model) -> TopologyModel:
        """
        Generates a :class:`TopologyModel` by parsing the body, joint and articulation arrays of a :class:`Model`.

        Args:
            model: A :class:`Model` instance containing the body, joint, and articulation arrays.

        Returns:
            A :class:`TopologyModel` instance containing the data from the provided model.
        """
        pass  # TODO: IMPLEMENT


###
# Backends
###


class TopologyComponentParser(TopologyComponentParserBase):
    """
    A default implementation of the `TopologyComponentParserBase` that parses
    the graph nodes and edges into components using a union-find / disjoint set
    data structure to efficiently group connected nodes into components, while
    also classifying them based on their connectivity to the implicit world node.
    """

    @override
    def parse_components(
        self,
        nodes: list[NodeType],
        edges: list[EdgeType],
        world: int = DEFAULT_WORLD_NODE_INDEX,
    ) -> list[TopologyComponent]:
        """
        Parses the given nodes and edges into a list of components.

        Args:
            nodes:
                List of body node indices in the graph.
            edges:
                List of joint edges in the graph, where each edge is a tuple of the form
                `(joint type, joint_index (predecessor_body_index, child_body_index))`.
            world:
                The index of the implicit world node in the graph.

        Returns:
            A list of `TopologyComponent` objects representing the components of the graph.

        Note:
            - The world node is not included in the list of nodes, but it is
              considered when determining the connectivity of components.

            - Components are classified as `islands` if they contain more than one node, and as
              `orphans` if they contain exactly one node. Each component is further classified
              as `connected` if it has at least one edge connecting it to the world node, and
              as `isolated` otherwise.
        """
        # Deduplicate edges and sort them by joint index so that the order of derived
        # structures (component edge lists, parent arrays, traversal orders, etc.) is
        # deterministic across graphs that share structure but differ in joint labelling.
        edges = sorted(set[EdgeType](edges), key=lambda e: e[1])
        msg.debug("edges: %s", edges)

        # Keep only the real (non-world) body nodes, deduplicated
        nodes = {n for n in nodes if n != world}
        msg.debug("nodes: %s", sorted(nodes))

        # Union-Find / Disjoint set
        parent = {n: n for n in nodes}
        rank = dict.fromkeys(nodes, 0)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        # Track adjacency for classification
        neighbors: dict[int, set[int]] = {n: set() for n in nodes}

        # Build subgraphs among non-world nodes only
        for _t, _j, (u, v) in edges:
            if u == world and v in nodes:
                neighbors[v].add(world)
            elif v == world and u in nodes:
                neighbors[u].add(world)
            elif u in nodes and v in nodes:
                neighbors[u].add(v)
                neighbors[v].add(u)
                union(u, v)

        # Gather connected subgraphs among body nodes
        component_node_map = defaultdict(list)
        for n in nodes:
            component_node_map[find(n)].append(n)
        component_nodes = [sorted(comp) for comp in component_node_map.values()]

        # Classify components as `islands` (size > 1) vs `orphans` (size == 1)
        islands = [nodes for nodes in component_nodes if len(nodes) > 1]
        orphans = [nodes for nodes in component_nodes if len(nodes) == 1]

        # Sort islands by descending size and orphans by index, then concatenate them
        islands.sort(key=len, reverse=True)
        orphans.sort(key=lambda c: c[0])
        component_nodes = islands + orphans
        msg.debug("component_nodes: %s", component_nodes)

        # Construct component objects with their properties
        components = []
        for comp_nodes in component_nodes:
            # Use a set for O(1) membership checks during the per-component edge sweep
            comp_nodes_set = set(comp_nodes)

            # Collect edges for the component and check
            # for connections to the implicit world node
            comp_edges: list[EdgeType] = []
            comp_grounding_edges: list[EdgeType] = []
            for e in edges:
                _t, _j, (u, v) = e
                is_comp_edge = u in comp_nodes_set and v in comp_nodes_set
                is_comp_ground_edge = (u in comp_nodes_set and v == world) or (v in comp_nodes_set and u == world)
                if is_comp_edge or is_comp_ground_edge:
                    comp_edges.append(e)
                # NOTE: We add all nodes/edges connecting the component to the world
                # as grounding nodes/edges, without yet distinguishing between base
                # vs grounding edges, as this will be handled later in the base
                # selection step after spanning tree generation.
                if is_comp_ground_edge:
                    comp_grounding_edges.append(e)

            # Assign the base node automatically if a single grounding edge is present
            comp_base_node: NodeType | None = None
            comp_base_edge: EdgeType | None = None
            if len(comp_grounding_edges) == 1:
                # Assign the unique grounding node as the base node,
                # and the unique grounding edge as the base edge
                comp_base_edge = comp_grounding_edges[0]
                comp_base_node = next(n for n in comp_base_edge[2] if n != world)
                # Drop the promoted edge from the grounding list — base and grounding
                # edges are mutually exclusive per the topology conventions
                comp_grounding_edges = []

            # If multiple grounding edges are present, and only one of them is a 6-DoF FREE joint,
            # then assign the node connected to the FREE joint as the base node, and remove it
            # from the grounding lists. If more than one of the grounding edges are FREE joints,
            # then raise an error, as this violates the modelling conventions.
            elif len(comp_grounding_edges) > 1:
                free_grounding_edges = [e for e in comp_grounding_edges if e[0] == JointType.FREE]
                if len(free_grounding_edges) == 1:
                    comp_base_edge = free_grounding_edges[0]
                    comp_base_node = next(n for n in comp_base_edge[2] if n != world)
                    comp_grounding_edges.remove(comp_base_edge)
                elif len(free_grounding_edges) > 1:
                    raise ValueError(
                        f"Component with nodes `{comp_nodes}` has multiple grounding edges `{comp_grounding_edges}` "
                        f"with more than one 6-DoF FREE joint, which violates modelling conventions."
                    )

            # Recompute the grounding-node set from the *final* grounding-edge list. This is the
            # source of truth for `ground_nodes` and avoids the duplicate-removal hazard that
            # arises when a body has multiple grounding edges (e.g. a Stewart platform leg).
            comp_grounding_nodes = sorted({n for _, _, pair in comp_grounding_edges for n in pair if n != world})

            # Add the new component object in the list of graph components
            components.append(
                TopologyComponent(
                    nodes=comp_nodes,
                    edges=comp_edges,
                    ground_nodes=comp_grounding_nodes,
                    ground_edges=comp_grounding_edges,
                    base_node=comp_base_node,
                    base_edge=comp_base_edge,
                    is_island=len(comp_nodes) > 1,
                    is_connected=comp_base_edge is not None or len(comp_grounding_edges) > 0,
                )
            )

        # Return the list of graph components
        msg.debug("components: %s", components)
        return components


class TopologyGraphVisualizer(TopologyGraphVisualizerBase):
    """
    A default implementation of the `TopologyGraphVisualizerBase`
    that renders a topology graph using networkx and matplotlib.

    Edge labels show a short abbreviation of the joint type plus the joint index.
    Because the integer in :data:`EdgeType` ``[0]`` (the joint type) is ambiguous —
    Newton's :class:`JointType` and Kamino's :class:`JointDoFType` use overlapping
    integer values (e.g. ``0`` is ``JointType.PRISMATIC`` but ``JointDoFType.FREE``) —
    the visualizer must be told which enum the integer values refer to. The default
    is :class:`JointDoFType` because this module lives in the Kamino subpackage and
    edges built from :class:`ModelBuilderKamino` use ``joint.dof_type.value``.
    """

    _PALETTE: tuple[str, ...] = (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    )
    """Cyclic palette used to color island components."""

    _JOINT_TYPE_ABBR: ClassVar[dict[int, str]] = {
        JointType.PRISMATIC: "PRISM",
        JointType.REVOLUTE: "REVL",
        JointType.BALL: "BAL",
        JointType.FIXED: "FIXD",
        JointType.FREE: "FREE",
        JointType.DISTANCE: "DIST",
        JointType.D6: "D6",
        JointType.CABLE: "CABL",
    }
    """Abbreviation table for Newton's :class:`JointType` integer values."""

    _JOINT_DOF_TYPE_ABBR: ClassVar[dict[int, str]] = {
        JointDoFType.FREE: "FREE",
        JointDoFType.REVOLUTE: "REVL",
        JointDoFType.PRISMATIC: "PRISM",
        JointDoFType.CYLINDRICAL: "CYLN",
        JointDoFType.UNIVERSAL: "UNIV",
        JointDoFType.SPHERICAL: "SPHE",
        JointDoFType.GIMBAL: "GIMB",
        JointDoFType.CARTESIAN: "CART",
        JointDoFType.FIXED: "FIXD",
    }
    """Abbreviation table for Kamino's :class:`JointDoFType` integer values."""

    def __init__(self, joint_type_enum: type[IntEnum] = JointDoFType):
        """Initializes the visualizer with the chosen joint-type enum.

        Args:
            joint_type_enum:
                The enum used to interpret the integer joint type stored in :data:`EdgeType` ``[0]``.
                Must be either :class:`JointType` (Newton's joint type enum) or :class:`JointDoFType`
                (Kamino's joint DoF type enum). Defaults to :class:`JointDoFType` because edges
                produced by :class:`ModelBuilderKamino` use ``joint.dof_type.value``.

        Raises:
            ValueError: If ``joint_type_enum`` is neither :class:`JointType` nor :class:`JointDoFType`.
        """
        if joint_type_enum is JointType:
            self._abbr_table = self._JOINT_TYPE_ABBR
        elif joint_type_enum is JointDoFType:
            self._abbr_table = self._JOINT_DOF_TYPE_ABBR
        else:
            raise ValueError(
                f"Unsupported `joint_type_enum={joint_type_enum!r}`: must be either "
                f"`JointType` (Newton) or `JointDoFType` (Kamino)."
            )
        self._joint_type_enum = joint_type_enum

    @override
    def render_graph(
        self,
        nodes: list[NodeType],
        edges: list[EdgeType],
        components: list[TopologyComponent],
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        joints: list[JointDescriptor] | None = None,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """
        Renders the given topology graph using networkx and matplotlib.

        The world node is placed at the global origin and components are packed radially
        around it. Each component receives its own per-component sublayout: a rooted
        layered layout when a base node is available (so the base anchors closest to the
        world), or a deterministic Kamada-Kawai layout otherwise. Edges are styled by
        structural role (base, grounding, internal) and labeled with a joint-type
        abbreviation and joint index.

        Args:
            nodes: A list of `NodeType` instances representing the nodes in the topology graph.
            edges: A list of `EdgeType` instances representing the edges in the topology graph.
            components: A list of `TopologyComponent` instances representing the components in the topology graph.
            world_node: The index of the world node in the topology graph.
            joints:
                Optional list of joint descriptors used to look up joint names for edge labels.
                When provided, an edge label has the form ``f"{name}_{index}_{type}"``; otherwise
                it falls back to ``f"{index}_{type}"``. ``joints`` is expected to be indexable
                by the global joint index stored in :data:`EdgeType` ``[1]``: out-of-range
                indices and missing/empty names are tolerated and silently fall back to the
                index-only label.
            figsize: Optional tuple specifying the figure size for the plot.
            path: Optional string specifying the file path to save the plot.
            show: Boolean indicating whether to display the plot.

        Raises:
            ImportError: If :mod:`networkx` or :mod:`matplotlib` are not installed.
        """
        try:
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx and matplotlib are required for rendering the topology graph. "
                "Please install them with `pip install networkx matplotlib`."
            ) from e

        if figsize is None:
            figsize = (12, 12)

        # The world node is only drawn if it appears as an endpoint of any edge,
        # which mirrors the modelling convention that an unreferenced world node
        # should not visually clutter the graph.
        world_in_graph = any(world_node in pair for _, _, pair in edges)

        # Compute per-component local layouts and their bounding-circle radii. Each
        # entry is `(local_pos, local_radius, is_rooted, base_node)`.
        comp_layouts: list[tuple[dict[NodeType, tuple[float, float]], float, bool, NodeType | None]] = []
        for comp in components:
            local_pos, local_radius, is_rooted = self._layout_component(comp, world_node)
            comp_layouts.append((local_pos, local_radius, is_rooted, comp.base_node))

        # Pack components radially around the world node. Returns a global position
        # dict keyed by node index, including the world node when applicable.
        pos = self._pack_components(comp_layouts, world_node, world_in_graph)

        # Build a single nx.Graph for drawing. We use a plain Graph (not MultiGraph)
        # because parallel edges between the same body pair are rare in topology
        # graphs and would only slightly clutter the labels — by contrast, sticking
        # to Graph keeps the per-edge styling logic simple.
        G = nx.Graph()
        G.add_nodes_from(nodes)
        if world_in_graph:
            G.add_node(world_node)
        for _t, _j, (u, v) in edges:
            G.add_edge(u, v)

        # Classify each input edge into one of three role buckets via a per-component
        # scan. We compare on `(joint_type, joint_index)` rather than the full tuple
        # so the comparison is cheap and unambiguous regardless of edge ordering.
        base_keys: set[tuple[int, int]] = set()
        ground_keys: set[tuple[int, int]] = set()
        for comp in components:
            if comp.base_edge is not None:
                base_keys.add((comp.base_edge[0], comp.base_edge[1]))
            if comp.ground_edges is not None:
                for e in comp.ground_edges:
                    ground_keys.add((e[0], e[1]))

        base_edges_uv: list[tuple[NodeType, NodeType]] = []
        ground_edges_uv: list[tuple[NodeType, NodeType]] = []
        internal_edges_uv: list[tuple[NodeType, NodeType]] = []
        edge_label_map: dict[tuple[NodeType, NodeType], str] = {}
        for jt, jid, (u, v) in edges:
            uv = (u, v)
            key = (jt, jid)
            if key in base_keys:
                base_edges_uv.append(uv)
            elif key in ground_keys:
                ground_edges_uv.append(uv)
            else:
                internal_edges_uv.append(uv)
            edge_label_map[uv] = self._build_edge_label(jt, jid, joints)

        # Build per-node styling. Defaults are overwritten by component- and role-
        # specific styling further below; this ordering matters because the base
        # node should win over the generic island styling.
        node_color_map: dict[NodeType, str] = {}
        node_size_map: dict[NodeType, int] = {}
        node_edge_color_map: dict[NodeType, str] = {}
        node_linewidth_map: dict[NodeType, float] = {}

        for n in G.nodes:
            node_color_map[n] = "lightgray"
            node_size_map[n] = 600
            node_edge_color_map[n] = "black"
            node_linewidth_map[n] = 1.0

        if world_in_graph:
            node_color_map[world_node] = "black"
            node_size_map[world_node] = 900
            node_edge_color_map[world_node] = "black"
            node_linewidth_map[world_node] = 1.5

        island_color_map: dict[int, str] = {}
        island_index = 0
        for comp in components:
            if comp.is_island:
                color = self._PALETTE[island_index % len(self._PALETTE)]
                island_color_map[island_index] = color
                for n in comp.nodes:
                    node_color_map[n] = color
                    node_size_map[n] = 700
                island_index += 1
            else:
                # Single-node component: connected vs isolated orphan
                n = comp.nodes[0]
                if comp.is_connected:
                    node_color_map[n] = "grey"
                else:
                    node_color_map[n] = "white"
                node_size_map[n] = 700

        # Base nodes get a thicker border to mark them as the local root, while
        # keeping their component fill so they remain visually grouped.
        for comp in components:
            if comp.base_node is not None:
                node_linewidth_map[comp.base_node] = 2.5

        # Ensure every node referenced in `pos` has styling — defensive against
        # mismatches between `nodes` and the per-component node lists.
        for n in G.nodes:
            node_color_map.setdefault(n, "lightgray")
            node_size_map.setdefault(n, 600)
            node_edge_color_map.setdefault(n, "black")
            node_linewidth_map.setdefault(n, 1.0)

        # Begin drawing
        fig, ax = plt.subplots(figsize=figsize)

        # Edges first, behind the nodes
        if internal_edges_uv:
            nx.draw_networkx_edges(G, pos, edgelist=internal_edges_uv, width=1.5, edge_color="0.55", ax=ax)
        if ground_edges_uv:
            nx.draw_networkx_edges(
                G, pos, edgelist=ground_edges_uv, width=1.8, style="dashed", edge_color="0.35", ax=ax
            )
        if base_edges_uv:
            nx.draw_networkx_edges(G, pos, edgelist=base_edges_uv, width=2.5, edge_color="black", ax=ax)

        # Nodes — `draw_networkx_nodes` requires per-node lists to be parallel to
        # the supplied `nodelist`, so we iterate in a stable node order.
        ordered_nodes = list(G.nodes)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=ordered_nodes,
            node_color=[node_color_map[n] for n in ordered_nodes],
            node_size=[node_size_map[n] for n in ordered_nodes],
            edgecolors=[node_edge_color_map[n] for n in ordered_nodes],
            linewidths=[node_linewidth_map[n] for n in ordered_nodes],
            ax=ax,
        )

        # Node labels — keep the world node label readable on the dark fill
        node_labels = {n: ("W" if n == world_node else str(n)) for n in G.nodes}
        # Draw world label in white, everything else in black, by splitting the call
        if world_in_graph:
            nx.draw_networkx_labels(G, pos, labels={world_node: "W"}, font_size=10, font_color="white", ax=ax)
            nx.draw_networkx_labels(
                G,
                pos,
                labels={n: lbl for n, lbl in node_labels.items() if n != world_node},
                font_size=10,
                font_color="black",
                ax=ax,
            )
        else:
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="black", ax=ax)

        # Edge labels with a small white bounding box so they remain legible
        # against the slightly grey internal-edge color.
        if edge_label_map:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_label_map,
                font_size=8,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
                ax=ax,
            )

        # Legend
        legend_handles: list = []
        if world_in_graph:
            legend_handles.append(mpatches.Patch(color="black", label=f"world ({world_node})"))
        legend_handles.append(mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.5, label="base node"))
        legend_handles.append(mpatches.Patch(color="grey", label="orphan - connected"))
        legend_handles.append(mpatches.Patch(facecolor="white", edgecolor="black", label="orphan - isolated"))
        for i in range(island_index):
            legend_handles.append(mpatches.Patch(color=island_color_map[i], label=f"island {i}"))
        legend_handles.append(mlines.Line2D([], [], color="black", linewidth=2.5, label="base edge"))
        legend_handles.append(
            mlines.Line2D([], [], color="0.35", linewidth=1.8, linestyle="--", label="grounding edge")
        )
        legend_handles.append(mlines.Line2D([], [], color="0.55", linewidth=1.5, label="internal edge"))
        ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)

        ax.set_axis_off()
        fig.tight_layout()

        if path is not None:
            fig.savefig(path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _layout_component(
        component: TopologyComponent,
        world_node: int,
    ) -> tuple[dict[NodeType, tuple[float, float]], float, bool]:
        """Computes a per-component layout in component-local coordinates.

        Args:
            component: The component subgraph to lay out.
            world_node: The index of the implicit world node, used to skip world endpoints.

        Returns:
            A tuple ``(local_pos, local_radius, is_rooted)`` where:

            - ``local_pos`` maps each node in the component to a local ``(x, y)`` position.
            - ``local_radius`` is the bounding-circle radius of the layout in local
              coordinates (with a small floor so single-node components still take a slot).
            - ``is_rooted`` is True when the layout grows along the local ``+x`` axis from
              the base node (``base_node`` first, at the local origin), enabling the radial
              packer to anchor the base toward the world. False when the layout has no
              preferred orientation.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for rendering the topology graph. Please install it with `pip install networkx`."
            ) from e

        comp_nodes = list(component.nodes) if component.nodes else []
        if not comp_nodes:
            return {}, 0.0, False

        # Orphan: a single-node component lays out trivially at the local origin.
        if len(comp_nodes) == 1:
            return {comp_nodes[0]: (0.0, 0.0)}, 0.0, False

        # Build the undirected internal subgraph (skip world endpoints).
        internal_pairs: list[tuple[NodeType, NodeType]] = []
        if component.edges:
            for _t, _j, (u, v) in component.edges:
                if u == world_node or v == world_node:
                    continue
                internal_pairs.append((u, v))

        if component.base_node is not None:
            local_pos = TopologyGraphVisualizer._rooted_layered_layout(
                comp_nodes, internal_pairs, root=component.base_node
            )
            is_rooted = True
        else:
            sub = nx.Graph()
            sub.add_nodes_from(comp_nodes)
            sub.add_edges_from(internal_pairs)
            try:
                local_pos = nx.kamada_kawai_layout(sub)
            except Exception:
                # Kamada-Kawai requires a connected graph and at least 2 nodes — fall
                # back to a deterministic spring layout if the heuristic fails.
                local_pos = nx.spring_layout(sub, seed=42)
            is_rooted = False

        # Bounding-circle radius (with a small floor so degenerate layouts still
        # take an angular slot during radial packing).
        max_r = 0.0
        for x, y in local_pos.values():
            max_r = max(max_r, (x * x + y * y) ** 0.5)
        local_radius = max(max_r, 0.5)

        return local_pos, local_radius, is_rooted

    @staticmethod
    def _rooted_layered_layout(
        nodes: list[NodeType],
        pairs: list[tuple[NodeType, NodeType]],
        root: NodeType,
    ) -> dict[NodeType, tuple[float, float]]:
        """Layered BFS layout rooted at ``root``, growing along the local ``+x`` axis.

        The root is placed at the local origin, children at depth ``d`` are placed at
        ``x = d`` and laterally distributed symmetrically around ``y = 0``. Nodes
        unreachable from the root (which should not occur for a well-formed component)
        are appended to the deepest layer to ensure every node receives a position.
        """
        adj: dict[NodeType, list[NodeType]] = {n: [] for n in nodes}
        for u, v in pairs:
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
        for n, neighbors in adj.items():
            adj[n] = sorted(set(neighbors))

        # BFS from the root to assign integer depths
        depth: dict[NodeType, int] = {root: 0}
        order: list[NodeType] = [root]
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    order.append(v)
                    q.append(v)

        # Append any unreachable nodes at the deepest layer + 1 so every node gets a position
        max_depth = max(depth.values(), default=0)
        for n in nodes:
            if n not in depth:
                depth[n] = max_depth + 1
                order.append(n)

        # Group nodes by depth, preserving BFS discovery order within each layer
        layers: dict[int, list[NodeType]] = defaultdict(list)
        for n in order:
            layers[depth[n]].append(n)

        # Lateral spacing chosen so total layer width matches the layer count, which
        # keeps the layout's aspect ratio roughly square as depth grows.
        local_pos: dict[NodeType, tuple[float, float]] = {}
        x_step = 1.0
        y_step = 1.0
        for d, members in layers.items():
            count = len(members)
            for i, n in enumerate(members):
                # Center each layer around y = 0
                y = (i - (count - 1) / 2.0) * y_step
                local_pos[n] = (d * x_step, y)
        return local_pos

    @staticmethod
    def _pack_components(
        comp_layouts: list[tuple[dict[NodeType, tuple[float, float]], float, bool, NodeType | None]],
        world_node: int,
        world_in_graph: bool,
    ) -> dict[NodeType, tuple[float, float]]:
        """Packs per-component local layouts radially around the world node.

        Args:
            comp_layouts: For each component, ``(local_pos, local_radius, is_rooted, base_node)``.
            world_node: The index of the implicit world node.
            world_in_graph: Whether the world node should be placed at the origin.

        Returns:
            A global position dict keyed by node index. The world node, when applicable, is at
            ``(0, 0)``.
        """
        pos: dict[NodeType, tuple[float, float]] = {}
        if world_in_graph:
            pos[world_node] = (0.0, 0.0)

        n_components = len(comp_layouts)
        if n_components == 0:
            return pos

        # Sort components by descending local radius for stable packing — the largest
        # components claim their angular slots first, smaller ones fill the gaps.
        order = sorted(range(n_components), key=lambda i: -comp_layouts[i][1])
        radii = [comp_layouts[i][1] for i in order]

        # Choose an anchor ring radius `R` large enough that adjacent components do
        # not overlap. The angular footprint each component requires is approximately
        # `2 * arcsin(r / R)`. We pick the smallest R such that the sum of these
        # angular footprints fits inside `2 * pi`. Using a closed-form lower bound
        # `R >= sum(r) / pi` is conservative but always feasible; we then add a small
        # margin so labels and node radii do not collide visually.
        sum_r = sum(radii)
        # Floor on R so that even tiny graphs (e.g. a few orphans) get a sensible
        # ring; otherwise everything would collapse onto the world node.
        min_radius = max(radii) if radii else 1.0
        R = max(sum_r / math.pi, 2.5 * min_radius, 2.5)

        # Compute angular slot widths and running mid-angles
        slots: list[float] = []
        for r in radii:
            # Clamp the asin argument to [-1, 1] in case of edge cases (shouldn't
            # actually occur given the choice of R above, but defensive).
            ratio = min(max(r / R, -1.0), 1.0)
            slot = 2.0 * math.asin(ratio)
            # Minimum angular slot to keep small components readable
            slot = max(slot, 2.0 * math.pi / max(n_components * 2, 1))
            slots.append(slot)

        # Normalize slot widths so they sum to 2*pi (in case the floor above pushed
        # the total above 2*pi for many small components).
        total_slot = sum(slots)
        if total_slot > 2.0 * math.pi:
            scale = (2.0 * math.pi) / total_slot
            slots = [s * scale for s in slots]
            total_slot = 2.0 * math.pi
        # Distribute any leftover angular budget evenly as padding between slots
        padding = (2.0 * math.pi - total_slot) / max(n_components, 1)

        running = 0.0
        for sort_idx, comp_idx in enumerate(order):
            slot = slots[sort_idx]
            theta = running + slot / 2.0
            running += slot + padding

            local_pos, local_radius, is_rooted, base_node = comp_layouts[comp_idx]

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            if is_rooted and base_node is not None and base_node in local_pos:
                # Rotate the local frame so local +x points outward at angle theta,
                # and translate so the base node lands on the ring at radius R.
                bx, by = local_pos[base_node]
                # Translation so base goes to (R*cos_t, R*sin_t) after rotation
                tx = R * cos_t
                ty = R * sin_t
                for n, (lx, ly) in local_pos.items():
                    rx = (lx - bx) * cos_t - (ly - by) * sin_t
                    ry = (lx - bx) * sin_t + (ly - by) * cos_t
                    pos[n] = (rx + tx, ry + ty)
            else:
                # Unrooted: rotate by theta and translate the layout's geometric
                # center (mean of local positions) onto the ring.
                if local_pos:
                    cx = sum(p[0] for p in local_pos.values()) / len(local_pos)
                    cy = sum(p[1] for p in local_pos.values()) / len(local_pos)
                else:
                    cx = cy = 0.0
                # Anchor distance pushes the component's center outward by
                # (R + local_radius) so even unrooted layouts respect the ring.
                anchor = R + 0.25 * local_radius
                tx = anchor * cos_t
                ty = anchor * sin_t
                for n, (lx, ly) in local_pos.items():
                    rx = (lx - cx) * cos_t - (ly - cy) * sin_t
                    ry = (lx - cx) * sin_t + (ly - cy) * cos_t
                    pos[n] = (rx + tx, ry + ty)

        return pos

    def _build_edge_label(
        self,
        joint_type: int,
        joint_index: int,
        joints: list[JointDescriptor] | None,
    ) -> str:
        """Builds an edge label of the form ``f"{name}_{index}_{type}"`` (or shorter variants).

        The joint name prefix is included only when ``joints`` is provided,
        ``joints[joint_index]`` exists, and the descriptor's ``name`` is a non-empty string.

        The joint-type suffix is included only when ``joint_type`` is a non-negative integer
        recognised by the active abbreviation table (selected at construction time via
        ``joint_type_enum``). The sentinel value ``-1`` is treated as "unspecified" and the
        suffix is omitted from the label. All other unrecognised joint-type integer values
        raise :class:`ValueError` so that incorrectly-categorised edges are surfaced loudly rather
        than silently rendered with an incorrect or generic label.

        Returns one of (in priority order, given the available pieces):

        - ``"{name}_{index}_{type}"``
        - ``"{index}_{type}"``
        - ``"{name}_{index}"``
        - ``"{index}"``

        Raises:
            ValueError: If ``joint_type`` is neither ``-1`` nor a value of the active
                joint-type enum (``JointType`` or ``JointDoFType``).
        """
        if joint_type == -1:
            abbr: str | None = None
        elif joint_type in self._abbr_table:
            abbr = self._abbr_table[joint_type]
        else:
            raise ValueError(
                f"Unsupported joint type `{joint_type}` for joint at index `{joint_index}`: "
                f"value is not a member of the active joint-type enum "
                f"`{self._joint_type_enum.__name__}` and is not the `-1` sentinel for "
                f"unspecified joint types."
            )

        name: str | None = None
        if joints is not None and 0 <= joint_index < len(joints):
            descriptor = joints[joint_index]
            candidate = getattr(descriptor, "name", None)
            if isinstance(candidate, str) and candidate:
                name = candidate

        prefix = f"{name}_" if name is not None else ""
        suffix = f"_{abbr}" if abbr is not None else ""
        return f"{prefix}{joint_index}{suffix}"

    @override
    def render_component_spanning_tree_candidates(
        self,
        component: TopologyComponent,
        candidates: list[TopologySpanningTree],
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """
        Renders the candidate spanning trees for each component of
        the given topology graph using networkx and matplotlib.

        Args:
            component: The `TopologyComponent` instance whose spanning tree candidates are to be rendered.
            candidates: A list of `TopologySpanningTree` instances representing the candidate spanning trees.
            world_node: The index of the world node in the topology graph.
            figsize: Optional tuple specifying the figure size for the plot.
            path: Optional string specifying the file path to save the plot.
            show: Boolean indicating whether to display the plot.
        """
        pass  # TODO: IMPLEMENT

    @override
    def render_component_spanning_tree(
        self,
        component: TopologyComponent,
        tree: TopologySpanningTree,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """
        Renders the selected spanning tree for a given component
        of the topology graph using networkx and matplotlib.

        Args:
            component: The `TopologyComponent` instance whose selected spanning tree is to be rendered.
            tree: The `TopologySpanningTree` instance representing the selected spanning tree.
            world_node: The index of the world node in the topology graph.
            figsize: Optional tuple specifying the figure size for the plot.
            path: Optional string specifying the file path to save the plot.
            show: Boolean indicating whether to display the plot.
        """
        pass  # TODO: IMPLEMENT


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
