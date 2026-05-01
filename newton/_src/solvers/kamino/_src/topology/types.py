# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines types & containers for constrained multi-body system topologies.

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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from ..core.bodies import RigidBodyDescriptor
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override

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
    "TopologyComponentParserBase",
    "TopologyGraphVisualizerBase",
    "TopologySpanningTree",
    "TopologySpanningTreeGeneratorBase",
    "TopologySpanningTreeSelectorBase",
    "_validate_max_candidates",
    "_validate_traversal_mode",
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
form ``(joint_type, joint_index, (predecessor_body_index, successor_body_index))``.

The ``joint_type`` field carries an integer matching a :class:`JointDoFType`
value (kamino DoF-typed joints), **not** a :class:`newton.JointType` value;
e.g. ``JointDoFType.FREE`` (``0``) marks a 6-DoF free-floating joint. Callers
constructing edges typically pass ``JointDoFType.<NAME>.value``.
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
        # not FREE joints which would violate modelling conventions.
        # `EdgeType.joint_type` carries a kamino :class:`JointDoFType` value (see
        # the parser/visualizer backends), not a Newton :class:`JointType`. Using
        # the wrong enum here silently never fires the invariant check.
        if self.ground_edges is not None:
            for edge in self.ground_edges:
                joint_type, _, _ = edge
                if joint_type == JointDoFType.FREE:
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
    Number of real joint edges included in the spanning tree, i.e. arcs.\n
    Equals ``num_bodies`` for a connected component (one arc per body, base
    edge included as ``arcs[0]``), and ``num_bodies - 1`` for an isolated
    island (the sentinel ``-1`` at ``arcs[0]`` is **not** counted). For
    orphans this equals ``len(arcs)`` directly (``0`` or ``1``).
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
                ``(joint_type, joint_index, (predecessor_body_index, successor_body_index))``.
            world:
                The index of the implicit world node in the graph.

        Returns:
            A list of :class:`TopologyComponent` objects representing the components of the graph.
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
        *,
        override_priorities: bool = False,
        prioritize_balanced: bool = False,
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
            override_priorities:
                If ``True``, ignore any base/grounding/degree-based root prioritization
                and enumerate spanning trees from every body node in the component.
                Backends that do not implement brute-force enumeration may raise
                :class:`NotImplementedError` for this combination.
            prioritize_balanced:
                If ``True``, request that the backend prefers balanced/symmetric
                trees (e.g. minimizing branching imbalance) when ordering or
                truncating candidates. Backends that do not support this hint may
                ignore it.

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
        joints: list[JointDescriptor] | None = None,
        skip_orphans: bool = True,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Renders the spanning tree candidates for the given topology graph.

        Args:
            component: The component containing the spanning tree candidates to visualize.
            candidates: The list of spanning tree candidates to visualize.
            world_node: The index of the implicit world node in the graph.
            joints:
                Optional list of joint descriptors used to look up joint names for edge
                labels. When provided, ``joints[joint_index].name`` is included in each
                edge label; when omitted, the label uses the joint index and joint-type
                abbreviation only.
            skip_orphans:
                When ``True`` (default), orphan components (single-body subgraphs whose
                spanning tree is trivial) are skipped and no figure is produced for them.
                Set to ``False`` to render the trivial candidate of every component.
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
        joints: list[JointDescriptor] | None = None,
        skip_orphans: bool = True,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Renders the selected spanning tree for the given topology graph.

        Args:
            component: The component containing the selected spanning tree to visualize.
            tree: The selected spanning tree to visualize.
            world_node: The index of the implicit world node in the graph.
            joints:
                Optional list of joint descriptors used to look up joint names for edge
                labels. When provided, ``joints[joint_index].name`` is included in each
                edge label; when omitted, the label uses the joint index and joint-type
                abbreviation only.
            skip_orphans:
                When ``True`` (default), orphan components (single-body subgraphs) are
                skipped since their spanning trees are trivial. Set to ``False`` to render
                every component regardless.
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
        """
        raise NotImplementedError("Subclasses must implement this method.")


###
# Utilities
###

_VALID_TRAVERSALS: tuple[SpanningTreeTraversal, ...] = ("dfs", "bfs")
"""Single source of truth for the runtime-validated set of :data:`SpanningTreeTraversal` values."""


def _validate_traversal_mode(mode: SpanningTreeTraversal | None) -> None:
    """Validate ``mode`` against :data:`_VALID_TRAVERSALS`.

    :data:`SpanningTreeTraversal` is a :data:`typing.Literal[...]` alias and is
    not iterable at runtime, so we validate against the explicit constant tuple.

    Raises:
        ValueError: If ``mode`` is not one of :data:`_VALID_TRAVERSALS`.
    """
    if mode is None:
        return
    if mode not in _VALID_TRAVERSALS:
        raise ValueError(f"Invalid traversal mode: {mode!r}; expected one of {_VALID_TRAVERSALS}.")


def _validate_max_candidates(max_candidates: int | None) -> None:
    """Validate the ``max_candidates`` argument used by spanning-tree generators.

    Raises:
        TypeError: If ``max_candidates`` is not an :class:`int` (and not ``None``).
        ValueError: If ``max_candidates`` is non-positive.
    """
    if max_candidates is None:
        return
    if not isinstance(max_candidates, int):
        raise TypeError(f"`max_candidates` must be an integer or None; got {type(max_candidates).__name__}.")
    if max_candidates <= 0:
        raise ValueError(f"`max_candidates` must be a positive integer; got {max_candidates}.")
