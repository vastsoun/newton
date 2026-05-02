# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ..core.bodies import RigidBodyDescriptor
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override

###
# Module interface
###

__all__ = [
    "DEFAULT_WORLD_NODE_INDEX",
    "NO_BASE_JOINT_INDEX",
    "ComponentConnectivity",
    "ComponentType",
    "EdgeType",
    "GraphEdge",
    "GraphNode",
    "NodeType",
    "SpanningTreeTraversal",
    "TopologyComponent",
    "TopologyComponentBaseSelectorBase",
    "TopologyComponentParserBase",
    "TopologyGraphVisualizerBase",
    "TopologySpanningTree",
    "TopologySpanningTreeGeneratorBase",
    "TopologySpanningTreeSelectorBase",
    "validate_max_candidates",
    "validate_traversal_mode",
]

###
# Constants
###

DEFAULT_WORLD_NODE_INDEX: int = -1
"""Default body-index sentinel for the implicit world node (Newton convention)."""

UNASSIGNED_JOINT_TYPE: int = -1
"""Joint-type sentinel marking the absence of a real joint type."""

NO_BASE_JOINT_INDEX: int = -1
"""Joint-index sentinel marking the absence of a real base joint at arc predecessor position ``0``."""

ComponentType = Literal["island", "orphan"]
"""Component subgraph type: ``"island"`` (multi-body) or ``"orphan"`` (single-body)."""

ComponentConnectivity = Literal["connected", "isolated"]
"""Component connectivity: ``"connected"`` (linked to the world node) or ``"isolated"``."""

SpanningTreeTraversal = Literal["dfs", "bfs"]
"""Spanning-tree traversal mode: ``"dfs"`` (depth-first) or ``"bfs"`` (breadth-first)."""


###
# Core Types
###


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A body node in a topology graph."""

    index: int
    """The unique body index (``>= 0`` for moving bodies; ``< 0`` reserved for the world)."""

    name: str | None = field(default=None, compare=False)
    """Optional human-readable name for the body. Excluded from equality/hashing."""

    def __post_init__(self) -> None:
        """Validates the GraphNode instance attributes."""
        if not isinstance(self.index, int):
            raise TypeError(f"`index` must be an integer; got {type(self.index).__name__}.")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"`name` must be a string or `None`; got {type(self.name).__name__}.")

    @classmethod
    def from_input(cls, node: NodeType) -> GraphNode:
        """Coerce ``node`` to a :class:`GraphNode` instance."""
        if isinstance(node, cls):
            return node
        if not isinstance(node, int):
            raise TypeError(f"Cannot coerce {node!r} to GraphNode; expected an int or GraphNode instance.")
        return cls(index=node)

    def __int__(self) -> int:
        """Return the underlying body index for arithmetic and indexing."""
        return self.index


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A joint edge in a topology graph."""

    _VALID_JOINT_TYPE_VALUES: ClassVar[frozenset[int]] = frozenset({-1, *(int(jt) for jt in JointDoFType)})
    """Permitted integer values for :attr:`joint_type` (every :class:`JointDoFType` value plus ``-1``)."""

    joint_type: int
    """The integer value of the joint's :class:`JointDoFType`, or ``-1`` for unspecified."""

    joint_index: int
    """The global joint index. Negative values are accepted as synthetic sentinels."""

    nodes: tuple[int, int]
    """The ``(predecessor, successor)`` body indices connected by the joint."""

    def __post_init__(self) -> None:
        """Validate the GraphEdge instance attributes."""
        if self.joint_type not in self._VALID_JOINT_TYPE_VALUES:
            raise ValueError(
                f"`joint_type={self.joint_type}` is not a valid `JointDoFType` value or the `-1` sentinel; "
                f"valid values are {sorted(self._VALID_JOINT_TYPE_VALUES)}."
            )
        if not isinstance(self.nodes, tuple) or len(self.nodes) != 2:
            raise TypeError(f"`nodes` must be a 2-tuple of integers; got {self.nodes!r}.")

    @classmethod
    def from_input(cls, edge: EdgeType | tuple[int, int, Iterable[int]]) -> GraphEdge:
        """Coerce ``edge`` to a :class:`GraphEdge` instance."""
        if isinstance(edge, cls):
            return edge
        if not isinstance(edge, tuple | list) or len(edge) != 3:
            raise TypeError(
                f"Cannot coerce {edge!r} to GraphEdge; expected a GraphEdge "
                f"instance or a 3-tuple `(joint_type, joint_index, (u, v))`."
            )
        jt, jid, nodes = edge
        return cls(
            joint_type=int(jt),
            joint_index=int(jid),
            nodes=tuple(nodes),  # type: ignore[arg-type]
        )

    def to_tuple(self) -> tuple[int, int, tuple[int, int]]:
        """Return the edge as a ``(joint_type, joint_index, nodes)`` tuple."""
        return (self.joint_type, self.joint_index, self.nodes)

    @property
    def dof_type(self) -> JointDoFType | None:
        """The :class:`JointDoFType` of the joint, or ``None`` for the `UNASSIGNED_JOINT_TYPE` sentinel."""
        if self.joint_type == UNASSIGNED_JOINT_TYPE:
            return None
        return JointDoFType(self.joint_type)


@dataclass(frozen=True, slots=True)
class OrientedEdge:
    """A joint edge oriented relative to a chosen spanning-tree root.

    Preserves the original input endpoint pair alongside the oriented
    parent-to-child pair. Used internally by spanning-tree generators to
    disambiguate edge polarity when a tree's parent/child relationship
    differs from the input edge's predecessor/successor order.
    """

    original: GraphEdge
    """The original edge as provided in the input subgraph."""

    oriented: GraphEdge
    """The oriented edge (parent-to-child) used in the returned tree."""


###
# Front-End Types
###

NodeType = GraphNode | int
"""Front-end union accepted by API boundaries; coerced internally to :class:`GraphNode`."""


EdgeType = GraphEdge | tuple[int, int, tuple[int, int]]
"""Front-end union accepted by API boundaries; coerced internally to :class:`GraphEdge`."""


@dataclass
class TopologyComponent:
    """A container for a component subgraph (its body nodes, joint edges, and metadata).

    Holds the parsed structure of a connected component along with optional
    grounding/base assignments used by downstream spanning-tree generation.
    """

    nodes: list[GraphNode] | None = None
    """Canonical body nodes of the component."""

    edges: list[GraphEdge] | None = None
    """List of joint edges associated with the component."""

    ground_nodes: list[GraphNode] | None = None
    """Body nodes that connect the component to the implicit world node."""

    ground_edges: list[GraphEdge] | None = None
    """List of joint edges that connect the component to the implicit world node."""

    base_node: GraphNode | None = None
    """Canonical base body node, or ``None`` if no base is assigned."""

    base_edge: GraphEdge | None = None
    """The base joint edge, or ``None`` if no base is assigned."""

    is_connected: bool | None = None
    """``True`` when the component has any grounding edge or an assigned base edge."""

    is_island: bool | None = None
    """``True`` for islands (multi-body components); ``False`` for orphans."""

    world_node: int = DEFAULT_WORLD_NODE_INDEX
    """The implicit world-node index used by the parent graph (must be negative)."""

    def __post_init__(self):
        """Validate component invariants and normalize node/edge fields.

        Coerces every front-end input to its canonical type:
        :data:`NodeType` (``nodes``, ``ground_nodes``, ``base_node``) is
        promoted to :class:`GraphNode`, and :data:`EdgeType` (``edges``,
        ``ground_edges``, ``base_edge``) to :class:`GraphEdge`. Then
        checks that the component's structural invariants hold.

        Raises:
            ValueError: If ``world_node`` is non-negative, a grounding edge is
                a 6-DoF FREE joint, ``base_node`` and ``base_edge`` are not
                both set or both unset, ``base_node`` is not in ``nodes``,
                ``base_node`` is not an endpoint of ``base_edge``,
                ``base_edge``'s peer endpoint is not ``world_node``,
                ``ground_nodes`` and ``ground_edges`` are not both set or
                both unset, ``ground_nodes`` contains duplicate body
                indices, ``ground_nodes`` does not match the non-world
                endpoints of ``ground_edges``, ``is_island`` is
                inconsistent with ``len(nodes) > 1``, or ``is_connected``
                is inconsistent with the actual world-link state.
            TypeError: If ``world_node`` is not an integer.
        """
        # First coerce all front-end inputs to their canonical types
        if self.nodes is not None:
            self.nodes = [GraphNode.from_input(n) for n in self.nodes]
        if self.edges is not None:
            self.edges = [GraphEdge.from_input(e) for e in self.edges]
        if self.ground_nodes is not None:
            self.ground_nodes = [GraphNode.from_input(n) for n in self.ground_nodes]
        if self.ground_edges is not None:
            self.ground_edges = [GraphEdge.from_input(e) for e in self.ground_edges]
        if self.base_node is not None:
            self.base_node = GraphNode.from_input(self.base_node)
        if self.base_edge is not None:
            self.base_edge = GraphEdge.from_input(self.base_edge)

        # Validate the world-node sentinel up front so the body-vs-world checks
        if self.world_node >= 0:
            raise ValueError(
                f"Component `world_node={self.world_node}` must be a negative integer "
                f"(the conventional sentinel for the implicit world)."
            )

        # Validate that `base_node` and `base_edge` are either both set or both unset,
        # and if set, that they are consistent with each other and the world-link state.
        if (self.base_node is None) != (self.base_edge is None):
            raise ValueError(
                f"Component must define both `base_node` and `base_edge`, or neither: "
                f"got base_node={self.base_node!r}, base_edge={self.base_edge!r}."
            )

        # Validate that `ground_nodes` and `ground_edges` are both set or
        # both unset, and if set, that they are consistent with each other.
        if (self.ground_nodes is None) != (self.ground_edges is None):
            raise ValueError(
                f"Component must define both `ground_nodes` and `ground_edges`, or neither: "
                f"got ground_nodes={self.ground_nodes!r}, ground_edges={self.ground_edges!r}."
            )

        # Validate that no grounding edge is a 6-DoF FREE joint, since that would violate
        # the exclusivity of grounding vs base edges and break the assumptions of downstream
        # consumers that rely on grounding edges to define loop constraints.
        if self.ground_edges is not None:
            for edge in self.ground_edges:
                if edge.joint_type == JointDoFType.FREE:
                    raise ValueError(f"Grounding edge `{edge}` cannot be 6-DoF FREE joint.")

        # When base_node is set, base_edge is also set (XOR check above), so we
        # do not need a redundant `if self.base_edge is not None:` guard here.
        if self.base_node is not None:
            base_idx = int(self.base_node)
            if self.nodes is not None and self.base_node not in self.nodes:
                raise ValueError(
                    f"Component base node `{self.base_node}` is not contained in component nodes `{self.nodes}`."
                )
            base_pair = self.base_edge.nodes
            if base_idx not in base_pair:
                raise ValueError(
                    f"Component base node `{self.base_node}` is not an endpoint of base edge `{self.base_edge}`."
                )
            other = base_pair[0] if base_pair[1] == base_idx else base_pair[1]
            if other != self.world_node:
                raise ValueError(
                    f"Component base edge `{self.base_edge}` must connect `base_node={self.base_node}` "
                    f"to the world node `{self.world_node}`; got non-world endpoint `{other}`."
                )

        # Ensure that `ground_nodes` does not contain duplicate body indices, which would violate
        # the assumption that each grounding node corresponds to a unique grounding edge.
        if self.ground_nodes is not None:
            ground_index_counts: dict[int, int] = {}
            for n in self.ground_nodes:
                ground_index_counts[int(n)] = ground_index_counts.get(int(n), 0) + 1
            duplicate_ground = sorted(idx for idx, count in ground_index_counts.items() if count > 1)
            if duplicate_ground:
                raise ValueError(f"Component `ground_nodes` contain duplicate body indices: {duplicate_ground}.")

        # Validate that the non-world endpoints of `ground_edges`
        # match the entries in `ground_nodes`, if both are set.
        if self.ground_edges is not None and self.ground_nodes is not None:
            implied_indices = {n for e in self.ground_edges for n in e.nodes if n != self.world_node}
            ground_indices = {int(n) for n in self.ground_nodes}
            if ground_indices != implied_indices:
                raise ValueError(
                    f"Component `ground_nodes` {sorted(ground_indices)} does not match the non-world "
                    f"endpoints {sorted(implied_indices)} of `ground_edges`."
                )

        # Validate that `is_island` is consistent with the number of nodes,
        # and that `is_connected` is consistent with the world-link state.
        if self.is_island is not None and self.nodes is not None:
            expected_is_island = len(self.nodes) > 1
            if self.is_island != expected_is_island:
                raise ValueError(
                    f"Component `is_island={self.is_island}` is inconsistent with `len(nodes)={len(self.nodes)}`; "
                    f"expected `is_island={expected_is_island}`."
                )

        # Validate that `is_connected` is consistent with the presence of
        # any world-linking edge, either a base edge or a grounding edge.
        if self.is_connected is not None:
            has_world_link = self.base_edge is not None or bool(self.ground_edges)
            if self.is_connected != has_world_link:
                raise ValueError(
                    f"Component `is_connected={self.is_connected}` is inconsistent with the "
                    f"world-link state (base_edge={self.base_edge!r}, "
                    f"ground_edges={self.ground_edges!r}); expected `is_connected={has_world_link}`."
                )

    def assign_base(self, base_node: NodeType, base_edge: EdgeType) -> None:
        """Assign the component's base node/edge and synchronize world-link state.

        Args:
            base_node: The body node to designate as the component's base.
            base_edge: The joint edge connecting ``base_node`` to the world node.

        Raises:
            ValueError: If the resulting state violates any
                :meth:`__post_init__` invariant.
        """
        # Coerce the inputs to their canonical types and assign them to the component.
        base_edge = GraphEdge.from_input(base_edge)
        self.base_node = GraphNode.from_input(base_node)
        self.base_edge = base_edge
        self.is_connected = True
        if self.ground_edges and base_edge in self.ground_edges:
            self.ground_edges = [g for g in self.ground_edges if g != base_edge]
            if self.ground_nodes is not None:
                remaining_indices = {n for e in self.ground_edges for n in e.nodes if n != self.world_node}
                self.ground_nodes = sorted(
                    (gn for gn in self.ground_nodes if int(gn) in remaining_indices),
                    key=int,
                )
        # Re-validate the component after mutation
        # to ensure the new state is consistent.
        self.__post_init__()

    @override
    def __str__(self) -> str:
        """Return a human-readable multi-line summary of the component."""
        return (
            f"TopologyComponent(\n"
            f"nodes: {self.nodes},\n"
            f"edges: {self.edges},\n"
            f"world_node: {self.world_node},\n"
            f"ground_nodes: {self.ground_nodes},\n"
            f"ground_edges: {self.ground_edges},\n"
            f"base_node: {self.base_node},\n"
            f"base_edge: {self.base_edge},\n"
            f"is_connected: {self.is_connected},\n"
            f"is_island: {self.is_island},\n"
            f")"
        )


@dataclass
class TopologySpanningTree:
    """A host-side container for a spanning tree of a component subgraph.

    A spanning tree of a component subgraph is the maximal acyclic subgraph
    that includes all body nodes; arcs are the joint edges in the tree and
    chords are the remaining edges that close kinematic loops.

    Note:
        Array-shaped fields (``arcs``, ``chords``, ``predecessors``,
        ``successors``, ``parents``, ``support``, ``children``, ``subtree``)
        are Python lists. They are converted to packed
        :class:`wp.array[wp.int32]` device buffers only when aggregated into
        a :class:`TopologyModel`; do not pass them directly to Warp kernels.
    """

    ###
    # Meta-Data
    ###

    traversal: SpanningTreeTraversal = "dfs"
    """The traversal mode used to generate the spanning tree (``"dfs"`` or ``"bfs"``)."""

    depth: int = 0
    """Maximum number of arcs on any root-to-leaf path."""

    directed: bool = False
    """``True`` when the arcs have a defined parent-to-child direction."""

    num_bodies: int = 0
    """Total number of moving body nodes ``N_B`` in the spanning tree (excludes the world)."""

    num_joints: int = 0
    """Total number of joint edges ``N_J`` (includes the base joint connecting to the world)."""

    num_tree_arcs: int = 0
    """Number of real joint edges included as arcs.

    Equals ``num_bodies`` for a connected component (one arc per body, base
    edge at ``arcs[0]``); ``num_bodies - 1`` for an isolated island (the
    sentinel at ``arcs[0]`` is not counted); ``len(arcs)`` for orphans.
    """

    num_tree_chords: int = 0
    """Number of joint edges not included as arcs (i.e. loop joints), equal to ``num_joints - num_bodies``."""

    ###
    # Definition
    ###

    component: TopologyComponent | None = None
    """The source component subgraph associated with the spanning tree."""

    root: int | None = None
    """The root body-node index, i.e. the unique body directly connected to the world."""

    arcs: list[int] | None = None
    """List of joint indices included in the spanning tree, shape ``(num_tree_arcs,)``."""

    chords: list[int] | None = None
    """List of joint indices not included in the spanning tree, shape ``(num_tree_chords,)``."""

    ###
    # Parameterization
    ###

    predecessors: list[int] | None = None
    """Per-joint predecessor body local positions, shape ``(num_joints,)``.

    The predecessor of the root is conventionally set to ``-1``.
    """

    successors: list[int] | None = None
    """Per-joint successor body local positions, shape ``(num_joints,)``.

    The successor of the root is conventionally set to ``-1``.
    """

    parents: list[int] | None = None
    """Per-body parent local positions, shape ``(num_bodies,)``.

    Featherstone's parent array ``λ`` (``lambda``) satisfying
    ``λ(i) = min(p(i), s(i))`` and ``λ(i) < i`` for ``0 ≤ i ≤ NB - 1``.
    The root's parent is conventionally set to ``-1``.
    """

    support: list[list[int]] | None = None
    """Per-body support arc local positions, shape ``(num_bodies,)``.

    Featherstone's support set ``κ`` (``kappa``): for any body ``i`` except
    the base, ``κ(i)`` is the set of all tree joint edges on the path from
    body ``i`` to the base, and satisfies ``j ∈ κ(i) ⇒ i ∈ v(j)``.
    """

    children: list[list[int]] | None = None
    """Per-body children body local positions, shape ``(num_bodies,)``.

    Featherstone's children set ``µ`` (``mu``) satisfying ``µ(i) ⊆ v(i)``.
    """

    subtree: list[list[int]] | None = None
    """Per-body subtree body local positions, shape ``(num_bodies,)``.

    Featherstone's subtree set ``v`` (``nu``): the set of bodies in the
    subtree rooted at body ``i``, satisfying ``µ(i) ⊆ v(i)`` and
    ``j ∈ κ(i) ⇒ i ∈ v(j)``.
    """

    ###
    # Operations
    ###

    def balanced_score(self) -> int:
        """Return the sum of squared per-parent child counts as a balance score.

        Lower scores correspond to more balanced trees: at fixed depth the
        minimum is reached when every internal node has roughly the same
        number of children, while a single-spine tree reaches the maximum.

        Returns:
            The integer balance score (lower is more balanced).

        Raises:
            ValueError: If :attr:`children` is ``None``.
        """
        if self.children is None:
            raise ValueError("Cannot score a `TopologySpanningTree` with `children=None`; the tree is malformed.")
        return sum(len(cs) * len(cs) for cs in self.children)


###
# Module Interfaces
###


class TopologyComponentParserBase:
    """Interface for modules that parse a topology graph into a list of components."""

    @abstractmethod
    def parse_components(self, nodes: list[NodeType], edges: list[EdgeType], world: int) -> list[TopologyComponent]:
        """Parse the given nodes and edges into a list of components.

        Args:
            nodes: List of body node indices.
            edges: List of joint edges in :data:`EdgeType` form.
            world: The implicit world node index.

        Returns:
            A list of :class:`TopologyComponent` instances.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TopologyComponentBaseSelectorBase:
    """Interface for modules that select the base node and edge for a component."""

    @abstractmethod
    def select_base(
        self,
        component: TopologyComponent,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> tuple[GraphNode, GraphEdge]:
        """Select the base node and edge for the given component.

        Args:
            component: The component subgraph to select a base for.
            bodies: Optional body descriptors used as input to the heuristic.
            joints: Optional joint descriptors used as input to the heuristic.

        Returns:
            A ``(base_node, base_edge)`` pair, where ``base_node`` is the
            canonical :class:`GraphNode` representation of the chosen body.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TopologySpanningTreeGeneratorBase:
    """Interface for modules that generate spanning-tree candidates for a component."""

    @abstractmethod
    def generate_spanning_trees(
        self,
        component: TopologyComponent,
        traversal_mode: SpanningTreeTraversal | None = None,
        max_candidates: int | None = None,
        roots: list[NodeType] | None = None,
        *,
        override_priorities: bool | None = None,
        prioritize_balanced: bool | None = None,
    ) -> list[TopologySpanningTree]:
        """Generate spanning-tree candidates for the given component.

        Keyword arguments default to ``None`` so callers can dispatch
        unconditionally; concrete backends substitute their configured
        defaults whenever an argument is left unset.

        Args:
            component: The component subgraph to enumerate trees for.
            traversal_mode: Traversal mode used for body ordering.
            max_candidates: Maximum number of candidates to produce.
            roots: Explicit root candidates to use, overriding the backend's
                priority cascade.
            override_priorities: If ``True``, skip the backend's priority
                cascade and brute-force enumerate over every body node.
            prioritize_balanced: If ``True``, prefer balanced/symmetric trees
                in candidate ordering and truncation.

        Returns:
            A list of :class:`TopologySpanningTree` candidates.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TopologySpanningTreeSelectorBase:
    """Interface for modules that pick a single spanning tree from a candidate list."""

    @abstractmethod
    def select_spanning_tree(
        self,
        candidates: list[TopologySpanningTree],
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> TopologySpanningTree:
        """Select the best spanning tree from the given candidate list.

        Args:
            candidates: Non-empty list of candidate spanning trees.
            bodies: Optional body descriptors used as input to the heuristic.
            joints: Optional joint descriptors used as input to the heuristic.

        Returns:
            The selected :class:`TopologySpanningTree`.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TopologyGraphVisualizerBase:
    """Interface for modules that render a topology graph, its components, and spanning trees."""

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
        """Render the given topology graph and its components.

        Args:
            nodes: Body node indices to visualize.
            edges: Joint edges to visualize.
            components: Components of the topology graph.
            world_node: Index of the implicit world node.
            joints: Optional joint descriptors for name-based edge labels.
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.
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
        """Render the spanning-tree candidates of a component.

        Args:
            component: The component the candidates belong to.
            candidates: List of candidate spanning trees.
            world_node: Index of the implicit world node.
            joints: Optional joint descriptors for name-based edge labels.
            skip_orphans: When ``True``, skip orphan components.
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.
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
        """Render the selected spanning tree of a component.

        Args:
            component: The component the tree belongs to.
            tree: The selected spanning tree to render.
            world_node: Index of the implicit world node.
            joints: Optional joint descriptors for name-based edge labels.
            skip_orphans: When ``True``, skip orphan components.
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.
        """
        raise NotImplementedError("Subclasses must implement this method.")


###
# Utilities
###

_VALID_TRAVERSALS: tuple[SpanningTreeTraversal, ...] = ("dfs", "bfs")
"""Runtime-validated set of :data:`SpanningTreeTraversal` values."""


def validate_traversal_mode(mode: SpanningTreeTraversal | None) -> None:
    """Validate ``mode`` against the supported traversal values.

    Args:
        mode: The traversal mode to validate, or ``None`` for a no-op.

    Raises:
        ValueError: If ``mode`` is not one of :data:`_VALID_TRAVERSALS`.
    """
    if mode is None:
        return
    if mode not in _VALID_TRAVERSALS:
        raise ValueError(f"Invalid traversal mode: {mode!r}; expected one of {_VALID_TRAVERSALS}.")


def validate_max_candidates(max_candidates: int | None) -> None:
    """Validate the ``max_candidates`` argument used by spanning-tree generators.

    Args:
        max_candidates: The maximum candidate count to validate, or ``None``
            for a no-op.

    Raises:
        TypeError: If ``max_candidates`` is not an :class:`int` (booleans are
            rejected even though ``bool`` subclasses ``int``).
        ValueError: If ``max_candidates`` is non-positive.
    """
    if max_candidates is None:
        return
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise TypeError(f"`max_candidates` must be an integer or None; got {type(max_candidates).__name__}.")
    if max_candidates <= 0:
        raise ValueError(f"`max_candidates` must be a positive integer; got {max_candidates}.")
