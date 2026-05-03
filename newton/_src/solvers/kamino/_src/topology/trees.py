# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Spanning-tree generator backends for the topology subsystem.

Concrete implementations of :class:`TopologySpanningTreeGeneratorBase`
that the :class:`TopologyGraph` pipeline can use to enumerate
spanning-tree candidates for each component of a topology graph.

Conventions
-----------
The generators in this module use **traversal-order local body indexing**
inside each returned :class:`TopologySpanningTree`. Body local position
``0`` is the spanning tree's root body, and the remaining bodies are
numbered ``1..N_B - 1`` in BFS or DFS discovery order from the root. All
per-body fields (``parents``, ``children``, ``support``, ``subtree``)
store local body positions; per-joint fields (``predecessors``,
``successors``) store local body positions for both the arc and chord
segments, with the implicit world body keeping the sentinel value ``-1``.

The ``arcs`` list stores **global** joint indices and is parallel to the
local body positions: ``arcs[i]`` is the global joint index of the joint
connecting the body at local position ``i`` to its parent body (or, for
``i == 0``, to the implicit world node via the component's base edge).
The ``chords`` list stores the global joint indices of all remaining
joints in the source component. This guarantees Featherstone's
regular-numbering invariant ``parents[i] < i`` for all ``i >= 1`` by
construction (since traversal discovers parents before children).
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import product

from ..core.types import override
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    NO_BASE_JOINT_INDEX,
    GraphEdge,
    NodeType,
    OrientedEdge,
    SpanningTreeTraversal,
    TopologyComponent,
    TopologySpanningTree,
    TopologySpanningTreeGeneratorBase,
    validate_max_candidates,
    validate_traversal_mode,
)

###
# Module interface
###

__all__ = [
    "TopologyMinimumDepthSpanningTreeGenerator",
]


###
# Backends
###


class TopologyMinimumDepthSpanningTreeGenerator(TopologySpanningTreeGeneratorBase):
    """Spanning-tree generator that enumerates minimum-depth candidates per component.

    For a chosen root body, a *minimum-depth* spanning tree is a tree
    whose maximum root-to-leaf arc count equals the eccentricity of the
    root in the component's body-only subgraph. The set of such trees is
    enumerated by allowing each non-root body to pick any incoming
    parent edge from a strictly preceding BFS layer.

    Root-selection priority cascade (overridable per call):

    1. An explicit ``roots`` argument always wins. With a single root and
       a component without an auto-assigned base, the root and its first
       connecting grounding edge (if any) are written back to
       ``component.base_node`` / ``component.base_edge``.
    2. Otherwise, ``component.base_node`` is used as the unique root.
    3. Otherwise, ``component.ground_nodes`` is used (one root each).
    4. Otherwise, the body with the unique maximum internal degree.
    5. Otherwise (degree tie or ``override_priorities=True``), brute-force
       over every body node.

    Args:
        directed: When ``True``, treat each input edge's ``(predecessor,
            successor)`` order as the only admissible polarity; when
            ``False`` (default), allow either polarity.
        traversal_mode: Default traversal order (``"bfs"`` or ``"dfs"``)
            used to assign local body positions inside each generated tree.
        max_candidates: Default cap on the number of candidate trees
            produced per component; ``None`` means no cap.
        override_priorities: Default value of the per-call argument; when
            ``True``, skip the priority cascade and brute-force over every
            body node as root.
        prioritize_balanced: Default value of the per-call argument; when
            ``True``, candidate trees are re-ordered (and truncated, if
            ``max_candidates`` is set) by an imbalance metric.
        prioritize_grounding_when_no_base: When ``True`` (default), use
            ``component.ground_nodes`` as roots in step ``3``; when
            ``False``, skip directly to the degree-based heuristic.
    """

    ###
    # Construction
    ###

    def __init__(
        self,
        *,
        directed: bool = False,
        traversal_mode: SpanningTreeTraversal = "dfs",
        max_candidates: int | None = None,
        override_priorities: bool = False,
        prioritize_balanced: bool = False,
        prioritize_grounding_when_no_base: bool = True,
    ) -> None:
        validate_traversal_mode(traversal_mode)
        validate_max_candidates(max_candidates)
        self._directed: bool = directed
        self._traversal_mode: SpanningTreeTraversal = traversal_mode
        self._max_candidates: int | None = max_candidates
        self._override_priorities: bool = override_priorities
        self._prioritize_balanced: bool = prioritize_balanced
        self._prioritize_grounding_when_no_base: bool = prioritize_grounding_when_no_base

    ###
    # Operations
    ###

    @override
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
        """Enumerate minimum-depth spanning trees for ``component``.

        Per-call arguments take precedence over the constructor defaults
        when supplied. See the class docstring for the priority cascade.

        Args:
            component: The component subgraph to enumerate trees for.
            traversal_mode: Per-call traversal override.
            max_candidates: Per-call cap on candidates per component.
            roots: Optional explicit root list (priority cascade step 1).
            override_priorities: Per-call override of the constructor
                default.
            prioritize_balanced: Per-call override of the constructor
                default.

        Returns:
            A list of :class:`TopologySpanningTree` candidates.

        Raises:
            ValueError: If ``component`` is ``None`` or has no body nodes.
            RuntimeError: If no root candidate could be selected.
        """
        traversal: SpanningTreeTraversal = traversal_mode if traversal_mode is not None else self._traversal_mode
        validate_traversal_mode(traversal)
        cap: int | None = max_candidates if max_candidates is not None else self._max_candidates
        validate_max_candidates(cap)
        ovp: bool = self._override_priorities if override_priorities is None else override_priorities
        bal: bool = self._prioritize_balanced if prioritize_balanced is None else prioritize_balanced

        if component is None:
            raise ValueError("`component` must not be None.")
        # Coerce the canonical `GraphNode` storage into raw int indices for the internal
        # graph algorithms (BFS, adjacency, root selection); per-node metadata (e.g.
        # names) is irrelevant for tree enumeration and stays on the source component.
        body_nodes: list[int] = [int(n) for n in (component.nodes or [])]
        if not body_nodes:
            raise ValueError("`component.nodes` must contain at least one body node.")

        # Orphan special case: a single-body component has exactly one trivial
        # spanning tree (with the body as root and at most the base edge as
        # the sole arc).
        if len(body_nodes) == 1:
            return [self._build_orphan_tree(component, body_nodes[0], traversal)]

        root_candidates = self._select_root_candidates(component, roots, ovp)
        if not root_candidates:
            raise RuntimeError(
                f"Could not select any root candidate for component with nodes `{body_nodes}` "
                f"(roots={roots!r}, override_priorities={ovp})."
            )

        candidates: list[TopologySpanningTree] = []
        remaining = cap
        for root in root_candidates:
            per_root = self._enumerate_min_depth_trees_from_root(
                component=component,
                root=root,
                traversal=traversal,
                max_count=remaining,
                directed=self._directed,
            )
            candidates.extend(per_root)
            if cap is not None:
                remaining = cap - len(candidates)
                if remaining <= 0:
                    break

        # Optional balanced-tree ordering. Stable sort preserves the
        # original enumeration order on ties.
        if bal and len(candidates) > 1:
            candidates.sort(key=lambda t: t.balanced_score())
        return candidates

    ###
    # Internals - graph utilities
    ###

    def _partition_edges(self, component: TopologyComponent, world: int) -> tuple[list[GraphEdge], list[GraphEdge]]:
        """Split ``component.edges`` into ``(internal, world)`` lists.

        Internal edges are body-to-body; world edges have the world
        sentinel as one endpoint.
        """
        internal: list[GraphEdge] = []
        world_edges: list[GraphEdge] = []
        for e in component.edges or []:
            u, v = e.nodes
            if u == world or v == world:
                world_edges.append(e)
            else:
                internal.append(e)
        return internal, world_edges

    def _compute_node_degrees(self, component: TopologyComponent) -> dict[int, int]:
        """Compute the internal degree of every body node in ``component``.

        Only body-to-body edges contribute, so grounding/base edges to
        the world are ignored. Self-loops contribute ``2`` to the node's
        degree, matching the standard graph-theoretic definition.
        """
        world = component.world_node
        degrees: dict[int, int] = {int(n): 0 for n in (component.nodes or [])}
        internal_edges, _ = self._partition_edges(component, world)
        for e in internal_edges:
            u, v = e.nodes
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1
        return degrees

    def _build_internal_adjacency(
        self,
        body_nodes: list[int],
        internal_edges: list[GraphEdge],
        directed: bool,
    ) -> dict[int, list[tuple[int, GraphEdge]]]:
        """Build an adjacency mapping for the body-only subgraph.

        Each entry is ``current -> [(next_body, edge), ...]``, where
        ``edge`` is the source :class:`GraphEdge` (carrying the original
        ``(predecessor, successor)`` polarity, joint type, and index).
        For undirected components, both polarities are emitted so that
        BFS / parent-discovery can traverse either direction.
        """
        adj: dict[int, list[tuple[int, GraphEdge]]] = {n: [] for n in body_nodes}
        for e in internal_edges:
            u, v = e.nodes
            if u in adj:
                adj[u].append((v, e))
            if not directed and u != v and v in adj:
                adj[v].append((u, e))
        # Stable adjacency ordering keeps enumeration deterministic across runs
        for neighbors in adj.values():
            neighbors.sort(key=lambda x: (x[1].joint_index, x[0]))
        return adj

    @staticmethod
    def _bfs_distances(
        root: int,
        adj: dict[int, list[tuple[int, GraphEdge]]],
    ) -> dict[int, int]:
        """Run BFS from ``root`` and return a ``body -> distance`` map."""
        dist: dict[int, int] = {root: 0}
        queue: deque[int] = deque([root])
        while queue:
            u = queue.popleft()
            for v, _edge in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    ###
    # Internals - root selection
    ###

    def _select_root_candidates(
        self,
        component: TopologyComponent,
        roots: list[NodeType] | None,
        override_priorities: bool,
    ) -> list[int]:
        """Apply the root-selection priority cascade described in the class docstring.

        May mutate ``component.base_node`` / ``component.base_edge`` when
        the caller passes a single explicit root and the component had
        no previously-assigned base.
        """
        # Coerce the canonical `GraphNode` storage into raw int indices for the cascade
        # below. The downstream enumeration routines all consume int indices.
        body_nodes: list[int] = [int(n) for n in (component.nodes or [])]

        # 1. Explicit roots argument wins
        if roots is not None:
            if not roots:
                raise ValueError("`roots` must be a non-empty list when supplied.")
            roots_idx: list[int] = [int(r) for r in roots]
            unknown = [r for r in roots_idx if r not in body_nodes]
            if unknown:
                raise ValueError(
                    f"`roots` contains body indices not in component.nodes: {unknown!r}; valid bodies: {body_nodes!r}."
                )
            # Stamp base_node/base_edge in-place for a single-root request on a
            # component that has no pre-assigned base. When a grounding edge
            # exists for the chosen root, promote the first such edge via
            # `assign_base` (which clears the promoted edge from the grounding
            # lists and re-validates the component).
            if len(roots_idx) == 1 and component.base_edge is None:
                world = component.world_node
                root = roots_idx[0]
                for e in component.ground_edges or []:
                    u, v = e.nodes
                    if (u == world and v == root) or (v == world and u == root):
                        component.assign_base(base_node=root, base_edge=e)
                        break
            return roots_idx

        # 5'. Override → brute-force
        if override_priorities:
            return list(body_nodes)

        # 2. Use base_node when available
        if component.base_node is not None:
            return [int(component.base_node)]

        # 3. Use grounding nodes when available
        if self._prioritize_grounding_when_no_base and component.ground_nodes:
            return [int(n) for n in component.ground_nodes]

        # 4. Use the unique max-degree node
        degrees = self._compute_node_degrees(component)
        if not degrees:
            return []
        max_deg = max(degrees.values())
        max_nodes = [n for n, d in degrees.items() if d == max_deg]
        if len(max_nodes) == 1:
            return max_nodes

        # 5. Tie → brute-force over all body nodes
        return list(body_nodes)

    ###
    # Internals - per-root enumeration
    ###

    def _enumerate_min_depth_trees_from_root(
        self,
        component: TopologyComponent,
        root: int,
        traversal: SpanningTreeTraversal,
        max_count: int | None,
        directed: bool,
    ) -> list[TopologySpanningTree]:
        """Enumerate every minimum-depth spanning tree rooted at ``root``.

        Returns an empty list (rather than raising) when the body-only
        subgraph is not fully reachable from ``root`` under the chosen
        polarity rules.
        """
        body_nodes: list[int] = [int(n) for n in (component.nodes or [])]
        if root not in body_nodes:
            raise ValueError(f"Root `{root}` is not contained in component nodes `{body_nodes}`.")
        if max_count is not None and max_count <= 0:
            return []

        world = component.world_node
        internal_edges, world_edges = self._partition_edges(component, world)
        adj = self._build_internal_adjacency(body_nodes, internal_edges, directed)

        # BFS layering from the root drives both the depth check and the per-body parent search
        dist = self._bfs_distances(root, adj)
        if len(dist) != len(body_nodes):
            return []
        depth = max(dist.values())

        # Bucket bodies by BFS distance so parent enumeration scans only the preceding
        # layer (not all bodies). Order within each bucket matches ``body_nodes``.
        nodes_at_dist: dict[int, list[int]] = defaultdict(list)
        for n in body_nodes:
            nodes_at_dist[dist[n]].append(n)

        # Per non-root body, collect every layered parent-edge candidate. Edge
        # deduplication is maintained by the parser (`set(edges)`); within a
        # single `adj[p]` traversal, the source `GraphEdge` instances are unique
        # by construction.
        non_root_nodes = [n for n in body_nodes if n != root]
        parent_choices: dict[int, list[OrientedEdge]] = {}
        for x in non_root_nodes:
            dx = dist[x]
            choices: list[OrientedEdge] = []
            for p in nodes_at_dist.get(dx - 1, []):
                # Enumerate edges from layer-(dx-1) parent `p` into `x`. The source
                # `GraphEdge` carries the original `(predecessor, successor)` ordering
                # while the oriented variant always pairs `(parent, child) = (p, x)`;
                # reuse the source edge directly when its polarity already matches.
                for nxt, edge in adj[p]:
                    if nxt != x:
                        continue
                    oriented = (
                        edge
                        if edge.nodes == (p, x)
                        else GraphEdge(joint_type=edge.joint_type, joint_index=edge.joint_index, nodes=(p, x))
                    )
                    choices.append(OrientedEdge(original=edge, oriented=oriented))
            choices.sort(key=lambda c: (c.oriented.joint_index, c.oriented.nodes[0], c.oriented.nodes[1]))
            parent_choices[x] = choices

        # Resolve the (provisional) base edge for this root: the first
        # world edge whose body endpoint matches `root`. May be `None`
        # for isolated islands; `_build_tree` handles that.
        base_edge = self._select_base_edge_for_root(root, component, world, world_edges)

        # Cartesian product over per-body parent choices yields all min-depth trees
        choice_lists = [parent_choices[n] for n in non_root_nodes]

        trees: list[TopologySpanningTree] = []
        for choice_tuple in product(*choice_lists):
            chosen_arcs: dict[int, OrientedEdge] = dict(zip(non_root_nodes, choice_tuple, strict=True))
            tree = self._build_tree(
                component=component,
                root=root,
                base_edge=base_edge,
                chosen_arcs=chosen_arcs,
                world_edges=world_edges,
                internal_edges=internal_edges,
                depth=depth,
                traversal=traversal,
                directed=directed,
                world=world,
            )
            trees.append(tree)
            if max_count is not None and len(trees) >= max_count:
                break
        return trees

    @staticmethod
    def _select_base_edge_for_root(
        root: int,
        component: TopologyComponent,
        world: int,
        world_edges: list[GraphEdge],
    ) -> GraphEdge | None:
        """Pick a base edge for ``root`` in deterministic priority order.

        Preference: the component's already-assigned base edge if
        ``base_node == root``, then any grounding edge connecting
        ``root`` to ``world``, then the first ``world_edge`` involving
        ``root``. Returns ``None`` for isolated components with no
        world edges.
        """
        if component.base_node == root and component.base_edge is not None:
            return component.base_edge
        for e in component.ground_edges or []:
            u, v = e.nodes
            if (u == world and v == root) or (v == world and u == root):
                return e
        for e in world_edges:
            u, v = e.nodes
            if (u == world and v == root) or (v == world and u == root):
                return e
        return None

    ###
    # Internals - tree construction
    ###

    @staticmethod
    def _children_dict(
        chosen_arcs: dict[int, OrientedEdge],
    ) -> dict[int, list[int]]:
        """Group each parent's children body indices in stable order."""
        children: dict[int, list[int]] = defaultdict(list)
        for child, oedge in chosen_arcs.items():
            parent = oedge.oriented.nodes[0]
            children[parent].append(child)
        for plist in children.values():
            plist.sort()
        return children

    @staticmethod
    def _traversal_order(
        root: int,
        children_of: dict[int, list[int]],
        mode: SpanningTreeTraversal,
    ) -> list[int]:
        """Linearize the chosen tree in BFS or DFS preorder from ``root``."""
        if mode == "bfs":
            order: list[int] = [root]
            queue: deque[int] = deque([root])
            while queue:
                u = queue.popleft()
                for c in children_of.get(u, []):
                    order.append(c)
                    queue.append(c)
            return order

        stack: list[int] = [root]
        visited: set[int] = {root}
        order_dfs: list[int] = []
        while stack:
            u = stack.pop()
            order_dfs.append(u)
            for c in reversed(children_of.get(u, [])):
                if c not in visited:
                    visited.add(c)
                    stack.append(c)
        return order_dfs

    def _build_tree(
        self,
        component: TopologyComponent,
        root: int,
        base_edge: GraphEdge | None,
        chosen_arcs: dict[int, OrientedEdge],
        world_edges: list[GraphEdge],
        internal_edges: list[GraphEdge],
        depth: int,
        traversal: SpanningTreeTraversal,
        directed: bool,
        world: int,
    ) -> TopologySpanningTree:
        """Assemble a fully-populated :class:`TopologySpanningTree`.

        Local body positions follow ``traversal`` order (root at ``0``).
        See the module docstring for the indexing convention. The base
        edge (if any) becomes ``arcs[0]`` so the per-body and per-arc
        portions of the parallel arrays stay aligned.
        """
        children_of = self._children_dict(chosen_arcs)
        body_order = self._traversal_order(root, children_of, traversal)
        local_of: dict[int, int] = {b: i for i, b in enumerate(body_order)}
        num_bodies = len(body_order)

        # Tree arcs: arcs[i] is the global joint index of the joint
        # connecting body at local position i to its parent (or to
        # the world via the base edge for i == 0).
        arcs: list[int] = [0] * num_bodies
        parents: list[int] = [0] * num_bodies
        if base_edge is not None:
            arcs[0] = base_edge.joint_index
        else:
            # Isolated island has no base joint; mark
            # slot 0 with the joint-index sentinel.
            arcs[0] = NO_BASE_JOINT_INDEX
        parents[0] = DEFAULT_WORLD_NODE_INDEX
        for i in range(1, num_bodies):
            child = body_order[i]
            oedge = chosen_arcs[child]
            arcs[i] = oedge.oriented.joint_index
            parent_body = oedge.oriented.nodes[0]
            parents[i] = local_of[parent_body]

        # Per-body local-position lists, parallel to body_order
        children: list[list[int]] = [[] for _ in range(num_bodies)]
        for i in range(1, num_bodies):
            children[parents[i]].append(i)

        # Subtree (v): post-order accumulation. Each body's subtree
        # contains itself plus its descendants, expressed as local positions.
        subtree: list[list[int]] = [[i] for i in range(num_bodies)]
        for i in reversed(range(1, num_bodies)):
            subtree[parents[i]].extend(subtree[i])
        for s in subtree:
            s.sort()

        # Support (κ): per-body list of arc local positions on the path
        # from that body to the root, excluding the implicit world arc.
        support: list[list[int]] = [[] for _ in range(num_bodies)]
        for i in range(1, num_bodies):
            support[i] = support[parents[i]] + [i]

        # Chord joints: every joint in `component.edges` not used as an arc
        used_arc_joint_indices: set[int] = {arcs[i] for i in range(num_bodies) if arcs[i] != NO_BASE_JOINT_INDEX}
        chord_edges: list[GraphEdge] = []
        for e in component.edges or []:
            if e.joint_index not in used_arc_joint_indices:
                chord_edges.append(e)
        chord_joint_indices: list[int] = [e.joint_index for e in chord_edges]
        num_tree_chords = len(chord_joint_indices)

        # Per-joint predecessors / successors arrays. The first
        # `num_bodies` entries align with arcs (in body-order) so that
        # arc i has predecessor `parents[i]` (the world for the root)
        # and successor `i`. The remaining entries align with chords.
        # Local-frame sentinel for "world / no body" is
        # :data:`DEFAULT_WORLD_NODE_INDEX`.
        num_joints = num_bodies + num_tree_chords
        predecessors: list[int] = [DEFAULT_WORLD_NODE_INDEX] * num_joints
        successors: list[int] = [DEFAULT_WORLD_NODE_INDEX] * num_joints
        for i in range(num_bodies):
            if arcs[i] != NO_BASE_JOINT_INDEX:
                predecessors[i] = parents[i]
                successors[i] = i
            else:
                # Sentinel arc (no base edge): predecessor stays at the
                # world sentinel; successor records the body's own local
                # position so the parent-child relationship is recoverable.
                successors[i] = i
        # Chord segment: chord polarity is taken directly from the source
        # edge in `directed` mode, otherwise oriented from the lower BFS
        # layer to the higher one. World endpoints map to the world
        # sentinel in the local frame.
        for k, e in enumerate(chord_edges):
            j_idx = num_bodies + k
            u, v = e.nodes
            if directed:
                pu, pv = u, v
            else:
                pu, pv = self._oriented_chord_endpoints(u, v, local_of, world)
            predecessors[j_idx] = (
                local_of.get(pu, DEFAULT_WORLD_NODE_INDEX) if pu != world else DEFAULT_WORLD_NODE_INDEX
            )
            successors[j_idx] = local_of.get(pv, DEFAULT_WORLD_NODE_INDEX) if pv != world else DEFAULT_WORLD_NODE_INDEX

        # `num_tree_arcs` counts only real arcs (i.e. those whose joint
        # index is not the :data:`NO_BASE_JOINT_INDEX` sentinel),
        # matching the orphan convention in `_build_orphan_tree`.
        num_tree_arcs = sum(1 for a in arcs if a != NO_BASE_JOINT_INDEX)

        return TopologySpanningTree(
            traversal=traversal,
            depth=depth,
            directed=directed,
            num_bodies=num_bodies,
            num_joints=num_joints,
            num_tree_arcs=num_tree_arcs,
            num_tree_chords=num_tree_chords,
            component=component,
            root=root,
            arcs=arcs,
            chords=chord_joint_indices,
            predecessors=predecessors,
            successors=successors,
            parents=parents,
            support=support,
            children=children,
            subtree=subtree,
        )

    @staticmethod
    def _oriented_chord_endpoints(
        u: int,
        v: int,
        local_of: dict[int, int],
        world: int,
    ) -> tuple[int, int]:
        """Choose chord polarity favouring lower-local-position to higher.

        World endpoints are kept as-is so the caller can route them to
        the ``-1`` sentinel in the per-joint arrays.
        """
        if u == world or v == world:
            return (u, v)
        lu = local_of.get(u, -1)
        lv = local_of.get(v, -1)
        if lu <= lv:
            return (u, v)
        return (v, u)

    def _build_orphan_tree(
        self,
        component: TopologyComponent,
        body: int,
        traversal: SpanningTreeTraversal,
    ) -> TopologySpanningTree:
        """Construct the trivial single-body spanning tree.

        When the component carries a base edge it becomes ``arcs[0]``;
        otherwise ``arcs`` is empty. Any other world edges in the
        component become chords.
        """
        world = component.world_node
        _internal, world_edges = self._partition_edges(component, world)

        if component.base_edge is not None:
            base_edge = component.base_edge
        elif world_edges:
            base_edge = world_edges[0]
        else:
            base_edge = None

        if base_edge is not None:
            arcs = [base_edge.joint_index]
            parents = [DEFAULT_WORLD_NODE_INDEX]
            num_bodies = 1
            used = {base_edge.joint_index}
            chord_edges = [e for e in (component.edges or []) if e.joint_index not in used]
            chord_joint_indices = [e.joint_index for e in chord_edges]
            num_tree_chords = len(chord_joint_indices)
            num_joints = num_bodies + num_tree_chords
            predecessors = [DEFAULT_WORLD_NODE_INDEX] * num_joints
            successors = [DEFAULT_WORLD_NODE_INDEX] * num_joints
            successors[0] = 0
            for k, e in enumerate(chord_edges):
                j_idx = num_bodies + k
                u, v = e.nodes
                pu = DEFAULT_WORLD_NODE_INDEX if u == world else (0 if u == body else DEFAULT_WORLD_NODE_INDEX)
                pv = DEFAULT_WORLD_NODE_INDEX if v == world else (0 if v == body else DEFAULT_WORLD_NODE_INDEX)
                predecessors[j_idx] = pu
                successors[j_idx] = pv
        else:
            arcs = []
            parents = [DEFAULT_WORLD_NODE_INDEX]
            num_bodies = 1
            chord_joint_indices = [e.joint_index for e in (component.edges or [])]
            num_tree_chords = len(chord_joint_indices)
            num_joints = num_tree_chords
            predecessors = [DEFAULT_WORLD_NODE_INDEX] * num_joints
            successors = [DEFAULT_WORLD_NODE_INDEX] * num_joints
            for k, e in enumerate(component.edges or []):
                u, v = e.nodes
                pu = DEFAULT_WORLD_NODE_INDEX if u == world else (0 if u == body else DEFAULT_WORLD_NODE_INDEX)
                pv = DEFAULT_WORLD_NODE_INDEX if v == world else (0 if v == body else DEFAULT_WORLD_NODE_INDEX)
                predecessors[k] = pu
                successors[k] = pv

        return TopologySpanningTree(
            traversal=traversal,
            depth=0,
            directed=self._directed,
            num_bodies=num_bodies,
            num_joints=num_joints,
            num_tree_arcs=len(arcs),
            num_tree_chords=num_tree_chords,
            component=component,
            root=body,
            arcs=arcs,
            chords=chord_joint_indices,
            predecessors=predecessors,
            successors=successors,
            parents=parents,
            support=[[]],
            children=[[]],
            subtree=[[0]],
        )
