# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Spanning-tree generator backends for the topology subsystem.

This module ships concrete implementations of
:class:`TopologySpanningTreeGeneratorBase` that the :class:`TopologyGraph`
pipeline can use to enumerate spanning-tree candidates for each component
of a topology graph.

Conventions
-----------
The generators in this module use **traversal-order local body indexing**
inside each returned :class:`TopologySpanningTree`. Body local position
``0`` is the spanning tree's root body, and the remaining bodies are
numbered ``1..N_B - 1`` in BFS or DFS discovery order from the root. All
per-body fields (``parents``, ``children``, ``support``, ``subtree``)
store local body positions; per-joint fields (``predecessors``,
``successors``) store local body positions for both the arc segment and
the chord segment of the array, with the implicit world body keeping the
sentinel value ``-1``.

The ``arcs`` list stores **global** joint indices and is parallel to the
local body positions: ``arcs[i]`` is the global joint index of the joint
connecting the body at local position ``i`` to its parent body (or, for
``i == 0``, to the implicit world node via the component's base edge).
The ``chords`` list stores the global joint indices of all remaining
joints in the source component.

This guarantees Featherstone's regular-numbering invariant
``parents[i] < i`` for all ``i >= 1`` *by construction* (since traversal
discovers parents before children), so the optional global re-numbering
step (``TODO IMPLEMENTATIONS #4`` in the kamino ``core.topology`` module)
is only ever needed to remap local positions back to global
:class:`TopologyGraph` indices.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import product

from ..core.types import override
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    EdgeType,
    NodeType,
    OrientedEdge,
    SpanningTreeTraversal,
    TopologyComponent,
    TopologySpanningTree,
    TopologySpanningTreeGeneratorBase,
    _validate_max_candidates,
    _validate_traversal_mode,
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
    """A :class:`TopologySpanningTreeGeneratorBase` backend that enumerates
    minimum-depth spanning-tree candidates for each component subgraph.

    Given a chosen root body, a *minimum-depth* spanning tree is a tree
    whose maximum root-to-leaf arc count equals the eccentricity of the
    root in the component's body-only subgraph. The set of such trees is
    enumerated by allowing each non-root body to pick any incoming
    parent edge from a strictly preceding BFS layer.

    The enumeration source for the root body is determined by the
    following priority cascade (see ``TODO IMPLEMENTATIONS`` item ``2d``
    in the kamino ``core.topology`` module), which can be overridden per-call:

    1. An explicit ``roots`` argument always wins. If a single root is
       given and the source component had no auto-assigned base, the
       root and its first connecting grounding edge (if any) are
       written back to ``component.base_node`` / ``component.base_edge``.
    2. Otherwise, ``component.base_node`` is used as the unique root.
    3. Otherwise, ``component.ground_nodes`` is used (one root per
       grounding node).
    4. Otherwise, the body with the unique maximum internal degree is
       used.
    5. Otherwise (degree tie or ``override_priorities=True``), a
       brute-force enumeration over every body node is performed.

    Args:
        directed:
            When ``True``, treat each input edge's ``(predecessor,
            successor)`` order as the only admissible polarity. When
            ``False`` (default), allow either polarity when building the
            tree.
        traversal_mode:
            Default traversal order (``"bfs"`` or ``"dfs"``) used to
            assign local body positions inside each generated tree.
            Overridable per call via
            :meth:`generate_spanning_trees`.
        max_candidates:
            Default cap on the number of candidate trees produced per
            component. Overridable per call via
            :meth:`generate_spanning_trees`. ``None`` means no cap.
        override_priorities:
            Default value of the per-call ``override_priorities``
            keyword argument. When ``True``, skip the priority cascade
            and brute-force every body node as a root.
        prioritize_balanced:
            Default value of the per-call ``prioritize_balanced``
            keyword argument. When ``True``, candidate trees are
            re-ordered (and truncated, if ``max_candidates`` is set) by
            an imbalance metric that prefers trees whose internal nodes
            distribute children evenly.
        prioritize_grounding_when_no_base:
            When ``True`` (default), use ``component.ground_nodes`` as
            roots in step ``3`` of the priority cascade above. When
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
        _validate_traversal_mode(traversal_mode)
        _validate_max_candidates(max_candidates)
        self._directed: bool = directed
        self._traversal_mode: SpanningTreeTraversal = traversal_mode
        self._max_candidates: int | None = max_candidates
        self._override_priorities: bool = override_priorities
        self._prioritize_balanced: bool = prioritize_balanced
        self._prioritize_grounding_when_no_base: bool = prioritize_grounding_when_no_base

    ###
    # Public API
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
        when supplied. See the class-level docstring for the meaning of
        each argument.

        Args:
            component: The component subgraph to enumerate spanning trees for.
            traversal_mode: Per-call override of the constructor default.
            max_candidates: Per-call override of the constructor default.
            roots: Optional explicit root list (priority cascade step ``1``).
            override_priorities: Per-call override of the constructor default.
            prioritize_balanced: Per-call override of the constructor default.

        Returns:
            A list of :class:`TopologySpanningTree` candidates, each
            populated with the schema described in the module docstring.
        """
        # Resolve effective settings: per-call takes precedence over constructor defaults.
        traversal: SpanningTreeTraversal = traversal_mode if traversal_mode is not None else self._traversal_mode
        _validate_traversal_mode(traversal)
        cap: int | None = max_candidates if max_candidates is not None else self._max_candidates
        _validate_max_candidates(cap)
        ovp: bool = self._override_priorities if override_priorities is None else override_priorities
        bal: bool = self._prioritize_balanced if prioritize_balanced is None else prioritize_balanced

        # Validate the component
        if component is None:
            raise ValueError("`component` must not be None.")
        body_nodes = list(component.nodes) if component.nodes is not None else []
        if not body_nodes:
            raise ValueError("`component.nodes` must contain at least one body node.")

        # Orphan special case: a single-body component has exactly one
        # trivial spanning tree (with the body as root and at most the
        # base edge as the sole arc). Skip the enumeration machinery.
        if len(body_nodes) == 1:
            return [self._build_orphan_tree(component, body_nodes[0], traversal)]

        # Resolve root candidates per the priority cascade
        root_candidates = self._select_root_candidates(component, roots, ovp)
        if not root_candidates:
            raise RuntimeError(
                f"Could not select any root candidate for component with nodes `{body_nodes}` "
                f"(roots={roots!r}, override_priorities={ovp})."
            )

        # Enumerate per-root with a global cap that lets the loop short-circuit
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
        # original enumeration order for trees that tie on imbalance.
        if bal and len(candidates) > 1:
            candidates.sort(key=self._balanced_score)

        return candidates

    ###
    # Helpers - graph utilities (TODO 2c)
    ###

    @staticmethod
    def _world_index(component: TopologyComponent) -> int:
        """Returns the world-node sentinel index used by ``component``'s edges.

        Components carry their world node implicitly via edge endpoints.
        We treat any negative endpoint as ``world`` for parsing
        purposes, defaulting to :data:`DEFAULT_WORLD_NODE_INDEX` when
        no world endpoint appears in the edge list.
        """
        if component.edges:
            for _t, _j, (u, v) in component.edges:
                if u < 0:
                    return u
                if v < 0:
                    return v
        return DEFAULT_WORLD_NODE_INDEX

    def _partition_edges(self, component: TopologyComponent, world: int) -> tuple[list[EdgeType], list[EdgeType]]:
        """Split ``component.edges`` into ``(internal, world)`` lists.

        Internal edges are body-to-body edges; world edges are those
        with the world sentinel as one endpoint.
        """
        internal: list[EdgeType] = []
        world_edges: list[EdgeType] = []
        for e in component.edges or []:
            _t, _j, (u, v) = e
            if u == world or v == world:
                world_edges.append(e)
            else:
                internal.append(e)
        return internal, world_edges

    def _compute_node_degrees(self, component: TopologyComponent) -> dict[NodeType, int]:
        """Compute the internal degree of every body node in ``component``.

        Implements TODO IMPLEMENTATIONS item ``2c``: only body-to-body
        edges contribute, so grounding/base edges to the world are
        ignored. Self-loops (``u == v``) contribute ``2`` to the
        node's degree, matching the standard graph-theoretic
        definition.
        """
        world = self._world_index(component)
        degrees: dict[NodeType, int] = dict.fromkeys(component.nodes or [], 0)
        internal_edges, _ = self._partition_edges(component, world)
        for _t, _j, (u, v) in internal_edges:
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1
        return degrees

    def _build_internal_adjacency(
        self,
        body_nodes: list[NodeType],
        internal_edges: list[EdgeType],
        directed: bool,
    ) -> dict[NodeType, list[tuple[NodeType, int, int, tuple[NodeType, NodeType]]]]:
        """Build an adjacency mapping for the body-only subgraph.

        Each entry is ``current -> [(next_body, joint_type, joint_index, original_uv), ...]``.
        For undirected components, both polarities are emitted so that
        BFS / parent-discovery can traverse either direction.
        """
        adj: dict[NodeType, list[tuple[NodeType, int, int, tuple[NodeType, NodeType]]]] = {n: [] for n in body_nodes}
        for jt, jid, (u, v) in internal_edges:
            if u in adj:
                adj[u].append((v, jt, jid, (u, v)))
            if not directed and u != v and v in adj:
                adj[v].append((u, jt, jid, (u, v)))
        # Stable adjacency ordering keeps enumeration deterministic across runs
        for neighbors in adj.values():
            neighbors.sort(key=lambda x: (x[2], x[0]))
        return adj

    @staticmethod
    def _bfs_distances(
        root: NodeType,
        adj: dict[NodeType, list[tuple[NodeType, int, int, tuple[NodeType, NodeType]]]],
    ) -> dict[NodeType, int]:
        """Standard BFS that returns ``body -> distance`` for every reachable body."""
        dist: dict[NodeType, int] = {root: 0}
        queue: deque[NodeType] = deque([root])
        while queue:
            u = queue.popleft()
            for v, _jt, _jid, _orig in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    ###
    # Helpers - root selection (TODO 2d)
    ###

    def _select_root_candidates(
        self,
        component: TopologyComponent,
        roots: list[NodeType] | None,
        override_priorities: bool,
    ) -> list[NodeType]:
        """Apply the root-selection priority cascade described in the class docstring.

        Implements TODO IMPLEMENTATIONS item ``2d``. May mutate
        ``component.base_node`` / ``component.base_edge`` when the
        caller passes a single explicit root and the component had no
        previously-assigned base (per spec line ``2524``).
        """
        body_nodes = list(component.nodes or [])

        # 1. Explicit roots argument wins
        if roots is not None:
            if not roots:
                raise ValueError("`roots` must be a non-empty list when supplied.")
            unknown = [r for r in roots if r not in body_nodes]
            if unknown:
                raise ValueError(
                    f"`roots` contains body indices not in component.nodes: {unknown!r}; valid bodies: {body_nodes!r}."
                )
            # Stamp base_node/base_edge in-place when admissible. This is
            # only done for a single-root request on a component that
            # has no pre-assigned base, matching the spec's admissibility
            # condition. When a grounding edge exists for the chosen
            # root, promote the first such edge; otherwise leave the
            # base unset so a later base-selector module can decide.
            if len(roots) == 1 and component.base_edge is None:
                world = self._world_index(component)
                root = roots[0]
                for e in component.ground_edges or []:
                    _t, _j, (u, v) = e
                    if (u == world and v == root) or (v == world and u == root):
                        component.base_node = root
                        component.base_edge = e
                        component.is_connected = True
                        # Drop the promoted edge from the grounding list. Recompute
                        # `ground_nodes` from the remaining grounding edges rather
                        # than blindly removing `root`: a body can have several
                        # grounding edges (e.g. a Stewart-platform leg), and
                        # `root` must remain a grounding node when any of those
                        # edges still references it. Otherwise the
                        # ``set(ground_nodes) == implied_endpoints_of(ground_edges)``
                        # invariant in :meth:`TopologyComponent.__post_init__`
                        # is silently violated.
                        if component.ground_edges is not None:
                            component.ground_edges = [g for g in component.ground_edges if g != e]
                            if component.ground_nodes is not None:
                                remaining = {n for _, _, pair in component.ground_edges for n in pair if n != world}
                                component.ground_nodes = sorted(remaining)
                        break
            return list(roots)

        # 5'. Override → brute-force
        if override_priorities:
            return list(body_nodes)

        # 2. Use base_node when available
        if component.base_node is not None:
            return [component.base_node]

        # 3. Use grounding nodes when available
        if self._prioritize_grounding_when_no_base and component.ground_nodes:
            return list(component.ground_nodes)

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
    # Helpers - per-root enumeration (TODO 2b)
    ###

    def _enumerate_min_depth_trees_from_root(
        self,
        component: TopologyComponent,
        root: NodeType,
        traversal: SpanningTreeTraversal,
        max_count: int | None,
        directed: bool,
    ) -> list[TopologySpanningTree]:
        """Enumerate every minimum-depth spanning tree rooted at ``root``.

        Returns an empty list (rather than raising) when the body-only
        subgraph is not fully reachable from ``root`` under the chosen
        polarity rules — that root simply admits no spanning tree.
        Implements TODO IMPLEMENTATIONS item ``2b``.
        """
        body_nodes = list(component.nodes or [])
        if root not in body_nodes:
            raise ValueError(f"Root `{root}` is not contained in component nodes `{body_nodes}`.")
        if max_count is not None and max_count <= 0:
            return []

        world = self._world_index(component)
        internal_edges, world_edges = self._partition_edges(component, world)
        adj = self._build_internal_adjacency(body_nodes, internal_edges, directed)

        # BFS layering from the root drives both the depth check and the per-body parent search
        dist = self._bfs_distances(root, adj)
        if len(dist) != len(body_nodes):
            return []
        depth = max(dist.values())

        # Bucket bodies by BFS distance so parent enumeration scans only the preceding
        # layer (not all bodies). Order within each bucket matches ``body_nodes``.
        nodes_at_dist: dict[int, list[NodeType]] = defaultdict(list)
        for n in body_nodes:
            nodes_at_dist[dist[n]].append(n)

        # Per non-root body: collect every layered parent-edge candidate
        non_root_nodes = [n for n in body_nodes if n != root]
        parent_choices: dict[NodeType, list[OrientedEdge]] = {}
        for x in non_root_nodes:
            dx = dist[x]
            choices: list[OrientedEdge] = []
            seen: set[tuple[int, NodeType, NodeType]] = set()
            for p in nodes_at_dist.get(dx - 1, []):
                # Enumerate edges p -> x; the recorded `oriented` pair is (p, x)
                for nxt, jt, jid, original in adj[p]:
                    if nxt != x:
                        continue
                    key = (jid, p, x)
                    if key in seen:
                        continue
                    seen.add(key)
                    choices.append(OrientedEdge(joint_type=jt, joint_index=jid, original=original, oriented=(p, x)))
            if not choices:
                # The graph claims `x` is reachable yet no layered parent exists.
                # This indicates an inconsistency in the BFS distances and is treated
                # as a hard error so callers can surface bad inputs immediately.
                raise RuntimeError(f"No layered parent edge found for body `{x}` at depth `{dx}` from root `{root}`.")
            choices.sort(key=lambda c: (c.joint_index, c.oriented[0], c.oriented[1]))
            parent_choices[x] = choices

        # Resolve the (provisional) base edge for this root: the first
        # world edge whose body endpoint matches `root`. May be `None`
        # for isolated islands; `_build_tree` handles that.
        base_edge = self._select_base_edge_for_root(root, component, world, world_edges)

        # Cartesian product over per-body parent choices yields all min-depth trees
        choice_lists = [parent_choices[n] for n in non_root_nodes]

        trees: list[TopologySpanningTree] = []
        for choice_tuple in product(*choice_lists):
            chosen_arcs: dict[NodeType, OrientedEdge] = dict(zip(non_root_nodes, choice_tuple, strict=True))
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
        root: NodeType,
        component: TopologyComponent,
        world: int,
        world_edges: list[EdgeType],
    ) -> EdgeType | None:
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
            _t, _j, (u, v) = e
            if (u == world and v == root) or (v == world and u == root):
                return e
        for e in world_edges:
            _t, _j, (u, v) = e
            if (u == world and v == root) or (v == world and u == root):
                return e
        return None

    ###
    # Helpers - tree construction (TopologySpanningTree population)
    ###

    @staticmethod
    def _children_dict(
        chosen_arcs: dict[NodeType, OrientedEdge],
    ) -> dict[NodeType, list[NodeType]]:
        """Group each parent's children body indices in stable order."""
        children: dict[NodeType, list[NodeType]] = defaultdict(list)
        for child, oedge in chosen_arcs.items():
            parent = oedge.oriented[0]
            children[parent].append(child)
        for plist in children.values():
            plist.sort()
        return children

    @staticmethod
    def _traversal_order(
        root: NodeType,
        children_of: dict[NodeType, list[NodeType]],
        mode: SpanningTreeTraversal,
    ) -> list[NodeType]:
        """Linearize the chosen tree in BFS or DFS preorder from ``root``."""
        if mode == "bfs":
            order: list[NodeType] = [root]
            queue: deque[NodeType] = deque([root])
            while queue:
                u = queue.popleft()
                for c in children_of.get(u, []):
                    order.append(c)
                    queue.append(c)
            return order

        stack: list[NodeType] = [root]
        visited: set[NodeType] = {root}
        order_dfs: list[NodeType] = []
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
        root: NodeType,
        base_edge: EdgeType | None,
        chosen_arcs: dict[NodeType, OrientedEdge],
        world_edges: list[EdgeType],
        internal_edges: list[EdgeType],
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
        # Linearize bodies in traversal order (root first); build local index map
        children_of = self._children_dict(chosen_arcs)
        body_order = self._traversal_order(root, children_of, traversal)
        local_of: dict[NodeType, int] = {b: i for i, b in enumerate(body_order)}
        num_bodies = len(body_order)

        # Tree arcs: arcs[i] is the global joint index of the joint
        # connecting body at local position i to its parent (or to
        # the world via the base edge for i == 0).
        arcs: list[int] = [0] * num_bodies
        parents: list[int] = [0] * num_bodies
        # arcs[0]: the base edge connects the root to the world
        if base_edge is not None:
            arcs[0] = base_edge[1]
        else:
            # Sentinel: -1 indicates "no base joint" (isolated island).
            # This is consistent with Featherstone's "world has no joint"
            # and matches the orphan handling in `_build_orphan_tree`.
            arcs[0] = -1
        parents[0] = -1
        for i in range(1, num_bodies):
            child = body_order[i]
            oedge = chosen_arcs[child]
            arcs[i] = oedge.joint_index
            parent_body = oedge.oriented[0]
            parents[i] = local_of[parent_body]

        # Children: per-body local-position lists, parallel to body_order
        children: list[list[int]] = [[] for _ in range(num_bodies)]
        for i in range(1, num_bodies):
            children[parents[i]].append(i)

        # Subtree (v): post-order accumulation. Each body's subtree
        # contains itself plus its descendants, expressed as local positions.
        subtree: list[list[int]] = [[i] for i in range(num_bodies)]
        # Process in reverse traversal order so children are processed before parents.
        for i in reversed(range(1, num_bodies)):
            subtree[parents[i]].extend(subtree[i])
        # Stable per-list sorting keeps the field deterministic and easy to compare in tests.
        for s in subtree:
            s.sort()

        # Support (κ): per-body list of arc local positions on the path
        # from that body to the root, excluding the implicit world arc.
        # Equivalently, support[i] = support[parents[i]] + [i] for i > 0.
        support: list[list[int]] = [[] for _ in range(num_bodies)]
        for i in range(1, num_bodies):
            support[i] = support[parents[i]] + [i]

        # Chord joints: every joint in `component.edges` not used as an arc
        used_arc_joint_indices: set[int] = {arcs[i] for i in range(num_bodies) if arcs[i] >= 0}
        chord_edges: list[EdgeType] = []
        for e in component.edges or []:
            jid = e[1]
            if jid not in used_arc_joint_indices:
                chord_edges.append(e)
        chord_joint_indices: list[int] = [e[1] for e in chord_edges]
        num_tree_chords = len(chord_joint_indices)

        # Per-joint predecessors / successors arrays. The first
        # `num_bodies` entries align with arcs (in body-order) so that
        # arc i has predecessor `parents[i]` (or -1 for the root) and
        # successor `i`. The remaining entries align with chords.
        num_joints = num_bodies + num_tree_chords
        predecessors: list[int] = [-1] * num_joints
        successors: list[int] = [-1] * num_joints
        # Arc segment
        for i in range(num_bodies):
            if arcs[i] >= 0:
                predecessors[i] = parents[i]
                successors[i] = i
            else:
                # Sentinel arc (no base edge): predecessor and successor unset (-1)
                # but we still record the body's own local position as the successor
                # so the parent-child relationship of the tree is recoverable.
                successors[i] = i
        # Chord segment: chord polarity is taken directly from the
        # source edge for `directed` mode, otherwise oriented from the
        # lower BFS layer to the higher one to match the arc convention.
        # World endpoints map to the local index `-1`.
        for k, (_jt, _jid, (u, v)) in enumerate(chord_edges):
            j_idx = num_bodies + k
            if directed:
                pu, pv = u, v
            else:
                pu, pv = self._oriented_chord_endpoints(u, v, local_of, world)
            predecessors[j_idx] = local_of.get(pu, -1) if pu != world else -1
            successors[j_idx] = local_of.get(pv, -1) if pv != world else -1

        # `num_tree_arcs` counts only **real** arcs (joint indices ≥ 0),
        # matching the orphan convention in `_build_orphan_tree`. For an
        # isolated island ``arcs[0] == -1`` is a sentinel for "no base
        # joint" and must not be counted.
        num_tree_arcs = sum(1 for a in arcs if a >= 0)

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
        u: NodeType,
        v: NodeType,
        local_of: dict[NodeType, int],
        world: int,
    ) -> tuple[NodeType, NodeType]:
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
        body: NodeType,
        traversal: SpanningTreeTraversal,
    ) -> TopologySpanningTree:
        """Construct the trivial single-body spanning tree.

        Used for orphan (single-node) components. When the component
        carries a base edge it becomes ``arcs[0]``; otherwise ``arcs``
        is empty. Any other world edges in the component become chords.
        """
        world = self._world_index(component)
        _internal, world_edges = self._partition_edges(component, world)

        if component.base_edge is not None:
            base_edge = component.base_edge
        elif world_edges:
            base_edge = world_edges[0]
        else:
            base_edge = None

        if base_edge is not None:
            arcs = [base_edge[1]]
            parents = [-1]
            num_bodies = 1
            used = {base_edge[1]}
            chord_edges = [e for e in (component.edges or []) if e[1] not in used]
            chord_joint_indices = [e[1] for e in chord_edges]
            num_tree_chords = len(chord_joint_indices)
            num_joints = num_bodies + num_tree_chords
            predecessors = [-1] * num_joints
            successors = [-1] * num_joints
            successors[0] = 0
            for k, (_jt, _jid, (u, v)) in enumerate(chord_edges):
                j_idx = num_bodies + k
                pu = -1 if u == world else (0 if u == body else -1)
                pv = -1 if v == world else (0 if v == body else -1)
                predecessors[j_idx] = pu
                successors[j_idx] = pv
        else:
            arcs = []
            parents = [-1]
            num_bodies = 1
            chord_joint_indices = [e[1] for e in (component.edges or [])]
            num_tree_chords = len(chord_joint_indices)
            num_joints = num_tree_chords
            predecessors = [-1] * num_joints
            successors = [-1] * num_joints
            for k, (_jt, _jid, (u, v)) in enumerate(component.edges or []):
                pu = -1 if u == world else (0 if u == body else -1)
                pv = -1 if v == world else (0 if v == body else -1)
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

    ###
    # Helpers - candidate scoring
    ###

    @staticmethod
    def _balanced_score(tree: TopologySpanningTree) -> int:
        """Sum of squared per-parent child counts.

        Smaller scores correspond to more balanced trees. For a fixed
        depth the minimum is achieved when every internal node has the
        same number of children, and the maximum is reached for a
        single-spine tree where one parent owns ``N_B - 1`` children.

        ``tree.children`` is always populated by the generator's tree
        builders; an unset value here indicates a malformed candidate
        and is surfaced loudly rather than silently scored as ``0``.
        """
        if tree.children is None:
            raise ValueError("Cannot score a TopologySpanningTree with `children=None`; the tree is malformed.")
        return sum(len(cs) * len(cs) for cs in tree.children)
