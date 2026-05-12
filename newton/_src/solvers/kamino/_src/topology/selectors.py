# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Component base-selector and spanning-tree selector backends.

Concrete implementations of :class:`TopologyComponentBaseSelectorBase`
and :class:`TopologySpanningTreeSelectorBase` used by the
:class:`TopologyGraph` pipeline:

- :class:`TopologyHeaviestBodyBaseSelector`: picks the heaviest body as
  the base node and an incident grounding edge (if any), otherwise
  synthesizes a 6-DoF FREE base edge to the world.
- :class:`TopologyMinimumDepthSpanningTreeSelector`: picks the minimum
  depth (and optionally most balanced) candidate per component.
"""

from __future__ import annotations

from collections import defaultdict, deque

from ..core.bodies import RigidBodyDescriptor
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    NO_BASE_JOINT_INDEX,
    GraphEdge,
    GraphNode,
    TopologyComponent,
    TopologyComponentBaseSelectorBase,
    TopologySpanningTree,
    TopologySpanningTreeSelectorBase,
)

###
# Module interface
###

__all__ = [
    "TopologyHeaviestBodyBaseSelector",
    "TopologyMinimumDepthSpanningTreeSelector",
    "TopologyNamedBodyBaseSelector",
]


###
# Base Node/Edge Selectors
###


class TopologyHeaviestBodyBaseSelector(TopologyComponentBaseSelectorBase):
    """
    Base-selector backend that picks the heaviest body as the component base.

    Among the bodies in ``component.nodes``, picks the one with the
    largest mass :attr:`RigidBodyDescriptor.m_i`, breaking ties by lowest
    body index. The base edge is then chosen as follows:

    1. If the heaviest body has any incident grounding edge in
       ``component.ground_edges``, pick one — preferring a 6-DoF FREE
       joint when ``prefer_free_when_available=True``.
    2. Otherwise (e.g. an isolated island, or a Stewart-platform-style
       end-effector body without its own grounding), synthesize a
       6-DoF FREE base edge connecting the body to the world.

    The returned ``(base_node, base_edge)`` is purely functional: this
    selector does not mutate the input ``component``. The orchestrator
    :meth:`TopologyGraph.select_component_bases` is responsible for
    committing the returned base via
    :meth:`TopologyComponent.assign_base`.

    Synthesized base edges carry the ``NO_BASE_JOINT_INDEX`` sentinel
    on their ``joint_index`` to flag them for the orchestrator. The
    orchestrator (:class:`TopologyGraph`) detects that sentinel and
    re-issues the edge with a fresh provisional joint index of the form
    ``NJ + k`` (where ``NJ`` is the number of user-supplied edges and
    ``k`` counts the synthetic edges committed so far) before committing
    it to the component, so multiple isolated components can each be
    given a unique synthetic base edge.

    Args:
        world_node: The world-node sentinel used when synthesizing a
            FREE base edge; must be a negative integer.
        prefer_free_when_available: When ``True`` (default), prefer a
            6-DoF FREE joint over other grounding-edge types incident
            to the chosen body.
    """

    ###
    # Construction
    ###

    def __init__(
        self,
        *,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        prefer_free_when_available: bool = True,
    ) -> None:
        if not isinstance(world_node, int):
            raise TypeError(f"`world_node` must be an integer; got {type(world_node).__name__}.")
        if world_node >= 0:
            raise ValueError(
                f"`world_node` must be a negative integer (sentinel for the implicit world); got {world_node}."
            )
        self._world_node: int = world_node
        self._prefer_free_when_available: bool = prefer_free_when_available

    ###
    # Public API
    ###

    @override
    def select_base(
        self,
        component: TopologyComponent,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> tuple[GraphNode, GraphEdge]:
        """
        Pick the heaviest body as base node and a corresponding base edge.

        Args:
            component:
                The component subgraph to select a base for; must have a non-empty ``nodes`` list.
            bodies:
                Per-world list of :class:`RigidBodyDescriptor`. The
                selector reads :attr:`RigidBodyDescriptor.m_i` from
                ``bodies[node]`` to score candidates.
            joints:
                Currently unused; accepted for interface compatibility.

        Returns:
            A ``(base_node, base_edge)`` pair where ``base_node`` is the
            canonical :class:`GraphNode` for the heaviest body in
            ``component.nodes`` (preserving any optional metadata) and
            ``base_edge`` is either an incident grounding edge of that
            body or a synthesized 6-DoF FREE edge connecting it to the
            world.

        Raises:
            ValueError:
                If ``component`` is ``None`` or has no nodes, ``bodies``
                is ``None``, or if any node is out of range of ``bodies``.
        """
        # Ensure component and bodies are provided and valid
        if component is None:
            raise ValueError("`component` must not be None.")
        if bodies is None:
            raise ValueError("`bodies` must be provided to select the heaviest base body.")

        # Create a local copy of the component nodes list to validate and work with,
        # ensure it is not empty and that all nodes are within the valid range.
        nodes: list[GraphNode] = list(component.nodes) if component.nodes is not None else []
        if not nodes:
            raise ValueError("`component.nodes` must contain at least one body node.")
        for n in nodes:
            idx = int(n)
            if idx < 0 or idx >= len(bodies):
                raise ValueError(f"Body node `{n}` is out of range for `bodies` (size={len(bodies)}).")

        # Score by (mass, -index) so the tie-breaker prefers the lowest body index.
        # Returning the originating `GraphNode` preserves any optional `name` metadata
        # for downstream consumers.
        base_node: GraphNode = max(nodes, key=lambda n: (bodies[int(n)].m_i, -int(n)))
        base_idx: int = int(base_node)

        # Collect grounding edges incident to the heaviest body. Coerce defensively
        # in case a caller mutated `ground_edges` post-construction with raw tuples.
        found_base_edge: list[GraphEdge] = []
        for e in component.ground_edges or []:
            edge = GraphEdge.from_input(e)
            u, v = edge.nodes
            if u == base_idx or v == base_idx:
                found_base_edge.append(edge)
        if found_base_edge:
            if self._prefer_free_when_available:
                found_free_base_edge = [e for e in found_base_edge if e.joint_type == JointDoFType.FREE]
                if found_free_base_edge:
                    return base_node, found_free_base_edge[0]
            return base_node, found_base_edge[0]

        # If no grounding edge is incident to the heaviest body, synthesize a 6-DoF FREE base edge to the world.
        synthetic_edge = GraphEdge(
            joint_type=int(JointDoFType.FREE),
            joint_index=int(NO_BASE_JOINT_INDEX),
            nodes=(self._world_node, base_idx),
        )
        return base_node, synthetic_edge


class TopologyNamedBodyBaseSelector(TopologyComponentBaseSelectorBase):
    """
    Base-selector backend that picks the body with a specific name as the
    component base.
    """

    ###
    # Construction
    ###

    def __init__(
        self,
        body_name: str,
        *,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        prefer_free_when_available: bool = True,
    ) -> None:
        if not isinstance(world_node, int):
            raise TypeError(f"`world_node` must be an integer; got {type(world_node).__name__}.")
        if world_node >= 0:
            raise ValueError(
                f"`world_node` must be a negative integer (sentinel for the implicit world); got {world_node}."
            )
        self._body_name: int = body_name
        self._world_node: int = world_node
        self._prefer_free_when_available: bool = prefer_free_when_available

    ###
    # Public API
    ###

    @override
    def select_base(
        self,
        component: TopologyComponent,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> tuple[GraphNode, GraphEdge]:
        """
        Pick the body with a specific index as the base node and a corresponding base edge.

        Args:
            component:
                The component subgraph to select a base for; must have a non-empty ``nodes`` list.
            bodies:
                Per-world list of :class:`RigidBodyDescriptor`. The
                selector reads :attr:`RigidBodyDescriptor.m_i` from
                ``bodies[node]`` to score candidates.
            joints:
                Currently unused; accepted for interface compatibility.

        Returns:
            A ``(base_node, base_edge)`` pair where ``base_node`` is the
            canonical :class:`GraphNode` for the body with a specific index in
            ``component.nodes`` (preserving any optional metadata) and
            ``base_edge`` is either an incident grounding edge of that
            body or a synthesized 6-DoF FREE edge connecting it to the
            world.

        Raises:
            ValueError:
                If ``component`` is ``None`` or has no nodes, ``bodies``
                is ``None``, or if any node is out of range of ``bodies``.
        """
        # Ensure component and bodies are provided and valid
        if component is None:
            raise ValueError("`component` must not be None.")
        if not component.nodes:
            raise ValueError("`component.nodes` must contain at least one body node.")
        if bodies is None:
            raise ValueError("`bodies` must be provided to select the base body.")

        # Get index and node for body name
        base_idx: int = -1
        for b_id, b in enumerate(bodies):
            if b.name == self._body_name:
                base_idx = b_id
                break
        if base_idx < 0:
            raise ValueError(f"Body name `{self._body_name}` not found in list of bodies.")
        base_node: GraphNode | None = None
        for n in component.nodes:
            if int(n) == base_idx:
                base_node = n
        if base_node is None:
            raise ValueError(f"Body index `{base_idx}` not found in list of nodes.")

        # Collect grounding edges incident to the base body. Coerce defensively
        # in case a caller mutated `ground_edges` post-construction with raw tuples.
        found_base_edge: list[GraphEdge] = []
        for e in component.ground_edges or []:
            edge = GraphEdge.from_input(e)
            u, v = edge.nodes
            if u == base_idx or v == base_idx:
                found_base_edge.append(edge)
        if found_base_edge:
            if self._prefer_free_when_available:
                found_free_base_edge = [e for e in found_base_edge if e.joint_type == JointDoFType.FREE]
                if found_free_base_edge:
                    return base_node, found_free_base_edge[0]
            return base_node, found_base_edge[0]

        # If no grounding edge is incident to the base body, synthesize a 6-DoF FREE base edge to the world.
        synthetic_edge = GraphEdge(
            joint_type=int(JointDoFType.FREE),
            joint_index=int(NO_BASE_JOINT_INDEX),
            nodes=(self._world_node, base_idx),
        )
        return base_node, synthetic_edge


###
# Spanning-Tree Selectors
###


class TopologyMinimumDepthSpanningTreeSelector(TopologySpanningTreeSelectorBase):
    """Spanning-tree selector that picks the minimum-depth (and optionally most balanced) candidate.

    Selection rules:

    - For orphan components (single-body candidates with
      ``num_bodies <= 1``), return the first candidate as-is.
    - For island components, pick the candidate that minimizes the
      *eccentricity of the component's assigned base body within the
      tree* (i.e. the longest path from the base body to any other body
      when the tree is treated as undirected). When the tree is rooted
      at the base body this collapses to :attr:`TopologySpanningTree.depth`;
      for trees rooted elsewhere it can be strictly larger. The legacy
      ``min(t.depth)`` ordering is used as a fallback when the source
      component is missing or has no assigned ``base_node``. On an
      eccentricity tie, minimize the imbalance score
      ``sum(len(c) * len(c) for c in tree.children)`` when
      ``prioritize_balanced=True`` (lower is more balanced); otherwise
      keep eccentricity-only ordering. When ``joint_chord_weight`` is set,
      total chord penalty is minimized next. Remaining ties follow stable
      ``min`` semantics.

    Selecting on the eccentricity from the base — rather than from each
    candidate's own root — matters whenever the spanning-tree generator
    has been allowed to enumerate candidates rooted at multiple bodies
    (e.g. ``override_priorities=True`` or a degree-tied root cascade).
    All such candidates share the same minimum ``t.depth`` from their
    own root by construction, so the legacy depth-only ordering would
    silently pick whichever candidate happened to come first in the
    input list — typically a tree where the assigned base sits near a
    leaf instead of near the geometric center.

    Args:
        prioritize_balanced: When ``True`` (default), use imbalance
            score as a secondary ordering key; when ``False``, only
            the eccentricity-from-base is considered.
        joint_chord_weight: Optional map ``joint_index ->`` non-negative
            chord penalty. After eccentricity (and balance when enabled),
            candidates with lower total chord penalty are preferred.
    """

    ###
    # Construction
    ###

    def __init__(
        self,
        *,
        prioritize_balanced: bool = True,
        joint_chord_weight: dict[int, float] | None = None,
        result_index: int = 0,
    ) -> None:
        self._prioritize_balanced: bool = prioritize_balanced
        self._joint_chord_weight: dict[int, float] | None = joint_chord_weight
        self._result_index: int = result_index

    ###
    # Public API
    ###

    @override
    def select_spanning_tree(
        self,
        candidates: list[TopologySpanningTree],
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> TopologySpanningTree:
        """Select the minimum-eccentricity-from-base (and optionally most balanced) candidate.

        Args:
            candidates: Non-empty list of candidate spanning trees.
            bodies: Currently unused; accepted for interface compatibility.
            joints: Currently unused; accepted for interface compatibility.

        Returns:
            The selected :class:`TopologySpanningTree`.

        Raises:
            ValueError:
                If ``candidates`` is empty, or if any island
                candidate has ``children`` set to ``None``
                while balanced ordering is requested.
        """
        # Ensure candidates is non-empty
        if not candidates:
            raise ValueError("`candidates` must contain at least one spanning tree.")

        # Orphans: the generator emits a single trivial candidate; all
        # candidates for the same component share the same `num_bodies`.
        if candidates[0].num_bodies <= 1:
            return candidates[0]

        # Islands: pick the minimum eccentricity from the component's
        # assigned base node, breaking ties by balance if requested. The
        # eccentricity helper falls back to `t.depth` when no base is
        # available, so candidates assembled outside the standard
        # pipeline still get the legacy depth-only ordering.
        if self._result_index > 0:
            if self._prioritize_balanced:
                candidate_ids = [
                    (idx, (self._chord_penalty_sum(t), self._eccentricity_from_base(t), t.balanced_score()))
                    for idx, t in enumerate(candidates)
                ]
            else:
                candidate_ids = [
                    (idx, (self._chord_penalty_sum(t), self._eccentricity_from_base(t)))
                    for idx, t in enumerate(candidates)
                ]
            candidate_ids.sort(key=lambda t: t[1])
            min_weight = candidate_ids[0][1]
            candidate_ids = [entry for entry in candidate_ids if entry[1] == min_weight]
            result_id = candidate_ids[self._result_index % len(candidate_ids)][0]
            return candidates[result_id]

        if self._prioritize_balanced:
            return min(
                candidates,
                key=lambda t: (
                    self._chord_penalty_sum(t),
                    self._eccentricity_from_base(t),
                    t.balanced_score(),
                ),
            )
        return min(candidates, key=lambda t: (self._chord_penalty_sum(t), self._eccentricity_from_base(t)))

    ###
    # Internals
    ###

    def _chord_penalty_sum(self, tree: TopologySpanningTree) -> float:
        if self._joint_chord_weight is None:
            return 0.0
        w = self._joint_chord_weight
        return sum(float(w.get(int(j), 0.0)) for j in tree.chords)

    @staticmethod
    def _eccentricity_from_base(tree: TopologySpanningTree) -> int:
        """Return the eccentricity of the tree's source-component base body.

        Treats the tree as undirected and returns the longest path (in
        arc count) from the component's :attr:`TopologyComponent.base_node`
        to any other body in the tree. When the tree is rooted at the
        base body this is identical to :attr:`TopologySpanningTree.depth`;
        for trees rooted elsewhere — which is the case whenever the
        generator brute-forces over multiple roots — it can be strictly
        larger and is the structurally meaningful "depth from the
        articulation root" we want the selector to minimize.

        Falls back to :attr:`TopologySpanningTree.depth` when the tree
        has no source component, no assigned base body, no arcs/parents
        bookkeeping, or the base body cannot be located in the tree
        (e.g. malformed candidates).
        """
        # Trivial trees and missing-bookkeeping cases all reduce to the
        # stored depth, which is `0` for orphans by construction.
        if tree.num_bodies <= 1 or tree.arcs is None or tree.parents is None:
            return tree.depth
        component = tree.component
        if component is None or component.base_node is None or tree.root is None:
            return tree.depth
        base_idx = int(component.base_node)
        if int(tree.root) == base_idx:
            return tree.depth

        # Reconstruct the local-position → global-body mapping by walking
        # the arcs in regular-numbering order. `arcs[i]` connects local
        # `parents[i]` to local `i`; the source `GraphEdge` carries the
        # canonical global endpoints, and the one that isn't the parent's
        # global index is local `i`'s global index.
        edge_endpoints: dict[int, tuple[int, int]] = {
            e.joint_index: e.nodes for e in (component.edges or []) if e.joint_index >= 0
        }
        nb = tree.num_bodies
        local_to_global: list[int] = [int(tree.root)] + [-1] * (nb - 1)
        for i in range(1, nb):
            joint_idx = tree.arcs[i]
            endpoints = edge_endpoints.get(joint_idx)
            if endpoints is None:
                # Missing source edge → can't reconstruct mapping; fall
                # back to the stored depth rather than risk a misleading
                # eccentricity computed against a partial mapping.
                return tree.depth
            parent_global = local_to_global[tree.parents[i]]
            u, v = endpoints
            local_to_global[i] = v if u == parent_global else u

        try:
            base_local = local_to_global.index(base_idx)
        except ValueError:
            return tree.depth

        # BFS over the undirected parent/child adjacency to find the
        # farthest body from the base.
        adj: dict[int, list[int]] = defaultdict(list)
        for i in range(1, nb):
            p = tree.parents[i]
            adj[p].append(i)
            adj[i].append(p)
        dist: dict[int, int] = {base_local: 0}
        queue: deque[int] = deque([base_local])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return max(dist.values())
