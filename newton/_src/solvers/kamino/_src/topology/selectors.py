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

from ..core.bodies import RigidBodyDescriptor
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    UNASSIGNED_JOINT_TYPE,
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
]


###
# Backends
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

    Args:
        world_node: The world-node sentinel used when synthesizing a
            FREE base edge; must be a negative integer.
        prefer_free_when_available: When ``True`` (default), prefer a
            6-DoF FREE joint over other grounding-edge types incident
            to the chosen body.
        synthetic_base_joint_index: The joint index assigned to a synthesized
            6-DoF FREE base edge; defaults to ``UNASSIGNED_JOINT_TYPE`` to
            flag it for creation by downstream consumers.
    """

    ###
    # Construction
    ###

    def __init__(
        self,
        *,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        synthetic_base_joint_index: int = UNASSIGNED_JOINT_TYPE,
        prefer_free_when_available: bool = True,
    ) -> None:
        if not isinstance(world_node, int):
            raise TypeError(f"`world_node` must be an integer; got {type(world_node).__name__}.")
        if world_node >= 0:
            raise ValueError(
                f"`world_node` must be a negative integer (sentinel for the implicit world); got {world_node}."
            )
        if not isinstance(synthetic_base_joint_index, int):
            raise TypeError(
                f"`synthetic_base_joint_index` must be an integer; got {type(synthetic_base_joint_index).__name__}."
            )
        self._world_node: int = world_node
        self._prefer_free_when_available: bool = prefer_free_when_available
        self._synthetic_base_joint_index: int = synthetic_base_joint_index

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
            joint_index=self._synthetic_base_joint_index,
            nodes=(self._world_node, base_idx),
        )
        return base_node, synthetic_edge


class TopologyMinimumDepthSpanningTreeSelector(TopologySpanningTreeSelectorBase):
    """Spanning-tree selector that picks the minimum-depth (and optionally most balanced) candidate.

    Selection rules:

    - For orphan components (single-body candidates with
      ``num_bodies <= 1``), return the first candidate as-is.
    - For island components, pick the candidate that minimizes
      :attr:`TopologySpanningTree.depth`. On a depth tie, minimize the
      imbalance score
      ``sum(len(c) * len(c) for c in tree.children)`` when
      ``prioritize_balanced=True`` (lower is more balanced); otherwise
      keep depth-only ordering. Remaining ties are broken by the input
      list order (stable ``min`` semantics).

    Args:
        prioritize_balanced: When ``True`` (default), use imbalance
            score as a secondary ordering key; when ``False``, only
            depth is considered.
    """

    ###
    # Construction
    ###

    def __init__(self, *, prioritize_balanced: bool = True) -> None:
        self._prioritize_balanced: bool = prioritize_balanced

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
        """Select the minimum-depth (and optionally most balanced) candidate.

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

        # Islands: pick the minimum depth, breaking ties by balance if requested.
        if self._prioritize_balanced:
            return min(candidates, key=lambda t: (t.depth, t.balanced_score()))
        return min(candidates, key=lambda t: t.depth)
