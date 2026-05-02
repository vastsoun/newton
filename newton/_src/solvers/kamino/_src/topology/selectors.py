# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Component base-selector and spanning-tree selector backends for the topology subsystem.

This module ships concrete first-example implementations of
:class:`TopologyComponentBaseSelectorBase` and
:class:`TopologySpanningTreeSelectorBase` that the
:class:`TopologyGraph` pipeline can use to (a) select the base
node/edge of components that the parser was unable to auto-assign and
(b) select a single spanning tree per component from the candidate list
emitted by a :class:`TopologySpanningTreeGeneratorBase` backend.

The shipped backends are:

- :class:`TopologyHeaviestBodyBaseSelector`: assigns the base node of a
  component to the moving body with the largest mass; picks an existing
  grounding edge incident to that body when available (preferring
  6-DoF FREE), and otherwise synthesizes a new FREE base edge to the
  world.

- :class:`TopologyMinimumDepthSpanningTreeSelector`: selects, per
  component, the candidate spanning tree with the smallest depth and
  (optionally) the most balanced internal branching.
"""

from __future__ import annotations

from ..core.bodies import RigidBodyDescriptor
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    EdgeType,
    NodeType,
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
    A :class:`TopologyComponentBaseSelectorBase` backend that
    assigns a component's base node to its heaviest moving body.

    The heuristic is intentionally simple: among the bodies in
    ``component.nodes``, pick the one with the largest mass
    :attr:`RigidBodyDescriptor.m_i`, breaking ties by the lowest body
    index for determinism. The base edge is then chosen as follows:

    1. If the heaviest body has any incident grounding edge in
       ``component.ground_edges``, pick one — preferring a 6-DoF FREE
       joint when ``prefer_free_when_available=True`` and one is
       incident; otherwise the first listed incident grounding edge.
    2. Otherwise (e.g. an isolated island, or a Stewart-platform-style
       end-effector body without its own grounding), synthesize a
       6-DoF FREE base edge of the form
       ``(JointDoFType.FREE.value, synthetic_base_joint_index, (world_node, base_node))``.
       The default ``synthetic_base_joint_index`` of ``-1`` is a
       sentinel indicating that the joint must be materialized later by
       the model-builder pipeline.

    The returned ``(base_node, base_edge)`` is purely functional: this
    selector does not mutate the input ``component``. The orchestrating
    :meth:`TopologyGraph.select_component_bases` is responsible for
    committing the returned base to the component and for cleaning up
    ``ground_edges`` / ``ground_nodes`` when an existing grounding edge
    is promoted to base.

    Args:
        world_node:
            The world-node sentinel index used when synthesizing a FREE
            base edge for components that have no incident grounding
            edge. Must be a negative integer; defaults to
            :data:`DEFAULT_WORLD_NODE_INDEX`.
        prefer_free_when_available:
            When ``True`` (default), prefer a 6-DoF FREE joint over
            other grounding-edge types incident to the chosen heaviest
            body. When ``False``, use the first listed incident
            grounding edge regardless of joint type.
        synthetic_base_joint_index:
            The joint index assigned to a synthesized FREE base edge
            when the heaviest body has no incident grounding edge.
            Defaults to ``-1`` to flag the edge as "to be materialized
            later" by the model-builder pipeline.
    """

    ###
    # Construction
    ###

    def __init__(
        self,
        *,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        prefer_free_when_available: bool = True,
        synthetic_base_joint_index: int = -1,
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
    ) -> tuple[NodeType, EdgeType]:
        """
        Select the heaviest-body base node and a corresponding base edge.

        Args:
            component:
                The component subgraph to select the base for. Must have a
                non-empty ``nodes`` list.
            bodies:
                Per-world list of :class:`RigidBodyDescriptor`. Required —
                the selector reads :attr:`RigidBodyDescriptor.m_i` from
                ``bodies[node]`` to score candidates.
            joints:
                Unused by this backend; accepted for interface compatibility.

        Returns:
            A tuple ``(base_node, base_edge)`` where ``base_node`` is the
            heaviest body in ``component.nodes`` and ``base_edge`` is
            either an incident grounding edge of that body or a
            synthesized 6-DoF FREE edge connecting it to the world.

        Raises:
            ValueError:
                If ``component`` is ``None``, ``component.nodes`` is empty
                or ``None``, ``bodies`` is ``None``, or any node in
                ``component.nodes`` is out of range for ``bodies``.
        """
        del joints  # unused but accepted for interface compatibility

        if component is None:
            raise ValueError("`component` must not be None.")
        nodes = list(component.nodes) if component.nodes is not None else []
        if not nodes:
            raise ValueError("`component.nodes` must contain at least one body node.")
        if bodies is None:
            raise ValueError("`bodies` must be provided to select the heaviest base body.")

        # Validate that every node maps into `bodies`, fail loudly otherwise so
        # the caller can detect mismatched per-world body lists immediately.
        for n in nodes:
            if n < 0 or n >= len(bodies):
                raise ValueError(f"Body node `{n}` is out of range for `bodies` (size={len(bodies)}).")

        # Score by (mass, -index) so that the tie-breaker prefers the lowest body
        # index. `max` returns the candidate with the largest tuple.
        base_node = max(nodes, key=lambda n: (bodies[n].m_i, -n))

        # Collect grounding edges incident to the heaviest body. Endpoints can
        # appear in either order, so we scan both.
        incident: list[EdgeType] = []
        for e in component.ground_edges or []:
            _t, _j, (u, v) = e
            if u == base_node or v == base_node:
                incident.append(e)

        if incident:
            if self._prefer_free_when_available:
                free_incident = [e for e in incident if e[0] == JointDoFType.FREE.value]
                if free_incident:
                    return base_node, free_incident[0]
            return base_node, incident[0]

        # No incident grounding edge — synthesize a FREE base edge to the world.
        synthetic_edge: EdgeType = (
            JointDoFType.FREE.value,
            self._synthetic_base_joint_index,
            (self._world_node, base_node),
        )
        return base_node, synthetic_edge


class TopologyMinimumDepthSpanningTreeSelector(TopologySpanningTreeSelectorBase):
    """
    A :class:`TopologySpanningTreeSelectorBase` backend that picks the
    minimum-depth (and optionally most balanced) candidate per component.

    Selection rules:

    - For orphan components (single-body candidates with
      ``num_bodies <= 1``), return the first candidate as-is. The
      generator emits a single trivial tree per orphan; there is
      nothing to choose between.
    - For island components (multi-body candidates), pick the candidate
      that minimizes :attr:`TopologySpanningTree.depth`. On a depth tie,
      minimize the imbalance score
      ``sum(len(c) * len(c) for c in tree.children)`` when
      ``prioritize_balanced=True`` (lower is more balanced); otherwise
      keep depth-only ordering. Remaining ties are broken by the
      candidate's position in the input list (stable ``min`` semantics).

    Args:
        prioritize_balanced:
            When ``True`` (default), use imbalance score as a secondary
            ordering key for island components after depth. When
            ``False``, only depth is considered and ties go to the
            first candidate.
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
        """
        Select the minimum-depth (and optionally most balanced) candidate.

        Args:
            candidates:
                Non-empty list of candidate spanning trees produced by a
                :class:`TopologySpanningTreeGeneratorBase` backend for
                a single component.
            bodies:
                Unused by this backend; accepted for interface compatibility.
            joints:
                Unused by this backend; accepted for interface compatibility.

        Returns:
            The selected :class:`TopologySpanningTree`.

        Raises:
            ValueError:
                If ``candidates`` is empty, or if any candidate has
                ``children`` set to ``None`` while balanced ordering is
                requested for an island.
        """
        del bodies, joints  # unused but accepted for interface compatibility

        if not candidates:
            raise ValueError("`candidates` must contain at least one spanning tree.")

        # Orphans: the generator emits a single trivial candidate. All
        # candidates for the same component share the same `num_bodies`,
        # so inspecting the first is sufficient for the orphan check.
        if candidates[0].num_bodies <= 1:
            return candidates[0]

        # Islands: stable `min` over (depth[, balance])
        # keeps the first candidate on full ties.
        if self._prioritize_balanced:
            return min(candidates, key=lambda t: (t.depth, _balanced_score(t)))
        return min(candidates, key=lambda t: t.depth)


###
# Helpers
###


def _balanced_score(tree: TopologySpanningTree) -> int:
    """
    Return the sum of squared per-parent child counts.

    Smaller scores indicate more balanced trees: at fixed depth the
    minimum is achieved when every internal node owns roughly the same
    number of children, while a single-spine tree (one parent owns
    ``N_B - 1`` children) reaches the maximum.

    Mirrors :meth:`TopologyMinimumDepthSpanningTreeGenerator._balanced_score`
    so the tree selector can score candidates without coupling to the
    generator module. A future refactor may promote this helper to a
    shared utility in :mod:`.types`.

    Raises:
        ValueError: If ``tree.children`` is ``None``.
    """
    if tree.children is None:
        raise ValueError("Cannot score a TopologySpanningTree with `children=None`; the tree is malformed.")
    return sum(len(cs) * len(cs) for cs in tree.children)
