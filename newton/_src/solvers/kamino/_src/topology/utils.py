# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""TopologyGraph interop helpers for ``newton.ModelBuilder`` and OpenUSD assets.

The public helpers in this module are:

- :func:`extract_graph_inputs_from_builder`, :func:`bodies_from_builder`, and
  :func:`joints_from_builder`: cheap extractors that derive the inputs of a
  :class:`~kamino.topology.TopologyGraph` from a Newton :class:`ModelBuilder`.

- :func:`discover_topology_for_builder`: convenience wrapper that returns an
  autoparsed :class:`~kamino.topology.TopologyGraph` for a single-world builder.

- :func:`apply_discovered_topology_to_builder`: returns a *new* Newton
  :class:`ModelBuilder` with the discovered topology baked-in (synthesized
  base joints, regular-numbering body/joint reordering, articulation
  bookkeeping populated).

- :func:`export_usd_with_discovered_topology`: writes a copy of a source USD
  asset with ``PhysicsArticulationRootAPI`` and ``physics:excludeFromArticulation``
  authored according to the discovered topology.

----
TopologyGraph use-cases:

    Notes:
        - Point (1) is the common workflow starting point.
        - Points (2) and (3) define branching workflows for different downstream use cases.

    1. A USD model comes as an arbitrary "soup" of bodies and joints with no explicit
        meta-data such as articulation info ("PhysicsArticulationRootAPI" on body prims,
        or "excludeFromArticulation" on joint prims), or even explicit FREE joints (i.e.
        "PhysicsJoint" w/ or w/o "PhysicsLimitAPI:<DOF>"). In these cases we want to
        discover the underlying topology of the model at run-time. To do this we can:
        1a. Use ``newton.ModelBuilder.add_usd(source=asset_file, joint_ordering=None, ...)``
            to first load the USD asset into a builder without invoking topology checks.
        1b. Use :func:`extract_graph_inputs_from_builder`, :func:`bodies_from_builder`,
            and :func:`joints_from_builder` to derive the inputs for a
            :class:`~kamino.topology.TopologyGraph`. The graph parses the body-joint
            connectivity and discovers the underlying topology of the asset, decomposing
            the graph into a set of spanning trees ordered by size, a list of body/joint
            index remappings (to satisfy Featherstone's regular numbering rules), and a
            list of synthesized FREE base joints needed to connect any isolated component
            in the graph.

    2. Re-export USD asset with discovered topology baked-in:
        2a. :func:`apply_discovered_topology_to_builder` deep-copies the source
            ``newton.ModelBuilder``, appends the synthesized FREE base joints, permutes
            every body/joint-indexed array according to the topology remap, and rebuilds
            the articulation bookkeeping (``articulation_*``) so each spanning tree
            surfaces as one ``newton.ModelBuilder`` articulation.
        2b. :func:`export_usd_with_discovered_topology` writes a copy of the source USD
            with the discovered topology baked-in:
                - Root bodies have the ``PhysicsArticulationRootAPI`` applied schema
                added to their prim defs.
                - Chord joints have the ``uniform bool physics:excludeFromArticulation = 1``
                attribute added.
                - Synthesized base joints are NOT written to the USD asset in this PR
                (the in-builder representation in 2a already covers that case for
                downstream solvers).

    3. Use front-end Newton API: take the builder returned by 2a and feed it into
        ``newton.Model`` as usual.

    4. Use Kamino ``USDImporter`` to load the modified USD asset (from 2b) into a
        ``ModelBuilderKamino``. Returning here is gated by the still-unimplemented
        ``ModelBuilderKamino.add_topology_descriptor``, ``TopologyModel.from_descriptors``,
        and ``TopologyModel.from_newton`` factories — once those land, the dropped
        Kamino-flavored helpers (``make_topology_model_for_existing_*_kamino``) can
        return.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import newton

from .....sim import ModelBuilder
from ..._src import topology
from ..._src.core.bodies import RigidBodyDescriptor
from ..._src.core.joints import JointDescriptor, JointDoFType
from ..._src.utils import logger as msg

###
# Module interface
###

__all__ = [
    "NEWTON_TO_KAMINO_JOINT_TYPE",
    "apply_discovered_topology_to_builder",
    "bodies_from_builder",
    "discover_topology_for_builder",
    "export_usd_with_discovered_topology",
    "extract_graph_inputs_from_builder",
    "joints_from_builder",
]

###
# Constants
###

NEWTON_TO_KAMINO_JOINT_TYPE: dict[int, int] = {
    int(newton.JointType.PRISMATIC): int(JointDoFType.PRISMATIC),
    int(newton.JointType.REVOLUTE): int(JointDoFType.REVOLUTE),
    int(newton.JointType.BALL): int(JointDoFType.SPHERICAL),
    int(newton.JointType.FIXED): int(JointDoFType.FIXED),
    int(newton.JointType.FREE): int(JointDoFType.FREE),
    int(newton.JointType.D6): int(JointDoFType.REVOLUTE),
    int(newton.JointType.DISTANCE): int(JointDoFType.REVOLUTE),
}
"""Dispatch table from Newton's :class:`newton.JointType` integer values to
Kamino's :class:`JointDoFType` integer values.

Only the FREE → FREE mapping matters for FREE-vs-non-FREE classification by
:class:`~kamino.topology.TopologyComponentParser`'s grounding-edge auto-promotion;
every other Newton joint type is collapsed onto :attr:`JointDoFType.REVOLUTE`
since the topology pipeline does not differentiate non-FREE types.
"""


###
# Extraction helpers (TopologyGraph inputs from a Newton ModelBuilder)
###


def extract_graph_inputs_from_builder(
    builder: ModelBuilder,
) -> tuple[list[topology.NodeType], list[topology.EdgeType]]:
    """Extract ``(nodes, edges)`` for a :class:`TopologyGraph` from a builder.

    All bodies and joints in the builder are flattened into a single graph;
    :class:`~kamino.topology.TopologyComponentParser` recovers connected
    components automatically.

    Args:
        builder: The source Newton :class:`ModelBuilder`.

    Returns:
        A pair ``(nodes, edges)`` where ``nodes`` is the list of body indices
        ``range(builder.body_count)`` and ``edges`` is the list of
        ``(joint_type, joint_index, (parent, child))`` tuples ready to feed
        into :class:`~kamino.topology.TopologyGraph`. ``joint_type`` is
        translated to :class:`JointDoFType` integer values via
        :data:`NEWTON_TO_KAMINO_JOINT_TYPE`.
    """
    nodes: list[topology.NodeType] = list(range(builder.body_count))
    edges: list[topology.EdgeType] = []
    for j in range(builder.joint_count):
        joint_type = NEWTON_TO_KAMINO_JOINT_TYPE.get(int(builder.joint_type[j]), int(JointDoFType.REVOLUTE))
        edges.append((joint_type, j, (int(builder.joint_parent[j]), int(builder.joint_child[j]))))
    return nodes, edges


def bodies_from_builder(builder: ModelBuilder) -> list[RigidBodyDescriptor]:
    """Synthesize :class:`RigidBodyDescriptor` instances from a Newton builder.

    Only the fields read by the shipped pipeline backends are populated:
    :attr:`RigidBodyDescriptor.name` from :attr:`ModelBuilder.body_label` (so
    the topology renderer can surface meaningful per-body labels) and
    :attr:`RigidBodyDescriptor.m_i` from :attr:`ModelBuilder.body_mass` (so
    :class:`~kamino.topology.TopologyHeaviestBodyBaseSelector` can score
    candidate bases). The remaining fields default per their dataclass.

    Args:
        builder: The source Newton :class:`ModelBuilder`.

    Returns:
        One :class:`RigidBodyDescriptor` per body in ``builder``, in body-index order.
    """
    return [
        RigidBodyDescriptor(
            name=builder.body_label[i] if i < len(builder.body_label) else f"body_{i}",
            m_i=float(builder.body_mass[i]),
            bid=i,
        )
        for i in range(builder.body_count)
    ]


def joints_from_builder(builder: ModelBuilder) -> list[JointDescriptor]:
    """Synthesize :class:`JointDescriptor` instances from a Newton builder.

    Populates :attr:`JointDescriptor.name` from :attr:`ModelBuilder.joint_label`
    and :attr:`JointDescriptor.dof_type` from the Newton joint-type integer via
    :data:`NEWTON_TO_KAMINO_JOINT_TYPE`. Other fields default per the dataclass
    — this is enough for consumers that only inspect ``name`` (e.g. the
    topology renderer).

    Args:
        builder: The source Newton :class:`ModelBuilder`.

    Returns:
        One :class:`JointDescriptor` per joint in ``builder``, in joint-index order.
    """
    return [
        JointDescriptor(
            name=builder.joint_label[j] if j < len(builder.joint_label) else f"joint_{j}",
            dof_type=JointDoFType(
                NEWTON_TO_KAMINO_JOINT_TYPE.get(int(builder.joint_type[j]), int(JointDoFType.REVOLUTE))
            ),
        )
        for j in range(builder.joint_count)
    ]


def discover_topology_for_builder(
    builder: ModelBuilder,
    *,
    base_selector: topology.TopologyComponentBaseSelectorBase | None = None,
    tree_traversal_mode: topology.SpanningTreeTraversal = "dfs",
    max_tree_candidates: int = 32,
    reassign_indices_inplace: bool = True,
) -> topology.TopologyGraph:
    """Build and autoparse a :class:`TopologyGraph` for a single-world builder.

    Convenience wrapper around the extraction helpers + the
    :class:`~kamino.topology.TopologyGraph` orchestrator. Validates the
    single-world precondition required by :func:`apply_discovered_topology_to_builder`.

    Args:
        builder: The source Newton :class:`ModelBuilder` (must contain a
            single world's worth of bodies and joints, i.e.
            ``builder.world_count == 1``).
        base_selector: Optional override for the base-selector backend; defaults
            to :class:`~kamino.topology.TopologyHeaviestBodyBaseSelector`.
        tree_traversal_mode: Spanning-tree traversal mode forwarded to the
            graph (``"dfs"`` or ``"bfs"``).
        max_tree_candidates: Per-component cap on candidate spanning trees.
        reassign_indices_inplace: When ``True`` (default), the spanning trees
            cached on the graph are remapped through the regular-numbering
            permutation in place; pass ``False`` to keep the original indices
            on :attr:`TopologyGraph.trees` and access remapped copies via
            :attr:`TopologyGraph.trees_remapped`.

    Returns:
        An autoparsed :class:`~kamino.topology.TopologyGraph` instance.

    Raises:
        ValueError: If ``builder.world_count != 1``.
    """
    if builder.world_count != 1:
        raise ValueError(
            f"`discover_topology_for_builder` requires a single-world `ModelBuilder`; "
            f"got `world_count={builder.world_count}`."
        )

    nodes, edges = extract_graph_inputs_from_builder(builder)
    bodies = bodies_from_builder(builder)
    joints = joints_from_builder(builder)

    if base_selector is None:
        base_selector = topology.TopologyHeaviestBodyBaseSelector()

    graph = topology.TopologyGraph(
        nodes=nodes,
        edges=edges,
        base_selector=base_selector,
        bodies=bodies,
        joints=joints,
        tree_traversal_mode=tree_traversal_mode,
        max_tree_candidates=max_tree_candidates,
        reassign_indices_inplace=reassign_indices_inplace,
        autoparse=True,
    )
    msg.debug("Discovered topology graph with %d component(s)", len(graph.components))
    return graph


###
# Builder-side topology baking (use cases 2a and 3)
###


def apply_discovered_topology_to_builder(
    builder: ModelBuilder,
    graph: topology.TopologyGraph | None = None,
    *,
    label: str | None = None,
) -> ModelBuilder:
    """Return a new builder with the discovered topology baked into its layout.

    Strategy: deep-copy ``builder``, append the synthesized FREE base joints
    from :attr:`TopologyGraph.new_base_edges`, then permute every body- and
    joint-indexed array of the deep-copied builder according to
    :attr:`TopologyGraph.body_node_remap` / :attr:`TopologyGraph.joint_edge_remap`.
    For every joint flagged on :attr:`TopologyGraph.reversed_joint_edges`, the
    rebuilt parent/child indices and the parent/child anchor transforms
    (``joint_X_p`` / ``joint_X_c``) are also swapped so the surface joint row
    matches the BFS-driven parent → child direction baked into the selected
    spanning tree (otherwise Featherstone's ``parents[i] < i`` invariant
    would fail at the first arc whose source edge was already polarized
    ``child → parent``). Finally, reset and rebuild the articulation
    bookkeeping (:attr:`ModelBuilder.articulation_start`,
    :attr:`ModelBuilder.articulation_label`, :attr:`ModelBuilder.articulation_world`,
    :attr:`ModelBuilder.joint_articulation`) so each spanning tree surfaces
    as one Newton articulation.

    The contiguous-arc invariant required by
    :meth:`ModelBuilder.add_articulation` is guaranteed by
    :class:`~kamino.topology.TopologyIndexReassignment`, which lays out joints
    as ``[tree_0_arcs, tree_0_chords, tree_1_arcs, tree_1_chords, ...]``. Chord
    joints are intentionally left with ``joint_articulation == -1`` so loop
    closures don't trip the multi-parent validation.

    Note:
        The reversed-joint swap reorients ``joint_parent`` / ``joint_child``
        and ``joint_X_p`` / ``joint_X_c`` only. Per-axis directions
        (``joint_axis``) are left unchanged; for USD-derived assets the
        parent/child anchor frames coincide at rest so the existing axis
        vectors remain valid in the new parent anchor frame. Callers with
        non-coincident anchor frames must rotate the axes themselves.

    Args:
        builder: The source Newton :class:`ModelBuilder` (single world).
        graph: Optional pre-built :class:`~kamino.topology.TopologyGraph`. When
            ``None``, this helper invokes :func:`discover_topology_for_builder`
            with default kwargs.
        label: Optional prefix for the rebuilt articulation labels. When
            provided, articulation ``i`` is labeled ``"<label>_articulation_<i>"``;
            otherwise it falls back to the default ``"articulation_<i>"`` used
            by :meth:`ModelBuilder.add_articulation`.

    Returns:
        A new :class:`ModelBuilder` instance with the topology baked-in. The
        original ``builder`` is not modified.

    Raises:
        ValueError: If ``builder.world_count != 1``, the graph's remap arrays
            are inconsistent with the post-synthesis builder size, or the
            remapped tree arcs are not contiguous in the new joint indexing
            (this would indicate a bug in the index reassignment backend).
    """
    if builder.world_count != 1:
        raise ValueError(
            f"`apply_discovered_topology_to_builder` requires a single-world `ModelBuilder`; "
            f"got `world_count={builder.world_count}`."
        )

    if graph is None:
        graph = discover_topology_for_builder(builder)

    new = copy.deepcopy(builder)

    # All bodies and joints synthesized below must be tagged with the single
    # world's index. ``add_joint_free`` reads ``current_world`` for that, so
    # force it to ``0`` regardless of whether the source builder is currently
    # inside a ``begin_world()`` context. We restore to ``-1`` (clean global
    # scope) on the way out so callers can continue editing the result with
    # their own ``begin_world()`` calls.
    new._current_world = 0

    # 1. Append synthetic FREE base joints. Their provisional ``joint_index``
    # of the form ``original_num_edges + k`` is chosen by
    # ``TopologyGraph._commit_base_edge`` to land exactly at the next slot
    # appended via ``add_joint_free``.
    if graph.new_base_edges:
        for k, edge in enumerate(graph.new_base_edges):
            world_node = graph.world_node
            try:
                child = next(n for n in edge.nodes if n != world_node)
            except StopIteration as exc:
                raise ValueError(
                    f"Synthetic base edge {edge!r} has no non-world endpoint; cannot determine child body."
                ) from exc
            joint_idx = new.add_joint_free(
                child=child,
                label=f"{label}_synthetic_base_{k}" if label else f"synthetic_base_{k}",
            )
            if joint_idx != edge.joint_index:
                raise ValueError(
                    f"Synthetic base joint index mismatch: graph promised "
                    f"`joint_index={edge.joint_index}` but `add_joint_free` returned "
                    f"`{joint_idx}`."
                )

    # 2. Recover the body/joint permutations and validate sizes.
    body_remap: list[int] = list(graph.body_node_remap) if graph.body_node_remap is not None else []
    joint_remap: list[int] = list(graph.joint_edge_remap) if graph.joint_edge_remap is not None else []

    if not body_remap:
        body_remap = list(range(new.body_count))
    if not joint_remap:
        joint_remap = list(range(new.joint_count))

    if len(body_remap) != new.body_count:
        raise ValueError(
            f"Body remap length ({len(body_remap)}) does not match post-synthesis body count ({new.body_count})."
        )
    if len(joint_remap) != new.joint_count:
        raise ValueError(
            f"Joint remap length ({len(joint_remap)}) does not match post-synthesis joint count ({new.joint_count})."
        )

    # Inverse permutations: ``new_to_old[new_idx] = old_idx``. We use these to
    # rebuild every per-body / per-joint array by pulling old slots in their
    # new order — this is more robust than trying to mutate the same list
    # in place (which can shadow earlier writes when the permutation has cycles).
    new_to_old_body: list[int] = [-1] * len(body_remap)
    for old_idx, new_idx in enumerate(body_remap):
        new_to_old_body[new_idx] = old_idx
    new_to_old_joint: list[int] = [-1] * len(joint_remap)
    for old_idx, new_idx in enumerate(joint_remap):
        new_to_old_joint[new_idx] = old_idx

    # 3. Snapshot per-joint slice metadata BEFORE permuting, since the
    # per-axis / per-coord / per-dof / per-cts permutation needs the old
    # joint_q_start, joint_qd_start, joint_cts_start, and joint_dof_dim
    # arrays to know how to slice.
    nj = new.joint_count
    old_q_starts = list(new.joint_q_start)
    old_qd_starts = list(new.joint_qd_start)
    old_cts_starts = list(new.joint_cts_start)
    old_q_total = new.joint_coord_count
    old_qd_total = new.joint_dof_count
    old_cts_total = new.joint_constraint_count
    old_dof_dims = list(new.joint_dof_dim)

    old_q_counts = _per_joint_counts(old_q_starts, old_q_total)
    old_qd_counts = _per_joint_counts(old_qd_starts, old_qd_total)
    old_cts_counts = _per_joint_counts(old_cts_starts, old_cts_total)
    old_axis_counts = [sum(d) for d in old_dof_dims]
    old_axis_starts: list[int] = [0] * nj
    for j in range(1, nj):
        old_axis_starts[j] = old_axis_starts[j - 1] + old_axis_counts[j - 1]

    # 4. Permute all per-body lists in lockstep.
    _PER_BODY_ATTRS = (
        "body_mass",
        "body_inertia",
        "body_inv_mass",
        "body_inv_inertia",
        "body_com",
        "body_q",
        "body_qd",
        "body_label",
        "body_lock_inertia",
        "body_flags",
        "body_world",
        "body_color_groups",
    )
    for attr in _PER_BODY_ATTRS:
        old_list = getattr(new, attr)
        if not old_list:
            continue
        if len(old_list) != new.body_count:
            # Safety net for builders whose internal state is partially
            # populated; skip rather than scrambling unrelated data.
            continue
        setattr(new, attr, [old_list[new_to_old_body[i]] for i in range(new.body_count)])

    # 5. Remap shape→body and rebuild ``body_shapes`` keys.
    for s in range(new.shape_count):
        old_b = new.shape_body[s]
        if old_b >= 0:
            new.shape_body[s] = body_remap[old_b]
    new.body_shapes = {(-1 if k == -1 else body_remap[k]): list(v) for k, v in new.body_shapes.items()}

    # 6. Permute per-joint scalar/tuple lists.
    _PER_JOINT_ATTRS = (
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_X_p",
        "joint_X_c",
        "joint_label",
        "joint_enabled",
        "joint_world",
        "joint_dof_dim",
        "joint_twist_lower",
        "joint_twist_upper",
    )
    for attr in _PER_JOINT_ATTRS:
        old_list = getattr(new, attr)
        if not old_list:
            continue
        if len(old_list) != nj:
            continue
        setattr(new, attr, [old_list[new_to_old_joint[i]] for i in range(nj)])

    # 7. Remap body indices in joint_parent/joint_child (preserving the -1
    # sentinel for world-attached joints).
    for j in range(nj):
        if new.joint_parent[j] >= 0:
            new.joint_parent[j] = body_remap[new.joint_parent[j]]
        if new.joint_child[j] >= 0:
            new.joint_child[j] = body_remap[new.joint_child[j]]

    # 7b. Swap parent ↔ child (and the matching anchor transforms) for joints
    # whose source-edge polarity opposed the BFS-driven parent → child
    # direction baked into the spanning tree (see :attr:`TopologyGraph
    # .reversed_joint_edges`). Without this swap the rebuilt builder would
    # carry joint rows that violate Featherstone's regular-numbering
    # convention even though the discovered topology is correct.
    #
    # Limitation: per-axis directions (``joint_axis``) are NOT renegotiated
    # across the swap — they still live in what is now the new child anchor
    # frame. For USD-derived assets the parent/child anchor frames coincide
    # at rest, so the same vector value remains valid; for assets with
    # non-coincident anchor frames callers must rotate the axis themselves
    # using ``joint_X_p``/``joint_X_c``. Per-DOF state arrays (``joint_q``,
    # ``joint_qd``, ``joint_act``, etc.) similarly retain their stored
    # values and may need a sign flip if the caller relies on a signed
    # convention for q.
    for old_j in graph.reversed_joint_edges:
        if old_j < 0 or old_j >= len(joint_remap):
            continue
        new_j = joint_remap[old_j]
        if new_j < 0 or new_j >= nj:
            continue
        new.joint_parent[new_j], new.joint_child[new_j] = (
            new.joint_child[new_j],
            new.joint_parent[new_j],
        )
        new.joint_X_p[new_j], new.joint_X_c[new_j] = (
            new.joint_X_c[new_j],
            new.joint_X_p[new_j],
        )

    # 7c. Rebuild ``joint_parents`` / ``joint_children`` dicts from the
    # post-swap parent/child arrays so they stay consistent.
    new.joint_parents = {}
    new.joint_children = {}
    for j in range(nj):
        p = new.joint_parent[j]
        c = new.joint_child[j]
        new.joint_parents.setdefault(c, []).append(p)
        new.joint_children.setdefault(p, []).append(c)

    # 8. Permute per-axis / per-coord / per-dof / per-cts arrays by walking
    # the OLD joint slice layout in NEW joint order.
    _PER_AXIS_ATTRS = (
        "joint_axis",
        "joint_target_pos",
        "joint_target_vel",
        "joint_target_mode",
        "joint_target_ke",
        "joint_target_kd",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_armature",
        "joint_effort_limit",
        "joint_velocity_limit",
        "joint_friction",
        "joint_limit_lower",
        "joint_limit_upper",
    )
    _permute_per_slot_arrays(new, _PER_AXIS_ATTRS, new_to_old_joint, old_axis_starts, old_axis_counts)
    _permute_per_slot_arrays(new, ("joint_q",), new_to_old_joint, old_q_starts, old_q_counts)
    _permute_per_slot_arrays(new, ("joint_qd", "joint_f", "joint_act"), new_to_old_joint, old_qd_starts, old_qd_counts)
    _permute_per_slot_arrays(new, ("joint_cts",), new_to_old_joint, old_cts_starts, old_cts_counts)

    # 9. Refresh the cumulative ``joint_*_start`` arrays in the new order.
    # Counts (totals) stay the same — we only re-laid the same chunks in a
    # new order, so the running prefix sum changes per joint but not the total.
    new_q_starts: list[int] = [0] * nj
    new_qd_starts: list[int] = [0] * nj
    new_cts_starts: list[int] = [0] * nj
    for new_j in range(1, nj):
        prev_old_j = new_to_old_joint[new_j - 1]
        new_q_starts[new_j] = new_q_starts[new_j - 1] + old_q_counts[prev_old_j]
        new_qd_starts[new_j] = new_qd_starts[new_j - 1] + old_qd_counts[prev_old_j]
        new_cts_starts[new_j] = new_cts_starts[new_j - 1] + old_cts_counts[prev_old_j]
    new.joint_q_start = new_q_starts
    new.joint_qd_start = new_qd_starts
    new.joint_cts_start = new_cts_starts

    # 10. Build per-DOF / per-coord / per-cts global remaps so we can update
    # cross-references that hold flat indices (actuator entries currently;
    # equality / mimic constraints reference whole joints, not DOFs).
    dof_remap = _build_per_slot_index_remap(joint_remap, old_qd_starts, old_qd_counts, new_qd_starts)
    coord_remap = _build_per_slot_index_remap(joint_remap, old_q_starts, old_q_counts, new_q_starts)

    # 11. Remap cross-reference lists — body indices and joint indices
    # stored on equality/mimic constraints, muscle bodies, and actuator
    # entries (which carry per-DOF/per-coord indices).
    _remap_inplace(new.equality_constraint_body1, body_remap)
    _remap_inplace(new.equality_constraint_body2, body_remap)
    _remap_inplace(new.equality_constraint_joint1, joint_remap)
    _remap_inplace(new.equality_constraint_joint2, joint_remap)
    _remap_inplace(new.constraint_mimic_joint0, joint_remap)
    _remap_inplace(new.constraint_mimic_joint1, joint_remap)
    _remap_inplace(new.muscle_bodies, body_remap)

    for entry in new.actuator_entries.values():
        entry.indices = [dof_remap[i] for i in entry.indices]
        entry.pos_indices = [coord_remap[i] for i in entry.pos_indices]

    # 12. Reset and rebuild articulation bookkeeping from the discovered
    # spanning trees. Walk the trees in ``graph.trees_remapped`` order so the
    # articulation layout matches the regular-numbering joint layout produced
    # by :class:`TopologyIndexReassignment`.
    new.articulation_start = []
    new.articulation_label = []
    new.articulation_world = []
    new.joint_articulation = [-1] * nj

    trees_remapped = graph.trees_remapped if graph.trees_remapped is not None else graph.trees
    for tree_idx, tree in enumerate(trees_remapped):
        if tree.arcs is None:
            continue
        # Filter out the ``NO_BASE_JOINT_INDEX`` sentinel some trees carry at
        # ``arcs[0]``; only real (or already-remapped synthetic) joint indices
        # should be added to the articulation.
        arcs = [a for a in tree.arcs if a >= 0]
        if not arcs:
            continue
        sorted_arcs = sorted(arcs)
        for i in range(1, len(sorted_arcs)):
            if sorted_arcs[i] != sorted_arcs[i - 1] + 1:
                raise ValueError(
                    f"Spanning tree {tree_idx}'s arc joint indices are not contiguous after "
                    f"index reassignment: {sorted_arcs}. This indicates a bug in the index "
                    f"reassignment backend (chords should be placed AFTER all arcs of a tree)."
                )
        articulation_idx = len(new.articulation_start)
        new.articulation_start.append(sorted_arcs[0])
        tree_label = f"{label}_articulation_{articulation_idx}" if label else f"articulation_{articulation_idx}"
        new.articulation_label.append(tree_label)
        new.articulation_world.append(0)
        for j_idx in sorted_arcs:
            new.joint_articulation[j_idx] = articulation_idx

    # 13. Restore a clean global scope so callers can chain further
    # ``begin_world()`` calls if they need to extend the result.
    new._current_world = -1

    return new


###
# USD-side topology baking (use case 2b)
###


def export_usd_with_discovered_topology(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    add_usd_kwargs: dict[str, Any] | None = None,
    plotfig: bool = False,
    savefig: bool = False,
    figpath: str | os.PathLike[str] | None = None,
) -> Path:
    """Export a copy of a USD asset with the discovered topology baked-in.

    Loads the USD asset into a Newton :class:`ModelBuilder`, runs topology
    discovery, then reopens a copy of the source USD and authors:

    - :class:`UsdPhysics.ArticulationRootAPI` on each spanning tree's root
      body prim (located via :attr:`ModelBuilder.body_label`, which mirrors
      the prim path for USD-imported assets).
    - ``uniform bool physics:excludeFromArticulation = 1`` on each chord joint
      prim so downstream USD consumers (incl. :class:`USDImporter`) treat
      these joints as loop closures rather than tree arcs.
    - For every joint listed on :attr:`TopologyGraph.reversed_joint_edges`,
      the ``physics:body0`` / ``physics:body1`` relationship targets are
      swapped along with the matching ``physics:localPos0`` /
      ``physics:localPos1`` and ``physics:localRot0`` / ``physics:localRot1``
      anchor attributes, so the authored joint row respects the BFS-driven
      ``parent → child`` direction baked into the spanning tree.

    Synthesized FREE base joints are NOT authored into the USD asset by this
    helper; that responsibility lives with
    :func:`apply_discovered_topology_to_builder` (the in-builder
    representation that downstream solvers consume).

    Note:
        The reversed-joint swap reorients the ``body0`` / ``body1``
        relationships and their matching local-anchor attributes
        (``localPos0`` / ``localPos1``, ``localRot0`` / ``localRot1``).
        Joint-local axis tokens (``physics:axis`` on Revolute /
        Prismatic, the per-DOF schemas on D6, etc.) are left untouched:
        for anchor frames that coincide at rest the same axis token
        remains valid in the new ``body0`` anchor frame, but assets
        with non-coincident anchor frames may need additional axis
        rotation which callers must perform themselves.

    Args:
        source: Path to the source USD asset (``.usd`` / ``.usda`` / ``.usdc``).
        output: Path to the destination USD asset. Existing files are
            overwritten.
        add_usd_kwargs: Optional dictionary of keyword arguments forwarded to
            :meth:`ModelBuilder.add_usd`. Use this to override defaults like
            ``joint_ordering`` (defaults to the importer's ``"dfs"`` mode),
            ``collapse_fixed_joints``, etc., for assets that need special
            handling.
        plotfig: When ``True``, display the rendered topology figures
            (forwarded to :meth:`TopologyGraph.render_graph` and friends).
        savefig: When ``True``, persist the rendered topology figures to
            ``figpath``. No effect when ``figpath`` is ``None``.
        figpath: Directory under which the rendered topology figures are
            written. Created on demand. Used only when ``plotfig`` or
            ``savefig`` is ``True``.

    Returns:
        The :class:`pathlib.Path` of the written output USD.

    Raises:
        ImportError: If the ``pxr`` (``usd-core``) Python bindings are not
            installed.
        ValueError: If the loaded builder does not have exactly one world or
            if any body/joint label does not resolve to a prim in the cloned
            stage.
    """
    Sdf, Usd, UsdPhysics = _load_pxr_modules()

    source_path = Path(source)
    output_path = Path(output)

    # 1. Load the source USD into a Newton builder. Default ``add_usd``
    # behaviour (``joint_ordering="dfs"``) works for tree-like assets; loop-
    # closing assets can pass ``joint_ordering=None`` via ``add_usd_kwargs``.
    add_usd_kwargs = dict(add_usd_kwargs) if add_usd_kwargs is not None else {}
    add_usd_kwargs.setdefault("source", str(source_path))
    tmp_builder = ModelBuilder()
    tmp_builder.begin_world()
    usd_result: dict[str, Any] = tmp_builder.add_usd(**add_usd_kwargs)
    tmp_builder.end_world()

    # 2. Discover the topology. We deliberately keep the original body/joint
    # indices on ``graph.trees`` (``reassign_indices_inplace=False``) so that
    # the per-tree ``root`` and ``chords`` indices can be resolved directly
    # through ``path_body_map`` / ``path_joint_map`` (which use the import-
    # order indices from :func:`parse_usd`).
    graph = discover_topology_for_builder(tmp_builder, reassign_indices_inplace=False)

    # 3. Optional rendering of the discovered topology graph and its trees.
    if plotfig or savefig:
        figdir = Path(figpath) if figpath is not None else None
        if figdir is not None:
            figdir.mkdir(parents=True, exist_ok=True)
        graph.render_graph(
            path=str(figdir / "topology_graph.pdf") if figdir is not None else None,
            show=plotfig,
        )
        graph.render_spanning_tree_candidates(
            path=str(figdir / "spanning_tree_candidates.pdf") if figdir is not None else None,
            show=plotfig,
        )
        graph.render_spanning_trees(
            path=str(figdir / "spanning_trees.pdf") if figdir is not None else None,
            show=plotfig,
        )

    # 4. Resolve body / joint indices to USD prim paths. ``parse_usd`` returns
    # the path→index maps; we invert them and additionally cross-check with
    # ``builder.body_label`` / ``builder.joint_label``, which mirror the prim
    # paths for USD-imported assets.
    path_to_body: dict[str, int] = usd_result.get("path_body_map", {}) or {}
    path_to_joint: dict[str, int] = usd_result.get("path_joint_map", {}) or {}
    body_to_path: dict[int, str] = {bid: p for p, bid in path_to_body.items()}
    joint_to_path: dict[int, str] = {jid: p for p, jid in path_to_joint.items()}
    # Fall back to the labels for any indices missing from the maps. ``parse_usd``
    # does not always populate the maps for joints introduced via ``base_joint``
    # arguments etc., but those still get sensible labels from the importer.
    for i in range(tmp_builder.body_count):
        body_to_path.setdefault(i, tmp_builder.body_label[i])
    for j in range(tmp_builder.joint_count):
        joint_to_path.setdefault(j, tmp_builder.joint_label[j])

    # 5. Open the source stage, export an unedited copy to ``output``, then
    # re-open ``output`` for editing so the original asset is untouched.
    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise ValueError(f"Failed to open USD source `{source_path}`.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_stage.GetRootLayer().Export(str(output_path))

    output_stage = Usd.Stage.Open(str(output_path))
    if output_stage is None:
        raise ValueError(f"Failed to open exported USD copy `{output_path}` for editing.")

    # 6. For each spanning tree, mark the root body prim with
    # ``UsdPhysics.ArticulationRootAPI`` and stamp every chord joint with
    # ``physics:excludeFromArticulation = 1``.
    for tree_idx, tree in enumerate(graph.trees):
        if tree.root is None or int(tree.root) < 0:
            continue
        root_path = body_to_path.get(int(tree.root))
        if root_path is None:
            msg.warning("Skipping articulation root for tree %d: body index %s has no USD prim.", tree_idx, tree.root)
        else:
            root_prim = output_stage.GetPrimAtPath(Sdf.Path(root_path))
            if not root_prim or not root_prim.IsValid():
                msg.warning(
                    "Skipping articulation root for tree %d: prim at `%s` is not valid in the output stage.",
                    tree_idx,
                    root_path,
                )
            else:
                UsdPhysics.ArticulationRootAPI.Apply(root_prim)

        if tree.chords is None:
            continue
        for chord_jid in tree.chords:
            if chord_jid < 0:
                continue
            chord_path = joint_to_path.get(int(chord_jid))
            if chord_path is None:
                msg.warning(
                    "Skipping `excludeFromArticulation` for tree %d's chord %s: no USD prim.",
                    tree_idx,
                    chord_jid,
                )
                continue
            chord_prim = output_stage.GetPrimAtPath(Sdf.Path(chord_path))
            if not chord_prim or not chord_prim.IsValid():
                msg.warning(
                    "Skipping `excludeFromArticulation` for tree %d's chord %s: prim at `%s` is invalid.",
                    tree_idx,
                    chord_jid,
                    chord_path,
                )
                continue
            attr = chord_prim.CreateAttribute(
                "physics:excludeFromArticulation",
                Sdf.ValueTypeNames.Bool,
                custom=False,
                variability=Sdf.VariabilityUniform,
            )
            attr.Set(True)

    # 7. Swap ``body0`` / ``body1`` relationship targets (and the matching
    # ``localPos`` / ``localRot`` anchor attributes) for every joint flagged
    # on :attr:`TopologyGraph.reversed_joint_edges`. This mirrors the
    # parent/child swap performed by
    # :func:`apply_discovered_topology_to_builder`, ensuring downstream
    # consumers that re-import the output USD see the BFS-driven parent →
    # child direction baked into the spanning tree.
    for old_j in graph.reversed_joint_edges:
        if old_j < 0:
            continue
        joint_path = joint_to_path.get(int(old_j))
        if joint_path is None:
            msg.warning("Skipping polarity swap for reversed joint %s: no USD prim.", old_j)
            continue
        joint_prim = output_stage.GetPrimAtPath(Sdf.Path(joint_path))
        if not joint_prim or not joint_prim.IsValid():
            msg.warning(
                "Skipping polarity swap for reversed joint %s: prim at `%s` is invalid.",
                old_j,
                joint_path,
            )
            continue
        _swap_usd_joint_polarity(joint_prim)

    # 8. Save and return.
    output_stage.GetRootLayer().Save()
    return output_path


###
# Internals
###


def _per_joint_counts(starts: list[int], total: int) -> list[int]:
    """Return ``counts[j] = starts[j+1] - starts[j]`` (with ``total`` for the last entry)."""
    n = len(starts)
    counts = [0] * n
    for j in range(n - 1):
        counts[j] = starts[j + 1] - starts[j]
    if n > 0:
        counts[n - 1] = total - starts[n - 1]
    return counts


def _permute_per_slot_arrays(
    builder: ModelBuilder,
    attr_names: tuple[str, ...],
    new_to_old_joint: list[int],
    old_starts: list[int],
    old_counts: list[int],
) -> None:
    """Re-emit per-axis / per-coord / per-dof / per-cts arrays in new joint order.

    Reads the OLD slice ``[old_starts[j] : old_starts[j] + old_counts[j]]`` for
    each old joint ``j`` (looked up via ``new_to_old_joint``), and concatenates
    the slices into the new flat array.

    Args:
        builder: The builder being permuted in place.
        attr_names: Tuple of attribute names that share the same slice layout
            (e.g. ``("joint_q",)`` for per-coord arrays).
        new_to_old_joint: Inverse joint permutation.
        old_starts: Per-joint start offsets in the original layout.
        old_counts: Per-joint slice lengths in the original layout.
    """
    nj = len(new_to_old_joint)
    for attr in attr_names:
        old_list = getattr(builder, attr)
        if not old_list:
            continue
        new_list: list = []
        for new_j in range(nj):
            old_j = new_to_old_joint[new_j]
            start = old_starts[old_j]
            count = old_counts[old_j]
            new_list.extend(old_list[start : start + count])
        setattr(builder, attr, new_list)


def _build_per_slot_index_remap(
    joint_remap: list[int],
    old_starts: list[int],
    old_counts: list[int],
    new_starts: list[int],
) -> list[int]:
    """Build a per-slot (per-DOF / per-coord / per-cts) old→new index remap.

    For each old joint ``j``, the slice ``[old_starts[j] : old_starts[j] + old_counts[j]]``
    moves to ``[new_starts[joint_remap[j]] : ...]`` in the new layout. We
    flatten that into a parallel old-index → new-index lookup table.

    Args:
        joint_remap: Forward joint permutation (``joint_remap[old_j] = new_j``).
        old_starts: Per-joint start offsets in the original layout.
        old_counts: Per-joint slice lengths in the original layout.
        new_starts: Per-joint start offsets in the new (post-permute) layout.

    Returns:
        A list ``remap`` with ``len(remap) == sum(old_counts)`` such that
        ``remap[old_idx] == new_idx``.
    """
    total = sum(old_counts)
    remap = [0] * total
    for old_j, new_j in enumerate(joint_remap):
        new_start = new_starts[new_j]
        old_start = old_starts[old_j]
        for k in range(old_counts[old_j]):
            remap[old_start + k] = new_start + k
    return remap


def _remap_inplace(values: list[int], remap: list[int], *, sentinel: int = -1) -> None:
    """Apply ``remap`` to every non-``sentinel`` entry of ``values`` in place."""
    for i, v in enumerate(values):
        if v == sentinel:
            continue
        values[i] = remap[v]


def _load_pxr_modules() -> tuple[Any, Any, Any]:
    """Lazy-import the ``pxr`` modules used by :func:`export_usd_with_discovered_topology`.

    Returns:
        A tuple ``(Sdf, Usd, UsdPhysics)`` of the imported pxr submodules.

    Raises:
        ImportError: If ``pxr`` is not installed (typically via ``usd-core``).
    """
    try:
        from pxr import Sdf, Usd, UsdPhysics
    except ImportError as e:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e
    return Sdf, Usd, UsdPhysics


def _swap_usd_joint_polarity(joint_prim: Any) -> None:
    """Swap ``body0`` ↔ ``body1`` and the matching local-anchor attributes on a USD joint prim.

    Operates on raw attribute / relationship names so the helper works
    uniformly across every concrete :class:`UsdPhysics.Joint` subclass
    (Revolute, Prismatic, Spherical, Fixed, Distance, D6, …) — the
    swapped attributes (``physics:body0`` / ``physics:body1`` and the
    matching ``physics:localPos0`` / ``physics:localPos1`` /
    ``physics:localRot0`` / ``physics:localRot1``) are defined on the
    base :class:`UsdPhysics.Joint` schema. Joint-local axis tokens
    (``physics:axis``, D6 limit / drive APIs, etc.) are intentionally
    left untouched; see :func:`export_usd_with_discovered_topology` for
    the rationale.

    Missing or unauthored anchor attributes are tolerated: only the
    sides that report :meth:`Usd.Attribute.IsValid` are read and written.
    Missing ``physics:body0`` / ``physics:body1`` relationships cause
    the function to no-op rather than raise so callers can apply the
    swap defensively across mixed asset libraries.

    Args:
        joint_prim: A :class:`Usd.Prim` referring to a USD physics joint
            whose ``body0`` / ``body1`` polarity should be swapped.
    """
    body0_rel = joint_prim.GetRelationship("physics:body0")
    body1_rel = joint_prim.GetRelationship("physics:body1")
    if not body0_rel or not body1_rel:
        return

    body0_targets = list(body0_rel.GetTargets())
    body1_targets = list(body1_rel.GetTargets())
    body0_rel.SetTargets(body1_targets)
    body1_rel.SetTargets(body0_targets)

    local_pos_0 = joint_prim.GetAttribute("physics:localPos0")
    local_pos_1 = joint_prim.GetAttribute("physics:localPos1")
    local_rot_0 = joint_prim.GetAttribute("physics:localRot0")
    local_rot_1 = joint_prim.GetAttribute("physics:localRot1")

    p0 = local_pos_0.Get() if (local_pos_0 and local_pos_0.IsValid()) else None
    p1 = local_pos_1.Get() if (local_pos_1 and local_pos_1.IsValid()) else None
    r0 = local_rot_0.Get() if (local_rot_0 and local_rot_0.IsValid()) else None
    r1 = local_rot_1.Get() if (local_rot_1 and local_rot_1.IsValid()) else None

    if p0 is not None and local_pos_1 and local_pos_1.IsValid():
        local_pos_1.Set(p0)
    if p1 is not None and local_pos_0 and local_pos_0.IsValid():
        local_pos_0.Set(p1)
    if r0 is not None and local_rot_1 and local_rot_1.IsValid():
        local_rot_1.Set(r0)
    if r1 is not None and local_rot_0 and local_rot_0.IsValid():
        local_rot_0.Set(r1)
