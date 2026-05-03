# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``kamino.topology`` module.

Covers:

- :class:`TopologyComponent` ``__post_init__`` invariants.

- :class:`TopologyGraph` constructor validation, component parsing, and
  end-to-end topology discovery against imported asset models.

- :class:`TopologyMinimumDepthSpanningTreeGenerator` tree enumeration, root
  selection, traversal/balancing options and the Featherstone numbering
  invariants of the produced :class:`TopologySpanningTree` instances.

- :class:`TopologyHeaviestBodyBaseSelector` mass-based base
  selection and synthetic FREE base-edge generation.

- :class:`TopologyMinimumDepthSpanningTreeSelector`
  depth-and-balance candidate selection.
"""

from __future__ import annotations

import unittest

import warp as wp

import newton
from newton._src.sim import Model, ModelBuilder
from newton._src.solvers.kamino._src import topology
from newton._src.solvers.kamino._src.core.bodies import RigidBodyDescriptor
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.joints import JointDescriptor, JointDoFType
from newton._src.solvers.kamino._src.topology.utils import (
    NEWTON_TO_KAMINO_JOINT_TYPE,
    apply_discovered_topology_to_builder,
    bodies_from_builder,
    discover_topology_for_builder,
    export_usd_with_discovered_topology,
    extract_graph_inputs_from_builder,
    joints_from_builder,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Helpers
###


def _make_test_graph(unsorted_nodes: bool = False) -> tuple[list[int], list[tuple[int, int, tuple[int, int]]]]:
    # Define a test graph with various node types:
    if unsorted_nodes:
        nodes = [5, 4, 10, 14, 7, 2, 3, 0, 6, 8, 9, 11, 12, 15, 13, 1]
    else:
        nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    edges = [
        (-1, 0, (-1, 0)),
        (-1, 1, (0, 1)),
        (-1, 2, (1, 2)),
        (-1, 3, (2, 3)),
        (-1, 4, (3, 4)),
        (-1, 5, (4, 1)),
        (-1, 6, (4, 0)),  # island: [0, 1, 2, 3, 4]
        (-1, 7, (-1, 8)),  # connected orphan: 8
        (-1, 8, (5, 6)),
        (-1, 9, (6, 7)),
        (-1, 10, (7, 10)),
        (-1, 11, (10, 11)),
        (-1, 12, (11, 5)),  # island: [5, 6, 7, 10, 11]
        (-1, 13, (-1, 9)),  # connected orphan: 9
        # 12-15: isolated orphans with no edges
    ]
    return nodes, edges


def _build_graph(nodes: list[int], edges: list[tuple]) -> topology.TopologyGraph:
    """Construct a :class:`TopologyGraph` and parse its components without triggering the full auto-pipeline."""
    G = topology.TopologyGraph(nodes, edges, autoparse=False)
    G.parse_components()
    return G


def _assert_featherstone_invariants(testcase: unittest.TestCase, tree: topology.TopologySpanningTree) -> None:
    """
    Validate the per-tree numbering invariants documented on the schema.

    - ``parents[0] == -1`` (root's parent is the world node).
    - ``parents[k] < k`` for ``k >= 1`` (regular numbering).
    - Tree consistency: ``parents[child] == p`` iff ``child in children[p]``.
    - Subtree (``v``) and support (``κ``) cross-relations:
      ``μ(i) ⊆ v(i)`` and ``j ∈ κ(i) ⇒ i ∈ v(j)``.
    """
    nb = tree.num_bodies
    testcase.assertEqual(tree.parents[0], -1)
    for k in range(1, nb):
        testcase.assertLess(tree.parents[k], k, f"parents[{k}]={tree.parents[k]} must be < {k}")
    # children<->parents bijection
    for k in range(1, nb):
        p = tree.parents[k]
        testcase.assertIn(k, tree.children[p])
    for i in range(nb):
        for c in tree.children[i]:
            testcase.assertEqual(tree.parents[c], i)
    # μ(i) ⊆ v(i)
    for i in range(nb):
        for c in tree.children[i]:
            testcase.assertIn(c, tree.subtree[i])
    # j ∈ κ(i) ⇒ i ∈ v(j) (here support stores arc local positions which
    # equal the local body position of the child end of the arc).
    for i in range(nb):
        for j in tree.support[i]:
            testcase.assertIn(i, tree.subtree[j])


def _make_tree(
    *,
    depth: int,
    num_bodies: int = 4,
    children: list[list[int]] | None = None,
) -> topology.TopologySpanningTree:
    """Lightweight :class:`TopologySpanningTree` factory for selector tests.

    Only the fields read by :class:`TopologyMinimumDepthSpanningTreeSelector`
    (``depth``, ``num_bodies``, ``children``) are populated — the other
    array-shaped fields are left at their dataclass defaults of ``None``.
    """
    if children is None:
        # Default: single-spine tree (most unbalanced for the given size).
        children = [[i + 1] if i + 1 < num_bodies else [] for i in range(num_bodies)]
    return topology.TopologySpanningTree(
        traversal="bfs",
        depth=depth,
        num_bodies=num_bodies,
        num_joints=num_bodies,
        num_tree_arcs=num_bodies,
        num_tree_chords=0,
        children=children,
    )


def _topology_inputs_from_kamino_builder(
    builder: ModelBuilderKamino,
) -> list[tuple[list[int], list[topology.EdgeType]]]:
    """Extract per-world ``(nodes, edges)`` pairs from a :class:`ModelBuilderKamino`.

    Returns one ``(nodes, edges)`` pair per world. Body and joint indices in the
    edges are the global indices in the merged builder, so the resulting
    ``TopologyGraph`` will discover one component per articulation within each
    world automatically.
    """
    per_world_nodes: list[list[int]] = [[] for _ in range(builder.num_worlds)]
    for i, body in enumerate(builder.all_bodies):
        per_world_nodes[body.wid].append(i)

    world_body_offsets = [0]
    for w in range(builder.num_worlds):
        world_body_offsets.append(world_body_offsets[-1] + len(builder.bodies[w]))

    per_world_edges: list[list[topology.EdgeType]] = [[] for _ in range(builder.num_worlds)]
    for j, joint in enumerate(builder.all_joints):
        bid_p = joint.bid_B + world_body_offsets[joint.wid] if joint.bid_B >= 0 else -1
        bid_s = joint.bid_F + world_body_offsets[joint.wid] if joint.bid_F >= 0 else -1
        per_world_edges[joint.wid].append((joint.dof_type.value, j, (bid_p, bid_s)))

    return list(zip(per_world_nodes, per_world_edges, strict=True))


def _assert_grounded_topology_invariants(
    testcase: unittest.TestCase,
    graph: topology.TopologyGraph,
    *,
    expected_num_components: int | None = None,
    expected_num_synthetic_edges: int = 0,
) -> None:
    """Assert the structural invariants we expect from a discovered topology graph.

    Used as a single shared check by the USD-asset tests so they only need to
    declare the expected component count (one per robot instance) and the
    expected number of synthetic base edges (zero for already-grounded assets,
    one per ungrounded component otherwise) and let this helper enforce:
    connectedness, the synthetic-edge accounting, contiguous remap coverage
    over the body indices each tree references, and Featherstone numbering on
    every selected tree.
    """
    testcase.assertIsNotNone(graph.components)
    if expected_num_components is not None:
        testcase.assertEqual(len(graph.components), expected_num_components)
    testcase.assertIsNotNone(graph.trees)
    testcase.assertEqual(len(graph.trees), len(graph.components))
    # Every component must end up connected — either directly via a base/
    # grounding edge or indirectly via a synthetic edge minted by the selector.
    for comp in graph.components:
        testcase.assertTrue(comp.is_connected, f"Component is not connected: {comp}")
    # Synthetic base edges must match the expected count.
    actual_synthetic = len(graph.new_base_edges) if graph.new_base_edges else 0
    testcase.assertEqual(
        actual_synthetic,
        expected_num_synthetic_edges,
        f"Expected {expected_num_synthetic_edges} synthetic base edges, got {actual_synthetic}: {graph.new_base_edges}",
    )
    # Every selected tree must obey Featherstone numbering.
    for tree in graph.trees:
        _assert_featherstone_invariants(testcase, tree)
    # Body remap is dense over the body indices each tree references; the
    # remapped slice over those bodies must be a permutation of [0, N).
    body_remap = graph.body_node_remap
    if body_remap:
        referenced_bodies: set[int] = set()
        for tree in graph.trees:
            if tree.component is not None and tree.component.nodes is not None:
                referenced_bodies.update(int(n) for n in tree.component.nodes)
        for tree in graph.trees:
            if tree.root is not None:
                testcase.assertGreaterEqual(body_remap[int(tree.root)], 0)
        remapped_referenced = sorted(body_remap[b] for b in referenced_bodies)
        testcase.assertEqual(remapped_referenced, list(range(len(referenced_bodies))))


###
# Validation helpers
###


class TestValidationHelpers(unittest.TestCase):
    """Tests for the module-level validation helpers in :mod:`...topology.types`."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_validate_max_candidates_rejects_bool(self):
        """Boolean inputs are rejected because ``bool`` subclasses ``int``."""
        with self.assertRaises(TypeError):
            topology.types.validate_max_candidates(True)
        with self.assertRaises(TypeError):
            topology.types.validate_max_candidates(False)

    def test_validate_max_candidates_accepts_none_and_positive_int(self):
        """``None`` and positive integers must pass."""
        topology.types.validate_max_candidates(None)
        topology.types.validate_max_candidates(1)
        topology.types.validate_max_candidates(32)

    def test_validate_max_candidates_rejects_non_positive(self):
        """Zero and negative integers must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            topology.types.validate_max_candidates(0)
        with self.assertRaises(ValueError):
            topology.types.validate_max_candidates(-1)


###
# Nodes & Edges
###


class TestNodesEdges(unittest.TestCase):
    """
    Sanity checks that the user-facing :data:`EdgeType` / :data:`NodeType`
    unions accept both the dataclass form and the legacy primitive form,
    and that the canonical :meth:`from_input` coercion is idempotent.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_graph_edge_from_input_accepts_tuple_and_graph_edge(self):
        """``GraphEdge.from_input`` accepts both forms and round-trips via :meth:`to_tuple`."""
        legacy = (JointDoFType.REVOLUTE.value, 7, (0, 1))
        from_tuple = topology.GraphEdge.from_input(legacy)
        self.assertIsInstance(from_tuple, topology.GraphEdge)
        self.assertEqual(from_tuple.joint_type, JointDoFType.REVOLUTE.value)
        self.assertEqual(from_tuple.joint_index, 7)
        self.assertEqual(from_tuple.nodes, (0, 1))
        self.assertEqual(from_tuple.to_tuple(), legacy)
        # Idempotent on an existing GraphEdge.
        self.assertIs(topology.GraphEdge.from_input(from_tuple), from_tuple)

    def test_graph_node_from_input_accepts_int_and_graph_node(self):
        """``GraphNode.from_input`` accepts both forms; ``__int__`` returns the index."""
        from_int = topology.GraphNode.from_input(5)
        self.assertIsInstance(from_int, topology.GraphNode)
        self.assertEqual(from_int.index, 5)
        self.assertIsNone(from_int.name)
        self.assertEqual(int(from_int), 5)
        # Idempotent on an existing GraphNode.
        self.assertIs(topology.GraphNode.from_input(from_int), from_int)

    def test_graph_node_equality_ignores_name(self):
        """``GraphNode`` equality and hashing must depend on ``index`` only.

        Two nodes with the same ``index`` but different ``name`` values
        represent the same body and must compare equal so that
        membership checks (``base_node in nodes``) work consistently
        regardless of the user passing a plain ``int`` or a named
        :class:`GraphNode`.
        """
        named = topology.GraphNode(index=3, name="body_3")
        unnamed = topology.GraphNode(index=3)
        renamed = topology.GraphNode(index=3, name="other")
        self.assertEqual(named, unnamed)
        self.assertEqual(named, renamed)
        self.assertEqual(hash(named), hash(unnamed))
        self.assertEqual(hash(named), hash(renamed))
        # Distinct indices are still distinct.
        self.assertNotEqual(named, topology.GraphNode(index=4, name="body_3"))

    def test_topology_component_base_node_membership_with_named_nodes(self):
        """``base_node`` (plain int) must be accepted when ``nodes`` carries named :class:`GraphNode` instances.

        Regression for an over-strict membership check that used full
        dataclass equality (including ``name``) and rejected a plain
        ``int`` ``base_node`` against a list of named nodes.
        """
        named_b0 = topology.GraphNode(index=0, name="body_0")
        named_b1 = topology.GraphNode(index=1, name="body_1")
        comp = topology.TopologyComponent(
            nodes=[named_b0, named_b1],
            edges=[
                (JointDoFType.FREE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            ],
            base_node=0,
            base_edge=(JointDoFType.FREE.value, 0, (-1, 0)),
            is_island=True,
            is_connected=True,
        )
        self.assertEqual(int(comp.base_node), 0)
        self.assertIn(comp.base_node, comp.nodes)

    def test_topology_component_accepts_either_edge_form(self):
        """``TopologyComponent`` normalizes mixed :data:`EdgeType` inputs to :class:`GraphEdge`."""
        legacy = (JointDoFType.REVOLUTE.value, 0, (-1, 0))
        graph_edge = topology.GraphEdge(joint_type=JointDoFType.REVOLUTE.value, joint_index=1, nodes=(0, 1))
        comp = topology.TopologyComponent(
            nodes=[0, 1],
            edges=[legacy, graph_edge],
            ground_nodes=[0],
            ground_edges=[legacy],
            base_node=None,
            base_edge=None,
            is_island=True,
            is_connected=True,
        )
        self.assertTrue(all(isinstance(e, topology.GraphEdge) for e in comp.edges))
        self.assertTrue(all(isinstance(e, topology.GraphEdge) for e in comp.ground_edges))


###
# TopologyComponent
###


class TestTopologyComponent(unittest.TestCase):
    """Negative tests for :meth:`TopologyComponent.__post_init__` invariants."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_grounding_edge_with_free_joint_is_rejected(self):
        """A 6-DoF FREE joint among ``ground_edges`` violates modelling conventions."""
        with self.assertRaisesRegex(ValueError, "FREE"):
            topology.TopologyComponent(
                nodes=[0],
                edges=[(JointDoFType.FREE.value, 0, (-1, 0))],
                ground_nodes=[0],
                ground_edges=[(JointDoFType.FREE.value, 0, (-1, 0))],
                is_island=False,
            )

    def test_base_node_without_base_edge_is_rejected(self):
        """``base_node`` set without ``base_edge`` (and vice versa) must fail."""
        with self.assertRaisesRegex(ValueError, "both"):
            topology.TopologyComponent(nodes=[0, 1], base_node=0, is_island=True)
        with self.assertRaisesRegex(ValueError, "both"):
            topology.TopologyComponent(
                nodes=[0, 1],
                base_edge=(JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                is_island=True,
            )

    def test_base_node_not_in_nodes_is_rejected(self):
        """``base_node`` must be a member of ``nodes``."""
        with self.assertRaisesRegex(ValueError, "not contained in component nodes"):
            topology.TopologyComponent(
                nodes=[0, 1],
                base_node=2,
                base_edge=(JointDoFType.REVOLUTE.value, 0, (-1, 2)),
                is_island=True,
            )

    def test_base_node_not_endpoint_of_base_edge_is_rejected(self):
        """``base_node`` must appear in the body-pair of ``base_edge``."""
        with self.assertRaisesRegex(ValueError, "not an endpoint"):
            topology.TopologyComponent(
                nodes=[0, 1, 2],
                base_node=0,
                base_edge=(JointDoFType.REVOLUTE.value, 0, (-1, 1)),
                is_island=True,
            )

    def test_is_island_inconsistent_with_node_count_is_rejected(self):
        """``is_island`` must agree with ``len(nodes) > 1``."""
        with self.assertRaisesRegex(ValueError, "is_island"):
            topology.TopologyComponent(nodes=[0], is_island=True)
        with self.assertRaisesRegex(ValueError, "is_island"):
            topology.TopologyComponent(nodes=[0, 1], is_island=False)

    def test_ground_nodes_must_match_endpoints_of_ground_edges(self):
        """``set(ground_nodes)`` must equal the non-world endpoints of ``ground_edges``."""
        with self.assertRaisesRegex(ValueError, "does not match"):
            topology.TopologyComponent(
                nodes=[0, 1, 2],
                edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
                ground_nodes=[0, 1],
                ground_edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
                is_island=True,
            )

    def test_ground_nodes_set_without_ground_edges_is_rejected(self):
        """Regression: ``ground_nodes`` and ``ground_edges`` must be both set or both unset.

        A populated ``ground_nodes`` paired with ``ground_edges=None``
        violated the dual-storage contract relied on by every downstream
        consumer (selectors, base-promotion helpers) but was previously
        accepted because the existing endpoint-matching guard only ran
        when *both* fields were non-``None``.
        """
        with self.assertRaisesRegex(ValueError, "both"):
            topology.TopologyComponent(
                nodes=[0, 1],
                edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
                ground_nodes=[0],
                ground_edges=None,
                is_island=True,
            )

    def test_ground_edges_set_without_ground_nodes_is_rejected(self):
        """Regression: companion to the above — ``ground_edges`` set with ``ground_nodes=None`` must fail."""
        with self.assertRaisesRegex(ValueError, "both"):
            topology.TopologyComponent(
                nodes=[0, 1],
                edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
                ground_nodes=None,
                ground_edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
                is_island=True,
            )

    def test_duplicate_ground_node_indices_are_rejected(self):
        """Regression: ``ground_nodes`` must not list the same body index twice.

        Two :class:`GraphNode` instances differing only in their optional
        ``name`` metadata still collide on body index and would silently
        inflate per-component grounding statistics.
        """
        named_a = topology.GraphNode(index=0, name="a")
        named_b = topology.GraphNode(index=0, name="b")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            topology.TopologyComponent(
                nodes=[topology.GraphNode(index=0, name="canonical"), 1],
                edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
                ground_nodes=[named_a, named_b],
                ground_edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
                is_island=True,
                is_connected=True,
            )

    ###
    # Cross-validator interactions
    ###

    def test_base_edge_other_endpoint_must_be_world(self):
        """``base_edge`` must connect ``base_node`` to the implicit world node."""
        with self.assertRaisesRegex(ValueError, "world"):
            topology.TopologyComponent(
                nodes=[5, 7],
                edges=[(JointDoFType.REVOLUTE.value, 0, (5, 7))],
                base_node=5,
                # Both endpoints are body nodes — invalid base edge.
                base_edge=(JointDoFType.REVOLUTE.value, 0, (5, 7)),
                is_island=True,
                is_connected=True,
            )

    def test_is_connected_must_match_world_link_state(self):
        """``is_connected`` must agree with the presence of a base or grounding edge."""
        # is_connected=True but no base/grounding link → must raise.
        with self.assertRaisesRegex(ValueError, "is_connected"):
            topology.TopologyComponent(
                nodes=[1, 2],
                edges=[(JointDoFType.REVOLUTE.value, 0, (1, 2))],
                ground_nodes=[],
                ground_edges=[],
                base_node=None,
                base_edge=None,
                is_island=True,
                is_connected=True,
            )
        # is_connected=False but a grounding edge exists → must raise.
        with self.assertRaisesRegex(ValueError, "is_connected"):
            topology.TopologyComponent(
                nodes=[1, 2],
                edges=[
                    (JointDoFType.REVOLUTE.value, 0, (1, 2)),
                    (JointDoFType.REVOLUTE.value, 1, (1, -1)),
                ],
                ground_nodes=[1],
                ground_edges=[(JointDoFType.REVOLUTE.value, 1, (1, -1))],
                base_node=None,
                base_edge=None,
                is_island=True,
                is_connected=False,
            )

    def test_is_connected_true_without_links_raises_when_fields_unset(self):
        """``is_connected=True`` must raise even when ``base_edge`` and ``ground_edges`` are left ``None``.

        Regression for a validation gap where the outer guard required at
        least one of the link fields to be non-``None`` before checking
        the connectivity assertion, allowing
        ``is_connected=True`` + ``base_edge=None`` + ``ground_edges=None``
        to silently slip through.
        """
        with self.assertRaisesRegex(ValueError, "is_connected"):
            topology.TopologyComponent(
                nodes=[1, 2],
                edges=[(JointDoFType.REVOLUTE.value, 0, (1, 2))],
                is_island=True,
                is_connected=True,
            )

    def test_joint_type_out_of_jointdoftype_range_is_rejected(self):
        """Edge ``joint_type`` integers must be in ``JointDoFType``'s range or ``-1``."""
        # 999 is far outside JointDoFType's 0..8 range, and is not the -1 sentinel.
        with self.assertRaises(ValueError):
            topology.TopologyComponent(
                nodes=[0, 1],
                edges=[(999, 0, (0, 1))],
                is_island=True,
                is_connected=False,
            )

    ###
    # assign_base() — shared base-promotion helper
    ###

    def test_assign_base_promotes_grounding_edge(self):
        """``assign_base`` drops the promoted edge from ``ground_edges`` and recomputes ``ground_nodes``."""
        # Body 0 has two grounding edges; promote one of them to base.
        ground = [
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (-1, 0)),
        ]
        comp = topology.TopologyComponent(
            nodes=[0, 1],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 2, (0, 1)),
            ],
            ground_nodes=[0],
            ground_edges=list(ground),
            is_island=True,
            is_connected=True,
        )

        comp.assign_base(base_node=0, base_edge=ground[0])

        self.assertEqual(int(comp.base_node), 0)
        self.assertEqual(comp.base_edge.to_tuple(), ground[0])
        self.assertTrue(comp.is_connected)
        # The promoted edge must no longer appear among ground_edges, but the
        # *other* grounding edge (and its endpoint) must still be tracked.
        self.assertEqual([e.to_tuple() for e in comp.ground_edges], [ground[1]])
        self.assertEqual([int(n) for n in comp.ground_nodes], [0])

    def test_assign_base_rejects_non_world_peer(self):
        """``assign_base`` re-runs ``__post_init__`` and surfaces invalid bases."""
        comp = topology.TopologyComponent(
            nodes=[0, 1],
            edges=[(JointDoFType.REVOLUTE.value, 0, (0, 1))],
            is_island=True,
            is_connected=False,
        )
        # Edge endpoints are both bodies (no world endpoint) — must raise.
        with self.assertRaisesRegex(ValueError, "world"):
            comp.assign_base(base_node=0, base_edge=(JointDoFType.REVOLUTE.value, 0, (0, 1)))

    def test_assign_base_preserves_remaining_ground_node_names(self):
        """``assign_base`` keeps the canonical :class:`GraphNode` instances of the surviving grounding nodes.

        The remaining-grounding-nodes recomputation must filter the
        existing ``GraphNode`` list so optional ``name`` metadata is not
        dropped, rather than rebuilding the list from raw integer indices.
        """
        named_b0 = topology.GraphNode(index=0, name="body_0")
        named_b1 = topology.GraphNode(index=1, name="body_1")
        # Body 0 has its own grounding edge; body 1 has a separate grounding edge
        # that we will promote to base. Body 0's grounding (and its name) must
        # remain after the promotion.
        comp = topology.TopologyComponent(
            nodes=[named_b0, named_b1],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (-1, 1)),
                (JointDoFType.REVOLUTE.value, 2, (0, 1)),
            ],
            ground_nodes=[named_b0, named_b1],
            ground_edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (-1, 1)),
            ],
            is_island=True,
            is_connected=True,
        )

        comp.assign_base(base_node=1, base_edge=(JointDoFType.REVOLUTE.value, 1, (-1, 1)))

        self.assertEqual(int(comp.base_node), 1)
        self.assertEqual([int(n) for n in comp.ground_nodes], [0])
        self.assertEqual(comp.ground_nodes[0].name, "body_0")


###
# TopologySpanningTree
###


class TestTopologySpanningTree(unittest.TestCase):
    """Direct tests for :class:`TopologySpanningTree` operations."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_balanced_score_matches_sum_of_squared_child_counts(self):
        """``balanced_score`` returns ``sum(len(c) ** 2 for c in children)``.

        Lower scores indicate more balanced trees: a fully balanced
        binary split scores lower than a single spine of equal depth.
        """
        # A tree with root having 2 children, each with 2 children = score 4 + 4 + 4 = 12.
        balanced = topology.TopologySpanningTree(num_bodies=7, children=[[1, 2], [3, 4], [5, 6], [], [], [], []])
        self.assertEqual(balanced.balanced_score(), 4 + 4 + 4)

        # A spine: every internal node has one child = score 1 + 1 + 1 = 3 (most balanced for spine shape).
        spine = topology.TopologySpanningTree(num_bodies=4, children=[[1], [2], [3], []])
        self.assertEqual(spine.balanced_score(), 3)

        # A star: root has 3 children = score 9.
        star = topology.TopologySpanningTree(num_bodies=4, children=[[1, 2, 3], [], [], []])
        self.assertEqual(star.balanced_score(), 9)

    def test_balanced_score_raises_when_children_is_none(self):
        """``balanced_score`` requires the ``children`` array to be populated."""
        tree = topology.TopologySpanningTree(num_bodies=3, children=None)
        with self.assertRaisesRegex(ValueError, "children=None"):
            tree.balanced_score()


###
# TopologySpanningTree.with_offsets / .remapped
###


def _build_chain_tree(num_bodies: int = 3, base_joint_idx: int = 0) -> topology.TopologySpanningTree:
    """Run the live pipeline to produce a single-component chain tree.

    The resulting :class:`TopologySpanningTree` has every parallel-array
    field populated by the real generator, so the helper is a convenient
    fixture for the offset / remap / reassignment tests below.
    """
    nodes = list(range(num_bodies))
    edges = [(JointDoFType.FREE.value, base_joint_idx, (-1, 0))]
    for i in range(1, num_bodies):
        edges.append((JointDoFType.REVOLUTE.value, base_joint_idx + i, (i - 1, i)))
    G = topology.TopologyGraph(nodes, edges, autoparse=True, reassign_indices_inplace=False)
    return G.trees[0]


def _build_isolated_orphan_tree() -> topology.TopologySpanningTree:
    """Build the trivial tree for an orphan with no edges (sentinel ``arcs[0]``).

    The resulting tree intentionally has ``arcs == []`` and ``num_tree_arcs == 0``
    so the offset / remap helpers can be exercised against the sentinel-edge case.
    """
    nodes = [0]
    G = topology.TopologyGraph(nodes, edges=[], autoparse=False)
    G.parse_components()
    G.generate_spanning_trees()
    G.select_spanning_trees()
    return G.trees[0]


def _build_isolated_island_tree_with_sentinel_arc() -> topology.TopologySpanningTree:
    """Build a tree from an isolated island whose ``arcs[0]`` carries the sentinel.

    Built without invoking the base selector so the spanning-tree generator
    falls back to the ``NO_BASE_JOINT_INDEX`` sentinel for slot 0 — exactly
    the case the offset / remap helpers must pass through unchanged.
    """
    nodes = [3, 4]
    edges = [(JointDoFType.REVOLUTE.value, 0, (3, 4))]
    G = topology.TopologyGraph(nodes, edges, autoparse=False)
    G.parse_components()
    G.generate_spanning_trees(override_priorities=True)
    G.select_spanning_trees()
    return G.trees[0]


class TestTopologySpanningTreeOffsets(unittest.TestCase):
    """Tests for :meth:`TopologySpanningTree.with_offsets`."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_zero_offsets_returns_an_independent_copy(self):
        """``with_offsets(0, 0)`` preserves indices but returns a fresh instance."""
        tree = _build_chain_tree()
        shifted = tree.with_offsets(0, 0)
        self.assertIsNot(shifted, tree)
        self.assertEqual(shifted.root, tree.root)
        self.assertEqual(shifted.arcs, tree.arcs)
        self.assertEqual(shifted.chords, tree.chords)

    def test_offsets_shift_root_arcs_and_chords(self):
        """Body and joint offsets translate the global-index fields."""
        tree = _build_chain_tree(num_bodies=3, base_joint_idx=0)
        shifted = tree.with_offsets(body_node_offset=10, joint_edge_offset=20)
        self.assertEqual(shifted.root, tree.root + 10)
        self.assertEqual(shifted.arcs, [a + 20 for a in tree.arcs])
        # Empty chord list survives the shift unchanged.
        self.assertEqual(shifted.chords, [])

    def test_offsets_preserve_local_position_fields(self):
        """Local-position fields are local to the tree and must be left alone."""
        tree = _build_chain_tree()
        shifted = tree.with_offsets(7, 11)
        self.assertEqual(shifted.parents, tree.parents)
        self.assertEqual(shifted.predecessors, tree.predecessors)
        self.assertEqual(shifted.successors, tree.successors)
        self.assertEqual(shifted.children, tree.children)
        self.assertEqual(shifted.subtree, tree.subtree)
        self.assertEqual(shifted.support, tree.support)

    def test_offsets_drop_component_reference(self):
        """The returned tree drops the source ``component`` reference."""
        tree = _build_chain_tree()
        self.assertIsNotNone(tree.component)
        shifted = tree.with_offsets(1, 1)
        self.assertIsNone(shifted.component)

    def test_offsets_pass_sentinel_arcs_through(self):
        """Sentinel arcs (``NO_BASE_JOINT_INDEX``) must not be shifted."""
        tree = _build_isolated_island_tree_with_sentinel_arc()
        # Sanity: the fixture must actually surface a sentinel arc to make
        # this test meaningful.
        self.assertEqual(tree.arcs[0], topology.types.NO_BASE_JOINT_INDEX)
        shifted = tree.with_offsets(50, 50)
        self.assertEqual(shifted.arcs[0], topology.types.NO_BASE_JOINT_INDEX)
        # Real arcs (after the sentinel slot) must still be shifted.
        for i in range(1, len(tree.arcs)):
            self.assertEqual(shifted.arcs[i], tree.arcs[i] + 50)

    def test_offsets_round_trip_is_identity(self):
        """Shifting by ``(a, b)`` and then ``(-a, -b)`` returns the original indices."""
        tree = _build_chain_tree()
        shifted = tree.with_offsets(13, 17).with_offsets(-13, -17)
        self.assertEqual(shifted.root, tree.root)
        self.assertEqual(shifted.arcs, tree.arcs)
        self.assertEqual(shifted.chords, tree.chords)

    def test_offsets_handle_orphan_with_empty_arcs(self):
        """Orphans whose ``arcs`` list is empty must return an empty arcs list."""
        tree = _build_isolated_orphan_tree()
        shifted = tree.with_offsets(5, 5)
        self.assertEqual(shifted.arcs, [])
        self.assertEqual(shifted.root, tree.root + 5)


class TestTopologySpanningTreeRemapped(unittest.TestCase):
    """Tests for :meth:`TopologySpanningTree.remapped`."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_none_remap_keeps_indices(self):
        """``None`` remap arguments are no-ops."""
        tree = _build_chain_tree()
        out = tree.remapped(None, None)
        self.assertEqual(out.root, tree.root)
        self.assertEqual(out.arcs, tree.arcs)
        self.assertEqual(out.chords, tree.chords)

    def test_identity_remap_keeps_indices(self):
        """An identity-list remap returns the same global indices."""
        tree = _build_chain_tree(num_bodies=3, base_joint_idx=0)
        body_remap = list(range(tree.num_bodies))
        joint_remap = list(range(tree.num_joints))
        out = tree.remapped(body_remap, joint_remap)
        self.assertEqual(out.root, tree.root)
        self.assertEqual(out.arcs, tree.arcs)
        self.assertEqual(out.chords, tree.chords)

    def test_permutation_remap_translates_root_arcs_and_chords(self):
        """A non-identity remap permutes ``root``, ``arcs``, ``chords`` accordingly."""
        tree = _build_chain_tree(num_bodies=3, base_joint_idx=0)
        # Reverse body/joint orders so the test catches "off-by-one" mistakes.
        body_remap = [tree.num_bodies - 1 - i for i in range(tree.num_bodies)]
        joint_remap = [tree.num_joints - 1 - i for i in range(tree.num_joints)]
        out = tree.remapped(body_remap, joint_remap)
        self.assertEqual(out.root, body_remap[tree.root])
        self.assertEqual(out.arcs, [joint_remap[a] for a in tree.arcs])
        self.assertEqual(out.chords, [joint_remap[c] for c in tree.chords])

    def test_remap_preserves_local_position_fields(self):
        """Local-position fields are local to the tree and must be left alone."""
        tree = _build_chain_tree()
        body_remap = list(range(tree.num_bodies))
        joint_remap = list(range(tree.num_joints))
        out = tree.remapped(body_remap, joint_remap)
        self.assertEqual(out.parents, tree.parents)
        self.assertEqual(out.predecessors, tree.predecessors)
        self.assertEqual(out.successors, tree.successors)
        self.assertEqual(out.children, tree.children)
        self.assertEqual(out.subtree, tree.subtree)
        self.assertEqual(out.support, tree.support)

    def test_remap_drops_component_reference(self):
        """The returned tree drops the source ``component`` reference."""
        tree = _build_chain_tree()
        self.assertIsNotNone(tree.component)
        out = tree.remapped(None, None)
        self.assertIsNone(out.component)

    def test_remap_passes_sentinel_arcs_through(self):
        """Sentinel arc entries must be passed through both remaps unchanged."""
        tree = _build_isolated_island_tree_with_sentinel_arc()
        self.assertEqual(tree.arcs[0], topology.types.NO_BASE_JOINT_INDEX)
        # Provide a remap that would otherwise raise on the sentinel value.
        body_remap = list(range(10))
        joint_remap = list(range(10))
        out = tree.remapped(body_remap, joint_remap)
        self.assertEqual(out.arcs[0], topology.types.NO_BASE_JOINT_INDEX)

    def test_remap_raises_on_out_of_range_body(self):
        """A body index referenced by ``root`` outside the remap raises ``IndexError``."""
        tree = _build_chain_tree(num_bodies=3, base_joint_idx=0)
        # Empty body remap + a non-negative root → out-of-range lookup.
        with self.assertRaises(IndexError):
            tree.remapped([], list(range(tree.num_joints)))

    def test_remap_raises_on_out_of_range_joint(self):
        """A joint index referenced by ``arcs`` outside the remap raises ``IndexError``."""
        tree = _build_chain_tree(num_bodies=3, base_joint_idx=0)
        # Empty joint remap + non-empty arcs → out-of-range lookup.
        with self.assertRaises(IndexError):
            tree.remapped(list(range(tree.num_bodies)), [])


###
# Heaviest-body base selector
###


class TestTopologyHeaviestBodyBaseSelector(unittest.TestCase):
    """Tests for :class:`TopologyHeaviestBodyBaseSelector` in isolation.

    Covers the heaviest-mass selection rule, tie-breaking by lowest body index,
    incident-grounding-edge promotion, FREE preference toggling, isolated-island
    fallback to a synthesized FREE base edge, and the constructor-level input
    validation for the world-node sentinel.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def _bodies(self, masses: list[float]) -> list[RigidBodyDescriptor]:
        """Return a list of :class:`RigidBodyDescriptor` with the given masses."""
        return [RigidBodyDescriptor(name=f"body_{i}", m_i=m, bid=i) for i, m in enumerate(masses)]

    ###
    # Basic selection
    ###

    def test_picks_heaviest_with_own_incident_grounding(self):
        """When the heaviest body has its own grounding edge, that edge is promoted."""
        # Body 2 is heaviest and carries the (-1, 2) grounding edge.
        comp = topology.TopologyComponent(
            nodes=[0, 1, 2],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (-1, 2)),
                (JointDoFType.REVOLUTE.value, 2, (0, 1)),
                (JointDoFType.REVOLUTE.value, 3, (1, 2)),
            ],
            ground_nodes=[0, 2],
            ground_edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (-1, 2)),
            ],
            is_island=True,
            is_connected=True,
        )
        bodies = self._bodies([1.0, 1.0, 4.0])

        sel = topology.TopologyHeaviestBodyBaseSelector()
        base_node, base_edge = sel.select_base(component=comp, bodies=bodies)

        self.assertEqual(int(base_node), 2)
        self.assertEqual(base_edge.joint_index, 1)  # joint index 1 = (-1, 2)
        self.assertEqual(base_edge.joint_type, JointDoFType.REVOLUTE.value)

    def test_synthesizes_free_when_heaviest_has_no_incident_grounding(self):
        """When the heaviest body has no incident grounding edge, a FREE edge is synthesized."""
        # Body 1 is heaviest but has no grounding edge incident; bodies 0 and 2 do.
        comp = topology.TopologyComponent(
            nodes=[0, 1, 2],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (-1, 2)),
                (JointDoFType.REVOLUTE.value, 2, (0, 1)),
                (JointDoFType.REVOLUTE.value, 3, (1, 2)),
            ],
            ground_nodes=[0, 2],
            ground_edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (-1, 2)),
            ],
            is_island=True,
            is_connected=True,
        )
        bodies = self._bodies([1.0, 3.0, 2.0])

        sel = topology.TopologyHeaviestBodyBaseSelector()
        base_node, base_edge = sel.select_base(component=comp, bodies=bodies)

        self.assertEqual(int(base_node), 1)
        self.assertEqual(base_edge.joint_type, JointDoFType.FREE.value)
        self.assertEqual(base_edge.joint_index, -1)
        self.assertEqual(base_edge.nodes, (-1, 1))

    def test_synthesizes_free_for_isolated_island(self):
        """Isolated island (no grounding edges) gets a synthetic FREE base on its heaviest body."""
        # Triangle 0-1-2 with no grounding; body 0 is heaviest.
        comp = topology.TopologyComponent(
            nodes=[0, 1, 2],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (0, 1)),
                (JointDoFType.REVOLUTE.value, 1, (1, 2)),
                (JointDoFType.REVOLUTE.value, 2, (2, 0)),
            ],
            ground_nodes=[],
            ground_edges=[],
            is_island=True,
            is_connected=False,
        )
        bodies = self._bodies([5.0, 1.0, 2.0])

        sel = topology.TopologyHeaviestBodyBaseSelector()
        base_node, base_edge = sel.select_base(component=comp, bodies=bodies)

        self.assertEqual(int(base_node), 0)
        self.assertEqual(base_edge.to_tuple(), (JointDoFType.FREE.value, -1, (-1, 0)))

    def test_orphan_with_no_grounding_synthesizes_free(self):
        """A single-body, edgeless component returns that body and a synthetic FREE edge."""
        comp = topology.TopologyComponent(
            nodes=[3],
            edges=[],
            ground_nodes=[],
            ground_edges=[],
            is_island=False,
            is_connected=False,
        )
        # Provide enough bodies so that index 3 is in range.
        bodies = self._bodies([0.0, 0.0, 0.0, 7.0])

        sel = topology.TopologyHeaviestBodyBaseSelector()
        base_node, base_edge = sel.select_base(component=comp, bodies=bodies)

        self.assertEqual(int(base_node), 3)
        self.assertEqual(base_edge.to_tuple(), (JointDoFType.FREE.value, -1, (-1, 3)))

    ###
    # Tie-breaking
    ###

    def test_tie_on_mass_uses_lowest_index(self):
        """When multiple bodies have identical mass, the lowest body index is selected."""
        comp = topology.TopologyComponent(
            nodes=[0, 1, 2],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (0, 1)),
                (JointDoFType.REVOLUTE.value, 1, (1, 2)),
            ],
            ground_nodes=[],
            ground_edges=[],
            is_island=True,
            is_connected=False,
        )
        bodies = self._bodies([2.0, 2.0, 2.0])

        sel = topology.TopologyHeaviestBodyBaseSelector()
        base_node, _ = sel.select_base(component=comp, bodies=bodies)

        self.assertEqual(int(base_node), 0)

    ###
    # FREE preference flag
    ###

    def test_free_preference_picks_free_over_revolute_when_both_incident(self):
        """`prefer_free_when_available=True` picks a FREE incident edge over a REVOLUTE one.

        Uses post-construction assignment to bypass
        :meth:`TopologyComponent.__post_init__`'s "no FREE in ground_edges" rule —
        this configuration is unreachable through the standard parser flow but
        validates the selector's preference logic in isolation.
        """
        comp = topology.TopologyComponent(
            nodes=[0, 1],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            ],
            ground_nodes=[0],
            ground_edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
            is_island=True,
            is_connected=True,
        )
        # Inject a synthetic FREE incident on the same body to test the preference flag.
        comp.ground_edges = [
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
            (JointDoFType.FREE.value, 1, (-1, 0)),
        ]
        bodies = self._bodies([3.0, 1.0])

        sel = topology.TopologyHeaviestBodyBaseSelector(prefer_free_when_available=True)
        _, base_edge = sel.select_base(component=comp, bodies=bodies)

        self.assertEqual(base_edge.joint_type, JointDoFType.FREE.value)
        self.assertEqual(base_edge.joint_index, 1)

    def test_disable_free_preference_uses_first_incident(self):
        """`prefer_free_when_available=False` picks the first incident grounding edge regardless of joint type."""
        comp = topology.TopologyComponent(
            nodes=[0, 1],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
                (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            ],
            ground_nodes=[0],
            ground_edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
            is_island=True,
            is_connected=True,
        )
        comp.ground_edges = [
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
            (JointDoFType.FREE.value, 1, (-1, 0)),
        ]
        bodies = self._bodies([3.0, 1.0])

        sel = topology.TopologyHeaviestBodyBaseSelector(prefer_free_when_available=False)
        _, base_edge = sel.select_base(component=comp, bodies=bodies)

        # First listed incident edge, regardless of FREE.
        self.assertEqual(base_edge.joint_type, JointDoFType.REVOLUTE.value)
        self.assertEqual(base_edge.joint_index, 0)

    ###
    # Configurable synthetic edge
    ###

    def test_synthetic_edge_uses_configured_world_node(self):
        """A custom ``world_node`` is honored when synthesizing the FREE base edge."""
        comp = topology.TopologyComponent(
            nodes=[0, 1],
            edges=[(JointDoFType.REVOLUTE.value, 0, (0, 1))],
            ground_nodes=[],
            ground_edges=[],
            is_island=True,
            is_connected=False,
        )
        bodies = self._bodies([1.0, 2.0])

        sel = topology.TopologyHeaviestBodyBaseSelector(world_node=-7)
        base_node, base_edge = sel.select_base(component=comp, bodies=bodies)

        self.assertEqual(int(base_node), 1)
        self.assertEqual(base_edge.nodes, (-7, 1))

    ###
    # Failure modes
    ###

    def test_missing_bodies_raises(self):
        """``bodies=None`` raises ``ValueError``."""
        comp = topology.TopologyComponent(
            nodes=[0, 1],
            edges=[(JointDoFType.REVOLUTE.value, 0, (0, 1))],
            ground_nodes=[],
            ground_edges=[],
            is_island=True,
            is_connected=False,
        )
        sel = topology.TopologyHeaviestBodyBaseSelector()
        with self.assertRaisesRegex(ValueError, "bodies"):
            sel.select_base(component=comp, bodies=None)

    def test_none_component_raises(self):
        """``component=None`` raises ``ValueError``."""
        sel = topology.TopologyHeaviestBodyBaseSelector()
        with self.assertRaisesRegex(ValueError, "component"):
            sel.select_base(component=None, bodies=self._bodies([1.0]))

    def test_empty_component_nodes_raises(self):
        """A component with empty/None ``nodes`` raises ``ValueError``."""
        comp = topology.TopologyComponent(
            nodes=[],
            edges=[],
            ground_nodes=[],
            ground_edges=[],
            is_island=False,
        )
        sel = topology.TopologyHeaviestBodyBaseSelector()
        with self.assertRaisesRegex(ValueError, "at least one body"):
            sel.select_base(component=comp, bodies=self._bodies([1.0]))

    def test_node_out_of_range_for_bodies_raises(self):
        """Body indices outside the ``bodies`` list raise ``ValueError``."""
        comp = topology.TopologyComponent(
            nodes=[0, 1, 5],  # body 5 is not present in the bodies list
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (0, 1)),
                (JointDoFType.REVOLUTE.value, 1, (1, 5)),
            ],
            ground_nodes=[],
            ground_edges=[],
            is_island=True,
            is_connected=False,
        )
        sel = topology.TopologyHeaviestBodyBaseSelector()
        with self.assertRaisesRegex(ValueError, "out of range"):
            sel.select_base(component=comp, bodies=self._bodies([1.0, 1.0, 1.0]))

    def test_constructor_rejects_non_negative_world_node(self):
        """``world_node >= 0`` raises ``ValueError`` at construction."""
        with self.assertRaisesRegex(ValueError, "negative integer"):
            topology.TopologyHeaviestBodyBaseSelector(world_node=0)

    def test_constructor_rejects_non_int_world_node(self):
        """Non-integer ``world_node`` raises ``TypeError`` at construction."""
        with self.assertRaisesRegex(TypeError, "integer"):
            topology.TopologyHeaviestBodyBaseSelector(world_node="-1")


###
# Minimum-depth spanning-tree generator
###


class TestTopologySpanningTreeGenerator(unittest.TestCase):
    """Tests for :class:`TopologyMinimumDepthSpanningTreeGenerator`.

    Covers the priority cascade, per-root enumeration, brute-force fallback,
    direct root override, max-candidate truncation, the orphan special case,
    the optional balanced-tree ordering and Featherstone numbering invariants
    on the produced :class:`TopologySpanningTree` instances.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.savefig = False  # Set to True for saving plotting output
        self.plotfig = False  # Set to True for render plotting output
        self.output_path = test_context.output_path / "test_topology" / "trees"

        # Create output directory if saving figures
        if self.savefig:
            self.output_path.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        msg.reset_log_level()

    ###
    # Serial chain
    ###

    def test_serial_chain_yields_unique_min_depth_tree(self):
        """A 5-body serial chain rooted at `0` admits exactly one min-depth tree."""
        nodes = [0, 1, 2, 3, 4]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),  # base
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
            (JointDoFType.REVOLUTE.value, 3, (2, 3)),
            (JointDoFType.REVOLUTE.value, 4, (3, 4)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")

        self.assertEqual(len(trees), 1)
        t = trees[0]
        self.assertEqual(t.root, 0)
        self.assertEqual(t.depth, 4)
        self.assertEqual(t.num_bodies, 5)
        self.assertEqual(t.num_tree_arcs, 5)
        self.assertEqual(t.num_tree_chords, 0)
        self.assertEqual(t.parents, [-1, 0, 1, 2, 3])
        self.assertEqual(t.arcs[0], 0)  # base edge joint index
        self.assertEqual(t.chords, [])
        _assert_featherstone_invariants(self, t)

    ###
    # 4-bar closed loop
    ###

    def test_four_bar_loop_yields_two_min_depth_trees(self):
        """A 4-body cycle rooted at `0` admits exactly two min-depth trees of depth 2."""
        nodes = [0, 1, 2, 3]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),  # base
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
            (JointDoFType.REVOLUTE.value, 3, (2, 3)),
            (JointDoFType.REVOLUTE.value, 4, (3, 0)),  # closes the loop
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")

        self.assertEqual(len(trees), 2)
        for t in trees:
            self.assertEqual(t.root, 0)
            self.assertEqual(t.depth, 2)
            self.assertEqual(t.num_bodies, 4)
            self.assertEqual(t.num_tree_arcs, 4)
            self.assertEqual(t.num_tree_chords, 1)
            self.assertEqual(t.arcs[0], 0)  # base edge
            _assert_featherstone_invariants(self, t)

        # The two trees must differ on which loop edge is used as a chord.
        chord_sets = {tuple(t.chords) for t in trees}
        self.assertEqual(len(chord_sets), 2)

    ###
    # Multi-grounding (no auto-base)
    ###

    def test_multiple_grounding_nodes_used_as_roots(self):
        """When the component has no base but multiple grounding nodes, each becomes a root."""
        nodes = [0, 1, 2]
        edges = [
            # Two REVOLUTE grounding edges to different bodies; no auto-base assignment.
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (-1, 2)),
            # Internal connectivity
            (JointDoFType.REVOLUTE.value, 2, (0, 1)),
            (JointDoFType.REVOLUTE.value, 3, (1, 2)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]
        self.assertIsNone(comp.base_edge)
        self.assertEqual(sorted(int(n) for n in comp.ground_nodes), [0, 2])

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")

        self.assertGreater(len(trees), 0)
        ground_indices = [int(n) for n in comp.ground_nodes]
        for t in trees:
            self.assertIn(t.root, ground_indices)
            _assert_featherstone_invariants(self, t)

    ###
    # Isolated island - degree heuristic with a unique max
    ###

    def test_isolated_island_uses_unique_max_degree_root(self):
        """Without base or grounding nodes, the unique max-degree body is selected."""
        # Star graph: body 1 is connected to all other bodies (degree 3).
        nodes = [0, 1, 2, 3]
        edges = [
            (JointDoFType.REVOLUTE.value, 0, (1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (1, 2)),
            (JointDoFType.REVOLUTE.value, 2, (1, 3)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]
        self.assertIsNone(comp.base_edge)
        self.assertEqual(comp.ground_edges, [])

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")

        self.assertGreater(len(trees), 0)
        for t in trees:
            self.assertEqual(t.root, 1)
            _assert_featherstone_invariants(self, t)

    ###
    # Isolated island - degree tie -> brute-force
    ###

    def test_isolated_island_degree_tie_brute_forces_all_nodes(self):
        """A symmetric island with no base/grounding triggers brute-force over all bodies."""
        # Triangle: every node has degree 2 -> degree tie.
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.REVOLUTE.value, 0, (0, 1)),
            (JointDoFType.REVOLUTE.value, 1, (1, 2)),
            (JointDoFType.REVOLUTE.value, 2, (2, 0)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")

        self.assertGreater(len(trees), 0)
        roots_seen = {t.root for t in trees}
        self.assertEqual(roots_seen, set(nodes))
        for t in trees:
            _assert_featherstone_invariants(self, t)

    ###
    # Orphans
    ###

    def test_orphan_with_base_edge_yields_trivial_tree(self):
        """A single-body component with a base edge produces a trivial 1-arc tree."""
        nodes = [0]
        edges = [(JointDoFType.FREE.value, 7, (-1, 0))]
        G = _build_graph(nodes, edges)
        comp = G.components[0]

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")

        self.assertEqual(len(trees), 1)
        t = trees[0]
        self.assertEqual(t.root, 0)
        self.assertEqual(t.num_bodies, 1)
        self.assertEqual(t.depth, 0)
        self.assertEqual(t.arcs, [7])
        self.assertEqual(t.chords, [])
        self.assertEqual(t.parents, [-1])
        self.assertEqual(t.children, [[]])
        self.assertEqual(t.subtree, [[0]])
        self.assertEqual(t.support, [[]])

    def test_isolated_orphan_yields_trivial_tree_no_arcs(self):
        """A single-body, edgeless component still produces one (empty) trivial tree."""
        nodes = [0]
        G = _build_graph(nodes, edges=[])
        comp = G.components[0]

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")

        self.assertEqual(len(trees), 1)
        t = trees[0]
        self.assertEqual(t.root, 0)
        self.assertEqual(t.num_bodies, 1)
        self.assertEqual(t.arcs, [])
        self.assertEqual(t.chords, [])
        self.assertEqual(t.parents, [-1])

    ###
    # override_priorities
    ###

    def test_override_priorities_brute_forces_when_base_exists(self):
        """`override_priorities=True` must ignore the base node and brute-force every body."""
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),  # base assigned to 0
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
            (JointDoFType.REVOLUTE.value, 3, (2, 0)),  # closes the loop
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]
        self.assertEqual(int(comp.base_node), 0)

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs", override_priorities=True)

        roots_seen = {t.root for t in trees}
        self.assertEqual(roots_seen, set(nodes))
        for t in trees:
            _assert_featherstone_invariants(self, t)

    ###
    # roots argument override (and component mutation)
    ###

    def test_explicit_root_override_uses_only_supplied_root(self):
        """A single explicit `roots=[r]` must restrict enumeration to that root."""
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.REVOLUTE.value, 0, (0, 1)),
            (JointDoFType.REVOLUTE.value, 1, (1, 2)),
            (JointDoFType.REVOLUTE.value, 2, (2, 0)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]
        self.assertIsNone(comp.base_edge)

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs", roots=[2])

        self.assertGreater(len(trees), 0)
        for t in trees:
            self.assertEqual(t.root, 2)
            _assert_featherstone_invariants(self, t)

    def test_explicit_root_override_stamps_base_when_grounding_exists(self):
        """A single explicit root with a matching grounding edge auto-promotes that edge to base."""
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.REVOLUTE.value, 0, (-1, 2)),  # only grounding edge -> auto-promoted
        ]
        # The single grounding edge is auto-promoted by the parser to base, so we
        # build a graph with two grounding edges instead so that no auto-promotion
        # occurs and `_select_root_candidates` performs the stamping.
        edges = [
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (-1, 2)),
            (JointDoFType.REVOLUTE.value, 2, (0, 1)),
            (JointDoFType.REVOLUTE.value, 3, (1, 2)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]
        self.assertIsNone(comp.base_edge)
        self.assertEqual(sorted(int(n) for n in comp.ground_nodes), [0, 2])

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs", roots=[2])

        self.assertGreater(len(trees), 0)
        # The grounding edge involving body 2 should now be the component's base edge.
        self.assertEqual(int(comp.base_node), 2)
        self.assertIsNotNone(comp.base_edge)
        self.assertEqual(comp.base_edge.joint_index, 1)  # joint index 1 is the (-1, 2) edge
        # ...and removed from the grounding lists.
        self.assertNotIn(2, [int(n) for n in comp.ground_nodes])
        self.assertTrue(all(e.joint_index != 1 for e in comp.ground_edges))
        for t in trees:
            self.assertEqual(t.root, 2)

    ###
    # max_candidates cap
    ###

    def test_max_candidates_truncates_prefix(self):
        """`max_candidates=N` must return the first `N` candidates of the unbounded enumeration."""
        # A graph with multiple min-depth candidates from the same root.
        nodes = [0, 1, 2, 3]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (0, 2)),
            (JointDoFType.REVOLUTE.value, 3, (1, 3)),
            (JointDoFType.REVOLUTE.value, 4, (2, 3)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        unbounded = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")
        capped = gen.generate_spanning_trees(component=comp, traversal_mode="bfs", max_candidates=1)

        self.assertGreaterEqual(len(unbounded), 1)
        self.assertEqual(len(capped), 1)
        # Prefix property: the first N candidates are identical.
        self.assertEqual(capped[0].arcs, unbounded[0].arcs)
        self.assertEqual(capped[0].chords, unbounded[0].chords)

    ###
    # BFS vs DFS traversal
    ###

    def test_bfs_dfs_produce_same_edge_sets(self):
        """BFS and DFS only change the per-tree ordering, never the set of edges chosen."""
        nodes = [0, 1, 2, 3]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (0, 2)),
            (JointDoFType.REVOLUTE.value, 3, (1, 3)),
            (JointDoFType.REVOLUTE.value, 4, (2, 3)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        bfs_trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")
        dfs_trees = gen.generate_spanning_trees(component=comp, traversal_mode="dfs")

        self.assertEqual(len(bfs_trees), len(dfs_trees))

        def _edge_signatures(trees):
            return {(frozenset(t.arcs), frozenset(t.chords)) for t in trees}

        self.assertEqual(_edge_signatures(bfs_trees), _edge_signatures(dfs_trees))
        for t in bfs_trees:
            self.assertEqual(t.traversal, "bfs")
            _assert_featherstone_invariants(self, t)
        for t in dfs_trees:
            self.assertEqual(t.traversal, "dfs")
            _assert_featherstone_invariants(self, t)

    ###
    # prioritize_balanced ordering
    ###

    def test_prioritize_balanced_ranks_balanced_trees_first(self):
        """When requested, balanced trees (lower imbalance score) must come first."""
        # Same diamond graph as the cap test; root=0 admits multiple trees with
        # different balance characteristics.
        nodes = [0, 1, 2, 3]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (0, 2)),
            (JointDoFType.REVOLUTE.value, 3, (1, 3)),
            (JointDoFType.REVOLUTE.value, 4, (2, 3)),
        ]
        G = _build_graph(nodes, edges)
        comp = G.components[0]

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        plain = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")
        balanced = gen.generate_spanning_trees(component=comp, traversal_mode="bfs", prioritize_balanced=True)

        # Same set of trees, just possibly re-ordered.
        self.assertEqual(len(plain), len(balanced))
        # The score of the first balanced candidate must be <= every other score.
        scores = [sum(len(c) * len(c) for c in t.children) for t in balanced]
        self.assertEqual(scores[0], min(scores))


###
# Minimum-depth spanning-tree selector
###


class TestTopologySpanningTreeSelector(unittest.TestCase):
    """Tests for :class:`TopologyMinimumDepthSpanningTreeSelector` in isolation.

    Covers depth-based selection, balance-based tie-breaking, the orphan
    short-circuit, and validation of the ``candidates`` argument.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    ###
    # Basic depth-based selection
    ###

    def test_picks_min_depth_when_unambiguous(self):
        """The candidate with the smallest ``depth`` wins outright."""
        candidates = [_make_tree(depth=5), _make_tree(depth=2), _make_tree(depth=7)]
        sel = topology.TopologyMinimumDepthSpanningTreeSelector()
        chosen = sel.select_spanning_tree(candidates=candidates)
        self.assertIs(chosen, candidates[1])

    ###
    # Tie-breaking
    ###

    def test_breaks_depth_ties_by_balance(self):
        """On a depth tie, the more-balanced candidate (lower imbalance score) wins."""
        # Both trees have 5 nodes and depth=2, but different per-node child counts.
        # Tree A: balanced (root has 2 children, one child has 2 children)
        #   children = [[1, 2], [3, 4], [], [], []]; score = 4 + 4 = 8.
        # Tree B: spine-then-fork
        #   children = [[1], [2, 3, 4], [], [], []]; score = 1 + 9 = 10.
        a = _make_tree(depth=2, num_bodies=5, children=[[1, 2], [3, 4], [], [], []])
        b = _make_tree(depth=2, num_bodies=5, children=[[1], [2, 3, 4], [], [], []])

        sel = topology.TopologyMinimumDepthSpanningTreeSelector(prioritize_balanced=True)
        chosen_a_first = sel.select_spanning_tree(candidates=[a, b])
        chosen_b_first = sel.select_spanning_tree(candidates=[b, a])

        # Order-independent: the more-balanced tree (A) wins regardless of input order.
        self.assertIs(chosen_a_first, a)
        self.assertIs(chosen_b_first, a)

    def test_disable_balance_uses_input_order_for_depth_ties(self):
        """With ``prioritize_balanced=False``, depth ties go to the first listed candidate."""
        a = _make_tree(depth=2, num_bodies=5, children=[[1, 2], [3, 4], [], [], []])
        b = _make_tree(depth=2, num_bodies=5, children=[[1], [2, 3, 4], [], [], []])

        sel = topology.TopologyMinimumDepthSpanningTreeSelector(prioritize_balanced=False)
        chosen = sel.select_spanning_tree(candidates=[b, a])
        self.assertIs(chosen, b)

    def test_breaks_full_ties_by_input_order(self):
        """When candidates tie on depth and balance, the first listed candidate wins."""
        a = _make_tree(depth=3, num_bodies=4, children=[[1], [2], [3], []])
        b = _make_tree(depth=3, num_bodies=4, children=[[1], [2], [3], []])

        sel = topology.TopologyMinimumDepthSpanningTreeSelector(prioritize_balanced=True)
        chosen = sel.select_spanning_tree(candidates=[a, b])
        self.assertIs(chosen, a)

    ###
    # Orphan special case
    ###

    def test_returns_orphan_trivial_tree(self):
        """A single-body candidate is returned as-is regardless of ordering options."""
        orphan = topology.TopologySpanningTree(
            traversal="bfs",
            depth=0,
            num_bodies=1,
            num_joints=1,
            num_tree_arcs=1,
            num_tree_chords=0,
            children=[[]],
            parents=[-1],
            arcs=[7],
            chords=[],
            predecessors=[-1],
            successors=[0],
            support=[[]],
            subtree=[[0]],
        )
        sel = topology.TopologyMinimumDepthSpanningTreeSelector()
        chosen = sel.select_spanning_tree(candidates=[orphan])
        self.assertIs(chosen, orphan)

    ###
    # Effective-depth-from-base selection
    ###

    def test_picks_min_eccentricity_from_base_when_candidates_tie_on_depth(self):
        """On a ``t.depth`` tie, the selector picks the candidate whose tree
        is shallowest when re-rooted at the component's assigned base node.

        Regression for an over-eager ``min(candidates, key=t.depth)`` that
        ignored the component's assigned base. On a 5-cycle component
        ``[5, 6, 7, 10, 11]`` with ``base_node=7`` and ``override_priorities=True``,
        the generator emits one minimum-depth candidate per body root — all
        with ``t.depth == 2`` even though their tree structure (a 5-vertex
        path with a chord) places body 7 at very different positions.
        Only the candidate rooted at body 7 places the base at the geometric
        center of that path; every other candidate places it at an off-center
        position whose eccentricity from body 7 is 3 or 4. Without this fix
        the selector would pick the first depth=2 candidate (root=5), giving
        a tree whose articulation chain from base=7 has length 4.
        """
        nodes = [5, 6, 7, 10, 11]
        edges = [
            (JointDoFType.REVOLUTE.value, 8, (5, 6)),
            (JointDoFType.REVOLUTE.value, 9, (6, 7)),
            (JointDoFType.REVOLUTE.value, 10, (7, 10)),
            (JointDoFType.REVOLUTE.value, 11, (10, 11)),
            (JointDoFType.REVOLUTE.value, 12, (11, 5)),
            (JointDoFType.FREE.value, 13, (-1, 7)),
        ]
        G = topology.TopologyGraph(nodes, edges, autoparse=False)
        components = G.parse_components()
        self.assertEqual(len(components), 1)
        component = components[0]
        self.assertEqual(int(component.base_node), 7)

        generator = topology.TopologyMinimumDepthSpanningTreeGenerator()
        candidates = generator.generate_spanning_trees(component, override_priorities=True)
        self.assertEqual(len(candidates), 5)
        self.assertTrue(all(c.depth == 2 for c in candidates))

        sel = topology.TopologyMinimumDepthSpanningTreeSelector()
        chosen = sel.select_spanning_tree(candidates=candidates)
        self.assertEqual(chosen.root, 7)

    def test_falls_back_to_stored_depth_when_no_base_assigned(self):
        """Without an assigned ``base_node`` on the source component, the
        selector preserves its legacy ``min(t.depth)`` behaviour."""
        # Synthetic candidates without a `component` link bypass the new
        # base-aware path entirely and exercise the depth-only fallback.
        candidates = [_make_tree(depth=4), _make_tree(depth=2), _make_tree(depth=3)]
        sel = topology.TopologyMinimumDepthSpanningTreeSelector(prioritize_balanced=False)
        chosen = sel.select_spanning_tree(candidates=candidates)
        self.assertIs(chosen, candidates[1])

    ###
    # Failure modes
    ###

    def test_empty_candidates_raises(self):
        """An empty ``candidates`` list raises ``ValueError``."""
        sel = topology.TopologyMinimumDepthSpanningTreeSelector()
        with self.assertRaisesRegex(ValueError, "at least one"):
            sel.select_spanning_tree(candidates=[])

    def test_malformed_candidate_raises_when_balanced(self):
        """An island candidate with ``children=None`` raises when balance ordering is requested."""
        # Two depth-tied candidates so the selector reaches the balance score.
        good = _make_tree(depth=2)
        bad = topology.TopologySpanningTree(
            traversal="bfs",
            depth=2,
            num_bodies=4,
            num_joints=4,
            num_tree_arcs=4,
            num_tree_chords=0,
            children=None,
        )
        sel = topology.TopologyMinimumDepthSpanningTreeSelector(prioritize_balanced=True)
        with self.assertRaisesRegex(ValueError, "children=None"):
            sel.select_spanning_tree(candidates=[good, bad])

    def test_malformed_candidate_ok_when_balance_disabled(self):
        """An island candidate with ``children=None`` is acceptable when balance ordering is disabled."""
        good = _make_tree(depth=1)
        bad = topology.TopologySpanningTree(
            traversal="bfs",
            depth=2,
            num_bodies=4,
            num_joints=4,
            num_tree_arcs=4,
            num_tree_chords=0,
            children=None,
        )
        sel = topology.TopologyMinimumDepthSpanningTreeSelector(prioritize_balanced=False)
        chosen = sel.select_spanning_tree(candidates=[bad, good])
        # The depth-only path picks `good` (depth=1) over `bad` (depth=2).
        self.assertIs(chosen, good)


###
# TopologyIndexReassignment
###


class TestTopologyIndexReassignment(unittest.TestCase):
    """Tests for :class:`TopologyIndexReassignment.reassign_indices`.

    The reassignment back-end takes a list of selected spanning trees and
    rewrites the global body / joint index space so that:

    - Bodies and joints belonging to the same tree are grouped contiguously.
    - Larger trees come first (descending size).
    - Within each tree, bodies follow traversal order (root at the smallest
      new index) and joints follow Featherstone's regular numbering
      (arcs first in body order, then chords sorted by predecessor).
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = True  # Set to True for detailed output
        self.savefig = True  # Set to True for saving plotting output
        self.plotfig = False  # Set to True for render plotting output
        self.output_path = test_context.output_path / "test_topology" / "index_reassignment"

        # Create output directory if saving figures
        if self.savefig:
            self.output_path.mkdir(parents=True, exist_ok=True)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def _run(self, *, nodes, edges, bases=None, inplace=False, **graph_kwargs):
        """Build a graph, run the pipeline, and return ``(graph, body_remap, joint_remap)``."""
        G = topology.TopologyGraph(nodes, edges, autoparse=False, reassign_indices_inplace=inplace, **graph_kwargs)
        G.parse_components()
        G.select_component_bases(bases=bases)
        G.generate_spanning_trees()
        G.select_spanning_trees()
        body_remap, joint_remap = G.compute_index_reassignment(reassign_inplace=inplace)
        return G, body_remap, joint_remap

    def test_already_regular_chain_yields_identity_remaps(self):
        """A chain in regular numbering must produce identity remaps."""
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
        ]
        _G, body_remap, joint_remap = self._run(nodes=nodes, edges=edges)
        self.assertEqual(body_remap, [0, 1, 2])
        self.assertEqual(joint_remap, [0, 1, 2])

    def test_reverse_numbered_chain_is_remapped_forward(self):
        """A chain whose body indices are reversed gets remapped to forward order.

        Edges: 0-1, 1-2. Base on body 2 (FREE grounding) → root is body 2 → after
        reassignment, body 2 → 0, body 1 → 1, body 0 → 2.
        """
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 2)),
            (JointDoFType.REVOLUTE.value, 1, (1, 2)),
            (JointDoFType.REVOLUTE.value, 2, (0, 1)),
        ]
        G, body_remap, joint_remap = self._run(nodes=nodes, edges=edges)
        msg.info("body_remap: %s", body_remap)
        msg.info("joint_remap: %s", joint_remap)
        msg.info("tree_remapped: %s", G.trees_remapped[0])
        G.render_graph(
            path=self.output_path / "test_reverse_numbered_chain_is_remapped_forward_graph.pdf", show=self.plotfig
        )
        G.render_spanning_trees(
            path=self.output_path / "test_reverse_numbered_chain_is_remapped_forward_trees.pdf", show=self.plotfig
        )
        G._graph_visualizer.render_component_spanning_tree(
            G.components[0],
            G.trees_remapped[0],
            path=self.output_path / "test_reverse_numbered_chain_is_remapped_forward_tree_remapped.pdf",
            show=self.plotfig,
        )

        # Body 2 was the root → new index 0; body 1 → 1; body 0 → 2.
        self.assertEqual(body_remap[2], 0)
        self.assertEqual(body_remap[1], 1)
        self.assertEqual(body_remap[0], 2)
        # Joint 0 (base on body 2) → new index 0; remaining arcs follow body order.
        self.assertEqual(joint_remap[0], 0)
        # All trees satisfy Featherstone invariants after the reassignment.
        for tree in G.trees:
            _assert_featherstone_invariants(self, tree)

    def test_two_components_are_grouped_with_larger_first(self):
        """The larger component must occupy the lowest indices."""
        nodes = [0, 1, 2, 3, 4]
        edges = [
            # Larger chain 0-1-2 with auto-base on 0.
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
            # Smaller pair 3-4 with auto-base on 3.
            (JointDoFType.FREE.value, 3, (-1, 3)),
            (JointDoFType.REVOLUTE.value, 4, (3, 4)),
        ]
        _G, body_remap, joint_remap = self._run(nodes=nodes, edges=edges)
        # Larger component (nodes [0,1,2]) gets new indices [0,1,2].
        self.assertEqual({body_remap[i] for i in (0, 1, 2)}, {0, 1, 2})
        # Smaller component (nodes [3,4]) gets new indices [3,4].
        self.assertEqual({body_remap[i] for i in (3, 4)}, {3, 4})
        # Joint 0 (base of larger component) becomes new joint 0.
        self.assertEqual(joint_remap[0], 0)
        # Joint 3 (base of smaller component) lands AFTER all of the larger
        # component's joints (3 arcs, 0 chords).
        self.assertEqual(joint_remap[3], 3)

    def test_chord_is_placed_after_arcs_within_a_tree(self):
        """A 4-bar mechanism's chord lands after the arcs of its component."""
        nodes = [0, 1, 2, 3]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
            (JointDoFType.REVOLUTE.value, 3, (2, 3)),
            # Closing chord:
            (JointDoFType.REVOLUTE.value, 4, (3, 0)),
        ]
        G, body_remap, joint_remap = self._run(nodes=nodes, edges=edges)
        # Single component → all arcs at [0, num_bodies), chord at num_bodies.
        self.assertEqual(len(G.trees), 1)
        tree = G.trees[0]
        nb = tree.num_bodies
        new_arc_positions = {joint_remap[a] for a in tree.arcs if a >= 0}
        new_chord_positions = {joint_remap[c] for c in tree.chords}
        self.assertEqual(new_arc_positions, set(range(nb)))
        # Single chord goes at slot ``nb``.
        self.assertEqual(new_chord_positions, {nb})
        # And ``body_remap`` is a permutation over [0, nb) (no body left out).
        self.assertEqual({body_remap[i] for i in range(nb)}, set(range(nb)))

    def test_inplace_mode_mutates_trees(self):
        """``inplace=True`` propagates the remap into the live spanning trees."""
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 2)),
            (JointDoFType.REVOLUTE.value, 1, (1, 2)),
            (JointDoFType.REVOLUTE.value, 2, (0, 1)),
        ]
        G, body_remap, joint_remap = self._run(nodes=nodes, edges=edges, inplace=True)
        tree = G.trees[0]
        # The root must now be the post-remap value (body 2's new index).
        self.assertEqual(tree.root, body_remap[2])
        # And ``arcs`` must carry the post-remap joint indices.
        self.assertEqual(tree.arcs[0], joint_remap[0])
        # The component reference is dropped because the original component still
        # describes the un-remapped graph.
        self.assertIsNone(tree.component)
        # ``trees_remapped`` returns the live list under inplace mode.
        self.assertIs(G.trees_remapped, G.trees)

    def test_non_inplace_mode_leaves_trees_alone(self):
        """``inplace=False`` keeps the trees unmodified and offers a remapped copy."""
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 2)),
            (JointDoFType.REVOLUTE.value, 1, (1, 2)),
            (JointDoFType.REVOLUTE.value, 2, (0, 1)),
        ]
        G, body_remap, joint_remap = self._run(nodes=nodes, edges=edges, inplace=False)
        tree = G.trees[0]
        # The original (un-remapped) root is preserved.
        self.assertEqual(tree.root, 2)
        # ``trees_remapped`` materializes per-tree copies with the remap applied.
        remapped = G.trees_remapped
        self.assertIsNot(remapped, G.trees)
        self.assertEqual(remapped[0].root, body_remap[2])
        self.assertEqual(remapped[0].arcs[0], joint_remap[0])

    def test_synthetic_base_edges_get_nj_plus_k_indices_then_remap(self):
        """Isolated components receive synthetic base edges with ``NJ + k`` joint indices.

        The ``NJ + k`` indices then flow into the reassignment and end up at
        the front of their respective per-tree joint segments.
        """
        nodes = [0, 1, 2, 3]
        edges = [
            # Connected chain 0-1-2 with FREE base.
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
            # Isolated orphan 3 (no grounding) → synthetic base.
        ]
        # Orphan 3 needs an explicit base hint since the orphan-fallback path
        # never invokes the heaviest-body selector (which needs `bodies=`).
        G, _body_remap, joint_remap = self._run(nodes=nodes, edges=edges, bases=[3])
        # One synthetic base edge was minted.
        self.assertIsNotNone(G.new_base_edges)
        self.assertEqual(len(G.new_base_edges), 1)
        synthetic = G.new_base_edges[0]
        # NJ_orig was 3 → synthetic edge has joint_index = 3 + 0 = 3.
        self.assertEqual(synthetic.joint_index, 3)
        self.assertEqual(synthetic.joint_type, JointDoFType.FREE.value)
        # After reassignment, the synthetic edge ends up at the joint slot
        # immediately following the larger (chain) component (slots 0..2).
        self.assertEqual(joint_remap[synthetic.joint_index], 3)

    def test_multiple_synthetic_base_edges_get_consecutive_indices(self):
        """Each synthetic base edge gets a unique ``NJ + k`` index."""
        nodes = [0, 1, 2, 3, 4]
        edges = [
            (JointDoFType.REVOLUTE.value, 0, (0, 1)),  # isolated island [0, 1]
            (JointDoFType.REVOLUTE.value, 1, (2, 3)),  # isolated island [2, 3]
            # body 4 is an isolated orphan
        ]
        G, _body_remap, _joint_remap = self._run(nodes=nodes, edges=edges, bases=[1, 3, 4])
        # Three components needed a synthetic base edge.
        self.assertIsNotNone(G.new_base_edges)
        self.assertEqual(len(G.new_base_edges), 3)
        # NJ_orig was 2 → synthetic edges get joint_index 2, 3, 4 (in commit order).
        self.assertEqual(sorted(e.joint_index for e in G.new_base_edges), [2, 3, 4])
        # All three edges are FREE joints to the world.
        for e in G.new_base_edges:
            self.assertEqual(e.joint_type, JointDoFType.FREE.value)
            self.assertIn(-1, e.nodes)

    def test_empty_tree_list_returns_none_remaps(self):
        """An empty ``trees`` list short-circuits to ``(None, None)``."""
        reassigner = topology.graph.TopologyIndexReassignment()
        body_remap, joint_remap = reassigner.reassign_indices(trees=[], inplace=False)
        self.assertIsNone(body_remap)
        self.assertIsNone(joint_remap)


###
# TopologyGraph
###


class TestTopologyGraph(unittest.TestCase):
    """
    End-to-end coverage for :class:`TopologyGraph` and the component
    parsing pipeline (graph → components → grounding/base assignment).

    Test layout:
    - ``test_0*_*`` — constructor input-validation checks.
    - ``test_1*_graph_component_parsing_*`` — component-parsing unit tests on
      synthetic node/edge configurations.
    - ``test_2*_discover_topology_of_*`` — end-to-end topology discovery against
      imported asset models (USD).
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = True  # Set to True for detailed output
        self.savefig = True  # Set to True for saving plotting output
        self.plotfig = False  # Set to True for render plotting output
        self.output_path = test_context.output_path / "test_topology" / "graph"

        # Create output directory if saving figures
        if self.savefig:
            self.output_path.mkdir(parents=True, exist_ok=True)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Constructor validation
    ###

    def test_01_non_int_node_raises_type_error(self):
        with self.assertRaises(TypeError):
            topology.TopologyGraph(nodes=[0, "1"], autoparse=False)

    def test_02_negative_node_raises_value_error(self):
        with self.assertRaises(ValueError):
            topology.TopologyGraph(nodes=[0, -1], autoparse=False)

    def test_03_malformed_edge_tuple_raises_type_error(self):
        with self.assertRaises(TypeError):
            topology.TopologyGraph(nodes=[0, 1], edges=[(0, 0)], autoparse=False)

    def test_04_non_int_world_raises_type_error(self):
        with self.assertRaises(TypeError):
            topology.TopologyGraph(nodes=[0, 1], world_node="-1", autoparse=False)

    def test_05_non_negative_world_raises_value_error(self):
        with self.assertRaises(ValueError):
            topology.TopologyGraph(nodes=[0, 1], world_node=0, autoparse=False)

    def test_06_world_in_nodes_raises_value_error(self):
        with self.assertRaises(ValueError):
            topology.TopologyGraph(nodes=[0, -1], world_node=-1, autoparse=False)

    def test_07_bool_world_node_raises_type_error(self):
        """``world_node`` must be an integer; booleans (``int`` subclass) are rejected explicitly."""
        with self.assertRaises(TypeError):
            topology.TopologyGraph(nodes=[0, 1], world_node=True, autoparse=False)
        with self.assertRaises(TypeError):
            topology.TopologyGraph(nodes=[0, 1], world_node=False, autoparse=False)

    def test_08_duplicate_node_indices_raise_value_error(self):
        """``nodes`` must not contain duplicate body indices."""
        with self.assertRaisesRegex(ValueError, "duplicate"):
            topology.TopologyGraph(nodes=[0, 1, 0], autoparse=False)
        # Mixing the int and GraphNode forms with the same index is also a duplicate.
        with self.assertRaisesRegex(ValueError, "duplicate"):
            topology.TopologyGraph(
                nodes=[topology.GraphNode(index=2, name="a"), 2],
                autoparse=False,
            )

    def test_09_edges_none_yields_empty_edge_list(self):
        """``edges=None`` must be treated as an empty graph (isolated body nodes only)."""
        G = topology.TopologyGraph(nodes=[0, 1], edges=None, autoparse=False)
        self.assertEqual(G.edges, [])

    def test_09a_edge_endpoint_unknown_node_raises_value_error(self):
        """Regression: an edge endpoint not present in ``nodes`` must raise ``ValueError``.

        Previously :class:`TopologyComponentParser` silently dropped any
        edge whose non-world endpoint was missing from the node set, so a
        typo or model-import inconsistency could produce orphan
        components and inflated joint counts without any error surface.
        """
        with self.assertRaisesRegex(ValueError, "not contained in `nodes`"):
            topology.TopologyGraph(
                nodes=[0, 1],
                edges=[(JointDoFType.REVOLUTE.value, 0, (0, 99))],
                autoparse=False,
            )
        # The world endpoint must remain valid even when it is absent
        # from `nodes` — only body endpoints are checked against it.
        topology.TopologyGraph(
            nodes=[0, 1],
            edges=[(JointDoFType.REVOLUTE.value, 0, (-1, 0))],
            autoparse=False,
        )

    def test_09b_edge_joint_index_polarity_swap_raises_value_error(self):
        """Regression: the same ``joint_index`` cannot index two distinct edges.

        ``(REVOLUTE, 0, (0, 1))`` and ``(REVOLUTE, 0, (1, 0))`` represent
        the same joint with swapped polarity but were stored as two
        :class:`GraphEdge` records. Downstream consumers (degree
        counting, arc/chord classification, oriented-chord polarity)
        all assume each ``joint_index`` uniquely identifies one joint
        record, so these inputs must be rejected up front.
        """
        with self.assertRaisesRegex(ValueError, "joint_index"):
            topology.TopologyGraph(
                nodes=[0, 1],
                edges=[
                    (JointDoFType.REVOLUTE.value, 0, (0, 1)),
                    (JointDoFType.REVOLUTE.value, 0, (1, 0)),
                ],
                autoparse=False,
            )

    def test_09c_edge_joint_index_collision_different_type_raises_value_error(self):
        """Regression: two edges with the same ``joint_index`` but different ``joint_type`` must fail."""
        with self.assertRaisesRegex(ValueError, "joint_type"):
            topology.TopologyGraph(
                nodes=[0, 1],
                edges=[
                    (JointDoFType.REVOLUTE.value, 0, (0, 1)),
                    (JointDoFType.PRISMATIC.value, 0, (0, 1)),
                ],
                autoparse=False,
            )

    def test_09d_edge_joint_index_collision_different_bodies_raises_value_error(self):
        """Regression: two edges sharing a ``joint_index`` but linking different body pairs must fail."""
        with self.assertRaisesRegex(ValueError, "joint_index"):
            topology.TopologyGraph(
                nodes=[0, 1, 2],
                edges=[
                    (JointDoFType.REVOLUTE.value, 0, (0, 1)),
                    (JointDoFType.REVOLUTE.value, 0, (1, 2)),
                ],
                autoparse=False,
            )

    def test_09e_exact_duplicate_edges_are_accepted(self):
        """Regression: byte-identical duplicate edges remain a benign no-op.

        The constructor only enforces uniqueness when the
        ``(joint_type, nodes)`` pair differs across edges sharing a
        ``joint_index``; exact duplicates are collapsed by the parser
        via set deduplication and must not regress to a hard error.
        """
        G = topology.TopologyGraph(
            nodes=[0, 1],
            edges=[
                (JointDoFType.REVOLUTE.value, 0, (0, 1)),
                (JointDoFType.REVOLUTE.value, 0, (0, 1)),
            ],
            autoparse=False,
        )
        # `edges` preserves the raw input; `parse_components` collapses duplicates.
        self.assertEqual(len(G.edges), 2)

    ###
    # Component parsing
    ###

    def test_10_graph_component_parsing_test_graph(self):
        """Reference-graph parsing exercises every component classification path.

        ``_make_test_graph`` is a hand-rolled multi-component fixture covering:
        a connected island (cycle with a single grounding edge auto-promoted to
        base), an isolated island (cycle with no grounding edge), connected
        orphans (single body with a single grounding edge auto-promoted to
        base), and isolated orphans (single body with no incident edges).
        """
        nodes, edges = _make_test_graph()
        G = topology.TopologyGraph(nodes, edges, autoparse=False)
        C = G.parse_components()

        # ``_make_test_graph`` produces:
        #   - 1 connected island of 5 nodes: {0, 1, 2, 3, 4}        (cycle with world-anchor)
        #   - 1 isolated island of 5 nodes:  {5, 6, 7, 10, 11}      (cycle, no world-anchor)
        #   - 2 connected orphans:           {8}, {9}               (single body, single world-edge)
        #   - 4 isolated orphans:            {12}, {13}, {14}, {15} (no incident edges at all)
        self.assertEqual(len(C), 8)

        def find_component(idx: int) -> topology.TopologyComponent:
            """Locate the unique component containing body ``idx``."""
            return next(c for c in C if idx in [int(n) for n in (c.nodes or [])])

        # 1) Connected island {0, 1, 2, 3, 4}: the lone grounding edge (jid=0)
        #    is auto-promoted by the parser to base, leaving the grounding lists empty.
        c = find_component(0)
        self.assertEqual(sorted(int(n) for n in c.nodes), [0, 1, 2, 3, 4])
        self.assertTrue(c.is_island)
        self.assertTrue(c.is_connected)
        self.assertEqual(sorted(e.joint_index for e in c.edges), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(int(c.base_node), 0)
        self.assertIsNotNone(c.base_edge)
        self.assertEqual(c.base_edge.joint_index, 0)
        self.assertEqual(c.base_edge.nodes, (-1, 0))
        self.assertEqual(c.ground_nodes, [])
        self.assertEqual(c.ground_edges, [])

        # 2) Isolated island {5, 6, 7, 10, 11}: cycle without a world-anchor — the
        #    parser leaves the base unassigned and the grounding lists empty.
        c = find_component(5)
        self.assertEqual(sorted(int(n) for n in c.nodes), [5, 6, 7, 10, 11])
        self.assertTrue(c.is_island)
        self.assertFalse(c.is_connected)
        self.assertEqual(sorted(e.joint_index for e in c.edges), [8, 9, 10, 11, 12])
        self.assertIsNone(c.base_node)
        self.assertIsNone(c.base_edge)
        self.assertEqual(c.ground_nodes, [])
        self.assertEqual(c.ground_edges, [])

        # 3) Connected orphans {8} and {9}: each has a single grounding edge that
        #    the parser auto-promotes to base.
        for orphan_idx, base_jid in ((8, 7), (9, 13)):
            c = find_component(orphan_idx)
            self.assertEqual([int(n) for n in c.nodes], [orphan_idx])
            self.assertFalse(c.is_island)
            self.assertTrue(c.is_connected)
            self.assertEqual([e.joint_index for e in c.edges], [base_jid])
            self.assertEqual(int(c.base_node), orphan_idx)
            self.assertIsNotNone(c.base_edge)
            self.assertEqual(c.base_edge.joint_index, base_jid)
            self.assertEqual(c.base_edge.nodes, (-1, orphan_idx))
            self.assertEqual(c.ground_nodes, [])
            self.assertEqual(c.ground_edges, [])

        # 4) Isolated orphans {12}, {13}, {14}, {15}: no incident edges, so no base
        #    can be assigned and the component remains entirely unconnected.
        for orphan_idx in (12, 13, 14, 15):
            c = find_component(orphan_idx)
            self.assertEqual([int(n) for n in c.nodes], [orphan_idx])
            self.assertFalse(c.is_island)
            self.assertFalse(c.is_connected)
            self.assertEqual(c.edges, [])
            self.assertIsNone(c.base_node)
            self.assertIsNone(c.base_edge)
            self.assertEqual(c.ground_nodes, [])
            self.assertEqual(c.ground_edges, [])

    def test_11_graph_component_parsing_empty_edges(self):
        """
        Parsing a graph with no edges must succeed and produce one orphan
        component per node, each ``isolated`` and not an ``island``.
        """
        nodes = [0, 1, 2]

        # Empty list of edges
        G = topology.TopologyGraph(nodes, edges=[], autoparse=False)
        C = G.parse_components()
        self.assertEqual(len(C), len(nodes))
        for c in C:
            self.assertEqual(len(c.nodes), 1)
            self.assertFalse(c.is_island)
            self.assertFalse(c.is_connected)
            self.assertIsNone(c.base_node)
            self.assertIsNone(c.base_edge)
            self.assertEqual(c.ground_nodes, [])
            self.assertEqual(c.ground_edges, [])

        # `edges=None` (the constructor must coerce to `[]`) — same expected outcome
        G_none = topology.TopologyGraph(nodes, edges=None, autoparse=False)
        C_none = G_none.parse_components()
        self.assertEqual(len(C_none), len(nodes))

    def test_12_graph_component_parsing_multi_grounding(self):
        """
        A component with multiple grounding edges including a single 6-DoF FREE joint
        must promote the FREE joint to the base edge and leave the remaining grounding
        edges in ``ground_edges`` with their endpoints listed exactly once in
        ``ground_nodes`` (regression for the ``list.remove`` duplicate-leakage bug).
        """
        nodes = [0, 1, 2]
        edges = [
            # Body 0 has two REVOLUTE grounding edges -> ground_nodes must list it once.
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (-1, 0)),
            # Body 1 has a 6-DoF FREE joint -> auto-promotes to base.
            (JointDoFType.FREE.value, 2, (-1, 1)),
            # Internal connectivity.
            (JointDoFType.REVOLUTE.value, 3, (0, 1)),
            (JointDoFType.REVOLUTE.value, 4, (1, 2)),
        ]
        G = topology.TopologyGraph(nodes, edges, autoparse=False)
        C = G.parse_components()
        self.assertEqual(len(C), 1)
        c = C[0]
        self.assertEqual(int(c.base_node), 1)
        self.assertEqual(c.base_edge.joint_type, JointDoFType.FREE.value)
        self.assertEqual(c.base_edge.joint_index, 2)
        # The base node must NOT remain in ground_nodes.
        self.assertNotIn(1, [int(n) for n in c.ground_nodes])
        # Body 0 has two grounding edges but must appear in ground_nodes exactly once.
        self.assertEqual([int(n) for n in c.ground_nodes], [0])
        self.assertEqual(len(c.ground_edges), 2)

    def test_13_graph_component_parsing_jid_sort_invariant(self):
        """
        Two graphs with identical structure but a relabeling of joint *types* must
        produce the same per-component edge order — i.e. ``parse_components`` must
        sort by joint index, not by joint type.
        """
        nodes = [0, 1, 2]
        edges_a = [
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
            (JointDoFType.PRISMATIC.value, 1, (0, 1)),
            (JointDoFType.SPHERICAL.value, 2, (1, 2)),
        ]
        edges_b = [
            (JointDoFType.SPHERICAL.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.PRISMATIC.value, 2, (1, 2)),
        ]
        ca = topology.TopologyGraph(nodes, edges_a, autoparse=False).parse_components()[0]
        cb = topology.TopologyGraph(nodes, edges_b, autoparse=False).parse_components()[0]
        self.assertEqual([e.joint_index for e in ca.edges], [e.joint_index for e in cb.edges])
        self.assertEqual([e.joint_index for e in ca.edges], [0, 1, 2])

    def test_14_graph_component_parsing_multi_free_grounding_raises(self):
        """
        A component with more than one 6-DoF FREE grounding edge violates modelling
        conventions and must raise ``ValueError`` from :meth:`parse_components`.
        """
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.FREE.value, 1, (-1, 1)),
            (JointDoFType.REVOLUTE.value, 2, (0, 1)),
            (JointDoFType.REVOLUTE.value, 3, (1, 2)),
        ]
        G = topology.TopologyGraph(nodes, edges, autoparse=False)
        with self.assertRaises(ValueError) as ctx:
            G.parse_components()
        self.assertIn("FREE", str(ctx.exception))

    def test_15_graph_component_parsing_multi_grounding_no_free_no_base(self):
        """
        Connected island with multiple non-FREE grounding edges leaves the component
        without a base assignment after parsing: every grounding edge is preserved in
        ``ground_edges``, ``ground_nodes`` lists each non-world endpoint exactly once,
        and ``is_connected`` is ``True`` on the strength of the original graph topology.
        """
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (-1, 2)),
            (JointDoFType.REVOLUTE.value, 2, (0, 1)),
            (JointDoFType.REVOLUTE.value, 3, (1, 2)),
        ]
        G = topology.TopologyGraph(nodes, edges, autoparse=False)
        C = G.parse_components()
        self.assertEqual(len(C), 1)
        c = C[0]
        self.assertIsNone(c.base_node)
        self.assertIsNone(c.base_edge)
        self.assertTrue(c.is_connected)
        self.assertTrue(c.is_island)
        self.assertEqual(sorted(int(n) for n in c.ground_nodes), [0, 2])
        self.assertEqual(len(c.ground_edges), 2)
        # All grounding edges must be REVOLUTE (no auto-promotion to base).
        self.assertTrue(all(e.joint_type == JointDoFType.REVOLUTE.value for e in c.ground_edges))

    ###
    # End-to-end topology discovery against imported asset models
    ###

    def test_20_discovery_topology_of_test_graph(self):
        """
        Tests the TopologyGraph class with a variety of node and edge configurations
        to ensure it correctly identifies components: islands, and orphans.
        """
        # Define a test graph with various node types:
        nodes, edges = _make_test_graph(unsorted_nodes=False)

        # Create a topology graph to manually run the pipeline step by step.
        graph_0 = topology.TopologyGraph(nodes, edges, base_selector=None)

        # Run the full pipeline manually to control the order of operations.
        graph_0.parse_components()
        graph_0.select_component_bases(bases=[7, 12, 13, 14, 15])
        graph_0.generate_spanning_trees(override_priorities=True, prioritize_balanced=False)
        graph_0.select_spanning_trees()
        # TODO: graph_0.compute_index_reassignment(reassign_inplace=True)

        # For each component, print the current base node/edge assignment:
        for i, comp in enumerate(graph_0.components):
            msg.info("Component %d: base_node=%s, base_edge=%s", i, comp.base_node, comp.base_edge)

        # Optional rendering output
        if self.plotfig or self.savefig:
            graph_0.render_graph(
                figsize=(10, 10), path=self.output_path / "test_20_graph_component_parsing.pdf", show=self.plotfig
            )
            graph_0.render_spanning_tree_candidates(
                figsize=(10, 10),
                path=self.output_path / "test_20_graph_component_parsing_candidates.pdf",
                show=self.plotfig,
            )
            graph_0.render_spanning_trees(
                figsize=(10, 10), path=self.output_path / "test_20_graph_component_parsing_trees.pdf", show=self.plotfig
            )

    def test_21_discover_topology_of_testmechanism(self):
        """End-to-end topology discovery on the Disney Research TestMechanism USD asset.

        Two instances of the asset are loaded into a ``ModelBuilderKamino`` and a
        :class:`TopologyGraph` is constructed for each world. Each per-world graph
        must yield a single (well-grounded) component, never need a synthetic base
        edge, and produce a Featherstone-numbered spanning tree.
        """
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")

        usd_importer = USDImporter()
        asset_builder: ModelBuilderKamino = usd_importer.import_from(
            source=asset_file,
            load_static_geometry=True,
            retain_joint_ordering=False,
            meshes_are_collidable=True,
            force_show_colliders=True,
            use_prim_path_names=True,
        )

        # Stack two copies of the asset into a single multi-world builder so the
        # test exercises both the per-world parsing path and component-level
        # invariants simultaneously.
        num_worlds = 2
        builder: ModelBuilderKamino = ModelBuilderKamino()
        for _i in range(num_worlds):
            builder.add_builder(asset_builder)

        per_world_inputs = _topology_inputs_from_kamino_builder(builder)
        self.assertEqual(len(per_world_inputs), num_worlds)

        for w, (nodes, edges) in enumerate(per_world_inputs):
            with self.subTest(world=w):
                base_selector = topology.TopologyHeaviestBodyBaseSelector()
                G = topology.TopologyGraph(
                    nodes,
                    edges,
                    base_selector=base_selector,
                    autoparse=True,
                )
                _assert_grounded_topology_invariants(self, G, expected_num_components=1)

                if self.plotfig or self.savefig:
                    G.render_graph(
                        figsize=(10, 10),
                        path=self.output_path / f"test_21_testmech_world_{w}_graph.pdf",
                        show=self.plotfig,
                        graph_labels=["tables"],
                    )
                    G.render_spanning_trees(
                        figsize=(10, 10),
                        path=self.output_path / f"test_21_testmech_world_{w}_trees.pdf",
                        show=self.plotfig,
                        graph_labels=["tables"],
                    )

    def test_22_discover_topology_of_anymal_d(self):
        """End-to-end topology discovery on the ANYbotics ANYmal D USD asset.

        Two instances of the asset are loaded into a single Newton :class:`ModelBuilder`
        world and a :class:`TopologyGraph` is built directly from the builder's
        joint connectivity arrays. The two robot instances must surface as two
        components with no synthetic base edges (each robot ships with a 6-DoF
        FREE base joint).
        """

        def _load_anymal_d_from_usd(builder: ModelBuilder):
            asset_path = newton.utils.download_asset("anybotics_anymal_d")
            asset_file = str(asset_path / "usd" / "anymal_d.usda")
            builder.add_usd(
                source=asset_file,
                collapse_fixed_joints=True,
                enable_self_collisions=False,
                force_show_colliders=True,
            )

        builder: ModelBuilder = ModelBuilder()
        builder.begin_world()
        _load_anymal_d_from_usd(builder)
        _load_anymal_d_from_usd(builder)
        builder.end_world()

        # Two instances of the same articulated robot → two articulations.
        self.assertEqual(builder.articulation_count, 2)

        nodes, edges = extract_graph_inputs_from_builder(builder)
        bodies = bodies_from_builder(builder)
        joints = joints_from_builder(builder)
        base_selector = topology.TopologyHeaviestBodyBaseSelector()
        G = topology.TopologyGraph(
            nodes,
            edges,
            base_selector=base_selector,
            bodies=bodies,
            joints=joints,
            autoparse=True,
        )
        # ANYmal D ships with a 6-DoF FREE base joint per instance, so no
        # synthetic base edges should be needed.
        _assert_grounded_topology_invariants(self, G, expected_num_components=2, expected_num_synthetic_edges=0)

        if self.plotfig or self.savefig:
            G.render_graph(
                figsize=(12, 12),
                path=self.output_path / "test_22_anymal_d_graph.pdf",
                show=self.plotfig,
                graph_labels={"inline", "tables"},
            )
            G.render_spanning_trees(
                figsize=(12, 12),
                path=self.output_path / "test_22_anymal_d_trees.pdf",
                show=self.plotfig,
                graph_labels={"inline", "tables"},
            )

        # As a sanity check, also verify that we can finalize a model and that
        # its joint-articulation grouping agrees with the discovered components
        # (each ANYmal D instance's 13 joints fall into one component / one
        # articulation).
        model: Model = builder.finalize()
        self.assertEqual(model.articulation_count, 2)
        msg.info("Discovered %d components matching %d articulations", len(G.components), model.articulation_count)

    def test_23_discover_topology_of_dr_legs(self):
        """End-to-end topology discovery on the Disney Research DR Legs USD asset.

        Loads two instances of the asset into a Newton :class:`ModelBuilder`
        and builds a :class:`TopologyGraph` from the joint connectivity. The
        DR Legs asset ships *without* a 6-DoF FREE base joint per instance —
        each leg is purely an articulated tree-with-loops with no world
        anchor — so the heaviest-body selector must mint one synthetic base
        edge per instance to make the components connected. The reassignment
        back-end then has to remap those ``NJ + k`` synthetic indices into
        the new joint space.
        """
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")

        builder: ModelBuilder = ModelBuilder()
        builder.begin_world()
        builder.add_usd(
            source=asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder.add_usd(
            source=asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder.end_world()

        nodes, edges = extract_graph_inputs_from_builder(builder)
        bodies = bodies_from_builder(builder)
        joints = joints_from_builder(builder)
        nj_orig = len(edges)

        base_selector = topology.TopologyHeaviestBodyBaseSelector()
        G = topology.TopologyGraph(
            nodes,
            edges,
            base_selector=base_selector,
            bodies=bodies,
            joints=joints,
            autoparse=True,
        )
        # Two robot instances → two components, each needing a synthetic FREE base edge.
        _assert_grounded_topology_invariants(self, G, expected_num_components=2, expected_num_synthetic_edges=2)

        # DEBUGGING OUTPUT
        for nbe in G.new_base_edges:
            msg.warning("new_base_edge: %s\n", nbe)

        msg.warning("body_node_remap: %s", G.body_node_remap)
        msg.warning("joint_edge_remap: %s\n", G.joint_edge_remap)

        msg.warning("trees[0]: %s\n", G.trees[0])
        msg.warning("trees_remapped[0]: %s\n", G.trees_remapped[0])

        # Synthetic edges must be tagged with consecutive ``NJ + k`` indices so the
        # downstream model builder can inject them at unambiguous positions.
        synthetic_indices = sorted(e.joint_index for e in G.new_base_edges)
        self.assertEqual(synthetic_indices, [nj_orig, nj_orig + 1])
        for e in G.new_base_edges:
            self.assertEqual(e.joint_type, JointDoFType.FREE.value)

        if self.plotfig or self.savefig:
            G.render_graph(
                figsize=(12, 12),
                path=self.output_path / "test_23_dr_legs_graph.pdf",
                show=self.plotfig,
                graph_labels={"tables"},
            )
            G.render_spanning_trees(
                figsize=(12, 12),
                path=self.output_path / "test_23_dr_legs_trees.pdf",
                show=self.plotfig,
                graph_labels={"tables"},
            )

        # Check re-assignment final output:
        # TODO: the builder.joint_parent array should be equal to the

    ###
    # End-to-end pipeline with the shipped selector backends
    ###

    def test_30_graph_pipeline_end_to_end_with_selectors(self):
        """End-to-end :meth:`TopologyGraph.parse` with both shipped selector backends.

        Builds a graph with one connected island (single FREE grounding edge -> auto-
        promoted base) plus one isolated island (no grounding edges -> base selector
        synthesizes a FREE edge on the heaviest body) and checks that ``G.trees`` is
        populated, the connected component keeps the auto-assigned FREE base, and the
        isolated component receives a synthesized FREE base on its heaviest body —
        with the synthetic edge committed at the next ``NJ + 0`` joint index.
        """
        nodes = [0, 1, 2, 3, 4]
        edges = [
            # Connected chain 0-1-2 with a FREE grounding on body 0 (auto-base).
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
            # Isolated island 3-4 (no grounding); the base selector decides the base.
            (JointDoFType.REVOLUTE.value, 3, (3, 4)),
        ]
        nj_orig = len(edges)
        bodies = [
            RigidBodyDescriptor(name="body_0", m_i=2.0, bid=0),
            RigidBodyDescriptor(name="body_1", m_i=1.0, bid=1),
            RigidBodyDescriptor(name="body_2", m_i=1.0, bid=2),
            RigidBodyDescriptor(name="body_3", m_i=1.0, bid=3),
            RigidBodyDescriptor(name="body_4", m_i=5.0, bid=4),
        ]

        base_selector = topology.TopologyHeaviestBodyBaseSelector()
        G = topology.TopologyGraph(
            nodes,
            edges,
            base_selector=base_selector,
            bodies=bodies,
            autoparse=True,
        )

        # Two components, two selected trees.
        self.assertEqual(len(G.components), 2)
        self.assertEqual(len(G.trees), 2)

        # Locate the connected and isolated components by their node sets.
        chain = next(c for c in G.components if 0 in [int(n) for n in (c.nodes or [])])
        island = next(c for c in G.components if 3 in [int(n) for n in (c.nodes or [])])

        # Connected chain: parser auto-promoted the single FREE grounding.
        self.assertEqual(int(chain.base_node), 0)
        self.assertIsNotNone(chain.base_edge)
        self.assertEqual(chain.base_edge.joint_type, JointDoFType.FREE.value)
        self.assertEqual(chain.base_edge.joint_index, 0)
        self.assertEqual(chain.ground_edges, [])

        # Isolated island: heaviest body 4 is selected; synthetic FREE edge to world
        # gets a fresh ``NJ + 0`` joint index so downstream consumers can still
        # remap it through ``G.joint_edge_remap`` after reassignment.
        self.assertEqual(int(island.base_node), 4)
        self.assertIsNotNone(island.base_edge)
        self.assertEqual(island.base_edge.joint_type, JointDoFType.FREE.value)
        self.assertEqual(island.base_edge.joint_index, nj_orig)
        # Heaviest selector orients the synthetic edge as ``(world, base_idx)``.
        self.assertIn(-1, island.base_edge.nodes)
        self.assertIn(4, island.base_edge.nodes)
        self.assertTrue(island.is_connected)

        # The synthetic edge is also tracked in ``G.new_base_edges`` for downstream injection.
        self.assertIsNotNone(G.new_base_edges)
        self.assertEqual(len(G.new_base_edges), 1)
        self.assertEqual(G.new_base_edges[0].joint_index, nj_orig)

        # Selected trees match the components' base nodes and satisfy Featherstone numbering.
        for tree in G.trees:
            _assert_featherstone_invariants(self, tree)
            if tree.component is chain:
                self.assertEqual(tree.root, 0)
            elif tree.component is island:
                self.assertEqual(tree.root, 4)

    def test_31_graph_base_selector_invoked_for_multi_grounding_no_free(self):
        """Connected island with multiple non-FREE groundings invokes the base selector,
        which picks the heaviest body and promotes one of its incident grounding edges
        to base. The promoted edge must be removed from ``ground_edges`` and
        ``ground_nodes`` recomputed accordingly.
        """
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.REVOLUTE.value, 0, (-1, 0)),  # grounding on body 0
            (JointDoFType.REVOLUTE.value, 1, (-1, 2)),  # grounding on body 2
            (JointDoFType.REVOLUTE.value, 2, (0, 1)),
            (JointDoFType.REVOLUTE.value, 3, (1, 2)),
        ]
        # Body 2 is the heaviest -> selector picks body 2 and promotes the (-1, 2)
        # grounding edge (joint index 1) to base.
        bodies = [
            RigidBodyDescriptor(name="body_0", m_i=1.0, bid=0),
            RigidBodyDescriptor(name="body_1", m_i=1.0, bid=1),
            RigidBodyDescriptor(name="body_2", m_i=4.0, bid=2),
        ]

        base_selector = topology.TopologyHeaviestBodyBaseSelector()
        G = topology.TopologyGraph(
            nodes,
            edges,
            base_selector=base_selector,
            bodies=bodies,
            autoparse=True,
        )

        self.assertEqual(len(G.components), 1)
        c = G.components[0]
        self.assertEqual(int(c.base_node), 2)
        self.assertIsNotNone(c.base_edge)
        self.assertEqual(c.base_edge.joint_index, 1)  # joint index 1 is the (-1, 2) grounding
        self.assertEqual(c.base_edge.joint_type, JointDoFType.REVOLUTE.value)

        # The promoted edge must be removed from `ground_edges`, leaving only the
        # remaining grounding edge on body 0.
        self.assertEqual(len(c.ground_edges), 1)
        self.assertEqual(c.ground_edges[0].joint_index, 0)
        self.assertEqual([int(n) for n in c.ground_nodes], [0])
        self.assertTrue(c.is_connected)

        # The selected tree is rooted at body 2.
        self.assertEqual(len(G.trees), 1)
        self.assertEqual(G.trees[0].root, 2)

    def test_32_parse_pipeline_explicit_call_runs_all_stages(self):
        """Calling :meth:`TopologyGraph.parse` explicitly populates components, candidates, and trees."""
        nodes = [0, 1, 2]
        edges = [
            (JointDoFType.FREE.value, 0, (-1, 0)),
            (JointDoFType.REVOLUTE.value, 1, (0, 1)),
            (JointDoFType.REVOLUTE.value, 2, (1, 2)),
        ]
        G = topology.TopologyGraph(nodes, edges, autoparse=False)
        # Before `parse()` runs, the derived properties must raise.
        with self.assertRaises(ValueError):
            _ = G.components
        with self.assertRaises(ValueError):
            _ = G.candidates
        with self.assertRaises(ValueError):
            _ = G.trees

        G.parse()

        self.assertEqual(len(G.components), 1)
        self.assertEqual(len(G.candidates), 1)
        self.assertEqual(len(G.trees), 1)
        self.assertEqual(G.trees[0].root, 0)
        self.assertEqual(G.trees[0].num_bodies, 3)

    def test_33_parse_pipeline_missing_required_module_raises(self):
        """:meth:`parse` must validate required pipeline modules up front, not lazily."""
        # Construct a graph with a stripped pipeline (no tree generator).
        G = topology.TopologyGraph(
            nodes=[0, 1],
            edges=[(JointDoFType.FREE.value, 0, (-1, 0)), (JointDoFType.REVOLUTE.value, 1, (0, 1))],
            tree_generator=None,
            autoparse=False,
        )
        # The default tree generator is auto-installed by the constructor, so
        # explicitly clear it to simulate a custom pipeline missing a module.
        G._tree_generator = None
        with self.assertRaisesRegex(ValueError, "tree_generator"):
            G.parse()


class TestTopologyInteropUtils(unittest.TestCase):
    """End-to-end coverage for :mod:`kamino._src.topology.utils` interop helpers.

    Test layout:
    - ``test_0*_extract_*`` — extraction helpers from a Newton ``ModelBuilder``.
    - ``test_1*_apply_*`` — :func:`apply_discovered_topology_to_builder` on
      synthetic builders (already-grounded chain, ungrounded chain).
    - ``test_2*_export_usd_*`` — :func:`export_usd_with_discovered_topology`
      round-trip on a real USD asset.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = True
        self.savefig = False
        self.plotfig = False
        self.output_path = test_context.output_path / "test_topology" / "interop_utils"
        if self.savefig:
            self.output_path.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print("\n")
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    @staticmethod
    def _make_grounded_chain(num_links: int = 4) -> ModelBuilder:
        """Return a Newton :class:`ModelBuilder` with a FREE-grounded revolute chain."""
        builder = ModelBuilder()
        builder.begin_world()
        prev_body = -1
        for i in range(num_links):
            body = builder.add_link(label=f"link_{i}", mass=1.0)
            if i == 0:
                builder.add_joint_free(child=body, label=f"joint_{i}")
            else:
                builder.add_joint_revolute(parent=prev_body, child=body, axis=newton.Axis.Z, label=f"joint_{i}")
            prev_body = body
        builder.end_world()
        return builder

    @staticmethod
    def _make_ungrounded_chain(num_links: int = 3) -> ModelBuilder:
        """Return a builder with a non-grounded revolute chain (no FREE base)."""
        builder = ModelBuilder()
        builder.begin_world()
        prev_body = -1
        for i in range(num_links):
            body = builder.add_link(label=f"link_{i}", mass=1.0)
            if i > 0:
                builder.add_joint_revolute(parent=prev_body, child=body, axis=newton.Axis.Z, label=f"joint_{i}")
            prev_body = body
        builder.end_world()
        return builder

    ###
    # Extraction helpers
    ###

    def test_01_extract_graph_inputs_round_trips_labels_and_types(self):
        """``extract_graph_inputs_from_builder`` mirrors per-joint labels, parent/child,
        and Newton→Kamino joint type translation.
        """
        builder = self._make_grounded_chain(num_links=3)
        nodes, edges = extract_graph_inputs_from_builder(builder)

        self.assertEqual(nodes, list(range(builder.body_count)))
        self.assertEqual(len(edges), builder.joint_count)
        for j, (joint_type, joint_index, (parent, child)) in enumerate(edges):
            self.assertEqual(joint_index, j)
            self.assertEqual(parent, int(builder.joint_parent[j]))
            self.assertEqual(child, int(builder.joint_child[j]))
            self.assertEqual(joint_type, NEWTON_TO_KAMINO_JOINT_TYPE[int(builder.joint_type[j])])
        # FREE base (joint 0) and revolute spine (joints 1, 2).
        self.assertEqual(edges[0][0], int(JointDoFType.FREE))
        for j in range(1, 3):
            self.assertEqual(edges[j][0], int(JointDoFType.REVOLUTE))

    def test_02_bodies_and_joints_from_builder_propagate_metadata(self):
        """``bodies_from_builder`` and ``joints_from_builder`` pull labels and mass."""
        builder = self._make_grounded_chain(num_links=3)
        bodies = bodies_from_builder(builder)
        joints = joints_from_builder(builder)

        self.assertEqual(len(bodies), builder.body_count)
        self.assertEqual(len(joints), builder.joint_count)
        for i, body in enumerate(bodies):
            self.assertIsInstance(body, RigidBodyDescriptor)
            self.assertEqual(body.name, builder.body_label[i])
            self.assertEqual(body.bid, i)
            self.assertAlmostEqual(body.m_i, float(builder.body_mass[i]))
        for j, joint in enumerate(joints):
            self.assertIsInstance(joint, JointDescriptor)
            self.assertEqual(joint.name, builder.joint_label[j])
            self.assertEqual(int(joint.dof_type), NEWTON_TO_KAMINO_JOINT_TYPE[int(builder.joint_type[j])])

    def test_03_discover_topology_rejects_multi_world_builder(self):
        """``discover_topology_for_builder`` enforces the single-world precondition."""
        builder = ModelBuilder()
        builder.begin_world()
        builder.add_link(label="w0_link", mass=1.0)
        builder.end_world()
        builder.begin_world()
        builder.add_link(label="w1_link", mass=1.0)
        builder.end_world()
        with self.assertRaisesRegex(ValueError, "single-world"):
            discover_topology_for_builder(builder)

    ###
    # Builder round-trip
    ###

    def test_11_apply_topology_grounded_chain_no_synthetic_edges(self):
        """A FREE-grounded revolute chain rebuilds into a single articulation
        spanning every joint with no synthetic base edges.
        """
        builder = self._make_grounded_chain(num_links=4)
        new_builder = apply_discovered_topology_to_builder(builder)

        self.assertEqual(new_builder.body_count, builder.body_count)
        self.assertEqual(new_builder.joint_count, builder.joint_count)
        self.assertEqual(new_builder.articulation_count, 1)
        self.assertEqual(new_builder.articulation_start, [0])
        self.assertEqual(list(new_builder.joint_articulation), [0] * new_builder.joint_count)
        self.assertEqual(new_builder.joint_parent[0], -1)
        # The sums over per-coord/per-dof arrays are conserved by a permutation.
        self.assertEqual(new_builder.joint_coord_count, builder.joint_coord_count)
        self.assertEqual(new_builder.joint_dof_count, builder.joint_dof_count)
        self.assertEqual(new_builder.joint_constraint_count, builder.joint_constraint_count)
        # The new builder must finalize cleanly into a Newton model.
        model: Model = new_builder.finalize()
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(model.body_count, new_builder.body_count)
        self.assertEqual(model.joint_count, new_builder.joint_count)

    def test_12_apply_topology_synthesizes_free_base_for_ungrounded_chain(self):
        """An ungrounded chain gains exactly one synthetic FREE base joint at
        the start of the rebuilt articulation.
        """
        builder = self._make_ungrounded_chain(num_links=3)
        new_builder = apply_discovered_topology_to_builder(builder)

        # +1 joint over the original (the synthetic FREE base).
        self.assertEqual(new_builder.body_count, builder.body_count)
        self.assertEqual(new_builder.joint_count, builder.joint_count + 1)
        self.assertEqual(new_builder.articulation_count, 1)
        self.assertEqual(new_builder.articulation_start, [0])
        # The synthetic FREE joint occupies the first slot of the articulation.
        self.assertEqual(int(new_builder.joint_type[0]), int(newton.JointType.FREE))
        self.assertEqual(new_builder.joint_parent[0], -1)
        self.assertTrue(new_builder.joint_label[0].startswith("synthetic_base_"))
        # Every remapped joint sits inside the same articulation.
        self.assertEqual(list(new_builder.joint_articulation), [0] * new_builder.joint_count)
        # Per-DOF totals: 6 (FREE base) + 1 + 1 (revolute spine) = 8 dofs,
        # 7 (FREE) + 1 + 1 = 9 coords.
        self.assertEqual(new_builder.joint_coord_count, 9)
        self.assertEqual(new_builder.joint_dof_count, 8)
        # Finalization must succeed.
        model: Model = new_builder.finalize()
        self.assertEqual(model.articulation_count, 1)

    def test_13_apply_topology_does_not_mutate_source_builder(self):
        """``apply_discovered_topology_to_builder`` deep-copies its input."""
        builder = self._make_ungrounded_chain(num_links=3)
        original_joint_count = builder.joint_count
        original_body_count = builder.body_count
        original_articulation_count = builder.articulation_count
        _ = apply_discovered_topology_to_builder(builder)
        self.assertEqual(builder.joint_count, original_joint_count)
        self.assertEqual(builder.body_count, original_body_count)
        self.assertEqual(builder.articulation_count, original_articulation_count)

    def test_14_apply_topology_propagates_label_prefix(self):
        """The ``label`` kwarg prefixes both articulation labels and synthetic joint labels."""
        builder = self._make_ungrounded_chain(num_links=2)
        new_builder = apply_discovered_topology_to_builder(builder, label="robot")
        self.assertEqual(new_builder.articulation_label, ["robot_articulation_0"])
        self.assertTrue(new_builder.joint_label[0].startswith("robot_synthetic_base_"))

    def test_15_apply_topology_anymal_d_round_trips_via_builder(self):
        """End-to-end: load ANYmal D from USD, apply topology, finalize cleanly.

        The pre-baked asset already has FREE bases, so this exercises the
        regular-numbering permutation rather than the synthetic-FREE injection.
        """
        try:
            asset_path = newton.utils.download_asset("anybotics_anymal_d")
        except Exception as exc:  # pragma: no cover — network/asset issues
            self.skipTest(f"ANYmal D asset unavailable: {exc!r}")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")

        builder = ModelBuilder()
        builder.begin_world()
        builder.add_usd(
            source=asset_file,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        builder.end_world()

        graph = discover_topology_for_builder(builder)
        new_builder = apply_discovered_topology_to_builder(builder, graph=graph)

        # A single ANYmal D instance → exactly one articulation in the rebuilt builder.
        self.assertEqual(new_builder.articulation_count, len(graph.trees))
        self.assertEqual(new_builder.body_count, builder.body_count)
        self.assertEqual(new_builder.joint_count, builder.joint_count + len(graph.new_base_edges or []))

        model: Model = new_builder.finalize()
        self.assertEqual(model.articulation_count, new_builder.articulation_count)

    ###
    # USD round-trip
    ###

    def test_21_export_usd_writes_articulation_root_api(self):
        """Round-trip a USD asset and verify the discovered topology is authored.

        Uses the ANYmal D asset (FREE-grounded so no synthetic edges are needed)
        and asserts that:

        - The output stage opens cleanly with ``pxr``.
        - At least one prim in the output stage carries the
          :class:`UsdPhysics.ArticulationRootAPI` applied schema.
        """
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("pxr (usd-core) is not installed")

        try:
            asset_path = newton.utils.download_asset("anybotics_anymal_d")
        except Exception as exc:  # pragma: no cover — network/asset issues
            self.skipTest(f"ANYmal D asset unavailable: {exc!r}")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")

        out_dir = test_context.output_path / "test_topology" / "interop_utils"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "anymal_d_with_topology.usda"
        if out_path.exists():
            out_path.unlink()

        result = export_usd_with_discovered_topology(
            asset_file,
            out_path,
            add_usd_kwargs={"collapse_fixed_joints": True},
        )
        self.assertEqual(result, out_path)
        self.assertTrue(out_path.exists())

        stage = Usd.Stage.Open(str(out_path))
        self.assertIsNotNone(stage)
        roots_with_api = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        self.assertGreaterEqual(
            len(roots_with_api),
            1,
            f"Expected at least one prim with `UsdPhysics.ArticulationRootAPI` in `{out_path}`.",
        )

    def test_22_export_usd_writes_chord_excludes_for_dr_testmech(self):
        """Round-trip the DR TestMechanism USD asset and verify topology authoring.

        DR TestMechanism is a well-grounded mechanism *with loop closures*:
        unlike ANYmal D it does have chord joints, but unlike DR Legs it does
        not need a synthesized FREE base (its grounding is already in the
        source asset). The test asserts:

        - The output stage opens cleanly with ``pxr``.
        - At least one prim in the output stage carries the
          :class:`UsdPhysics.ArticulationRootAPI` applied schema.
        - At least one joint prim in the output stage carries
          ``physics:excludeFromArticulation = True``.
        """
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("pxr (usd-core) is not installed")

        try:
            asset_path = newton.utils.download_asset("disneyresearch")
        except Exception as exc:  # pragma: no cover — network/asset issues
            self.skipTest(f"DR TestMechanism asset unavailable: {exc!r}")
        asset_file = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")

        out_dir = test_context.output_path / "test_topology" / "interop_utils"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dr_testmech_with_topology.usda"
        if out_path.exists():
            out_path.unlink()

        result = export_usd_with_discovered_topology(asset_file, out_path)
        self.assertEqual(result, out_path)
        self.assertTrue(out_path.exists())

        stage = Usd.Stage.Open(str(out_path))
        self.assertIsNotNone(stage)

        roots_with_api = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        self.assertGreaterEqual(
            len(roots_with_api),
            1,
            f"Expected at least one prim with `UsdPhysics.ArticulationRootAPI` in `{out_path}`.",
        )

        excluded_chord_prims = []
        for prim in stage.Traverse():
            attr = prim.GetAttribute("physics:excludeFromArticulation")
            if attr and attr.IsValid() and attr.Get() is True:
                excluded_chord_prims.append(prim.GetPath())
        self.assertGreaterEqual(
            len(excluded_chord_prims),
            1,
            f"Expected at least one joint prim with `physics:excludeFromArticulation=True` in `{out_path}`.",
        )

    def test_23_export_usd_writes_chord_excludes_for_dr_legs(self):
        """Round-trip the DR Legs USD asset and verify chord-joint authoring.

        Unlike ANYmal D (a pure tree), the DR Legs asset ships as a
        articulated mechanism *with loop closures*. This exercises the chord-
        joint side of :func:`export_usd_with_discovered_topology`:

        - The output stage opens cleanly with ``pxr``.
        - At least one prim in the output stage carries the
          :class:`UsdPhysics.ArticulationRootAPI` applied schema (the
          synthesized FREE base lands on the heaviest body's prim).
        - At least one joint prim in the output stage carries
          ``physics:excludeFromArticulation = True``, which is the unique
          USD-level signal authored for chord (loop-closing) joints.
        """
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("pxr (usd-core) is not installed")

        try:
            asset_path = newton.utils.download_asset("disneyresearch")
        except Exception as exc:  # pragma: no cover — network/asset issues
            self.skipTest(f"DR Legs asset unavailable: {exc!r}")
        asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")

        out_dir = test_context.output_path / "test_topology" / "interop_utils"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dr_legs_with_topology.usda"
        if out_path.exists():
            out_path.unlink()

        result = export_usd_with_discovered_topology(
            asset_file,
            out_path,
            add_usd_kwargs={
                "joint_ordering": None,
                "force_show_colliders": True,
                "force_position_velocity_actuation": True,
            },
        )
        self.assertEqual(result, out_path)
        self.assertTrue(out_path.exists())

        stage = Usd.Stage.Open(str(out_path))
        self.assertIsNotNone(stage)

        roots_with_api = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        self.assertGreaterEqual(
            len(roots_with_api),
            1,
            f"Expected at least one prim with `UsdPhysics.ArticulationRootAPI` in `{out_path}`.",
        )

        excluded_chord_prims = []
        for prim in stage.Traverse():
            attr = prim.GetAttribute("physics:excludeFromArticulation")
            if attr and attr.IsValid() and attr.Get() is True:
                excluded_chord_prims.append(prim.GetPath())
        self.assertGreaterEqual(
            len(excluded_chord_prims),
            1,
            f"Expected at least one joint prim with `physics:excludeFromArticulation=True` in `{out_path}`.",
        )


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
