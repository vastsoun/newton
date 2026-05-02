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
from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Helpers
###


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


###
# TopologyComponent
###


class TestTopologyComponentInvariants(unittest.TestCase):
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
    - ``test_2*_discovery_topology_*`` — end-to-end topology discovery against
      imported asset models (USD).
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.savefig = False  # Set to True for saving plotting output
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

    ###
    # Component parsing
    ###

    def test_10_graph_component_parsing(self):
        """
        Tests the TopologyGraph class with a variety of node and edge configurations
        to ensure it correctly identifies components: islands, and orphans.
        """
        # Define a test graph with various node types:
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
        G = topology.TopologyGraph(nodes, edges, autoparse=False)
        C = G.parse_components()
        T = G.generate_spanning_trees(override_priorities=True)
        S = G.select_spanning_trees()

        # Optional debug output
        print("\n")
        msg.info("G.components:\n%s", C)
        msg.info("G.spanning_tree_candidates:\n%s", T)
        msg.info("G.spanning_trees:\n%s", S)

        # Optional rendering output
        if self.plotfig or self.savefig:
            G.render_graph(
                figsize=(10, 10), path=self.output_path / "test_10_graph_component_parsing.pdf", show=self.plotfig
            )
            G.render_spanning_tree_candidates(
                figsize=(10, 10),
                path=self.output_path / "test_10_graph_component_parsing_candidates.pdf",
                show=self.plotfig,
            )
            G.render_spanning_trees(
                figsize=(10, 10), path=self.output_path / "test_10_graph_component_parsing_trees.pdf", show=self.plotfig
            )

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
        self.assertEqual(c.base_node, 1)
        self.assertEqual(c.base_edge[0], JointDoFType.FREE.value)
        self.assertEqual(c.base_edge[1], 2)
        # The base node must NOT remain in ground_nodes.
        self.assertNotIn(1, c.ground_nodes)
        # Body 0 has two grounding edges but must appear in ground_nodes exactly once.
        self.assertEqual(c.ground_nodes, [0])
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
        self.assertEqual([e[1] for e in ca.edges], [e[1] for e in cb.edges])
        self.assertEqual([e[1] for e in ca.edges], [0, 1, 2])

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
        self.assertEqual(sorted(c.ground_nodes), [0, 2])
        self.assertEqual(len(c.ground_edges), 2)
        # All grounding edges must be REVOLUTE (no auto-promotion to base).
        self.assertTrue(all(e[0] == JointDoFType.REVOLUTE.value for e in c.ground_edges))

    ###
    # End-to-end topology discovery against imported asset models
    ###

    def test_20_discovery_topology_testmechanism(self):
        """End-to-end sketch exercising the kamino USD import → ``ModelBuilderKamino`` →
        :class:`TopologyGraph` flow without making strict topological assertions; it
        currently serves as a smoke test that the pipeline runs to completion. See the
        adjacent ``test_1*_graph_component_parsing*`` cases for assertion-based checks.
        """
        # Define the path to the USD file for the DR testmechanism model
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        usd_importer = USDImporter()
        asset_builder: ModelBuilderKamino = usd_importer.import_from(
            source=asset_file,
            load_static_geometry=True,
            retain_joint_ordering=False,
            meshes_are_collidable=True,
            force_show_colliders=True,
            use_prim_path_names=True,
        )

        # Create a main builder to add the asset builder multiple times
        num_worlds = 2
        builder: ModelBuilderKamino = ModelBuilderKamino()
        for _i in range(num_worlds):
            builder.add_builder(asset_builder)

        # --------------------------------------------------------
        # Parse from ModelBuilderKamino

        def _parse_nodes_from_builder_kamino_bodies(builder: ModelBuilderKamino) -> list[topology.NodeType]:
            """Group rigid-body indices into a per-world list of topology graph nodes."""
            per_world_graph_nodes = [[] for _ in range(builder.num_worlds)]
            for i, body in enumerate(builder.all_bodies):
                per_world_graph_nodes[body.wid].append(i)
            return per_world_graph_nodes

        def _parse_edges_from_builder_kamino_joints(builder: ModelBuilderKamino) -> list[topology.EdgeType]:
            """Group joint descriptors into a per-world list of topology graph edges
            with global body indices for the predecessor/successor endpoints."""
            per_world_graph_edges = [[] for _ in range(builder.num_worlds)]
            world_bio = [0]
            for w in range(builder.num_worlds):
                world_bio.append(world_bio[-1] + len(builder.bodies[w]))
            for j, joint in enumerate(builder.all_joints):
                bid_P = joint.bid_B + world_bio[joint.wid] if joint.bid_B >= 0 else -1
                bid_S = joint.bid_F + world_bio[joint.wid] if joint.bid_F >= 0 else -1
                per_world_graph_edges[joint.wid].append((joint.dof_type.value, j, (bid_P, bid_S)))
            return per_world_graph_edges

        def _parse_nodes_and_edges_from_builder_kamino(
            builder: ModelBuilderKamino,
        ) -> tuple[list[topology.NodeType], list[topology.EdgeType]]:
            """Convenience wrapper returning ``(per_world_nodes, per_world_edges)``."""
            per_world_graph_nodes = _parse_nodes_from_builder_kamino_bodies(builder)
            per_world_graph_edges = _parse_edges_from_builder_kamino_joints(builder)
            return per_world_graph_nodes, per_world_graph_edges

        # --------------------------------------------------------
        # Parse from newton.ModelBuilder
        # TODO

        # --------------------------------------------------------
        # Parse from ModelKamino
        # TODO

        # --------------------------------------------------------
        # Parse from newton.Model
        # TODO

        # --------------------------------------------------------

        # TODO
        per_world_graph_nodes, per_world_graph_edges = _parse_nodes_and_edges_from_builder_kamino(builder)
        msg.info("Graph Nodes:\n%s", per_world_graph_nodes)
        msg.info("Graph Edges:\n%s", per_world_graph_edges)

        # Generate the topology graph for each world
        for w in range(builder.num_worlds):
            # Create the topology graph for the current world
            graph_w = topology.TopologyGraph(
                per_world_graph_nodes[w],
                per_world_graph_edges[w],
            )

            # Parse components before accessing the `.components` property — the full
            # pipeline (tree generator/selector backends) is not yet wired up here.
            graph_w.parse_components()
            graph_w.generate_spanning_trees()
            graph_w.select_spanning_trees()

            # For each component, generate a list of candidate spanning trees
            for c in graph_w.components:
                msg.info("Component:\n%s", c)

                # Optional rendering output
            if self.plotfig or self.savefig:
                graph_w.render_graph(
                    figsize=(10, 10),
                    path=self.output_path / f"test_20_discovery_topology_testmechanism_{w}.pdf",
                    show=self.plotfig,
                )
                graph_w.render_spanning_tree_candidates(
                    figsize=(10, 10),
                    path=self.output_path / f"test_20_discovery_topology_testmechanism_{w}_candidates.pdf",
                    show=self.plotfig,
                )
                graph_w.render_spanning_trees(
                    figsize=(10, 10),
                    path=self.output_path / f"test_20_discovery_topology_testmechanism_{w}_trees.pdf",
                    show=self.plotfig,
                )

    def test_21_discovery_topology_anymal_d(self):
        """
        Test the conversion operations between :class:`newton.Model` and ``kamino.ModelKamino``
        on the Anymal D model loaded from USD.
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

        # Create a model builder and load multiple instances of the Anymal D model
        # from USD to test the topology graph generation on a more complex system
        # with multiple components and articulation structures.
        builder_0: ModelBuilder = ModelBuilder()
        builder_0.begin_world()
        _load_anymal_d_from_usd(builder_0)
        _load_anymal_d_from_usd(builder_0)
        builder_0.end_world()

        # Generate a model from the builder
        model_0: Model = builder_0.finalize()

        # Print out the topology-related attributes of the model for debugging.
        # Use `info` rather than `warning` — these are diagnostic dumps, not problems.
        msg.info("model_0.joint_type:         %s", model_0.joint_type)
        msg.info("model_0.joint_parent:       %s", model_0.joint_parent)
        msg.info("model_0.joint_child:        %s", model_0.joint_child)
        msg.info("model_0.joint_ancestor:     %s", model_0.joint_ancestor)
        msg.info("model_0.joint_articulation: %s", model_0.joint_articulation)
        msg.info("model_0.articulation_count: %s", model_0.articulation_count)
        msg.info("model_0.articulation_label: %s", model_0.articulation_label)
        msg.info("model_0.articulation_start: %s", model_0.articulation_start)
        msg.info("model_0.articulation_world: %s", model_0.articulation_world)
        msg.info("model_0.articulation_world_start: %s", model_0.articulation_world_start)
        msg.info("model_0.max_joints_per_articulation: %s", model_0.max_joints_per_articulation)
        msg.info("model_0.max_dofs_per_articulation: %s", model_0.max_dofs_per_articulation)

    ###
    # End-to-end pipeline with the shipped selector backends
    ###

    def test_30_graph_pipeline_end_to_end_with_selectors(self):
        """End-to-end :meth:`TopologyGraph.parse` with both shipped selector backends.

        Builds a graph with one connected island (single FREE grounding edge -> auto-
        promoted base) plus one isolated island (no grounding edges -> base selector
        synthesizes a FREE edge on the heaviest body) and checks that ``G.trees`` is
        populated, the connected component keeps the auto-assigned FREE base, and the
        isolated component receives a synthesized FREE base on its heaviest body.
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
        chain = next(c for c in G.components if 0 in (c.nodes or []))
        island = next(c for c in G.components if 3 in (c.nodes or []))

        # Connected chain: parser auto-promoted the single FREE grounding.
        self.assertEqual(chain.base_node, 0)
        self.assertIsNotNone(chain.base_edge)
        self.assertEqual(chain.base_edge[0], JointDoFType.FREE.value)
        self.assertEqual(chain.base_edge[1], 0)
        self.assertEqual(chain.ground_edges, [])

        # Isolated island: heaviest body 4 is selected; synthetic FREE edge to world.
        self.assertEqual(island.base_node, 4)
        self.assertIsNotNone(island.base_edge)
        self.assertEqual(island.base_edge[0], JointDoFType.FREE.value)
        self.assertEqual(island.base_edge[1], -1)  # synthetic sentinel index
        self.assertEqual(island.base_edge[2], (-1, 4))
        self.assertTrue(island.is_connected)

        # Selected trees match the components' base nodes.
        for tree in G.trees:
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
        self.assertEqual(c.base_node, 2)
        self.assertIsNotNone(c.base_edge)
        self.assertEqual(c.base_edge[1], 1)  # joint index 1 is the (-1, 2) grounding
        self.assertEqual(c.base_edge[0], JointDoFType.REVOLUTE.value)

        # The promoted edge must be removed from `ground_edges`, leaving only the
        # remaining grounding edge on body 0.
        self.assertEqual(len(c.ground_edges), 1)
        self.assertEqual(c.ground_edges[0][1], 0)
        self.assertEqual(c.ground_nodes, [0])
        self.assertTrue(c.is_connected)

        # The selected tree is rooted at body 2.
        self.assertEqual(len(G.trees), 1)
        self.assertEqual(G.trees[0].root, 2)


###
# Minimum-depth spanning-tree generator
###


class TestTopologyMinimumDepthSpanningTreeGenerator(unittest.TestCase):
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
        self.assertEqual(sorted(comp.ground_nodes), [0, 2])

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs")

        self.assertGreater(len(trees), 0)
        for t in trees:
            self.assertIn(t.root, comp.ground_nodes)
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
        self.assertEqual(comp.base_node, 0)

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
        self.assertEqual(sorted(comp.ground_nodes), [0, 2])

        gen = topology.TopologyMinimumDepthSpanningTreeGenerator()
        trees = gen.generate_spanning_trees(component=comp, traversal_mode="bfs", roots=[2])

        self.assertGreater(len(trees), 0)
        # The grounding edge involving body 2 should now be the component's base edge.
        self.assertEqual(comp.base_node, 2)
        self.assertIsNotNone(comp.base_edge)
        self.assertEqual(comp.base_edge[1], 1)  # joint index 1 is the (-1, 2) edge
        # ...and removed from the grounding lists.
        self.assertNotIn(2, comp.ground_nodes)
        self.assertTrue(all(e[1] != 1 for e in comp.ground_edges))
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

        self.assertEqual(base_node, 2)
        self.assertEqual(base_edge[1], 1)  # joint index 1 = (-1, 2)
        self.assertEqual(base_edge[0], JointDoFType.REVOLUTE.value)

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

        self.assertEqual(base_node, 1)
        self.assertEqual(base_edge[0], JointDoFType.FREE.value)
        self.assertEqual(base_edge[1], -1)
        self.assertEqual(base_edge[2], (-1, 1))

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

        self.assertEqual(base_node, 0)
        self.assertEqual(base_edge, (JointDoFType.FREE.value, -1, (-1, 0)))

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

        self.assertEqual(base_node, 3)
        self.assertEqual(base_edge, (JointDoFType.FREE.value, -1, (-1, 3)))

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

        self.assertEqual(base_node, 0)

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

        self.assertEqual(base_edge[0], JointDoFType.FREE.value)
        self.assertEqual(base_edge[1], 1)

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
        self.assertEqual(base_edge[0], JointDoFType.REVOLUTE.value)
        self.assertEqual(base_edge[1], 0)

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

        self.assertEqual(base_node, 1)
        self.assertEqual(base_edge[2], (-7, 1))

    def test_synthetic_edge_uses_configured_joint_index(self):
        """A custom ``synthetic_base_joint_index`` is honored when synthesizing the FREE base edge."""
        comp = topology.TopologyComponent(
            nodes=[0],
            edges=[],
            ground_nodes=[],
            ground_edges=[],
            is_island=False,
            is_connected=False,
        )
        bodies = self._bodies([1.0])

        sel = topology.TopologyHeaviestBodyBaseSelector(synthetic_base_joint_index=42)
        _, base_edge = sel.select_base(component=comp, bodies=bodies)

        self.assertEqual(base_edge[1], 42)

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

    def test_constructor_rejects_non_int_synthetic_joint_index(self):
        """Non-integer ``synthetic_base_joint_index`` raises ``TypeError`` at construction."""
        with self.assertRaisesRegex(TypeError, "integer"):
            topology.TopologyHeaviestBodyBaseSelector(synthetic_base_joint_index="x")


###
# Minimum-depth spanning-tree selector
###


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


class TestTopologyMinimumDepthSpanningTreeSelector(unittest.TestCase):
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
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
