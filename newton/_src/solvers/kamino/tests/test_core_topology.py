# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""TODO"""

import unittest

import warp as wp

import newton
from newton._src.sim import JointType, Model, ModelBuilder
from newton._src.solvers.kamino._src.core import topology
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestCoreTopology(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        # self.verbose = test_context.verbose  # Set to True for detailed output
        self.verbose = True  # Set to True for detailed output
        self.savefig = False  # Set to True for saving plotting output

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

    def test_XX_topology_anymal_d(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
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

        # Print out the topology-related attributes of the model for debugging
        msg.warning("model_0.joint_type:         %s", model_0.joint_type)
        msg.warning("model_0.joint_parent:       %s", model_0.joint_parent)
        msg.warning("model_0.joint_child:        %s", model_0.joint_child)
        msg.warning("model_0.joint_ancestor:     %s", model_0.joint_ancestor)
        msg.warning("model_0.joint_articulation: %s", model_0.joint_articulation)
        msg.warning("model_0.articulation_count: %s", model_0.articulation_count)
        msg.warning("model_0.articulation_label: %s", model_0.articulation_label)
        msg.warning("model_0.articulation_start: %s", model_0.articulation_start)
        msg.warning("model_0.articulation_world: %s", model_0.articulation_world)
        msg.warning("model_0.articulation_world_start: %s", model_0.articulation_world_start)
        msg.warning("model_0.max_joints_per_articulation: %s", model_0.max_joints_per_articulation)
        msg.warning("model_0.max_dofs_per_articulation: %s", model_0.max_dofs_per_articulation)

    def test_00_sketch_topology_api(self):
        """
        TODO
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
            """TODO"""
            per_world_graph_nodes = [[] for _ in range(builder.num_worlds)]
            for i, body in enumerate(builder.all_bodies):
                per_world_graph_nodes[body.wid].append(i)
            return per_world_graph_nodes

        def _parse_edges_from_builder_kamino_joints(builder: ModelBuilderKamino) -> list[topology.EdgeType]:
            """TODO"""
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
            """TODO"""
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
                # joints=builder.joints[w],
            )

            # Parse components before accessing the `.components` property — the full
            # pipeline (tree generator/selector backends) is not yet wired up here.
            graph_w.parse_components()

            # For each component, generate a list of candidate spanning trees
            for c in graph_w.components:
                msg.info("Component:\n%s", c)

                # Optional rendering output
            if self.verbose:
                graph_w.render_graph(figsize=(10, 10), show=True)

    def test_00_graph_component_parsing(self):
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

        # Optional debug output
        print("\n")
        msg.info("G.components:\n%s", C)

        # Optional rendering output
        if self.verbose:
            G.render_graph(figsize=(10, 10), show=True)

    def test_00_graph_component_parsing_empty_edges(self):
        """
        Parsing a graph with no edges must succeed and produce one orphan component
        per node, each ``isolated`` and not an ``island``.
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

    def test_00_graph_component_parsing_multi_grounding(self):
        """
        A component with multiple grounding edges including a single 6-DoF FREE joint
        must promote the FREE joint to the base edge and leave the remaining grounding
        edges in ``ground_edges`` with their endpoints listed exactly once in
        ``ground_nodes`` (regression for the ``list.remove`` duplicate-leakage bug).
        """
        nodes = [0, 1, 2]
        edges = [
            # Body 0 has two REVOLUTE grounding edges -> ground_nodes must list it once.
            (JointType.REVOLUTE.value, 0, (-1, 0)),
            (JointType.REVOLUTE.value, 1, (-1, 0)),
            # Body 1 has a 6-DoF FREE joint -> auto-promotes to base.
            (JointType.FREE.value, 2, (-1, 1)),
            # Internal connectivity.
            (JointType.REVOLUTE.value, 3, (0, 1)),
            (JointType.REVOLUTE.value, 4, (1, 2)),
        ]
        G = topology.TopologyGraph(nodes, edges, autoparse=False)
        C = G.parse_components()
        self.assertEqual(len(C), 1)
        c = C[0]
        self.assertEqual(c.base_node, 1)
        self.assertEqual(c.base_edge[0], JointType.FREE.value)
        self.assertEqual(c.base_edge[1], 2)
        # The base node must NOT remain in ground_nodes.
        self.assertNotIn(1, c.ground_nodes)
        # Body 0 has two grounding edges but must appear in ground_nodes exactly once.
        self.assertEqual(c.ground_nodes, [0])
        self.assertEqual(len(c.ground_edges), 2)

    def test_00_graph_component_parsing_jid_sort_invariant(self):
        """
        Two graphs with identical structure but a relabeling of joint *types* must
        produce the same per-component edge order — i.e. ``parse_components`` must
        sort by joint index, not by joint type.
        """
        nodes = [0, 1, 2]
        edges_a = [
            (JointType.REVOLUTE.value, 0, (-1, 0)),
            (JointType.PRISMATIC.value, 1, (0, 1)),
            (JointType.BALL.value, 2, (1, 2)),
        ]
        edges_b = [
            (JointType.BALL.value, 0, (-1, 0)),
            (JointType.REVOLUTE.value, 1, (0, 1)),
            (JointType.PRISMATIC.value, 2, (1, 2)),
        ]
        ca = topology.TopologyGraph(nodes, edges_a, autoparse=False).parse_components()[0]
        cb = topology.TopologyGraph(nodes, edges_b, autoparse=False).parse_components()[0]
        self.assertEqual([e[1] for e in ca.edges], [e[1] for e in cb.edges])
        self.assertEqual([e[1] for e in ca.edges], [0, 1, 2])


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
