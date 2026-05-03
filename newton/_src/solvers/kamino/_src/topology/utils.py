# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""TODO"""

from __future__ import annotations

import numpy as np

from .....sim import ModelBuilder
from ..._src import topology
from ..._src.core.builder import ModelBuilderKamino
from ..._src.core.model import ModelKamino
from ..._src.core.topology import TopologyDescriptor, TopologyModel
from ..._src.utils import logger as msg

###
# Topology Utilities
###

# TODO: Create helper for the plotting output and re-use it across the different helper methods


def make_usd_asset_with_discovered_topology(
    source: str,
    usdpath: str,
    plotfig: bool = False,
    savefig: bool = False,
    figpath: str | None = None,
):
    """
    Loads a USD asset from file, discovers its topology, and saves a copy of the asset
    with the discovered topology graph and spanning tree candidates rendered as metadata.
    """
    pass


def make_topology_model_for_existing_single_world_builder_kamino(
    builder: ModelBuilderKamino,
    plotfig: bool = False,
    savefig: bool = False,
    figpath: str | None = None,
):
    # Ensure that the builder contains only a single world
    if builder.num_worlds != 1:
        raise ValueError(f"Expected a single world in the builder, but got {builder.num_worlds} worlds.")

    def _parse_nodes_from_builder_kamino_bodies(builder: ModelBuilderKamino) -> list[topology.NodeType]:
        return list(range(builder.num_bodies))

    def _parse_edges_from_builder_kamino_joints(builder: ModelBuilderKamino) -> list[topology.EdgeType]:
        world_graph_edges = []
        for j, joint in enumerate(builder.all_joints):
            bid_P = joint.bid_B if joint.bid_B >= 0 else -1
            bid_S = joint.bid_F if joint.bid_F >= 0 else -1
            world_graph_edges.append((joint.dof_type.value, j, (bid_P, bid_S)))
        return world_graph_edges

    def _parse_nodes_and_edges_from_builder_kamino(
        builder: ModelBuilderKamino,
    ) -> tuple[list[topology.NodeType], list[topology.EdgeType]]:
        world_graph_nodes = _parse_nodes_from_builder_kamino_bodies(builder)
        world_graph_edges = _parse_edges_from_builder_kamino_joints(builder)
        return world_graph_nodes, world_graph_edges

    # First parse the graph nodes and edges for each world in the builder based on the body and joint
    # descriptors, which contain the necessary information about world index and body/joint connectivity
    world_graph_nodes, world_graph_edges = _parse_nodes_and_edges_from_builder_kamino(builder)
    msg.info("Graph Nodes:\n%s", world_graph_nodes)
    msg.info("Graph Edges:\n%s", world_graph_edges)

    # Generate the topology graph for each world
    # Create the topology graph for the current world
    graph = topology.TopologyGraph(world_graph_nodes, world_graph_edges, autoparse=True)

    # Optional rendering output
    if plotfig or savefig:
        graph.render_graph(
            path=figpath / "topology_graph.pdf" if figpath else None,
            show=plotfig,
        )
        graph.render_spanning_tree_candidates(
            path=figpath / "spanning_tree_candidates.pdf" if figpath else None,
            show=plotfig,
        )
        graph.render_spanning_trees(
            path=figpath / "spanning_trees.pdf" if figpath else None,
            show=plotfig,
        )


def make_topology_model_for_existing_single_world_builder(
    builder: ModelBuilder,
    plotfig: bool = False,
    savefig: bool = False,
    figpath: str | None = None,
) -> ModelBuilder:
    # Ensure that the builder contains only a single world
    if builder.world_count != 1:
        raise ValueError(f"Expected a single world in the builder, but got {builder.world_count} worlds.")

    def _parse_nodes_from_builder_bodies(builder: ModelBuilder) -> list[topology.NodeType]:
        return list(range(builder.body_count))

    def _parse_edges_from_builder_joints(builder: ModelBuilder) -> list[topology.EdgeType]:
        world_graph_edges = []
        for j in range(builder.joint_count):
            world_graph_edges.append((builder.joint_type[j], j, (builder.joint_parent[j], builder.joint_child[j])))
        return world_graph_edges

    def _parse_nodes_and_edges_from_builder(
        builder: ModelBuilder,
    ) -> tuple[list[topology.NodeType], list[topology.EdgeType]]:
        world_graph_nodes = _parse_nodes_from_builder_bodies(builder)
        world_graph_edges = _parse_edges_from_builder_joints(builder)
        return world_graph_nodes, world_graph_edges

    # First parse the graph nodes and edges for each world in the builder based on the body and joint
    # descriptors, which contain the necessary information about world index and body/joint connectivity
    world_graph_nodes, world_graph_edges = _parse_nodes_and_edges_from_builder(builder)
    msg.info("Graph Nodes:\n%s", world_graph_nodes)
    msg.info("Graph Edges:\n%s", world_graph_edges)

    # Generate the topology graph for each world
    # Create the topology graph for the current world
    graph = topology.TopologyGraph(world_graph_nodes, world_graph_edges, autoparse=True)

    # Optional rendering output
    if plotfig or savefig:
        graph.render_graph(
            path=figpath / "topology_graph.pdf" if figpath else None,
            show=plotfig,
        )
        graph.render_spanning_tree_candidates(
            path=figpath / "spanning_tree_candidates.pdf" if figpath else None,
            show=plotfig,
        )
        graph.render_spanning_trees(
            path=figpath / "spanning_trees.pdf" if figpath else None,
            show=plotfig,
        )

    # Create a new builder which will have re-mapped bodies/joints according to the
    # optimized topology trees, and will be used to generate the final TopologyModel.
    _builder = (
        ModelBuilder()
    )  # TODO: Initialize common attributes like gravity, materials, etc. from the original builder

    # TODO:
    #   1. Get re-mapped topology spanning trees and index remaps
    #   2. Create a new builder, and duplicate all unaffected entities
    #   3. Re-add all bodies and joints in the new order defined by the re-mapped topology trees.
    #   4. For each list of topology joint indices use `builder.add_articulation()` to add the corresponding articulation info

    return _builder


def make_topology_model_for_existing_model_kamino(
    model: ModelKamino,
    plotfig: bool = False,
    savefig: bool = False,
    figpath: str | None = None,
):
    # TODO: Create a utility for batched simulation that looks at the first world and creates
    # a TopologyGraph for it, to then generate the TopologyModel as a batched instance of the
    # first world. This will be the main use case, so we should prioritize it.
    # TODO: How to handle slight heterogeneity across worlds in the builder?

    # Retrieve the number of bodies and joints in a single world from the max over all worlds.
    nb = model.size.max_of_num_bodies
    nj = model.size.max_of_num_joints

    # Initialize the prototype world's body and joint index lists based on the maximum counts across worlds.
    proto_body_indices_np = np.arange(nb)
    proto_joint_indices_np = np.arange(nj)
    proto_body_indices = proto_body_indices_np.astype(int).tolist()
    proto_joint_indices = proto_joint_indices_np.astype(int).tolist()
    msg.info("ModelKamino body indices:\n%s", proto_body_indices)
    msg.info("ModelKamino joint indices:\n%s", proto_joint_indices)

    # The predecessor/successor body indices for each joint are stored in the `bid_B` and `bid_F`
    proto_joint_predecessors = model.joints.bid_B.numpy()[:nj].tolist()
    proto_joint_successors = model.joints.bid_F.numpy()[:nj].tolist()
    proto_joint_types = model.joints.dof_type.numpy()[:nj].tolist()

    # Define the input nodes and edges for the prototype world
    # based on the joint connectivity and body indices.
    proto_nodes = list(proto_body_indices)
    proto_edges = []
    for j in range(nj):
        proto_edges.append((proto_joint_types[j], j, (proto_joint_predecessors[j], proto_joint_successors[j])))
    msg.info("Prototype world nodes:\n%s", proto_nodes)
    msg.info("Prototype world edges:\n%s", proto_edges)

    # The prototype world represents the "template" topology for all worlds in the builder, so we can use it
    # to construct a single TopologyGraph and check that it parses correctly before scaling up to all worlds.
    proto_topgraph = topology.TopologyGraph(nodes=proto_nodes, edges=proto_edges, autoparse=True)

    # Optional rendering output
    if plotfig or savefig:
        proto_topgraph.render_graph(
            path=figpath / "topology_graph.pdf" if figpath else None,
            show=plotfig,
        )
        proto_topgraph.render_spanning_tree_candidates(
            path=figpath / "spanning_tree_candidates.pdf" if figpath else None,
            show=plotfig,
        )
        proto_topgraph.render_spanning_trees(
            path=figpath / "spanning_trees.pdf" if figpath else None,
            show=plotfig,
        )

    # OK, now suppose we have all topology discovery outputs, namely:
    # - proto_topgraph.trees: a list of spanning trees to generate TopologyDescriptors from
    # - proto_topgraph.new_base_edges: a list of missing base edges to connect isolated components
    # - proto_topgraph.body_node_remap: body index remapping from the original to optimized ordering
    # - proto_topgraph.joint_edge_remap: joint index remapping from the original to optimized ordering

    # First extract the index offsets for bodies and joints in each world
    per_world_bodies_start = model.info.bodies_offset.numpy().astype(int).tolist()
    per_world_joints_start = model.info.joints_offset.numpy().astype(int).tolist()

    # First expand the proto_topgraph body/joint remapping lists to the full size of the prototype world by filling in identity mappings for any missing indices
    world_body_node_remapped_full = proto_body_indices_np.copy()  # TODO: Is this copy necessary?
    world_joint_edge_remapped_full = proto_joint_indices_np.copy()  # TODO: Is this copy necessary?
    if proto_topgraph.body_node_remap:
        for original_idx, remapped_idx in enumerate(proto_topgraph.body_node_remap):
            world_body_node_remapped_full[original_idx] = remapped_idx
    if proto_topgraph.joint_edge_remap:
        for original_idx, remapped_idx in enumerate(proto_topgraph.joint_edge_remap):
            world_joint_edge_remapped_full[original_idx] = remapped_idx
    world_body_node_remapped_full = world_body_node_remapped_full.astype(np.float32)
    world_joint_edge_remapped_full = world_joint_edge_remapped_full.astype(np.float32)
    msg.info("World body node remapped:\n%s", world_body_node_remapped_full)
    msg.info("World joint edge remapped:\n%s", world_joint_edge_remapped_full)

    # Now tile the prototype graph's body/joint remapping arrays across all worlds into corresponding 2d
    # arrays to get the full remapping from original to optimized indices for the entire multi-world model.
    model_body_node_remap_np = np.tile(world_body_node_remapped_full, reps=model.size.num_worlds)
    model_joint_edge_remap_np = np.tile(world_joint_edge_remapped_full, reps=model.size.num_worlds)
    msg.info("Model body node remap:\n%s", model_body_node_remap_np)
    msg.info("Model joint edge remap:\n%s", model_joint_edge_remap_np)

    # Now apply the per-world body/joint offset to each row of the tiled remapping arrays to get the
    # final global remapping from original to optimized indices for the entire multi-world model.
    for w in range(model.size.num_worlds):
        model_body_node_remap_np[per_world_bodies_start[w] : per_world_bodies_start[w + 1]] += per_world_bodies_start[w]
        model_joint_edge_remap_np[per_world_joints_start[w] : per_world_joints_start[w + 1]] += per_world_joints_start[
            w
        ]
    msg.info("Model body node remap with offsets:\n%s", model_body_node_remap_np)
    msg.info("Model joint edge remap with offsets:\n%s", model_joint_edge_remap_np)

    # The final remapping arrays can now be used to translate the original body/joint indices
    # in each world's graph to the optimized indices in the prototype graph, which should match
    # the spanning tree candidates and selected trees generated by the prototype graph.
    # TODO: Create a copy of model.joints and model.bodies to cache the original model data
    # TODO: Implement a set of Warp kernels to efficiently do this in parallel over:
    # - all bodies: update per-body data at the current index from the original one
    # - all shapes: change the per-shape bid to the new body index
    # - all joints: update per-joint data at the current index from the original one, and change the per-joint bid_B and bid_F to the new body indices

    # Retrieve the list prototype world's spanning trees re-mapped to the optimized indices in the prototype graph,
    # which should match the spanning tree candidates and selected trees generated by the prototype graph.
    proto_topology_descriptors: list[TopologyDescriptor] = []
    for tid, tree in enumerate(proto_topgraph.trees):
        proto_topology_descriptors.append(
            TopologyDescriptor(
                tree=tree,
                tid=tid,
                wid=0,
            )
        )

    # Replicate the prototype topology descriptors across all worlds in the model, applying the appropriate world index
    # and body/joint index offsets to get the final list of topology descriptors for the entire multi-world model.
    topology_descriptors = [proto_topology_descriptors]
    tid_counter = len(proto_topology_descriptors)
    for w in range(1, model.size.num_worlds):
        topology_descriptors.append([])
        for desc in proto_topology_descriptors:
            topology_descriptors[w].append(
                TopologyDescriptor(
                    tree=desc.tree.with_offsets(per_world_bodies_start[w], per_world_joints_start[w]),
                    tid=tid_counter,
                    wid=w,
                )
            )
            tid_counter += 1

    # Now create a TopologyModel instance in the model
    model.topology = TopologyModel.from_descriptors(descriptors=topology_descriptors)
