# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines Types & Containers for Rigid-Body Topology."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .bodies import RigidBodyDescriptor
from .joints import JointDescriptor
from .types import Descriptor, override

###
# Module interface
###

__all__ = [
    "TopologyDescriptor",
    "draw_graph",
    "parse_graph",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


def parse_graph(
    nodes: Iterable[int],
    edges: Iterable[tuple[int, int]],
    root: int = -1,
) -> dict[str, Any]:
    """
    Identify connected subgraphs ("islands") and orphan nodes in a graph.

    Parameters
    ----------
    nodes
        Iterable of node indices. May include `root` (-1 by default).
    edges
        Iterable of (u, v) pairs.
    root
        Index of the global root node.

    Returns
    -------
    dict with keys:
        - "islands": list[list[int]]
            Connected components of size > 1 among non-root nodes.
        - "orphans": list[int]
            Nodes whose only connection is to the root.
        - "isolated": list[int]
            Nodes with no edges at all.
        - "components": list[list[int]]
            All connected components among non-root nodes.
    """
    nodes = set(nodes)
    edges = list(edges)

    # Keep only real nodes, not the global root
    real_nodes = {n for n in nodes if n != root}

    # Union-Find / Disjoint set
    parent = {n: n for n in real_nodes}
    rank = dict.fromkeys(real_nodes, 0)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Track adjacency for classification
    neighbors: dict[int, set[int]] = {n: set() for n in real_nodes}

    # Build components among non-root nodes only
    for u, v in edges:
        if u == root and v in real_nodes:
            neighbors[v].add(root)
        elif v == root and u in real_nodes:
            neighbors[u].add(root)
        elif u in real_nodes and v in real_nodes:
            neighbors[u].add(v)
            neighbors[v].add(u)
            union(u, v)

    # Gather connected components among real nodes
    components_map = defaultdict(list)
    for n in real_nodes:
        components_map[find(n)].append(n)

    components = [sorted(comp) for comp in components_map.values()]
    components.sort(key=lambda c: (len(c), c))

    islands = []
    orphans = []
    isolated = []

    for comp in components:
        if len(comp) > 1:
            islands.append(comp)
        else:
            n = comp[0]
            nbrs = neighbors[n]
            if nbrs == {root}:
                orphans.append(n)
            elif len(nbrs) == 0:
                isolated.append(n)

    return {
        "islands": islands,
        "orphans": sorted(orphans),
        "isolated": sorted(isolated),
        "components": components,
    }


def draw_graph(nodes, edges, root=-1, figsize=(10, 8)):
    # Try to import networkx and matplotlib, and skip drawing if not available
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("networkx and matplotlib are required for drawing the graph.")
        return None

    info = parse_graph(nodes, edges, root=root)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Root in the center tends to look good for this kind of structure
    k = 6.0 / (len(G.nodes) ** 0.5)  # more spread
    pos = nx.spring_layout(G, seed=42, k=k, iterations=100, scale=3.0)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.shell_layout(G)

    island_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    node_color_map = {}
    node_size_map = {}

    # Default styling
    for n in G.nodes:
        node_color_map[n] = "lightgray"
        node_size_map[n] = 500

    # Root styling
    if root in G:
        node_color_map[root] = "black"
        node_size_map[root] = 900

    # Color each island differently
    for i, island in enumerate(info["islands"]):
        color = island_colors[i % len(island_colors)]
        for n in island:
            node_color_map[n] = color
            node_size_map[n] = 700

    # Orphans
    for n in info["orphans"]:
        node_color_map[n] = "gold"
        node_size_map[n] = 700

    # Isolated
    for n in info["isolated"]:
        node_color_map[n] = "white"
        node_size_map[n] = 700

    node_colors = [node_color_map[n] for n in G.nodes]
    node_sizes = [node_size_map[n] for n in G.nodes]

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(G, pos, width=1.5, edge_color="0.6")
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Legend
    import matplotlib.patches as mpatches

    legend_items = [
        mpatches.Patch(color="black", label="root (-1)"),
        mpatches.Patch(color="gold", label="orphan"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="isolated"),
        mpatches.Patch(color="lightgray", label="other / default"),
    ]

    for i, _island in enumerate(info["islands"][: len(island_colors)]):
        legend_items.append(mpatches.Patch(color=island_colors[i], label=f"island {i}"))

    plt.legend(handles=legend_items, loc="best")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return info


###
# Types
###


@dataclass
class TopologyDescriptor(Descriptor):
    """
    A container to describe a single rigid-body topology present in a world.

    Used to detect a body-join topology when an explicitly defined articulation is not provided.

    A `topology` represents a collection of bodies connected by joints
    that captures their connectivity in the form of a graph. Concretely,
    it is defined by two ordered sets of indices: a set of `bodies` and
    a set of `joints`, that refer to the corresponding entities in the
    world. The bodies are thus the nodes and the joints are the edges.

    Within the context of mechanical systems, and robotics in particular, a
    topology represents a generalization of an `articulation`, which, is a
    specific type of topology where the graph reduces to a tree structure
    (i.e., without containing loops), a.k.a. a kinematic tree, [1].

    ...

    See [1] for more details on the concept of topology and its relevance to rigid-body dynamics algorithms.
    See [2] for a concise introduction on branch-induced sparsity in rigid-body dynamics.

    Spanning Trees:
    - N body nodes, M joint edges, N - 1 arcs (i.e. tree joints), M - (N - 1) chords (i.e. loop joints)
    - The importance of the one-chord cycles is that they identify the set of independent kinematic loops
    - The number of independent kinematic loops is `L = M - N`

    Numbering Scheme:
    - The root node is numbered `0`
    - All other nodes are numbered `1` to `N` in any order such that each node has a higher number than its parent
    - Edges are numbered such that edge `i` connects node `i` to its parent node
    - Bodies and joints have the same numbers as their corresponding nodes and edges, respectively

    Numbering Scheme (i.e. regular numbering):
    1. Choose a spanning tree, `Gt`.
    2. Assign the number 0 to the node representing the fixed base, and define this node to be the root node of `Gt`.
    3. Number the remaining nodes from 1 to `N` in any order such that each node has a higher number than its parent in `Gt`.
    4. Number the arcs in `Gt` from 1 to `N` such that arc `i` connects between node `i` and its parent.
    5. Number all remaining arcs from `N + 1` to `M` in any order.
    6. Each body gets the same number as its node, and each joint gets the same number as its arc.

    References:
    - [1] Featherstone, Roy. Rigid body dynamics algorithms. Boston, MA: Springer US, 2008.
    - [2] Roy Featherstone, Branch-Induced Sparsity in Rigid-Body Dynamics
          https://royfeatherstone.org/talks/BISinRBD.pdf
    """

    ###
    # Input Attributes
    ###

    floating: bool | None = None
    """
    Whether the topology has a floating or fixed base body.

    This will result cause an automatic detection of whether the root body is connected to the
    world body with index `-1`, and the appropriate joint will be added if not already present.

    Specifically:
    - If `True`, a :attr:`JointDoFType.FREE` joint will be added.
    - If `False`, a :attr:`JointDoFType.FIXED` joint will be added.
    """

    bodies: list[RigidBodyDescriptor] | None = None
    """
    List of body descriptors belonging to the topology.\n
    `N := len(bodies)` is the number of bodies in the topology.\n
    """

    joints: list[JointDescriptor] | None = None
    """
    List of joint descriptors belonging to the topology.\n
    `M := len(joints)` is the number of joints in the topology.\n
    """

    ###
    # Derived Attributes
    ###

    _nodes: np.ndarray | None = None
    """
    List of body node indices defining the topology.\n
    `len(nodes) = len(bodies) = N` is the number of nodes in the topology.\n
    Defaults to `None`, indicating that nodes have not been generated.
    """

    _edges: np.ndarray | None = None
    """
    List of joint edge indices defining the topology.\n
    `len(edges) = len(joints) = M` is the number of edges in the topology.\n
    Defaults to `None`, indicating that edges have not been generated.
    """

    _arcs: np.ndarray | None = None
    """
    List of joint arc indices in the topology.\n
    Arcs are the subset of `edges` that form a spanning tree of the topology.\n
    `len(arcs) = N - 1 <= M = len(joints)` is the number of arcs in the topology.\n
    Defaults to `None`, indicating that arcs have not been generated.
    """

    _chords: np.ndarray | None = None
    """
    List of joint chord indices in the topology.\n
    Chords are the subset of `edges` that do not form a part of the spanning tree of the topology.\n
    `len(chords) = M - (N - 1)` is the number of chords in the topology.\n
    Defaults to `None`, indicating that chords have not been generated.
    """

    _parent: np.ndarray | None = None
    """
    Array of per-joint parent body indices.\n
    Generated from the list of joints and the chosen spanning tree of the topology.\n
    Shape of `(M,)`, where `M` is the number of joints in the topology.\n
    Defaults to `None`, indicating that parent nodes have not been generated.
    """

    _child: np.ndarray | None = None
    """
    Array of per-joint child body indices.\n
    Generated from the list of joints and the chosen spanning tree of the topology.\n
    Shape of `(M,)`, where `M` is the number of joints in the topology.\n
    Defaults to `None`, indicating that child nodes have not been generated.
    """

    _lambdas: np.ndarray | None = None
    """
    The "parent array" containing the parent of each non-root node.\n
    Defines both the connectivity and numbering scheme of the topology.

    Satisfies the following fundamental property:
    ```
    lambdas[i] = min(parent(i), child(i)), and
    0 <= lambdas[i] < i, for all i in [1, N],
    for all i in [1, N]
    ```

    Shape of `(N,)`, where `N` is the number of bodies in the topology.\n
    Defaults to `None`, indicating that parent nodes have not been generated.
    """

    ###
    # Metadata - to be set by the WorldDescriptor when added
    ###

    wid: int = -1
    """
    Index of the world to which the topology belongs.\n
    Defaults to `-1`, indicating that the topology has not yet been added to a world.
    """

    tid: int = -1
    """
    Index of the topology w.r.t. its world.\n
    Defaults to `-1`, indicating that the topology has not yet been added to a world.
    """

    ###
    # Topological Queries
    ###

    # TODO: ????
    def lambda_of(self, i: int) -> int | None:
        """
        Returns the parent of node `i`, i.e.:\n
        `lambda(i)`: the parent of node `i`.
        """
        if self._lambdas:
            return self._lambdas[i]
        return None

    def kappa(self, i: int) -> list[int] | None:
        """
        Returns the "support set" of node `i`: a list containing all joints existing between node `i` and the root, i.e.:\n
        `kappa(i)`: all joints between node `i` and the root.
        """
        return None

    def mu(self, i: int) -> list[int] | None:
        """
        Returns the "child set" of node `i`: a list containing all children of node `i`, i.e.:\n
        `mu(i)`: the children of node `i`.
        """
        return None

    def nu(self, i: int) -> list[int] | None:
        """
        Returns the "subtree set" of node `i`: a list containing all descendant bodies of node `i`, i.e.:\n
        `nu(i)`: all the bodies beyond joint `i`.
        """
        return None

    ###
    # Internals
    ###

    def __post_init__(self):
        # First call the parent method to perform
        # necessary validation in the base class
        super().__post_init__()

        # Retrieve the topology size
        num_bodies = len(self.bodies)
        _num_joints = len(self.joints)

        # Validate that the topology has at least one body
        if num_bodies == 0:
            raise ValueError("A topology must contain at least one body.")

        # Pre-processing:
        # 1. Detect connected sub-graphs (i.e. islands) in the topology
        # 2.
        # 2. Find which bodies are not children of any joint (i.e. orphans), and add a FREE joint to connect them to the world body with index `-1`
        # 3.
        # 4.

        ###
        # Criteria for roots:
        #   - No parent joint (i.e. no incoming edge)
        #   - Connected to the world body with index `-1` through a world joint
        #   -
        #
        #
        #
        #
        #
        ###

        # def _joint_to_body(joint: JointDescriptor, body: RigidBodyDescriptor) -> bool:
        #     return joint.bid_B == body.bid or joint.bid_F == body.bid

        # def _joint_to_world(joint: JointDescriptor) -> bool:
        #     return joint.bid_B == -1 or joint.bid_F == -1

        # # Check for explicit world joints, i.e. joints that connect
        # # a body in the topology to the world body with index `-1`
        # world_joints = [joint for joint in self.joints if _joint_to_world(joint)]

        # # Check for explicit root bodies, i.e. bodies that are connected
        # # to the world body with index `-1` through a world joint
        # root_bodies = [body for body in self.bodies if any(_joint_to_body(joint, body) for joint in world_joints)]

        # # If not root bodies are found, we need to:
        # # 1. create a graph
        # # 2. Identify islands, i.e. unconnected subgraphs
        # # 2. find a minimal spanning tree
        # # 3. and add a world joint to connect the root body to the world

        # # Check for orphan bodies, i.e. bodies that are not connected to
        # # any joint, and add FREE joints to connect them to the world
        # orphan_bodies = [body for body in self.bodies if all(not _joint_to_body(joint, body) for joint in self.joints)]
        # for body in orphan_bodies:
        #     world_joint = JointDescriptor(
        #         name=f"world_to_{body.name}",
        #         bid_B=-1,
        #         bid_F=body.bid,
        #         dof_type=JointDoFType.FREE,
        #     )
        #     self.joints.append(world_joint)

        # # Detect world-joints, i.e. joints that connect a body in the topology to the world body with index `-1`
        # world_joints = [joint for joint in self.joints if joint.bid_B == -1 or joint.bid_F == -1]

        # # Determine the expected world-joint type based on the `floating` attribute of the topology
        # world_joint_type = JointDoFType.FREE if self.floating else JointDoFType.FIXED

        # # If the topology has no world-joint, insert one at the start
        # # of the joint list, connecting the root body to the world,
        # # and update the body and joint indices accordingly
        # if len(world_joints) == 0:
        #     root = self.bodies[0]
        #     world_joint = JointDescriptor(
        #         name=f"world_to_{root.name}",
        #         bid_B=-1,
        #         bid_F=root.bid,
        #         dof_type=world_joint_type,
        #     )
        #     self.joints.insert(0, world_joint)
        #     for joint in self.joints[1:]:
        #         joint.jid += 1

    @override
    def __str__(self) -> str:
        """Returns a human-readable string representation of the TopologyDescriptor."""
        return (
            f"TopologyDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"bodies: {self.bodies},\n"
            f"joints: {self.joints},\n"
            f"wid: {self.wid},\n"
            f"tid: {self.tid},\n"
            f")"
        )

    @override
    def __repr__(self) -> str:
        return self.__str__()
