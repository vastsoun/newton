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

from dataclasses import dataclass

import numpy as np
import warp as wp

from ....utils.topology import topological_sort
from .bodies import RigidBodyDescriptor
from .joints import JointDescriptor
from .types import Descriptor, override

###
# Module interface
###

__all__ = [
    "TopologyDescriptor",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class TopologyDescriptor(Descriptor):
    """
    A container to describe a single rigid-body topology present in a world.

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

    bodies: list[RigidBodyDescriptor]
    """
    List of body descriptors belonging to the topology.\n
    `N := len(bodies)` is the number of bodies in the topology.\n
    """

    joints: list[JointDescriptor]
    """
    List of joint descriptors belonging to the topology.\n
    `M := len(joints)` is the number of joints in the topology.\n
    """

    ###
    # Derived Attributes
    ###

    nodes: np.ndarray | None = None
    """
    List of body node indices defining the topology.\n
    `len(nodes) = len(bodies) = N` is the number of nodes in the topology.\n
    Defaults to `None`, indicating that nodes have not been generated.
    """

    edges: np.ndarray | None = None
    """
    List of joint edge indices defining the topology.\n
    `len(edges) = len(joints) = M` is the number of edges in the topology.\n
    Defaults to `None`, indicating that edges have not been generated.
    """

    arcs: np.ndarray | None = None
    """
    List of joint arc indices in the topology.\n
    Arcs are the subset of `edges` that form a spanning tree of the topology.\n
    `len(arcs) = N - 1 <= M = len(joints)` is the number of arcs in the topology.\n
    Defaults to `None`, indicating that arcs have not been generated.
    """

    chords: np.ndarray | None = None
    """
    List of joint chord indices in the topology.\n
    Chords are the subset of `edges` that do not form a part of the spanning tree of the topology.\n
    `len(chords) = M - (N - 1)` is the number of chords in the topology.\n
    Defaults to `None`, indicating that chords have not been generated.
    """

    parent: np.ndarray | None = None
    """
    Array of per-joint parent body indices.\n
    Generated from the list of joints and the chosen spanning tree of the topology.\n
    Shape of `(M,)`, where `M` is the number of joints in the topology.\n
    Defaults to `None`, indicating that parent nodes have not been generated.
    """

    child: np.ndarray | None = None
    """
    Array of per-joint child body indices.\n
    Generated from the list of joints and the chosen spanning tree of the topology.\n
    Shape of `(M,)`, where `M` is the number of joints in the topology.\n
    Defaults to `None`, indicating that child nodes have not been generated.
    """

    lambdas: np.ndarray | None = None
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

    def lamda(self, i: int) -> int | None:
        """
        Returns the parent of node `i`, i.e.:\n
        `lamda(i)`: the parent of node `i`.
        """
        if self.lambdas:
            return self.lambdas[i]
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
    # Operations
    ###

    @override
    def __repr__(self) -> str:
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
