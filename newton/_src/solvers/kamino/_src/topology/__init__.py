# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides primitives for constrained multi-body system topologies.

A `topology` represents a collection of bodies connected by joints
that captures their connectivity in the form of a graph. Concretely,
it is defined by two ordered sets of indices: a set of `bodies` and
a set of `joints` that refer to the corresponding entities in the
world. Bodies thus correspond to nodes of the graph and joints to edges.

Within the context of mechanical systems, and robotics in particular, a
topology represents a generalization of an `articulation`, which, is a
specific type of topology where the graph reduces to a topological tree
(i.e. without loops), a.k.a. a kinematic tree [1].

See [1] for more details on the topic of topology and its relevance to rigid-body dynamics algorithms.
See [2] for a concise introduction to branch-induced sparsity in rigid-body dynamics.

References:
- [1] Featherstone, Roy. Rigid body dynamics algorithms. Boston, MA: Springer US, 2008.
- [2] Roy Featherstone, Branch-Induced Sparsity in Rigid-Body Dynamics
      https://royfeatherstone.org/talks/BISinRBD.pdf
"""

from .graph import TopologyComponentParser, TopologyGraph
from .render import TopologyGraphVisualizer
from .selectors import (
    TopologyHeaviestBodyBaseSelector,
    TopologyMinimumDepthSpanningTreeSelector,
)
from .trees import TopologyMinimumDepthSpanningTreeGenerator
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    ComponentConnectivity,
    ComponentType,
    EdgeType,
    NodeType,
    OrientedEdge,
    SpanningTreeTraversal,
    TopologyComponent,
    TopologyComponentBaseSelectorBase,
    TopologyComponentParserBase,
    TopologyGraphVisualizerBase,
    TopologySpanningTree,
    TopologySpanningTreeGeneratorBase,
    TopologySpanningTreeSelectorBase,
)

###
# Module interface
###

__all__ = [
    "DEFAULT_WORLD_NODE_INDEX",
    "ComponentConnectivity",
    "ComponentType",
    "EdgeType",
    "NodeType",
    "OrientedEdge",
    "SpanningTreeTraversal",
    "TopologyComponent",
    "TopologyComponentBaseSelectorBase",
    "TopologyComponentParser",
    "TopologyComponentParserBase",
    "TopologyGraph",
    "TopologyGraphVisualizer",
    "TopologyGraphVisualizerBase",
    "TopologyHeaviestBodyBaseSelector",
    "TopologyMinimumDepthSpanningTreeGenerator",
    "TopologyMinimumDepthSpanningTreeSelector",
    "TopologySpanningTree",
    "TopologySpanningTreeGeneratorBase",
    "TopologySpanningTreeSelectorBase",
]
