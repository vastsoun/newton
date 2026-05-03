# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###
# IMPLEMENTATIONS:
#   1. Implement a simple subgraph component base/edge selector back-end that assigns the base node/edge
#      to the heaviest moving body node as a first example of a TopologyComponent generation back-end.
#      -----
#      [DONE] Subgraph component base/edge selector that assigns the base node/edge to the
#      heaviest moving body node — see :class:`TopologyHeaviestBodyBaseSelector` in :mod:`.selectors`.
#
#   ------------------------------------------------------------------------------------------------
#   2. Implement a minimum depth spanning tree generation back-end as a first example:
#       2a. Implement a brute-force minimum-depth spanning tree generation function that
#           generates all possible minimum-depth spanning trees of the component subgraph.
#       2b. Implement a function that generates all minimum-depth spanning trees starting from a given root node.
#       2c. Implement a function that computes the degree of each node in the component subgraph
#       2d. Implement a function that generates all minimum-depth spanning trees of the component subgraph by:
#           - first checking if a base node/edge is defined for the corresponding component, and use 2b to generate all spanning trees from the base node as root.
#           - if the component doesn't define a base node/edge, but has grounding nodes/edges, generate all minimum-depth spanning trees with those as root nodes using 2b.
#           - if no base or grounding nodes are present the generate all spanning trees setting the the node with the highest degree using 2c.
#           - in case of ties in degree, use the brute-force method of 2a. and generate all possible spanning trees.
#           - this function should accept arguments for:
#               - the spanning tree traversal mode (e.g. DFS, BFS)
#               - override all prioritization rules and just generate all possible spanning trees, if admissible.
#               - direct specification of the root node/edge for spanning tree generation, if admissible, and set that as the base node/edge in the source component.
#               - accept a maximum number of spanning trees to generate, and stop the generation process once that number is reached, if admissible.
#               - whether to prioritize balanced/symmetric trees over unbalanced ones, if admissible.
#      -----
#      [DONE] Minimum-depth spanning-tree generator — see
#      :class:`TopologyMinimumDepthSpanningTreeGenerator` in :mod:`.trees`. Implements:
#       2a. Brute-force minimum-depth spanning-tree generation (priority-cascade
#           fallback when ``override_priorities=True`` or on degree ties).
#       2b. Per-root minimum-depth spanning-tree enumeration.
#       2c. Per-node degree computation on the body-only subgraph.
#       2d. Priority cascade over (explicit roots → override → base → grounding →
#           max-degree → brute-force) with kwargs for traversal mode,
#           override, explicit roots, max-candidate cap, and balanced-tree
#           prioritization.
#
#
#   ------------------------------------------------------------------------------------------------
#   3. Implement a simple heuristic spanning tree selection back-end as a first example:
#       - For islands, select the spanning tree by:
#           - ordering based on tree depth, and selecting the one with minimum depth, if there are no ties.
#           - prioritize balanced/symmetric subtrees over unbalanced ones, if admissible.
#           - In case of remaining ties, just select and return the first candidate in the list.
#       - For orphans, select the trivial spanning tree with no edges.
#      -----
#      [DONE] Heuristic spanning-tree selector — see
#      :class:`TopologyMinimumDepthSpanningTreeSelector` in :mod:`.selectors`. Picks the
#      minimum-depth candidate (with optional balanced-tree tie-breaker) for
#      islands and the trivial candidate for orphans.
#
#   ------------------------------------------------------------------------------------------------
#   4. Implement a mechanism (class or function) for body/joint index reassignment to optimize
#      for better data locality and satisfaction of Featherstone's regular numbering rules.
#       4a. It should take as input a TopologyGraph, performing a copy of the original and operate on that
#       4b. The components and trees attributes should be re-ordered according to the number of nodes + number of edges
#       4c. Construct two lists to hold index re-mappings using an appropriate container: dict or list of tuples, etc.
#       4c. It should first reorder nodes/edges to exactly group them according to their component membership
#       4d. Then, it should reorder nodes/edges within each component according to their spanning tree membership, i.e. first the tree arcs, then the chords.
#       4e. Then, it should reorder nodes/edges within each tree according to the traversal mode of the original tree starting from the root node, if defined, or the node with the highest degree otherwise.
#           Following Featherstone's regular numbering rules, i.e. for each joint edge (i, j), where i is the predecessor body and j is the successor body, it should ensure that min(i, j) < max(i, j) and that the parent of body j is body i.
#       4f. It should generate and return:
#       - a new TopologyGraph with re-ordered components and spanning trees, as well as updated node and edge indices according to the new body/joint ordering.
#       - a list of body node index reassignments, where the entry at index `i` gives the new body index assigned to the body with original index `i`.
#       - a list of joint edge index reassignments, where the entry at index `j` gives the new joint index assigned to the joint with original index `j`.
#
#   ------------------------------------------------------------------------------------------------
#   Synthetic-edge index handoff:
#       Selectors that synthesize a 6-DoF FREE base edge (because no user-supplied grounding exists)
#       initialize it with ``joint_index = NO_BASE_JOINT_INDEX`` (-1). The orchestrator detects the
#       sentinel inside :meth:`TopologyGraph._commit_base_edge` and re-issues the edge with a fresh
#       provisional joint index of the form ``NJ + k``, where ``NJ`` is the count of user-supplied
#       edges captured at construction time (:attr:`TopologyGraph._original_num_edges`) and ``k``
#       is the count of synthetic edges committed so far. Synthetic edges are exposed verbatim via
#       :attr:`TopologyGraph.new_base_edges` so the downstream model builder knows which joints
#       still need to be added to its model. After index reassignment, the final joint index for
#       each synthetic edge is ``self.joint_edge_remap[edge.joint_index]``.
#
#   -------------------------------------------------------------------------------------------------
#   Next steps:
#       1. Implement a utility function that takes a source USD asset and creates a variant named `*_articulated`
#          with `uniform bool physics:excludeFromArticulation = 1`` added to the corresponding chord joints.
#       2. Implement a mechanism for tree selection that prioritizes trees with more equal subtree scores:
#          Add another (optional) tie-breaker heuristic to spanning tree selection that also assigns to each edge a value based on the effective lever arm length
#          defined by the joint transforms (i.e. total distance from predecessor frame to successor frame). The values of each edge are used to compute
#          an accumulated score on each parent node computed from its subtree edges (first backward pass). Then a forward pass starting at the root,
#          prioritizes trees with more equal subtree scores.
#
###

"""
Provides a modular pipeline for topology discovery and spanning tree generation.

Ships the user-facing :class:`TopologyGraph` container and the default
:class:`TopologyComponentParser` (a union-find based component grouper).
:class:`TopologyGraph` orchestrates the topology discovery pipeline:
component parsing, base/grounding selection, spanning-tree candidate
generation, and per-component spanning-tree selection. Also provides
a default graph component parser in :class:`TopologyComponentParser`.

See :mod:`.types` for schema definitions (:class:`GraphEdge`,
:class:`GraphNode`, :class:`TopologyComponent`,
:class:`TopologySpanningTree`, and the abstract module bases) and
:mod:`.trees` for shipped spanning-tree generator back-ends.

----
Topology Discovery Process

1. Parse each world's lists of :class:`RigidBodyDescriptor` and
   :class:`JointDescriptor` to generate lists of :data:`NodeType` and
   :data:`EdgeType`, and use them to create a :class:`TopologyGraph`.

2. Each :class:`TopologyGraph` parses its nodes and edges to generate its constituent
   components, i.e. subgraphs, using the configured component parser back-end:
   2a. Each component is classified as an island or an orphan, and as connected or isolated.
   2b. For each component, assign a base node/edge if there is a single grounding node/edge,
       or if multiple grounding edges are present but only one of them is a 6-DoF FREE joint,
       take that. If a base node/edge is assigned, remove it from the grounding lists. If
       multiple grounding edges are present and more than one of them are 6-DoF FREE joints,
       raise an error. If no base node/edge can be assigned, leave them as ``None`` for the
       next steps.

3. For each isolated component, run the configured base selector back-end:
   3a. For each isolated orphan, synthesize a 6-DoF FREE joint connecting
       the orphan node to the world node and assign it as a base edge.
   3b. For each isolated island, the base selector picks a base node
       and edge based on the contents of the :class:`RigidBodyDescriptor`
       and :class:`JointDescriptor` lists of the world, if provided, and
       synthesize a 6-DoF FREE joint connecting the orphan node to the
       world node.
   3c. All synthetic base edges are assigned the joint index ``-1`` to
       flag and cached within the component object for later reference,
       and construction by consumers such as the model-builder.

4. For each component, generate a list of :class:`TopologySpanningTree`
   candidates using the configured spanning tree generator back-end:
   4a. If a base node/edge is present, use it as the unique root.
   4b. Otherwise, if the component has grounding nodes/edges, use them as roots.
   4c. Otherwise, brute-force enumerate all possible roots.
   4d. The :data:`SpanningTreeTraversal` (e.g. DFS, BFS) argument controls body ordering.

5. For each component, select a single spanning tree from the list of
   candidates using the modular spanning tree selection back-end,
   taking as inputs the list of spanning tree candidates, as well as
   the RigidBodyDescriptor and JointDescriptor lists of the world. The
   latter are forwarded to the selector for context, if provided, and
   whose inertial and geometric properties are used to inform the
   selection process, e.g. by analyzing branch induced sparsity, branch
   balance, branch depth, and kinematic/dynamic singularities etc.

6. For each spanning tree, check the parent array against Featherstone's
   regular numbering rules and, if not satisfied, perform index reassignment
   of the bodies and joints in the TopologyGraph instance accordingly, using
   the modular index reassignment mechanism.

7. Generate a :class:`TopologyDescriptor` from the selected
   :class:`TopologySpanningTree` instance of each component.

8. Add the descriptor to the model builder, and assign it
   to the corresponding bodies and joints in the world.

9. Generate the :class:`TopologyModel` from the descriptors.

10. (Optional) Perform index reassignment of the bodies and joints in
    the graph to optimize for better data locality and satisfaction of
    Featherstone's regular numbering rules, by updating the
    TopologyGraph instance accordingly.

"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from ..core.bodies import RigidBodyDescriptor
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override
from ..utils import logger as msg
from .render import TopologyGraphVisualizer
from .selectors import (
    TopologyHeaviestBodyBaseSelector,
    TopologyMinimumDepthSpanningTreeSelector,
)
from .trees import TopologyMinimumDepthSpanningTreeGenerator
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    NO_BASE_JOINT_INDEX,
    UNASSIGNED_JOINT_TYPE,
    EdgeType,
    GraphEdge,
    GraphLabels,
    GraphNode,
    NodeType,
    SpanningTreeTraversal,
    TopologyComponent,
    TopologyComponentBaseSelectorBase,
    TopologyComponentParserBase,
    TopologyGraphVisualizerBase,
    TopologyIndexReassignmentBase,
    TopologySpanningTree,
    TopologySpanningTreeGeneratorBase,
    TopologySpanningTreeSelectorBase,
    validate_max_candidates,
    validate_traversal_mode,
)

###
# Module interface
###

__all__ = [
    "TopologyComponentParser",
    "TopologyGraph",
]

###
# Interfaces
###


class TopologyGraph:
    """Container to represent a topological undirected graph `G`.

    Holds the body nodes and joint edges of a graph plus the modular
    pipeline (component parser, base selector, tree generator, tree
    selector, visualizer) used to derive its components and spanning
    trees.
    """

    def __init__(
        self,
        # Graph attributes
        nodes: list[NodeType],
        edges: list[EdgeType] | None = None,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        # Pipeline modules
        component_parser: TopologyComponentParserBase | None = None,
        base_selector: TopologyComponentBaseSelectorBase | None = None,
        tree_generator: TopologySpanningTreeGeneratorBase | None = None,
        tree_selector: TopologySpanningTreeSelectorBase | None = None,
        index_reassigner: TopologyIndexReassignmentBase | None = None,
        graph_visualizer: TopologyGraphVisualizerBase | None = None,
        # Source model hints
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        bases: list[NodeType] | None = None,
        # Parsing configurations
        tree_traversal_mode: SpanningTreeTraversal = "dfs",
        max_tree_candidates: int = 32,
        override_priorities: bool = False,
        prioritize_balanced: bool = False,
        reassign_indices_inplace: bool = False,
        autoparse: bool = False,
    ):
        """Initialize the graph with nodes, edges, and optional pipeline modules.

        Args:
            nodes: List of body node indices.
            edges: List of joint edges in :data:`EdgeType` form (each entry
                is a :class:`GraphEdge` or a 3-tuple
                ``(joint_type, joint_index, (u, v))``); coerced to
                :class:`GraphEdge` at construction.
            world_node: Index of the implicit world node (must be negative
                and not in ``nodes``).
            component_parser: Module that parses nodes/edges into components;
                defaults to a shipped :class:`TopologyComponentParser`.
            base_selector: Module that selects a base for components without
                an auto-assigned base; required only if any parsed component
                lacks a base.
            tree_generator: Module that generates spanning-tree candidates
                per component.
            tree_selector: Module that selects one spanning tree per
                component from the candidate list.
            index_reassigner: Module that performs body/joint index
                reassignment for better locality and regularity.
            graph_visualizer: Module used by the ``render_*`` methods;
                defaults to a shipped :class:`TopologyGraphVisualizer`.
            bodies: Optional body descriptors forwarded to the base/tree
                selectors; can also be passed to :meth:`parse`.
            joints: Optional joint descriptors forwarded to the base/tree
                selectors; can also be passed to :meth:`parse`.
            tree_traversal_mode: Default traversal mode (``"dfs"`` or
                ``"bfs"``) used by :meth:`generate_spanning_trees`.
            max_tree_candidates: Default upper bound on candidate spanning
                trees per component.
            autoparse: If ``True``, run the full :meth:`parse` pipeline
                immediately after construction.

        Raises:
            ValueError: If any node has a negative index, ``nodes``
                contains duplicate body indices, any edge has an invalid
                format, an edge references a non-world body index that
                is not present in ``nodes``, two edges share a
                ``joint_index`` with conflicting ``joint_type`` or
                body-pair (including polarity-swapped duplicates),
                ``world_node`` is non-negative or contained in ``nodes``,
                ``tree_traversal_mode`` is not supported, or
                ``max_tree_candidates`` is non-positive.
            TypeError: If a node is not an :class:`int` or
                :class:`GraphNode`, an edge is not a :class:`GraphEdge`
                or a 3-tuple, ``world_node`` is not an integer, or
                ``max_tree_candidates`` is not an :class:`int`.
        """
        self._nodes: list[GraphNode] = [GraphNode.from_input(n) for n in nodes]
        """
        List of body nodes contained in the graph. Each node is uniquely identified by its
        associated index in the range ``[0, NB-1]``, where ``NB`` is the total number of body
        nodes in the graph. ``NB`` excludes the implicit world node with index ``-1``, which
        is present in the graph if any moving body node is connected to it.
        """
        # ``edges=None`` is treated as an empty graph (isolated body nodes only).
        self._edges: list[GraphEdge] = [GraphEdge.from_input(e) for e in edges] if edges is not None else []
        """
        List of joint edges contained in the graph. Each edge is uniquely identified by its
        associated index in the range ``[0, NJ-1]``, where ``NJ`` is the total number of joint
        edges. ``NJ`` excludes the implicit world node with index ``-1``, which is present in
        the graph if any joint edge is connected to it.
        """
        self._world_node: int = world_node
        """Index of the implicit world node (defaults to ``-1``)."""

        # Cache parsing configurations
        self._tree_traversal_mode: SpanningTreeTraversal = tree_traversal_mode
        """Traversal mode used for spanning-tree generation."""
        self._max_tree_candidates: int = max_tree_candidates
        """Maximum number of candidate spanning trees per component."""
        self._override_priorities: bool = override_priorities
        """Whether to override all prioritization rules and just generate all possible spanning trees."""
        self._prioritize_balanced: bool = prioritize_balanced
        """Whether to prioritize balanced/symmetric trees over unbalanced ones when selecting among candidates."""
        self._reassign_indices_inplace: bool = reassign_indices_inplace
        """Whether to also perform index reassignment in-place (``True``)
        as well as return the necessary index mappings."""

        # Validate the input graph attributes to ensure they are
        # consistent with the expected formats and conventions
        self._validate_inputs()

        # Store input modules for component parsing and spanning tree generation
        self._component_parser: TopologyComponentParserBase | None = component_parser
        """Module that parses graph nodes/edges into components."""
        self._base_selector: TopologyComponentBaseSelectorBase | None = base_selector
        """Module that selects the base node/edge for each component."""
        self._tree_generator: TopologySpanningTreeGeneratorBase | None = tree_generator
        """Module that generates spanning-tree candidates per component."""
        self._tree_selector: TopologySpanningTreeSelectorBase | None = tree_selector
        """Module that selects the best spanning tree from a candidate list."""
        self._index_reassigner: TopologyIndexReassignmentBase | None = index_reassigner
        """Module that performs body/joint index reassignment for better locality and regularity."""
        self._graph_visualizer: TopologyGraphVisualizerBase | None = graph_visualizer
        """Module that renders the graph, components, and spanning trees."""

        # Set default modules where shipped concrete defaults exist
        if self._component_parser is None:
            self._component_parser = TopologyComponentParser()
        if self._base_selector is None:
            self._base_selector = TopologyHeaviestBodyBaseSelector()
        if self._tree_generator is None:
            self._tree_generator = TopologyMinimumDepthSpanningTreeGenerator()
        if self._tree_selector is None:
            self._tree_selector = TopologyMinimumDepthSpanningTreeSelector()
        if self._index_reassigner is None:
            self._index_reassigner = TopologyIndexReassignment()
        if self._graph_visualizer is None:
            self._graph_visualizer = TopologyGraphVisualizer()

        # Declare and initialize internal caches for the source model descriptors
        self._bases: list[NodeType] | None = bases
        self._bodies: list[RigidBodyDescriptor] | None = bodies
        self._joints: list[JointDescriptor] | None = joints

        # Declare derived attributes
        self._components: list[TopologyComponent] | None = None
        """Parsed component subgraphs of the topology graph."""
        self._candidates: list[list[TopologySpanningTree]] | None = None
        """Candidate spanning trees per component."""
        self._trees: list[TopologySpanningTree] | None = None
        """Selected spanning tree per component."""
        self._trees_remapped: list[TopologySpanningTree] | None = None
        """Per-component spanning trees with indices rewritten through the reassignment remap."""
        self._new_base_edges: list[GraphEdge] | None = None
        """New edges that should be added to connect isolated components."""
        self._body_node_remap: list[int] | None = None
        """Index reassignment list to map original to new body indices."""
        self._joint_edge_remap: list[int] | None = None
        """Index reassignment list to map original to new joint indices."""

        # Snapshot the count of user-supplied edges before the pipeline starts mutating
        # ``self._edges`` with synthetic base edges. This is the ``NJ`` in ``NJ + k`` —
        # the provisional joint index handed to the ``k``-th synthetic base edge so that
        # downstream consumers (and the reassignment back-end) can disambiguate them.
        self._original_num_edges: int = len(self._edges)
        """Number of user-supplied joint edges captured at construction time."""

        # If `autoparse` is True, automatically parse the graph nodes
        # and edges into components and generate spanning trees
        if autoparse:
            self.parse()

    ###
    # Properties
    ###

    @property
    def nodes(self) -> list[GraphNode]:
        """Returns the list of body nodes in the graph."""
        return self._nodes

    @property
    def edges(self) -> list[GraphEdge]:
        """Returns the list of joint edges in the graph (empty if none)."""
        return self._edges

    @property
    def world_node(self) -> int:
        """Returns the index of the implicit world node."""
        return self._world_node

    @property
    def components(self) -> list[TopologyComponent]:
        """Returns the list of parsed components.

        Raises:
            ValueError: If components have not been parsed yet.
        """
        if self._components is None:
            raise ValueError("Graph components have not been parsed yet.")
        return self._components

    @property
    def candidates(self) -> list[list[TopologySpanningTree]]:
        """Returns the per-component lists of candidate spanning trees.

        Raises:
            ValueError: If candidates have not been generated yet.
        """
        if self._candidates is None:
            raise ValueError("Candidate spanning trees have not been generated yet.")
        return self._candidates

    @property
    def trees(self) -> list[TopologySpanningTree]:
        """Returns the per-component selected spanning trees.

        Raises:
            ValueError: If spanning trees have not been selected yet.
        """
        if self._trees is None:
            raise ValueError("Spanning trees have not been selected yet.")
        return self._trees

    @property
    def new_base_edges(self) -> list[GraphEdge] | None:
        """
        Returns the list of new base edges that should be added to connect isolated components.

        The length of the list depends on the number of isolated components in the
        graph, which can be less than or equal to the total number of components.

        If ``None``, then no new base edges are specified, indicating that
        all components in the graph are already connected to the world node.
        """
        return self._new_base_edges

    @property
    def body_node_remap(self) -> list[int] | None:
        """
        Returns the list of body node index reassignments for remapping original to new indices.

        The length of the list depends on the number of body nodes that need to be reassigned,
        which can be less than or equal to the total number of body nodes in the graph.

        If ``None``, no body index reassignment is needed by the graph topology.
        """
        return self._body_node_remap

    @property
    def joint_edge_remap(self) -> list[int] | None:
        """
        Returns the list of joint edge index reassignments for remapping original to new indices.

        The length of the list depends on the number of joint edges that need to be reassigned,
        which can be less than or equal to the total number of joint edges in the graph.

        If ``None``, no joint index reassignment is needed by the graph topology.
        """
        return self._joint_edge_remap

    @property
    def trees_remapped(self) -> list[TopologySpanningTree] | None:
        """
        Returns the list of spanning trees re-mapped to the optimized indices in the prototype graph.

        When :meth:`compute_index_reassignment` has been run with
        ``inplace=True``, the live :attr:`trees` already carry the remapped
        indices and are returned as-is (no copy). Otherwise the per-tree
        remapped copies are materialized once and cached.

        Raises:
            ValueError: If spanning trees have not been selected yet, or if
                the index reassignment has not been computed yet.
        """
        if self._trees is None:
            raise ValueError("Spanning trees have not been selected yet.")
        if self._body_node_remap is None and self._joint_edge_remap is None:
            raise ValueError("Index reassignment has not been computed yet.")
        if self._reassign_indices_inplace:
            return self._trees
        if self._trees_remapped is None:
            self._trees_remapped = [
                tree.remapped(self._body_node_remap, self._joint_edge_remap) for tree in self._trees
            ]
        return self._trees_remapped

    ###
    # Operations
    ###

    def parse(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        bases: list[NodeType] | None = None,
        tree_traversal_mode: SpanningTreeTraversal | None = None,
        max_tree_candidates: int | None = None,
        override_priorities: bool | None = None,
        prioritize_balanced: bool | None = None,
        reassign_indices_inplace: bool | None = None,
    ) -> None:
        """Run the full topology-discovery pipeline end-to-end.

        Args:
            bodies:
                Optional body descriptors forwarded to the base/tree
                selectors; falls back to the descriptors supplied at
                construction time when omitted.
            joints:
                Optional joint descriptors forwarded to the base/tree
                selectors; falls back to the descriptors supplied at
                construction time when omitted.
            bases:
                Optional base node/edge indices forwarded to the
                base/tree selectors; falls back to the indices
                supplied at construction time when omitted.

        Raises:
            ValueError: If any module required by the full pipeline is
                missing, parsing or generation fails, or the graph
                attributes are invalid.
        """
        # Validate up front that every module required by the full pipeline is available,
        # so that the user can fix them in one round instead of failing in step N of M.
        missing = [
            name
            for name, mod in (
                ("component_parser", self._component_parser),
                ("base_selector", self._base_selector),
                ("tree_generator", self._tree_generator),
                ("tree_selector", self._tree_selector),
                ("index_reassigner", self._index_reassigner),
            )
            if mod is None
        ]
        if missing:
            raise ValueError(
                f"Cannot run full topology parsing pipeline: missing required module(s): "
                f"{', '.join(missing)}. Provide them via the `TopologyGraph` constructor."
            )

        # Use the provided body and joint descriptors for parsing if given,
        # otherwise use the cached descriptors from initialization
        _bodies = bodies if bodies is not None else self._bodies
        _joints = joints if joints is not None else self._joints
        _bases = bases if bases is not None else self._bases

        # Check argument overrides for parsing configurations, and fall back to the cached values from initialization if not provided
        _tree_traversal_mode = tree_traversal_mode if tree_traversal_mode is not None else self._tree_traversal_mode
        _max_tree_candidates = max_tree_candidates if max_tree_candidates is not None else self._max_tree_candidates
        _override_priorities = override_priorities if override_priorities is not None else self._override_priorities
        _prioritize_balanced = prioritize_balanced if prioritize_balanced is not None else self._prioritize_balanced
        _reassign_indices_inplace = (
            reassign_indices_inplace if reassign_indices_inplace is not None else self._reassign_indices_inplace
        )

        # Parse the graph nodes and edges into components, and auto-assign
        # base nodes/edges where possible based on the discovery logic.
        self.parse_components()

        # For any remaining components that still lack a base
        # node/edge after parsing, invoke base selection
        self.select_component_bases(bodies=_bodies, joints=_joints, bases=_bases)

        # Generate candidate spanning trees for each component, and select one
        # per component using the configured generator and selector modules.
        self.generate_spanning_trees(
            traversal_mode=_tree_traversal_mode,
            max_candidates=_max_tree_candidates,
            roots=_bases,
            override_priorities=_override_priorities,
            prioritize_balanced=_prioritize_balanced,
        )

        # Perform spanning tree selection for each component using the configured
        # selector module, and cache the selected tree for each component in the graph.
        self.select_spanning_trees(bodies=_bodies, joints=_joints)

        # Reassign indices for the graph nodes and
        # edges based on the selected spanning trees.
        self.compute_index_reassignment(reassign_inplace=_reassign_indices_inplace)

    def render(
        self,
        graph_labels: Iterable[GraphLabels] | None = None,
        edge_label_offset_pts: float | None = None,
        force_path_labels: bool = False,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Render the topology graph, its components, and spanning trees.

        Args:
            graph_labels: Optional :data:`GraphLabels` set selecting
                which name-label variants to render. ``"inline"`` adds
                tiny on-graph annotations beside named nodes/edges;
                ``"tables"`` adds ``index | name`` reference tables
                below the graph. Both can be combined. Modes silently
                no-op when the corresponding descriptor list is missing
                or has no named entries.
            force_path_labels: When ``True``, preserve the full scoped
                name (e.g. ``/world/anymal/LF_HIP``) in inline
                annotations and tables. Defaults to ``False`` so
                USD-style ``/scope/path/leaf`` names are clipped to
                ``…/leaf`` when they exceed the per-label budget,
                keeping dense graphs readable.
            edge_label_offset_pts: Perpendicular distance, in display
                points (1 pt = 1/72 inch — matplotlib's standard
                typographic unit), between each edge and its primary
                ``index_TYPE`` label as well as the matching inline
                joint-name label on the opposite side. ``None``
                (default) uses the visualizer's built-in default.
                Increase to push labels further from their edges;
                decrease to bring them closer (set to ``0.0`` to
                recover NetworkX-style on-edge placement).
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.

        Raises:
            ValueError:
                If no graph visualizer is configured, components have
                not been parsed, candidates have not been generated,
                or spanning trees have not been selected.
        """
        # Ensure that a graph visualizer module is provided,
        # since this is required for any rendering to proceed.
        if self._graph_visualizer is None:
            raise ValueError("No graph visualizer module provided, cannot render topology graph.")
        if self._components is None:
            raise ValueError("Graph components must be generated before rendering.")
        if self._candidates is None:
            raise ValueError("Candidate spanning trees must be generated before rendering.")
        if self._trees is None:
            raise ValueError("Spanning trees must be selected before rendering.")

        # Set the sub-paths to each plot group
        graph_path = self._inject_path_index_suffix(path, "graph")
        candidates_path = self._inject_path_index_suffix(path, "candidates")
        trees_path = self._inject_path_index_suffix(path, "trees")

        # Render the graph, its components, and spanning trees.
        self.render_graph(
            graph_labels=graph_labels,
            edge_label_offset_pts=edge_label_offset_pts,
            force_path_labels=force_path_labels,
            figsize=figsize,
            path=graph_path,
            show=show,
        )
        self.render_spanning_tree_candidates(
            graph_labels=graph_labels,
            edge_label_offset_pts=edge_label_offset_pts,
            force_path_labels=force_path_labels,
            skip_orphans=True,
            figsize=figsize,
            path=candidates_path,
            show=show,
        )
        self.render_spanning_trees(
            graph_labels=graph_labels,
            edge_label_offset_pts=edge_label_offset_pts,
            force_path_labels=force_path_labels,
            skip_orphans=True,
            figsize=figsize,
            path=trees_path,
            show=show,
        )

    ###
    # Step-by-Step Pipeline Operations
    ###

    def parse_components(self) -> list[TopologyComponent]:
        """Parse the graph into a list of components using the configured parser.

        Returns:
            The list of parsed :class:`TopologyComponent` instances.

        Raises:
            ValueError: If no component parser is configured or the parser
                returns ``None``.
        """
        # Ensure that a component parser module is provided,
        # since this is required for any parsing to proceed.
        if self._component_parser is None:
            raise ValueError("No component parser module provided, cannot parse graph components.")

        # Parse the graph nodes and edges into components using the provided component parser module
        self._components = self._component_parser.parse_components(
            nodes=self._nodes, edges=self._edges, world=self._world_node
        )

        # Ensure that the parser returned a valid list
        # of components, and cache it for later access
        if self._components is None:
            raise ValueError("Graph component parsing failed.")

        # Return a reference to the cached components
        # so that users can access them immediately.
        return self._components

    def select_component_bases(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        bases: list[NodeType] | None = None,
    ) -> None:
        """Assign a base node/edge to every component that lacks one.

        Components with a single grounding edge (or a single FREE joint
        among multiple grounding edges) are auto-assigned a base by the
        parser; this method only invokes the base selector for the
        remaining components.

        Args:
            bodies:
                Optional body descriptors forwarded to the selector.
            joints:
                Optional joint descriptors forwarded to the selector.
            bases:
                Optional base nodes forwarded to the selector. If provided,
                this will be used instead of invoking the selector module,
                and the base node/edge will be assigned directly to the
                corresponding component. If a joint edge connecting the base
                node to the world node does not already exist, a new 6-DoF
                FREE joint will be created and assigned as the base edge.
        Raises:
            ValueError: If components have not been parsed, any component
                still lacks a base but no base selector is configured, or
                the selector returns ``None``.
        """
        # Ensure that the graph components are generated before
        # selecting the base node and edge for each component
        if self._components is None:
            raise ValueError("Graph components must be generated before base node/edge selection.")

        # Use the provided body and joint descriptors for parsing if given,
        # otherwise use the cached descriptors from initialization
        _bodies = bodies if bodies is not None else self._bodies
        _joints = joints if joints is not None else self._joints
        _bases = bases if bases is not None else self._bases

        # Determine which components still need a base assignment after parsing
        components_needing_base = [c for c in self._components if c.base_edge is None]
        if not components_needing_base:
            return

        # Check if the user provided explicit base node/edge indices, and if
        # yes attempt to match them to components that need base assignment.
        # NOTE: This may allow us to skip invoking the base selector module entirely.
        if _bases is not None:
            remaining_components: list[TopologyComponent] = []
            for component in components_needing_base:
                matched = False
                for node in component.nodes:
                    if node.index in _bases:
                        # Synthesize a 6-DoF FREE base edge connecting the hinted base node to the
                        # world. The provisional ``joint_index`` carries the synthetic sentinel so
                        # ``_commit_base_edge`` mints an unambiguous ``NJ + k`` index for it.
                        base_node = GraphNode(index=int(node.index))
                        base_edge = GraphEdge(
                            joint_type=int(JointDoFType.FREE),
                            joint_index=UNASSIGNED_JOINT_TYPE,
                            nodes=(node.index, self._world_node),
                        )
                        self._commit_base_edge(component, base_node, base_edge)

                        # Mark the component as matched so it is
                        # not processed by the selector loop below.
                        matched = True
                        break

                # Carry forward only components that were not matched
                # by any base hint, so the selector loop below sees an
                # accurate "still needs a base" list.
                if not matched:
                    remaining_components.append(component)
            # Update the list of components that need a base assignment
            components_needing_base = remaining_components

        # If this method is called explicitly by the
        # user ensure that a base selector is set
        if self._base_selector is None and components_needing_base:
            raise ValueError(
                f"No base selector module provided, but {len(components_needing_base)} component(s) "
                f"still lack a base node/edge after parsing. Provide a `base_selector` module via "
                f"the `TopologyGraph` constructor."
            )

        # Run base selection for components that need it
        for component in components_needing_base:
            base_node, base_edge = self._base_selector.select_base(component=component, bodies=_bodies, joints=_joints)
            # The selector contract returns a non-Optional `(NodeType, EdgeType)` tuple,
            # but defensively assert here so a misbehaving custom backend produces a
            # clear error at the integration site rather than a downstream type error.
            assert base_node is not None and base_edge is not None, (
                f"Base node/edge selection returned `None` for component: {component}"
            )
            self._commit_base_edge(component, base_node, base_edge)

    def generate_spanning_trees(
        self,
        traversal_mode: SpanningTreeTraversal | None = None,
        max_candidates: int | None = None,
        roots: list[NodeType] | None = None,
        *,
        override_priorities: bool = False,
        prioritize_balanced: bool = False,
    ) -> list[list[TopologySpanningTree]]:
        """Generate candidate spanning trees for every component.

        Args:
            traversal_mode: Per-call traversal override; falls back to the
                constructor default when ``None``.
            max_candidates: Per-call cap on candidates per component;
                falls back to the constructor default when ``None``.
            roots: Optional explicit root list forwarded to the generator.
            override_priorities: If ``True``, instructs the backend to
                ignore base/grounding/degree-based root prioritization.
            prioritize_balanced: If ``True``, instructs the backend to
                prefer balanced/symmetric trees in candidate ordering.

        Returns:
            A list of per-component candidate lists.

        Raises:
            ValueError: If components have not been parsed, no tree
                generator is configured, or generation fails for any
                component.
        """
        # Ensure that the graph components are generated before
        # generating spanning trees for each component of the graph
        if self._components is None:
            raise ValueError("Graph components must be generated before spanning tree generation.")

        # Ensure that a tree generator module is provided
        if self._tree_generator is None:
            raise ValueError("No tree generator module provided, cannot generate spanning trees.")

        # If a maximum number of candidates is provided, use it to limit the number of candidates generated
        validate_max_candidates(max_candidates)
        _max_candidates = max_candidates if max_candidates is not None else self._max_tree_candidates

        # Validate the traversal mode against the canonical set of supported values
        validate_traversal_mode(traversal_mode)
        _traversal_mode = traversal_mode if traversal_mode is not None else self._tree_traversal_mode

        # Proceed with generating spanning tree candidates for each component of the graph using
        # the provided tree generator module, if given, otherwise skip this and rely on
        # the user to generate the spanning trees directly from the components
        candidates = []
        for component in self._components:
            trees = self._tree_generator.generate_spanning_trees(
                component=component,
                traversal_mode=_traversal_mode,
                max_candidates=_max_candidates,
                roots=roots,
                override_priorities=override_priorities,
                prioritize_balanced=prioritize_balanced,
            )
            if trees is None:
                raise ValueError(f"Spanning tree generation failed for component: {component}")
            candidates.append(trees)
        self._candidates = candidates
        return self._candidates

    def select_spanning_trees(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> list[TopologySpanningTree]:
        """Select one spanning tree per component from the generated candidates.

        Args:
            bodies: Optional body descriptors forwarded to the selector.
            joints: Optional joint descriptors forwarded to the selector.

        Returns:
            The list of per-component selected spanning trees.

        Raises:
            ValueError: If candidates have not been generated, no tree
                selector is configured, or the selector returns ``None``.
        """
        # A tree selector module is required to populate the per-component selected
        # spanning tree list, since there is no shipped default selection heuristic.
        if self._tree_selector is None:
            raise ValueError(
                "No tree selector module provided, cannot select spanning trees. Provide a "
                "`tree_selector` module via the `TopologyGraph` constructor."
            )

        # Ensure that the candidate spanning trees are generated before
        # selecting the best spanning tree for each component of the graph
        if self._candidates is None:
            raise ValueError("Candidate spanning trees must be generated before spanning tree selection.")

        # Use the provided body and joint descriptors for parsing if given,
        # otherwise use the cached descriptors from initialization
        _bodies = bodies if bodies is not None else self._bodies
        _joints = joints if joints is not None else self._joints

        # Run tree selection for every component
        self._trees = []
        for trees in self._candidates:
            tree = self._tree_selector.select_spanning_tree(candidates=trees, bodies=_bodies, joints=_joints)
            if tree is None:
                raise ValueError("Spanning tree selection failed for component.")
            self._trees.append(tree)
        return self._trees

    def compute_index_reassignment(self, reassign_inplace: bool = False) -> tuple[list[int] | None, list[int] | None]:
        """Perform body and joint index reassignment for better locality and regularity."""
        # An index reassigner module is required to perform body and joint index
        # reassignment, since there is no shipped default reassignment heuristic.
        if self._index_reassigner is None:
            raise ValueError(
                "No index reassigner module provided, cannot perform index reassignment. Provide a "
                "`index_reassigner` module via the `TopologyGraph` constructor."
            )

        # Track the actual mode used for this run so the ``trees_remapped`` property
        # can decide whether to short-circuit to ``self._trees`` (inplace) or
        # materialize remapped copies (not inplace).
        self._reassign_indices_inplace = reassign_inplace

        # Invalidate any cached remapped trees from a previous call, since the
        # remap and/or trees may have changed.
        self._trees_remapped = None

        # Perform index reassignment based on the selected spanning trees, and
        # cache the resulting body and joint remapping lists for later reference.
        self._body_node_remap, self._joint_edge_remap = self._index_reassigner.reassign_indices(
            trees=self._trees,
            inplace=reassign_inplace,
        )

        # Return the remapping lists so they can be accessed immediately after computation.
        return self._body_node_remap, self._joint_edge_remap

    ###
    # Visualization
    ###

    def render_graph(
        self,
        graph_labels: Iterable[GraphLabels] | None = None,
        edge_label_offset_pts: float | None = None,
        force_path_labels: bool = False,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Render the graph and its components using the configured visualizer.

        The :class:`RigidBodyDescriptor` and :class:`JointDescriptor`
        lists captured at construction time are forwarded to the
        visualizer as the source of human-readable names; opting into
        either name-label variant requires those descriptors to be
        present.

        Args:
            graph_labels: Optional :data:`GraphLabels` set selecting
                which name-label variants to render. ``"inline"`` adds
                tiny on-graph annotations beside named nodes/edges;
                ``"tables"`` adds ``index | name`` reference tables
                below the graph. Both can be combined. Modes silently
                no-op when the corresponding descriptor list is missing
                or has no named entries.
            force_path_labels: When ``True``, preserve the full scoped
                name (e.g. ``/world/anymal/LF_HIP``) in inline
                annotations and tables. Defaults to ``False`` so
                USD-style ``/scope/path/leaf`` names are clipped to
                ``…/leaf`` when they exceed the per-label budget,
                keeping dense graphs readable.
            edge_label_offset_pts: Perpendicular distance, in display
                points (1 pt = 1/72 inch — matplotlib's standard
                typographic unit), between each edge and its primary
                ``index_TYPE`` label as well as the matching inline
                joint-name label on the opposite side. ``None``
                (default) uses the visualizer's built-in default.
                Increase to push labels further from their edges;
                decrease to bring them closer (set to ``0.0`` to
                recover NetworkX-style on-edge placement).
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.

        Raises:
            ValueError: If no visualizer is configured or components have
                not been parsed yet.
        """
        if self._graph_visualizer is None:
            raise ValueError("No graph visualizer module provided, cannot render graph.")
        if self._components is None:
            raise ValueError("Graph components must be generated before rendering.")
        self._graph_visualizer.render_graph(
            nodes=self._nodes,
            edges=self._edges,
            components=self._components,
            world_node=self._world_node,
            bodies=self._bodies,
            joints=self._joints,
            graph_labels=graph_labels,
            edge_label_offset_pts=edge_label_offset_pts,
            force_path_labels=force_path_labels,
            figsize=figsize,
            path=path,
            show=show,
        )

    def render_spanning_tree_candidates(
        self,
        graph_labels: Iterable[GraphLabels] | None = None,
        edge_label_offset_pts: float | None = None,
        force_path_labels: bool = False,
        skip_orphans: bool = True,
        figsize: tuple[int, int] | None = None,
        path: str | os.PathLike[str] | None = None,
        show: bool = False,
    ) -> None:
        """Render the candidate spanning trees of each component.

        Args:
            skip_orphans: When ``True``, skip orphan components.
            figsize: Optional figure size.
            path: Optional file path to save the figures. One file is written
                per rendered component, with the component's index injected
                before the file extension (e.g. ``"out.pdf"`` becomes
                ``"out_0.pdf"``, ``"out_1.pdf"``, ...). Components that the
                visualizer skips (e.g. orphans when ``skip_orphans=True``) do
                not produce a file, leaving gaps in the numbering that
                correspond to component positions in :attr:`components`.
            show: When ``True``, display the figure immediately.
            graph_labels: Optional :data:`GraphLabels` set selecting
                which name-label variants to render. See :meth:`render_graph`.
            force_path_labels: When ``True``, preserve full scoped names in
                inline annotations and tables. See :meth:`render_graph`.
            edge_label_offset_pts: Perpendicular edge-to-label distance
                in display points. See :meth:`render_graph`.

        Raises:
            ValueError: If no visualizer is configured, components have
                not been parsed, or candidates have not been generated.
        """
        if self._graph_visualizer is None:
            raise ValueError("No graph visualizer module provided, cannot render spanning tree candidates.")
        if self._components is None:
            raise ValueError("Graph components must be generated before rendering.")
        if self._candidates is None:
            raise ValueError("Candidate spanning trees must be generated before rendering.")
        for i, (component, candidates) in enumerate(zip(self._components, self._candidates, strict=True)):
            self._graph_visualizer.render_component_spanning_tree_candidates(
                component=component,
                candidates=candidates,
                world_node=self._world_node,
                bodies=self._bodies,
                joints=self._joints,
                graph_labels=graph_labels,
                edge_label_offset_pts=edge_label_offset_pts,
                force_path_labels=force_path_labels,
                skip_orphans=skip_orphans,
                figsize=figsize,
                path=self._inject_path_index_suffix(path, i),
                show=show,
            )

    def render_spanning_trees(
        self,
        graph_labels: Iterable[GraphLabels] | None = None,
        edge_label_offset_pts: float | None = None,
        force_path_labels: bool = False,
        skip_orphans: bool = True,
        figsize: tuple[int, int] | None = None,
        path: str | os.PathLike[str] | None = None,
        show: bool = False,
    ) -> None:
        """Render the selected spanning tree of each component.

        Args:
            skip_orphans: When ``True``, skip orphan components.
            figsize: Optional figure size.
            path: Optional file path to save the figures. One file is written
                per rendered component, with the component's index injected
                before the file extension (e.g. ``"out.pdf"`` becomes
                ``"out_0.pdf"``, ``"out_1.pdf"``, ...). Components that the
                visualizer skips (e.g. orphans when ``skip_orphans=True``) do
                not produce a file, leaving gaps in the numbering that
                correspond to component positions in :attr:`components`.
            show: When ``True``, display the figure immediately.
            graph_labels: Optional :data:`GraphLabels` set selecting
                which name-label variants to render. See :meth:`render_graph`.
            force_path_labels: When ``True``, preserve full scoped names in
                inline annotations and tables. See :meth:`render_graph`.
            edge_label_offset_pts: Perpendicular edge-to-label distance
                in display points. See :meth:`render_graph`.

        Raises:
            ValueError: If no visualizer is configured, components have
                not been parsed, or trees have not been selected.
        """
        if self._graph_visualizer is None:
            raise ValueError("No graph visualizer module provided, cannot render spanning trees.")
        if self._components is None:
            raise ValueError("Graph components must be generated before rendering.")
        if self._trees is None:
            raise ValueError("Selected spanning trees must be generated before rendering.")
        for i, (component, tree) in enumerate(zip(self._components, self._trees, strict=True)):
            self._graph_visualizer.render_component_spanning_tree(
                component=component,
                tree=tree,
                world_node=self._world_node,
                bodies=self._bodies,
                joints=self._joints,
                graph_labels=graph_labels,
                edge_label_offset_pts=edge_label_offset_pts,
                force_path_labels=force_path_labels,
                skip_orphans=skip_orphans,
                figsize=figsize,
                path=self._inject_path_index_suffix(path, i),
                show=show,
            )

    # TODO: Method to render the graph with reassigned indices, which would be the final output of the full
    # pipeline and most relevant for debugging the final result of the parsing and generation procedures.
    # TODO: Also add the corresponding method to the visualizer base class

    ###
    # Internals
    ###

    @staticmethod
    def _inject_path_index_suffix(path: str | os.PathLike[str] | None, suffix: int | str | None) -> Path | None:
        """Inject ``_<index>`` into ``path`` just before the file extension.

        Used by the per-component rendering helpers to expand a single
        user-supplied output path into one path per component, so the
        figure for each component lands in its own file rather than
        overwriting the previous one.

        Only the last extension is treated as the suffix, so paths like
        ``"out.pdf"`` become ``"out_<index>.pdf"`` and paths without an
        extension simply get ``_<index>`` appended.

        Args:
            path:
                User-supplied output path, or ``None`` to disable
                file output (in which case ``None`` is returned).
            suffix:
                Suffix to inject into the path, or ``None`` to disable
                file output (in which case ``None`` is returned). If a string,
                it is injected as is. If an integer, it is converted to a string
                and injected as the suffix. If ``None``, the suffix is not injected.
        """
        if path is None:
            return None
        if isinstance(suffix, int):
            suffix = str(suffix)
        return Path(path).with_name(f"{Path(path).stem}_{suffix}{Path(path).suffix}" if suffix else Path(path).name)

    @staticmethod
    def _assert_node_valid(node: GraphNode) -> None:
        """Assert that ``node`` is a non-negative-index body node.

        The integer/bool rejection happens earlier in
        :meth:`GraphNode.from_input`; this assertion only enforces the
        non-negative-index invariant on the canonical :class:`GraphNode`.

        Raises:
            TypeError: If ``node`` is not a :class:`GraphNode` instance.
            ValueError: If ``node.index`` is negative (reserved for the world).
        """
        if not isinstance(node, GraphNode):
            raise TypeError(f"Graph node `{node!r}` is not a `GraphNode` instance (got {type(node).__name__}).")
        if node.index < 0:
            raise ValueError(
                f"Graph node `{node}` has a negative index, which is reserved for the implicit world node."
            )

    @staticmethod
    def _assert_edge_valid(edge: GraphEdge) -> None:
        """Assert that ``edge`` has been normalized to a :class:`GraphEdge`.

        Constructor-time inputs are coerced to :class:`GraphEdge` via
        :meth:`GraphEdge.from_input` before this check runs, so a
        non-:class:`GraphEdge` value here indicates a programming error
        rather than user input.

        Raises:
            TypeError: If ``edge`` is not a :class:`GraphEdge` instance.
        """
        if not isinstance(edge, GraphEdge):
            raise TypeError(
                f"Graph edge `{edge!r}` is not a `GraphEdge` instance "
                f"(got {type(edge).__name__}); inputs must be `GraphEdge` or a "
                f"3-tuple `(joint_type, joint_index, (pbid, sbid))` and are "
                f"normalized at construction."
            )

    @staticmethod
    def _assert_world_node_valid(world_node: int, nodes: list[GraphNode]) -> None:
        """Assert that ``world_node`` is a negative integer not present in ``nodes``.

        Raises:
            TypeError: If ``world_node`` is not an integer (booleans are
                rejected even though ``bool`` subclasses ``int``).
            ValueError: If ``world_node`` is non-negative or contained in
                the body indices of ``nodes``.
        """
        if isinstance(world_node, bool) or not isinstance(world_node, int):
            raise TypeError(f"World index `{world_node!r}` is not an integer representing the world node index.")
        if world_node >= 0:
            raise ValueError(
                f"World index `{world_node}` is a non-negative integer, but it should be a "
                f"negative integer representing the implicit world node."
            )
        if world_node in {int(n) for n in nodes}:
            raise ValueError(f"World index `{world_node}` should not be included in the nodes list.")

    @staticmethod
    def _assert_no_duplicate_node_indices(nodes: list[GraphNode]) -> None:
        """Assert that ``nodes`` does not contain duplicate body indices.

        Each node represents a distinct body and must therefore appear at
        most once. Distinct :class:`GraphNode` instances that share an
        index are still considered duplicates (equality on
        :class:`GraphNode` only compares ``index``).

        Raises:
            ValueError: If any body index appears more than once in ``nodes``.
        """
        seen: dict[int, int] = {}
        for n in nodes:
            seen[int(n)] = seen.get(int(n), 0) + 1
        duplicates = sorted(idx for idx, count in seen.items() if count > 1)
        if duplicates:
            raise ValueError(f"Graph nodes contain duplicate body indices: {duplicates}.")

    @staticmethod
    def _assert_edge_endpoints_in_nodes(
        edges: list[GraphEdge],
        nodes: list[GraphNode],
        world_node: int,
    ) -> None:
        """Assert that every edge endpoint is either ``world_node`` or a body index in ``nodes``.

        A non-world endpoint that is not present in ``nodes`` indicates a
        malformed input graph (e.g. a typo in the joint's body indices)
        rather than a meaningful structural choice; surfacing the error
        eagerly is preferable to silently dropping the edge during
        component parsing.

        Raises:
            ValueError: If any edge has a non-world endpoint missing from
                ``nodes``.
        """
        body_indices = {int(n) for n in nodes}
        unknown: list[tuple[int, int, tuple[int, int]]] = []
        for e in edges:
            for endpoint in e.nodes:
                if endpoint == world_node:
                    continue
                if endpoint not in body_indices:
                    unknown.append((e.joint_type, e.joint_index, e.nodes))
                    break
        if unknown:
            raise ValueError(
                f"Graph edges reference body indices not contained in `nodes`: {unknown}; "
                f"every non-world edge endpoint must match a body index in `nodes` "
                f"(world_node={world_node})."
            )

    @staticmethod
    def _assert_unique_edge_joint_indices(edges: list[GraphEdge]) -> None:
        """Assert that ``joint_index`` uniquely identifies an edge.

        Exact-duplicate edges (same ``joint_type``, ``joint_index`` and
        ``nodes`` tuple) are harmless because the parser collapses them
        via ``set`` deduplication. Any other case where two edges share
        a ``joint_index`` — whether they differ in ``joint_type``, in
        the body-pair, or only in the ``(u, v)`` polarity — violates
        the "global ``joint_index`` uniquely identifies a joint"
        invariant assumed by every downstream algorithm (degree
        counting, arc/chord classification, oriented-chord polarity
        selection).

        Raises:
            ValueError: If any ``joint_index`` is shared by two edges
                whose ``(joint_type, nodes)`` pair differs (polarity
                swaps included).
        """
        groups: dict[int, set[tuple[int, tuple[int, int]]]] = {}
        for e in edges:
            groups.setdefault(e.joint_index, set()).add((e.joint_type, e.nodes))
        conflicts: list[tuple[int, list[tuple[int, tuple[int, int]]]]] = []
        for jid, group in groups.items():
            if len(group) > 1:
                conflicts.append((jid, sorted(group)))
        if conflicts:
            details = "; ".join(f"joint_index={jid}: {items!r}" for jid, items in conflicts)
            raise ValueError(
                f"Graph edges have conflicting entries that share a `joint_index` but differ in "
                f"`joint_type` or in the body-pair (including polarity swaps): {details}. "
                f"Each global `joint_index` must uniquely identify a joint."
            )

    def _validate_inputs(self) -> None:
        """Validate the input graph attributes.

        Raises:
            ValueError: If any node has a negative index, ``nodes``
                contains duplicate body indices, an edge references a
                body index not in ``nodes``, two edges share a
                ``joint_index`` with conflicting fields, the world
                index is non-negative or contained in ``nodes``, or the
                spanning-tree configurations are invalid.
            TypeError: If ``world_node`` is not an integer or
                ``max_tree_candidates`` is not an integer.
        """
        for node in self._nodes:
            self._assert_node_valid(node)

        self._assert_no_duplicate_node_indices(self._nodes)

        for edge in self._edges:
            self._assert_edge_valid(edge)

        self._assert_world_node_valid(self._world_node, self._nodes)

        # Run after the world-node and per-node checks so the lookup set
        # below reflects the validated body-index domain.
        self._assert_edge_endpoints_in_nodes(self._edges, self._nodes, self._world_node)

        # Catches ambiguous joint labelling (e.g. a joint specified twice
        # with swapped `(u, v)` polarity) before it pollutes the
        # downstream degree counts and arc/chord enumeration.
        self._assert_unique_edge_joint_indices(self._edges)

        validate_traversal_mode(self._tree_traversal_mode)
        validate_max_candidates(self._max_tree_candidates)

    def _commit_base_edge(
        self,
        component: TopologyComponent,
        base_node: GraphNode,
        base_edge: GraphEdge,
    ) -> GraphEdge:
        """Commit a base node/edge to ``component`` and bookkeep synthetic edges.

        Edges with a negative ``joint_index`` are treated as synthetic
        sentinels: a fresh :class:`GraphEdge` is minted with
        ``joint_index = self._original_num_edges + k``, where ``k`` is
        the number of synthetic edges committed so far on this graph.
        Synthetic edges are appended to both :attr:`_new_base_edges` and
        :attr:`_edges`; existing edges (already present in
        ``self._edges`` because they were supplied at construction) are
        only committed to the component and otherwise left alone.

        Args:
            component: The component receiving the new base assignment.
            base_node: The body node selected as the component's base.
            base_edge: The edge selected as the component's base. Edges with
                ``joint_index < 0`` are interpreted as synthetic and re-issued
                with the next ``NJ + k`` global index.

        Returns:
            The (possibly minted) :class:`GraphEdge` that was committed.
        """
        if base_edge.joint_index < 0:
            # Mint the next provisional NJ + k index. ``len(self._new_base_edges or [])`` reads
            # the count of synthetics committed BEFORE this one, which equals the next index
            # past ``self._original_num_edges``. Re-issuing as a fresh ``GraphEdge`` keeps the
            # frozen-dataclass invariant intact.
            next_idx = self._original_num_edges + len(self._new_base_edges or [])
            base_edge = GraphEdge(
                joint_type=base_edge.joint_type,
                joint_index=next_idx,
                nodes=base_edge.nodes,
            )
            if self._new_base_edges is None:
                self._new_base_edges = []
            self._new_base_edges.append(base_edge)
            self._edges.append(base_edge)

        # `assign_base` atomically commits the new base, flips `is_connected` to ``True``, drops
        # the promoted edge from the grounding lists when applicable, and re-validates state.
        component.assign_base(base_node=base_node, base_edge=base_edge)
        return base_edge


###
# Backends
###


class TopologyComponentParser(TopologyComponentParserBase):
    """Default :class:`TopologyComponentParserBase` backend using union-find.

    Groups connected nodes into components via a disjoint-set data
    structure and classifies each component by its connectivity to the
    implicit world node.
    """

    @override
    def parse_components(
        self,
        nodes: list[NodeType],
        edges: list[EdgeType],
        world: int = DEFAULT_WORLD_NODE_INDEX,
    ) -> list[TopologyComponent]:
        """Parse ``nodes`` and ``edges`` into a list of components.

        Args:
            nodes: List of body node indices.
            edges: List of joint edges in :data:`EdgeType` form.
            world: The implicit world node index.

        Returns:
            A list of :class:`TopologyComponent` instances.

        Note:
            Components with more than one body are classified as
            ``islands`` and single-body components as ``orphans``.
            Components with at least one edge to the world are
            ``connected``, otherwise ``isolated``.
        """
        # Deduplicate edges and sort by `(joint_index, joint_type, nodes)` so derived
        # structures (component edge lists, parent arrays, traversal orders, ...) are
        # deterministic across graphs that share structure but differ in joint labelling.
        unique_edges: list[GraphEdge] = sorted(
            {GraphEdge.from_input(e) for e in edges},
            key=lambda e: (e.joint_index, e.joint_type, e.nodes),
        )
        msg.debug("edges: %s", unique_edges)

        # Coerce the front-end `NodeType` union into canonical `GraphNode` instances and
        # build an `index -> GraphNode` mapping so the body-only union-find can keep using
        # cheap int keys while the component construction below preserves any optional
        # node metadata (e.g. names) carried by the original `GraphNode` inputs.
        canonical_nodes: list[GraphNode] = [GraphNode.from_input(n) for n in nodes]
        node_by_index: dict[int, GraphNode] = {n.index: n for n in canonical_nodes}

        # Keep only the real (non-world) body indices, deduplicated
        body_nodes: set[int] = {n.index for n in canonical_nodes if n.index != world}
        msg.debug("nodes: %s", sorted(body_nodes))

        # Union-Find / Disjoint set
        parent = {n: n for n in body_nodes}
        rank = dict.fromkeys(body_nodes, 0)

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

        # Body-to-body edges merge components; world-incident edges are classified later.
        for e in unique_edges:
            u, v = e.nodes
            if u in body_nodes and v in body_nodes:
                union(u, v)

        # Gather connected subgraphs among body nodes
        component_node_map = defaultdict(list)
        for n in body_nodes:
            component_node_map[find(n)].append(n)
        component_nodes = [sorted(comp) for comp in component_node_map.values()]

        # Classify components as `islands` (size > 1) vs `orphans` (size == 1)
        islands = [n for n in component_nodes if len(n) > 1]
        orphans = [n for n in component_nodes if len(n) == 1]

        # Sort islands by descending size and orphans by index, then concatenate them
        islands.sort(key=len, reverse=True)
        orphans.sort(key=lambda c: c[0])
        component_nodes = islands + orphans
        msg.debug("component_nodes: %s", component_nodes)

        # Construct component objects with their properties
        components = []
        for comp_nodes in component_nodes:
            # Use a set for O(1) membership checks during the per-component edge sweep
            comp_nodes_set = set(comp_nodes)

            # Collect edges for the component and identify world-incident ones
            comp_edges: list[GraphEdge] = []
            comp_grounding_edges: list[GraphEdge] = []
            for e in unique_edges:
                u, v = e.nodes
                is_comp_edge = u in comp_nodes_set and v in comp_nodes_set
                is_comp_ground_edge = (u in comp_nodes_set and v == world) or (v in comp_nodes_set and u == world)
                if is_comp_edge or is_comp_ground_edge:
                    comp_edges.append(e)
                # All world-incident edges start as grounding edges; the auto-base
                # promotion below decides which (if any) becomes the base edge.
                if is_comp_ground_edge:
                    comp_grounding_edges.append(e)

            # Auto-promote a single grounding edge to the base
            comp_base_node: int | None = None
            comp_base_edge: GraphEdge | None = None
            if len(comp_grounding_edges) == 1:
                comp_base_edge = comp_grounding_edges[0]
                comp_base_node = next(n for n in comp_base_edge.nodes if n != world)
                comp_grounding_edges = []

            # With multiple grounding edges, promote the unique FREE one (if any) to base;
            # raise on more than one FREE grounding edge as that violates conventions.
            elif len(comp_grounding_edges) > 1:
                free_grounding_edges = [e for e in comp_grounding_edges if e.joint_type == JointDoFType.FREE]
                if len(free_grounding_edges) == 1:
                    comp_base_edge = free_grounding_edges[0]
                    comp_base_node = next(n for n in comp_base_edge.nodes if n != world)
                    comp_grounding_edges.remove(comp_base_edge)
                elif len(free_grounding_edges) > 1:
                    raise ValueError(
                        f"Component with nodes `{comp_nodes}` has multiple grounding edges `{comp_grounding_edges}` "
                        f"with more than one 6-DoF FREE joint, which violates modelling conventions."
                    )

            # Recompute grounding nodes from the final grounding-edge list to keep the
            # `set(ground_nodes) == implied_endpoints_of(ground_edges)` invariant.
            comp_grounding_nodes = sorted({n for e in comp_grounding_edges for n in e.nodes if n != world})

            # Lift the int-indexed bookkeeping back through `node_by_index` so every
            # produced `TopologyComponent` carries the canonical `GraphNode` instances
            # (with any optional metadata) supplied at the front-end boundary.
            components.append(
                TopologyComponent(
                    nodes=[node_by_index[i] for i in comp_nodes],
                    edges=comp_edges,
                    ground_nodes=[node_by_index[i] for i in comp_grounding_nodes],
                    ground_edges=comp_grounding_edges,
                    base_node=node_by_index[comp_base_node] if comp_base_node is not None else None,
                    base_edge=comp_base_edge,
                    is_island=len(comp_nodes) > 1,
                    is_connected=comp_base_edge is not None or len(comp_grounding_edges) > 0,
                    world_node=world,
                )
            )

        # Return the list of graph components
        msg.debug("components: %s", components)
        return components


class TopologyIndexReassignment(TopologyIndexReassignmentBase):
    """Default :class:`TopologyIndexReassignmentBase` backend that reassigns body indices to a contiguous range.

    Reassigns body indices to a contiguous range starting from zero, and updates
    the edge endpoint indices accordingly. The world node index is left unchanged.

    The procedure groups bodies and joints contiguously by spanning tree (largest
    tree first for better locality) and lays them out within each tree according
    to Featherstone's regular numbering rules: bodies in the tree's traversal
    order (root at the smallest new index), then arcs at joint positions parallel
    to the bodies they connect to, then chords sorted by their predecessor body's
    new local position.
    """

    @override
    def reassign_indices(
        self,
        trees: list[TopologySpanningTree],
        inplace: bool = False,
    ) -> tuple[list[int] | None, list[int] | None]:
        """Re-assign body-node/joint-edge indices given the set of spanning trees present in the graph.

        This implementation realizes the following procedure to optimize for better
        data locality and satisfaction of Featherstone's regular numbering rules:

            1. Reorders the trees (via copy or in-place) so that they are organized according
               to the their size, i.e. `len(nodes) + len(edges)` of edges in descending order,
               so that larger components and trees are prioritized for better locality.

            2. Reorders nodes/edges to group them contiguously according to the spanning trees they belong to.

            3. Reorders nodes/edges within each tree according to the traversal mode of the original tree,
               starting from the root node which should always be the first node, and then following Featherstones
               regular numbering rules: for each joint arc ``(i, j)``, where ``i`` is the predecessor body and ``j``
               is the successor body, it should ensure that ``min(i, j) < max(i, j)`` and that the parent of body
               ``j`` is body ``i``. This means that bodies should be ordered according to traversal order, and joint
               arcs should be ordered according to the index of their predecessor body. Chords should then be ordered
               after arcs, and ordered according to the index of their predecessor body as well.

        Args:
            trees: The list of spanning trees used to derive optimized node/edge indices.
            inplace: If ``True``, also modifies the trees in-place with the new indices.

        Returns:
            A tuple of two lists: the first is a body-node index remapping list,
            and the second is a joint-edge index remapping list. Each list maps
            old to new indices, that span all nodes/edges present over all trees.
            Indices not covered by any tree default to identity (i.e. ``remap[i] == i``)
            so the remap is safe to apply unconditionally to a parallel array.
            If a remap list is ``None``, it indicates that no re-assignment
            should be performed for that category (body nodes or joint edges).
        """
        # Empty input → no remap to compute. Return ``(None, None)`` so the caller
        # can short-circuit (matches the documented "no re-assignment" semantics).
        if not trees:
            return None, None

        # ------------------------------------------------------------------
        # Step 1: order trees by descending size with a stable secondary key
        # ------------------------------------------------------------------
        # Larger trees go first so their dense bodies/joints occupy the lowest
        # indices — the locality/cache argument from the pipeline header.
        # Sizes fall back to the tree's own counters when the source component
        # is missing (e.g. for trees built directly from raw data in tests).
        ordered_pairs: list[tuple[int, TopologySpanningTree]] = sorted(
            enumerate(trees),
            key=lambda it: (-self._size_of(it[1]), it[0]),
        )

        # ------------------------------------------------------------------
        # Step 2: size the remap arrays from the maximum old index encountered
        # ------------------------------------------------------------------
        # Defaults are identity so unmapped slots act as no-ops at the call
        # site (which can otherwise blindly apply ``remap[old]``).
        max_body_idx, max_joint_idx = self._max_indices(trees)
        body_remap: list[int] = list(range(max_body_idx + 1)) if max_body_idx >= 0 else []
        joint_remap: list[int] = list(range(max_joint_idx + 1)) if max_joint_idx >= 0 else []

        # ------------------------------------------------------------------
        # Step 3: walk trees in size order, building the remaps
        # ------------------------------------------------------------------
        body_offset = 0
        joint_offset = 0
        for _, tree in ordered_pairs:
            nb = tree.num_bodies
            if nb == 0:
                continue

            # Recover the original global body index at each local position
            # by walking the arcs and falling back on the source component
            # for endpoint lookup (mirrors the helper in
            # :class:`TopologyMinimumDepthSpanningTreeSelector`).
            local_to_global = self._reconstruct_local_to_global(tree)

            # Body remap: traversal-order local position ``i`` claims the next
            # ``body_offset + i`` slot. Each component's bodies thus occupy a
            # contiguous, root-first block.
            for i in range(nb):
                old = local_to_global[i]
                if old < 0:
                    continue
                body_remap[old] = body_offset + i

            # Whether the tree's slot 0 holds a real arc (real or synthetic
            # base joint) or the ``NO_BASE_JOINT_INDEX`` sentinel determines
            # whether the joint segment starts at ``i == 0`` or ``i == 1``.
            has_base = tree.arcs is not None and len(tree.arcs) > 0 and tree.arcs[0] != NO_BASE_JOINT_INDEX

            # Arc remap: local arc at body position ``i`` lands at
            # joint slot ``joint_offset + i`` when there is a base arc and at
            # ``joint_offset + i - 1`` otherwise (no slot is reserved for the
            # missing base). Sentinel arcs are skipped — they have no old
            # global index to remap.
            if tree.arcs is not None:
                for i, old in enumerate(tree.arcs):
                    if old == NO_BASE_JOINT_INDEX:
                        continue
                    arc_pos = i if has_base else i - 1
                    joint_remap[old] = joint_offset + arc_pos

            # Chord remap: chords are placed after the arcs and ordered by
            # their predecessor body's local position (which is monotone in
            # the new global body index within this tree). Ties are broken by
            # the chord's original list position to keep the result stable.
            num_chords_in_tree = len(tree.chords) if tree.chords is not None else 0
            num_arcs_in_tree = nb if has_base else max(nb - 1, 0)
            if num_chords_in_tree > 0:
                chord_data: list[tuple[int, int, int]] = []
                preds = tree.predecessors if tree.predecessors is not None else []
                for k, chord_old_idx in enumerate(tree.chords):
                    local_pos_in_preds = nb + k
                    pred_local = (
                        preds[local_pos_in_preds] if local_pos_in_preds < len(preds) else DEFAULT_WORLD_NODE_INDEX
                    )
                    chord_data.append((chord_old_idx, pred_local, k))
                chord_data.sort(key=lambda c: (c[1], c[2]))
                for new_chord_pos, (chord_old_idx, _pred_local, _orig_k) in enumerate(chord_data):
                    if chord_old_idx >= 0:
                        joint_remap[chord_old_idx] = joint_offset + num_arcs_in_tree + new_chord_pos

            # Advance offsets by the actual number of bodies/joints in this
            # tree (no slot reserved for a sentinel base).
            body_offset += nb
            joint_offset += num_arcs_in_tree + num_chords_in_tree

        # ------------------------------------------------------------------
        # Step 4: optionally apply the remap to the trees in place
        # ------------------------------------------------------------------
        if inplace:
            for tree in trees:
                new_tree = tree.remapped(body_remap, joint_remap)
                # Local-position fields (``parents``, ``children``, ``subtree``,
                # ``support``, ``predecessors``, ``successors``) are unchanged
                # by remapping, so we only swap in the global-index fields and
                # drop the now-stale ``component`` reference.
                tree.root = new_tree.root
                tree.arcs = new_tree.arcs
                tree.chords = new_tree.chords
                tree.component = None

        return body_remap, joint_remap

    ###
    # Internals
    ###

    @staticmethod
    def _size_of(tree: TopologySpanningTree) -> int:
        """Return the descending-sort key used to prioritize larger trees first."""
        comp = tree.component
        if comp is not None and comp.nodes is not None and comp.edges is not None:
            return len(comp.nodes) + len(comp.edges)
        return tree.num_bodies + tree.num_joints

    @staticmethod
    def _max_indices(trees: list[TopologySpanningTree]) -> tuple[int, int]:
        """Return the highest body / joint global index referenced by ``trees``.

        Used to size the dense identity-initialised remap arrays. Both values
        default to ``-1`` when nothing is referenced (e.g. a list of empty
        trees), in which case the caller produces empty remaps.
        """
        max_body_idx = -1
        max_joint_idx = -1
        for tree in trees:
            if tree.root is not None and tree.root >= 0:
                max_body_idx = max(max_body_idx, int(tree.root))
            if tree.component is not None and tree.component.edges is not None:
                for e in tree.component.edges:
                    for n in e.nodes:
                        if n >= 0:
                            max_body_idx = max(max_body_idx, n)
                    if e.joint_index >= 0:
                        max_joint_idx = max(max_joint_idx, e.joint_index)
            if tree.arcs is not None:
                for a in tree.arcs:
                    if a >= 0:
                        max_joint_idx = max(max_joint_idx, a)
            if tree.chords is not None:
                for c in tree.chords:
                    if c >= 0:
                        max_joint_idx = max(max_joint_idx, c)
        return max_body_idx, max_joint_idx

    @staticmethod
    def _reconstruct_local_to_global(tree: TopologySpanningTree) -> list[int]:
        """Recover the global body index at each local position of ``tree``.

        Walks the arcs in regular-numbering order: ``arcs[i]`` connects the
        body at local position ``i`` to its parent at local position
        ``parents[i]``. The matching :class:`GraphEdge` in ``tree.component``
        yields the global ``(u, v)`` endpoints; the one that isn't the
        parent's already-known global index is the child's.

        Raises:
            ValueError: If the tree is missing the bookkeeping needed for
                reconstruction (no ``arcs``/``parents``, missing component
                edges, or a sentinel arc on a non-root local position).
        """
        nb = tree.num_bodies
        if nb == 0:
            return []
        if tree.root is None:
            raise ValueError("Cannot reconstruct local-to-global mapping: `tree.root` is None.")
        local_to_global: list[int] = [int(tree.root)] + [-1] * (nb - 1)
        if nb == 1:
            return local_to_global

        if tree.arcs is None or tree.parents is None:
            raise ValueError("Cannot reconstruct local-to-global mapping: `tree.arcs` and `tree.parents` are required.")
        if tree.component is None or tree.component.edges is None:
            raise ValueError("Cannot reconstruct local-to-global mapping: `tree.component.edges` is required.")

        edge_endpoints: dict[int, tuple[int, int]] = {e.joint_index: e.nodes for e in tree.component.edges}
        for i in range(1, nb):
            joint_idx = tree.arcs[i]
            if joint_idx == NO_BASE_JOINT_INDEX:
                # ``NO_BASE_JOINT_INDEX`` only ever occupies slot 0 (the missing
                # base joint of an isolated tree); seeing it on a non-root slot
                # means the tree was assembled inconsistently.
                raise ValueError(f"Cannot reconstruct local-to-global mapping: sentinel arc at local position {i}.")
            endpoints = edge_endpoints.get(joint_idx)
            if endpoints is None:
                raise ValueError(
                    f"Cannot reconstruct local-to-global mapping: joint {joint_idx} not in component edges."
                )
            parent_global = local_to_global[tree.parents[i]]
            u, v = endpoints
            local_to_global[i] = v if u == parent_global else u
        return local_to_global
