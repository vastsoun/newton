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
#       4e. Finally, it should reorder nodes/edges within each tree according to the traversal mode of the original tree starting from the base node, if defined, or the node with the highest degree otherwise, and following Featherstone's regular numbering rules, i.e. for each joint edge (i, j), where i is the predecessor body and j is the successor body, it should ensure that min(i, j) < max(i, j) and that the parent of body j is body i.
#       4f. It should generate and return:
#       - a new TopologyGraph with re-ordered components and spanning trees, as well as updated node and edge indices according to the new body/joint ordering.
#       - a list of body node index reassignments, where the entry at index `i` gives the new body index assigned to the body with original index `i`.
#       - a list of joint edge index reassignments, where the entry at index `j` gives the new joint index assigned to the joint with original index `j`.
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
   3c. All synthetic base edges are assigned the joint index ``-2`` to
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

from collections import defaultdict

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
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    SpanningTreeTraversal,
    TopologyComponent,
    TopologyComponentBaseSelectorBase,
    TopologyComponentParserBase,
    TopologyGraphVisualizerBase,
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
        graph_visualizer: TopologyGraphVisualizerBase | None = None,
        # Source model descriptors
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        # Parsing configurations
        tree_traversal_mode: SpanningTreeTraversal = "dfs",
        max_tree_candidates: int = 32,
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
        if self._graph_visualizer is None:
            self._graph_visualizer = TopologyGraphVisualizer()

        # Declare and initialize internal caches for the source model descriptors
        self._bodies: list[RigidBodyDescriptor] | None = bodies
        self._joints: list[JointDescriptor] | None = joints

        # Declare derived attributes
        self._components: list[TopologyComponent] | None = None
        """Parsed component subgraphs of the topology graph."""
        self._candidates: list[list[TopologySpanningTree]] | None = None
        """Candidate spanning trees per component."""
        self._trees: list[TopologySpanningTree] | None = None
        """Selected spanning tree per component."""

        # If `autoparse` is True, automatically parse the graph nodes
        # and edges into components and generate spanning trees
        if autoparse:
            self.parse()

    ###
    # Properties
    ###

    @property
    def nodes(self) -> list[GraphNode]:
        """Return the list of body nodes in the graph."""
        return self._nodes

    @property
    def edges(self) -> list[GraphEdge]:
        """Return the list of joint edges in the graph (empty if none)."""
        return self._edges

    @property
    def world_node(self) -> int:
        """Return the index of the implicit world node."""
        return self._world_node

    @property
    def components(self) -> list[TopologyComponent]:
        """Return the list of parsed components.

        Raises:
            ValueError: If components have not been parsed yet.
        """
        if self._components is None:
            raise ValueError("Graph components have not been parsed yet.")
        return self._components

    @property
    def candidates(self) -> list[list[TopologySpanningTree]]:
        """Return the per-component lists of candidate spanning trees.

        Raises:
            ValueError: If candidates have not been generated yet.
        """
        if self._candidates is None:
            raise ValueError("Candidate spanning trees have not been generated yet.")
        return self._candidates

    @property
    def trees(self) -> list[TopologySpanningTree]:
        """Return the per-component selected spanning trees.

        Raises:
            ValueError: If spanning trees have not been selected yet.
        """
        if self._trees is None:
            raise ValueError("Spanning trees have not been selected yet.")
        return self._trees

    ###
    # Operations
    ###

    def parse(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        # TODO: Add option to specify the component base node/edge indices and skip the base selector module
    ) -> None:
        """Run the full topology-discovery pipeline end-to-end.

        Calls :meth:`parse_components`, :meth:`select_component_bases`,
        :meth:`generate_spanning_trees`, and :meth:`select_spanning_trees`
        in order. The cached ``tree_traversal_mode`` and
        ``max_tree_candidates`` are forwarded to the generator so that
        constructor-time configuration is honored.

        Args:
            bodies: Optional body descriptors forwarded to the base/tree
                selectors; falls back to the descriptors supplied at
                construction time when omitted.
            joints: Optional joint descriptors forwarded to the base/tree
                selectors; falls back to the descriptors supplied at
                construction time when omitted.

        Raises:
            ValueError: If any module required by the full pipeline is
                missing, parsing or generation fails, or the graph
                attributes are invalid.
        """
        # Validate up front that every module required by the full pipeline is available,
        # so that the user can fix them in one round instead of failing in step N of M.
        # The base selector is treated as required only if at least one component lacks an
        # auto-assigned base after parsing — see :meth:`select_component_bases`.
        missing = [
            name
            for name, mod in (
                ("component_parser", self._component_parser),
                ("tree_generator", self._tree_generator),
                ("tree_selector", self._tree_selector),
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

        # Parse the graph nodes and edges into components, and auto-assign
        # base nodes/edges where possible based on the discovery logic.
        self.parse_components()

        # If no base selector module is provided, skip
        # base selection since this is a optional step.
        if self._base_selector is not None:
            self.select_component_bases(bodies=_bodies, joints=_joints)

        # Generate candidate spanning trees for each component, and select one
        # per component using the configured generator and selector modules.
        self.generate_spanning_trees(
            traversal_mode=self._tree_traversal_mode,
            max_candidates=self._max_tree_candidates,
        )

        #
        self.select_spanning_trees(bodies=_bodies, joints=_joints)

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
    ) -> None:
        """Assign a base node/edge to every component that lacks one.

        Components with a single grounding edge (or a single FREE joint
        among multiple grounding edges) are auto-assigned a base by the
        parser; this method only invokes the base selector for the
        remaining components.

        Args:
            bodies: Optional body descriptors forwarded to the selector.
            joints: Optional joint descriptors forwarded to the selector.

        Raises:
            ValueError: If components have not been parsed, any component
                still lacks a base but no base selector is configured, or
                the selector returns ``None``.
        """
        # If this method is called explicitly by the
        # user ensure that a base selector is set
        if self._base_selector is None:
            raise ValueError(
                f"No base selector module provided, but {len(components_needing_base)} component(s) "
                f"still lack a base node/edge after parsing. Provide a `base_selector` module via "
                f"the `TopologyGraph` constructor."
            )

        # Ensure that the graph components are generated before
        # selecting the base node and edge for each component
        if self._components is None:
            raise ValueError("Graph components must be generated before base node/edge selection.")

        # Determine which components still need a base assignment after parsing
        components_needing_base = [c for c in self._components if c.base_edge is None]
        if not components_needing_base:
            return

        # Use the provided body and joint descriptors for parsing if given,
        # otherwise use the cached descriptors from initialization
        _bodies = bodies if bodies is not None else self._bodies
        _joints = joints if joints is not None else self._joints

        # Run base selection for components that need it
        for component in components_needing_base:
            base_node, base_edge = self._base_selector.select_base(component=component, bodies=_bodies, joints=_joints)
            # The selector contract returns a non-Optional `(NodeType, EdgeType)` tuple,
            # but defensively assert here so a misbehaving custom backend produces a
            # clear error at the integration site rather than a downstream type error.
            assert base_node is not None and base_edge is not None, (
                f"Base node/edge selection returned `None` for component: {component}"
            )
            # `assign_base` atomically commits the new base, flips `is_connected`
            # to ``True``, drops the promoted edge from the grounding lists when
            # applicable, and re-validates the resulting state. See
            # :meth:`TopologyComponent.assign_base`.
            component.assign_base(base_node=base_node, base_edge=base_edge)

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

    ###
    # Visualization
    ###

    def render_graph(
        self,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Render the graph and its components using the configured visualizer.

        Args:
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
            joints=self._joints,
            figsize=figsize,
            path=path,
            show=show,
        )

    def render_spanning_tree_candidates(
        self,
        skip_orphans: bool = True,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Render the candidate spanning trees of each component.

        Args:
            skip_orphans: When ``True``, skip orphan components.
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.

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
        for component, candidates in zip(self._components, self._candidates, strict=True):
            self._graph_visualizer.render_component_spanning_tree_candidates(
                component=component,
                candidates=candidates,
                world_node=self._world_node,
                joints=self._joints,
                skip_orphans=skip_orphans,
                figsize=figsize,
                path=path,
                show=show,
            )

    def render_spanning_trees(
        self,
        skip_orphans: bool = True,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Render the selected spanning tree of each component.

        Args:
            skip_orphans: When ``True``, skip orphan components.
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.

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
        for component, tree in zip(self._components, self._trees, strict=True):
            self._graph_visualizer.render_component_spanning_tree(
                component=component,
                tree=tree,
                world_node=self._world_node,
                joints=self._joints,
                skip_orphans=skip_orphans,
                figsize=figsize,
                path=path,
                show=show,
            )

    ###
    # Internals
    ###

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
