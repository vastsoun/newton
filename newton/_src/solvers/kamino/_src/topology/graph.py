# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Topology graph and component parser back-end for the kamino solver.

This module ships the user-facing :class:`TopologyGraph` container — the
entry point that turns raw lists of body nodes and joint edges into a
deterministic pipeline of (parsed) components, base/grounding selection,
spanning-tree candidate generation, and per-component spanning-tree
selection. It also ships the default :class:`TopologyComponentParser`
back-end (a union-find based component grouper) used when the caller does
not provide a custom :class:`TopologyComponentParserBase` instance.

See :mod:`.types` for the schema definitions (``EdgeType``, ``NodeType``,
:class:`TopologyComponent`, :class:`TopologySpanningTree`, and the
abstract module bases) and :mod:`.trees` for shipped spanning-tree
generator back-ends.
"""

from __future__ import annotations

from collections import defaultdict

from ..core.bodies import RigidBodyDescriptor
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override
from ..utils import logger as msg
from .render import TopologyGraphVisualizer
from .trees import TopologyMinimumDepthSpanningTreeGenerator
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    EdgeType,
    NodeType,
    SpanningTreeTraversal,
    TopologyComponent,
    TopologyComponentBaseSelectorBase,
    TopologyComponentParserBase,
    TopologyGraphVisualizerBase,
    TopologySpanningTree,
    TopologySpanningTreeGeneratorBase,
    TopologySpanningTreeSelectorBase,
    _validate_max_candidates,
    _validate_traversal_mode,
)

###
# Module interface
###

__all__ = [
    "TopologyGraph",
]

###
# Interfaces
###


class TopologyGraph:
    """
    A container to represent a topological undirected graph `G`.
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
        """
        Initializes the TopologyGraph object with the given nodes, edges, and
        optional modules for component parsing and spanning tree generation.

        Args:
            nodes:
                List of body node indices in the graph.
            edges:
                Optional list of joint edges in the graph, where each edge is a tuple of the
                form ``(joint_type, joint_index, (predecessor_body_index, successor_body_index))``.
            world_node:
                The index of the implicit world node in the graph. Must be a negative integer
                that is not contained in ``nodes``. Defaults to :data:`DEFAULT_WORLD_NODE_INDEX`.
            component_parser:
                Optional module to parse the graph nodes and edges into a list of components.
                Defaults to a shipped :class:`TopologyComponentParser` instance when not provided.
            base_selector:
                Optional module to select the base node and edge for component subgraphs whose
                base could not be auto-assigned during parsing. Required only when at least one
                parsed component lacks an auto-assigned base; see :meth:`select_component_bases`.
            tree_generator:
                Optional module to generate spanning tree candidates for each component
                subgraph. Required when calling :meth:`generate_spanning_trees` or :meth:`parse`.
            tree_selector:
                Optional module to select the best spanning tree from a list of candidates.
                Required when calling :meth:`select_spanning_trees` or :meth:`parse`.
            graph_visualizer:
                Optional module to render the topology graph, components, and spanning trees.
                Defaults to a shipped :class:`TopologyGraphVisualizer` instance when not provided.
            bodies:
                Optional list of rigid body descriptors associated with the graph nodes,
                forwarded to the base selector and tree selector modules to inform their
                heuristics. Can also be supplied at :meth:`parse` time.
            joints:
                Optional list of joint descriptors associated with the graph edges, forwarded
                to the base selector and tree selector modules to inform their heuristics. Can
                also be supplied at :meth:`parse` time.
            tree_traversal_mode:
                Default traversal mode used by :meth:`generate_spanning_trees`, one of ``"dfs"``
                or ``"bfs"``. Defaults to ``"dfs"``.
            max_tree_candidates:
                Default upper bound on the number of candidate spanning trees generated per
                component by :meth:`generate_spanning_trees`. Defaults to ``32``.
            autoparse:
                If ``True``, run the full :meth:`parse` pipeline (component parsing, base
                selection, spanning tree generation, and spanning tree selection) immediately
                after construction. Requires ``tree_generator`` and ``tree_selector`` to be
                provided. Defaults to ``False``.

        Raises:
            ValueError:
                If ``nodes`` is ``None``, if any node has an invalid format (non-integer or
                negative), if any edge has an invalid format, if ``world_node`` is not a
                negative integer, or if ``world_node`` is contained in ``nodes``. Also if
                ``tree_traversal_mode`` is not ``"dfs"`` or ``"bfs"``, or if
                ``max_tree_candidates`` is not a positive integer. If ``autoparse=True``,
                also raises if any module required by the full pipeline is missing (see
                :meth:`parse`).
            TypeError:
                If ``max_tree_candidates`` is not an :class:`int`.
        """
        # Cache the input graph attributes that define the graph contents and structure
        self._nodes: list[NodeType] = nodes
        """
        List of body node indices contained in the graph.

        Each node is uniquely identified by its associated index in the range
        `[0, N-1]`, where `N` is the total number of nodes in the graph.

        `N` excludes the implicit world node with index `-1`, which
        is present in the graph if any node is connected to it.
        """
        self._edges: list[EdgeType] | None = edges
        """
        List of joint indices contained in the graph.

        Each edge is uniquely identified by its associated index in
        the range `[0, M-1]`, where `M` is the total number of edges.
        """
        self._world_node: int = world_node
        """
        The index of the implicit world node in the graph.\n
        Defaults to `-1`, which is the conventional index for the world body in Newton.
        """

        # Cache parsing configurations
        self._tree_traversal_mode: SpanningTreeTraversal = tree_traversal_mode
        """The traversal mode used for generating the spanning trees."""
        self._max_tree_candidates: int = max_tree_candidates
        """ The maximum number of candidate spanning trees to generate for each component of the graph."""

        # Validate the input graph attributes to ensure they are
        # consistent with the expected formats and conventions
        self._validate_inputs()

        # Store input modules for component parsing and spanning tree generation
        self._component_parser: TopologyComponentParserBase | None = component_parser
        """A module to parse the graph nodes and edges into a list of components."""
        self._base_selector: TopologyComponentBaseSelectorBase | None = base_selector
        """A module to select the base node for each component subgraph."""
        self._tree_generator: TopologySpanningTreeGeneratorBase | None = tree_generator
        """A module to generate a spanning tree for each component subgraph."""
        self._tree_selector: TopologySpanningTreeSelectorBase | None = tree_selector
        """A module to select the best spanning tree from the given list of spanning trees."""
        self._graph_visualizer: TopologyGraphVisualizerBase | None = graph_visualizer
        """A module to render the topology graph, components and spanning trees for visualization."""

        # Set default modules where shipped concrete defaults exist. Modules without a
        # default (`base_selector`, `tree_generator`, `tree_selector`) are left as `None`
        # and validated lazily in :meth:`parse` and the per-step methods so that callers
        # who only invoke a subset of the pipeline (e.g. :meth:`parse_components`) do not
        # need to provide every module up front.
        if self._component_parser is None:
            self._component_parser = TopologyComponentParser()
        # TODO: Add a default tree selector module
        # if self._tree_selector is None:
        #     self._tree_selector = TopologySpanningTreeSelector()
        if self._tree_generator is None:
            self._tree_generator = TopologyMinimumDepthSpanningTreeGenerator()
        if self._graph_visualizer is None:
            self._graph_visualizer = TopologyGraphVisualizer()

        # Declare and initialize internal caches for the source model descriptors
        self._bodies: list[RigidBodyDescriptor] | None = bodies
        self._joints: list[JointDescriptor] | None = joints

        # Declare derived attributes
        self._components: list[TopologyComponent] | None = None
        """
        A list of topology graph components, i.e. subgraphs.
        """
        self._candidates: list[list[TopologySpanningTree]] | None = None
        """
        A list of candidate spanning tree subgraphs corresponding to each component of the graph.
        """
        self._trees: list[TopologySpanningTree] | None = None
        """
        A list of selected spanning tree subgraphs corresponding to each component of the graph.
        """

        # If `autoparse` is True, automatically parse the graph nodes
        # and edges into components and generate spanning trees
        if autoparse:
            self.parse()

    ###
    # Properties
    ###

    @property
    def nodes(self) -> list[NodeType]:
        """Returns the list of body node indices contained in the graph."""
        return self._nodes

    @property
    def edges(self) -> list[EdgeType] | None:
        """Returns the list of joint edges contained in the graph."""
        return self._edges

    @property
    def world_node(self) -> int:
        """Returns the index of the implicit world node in the graph."""
        return self._world_node

    @property
    def components(self) -> list[TopologyComponent]:
        """Returns the list of components parsed from the graph."""
        if self._components is None:
            raise ValueError("Graph components have not been parsed yet.")
        return self._components

    @property
    def candidates(self) -> list[list[TopologySpanningTree]]:
        """Returns the list of candidate spanning trees generated for each component of the graph."""
        if self._candidates is None:
            raise ValueError("Candidate spanning trees have not been generated yet.")
        return self._candidates

    @property
    def trees(self) -> list[TopologySpanningTree]:
        """Returns the list of selected spanning trees for each component of the graph."""
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
        """
        Parses the graph nodes and edges into a list of components, and generates a spanning tree for each component.

        Args:
            bodies:
                Optional list of rigid body descriptors. Forwarded to the base selector
                and tree selector modules to inform their heuristics; not consumed by
                the component parser itself. Falls back to the descriptors supplied
                at construction time when omitted.
            joints:
                Optional list of joint descriptors. Forwarded alongside ``bodies`` to
                the base selector and tree selector modules; not consumed by the
                component parser itself. Falls back to the descriptors supplied at
                construction time when omitted.

        Raises:
            ValueError:
                If any module required by the full pipeline (component parser, tree generator,
                tree selector) is missing, if graph parsing or spanning tree generation fails,
                or if the graph attributes are invalid. All missing required modules are
                reported in a single error rather than failing partway through the pipeline.
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

        # Parse the graph nodes and edges into and run the topology discovery
        # pipeline to generate spanning trees for each component of the graph.
        # Forward the cached `tree_traversal_mode` and `max_tree_candidates`
        # so that constructor-time configuration is honored, rather than
        # silently falling back to the per-call defaults.
        self.parse_components()
        self.select_component_bases(bodies=_bodies, joints=_joints)
        self.generate_spanning_trees(
            traversal_mode=self._tree_traversal_mode,
            max_candidates=self._max_tree_candidates,
        )
        self.select_spanning_trees(bodies=_bodies, joints=_joints)

    def parse_components(self) -> list[TopologyComponent]:
        """
        Parses the graph nodes and edges into a list of components using the provided component parser module.

        Returns:
            A list of `TopologyComponent` objects representing the components of the graph.

        Raises:
            ValueError:
                If graph parsing fails, i.e. if the component parser
                returns `None` or if the graph attributes are invalid.
        """
        if self._component_parser is None:
            raise ValueError("No component parser module provided, cannot parse graph components.")
        self._components = self._component_parser.parse_components(
            nodes=self._nodes, edges=self._edges, world=self._world_node
        )
        if self._components is None:
            raise ValueError("Graph component parsing failed.")
        return self._components

    def select_component_bases(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> None:
        """
        Selects the base node and edge for each component of the graph using the provided base selector module.

        The component parser may already auto-assign a base node/edge for components with a
        single grounding edge, or with a single 6-DoF FREE joint among multiple grounding
        edges. This method only invokes the base selector for components that still lack a
        base after parsing, so a missing base selector is only an error when at least one
        component requires one.

        Args:
            bodies: Optional list of body descriptors to aid in base node/edge selection.
            joints: Optional list of joint descriptors to aid in grounding node/edge selection.

        Raises:
            ValueError:
                If components have not been parsed yet, if any component still lacks a base
                node/edge but no base selector module was provided, or if the base selector
                returns `None` for a component.
        """
        # Ensure that the graph components are generated before
        # selecting the base node and edge for each component
        if self._components is None:
            raise ValueError("Graph components must be generated before base node/edge selection.")

        # Determine which components still need a base assignment after parsing
        components_needing_base = [c for c in self._components if c.base_edge is None]
        if not components_needing_base:
            return

        # Components remain that need a base — a base selector module is now required
        if self._base_selector is None:
            raise ValueError(
                f"No base selector module provided, but {len(components_needing_base)} component(s) "
                f"still lack a base node/edge after parsing. Provide a `base_selector` module via "
                f"the `TopologyGraph` constructor."
            )

        # Use the provided body and joint descriptors for parsing if given,
        # otherwise use the cached descriptors from initialization
        _bodies = bodies if bodies is not None else self._bodies
        _joints = joints if joints is not None else self._joints

        # Run base selection for components that need it
        for component in components_needing_base:
            base_node, base_edge = self._base_selector.select_base(component=component, bodies=_bodies, joints=_joints)
            if base_node is None or base_edge is None:
                raise ValueError(f"Base node/edge selection failed for component: {component}")
            # TODO: Should this be done in-place in the base selector module instead of here?
            component.base_node = base_node
            component.base_edge = base_edge
            # The component is now connected to the world via the assigned base edge.
            # `is_connected` was set during parsing based on the *original* graph topology,
            # so it must be refreshed for components that were initially isolated.
            component.is_connected = True

    def generate_spanning_trees(
        self,
        traversal_mode: SpanningTreeTraversal | None = None,
        max_candidates: int | None = None,
        roots: list[NodeType] | None = None,
        *,
        override_priorities: bool = False,
        prioritize_balanced: bool = False,
    ) -> list[list[TopologySpanningTree]]:
        """
        Generates a spanning tree for each component of the graph using the provided tree generator module.

        Args:
            traversal_mode:
                The traversal mode used by the spanning-tree generator. Must be one
                of ``"dfs"`` or ``"bfs"``.
            max_candidates:
                Optional integer specifying the maximum number of spanning
                tree candidates to generate for each component of the graph.
                This overrides the maximum number of candidates specified
                in the object constructor.
            roots:
                Optional list of node indices to forward to the tree generator as
                root candidates. Treated as a hint by backends that support direct
                root specification.
            override_priorities:
                Forwarded to :meth:`TopologySpanningTreeGeneratorBase.generate_spanning_trees`.
                If ``True``, instructs the backend to ignore base/grounding/degree-based
                root prioritization and brute-force enumerate over all body nodes.
            prioritize_balanced:
                Forwarded to :meth:`TopologySpanningTreeGeneratorBase.generate_spanning_trees`.
                If ``True``, instructs the backend to prefer balanced/symmetric trees
                in candidate ordering and truncation.

        Returns:
            A list of lists of `TopologySpanningTree` objects representing
            the candidate spanning trees for each component of the graph.

        Raises:
            ValueError:
                If spanning tree generation fails, i.e. if the tree generator
                returns `None` or if the graph attributes are invalid.
        """
        # Ensure that the graph components are generated before
        # generating spanning trees for each component of the graph
        if self._components is None:
            raise ValueError("Graph components must be generated before spanning tree generation.")

        # Ensure that a tree generator module is provided
        if self._tree_generator is None:
            raise ValueError("No tree generator module provided, cannot generate spanning trees.")

        # If a maximum number of candidates is provided, use it to limit the number of candidates generated
        _validate_max_candidates(max_candidates)
        _max_candidates = max_candidates if max_candidates is not None else self._max_tree_candidates

        # Validate the traversal mode against the canonical set of supported values
        _validate_traversal_mode(traversal_mode)
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
        if len(candidates) != len(self._components):
            raise ValueError("Spanning tree generation failed for some components.")
        self._candidates = candidates
        return self._candidates

    def select_spanning_trees(
        self,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
    ) -> list[TopologySpanningTree]:
        """
        Selects the best spanning tree for each component of the graph from the
        generated candidate spanning trees using the provided tree selector module.

        Args:
            bodies: Optional list of body descriptors to aid in spanning tree selection.
            joints: Optional list of joint descriptors to aid in spanning tree selection.

        Returns:
            A list of `TopologySpanningTree` objects representing the
            selected spanning tree for each component of the graph.

        Raises:
            ValueError:
                If candidate spanning trees have not been generated yet, if no tree selector
                module was provided, or if the tree selector returns `None` for a component.
        """
        # Ensure that the candidate spanning trees are generated before
        # selecting the best spanning tree for each component of the graph
        if self._candidates is None:
            raise ValueError("Candidate spanning trees must be generated before spanning tree selection.")

        # A tree selector module is required to populate the per-component selected
        # spanning tree list, since there is no shipped default selection heuristic.
        if self._tree_selector is None:
            raise ValueError(
                "No tree selector module provided, cannot select spanning trees. Provide a "
                "`tree_selector` module via the `TopologyGraph` constructor."
            )

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
        """
        Renders the graph and its constituent components using the configured graph visualizer module.

        Args:
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
        """
        if self._graph_visualizer is None:
            raise ValueError("No graph visualizer module provided, cannot render graph.")
        if self._components is None:
            raise ValueError("Graph components must be generated before rendering.")
        self._graph_visualizer.render_graph(
            nodes=self._nodes,
            edges=self._edges if self._edges is not None else [],
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
        """
        Renders the candidate spanning trees for each component of the graph using the configured graph visualizer module.

        Args:
            skip_orphans:
                When ``True`` (default), orphan components (single-body subgraphs whose
                spanning tree is trivial) are skipped. Forwarded to
                :meth:`TopologyGraphVisualizerBase.render_component_spanning_tree_candidates`.
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
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
        """
        Renders the selected spanning trees for each component of the graph using the configured graph visualizer module.

        Args:
            skip_orphans:
                When ``True`` (default), orphan components (single-body subgraphs) are
                skipped since their spanning trees are trivial. Set to ``False`` to render
                every component regardless.
            figsize: Optional tuple specifying the figure size for the render.
            path: Optional string specifying the file path to save the render.
            show: Boolean indicating whether to display the render immediately.
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
    def _assert_node_valid(node: NodeType) -> None:
        """
        Asserts that the given node is in the correct format.

        Raises:
            TypeError: If ``node`` is not an integer.
            ValueError: If ``node`` is a negative integer (reserved for the world node).
        """
        if not isinstance(node, int):
            raise TypeError(f"Graph node `{node}` is not an integer representing a body index.")
        if node < 0:
            raise ValueError(
                f"Graph node `{node}` is a negative integer, which is reserved for the implicit world node."
            )

    @staticmethod
    def _assert_edge_valid(edge: EdgeType) -> None:
        """
        Asserts that the given edge is in the correct format.

        Raises:
            TypeError:
                If ``edge`` is not a 3-tuple, if its elements are not integers,
                or if the body-pair entry is not a 2-tuple of integers.
        """
        if not isinstance(edge, tuple) or len(edge) != 3:
            raise TypeError(f"Graph edge `{edge}` is not in the correct format (type, jid, (pbid, sbid)).")
        joint_type, joint_index, body_pair = edge
        if not isinstance(joint_type, int):
            raise TypeError(f"Graph edge `{edge}` has a non-integer joint type.")
        if not isinstance(joint_index, int):
            raise TypeError(f"Graph edge `{edge}` has a non-integer joint index.")
        if not isinstance(body_pair, tuple) or len(body_pair) != 2:
            raise TypeError(f"Graph edge `{edge}` has an invalid body pair format.")
        if not all(isinstance(b, int) for b in body_pair):
            raise TypeError(f"Graph edge `{edge}` has non-integer body indices.")

    @staticmethod
    def _assert_world_node_valid(world_node: int, nodes: list[NodeType]) -> None:
        """
        Asserts that the given world node index is in the correct format.

        Raises:
            TypeError: If ``world_node`` is not an integer.
            ValueError:
                If ``world_node`` is a non-negative integer or if it is contained in
                ``nodes``.
        """
        if not isinstance(world_node, int):
            raise TypeError(f"World index `{world_node}` is not an integer representing the world node index.")
        if world_node >= 0:
            raise ValueError(
                f"World index `{world_node}` is a non-negative integer, but it should be a negative integer representing the implicit world node."
            )
        if world_node in nodes:
            raise ValueError(f"World index `{world_node}` should not be included in the nodes list.")

    def _validate_inputs(self):
        """
        Checks that the input graph attributes are valid and
        consistent with the expected formats and conventions.

        Raises:
            ValueError:
                If nodes are not provided, if edges are not in the correct format, if the world
                index is not an integer, or if the world index is included in the nodes list.
        """
        # Ensure that nodes are provided, as they are necessary to define the graph contents
        if self._nodes is None:
            raise ValueError("Nodes must be provided to initialize the graph.")

        # Edges are optional, as a graph can consist of isolated nodes
        # only, but if provided, they must be in the correct format
        if self._edges is None:
            self._edges = []

        # Ensure that nodes are in the correct format
        for node in self._nodes:
            self._assert_node_valid(node)

        # Ensure edges are in the correct format
        for edge in self._edges:
            self._assert_edge_valid(edge)

        # Ensure that the world index is in the correct format
        self._assert_world_node_valid(self._world_node, self._nodes)

        # Validate the spanning tree generation configurations
        _validate_traversal_mode(self._tree_traversal_mode)
        _validate_max_candidates(self._max_tree_candidates)


###
# Backends
###


class TopologyComponentParser(TopologyComponentParserBase):
    """
    A default implementation of the :class:`TopologyComponentParserBase` that parses
    the graph nodes and edges into components using a union-find / disjoint set
    data structure to efficiently group connected nodes into components, while
    also classifying them based on their connectivity to the implicit world node.
    """

    @override
    def parse_components(
        self,
        nodes: list[NodeType],
        edges: list[EdgeType],
        world: int = DEFAULT_WORLD_NODE_INDEX,
    ) -> list[TopologyComponent]:
        """
        Parses the given nodes and edges into a list of components.

        Args:
            nodes:
                List of body node indices in the graph.
            edges:
                List of joint edges in the graph, where each edge is a tuple of the form
                ``(joint_type, joint_index, (predecessor_body_index, successor_body_index))``.
            world:
                The index of the implicit world node in the graph.

        Returns:
            A list of :class:`TopologyComponent` objects representing the components of the graph.

        Note:
            - The world node is not included in the list of nodes, but it is
              considered when determining the connectivity of components.

            - Components are classified as `islands` if they contain more than one node, and as
              `orphans` if they contain exactly one node. Each component is further classified
              as `connected` if it has at least one edge connecting it to the world node, and
              as `isolated` otherwise.
        """
        # Deduplicate edges and sort them by joint index so that the order of derived
        # structures (component edge lists, parent arrays, traversal orders, etc.) is
        # deterministic across graphs that share structure but differ in joint labelling.
        unique_edges: list[EdgeType] = sorted(set(edges), key=lambda e: e[1])
        msg.debug("edges: %s", unique_edges)

        # Keep only the real (non-world) body nodes, deduplicated
        body_nodes: set[NodeType] = {n for n in nodes if n != world}
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
        for _t, _j, (u, v) in unique_edges:
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

            # Collect edges for the component and check
            # for connections to the implicit world node
            comp_edges: list[EdgeType] = []
            comp_grounding_edges: list[EdgeType] = []
            for e in unique_edges:
                _t, _j, (u, v) = e
                is_comp_edge = u in comp_nodes_set and v in comp_nodes_set
                is_comp_ground_edge = (u in comp_nodes_set and v == world) or (v in comp_nodes_set and u == world)
                if is_comp_edge or is_comp_ground_edge:
                    comp_edges.append(e)
                # NOTE: We add all nodes/edges connecting the component to the world
                # as grounding nodes/edges, without yet distinguishing between base
                # vs grounding edges, as this will be handled later in the base
                # selection step after spanning tree generation.
                if is_comp_ground_edge:
                    comp_grounding_edges.append(e)

            # Assign the base node automatically if a single grounding edge is present
            comp_base_node: NodeType | None = None
            comp_base_edge: EdgeType | None = None
            if len(comp_grounding_edges) == 1:
                # Assign the unique grounding node as the base node,
                # and the unique grounding edge as the base edge
                comp_base_edge = comp_grounding_edges[0]
                comp_base_node = next(n for n in comp_base_edge[2] if n != world)
                # Drop the promoted edge from the grounding list — base and grounding
                # edges are mutually exclusive per the topology conventions
                comp_grounding_edges = []

            # If multiple grounding edges are present, and only one of them is a 6-DoF FREE joint,
            # then assign the node connected to the FREE joint as the base node, and remove it
            # from the grounding lists. If more than one of the grounding edges are FREE joints,
            # then raise an error, as this violates the modelling conventions.
            elif len(comp_grounding_edges) > 1:
                free_grounding_edges = [e for e in comp_grounding_edges if e[0] == JointDoFType.FREE]
                if len(free_grounding_edges) == 1:
                    comp_base_edge = free_grounding_edges[0]
                    comp_base_node = next(n for n in comp_base_edge[2] if n != world)
                    comp_grounding_edges.remove(comp_base_edge)
                elif len(free_grounding_edges) > 1:
                    raise ValueError(
                        f"Component with nodes `{comp_nodes}` has multiple grounding edges `{comp_grounding_edges}` "
                        f"with more than one 6-DoF FREE joint, which violates modelling conventions."
                    )

            # Recompute the grounding-node set from the *final* grounding-edge list. This is the
            # source of truth for `ground_nodes` and avoids the duplicate-removal hazard that
            # arises when a body has multiple grounding edges (e.g. a Stewart platform leg).
            comp_grounding_nodes = sorted({n for _, _, pair in comp_grounding_edges for n in pair if n != world})

            # Add the new component object in the list of graph components
            components.append(
                TopologyComponent(
                    nodes=comp_nodes,
                    edges=comp_edges,
                    ground_nodes=comp_grounding_nodes,
                    ground_edges=comp_grounding_edges,
                    base_node=comp_base_node,
                    base_edge=comp_base_edge,
                    is_island=len(comp_nodes) > 1,
                    is_connected=comp_base_edge is not None or len(comp_grounding_edges) > 0,
                )
            )

        # Return the list of graph components
        msg.debug("components: %s", components)
        return components
