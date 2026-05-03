# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a visualization backend for the topology subsystem.

Ships the default :class:`TopologyGraphVisualizer`, a ``matplotlib``-based and ``networkx``-based
:class:`TopologyGraphVisualizerBase` that renders :class:`TopologyGraph` instances together with
their components, spanning-tree candidates, and selected spanning trees.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import ClassVar

from ..core.bodies import RigidBodyDescriptor
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    EdgeType,
    GraphEdge,
    GraphLabels,
    GraphNode,
    NodeType,
    TopologyComponent,
    TopologyGraphVisualizerBase,
    TopologySpanningTree,
)

###
# Module interface
###

__all__ = [
    "TopologyGraphVisualizer",
]

###
# Interfaces
###


class TopologyGraphVisualizer(TopologyGraphVisualizerBase):
    """
    Default :class:`TopologyGraphVisualizerBase`
    backend using ``networkx`` and ``matplotlib``.

    Edge labels combine a joint-type abbreviation with
    the joint index. :attr:`GraphEdge.joint_type` is
    interpreted strictly as a :class:`JointDoFType` value
    (the kamino DoF-typed enum); validity is enforced at
    edge construction by :class:`GraphEdge`.
    """

    _PALETTE: tuple[str, ...] = (
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
    )
    """Cyclic palette used to color island components."""

    _JOINT_DOF_TYPE_ABBR: ClassVar[dict[int, str]] = {
        JointDoFType.FREE: "FREE",
        JointDoFType.REVOLUTE: "REVL",
        JointDoFType.PRISMATIC: "PRISM",
        JointDoFType.CYLINDRICAL: "CYLN",
        JointDoFType.UNIVERSAL: "UNIV",
        JointDoFType.SPHERICAL: "SPHE",
        JointDoFType.GIMBAL: "GIMB",
        JointDoFType.CARTESIAN: "CART",
        JointDoFType.FIXED: "FIXD",
    }
    """Abbreviation table for the integer values of :class:`JointDoFType`."""

    def __init__(self):
        """Initialize the visualizer with the default joint-type abbreviation table."""
        self._abbr_table = self._JOINT_DOF_TYPE_ABBR

    ###
    # Operations
    ###

    @override
    def render_graph(
        self,
        nodes: list[NodeType],
        edges: list[EdgeType],
        components: list[TopologyComponent],
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        graph_labels: Iterable[GraphLabels] | None = None,
        force_path_labels: bool = False,
        edge_label_offset_pts: float | None = None,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Render ``components`` of the topology graph in a single figure.

        The world node is placed at the global origin and components are
        packed radially around it. Each component is laid out with a
        rooted layered layout when a base node is available (so the base
        anchors closest to the world), or a deterministic Kamada-Kawai
        layout otherwise.

        Args:
            nodes: Body node indices to visualize.
            edges: Joint edges to visualize, in :data:`EdgeType` form.
            components: Components of the topology graph.
            world_node: Index of the implicit world node.
            bodies: Optional body descriptors used to source human-readable
                names when ``graph_labels`` is requested.
            joints: Optional joint descriptors used to enrich edge labels
                with joint names (also used as the edge-name source when
                ``graph_labels`` is requested).
            graph_labels: Optional :data:`GraphLabels` set selecting which
                name-label variants to render. See
                :meth:`TopologyGraphVisualizerBase.render_graph`.
            force_path_labels: When ``True``, preserve full scoped names in
                inline annotations and tables. See
                :meth:`TopologyGraphVisualizerBase.render_graph`.
            edge_label_offset_pts: Perpendicular edge-to-label distance
                in display points. ``None`` (default) uses
                :attr:`_EDGE_LABEL_OFFSET_PTS`.
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.

        Raises:
            ImportError: If :mod:`networkx` or :mod:`matplotlib` are not installed.
        """
        # Attempt to import the required modules.
        try:
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx and matplotlib are required for rendering the topology graph. "
                "Please install them with `pip install networkx matplotlib`."
            ) from e

        # Normalize `graph_labels` to two booleans so the rest of the body can
        # branch on simple flags. Unknown literals are silently ignored.
        modes = set(graph_labels) if graph_labels is not None else set()
        show_inline_names = "inline" in modes
        show_tables_mode = "tables" in modes

        # Pre-compute name maps so both the inline and tables paths share a
        # single source of truth (and the tables-visibility check below can
        # short-circuit when nothing is named).
        joint_name_map = self._build_joint_name_map(joints)
        node_name_map = self._build_node_name_map(nodes, bodies)
        tables_visible = show_tables_mode and (bool(joint_name_map) or bool(node_name_map))

        # When any name-label mode is active, the joint name surfaces via the
        # new variants (inline annotations and/or tables), so we drop the name
        # prefix from the existing on-edge ``name_index_TYPE`` label to keep
        # it compact (`index_TYPE`). When no mode is active the existing
        # behaviour (name prefix included) is preserved.
        joints_for_labels = None if (show_inline_names or show_tables_mode) else joints

        # Set default figure size if not provided. When the tables row is
        # active, we grow the figure vertically by `tables_h` so the graph
        # keeps its requested size and the tables sit underneath without
        # squashing anything. The 2.5-inch height comfortably fits up to
        # ~15 rows at the default cell font size.
        if figsize is None:
            figsize = (12, 12)
        tables_h = 2.5
        if tables_visible:
            figsize = (figsize[0], figsize[1] + tables_h)

        # Coerce the edges to GraphEdge instances and the nodes to raw int indices.
        # NetworkX uses node identity as a hashable key, and `GraphEdge.nodes` is
        # already a `tuple[int, int]`; mixing `GraphNode` and int identities here
        # would silently double up the same body in the resulting graph.
        graph_edges = [GraphEdge.from_input(e) for e in edges]
        int_nodes: list[int] = [int(n) for n in nodes]

        # The world node is only drawn if it appears as an endpoint of any edge,
        # which mirrors the modelling convention that an unreferenced world node
        # should not visually clutter the graph.
        world_in_graph = any(world_node in e.nodes for e in graph_edges)

        # Compute per-component local layouts and their bounding-circle radii. Each
        # entry is `(local_pos, local_radius, lat_extent, is_rooted, base_node)`.
        # `lat_extent` is the layout's half-extent perpendicular to the radial
        # direction, used by `_pack_components` to size the world-anchor ring.
        # The trailing `base_node` slot is normalized to `int | None` so the
        # radial packer can use it as a dict key alongside the int-keyed
        # `local_pos`.
        comp_layouts: list[tuple[dict[int, tuple[float, float]], float, float, bool, int | None]] = []
        for comp in components:
            local_pos, local_radius, lat_extent, is_rooted = self._layout_component(comp, world_node)
            base_idx = int(comp.base_node) if comp.base_node is not None else None
            comp_layouts.append((local_pos, local_radius, lat_extent, is_rooted, base_idx))

        # Pack components radially around the world node. Returns a global position
        # dict keyed by node index, including the world node when applicable.
        pos = self._pack_components(comp_layouts, world_node, world_in_graph)

        # Build a single nx.Graph for drawing. Plain Graph (not MultiGraph) is
        # sufficient because parallel edges between the same body pair are rare.
        G = nx.Graph()
        G.add_nodes_from(int_nodes)
        if world_in_graph:
            G.add_node(world_node)
        for e in graph_edges:
            G.add_edge(*e.nodes)

        # Classify each edge into one of three role buckets via a per-component
        # scan. Compare on `(joint_type, joint_index)` for cheap, unambiguous keys.
        base_keys: set[tuple[int, int]] = set()
        ground_keys: set[tuple[int, int]] = set()
        for comp in components:
            if comp.base_edge is not None:
                base_keys.add((comp.base_edge.joint_type, comp.base_edge.joint_index))
            if comp.ground_edges is not None:
                for ge in comp.ground_edges:
                    ground_keys.add((ge.joint_type, ge.joint_index))

        base_edges_uv: list[tuple[int, int]] = []
        ground_edges_uv: list[tuple[int, int]] = []
        internal_edges_uv: list[tuple[int, int]] = []
        edge_label_map: dict[tuple[int, int], str] = {}
        for e in graph_edges:
            uv = e.nodes
            key = (e.joint_type, e.joint_index)
            if key in base_keys:
                base_edges_uv.append(uv)
            elif key in ground_keys:
                ground_edges_uv.append(uv)
            else:
                internal_edges_uv.append(uv)
            edge_label_map[uv] = self._build_edge_label(e.joint_type, e.joint_index, joints_for_labels)

        # Build per-node styling. Defaults are overwritten by component- and role-
        # specific styling further below; this ordering matters because the base
        # node should win over the generic island styling.
        node_color_map: dict[int, str] = {}
        node_size_map: dict[int, int] = {}
        node_edge_color_map: dict[int, str] = {}
        node_linewidth_map: dict[int, float] = {}

        for n in G.nodes:
            node_color_map[n] = "lightgray"
            node_size_map[n] = 600
            node_edge_color_map[n] = "black"
            node_linewidth_map[n] = 1.0

        if world_in_graph:
            node_color_map[world_node] = "black"
            node_size_map[world_node] = 900
            node_edge_color_map[world_node] = "black"
            node_linewidth_map[world_node] = 1.5

        island_color_map: dict[int, str] = {}
        island_index = 0
        for comp in components:
            if comp.is_island:
                color = self._PALETTE[island_index % len(self._PALETTE)]
                island_color_map[island_index] = color
                for n in comp.nodes:
                    nidx = int(n)
                    node_color_map[nidx] = color
                    node_size_map[nidx] = 700
                island_index += 1
            else:
                # Single-node component: connected vs isolated orphan
                nidx = int(comp.nodes[0])
                if comp.is_connected:
                    node_color_map[nidx] = "grey"
                else:
                    node_color_map[nidx] = "white"
                node_size_map[nidx] = 700

        # Base nodes get a thicker border to mark them as the local root, while
        # keeping their component fill so they remain visually grouped.
        for comp in components:
            if comp.base_node is not None:
                node_linewidth_map[int(comp.base_node)] = 2.5

        # Ensure every node referenced in `pos` has styling — defensive against
        # mismatches between `nodes` and the per-component node lists.
        for n in G.nodes:
            node_color_map.setdefault(n, "lightgray")
            node_size_map.setdefault(n, 600)
            node_edge_color_map.setdefault(n, "black")
            node_linewidth_map.setdefault(n, 1.0)

        # Begin drawing. When tables are active we lay the figure out as a
        # 2-row gridspec (graph on top, tables row beneath), with the tables
        # row split into two equal-width axes (joints left, bodies right).
        fig = plt.figure(figsize=figsize)
        if tables_visible:
            graph_h = figsize[1] - tables_h
            gs = fig.add_gridspec(2, 1, height_ratios=[graph_h, tables_h])
            ax = fig.add_subplot(gs[0, 0])
            gs_tables = gs[1, 0].subgridspec(1, 2, wspace=0.08)
            ax_tables_joints = fig.add_subplot(gs_tables[0, 0])
            ax_tables_bodies = fig.add_subplot(gs_tables[0, 1])
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax_tables_joints = None
            ax_tables_bodies = None

        # Edges first, behind the nodes
        if internal_edges_uv:
            nx.draw_networkx_edges(G, pos, edgelist=internal_edges_uv, width=1.5, edge_color="0.55", ax=ax)
        if ground_edges_uv:
            nx.draw_networkx_edges(
                G, pos, edgelist=ground_edges_uv, width=1.8, style="dashed", edge_color="0.35", ax=ax
            )
        if base_edges_uv:
            nx.draw_networkx_edges(G, pos, edgelist=base_edges_uv, width=2.5, edge_color="black", ax=ax)

        # Nodes — `draw_networkx_nodes` requires per-node lists to be parallel to
        # the supplied `nodelist`, so we iterate in a stable node order.
        ordered_nodes = list(G.nodes)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=ordered_nodes,
            node_color=[node_color_map[n] for n in ordered_nodes],
            node_size=[node_size_map[n] for n in ordered_nodes],
            edgecolors=[node_edge_color_map[n] for n in ordered_nodes],
            linewidths=[node_linewidth_map[n] for n in ordered_nodes],
            ax=ax,
        )

        # Node labels — keep the world node label readable on the dark fill
        node_labels = {n: ("W" if n == world_node else str(n)) for n in G.nodes}
        # Draw world label in white, everything else in black, by splitting the call
        if world_in_graph:
            nx.draw_networkx_labels(G, pos, labels={world_node: "W"}, font_size=10, font_color="white", ax=ax)
            nx.draw_networkx_labels(
                G,
                pos,
                labels={n: lbl for n, lbl in node_labels.items() if n != world_node},
                font_size=10,
                font_color="black",
                ax=ax,
            )
        else:
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="black", ax=ax)

        # Edge labels with a small perpendicular offset so the underlying edge
        # stroke stays visible. Labels go to the opposite side from inline name
        # labels (which always flip "above") to avoid collisions when both
        # annotations are drawn together.
        if edge_label_map:
            self._draw_offset_edge_labels(ax, pos, edge_label_map, font_size=8, offset_pts=edge_label_offset_pts)

        # Optional inline name labels — drawn after the canonical index/type
        # labels so they sit on top of the heavier rendering and never mask it.
        if show_inline_names and (node_name_map or joint_name_map):
            self._draw_inline_names(
                ax,
                pos,
                graph_edges,
                node_name_map=node_name_map,
                joint_name_map=joint_name_map,
                world_node=world_node,
                edge_offset_pts=edge_label_offset_pts,
                full_paths=force_path_labels,
            )

        # Legend
        legend_handles: list = []
        if world_in_graph:
            legend_handles.append(mpatches.Patch(color="black", label=f"world ({world_node})"))
        legend_handles.append(mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.5, label="base node"))
        legend_handles.append(mpatches.Patch(color="grey", label="orphan - connected"))
        legend_handles.append(mpatches.Patch(facecolor="white", edgecolor="black", label="orphan - isolated"))
        for i in range(island_index):
            legend_handles.append(mpatches.Patch(color=island_color_map[i], label=f"island {i}"))
        legend_handles.append(mlines.Line2D([], [], color="black", linewidth=2.5, label="base edge"))
        legend_handles.append(
            mlines.Line2D([], [], color="0.35", linewidth=1.8, linestyle="--", label="grounding edge")
        )
        legend_handles.append(mlines.Line2D([], [], color="0.55", linewidth=1.5, label="internal edge"))
        ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)

        ax.set_axis_off()

        # Optional name-reference tables below the graph. Drawn last so the
        # tight-layout pass below fits both the graph and the tables together.
        if tables_visible and ax_tables_joints is not None and ax_tables_bodies is not None:
            self._draw_name_tables(
                ax_tables_joints,
                ax_tables_bodies,
                joint_name_map=joint_name_map,
                node_name_map=node_name_map,
                full_paths=force_path_labels,
            )

        fig.tight_layout()

        if path is not None:
            fig.savefig(path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    @override
    def render_component_spanning_tree_candidates(
        self,
        component: TopologyComponent,
        candidates: list[TopologySpanningTree],
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        graph_labels: Iterable[GraphLabels] | None = None,
        force_path_labels: bool = False,
        edge_label_offset_pts: float | None = None,
        skip_orphans: bool = True,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Render the candidate spanning trees of a component as a grid of panels.

        The figure has a top "Original Component" panel (using the same
        base / grounding / internal classification as :meth:`render_graph`)
        followed by one panel per candidate (base / arc / chord
        classification). All panels share the same node positions so
        candidates can be compared directly.

        Args:
            component: The component whose candidates are rendered.
            candidates: List of candidate spanning trees; an empty list
                draws only the top panel.
            world_node: Index of the implicit world node.
            bodies: Optional body descriptors used to source human-readable
                names when ``graph_labels`` is requested.
            joints: Optional joint descriptors for name-based edge labels
                (also used as the edge-name source when ``graph_labels`` is
                requested).
            graph_labels: Optional :data:`GraphLabels` set selecting which
                name-label variants to render. See
                :meth:`TopologyGraphVisualizerBase.render_graph`.
            force_path_labels: When ``True``, preserve full scoped names in
                inline annotations and tables. See
                :meth:`TopologyGraphVisualizerBase.render_graph`.
            edge_label_offset_pts: Perpendicular edge-to-label distance
                in display points. ``None`` (default) uses
                :attr:`_EDGE_LABEL_OFFSET_PTS`.
            skip_orphans: When ``True`` (default), orphan components are
                skipped entirely.
            figsize: Optional figure size; defaults are derived from the
                candidate count.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.

        Raises:
            ImportError: If :mod:`networkx` or :mod:`matplotlib` are not installed.
        """
        # Orphan short-circuit: skip components with a single body when requested.
        if skip_orphans and not component.is_island:
            return

        try:
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "networkx and matplotlib are required for rendering the topology graph. "
                "Please install them with `pip install networkx matplotlib`."
            ) from e

        # Normalize `graph_labels` and pre-build name maps so both per-panel
        # inline annotations and the figure-wide tables row share a single
        # source of truth. Tables are scoped to the component being rendered
        # (its own nodes/edges) so they double as a per-figure legend.
        modes = set(graph_labels) if graph_labels is not None else set()
        show_inline_names = "inline" in modes
        show_tables_mode = "tables" in modes
        joint_name_map = self._build_joint_name_map(joints)
        node_name_map = self._build_node_name_map(component.nodes, bodies)
        # Restrict the maps to entries that actually appear in this component
        # so the tables stay focused on the rendered subgraph.
        comp_node_indices = {int(n) for n in (component.nodes or [])}
        comp_joint_indices = {e.joint_index for e in (component.edges or []) if e.joint_index >= 0}
        node_name_map = {i: name for i, name in node_name_map.items() if i in comp_node_indices}
        joint_name_map = {j: name for j, name in joint_name_map.items() if j in comp_joint_indices}
        tables_visible = show_tables_mode and (bool(joint_name_map) or bool(node_name_map))

        # When any name-label mode is active, drop the joint-name prefix from
        # the existing on-edge labels so they stay compact (`index_TYPE`).
        joints_for_labels = None if (show_inline_names or show_tables_mode) else joints

        # Shared layout: every panel positions the component's nodes identically.
        pos, world_in_graph = self._compute_single_component_pos(component, world_node)

        # Pick the panel corner with the most empty space once per call —
        # all candidate panels share the same layout, so the corner choice is
        # the same everywhere. The metadata text overlay below uses these
        # axes-fraction coordinates.
        md_x, md_y, md_ha, md_va = self._pick_metadata_corner(pos)

        # Per-cell size is fixed (so PDF page sizes stay manageable). When the
        # component is large (``n_bodies > 10``), shrink node markers and font
        # sizes so the layout fits each cell. The square-root falloff keeps the
        # shrink gentle: ``n_bodies = 20 -> ~0.71``; ``n_bodies = 40 -> ~0.5``.
        n_bodies = len(component.nodes or [])
        size_scale = 1.0 if n_bodies <= 10 else math.sqrt(10.0 / n_bodies)

        # Font sizes shrink with the component but never below 6 / 7 / 8 pt so
        # labels stay legible at any size.
        edge_label_fs_top = max(6, int(round(8 * size_scale)))
        node_label_fs_top = max(7, int(round(10 * size_scale)))
        edge_label_fs_cand = max(6, int(round(7 * size_scale)))
        node_label_fs_cand = max(7, int(round(9 * size_scale)))
        meta_fs = max(6, int(round(7 * size_scale)))
        title_fs_cand = max(8, int(round(9 * size_scale)))
        title_fs_top = max(9, int(round(11 * size_scale)))

        n = len(candidates)
        # Square-ish grid; 1-4 candidates render in a single row and larger
        # counts wrap to a roughly square grid.
        if n <= 0:
            cols = 1
            rows = 0
        elif n <= 4:
            cols = max(1, n)
            rows = 1
        else:
            cols = max(1, math.ceil(math.sqrt(n)))
            rows = max(1, math.ceil(n / cols))

        # Per-cell sizing — fixed base values chosen to match `render_graph`'s
        # default ``(12, 12)`` figure for a 4-candidate view. Independent of
        # component complexity to keep saved PDFs from ballooning.
        per_w = 4.0
        per_h = 3.5
        top_h = 5.0
        tables_h = 2.5
        extra_h = tables_h if tables_visible else 0.0
        if figsize is None:
            fig_w = max(8.0, cols * per_w)
            fig_h = top_h + rows * per_h + extra_h
            figsize = (fig_w, fig_h)
        elif tables_visible:
            figsize = (figsize[0], figsize[1] + tables_h)

        fig = plt.figure(figsize=figsize)
        # Build the gridspec: top panel + per-candidate rows + optional tables row.
        # The trailing tables row is `gs[-1, :]`; we expand the per-row height
        # ratios in the same units (`per_h`) used by the candidate rows so the
        # tables row is sized like a short candidate slice.
        if rows > 0:
            base_ratios = [top_h / per_h] + [1.0] * rows
        else:
            base_ratios = [1.0]
        if tables_visible:
            height_ratios = [*base_ratios, tables_h / per_h]
            gs = fig.add_gridspec(nrows=len(height_ratios), ncols=cols, height_ratios=height_ratios)
        else:
            gs = fig.add_gridspec(nrows=rows + 1, ncols=cols, height_ratios=base_ratios)

        # Top panel: original component, classified as base / grounding / internal.
        ax_top = fig.add_subplot(gs[0, :] if cols > 1 else gs[0, 0])
        self._draw_component_on_axis(
            ax_top,
            component,
            pos,
            world_node,
            world_in_graph,
            joints_for_labels,
            mode="graph",
            island_color=self._PALETTE[0],
            edge_label_font_size=edge_label_fs_top,
            node_label_font_size=node_label_fs_top,
            node_size_scale=size_scale,
            show_inline_names=show_inline_names,
            node_name_map=node_name_map,
            joint_name_map=joint_name_map,
            force_path_labels=force_path_labels,
            edge_label_offset_pts=edge_label_offset_pts,
        )
        top_title = "Original Component"
        if n == 0:
            top_title += " (no candidates)"
        ax_top.set_title(top_title, fontsize=title_fs_top)
        ax_top.set_axis_off()

        # Candidate panels: one per spanning-tree candidate, sharing `pos`.
        for i, cand in enumerate(candidates):
            r = 1 + i // cols
            c = i % cols
            ax = fig.add_subplot(gs[r, c])
            arc_set = {a for a in (cand.arcs or []) if a >= 0}
            self._draw_component_on_axis(
                ax,
                component,
                pos,
                world_node,
                world_in_graph,
                joints_for_labels,
                mode="candidate",
                arc_joint_indices=arc_set,
                island_color=self._PALETTE[0],
                edge_label_font_size=edge_label_fs_cand,
                node_label_font_size=node_label_fs_cand,
                node_size_scale=size_scale,
                show_inline_names=show_inline_names,
                node_name_map=node_name_map,
                joint_name_map=joint_name_map,
                force_path_labels=force_path_labels,
                edge_label_offset_pts=edge_label_offset_pts,
            )
            # The metadata "table" is a monospace multiline text overlay
            # anchored to the corner with the most empty space (computed
            # once before the loop). Two-space padding between key and value
            # gives the columns a tabular feel.
            md_text = "\n".join(
                [
                    f"depth   {cand.depth}",
                    f"nodes   {cand.num_bodies}",
                    f"arcs    {cand.num_tree_arcs}",
                    f"chords  {cand.num_tree_chords}",
                ]
            )
            ax.text(
                md_x,
                md_y,
                md_text,
                transform=ax.transAxes,
                ha=md_ha,
                va=md_va,
                fontsize=meta_fs,
                family="monospace",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "lw": 0.8, "alpha": 0.9},
            )
            ax.set_title(
                f"Candidate {i}\nroot={cand.root}, traversal={cand.traversal}",
                fontsize=title_fs_cand,
            )
            ax.set_axis_off()

        # Hide unused trailing grid cells when the candidate count does not
        # exactly fill the last row.
        total_cells = rows * cols
        for j in range(n, total_cells):
            r = 1 + j // cols
            c = j % cols
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")

        # Optional name-reference tables — placed in the trailing gridspec row
        # so they sit just above the figure-level legend (which is anchored to
        # the figure bbox via `loc="lower center"`).
        if tables_visible:
            gs_tables = gs[-1, :].subgridspec(1, 2, wspace=0.08)
            ax_tables_joints = fig.add_subplot(gs_tables[0, 0])
            ax_tables_bodies = fig.add_subplot(gs_tables[0, 1])
            self._draw_name_tables(
                ax_tables_joints,
                ax_tables_bodies,
                joint_name_map=joint_name_map,
                node_name_map=node_name_map,
                full_paths=force_path_labels,
            )

        # Figure-level legend — covers both the top-panel and candidate-panel
        # edge-styling vocabularies in a single place.
        legend_handles: list = [
            mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.5, label="base node"),
            mlines.Line2D([], [], color="black", linewidth=2.5, label="base edge"),
            mlines.Line2D([], [], color="0.35", linewidth=1.8, linestyle="--", label="grounding edge (top)"),
            mlines.Line2D([], [], color="0.55", linewidth=1.5, label="internal edge (top)"),
            mlines.Line2D([], [], color="0.35", linewidth=1.8, label="arc edge (candidate)"),
            mlines.Line2D([], [], color="tab:red", linewidth=1.5, linestyle="--", label="chord edge (candidate)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            fontsize=9,
            framealpha=0.9,
        )

        fig.suptitle(f"Spanning-tree candidates ({n} shown)", fontsize=12)
        # Reserve room at the top for the suptitle and at the bottom for the legend.
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])

        if path is not None:
            fig.savefig(path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    @override
    def render_component_spanning_tree(
        self,
        component: TopologyComponent,
        tree: TopologySpanningTree,
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        bodies: list[RigidBodyDescriptor] | None = None,
        joints: list[JointDescriptor] | None = None,
        graph_labels: Iterable[GraphLabels] | None = None,
        force_path_labels: bool = False,
        edge_label_offset_pts: float | None = None,
        skip_orphans: bool = True,
        figsize: tuple[int, int] | None = None,
        path: str | None = None,
        show: bool = False,
    ) -> None:
        """Render a component's selected spanning tree side by side with the original.

        The figure has a top "Original Component" panel and a bottom
        "Selected Spanning Tree" panel sharing the same node positions.
        A monospace metadata overlay summarizes the selected tree.

        Args:
            component: The component the tree belongs to.
            tree: The selected spanning tree to render.
            world_node: Index of the implicit world node.
            bodies: Optional body descriptors used to source human-readable
                names when ``graph_labels`` is requested.
            joints: Optional joint descriptors for name-based edge labels
                (also used as the edge-name source when ``graph_labels`` is
                requested).
            graph_labels: Optional :data:`GraphLabels` set selecting which
                name-label variants to render. See
                :meth:`TopologyGraphVisualizerBase.render_graph`.
            force_path_labels: When ``True``, preserve full scoped names in
                inline annotations and tables. See
                :meth:`TopologyGraphVisualizerBase.render_graph`.
            edge_label_offset_pts: Perpendicular edge-to-label distance
                in display points. ``None`` (default) uses
                :attr:`_EDGE_LABEL_OFFSET_PTS`.
            skip_orphans: When ``True`` (default), orphan components are
                skipped entirely.
            figsize: Optional figure size.
            path: Optional file path to save the figure.
            show: When ``True``, display the figure immediately.

        Raises:
            ImportError: If :mod:`networkx` or :mod:`matplotlib` are not installed.
        """
        if skip_orphans and not component.is_island:
            return
        try:
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "networkx and matplotlib are required for rendering the topology graph. "
                "Please install them with `pip install networkx matplotlib`."
            ) from e

        # Normalize `graph_labels` and pre-build name maps; restrict them to
        # entries that belong to this component so the tables stay focused on
        # the rendered subgraph.
        modes = set(graph_labels) if graph_labels is not None else set()
        show_inline_names = "inline" in modes
        show_tables_mode = "tables" in modes
        joint_name_map = self._build_joint_name_map(joints)
        node_name_map = self._build_node_name_map(component.nodes, bodies)
        comp_node_indices = {int(n) for n in (component.nodes or [])}
        comp_joint_indices = {e.joint_index for e in (component.edges or []) if e.joint_index >= 0}
        node_name_map = {i: name for i, name in node_name_map.items() if i in comp_node_indices}
        joint_name_map = {j: name for j, name in joint_name_map.items() if j in comp_joint_indices}
        tables_visible = show_tables_mode and (bool(joint_name_map) or bool(node_name_map))

        # When any name-label mode is active, drop the joint-name prefix from
        # the existing on-edge labels so they stay compact (`index_TYPE`).
        joints_for_labels = None if (show_inline_names or show_tables_mode) else joints

        pos, world_in_graph = self._compute_single_component_pos(component, world_node)
        md_x, md_y, md_ha, md_va = self._pick_metadata_corner(pos)

        n_bodies = len(component.nodes or [])
        size_scale = 1.0 if n_bodies <= 10 else math.sqrt(10.0 / n_bodies)

        edge_label_fs_top = max(6, int(round(8 * size_scale)))
        node_label_fs_top = max(7, int(round(10 * size_scale)))
        edge_label_fs_tree = max(6, int(round(7 * size_scale)))
        node_label_fs_tree = max(7, int(round(9 * size_scale)))
        meta_fs = max(6, int(round(7 * size_scale)))
        title_fs_tree = max(8, int(round(9 * size_scale)))
        title_fs_top = max(9, int(round(11 * size_scale)))

        top_h = 5.0
        bot_h = 5.0
        tables_h = 2.5
        extra_h = tables_h if tables_visible else 0.0
        if figsize is None:
            figsize = (8.0, top_h + bot_h + extra_h)
        elif tables_visible:
            figsize = (figsize[0], figsize[1] + tables_h)

        fig = plt.figure(figsize=figsize)
        if tables_visible:
            gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[top_h, bot_h, tables_h])
        else:
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[top_h, bot_h])

        ax_top = fig.add_subplot(gs[0, 0])
        self._draw_component_on_axis(
            ax_top,
            component,
            pos,
            world_node,
            world_in_graph,
            joints_for_labels,
            mode="graph",
            island_color=self._PALETTE[0],
            edge_label_font_size=edge_label_fs_top,
            node_label_font_size=node_label_fs_top,
            node_size_scale=size_scale,
            show_inline_names=show_inline_names,
            node_name_map=node_name_map,
            joint_name_map=joint_name_map,
            force_path_labels=force_path_labels,
            edge_label_offset_pts=edge_label_offset_pts,
        )
        ax_top.set_title("Original Component", fontsize=title_fs_top)
        ax_top.set_axis_off()

        ax_bot = fig.add_subplot(gs[1, 0])
        arc_set = {a for a in (tree.arcs or []) if a >= 0}
        self._draw_component_on_axis(
            ax_bot,
            component,
            pos,
            world_node,
            world_in_graph,
            joints_for_labels,
            mode="candidate",
            arc_joint_indices=arc_set,
            island_color=self._PALETTE[0],
            edge_label_font_size=edge_label_fs_tree,
            node_label_font_size=node_label_fs_tree,
            node_size_scale=size_scale,
            show_inline_names=show_inline_names,
            node_name_map=node_name_map,
            joint_name_map=joint_name_map,
            force_path_labels=force_path_labels,
            edge_label_offset_pts=edge_label_offset_pts,
        )

        md_text = "\n".join(
            [
                f"root      {tree.root}",
                f"traversal {tree.traversal}",
                f"directed  {tree.directed}",
                f"depth     {tree.depth}",
                f"nodes     {tree.num_bodies}",
                f"arcs      {tree.num_tree_arcs}",
                f"chords    {tree.num_tree_chords}",
            ]
        )
        ax_bot.text(
            md_x,
            md_y,
            md_text,
            transform=ax_bot.transAxes,
            ha=md_ha,
            va=md_va,
            fontsize=meta_fs,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "lw": 0.8, "alpha": 0.9},
        )
        ax_bot.set_title(
            f"Selected Spanning Tree\nroot={tree.root}, traversal={tree.traversal}",
            fontsize=title_fs_tree,
        )
        ax_bot.set_axis_off()

        # Optional name-reference tables — placed in the trailing gridspec row
        # so they sit just above the figure-level legend.
        if tables_visible:
            gs_tables = gs[2, 0].subgridspec(1, 2, wspace=0.08)
            ax_tables_joints = fig.add_subplot(gs_tables[0, 0])
            ax_tables_bodies = fig.add_subplot(gs_tables[0, 1])
            self._draw_name_tables(
                ax_tables_joints,
                ax_tables_bodies,
                joint_name_map=joint_name_map,
                node_name_map=node_name_map,
                full_paths=force_path_labels,
            )

        legend_handles: list = [
            mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.5, label="base node"),
            mlines.Line2D([], [], color="black", linewidth=2.5, label="base edge"),
            mlines.Line2D([], [], color="0.35", linewidth=1.8, linestyle="--", label="grounding edge (top)"),
            mlines.Line2D([], [], color="0.55", linewidth=1.5, label="internal edge (top)"),
            mlines.Line2D([], [], color="0.35", linewidth=1.8, label="arc edge (tree)"),
            mlines.Line2D([], [], color="tab:red", linewidth=1.5, linestyle="--", label="chord edge (tree)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            fontsize=9,
            framealpha=0.9,
        )

        fig.suptitle("Selected Spanning Tree", fontsize=12)
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])

        if path is not None:
            fig.savefig(path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    ###
    # Internals
    ###

    @staticmethod
    def _layout_component(
        component: TopologyComponent,
        world_node: int,
    ) -> tuple[dict[int, tuple[float, float]], float, float, bool]:
        """Compute a per-component layout in component-local coordinates.

        Returns ``(local_pos, local_radius, lat_extent, is_rooted)``.
        ``is_rooted`` is ``True`` when the layout grows along the local
        ``+x`` axis from the base node, enabling the radial packer to
        anchor the base toward the world. The position dict is keyed by
        raw int body index so it stays interoperable with NetworkX edge
        endpoints. ``lat_extent`` is the half-extent perpendicular to
        the radial direction used by :meth:`_pack_components` to size
        the world-anchor ring without overlap.

        Args:
            component: The component subgraph to lay out.
            world_node: The implicit world node index (skipped during layout).

        Returns:
            A ``(local_pos, local_radius, lat_extent, is_rooted)`` tuple.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for rendering the topology graph. Please install it with `pip install networkx`."
            ) from e

        # Coerce the canonical `GraphNode` storage into raw int indices so the
        # NetworkX graph and the returned position dict use a single key type.
        comp_nodes: list[int] = [int(n) for n in component.nodes] if component.nodes else []
        if not comp_nodes:
            return {}, 0.0, 0.0, False

        if len(comp_nodes) == 1:
            # Floor the single-node lateral extent so it still claims a
            # sensible angular slot during radial packing.
            return {comp_nodes[0]: (0.0, 0.0)}, 0.0, 0.5, False

        # Build the undirected internal subgraph (skip world endpoints).
        internal_pairs: list[tuple[int, int]] = []
        if component.edges:
            for e in component.edges:
                u, v = e.nodes
                if u == world_node or v == world_node:
                    continue
                internal_pairs.append((u, v))

        if component.base_node is not None:
            local_pos = TopologyGraphVisualizer._rooted_layered_layout(
                comp_nodes, internal_pairs, root=int(component.base_node)
            )
            is_rooted = True
        else:
            sub = nx.Graph()
            sub.add_nodes_from(comp_nodes)
            sub.add_edges_from(internal_pairs)
            try:
                local_pos = nx.kamada_kawai_layout(sub)
            except Exception:
                # Kamada-Kawai requires a connected graph and at least 2 nodes;
                # fall back to a deterministic spring layout if it fails.
                local_pos = nx.spring_layout(sub, seed=42)
            is_rooted = False

        # Bounding-circle radius with a small floor so degenerate layouts still
        # take an angular slot during radial packing.
        max_r = 0.0
        for x, y in local_pos.values():
            max_r = max(max_r, (x * x + y * y) ** 0.5)
        local_radius = max(max_r, 0.5)

        # Lateral half-extent — used by the packer to enforce angular non-overlap
        # at the world anchor. Rooted layouts grow along local +x from the base,
        # so only `max(|y|)` (perpendicular to the radial direction) matters.
        # Unrooted layouts have no preferred direction; use the bounding-circle
        # radius as a conservative omnidirectional bound.
        if is_rooted:
            max_y = 0.0
            for _, y in local_pos.values():
                max_y = max(max_y, abs(y))
            lat_extent = max(max_y, 0.5)
        else:
            lat_extent = local_radius

        return local_pos, local_radius, lat_extent, is_rooted

    @staticmethod
    def _rooted_layered_layout(
        nodes: list[int],
        pairs: list[tuple[int, int]],
        root: int,
    ) -> dict[int, tuple[float, float]]:
        """Compute a layered BFS layout rooted at ``root`` growing along ``+x``.

        The root is placed at the local origin; children at depth ``d``
        are placed at ``x = d`` and distributed symmetrically around
        ``y = 0``. Unreachable nodes are appended to the deepest layer
        to ensure every node receives a position.
        """
        adj: dict[int, list[int]] = {n: [] for n in nodes}
        for u, v in pairs:
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
        for n, neighbors in adj.items():
            adj[n] = sorted(set(neighbors))

        # BFS from the root to assign integer depths
        depth: dict[int, int] = {root: 0}
        order: list[int] = [root]
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    order.append(v)
                    q.append(v)

        # Append any unreachable nodes at the deepest layer + 1 so every node gets a position
        max_depth = max(depth.values(), default=0)
        for n in nodes:
            if n not in depth:
                depth[n] = max_depth + 1
                order.append(n)

        # Group nodes by depth, preserving BFS discovery order within each layer
        layers: dict[int, list[int]] = defaultdict(list)
        for n in order:
            layers[depth[n]].append(n)

        # Lateral spacing chosen so total layer width matches the layer count, which
        # keeps the layout's aspect ratio roughly square as depth grows.
        local_pos: dict[int, tuple[float, float]] = {}
        x_step = 1.0
        y_step = 1.0
        for d, members in layers.items():
            count = len(members)
            for i, n in enumerate(members):
                # Center each layer around y = 0
                y = (i - (count - 1) / 2.0) * y_step
                local_pos[n] = (d * x_step, y)
        return local_pos

    @staticmethod
    def _pack_components(
        comp_layouts: list[tuple[dict[int, tuple[float, float]], float, float, bool, int | None]],
        world_node: int,
        world_in_graph: bool,
    ) -> dict[int, tuple[float, float]]:
        """Pack per-component local layouts radially around the world node.

        Each component is given a uniform ``2*pi / N`` angular slot, and
        the world-anchor ring radius ``R`` is chosen as the smallest value
        such that adjacent components do not angularly overlap at their
        base point — i.e. ``R >= max_lat / tan(pi / N)``, where
        ``max_lat`` is the largest per-component lateral half-extent. This
        keeps the world-to-base edges at the same visual scale as the
        intra-component edges (typically ``~1.0`` unit) for small ``N``,
        only growing ``R`` when the angular non-overlap constraint binds.

        Args:
            comp_layouts: Per-component
                ``(local_pos, local_radius, lat_extent, is_rooted, base_node)`` tuples.
            world_node: The implicit world node index.
            world_in_graph: When ``True``, place the world at the origin.

        Returns:
            A global ``node -> (x, y)`` position map; the world node, when
            applicable, is at ``(0, 0)``.
        """
        pos: dict[int, tuple[float, float]] = {}
        if world_in_graph:
            pos[world_node] = (0.0, 0.0)

        n_components = len(comp_layouts)
        if n_components == 0:
            return pos

        # Stable ordering by descending lateral extent — biggest components claim
        # their slots first so subsequent index-based rotation is deterministic.
        order = sorted(range(n_components), key=lambda i: -comp_layouts[i][2])

        # Pick `R` from the angular non-overlap constraint, with a unit floor so
        # the world-to-base edge stays visible even for tiny single-component
        # graphs. For `N == 1` `tan(pi / N) = tan(pi) = 0`, which would force
        # `R -> inf`; treat it specially with the floor.
        max_lat = max(comp_layouts[i][2] for i in range(n_components))
        if n_components == 1:
            R = 1.0
        else:
            R = max(1.0, max_lat / max(math.tan(math.pi / n_components), 1e-3))

        # Uniform angular slots — each component sits at the slot midpoint.
        slot = 2.0 * math.pi / n_components

        for sort_idx, comp_idx in enumerate(order):
            theta = sort_idx * slot

            local_pos, local_radius, _lat_extent, is_rooted, base_node = comp_layouts[comp_idx]

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            if is_rooted and base_node is not None and base_node in local_pos:
                # Rotate the local frame so local +x points outward at angle theta,
                # and translate so the base node lands on the ring at radius R.
                bx, by = local_pos[base_node]
                # Translation so base goes to (R*cos_t, R*sin_t) after rotation
                tx = R * cos_t
                ty = R * sin_t
                for n, (lx, ly) in local_pos.items():
                    rx = (lx - bx) * cos_t - (ly - by) * sin_t
                    ry = (lx - bx) * sin_t + (ly - by) * cos_t
                    pos[n] = (rx + tx, ry + ty)
            else:
                # Unrooted: rotate by theta and translate the layout's geometric
                # center (mean of local positions) onto the ring.
                if local_pos:
                    cx = sum(p[0] for p in local_pos.values()) / len(local_pos)
                    cy = sum(p[1] for p in local_pos.values()) / len(local_pos)
                else:
                    cx = cy = 0.0
                # Anchor distance pushes the component's center outward by
                # (R + local_radius) so even unrooted layouts respect the ring.
                anchor = R + 0.25 * local_radius
                tx = anchor * cos_t
                ty = anchor * sin_t
                for n, (lx, ly) in local_pos.items():
                    rx = (lx - cx) * cos_t - (ly - cy) * sin_t
                    ry = (lx - cx) * sin_t + (ly - cy) * cos_t
                    pos[n] = (rx + tx, ry + ty)

        return pos

    @staticmethod
    def _compute_single_component_pos(
        component: TopologyComponent,
        world_node: int,
    ) -> tuple[dict[int, tuple[float, float]], bool]:
        """Compute a node-position dict for a single-component figure.

        Reuses :meth:`_layout_component` to obtain body-node positions in
        local component coordinates and anchors the world node next to
        the layout when the component has any world-incident edge.

        Args:
            component: The component subgraph whose nodes need positions.
            world_node: The implicit world node index.

        Returns:
            A ``(pos, world_in_graph)`` pair; ``pos`` includes the world
            node only when ``world_in_graph`` is ``True``.
        """
        local_pos, _local_radius, _lat_extent, is_rooted = TopologyGraphVisualizer._layout_component(
            component, world_node
        )

        # Draw the world only when the component actually references it
        # via an edge endpoint, mirroring `render_graph`.
        world_in_graph = any(world_node in e.nodes for e in component.edges or [])

        pos: dict[int, tuple[float, float]] = dict(local_pos)
        if not world_in_graph:
            return pos, False

        base_idx = int(component.base_node) if component.base_node is not None else None
        if is_rooted and base_idx is not None and base_idx in local_pos:
            # Place the world one unit to the left of the base so the
            # rooted local +x axis points outward from the world.
            bx, by = local_pos[base_idx]
            pos[world_node] = (bx - 1.0, by)
        else:
            # Drop the world below the layout's bounding box; offsets
            # match `_rooted_layered_layout` so the world stays adjacent
            # without overlapping any body node.
            xs = [p[0] for p in local_pos.values()] if local_pos else [0.0]
            ys = [p[1] for p in local_pos.values()] if local_pos else [0.0]
            cx = (min(xs) + max(xs)) / 2.0
            min_y = min(ys)
            pos[world_node] = (cx, min_y - 1.0)

        return pos, True

    @staticmethod
    def _pick_metadata_corner(
        pos: dict[int, tuple[float, float]],
    ) -> tuple[float, float, str, str]:
        """Pick the panel corner with the most empty space for a metadata overlay.

        Returns the axes-fraction position and matplotlib alignment strings
        for the corner whose nearest node (in node-position-normalized
        coordinates) is the furthest. Top-right is preferred on ties.

        Args:
            pos: Shared node-position map for the component.

        Returns:
            A ``(x_frac, y_frac, ha, va)`` tuple ready for
            :meth:`matplotlib.axes.Axes.text` with ``transform=ax.transAxes``.
        """
        if not pos:
            return (0.98, 0.98, "right", "top")

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # Floor the ranges so degenerate (collinear) layouts still produce a
        # well-defined normalized coordinate space.
        xrange = max(xmax - xmin, 1e-9)
        yrange = max(ymax - ymin, 1e-9)

        # Normalize all node positions to ``[0, 1] x [0, 1]`` so x and y get
        # equal weight in the corner-distance metric (matplotlib stretches the
        # axes to the cell aspect ratio anyway).
        norm = [((x - xmin) / xrange, (y - ymin) / yrange) for x, y in pos.values()]

        # Corners are evaluated in priority order so ties break in favour of
        # the most natural reading corner (top-right, then top-left).
        corners = (
            (1.0, 1.0, 0.98, 0.98, "right", "top"),
            (0.0, 1.0, 0.02, 0.98, "left", "top"),
            (1.0, 0.0, 0.98, 0.02, "right", "bottom"),
            (0.0, 0.0, 0.02, 0.02, "left", "bottom"),
        )
        best_dist = -1.0
        best = corners[0]
        for entry in corners:
            cx, cy = entry[0], entry[1]
            d = min(math.hypot(cx - nx, cy - ny) for nx, ny in norm)
            if d > best_dist:
                best_dist = d
                best = entry
        return (best[2], best[3], best[4], best[5])

    def _build_edge_label(
        self,
        joint_type: int,
        joint_index: int,
        joints: list[JointDescriptor] | None,
    ) -> str:
        """Build an edge label of the form ``f"{name}_{index}_{type}"`` (or shorter).

        The joint-name prefix is included when ``joints[joint_index].name`` is
        a non-empty string. The joint-type suffix is included for any
        :class:`JointDoFType` value and omitted for the ``-1`` "unspecified"
        sentinel. ``joint_type`` is assumed pre-validated by :class:`GraphEdge`.

        Args:
            joint_type: Integer matching a :class:`JointDoFType` value, or
                ``-1`` for unspecified.
            joint_index: Global joint index used both as the body of the
                label and to look up an optional name.
            joints: Optional joint descriptor list; out-of-range indices and
                missing names fall back to the index-only label.

        Returns:
            A non-empty string label.
        """
        abbr: str | None = self._abbr_table.get(joint_type) if joint_type != -1 else None

        name: str | None = None
        if joints is not None and 0 <= joint_index < len(joints):
            descriptor = joints[joint_index]
            candidate = getattr(descriptor, "name", None)
            if isinstance(candidate, str) and candidate:
                name = candidate

        prefix = f"{name}_" if name is not None else ""
        suffix = f"_{abbr}" if abbr is not None else ""
        return f"{prefix}{joint_index}{suffix}"

    def _draw_component_on_axis(
        self,
        ax,
        component: TopologyComponent,
        pos: dict[int, tuple[float, float]],
        world_node: int,
        world_in_graph: bool,
        joints: list[JointDescriptor] | None,
        *,
        mode: str,
        arc_joint_indices: set[int] | None = None,
        island_color: str = "tab:blue",
        show_edge_labels: bool = True,
        edge_label_font_size: int = 8,
        node_label_font_size: int = 10,
        node_size_scale: float = 1.0,
        show_inline_names: bool = False,
        node_name_map: dict[int, str] | None = None,
        joint_name_map: dict[int, str] | None = None,
        force_path_labels: bool = False,
        edge_label_offset_pts: float | None = None,
    ) -> None:
        """Draw a single component on a matplotlib axis using a shared ``pos`` dict.

        Single source of truth for the per-component drawing vocabulary
        used by the candidate-tree renderer. Mirrors the styling choices
        of :meth:`render_graph` so that figures stay visually consistent.

        Args:
            ax: The matplotlib axis to draw on.
            component: The component to render.
            pos: Node-to-``(x, y)`` map covering every drawn node.
            world_node: Index of the implicit world node.
            world_in_graph: Draw the world node when ``True``.
            joints: Optional joint descriptors for name-based edge labels.
            mode: ``"graph"`` (base / grounding / internal classification)
                or ``"candidate"`` (base / arc / chord classification).
            arc_joint_indices: Required in ``"candidate"`` mode; the set
                of global joint indices that belong to the spanning tree.
            island_color: Fill color for island bodies.
            show_edge_labels: When ``True``, draw per-edge text labels.
            edge_label_font_size: Font size for edge labels.
            node_label_font_size: Font size for node labels.
            node_size_scale: Multiplier applied to all node marker sizes.
            show_inline_names: When ``True`` and either ``node_name_map``
                or ``joint_name_map`` is non-empty, draw tiny on-graph
                name annotations beside named nodes/edges via
                :meth:`_draw_inline_names`.
            node_name_map: ``{body_index: name}`` map (e.g. from
                :meth:`_build_node_name_map`); used when
                ``show_inline_names`` is ``True``.
            joint_name_map: ``{joint_index: name}`` map (e.g. from
                :meth:`_build_joint_name_map`); used when
                ``show_inline_names`` is ``True``.
            force_path_labels: When ``True``, preserve full scoped names
                in inline annotations instead of clipping to ``…/leaf``
                via :meth:`_format_name`.
            edge_label_offset_pts: Perpendicular distance, in display
                points, between each edge and its labels. ``None``
                (default) uses :attr:`_EDGE_LABEL_OFFSET_PTS`.

        Raises:
            ImportError: If :mod:`networkx` or :mod:`matplotlib` are not installed.
            ValueError: If ``mode`` is not one of ``"graph"`` or
                ``"candidate"``, or ``arc_joint_indices`` is missing in
                ``"candidate"`` mode.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for rendering the topology graph. Please install it with `pip install networkx`."
            ) from e

        if mode not in ("graph", "candidate"):
            raise ValueError(f"Unsupported mode `{mode!r}`: expected one of `'graph'` or `'candidate'`.")
        if mode == "candidate" and arc_joint_indices is None:
            raise ValueError("`arc_joint_indices` must be provided when `mode='candidate'`.")

        # Plain `nx.Graph` (not MultiGraph) keeps the per-edge styling
        # logic simple; parallel edges between the same body pair are rare.
        # Node identities are coerced to int so they line up with the
        # `tuple[int, int]` endpoints carried by every `GraphEdge`.
        G = nx.Graph()
        if component.nodes:
            G.add_nodes_from(int(n) for n in component.nodes)
        if world_in_graph:
            G.add_node(world_node)
        for e in component.edges or []:
            G.add_edge(*e.nodes)

        base_key: tuple[int, int] | None = None
        if component.base_edge is not None:
            base_key = (component.base_edge.joint_type, component.base_edge.joint_index)

        ground_keys: set[tuple[int, int]] = set()
        if mode == "graph" and component.ground_edges is not None:
            for ge in component.ground_edges:
                ground_keys.add((ge.joint_type, ge.joint_index))

        # Classify each edge into one of three buckets and build a single
        # edge-label map keyed by `(u, v)`.
        base_edges_uv: list[tuple[int, int]] = []
        secondary_edges_uv: list[tuple[int, int]] = []  # grounding (graph) or arc (candidate)
        tertiary_edges_uv: list[tuple[int, int]] = []  # internal (graph) or chord (candidate)
        edge_label_map: dict[tuple[int, int], str] = {}
        for e in component.edges or []:
            uv = e.nodes
            key = (e.joint_type, e.joint_index)
            if base_key is not None and key == base_key:
                base_edges_uv.append(uv)
            elif mode == "graph":
                if key in ground_keys:
                    secondary_edges_uv.append(uv)
                else:
                    tertiary_edges_uv.append(uv)
            else:
                # Candidate mode: arc vs chord. The base edge is already
                # peeled off above, so the membership check is unambiguous.
                if e.joint_index in arc_joint_indices:  # type: ignore[operator]
                    secondary_edges_uv.append(uv)
                else:
                    tertiary_edges_uv.append(uv)
            edge_label_map[uv] = self._build_edge_label(e.joint_type, e.joint_index, joints)

        # Edges first, behind the nodes. The ordering (tertiary → secondary →
        # base) puts the most prominent style on top, matching `render_graph`.
        if tertiary_edges_uv:
            if mode == "graph":
                nx.draw_networkx_edges(G, pos, edgelist=tertiary_edges_uv, width=1.5, edge_color="0.55", ax=ax)
            else:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=tertiary_edges_uv,
                    width=1.5,
                    style="dashed",
                    edge_color="tab:red",
                    alpha=0.7,
                    ax=ax,
                )
        if secondary_edges_uv:
            if mode == "graph":
                nx.draw_networkx_edges(
                    G, pos, edgelist=secondary_edges_uv, width=1.8, style="dashed", edge_color="0.35", ax=ax
                )
            else:
                nx.draw_networkx_edges(G, pos, edgelist=secondary_edges_uv, width=1.8, edge_color="0.35", ax=ax)
        if base_edges_uv:
            nx.draw_networkx_edges(G, pos, edgelist=base_edges_uv, width=2.5, edge_color="black", ax=ax)

        # Per-node styling — mirrors `render_graph`. Defaults are set first and
        # then overridden by component- and role-specific styling so the base
        # node always wins over the generic island styling. All sizes are
        # multiplied by `node_size_scale` so the candidate-grid renderer can
        # keep markers proportional to per-cell physical size.
        size_default = max(1, int(round(500 * node_size_scale)))
        size_world = max(1, int(round(400 * node_size_scale)))
        size_member = max(1, int(round(300 * node_size_scale)))

        node_color_map: dict[int, str] = {}
        node_size_map: dict[int, int] = {}
        node_edge_color_map: dict[int, str] = {}
        node_linewidth_map: dict[int, float] = {}
        for n in G.nodes:
            node_color_map[n] = "lightgray"
            node_size_map[n] = size_default
            node_edge_color_map[n] = "black"
            node_linewidth_map[n] = 1.0

        if world_in_graph:
            node_color_map[world_node] = "black"
            node_size_map[world_node] = size_world
            node_edge_color_map[world_node] = "black"
            node_linewidth_map[world_node] = 1.5

        if component.is_island:
            for n in component.nodes or []:
                nidx = int(n)
                node_color_map[nidx] = island_color
                node_size_map[nidx] = size_member
        elif component.nodes:
            nidx = int(component.nodes[0])
            if component.is_connected:
                node_color_map[nidx] = "grey"
            else:
                node_color_map[nidx] = "white"
            node_size_map[nidx] = size_member

        if component.base_node is not None:
            node_linewidth_map[int(component.base_node)] = 2.5

        ordered_nodes = list(G.nodes)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=ordered_nodes,
            node_color=[node_color_map[n] for n in ordered_nodes],
            node_size=[node_size_map[n] for n in ordered_nodes],
            edgecolors=[node_edge_color_map[n] for n in ordered_nodes],
            linewidths=[node_linewidth_map[n] for n in ordered_nodes],
            ax=ax,
        )

        # Node labels — keep the world label readable on its dark fill.
        if world_in_graph:
            nx.draw_networkx_labels(
                G, pos, labels={world_node: "W"}, font_size=node_label_font_size, font_color="white", ax=ax
            )
            other_labels = {n: str(n) for n in G.nodes if n != world_node}
            nx.draw_networkx_labels(
                G, pos, labels=other_labels, font_size=node_label_font_size, font_color="black", ax=ax
            )
        else:
            node_labels = {n: str(n) for n in G.nodes}
            nx.draw_networkx_labels(
                G, pos, labels=node_labels, font_size=node_label_font_size, font_color="black", ax=ax
            )

        if show_edge_labels and edge_label_map:
            self._draw_offset_edge_labels(
                ax,
                pos,
                edge_label_map,
                font_size=edge_label_font_size,
                offset_pts=edge_label_offset_pts,
            )

        # Optional inline name labels — drawn after the canonical labels so they
        # sit on top of the heavier rendering and never mask it. Edge entries
        # iterate over the component's edges (which is what `pos` covers).
        if show_inline_names and (node_name_map or joint_name_map):
            self._draw_inline_names(
                ax,
                pos,
                component.edges or [],
                node_name_map=node_name_map or {},
                joint_name_map=joint_name_map or {},
                world_node=world_node,
                edge_offset_pts=edge_label_offset_pts,
                full_paths=force_path_labels,
            )

    @staticmethod
    def _build_node_name_map(
        nodes: Iterable[NodeType] | None,
        bodies: list[RigidBodyDescriptor] | None,
    ) -> dict[int, str]:
        """Aggregate body-index → human-readable-name mappings from both sources.

        ``bodies[i].name`` (when non-empty) overrides any ``GraphNode.name``
        sourced from ``nodes``, since the descriptor list is the canonical
        body inventory and the per-node attribute is a free-form override.

        Args:
            nodes: Optional iterable of :data:`NodeType` providing
                ``GraphNode.name`` overrides keyed by body index.
            bodies: Optional list of :class:`RigidBodyDescriptor` whose
                ``name`` field is used as the primary name source.

        Returns:
            A ``{body_index: name}`` dict containing only entries with a
            non-empty name. Empty when no source provides any name.
        """
        name_map: dict[int, str] = {}
        if nodes is not None:
            for n in nodes:
                if isinstance(n, GraphNode) and isinstance(n.name, str) and n.name:
                    name_map[int(n)] = n.name
        if bodies is not None:
            for i, body in enumerate(bodies):
                if body is None:
                    continue
                name = getattr(body, "name", None)
                if isinstance(name, str) and name:
                    name_map[i] = name
        return name_map

    @staticmethod
    def _build_joint_name_map(
        joints: list[JointDescriptor] | None,
    ) -> dict[int, str]:
        """Aggregate joint-index → human-readable-name mappings.

        Args:
            joints: Optional list of :class:`JointDescriptor` whose ``name``
                field is the joint-name source.

        Returns:
            A ``{joint_index: name}`` dict containing only entries with a
            non-empty name. Empty when ``joints`` is ``None`` or every
            entry has an empty name.
        """
        name_map: dict[int, str] = {}
        if joints is None:
            return name_map
        for i, descriptor in enumerate(joints):
            if descriptor is None:
                continue
            name = getattr(descriptor, "name", None)
            if isinstance(name, str) and name:
                name_map[i] = name
        return name_map

    # Default perpendicular offset (display points) used by both
    # `_draw_offset_edge_labels` and `_draw_inline_names` to push their labels
    # off the edge midpoint. Expressed in points (1 pt = 1/72 inch, matching
    # matplotlib's standard typographic unit) so the visual gap stays
    # constant regardless of axes aspect ratio or figure size. Sized to clear
    # the default ``font_size=8`` label height (~10 pt) while keeping the
    # label visually anchored to its edge.
    _EDGE_LABEL_OFFSET_PTS: ClassVar[float] = 8.0

    @staticmethod
    def _estimate_label_radius_pts(
        text: str,
        font_size: int,
        *,
        bbox_pad_factor: float = 0.15,
        char_aspect: float = 0.55,
        line_height: float = 1.2,
    ) -> float:
        """Estimate a bounding-circle radius (display points) for a label.

        Approximates the rendered size of a matplotlib text label without
        having to call the renderer. Used by
        :meth:`_resolve_label_parallel_shifts` as the collision-test
        radius for both primary edge labels and inline name labels.

        Width is approximated as ``len(text) x font_size x char_aspect``,
        height as ``font_size x line_height``, then both are inflated by
        ``2 x bbox_pad_factor x font_size`` to account for the rounded
        bbox padding used on every label drawn by the visualizer. The
        returned radius is half the diagonal of the resulting rectangle,
        which over-approximates the worst-case rotation footprint while
        still giving a tight-enough bound to pack closely-spaced labels.

        Args:
            text: The label string.
            font_size: Font size in points.
            bbox_pad_factor: Bbox padding as a fraction of ``font_size``
                (matplotlib convention). Defaults to ``0.15`` which
                matches ``boxstyle="round,pad=0.15"`` used for primary
                labels; pass ``0.10`` to mirror the inline-name bbox.
            char_aspect: Ratio of mean character width to font size for
                the figure's text family. Calibrated for matplotlib's
                default DejaVu Sans (~0.55).
            line_height: Total line-height multiplier (text + leading)
                relative to ``font_size``.

        Returns:
            Bounding-circle radius in display points.
        """
        width = max(1, len(text)) * font_size * char_aspect + 2 * bbox_pad_factor * font_size
        height = font_size * line_height + 2 * bbox_pad_factor * font_size
        return 0.5 * math.hypot(width, height)

    @staticmethod
    def _resolve_label_parallel_shifts(
        label_pos_pts: list[tuple[float, float]],
        edge_dir_pts: list[tuple[float, float]],
        edge_length_pts: list[float],
        label_radii_pts: list[float],
        *,
        pad_pts: float = 1.5,
        max_shift_frac: float = 0.33,
        max_iterations: int = 4,
    ) -> list[float]:
        """Iteratively resolve edge-label overlaps by sliding along each edge.

        Pairwise greedy resolver: for every pair of edges whose
        bounding-circle labels overlap (distance < sum of radii + pad),
        both labels slide along their respective edges in opposite
        directions so they move apart. The slide direction is derived
        from the projection of the connecting vector onto each edge's
        tangent — naturally placing each label on the side of its
        midpoint that is further from the colliding neighbor — with a
        canonical ``+/-`` tie-break when the connecting vector is
        degenerate or nearly perpendicular to both edges.

        Each label's cumulative shift is clamped to
        ``±max_shift_frac x edge_length`` so labels stay anchored to
        their edge. Cumulative updates are damped at 50% per iteration
        to suppress oscillation when several pairs interact.

        Common collision pattern: two edges sharing an endpoint whose
        other endpoints are close in display space (e.g. ``(9, 3)``
        and ``(9, 4)`` when nodes 3 and 4 are stacked). The resolver
        slides one label toward the shared endpoint and the other away
        from it, separating them along the natural edge axis without
        breaking the existing perpendicular offset to the stroke.

        Args:
            label_pos_pts: Per-edge label centers in display points
                (already including any perpendicular offset applied by
                the caller).
            edge_dir_pts: Per-edge unit tangent in display space (same
                ordering as ``label_pos_pts``).
            edge_length_pts: Per-edge length in display points; used
                to clamp each label's cumulative parallel shift.
            label_radii_pts: Per-edge label bounding-circle radius in
                display points (typically from
                :meth:`_estimate_label_radius_pts`).
            pad_pts: Extra clearance enforced between adjacent labels,
                in display points.
            max_shift_frac: Maximum cumulative parallel shift as a
                fraction of edge length.
            max_iterations: Maximum number of resolver passes; the
                loop terminates early once a pass produces no shifts.

        Returns:
            Per-edge cumulative parallel shift in display points,
            same length as the inputs.
        """
        n = len(label_pos_pts)
        shifts = [0.0] * n
        if n < 2:
            return shifts
        max_shifts = [max(0.0, max_shift_frac * elen) for elen in edge_length_pts]
        # Guard against pathological projections: if an edge is nearly perpendicular
        # to the connecting vector, sliding along it barely separates the labels.
        # Clamping `|proj|` to this floor caps the per-iteration step so we don't
        # scale the slide by 1/0 and end up shooting the label off the edge.
        min_proj = 0.20
        for _ in range(max_iterations):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    pi_x = label_pos_pts[i][0] + shifts[i] * edge_dir_pts[i][0]
                    pi_y = label_pos_pts[i][1] + shifts[i] * edge_dir_pts[i][1]
                    pj_x = label_pos_pts[j][0] + shifts[j] * edge_dir_pts[j][0]
                    pj_y = label_pos_pts[j][1] + shifts[j] * edge_dir_pts[j][1]
                    dx = pi_x - pj_x
                    dy = pi_y - pj_y
                    d = math.hypot(dx, dy)
                    req = label_radii_pts[i] + label_radii_pts[j] + pad_pts
                    if d >= req:
                        continue
                    if d > 1e-6:
                        proj_i = (dx * edge_dir_pts[i][0] + dy * edge_dir_pts[i][1]) / d
                        proj_j = (-dx * edge_dir_pts[j][0] - dy * edge_dir_pts[j][1]) / d
                    else:
                        # Degenerate: labels exactly coincident. Use canonical opposite
                        # directions (lower-index slides forward along its edge, higher-index
                        # slides backward along its own) so the choice is stable.
                        proj_i = 1.0
                        proj_j = -1.0
                    if abs(proj_i) < min_proj and abs(proj_j) < min_proj:
                        # Both edges nearly orthogonal to the connecting vector: sliding
                        # along the tangents won't separate them. Skip and let other
                        # interactions push the labels apart.
                        continue
                    proj_i = math.copysign(max(abs(proj_i), min_proj), proj_i if proj_i != 0 else 1.0)
                    proj_j = math.copysign(max(abs(proj_j), min_proj), proj_j if proj_j != 0 else -1.0)
                    gap = req - d
                    step = 0.5 * gap + 0.5
                    step_i = step / abs(proj_i)
                    step_j = step / abs(proj_j)
                    # Damp the per-iteration update so oscillations between pairs settle.
                    delta_i = 0.5 * math.copysign(step_i, proj_i)
                    delta_j = 0.5 * math.copysign(step_j, proj_j)
                    new_i = shifts[i] + delta_i
                    new_j = shifts[j] + delta_j
                    new_i = max(-max_shifts[i], min(max_shifts[i], new_i))
                    new_j = max(-max_shifts[j], min(max_shifts[j], new_j))
                    if new_i != shifts[i] or new_j != shifts[j]:
                        shifts[i] = new_i
                        shifts[j] = new_j
                        moved = True
            if not moved:
                break
        return shifts

    @staticmethod
    def _draw_offset_edge_labels(
        ax,
        pos: dict[int, tuple[float, float]],
        edge_label_map: dict[tuple[int, int], str],
        *,
        font_size: int,
        offset_pts: float | None = None,
        rotate: bool = True,
        font_color: str = "black",
        bbox: dict | None = None,
    ) -> None:
        """Draw primary ``index_TYPE`` edge labels alongside their edge lines.

        Mirrors :func:`networkx.draw_networkx_edge_labels` but shifts
        each label perpendicular to its edge direction by ``offset_pts``
        display points so the underlying stroke stays visible. Both the
        rotation angle and the perpendicular offset are computed in
        display space (after :attr:`Axes.transData`), so labels stay
        aligned with the on-screen edge orientation even when the axes
        use the default ``aspect="auto"`` scaling — where data x and
        y can have different display scales and a data-space ``atan2``
        angle would not match the visual edge direction.

        The offset always points toward "below" each edge — the
        opposite side of :meth:`_draw_inline_names`, which always flips
        to "above" — so when both annotations are drawn they never
        collide and the edge line is fully visible between them.

        Pairs of labels whose bounding circles would otherwise overlap
        in display space (e.g. two edges sharing an endpoint with
        nearby midpoints) are separated by
        :meth:`_resolve_label_parallel_shifts`, which slides each
        colliding label along its own edge direction in opposite
        directions until they pull apart, while keeping each label
        anchored to its edge (cumulative shift capped at ~⅓ of the
        edge's length).

        Args:
            ax: The matplotlib axis to draw on.
            pos: Node-to-``(x, y)`` map covering both endpoints of every
                labelled edge.
            edge_label_map: ``{(u, v): label}`` map matching the input
                accepted by :func:`networkx.draw_networkx_edge_labels`.
            font_size: Font size for the labels.
            offset_pts: Perpendicular offset, in display points
                (1 pt = 1/72 inch). ``None`` (default) uses
                :attr:`_EDGE_LABEL_OFFSET_PTS`. Set to ``0.0`` to
                recover the NetworkX-default on-edge placement.
            rotate: When ``True`` (default), each label is rotated to
                follow the edge direction in the range ``(-90°, 90°]``
                — same readable convention as the NetworkX default.
            font_color: Text color.
            bbox: Optional bbox dict; defaults to a soft white-rounded
                background that keeps labels legible against edges.
        """
        try:
            from matplotlib.transforms import offset_copy
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for rendering offset edge labels. "
                "Please install it with `pip install matplotlib`."
            ) from e

        if bbox is None:
            bbox = {"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85}
        if offset_pts is None:
            offset_pts = TopologyGraphVisualizer._EDGE_LABEL_OFFSET_PTS
        # Convert pixel-space distances to display points so the resolver works
        # in the same units as `offset_copy(units="points")`.
        px_per_pt = ax.figure.dpi / 72.0
        # Pass 1: per-edge geometry. We collect everything the resolver and the
        # eventual `ax.text` call need so we don't redo the transData round-trip.
        records: list[
            tuple[
                str,  # label
                float,  # midx (data)
                float,  # midy (data)
                float,  # angle (deg, display-aligned)
                tuple[float, float],  # perpendicular unit vector (display)
                tuple[float, float],  # tangent unit vector (display)
                float,  # edge length (pts)
                tuple[float, float],  # post-perpendicular label center (pts)
                float,  # bounding-circle radius (pts)
            ]
        ] = []
        for (u, v), label in edge_label_map.items():
            if u not in pos or v not in pos:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            midx = 0.5 * (x1 + x2)
            midy = 0.5 * (y1 + y2)
            # Compute edge direction in display space so rotation and offset
            # both align with how the edge actually renders on screen.
            disp1 = ax.transData.transform((x1, y1))
            disp2 = ax.transData.transform((x2, y2))
            ddx_px = float(disp2[0] - disp1[0])
            ddy_px = float(disp2[1] - disp1[1])
            disp_length_px = math.hypot(ddx_px, ddy_px)
            if rotate and disp_length_px > 1e-9:
                angle = math.degrees(math.atan2(ddy_px, ddx_px))
                # Constrain to (-90°, 90°] so labels stay right-side-up.
                if angle > 90.0:
                    angle -= 180.0
                elif angle <= -90.0:
                    angle += 180.0
            else:
                angle = 0.0
            if disp_length_px > 1e-9:
                tan_x = ddx_px / disp_length_px
                tan_y = ddy_px / disp_length_px
                # Display-space perpendicular pointing toward "below" the edge
                # — opposite of inline name labels (which always flip "above").
                perp_x = ddy_px / disp_length_px
                perp_y = -ddx_px / disp_length_px
                if perp_y > 0.0:
                    perp_x = -perp_x
                    perp_y = -perp_y
            else:
                tan_x, tan_y = 1.0, 0.0
                perp_x, perp_y = 0.0, -1.0
            # Convert into display-points so collision detection and slide
            # offsets share units with `offset_copy(units="points")`.
            mid_disp_px = ax.transData.transform((midx, midy))
            mid_pos_pts = (mid_disp_px[0] / px_per_pt, mid_disp_px[1] / px_per_pt)
            label_pos_pts = (mid_pos_pts[0] + offset_pts * perp_x, mid_pos_pts[1] + offset_pts * perp_y)
            edge_length_pts = disp_length_px / px_per_pt
            radius_pts = TopologyGraphVisualizer._estimate_label_radius_pts(label, font_size, bbox_pad_factor=0.15)
            records.append(
                (label, midx, midy, angle, (perp_x, perp_y), (tan_x, tan_y), edge_length_pts, label_pos_pts, radius_pts)
            )

        # Pass 2: resolve overlaps by sliding each colliding label along its own
        # edge in opposite directions, then draw with the combined offset.
        if records:
            label_centers = [rec[7] for rec in records]
            edge_dirs = [rec[5] for rec in records]
            edge_lens = [rec[6] for rec in records]
            radii = [rec[8] for rec in records]
            shifts = TopologyGraphVisualizer._resolve_label_parallel_shifts(label_centers, edge_dirs, edge_lens, radii)
        else:
            shifts = []
        for rec, parallel_pts in zip(records, shifts, strict=True):
            label, midx, midy, angle, (perp_x, perp_y), (tan_x, tan_y), _elen, _pos, _rad = rec
            # Anchor at the data-space midpoint, then offset by the requested
            # display-points distance along the on-screen perpendicular plus the
            # collision-resolved parallel shift. This keeps the visual gap
            # between edge and label constant across aspect ratios and figure
            # sizes while spreading colliding labels along their respective
            # edges.
            total_off_x = offset_pts * perp_x + parallel_pts * tan_x
            total_off_y = offset_pts * perp_y + parallel_pts * tan_y
            text_trans = offset_copy(
                ax.transData,
                fig=ax.figure,
                x=total_off_x,
                y=total_off_y,
                units="points",
            )
            ax.text(
                midx,
                midy,
                label,
                fontsize=font_size,
                color=font_color,
                ha="center",
                va="center",
                rotation=angle,
                rotation_mode="anchor",
                transform=text_trans,
                bbox=bbox,
                # Above edges (default zorder=2) but below nodes (zorder=3).
                zorder=2.5,
            )

    @staticmethod
    def _draw_inline_names(
        ax,
        pos: dict[int, tuple[float, float]],
        edges: Iterable[EdgeType],
        node_name_map: dict[int, str],
        joint_name_map: dict[int, str],
        world_node: int = DEFAULT_WORLD_NODE_INDEX,
        *,
        font_size: int = 5,
        edge_offset_pts: float | None = None,
        color: str = "0.2",
        max_chars: int = 24,
        full_paths: bool = False,
    ) -> None:
        """Annotate named nodes and edges with tiny on-graph name labels.

        Node labels are drawn above-and-right of each named node center
        with a fixed display-points offset so they look the same at any
        figure scale. Edge labels are placed at the edge midpoint with a
        small perpendicular offset (in data coordinates) so they don't
        collide with the existing index/type edge label rendered by
        :func:`networkx.draw_networkx_edge_labels`.

        Names are formatted via :meth:`_format_name` so USD-style scoped
        paths (``/.../leaf``) collapse to ``…/leaf`` when they exceed
        ``max_chars``, keeping inline labels readable for densely-named
        graphs. Pass ``full_paths=True`` to preserve full scopes.

        Args:
            ax: The matplotlib axis to draw on.
            pos: Node-to-``(x, y)`` map covering every drawn node.
            edges: Iterable of :data:`EdgeType` whose names should be drawn.
            node_name_map: Pre-built ``{body_index: name}`` map (from
                :meth:`_build_node_name_map`); empty disables node labels.
            joint_name_map: Pre-built ``{joint_index: name}`` map (from
                :meth:`_build_joint_name_map`); empty disables edge labels.
            world_node: Implicit world-node index (always skipped).
            font_size: Font size for both node and edge labels.
            edge_offset_pts: Edge-label perpendicular offset, in
                display points. ``None`` (default) uses
                :attr:`_EDGE_LABEL_OFFSET_PTS` so inline labels share
                the same visual gap as :meth:`_draw_offset_edge_labels`
                (placed on the opposite side of each edge). Computed
                in display space so the offset stays perpendicular to
                the on-screen edge orientation regardless of axis
                aspect ratio.
            color: Text color for the labels.
            max_chars: Maximum characters per displayed name; longer
                names are clipped via :meth:`_format_name`.
            full_paths: When ``True``, preserve the full scoped name
                (mid-truncating only when still over ``max_chars``)
                instead of clipping the parent path.
        """
        try:
            from matplotlib.transforms import offset_copy
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for rendering inline name labels. "
                "Please install it with `pip install matplotlib`."
            ) from e

        if edge_offset_pts is None:
            edge_offset_pts = TopologyGraphVisualizer._EDGE_LABEL_OFFSET_PTS

        # Node labels: top-right of the node center, in DPI-independent points
        # so the offset stays visually consistent across figure sizes.
        if node_name_map:
            node_trans = offset_copy(ax.transData, fig=ax.figure, x=6.0, y=6.0, units="points")
            label_bbox = {"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.6}
            for nidx, (x, y) in pos.items():
                if nidx == world_node:
                    continue
                name = node_name_map.get(int(nidx))
                if not name:
                    continue
                ax.text(
                    x,
                    y,
                    TopologyGraphVisualizer._format_name(name, max_chars, full_paths),
                    transform=node_trans,
                    fontsize=font_size,
                    color=color,
                    ha="left",
                    va="bottom",
                    clip_on=False,
                    bbox=label_bbox,
                )

        # Edge labels: perpendicular offset applied in display points along
        # the on-screen perpendicular direction, so the visual gap stays
        # constant across aspect ratios. Always flips toward positive y so
        # labels read "above" each edge — opposite side of
        # :meth:`_draw_offset_edge_labels`, which goes "below". Each label
        # is rotated to follow the on-screen edge direction (display-space
        # ``atan2`` constrained to ``(-90°, 90°]``), so inline names line
        # up with the primary ``index_TYPE`` labels they mirror across the
        # edge stroke. Pairs of inline names that would otherwise overlap
        # (e.g. two edges sharing an endpoint) are spread along their own
        # edges by :meth:`_resolve_label_parallel_shifts` — independently
        # from the primary-label resolution, since the two annotation
        # layers sit on opposite sides of each edge and never collide
        # cross-layer.
        if joint_name_map:
            edge_bbox = {"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.6}
            px_per_pt = ax.figure.dpi / 72.0
            records: list[
                tuple[
                    str,  # display name
                    float,  # midx (data)
                    float,  # midy (data)
                    float,  # rotation angle (deg, display-aligned)
                    tuple[float, float],  # perpendicular unit (display)
                    tuple[float, float],  # tangent unit (display)
                    float,  # edge length (pts)
                    tuple[float, float],  # post-perpendicular label center (pts)
                    float,  # bounding-circle radius (pts)
                ]
            ] = []
            for e in edges:
                ge = GraphEdge.from_input(e)
                name = joint_name_map.get(ge.joint_index)
                if not name:
                    continue
                u, v = ge.nodes
                if u not in pos or v not in pos:
                    continue
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                midx = 0.5 * (x1 + x2)
                midy = 0.5 * (y1 + y2)
                disp1 = ax.transData.transform((x1, y1))
                disp2 = ax.transData.transform((x2, y2))
                ddx_px = float(disp2[0] - disp1[0])
                ddy_px = float(disp2[1] - disp1[1])
                disp_length_px = math.hypot(ddx_px, ddy_px)
                if disp_length_px > 1e-9:
                    angle = math.degrees(math.atan2(ddy_px, ddx_px))
                    # Constrain to (-90°, 90°] — same readable convention as
                    # the primary labels — so inline names are always
                    # right-side-up regardless of edge direction.
                    if angle > 90.0:
                        angle -= 180.0
                    elif angle <= -90.0:
                        angle += 180.0
                    tan_x = ddx_px / disp_length_px
                    tan_y = ddy_px / disp_length_px
                    perp_x = -ddy_px / disp_length_px
                    perp_y = ddx_px / disp_length_px
                    if perp_y < 0.0:
                        perp_x = -perp_x
                        perp_y = -perp_y
                else:
                    angle = 0.0
                    tan_x, tan_y = 1.0, 0.0
                    perp_x, perp_y = 0.0, 1.0
                display_name = TopologyGraphVisualizer._format_name(name, max_chars, full_paths)
                mid_disp_px = ax.transData.transform((midx, midy))
                mid_pos_pts = (mid_disp_px[0] / px_per_pt, mid_disp_px[1] / px_per_pt)
                label_pos_pts = (
                    mid_pos_pts[0] + edge_offset_pts * perp_x,
                    mid_pos_pts[1] + edge_offset_pts * perp_y,
                )
                edge_length_pts = disp_length_px / px_per_pt
                radius_pts = TopologyGraphVisualizer._estimate_label_radius_pts(
                    display_name, font_size, bbox_pad_factor=0.10
                )
                records.append(
                    (
                        display_name,
                        midx,
                        midy,
                        angle,
                        (perp_x, perp_y),
                        (tan_x, tan_y),
                        edge_length_pts,
                        label_pos_pts,
                        radius_pts,
                    )
                )

            if records:
                label_centers = [rec[7] for rec in records]
                edge_dirs = [rec[5] for rec in records]
                edge_lens = [rec[6] for rec in records]
                radii = [rec[8] for rec in records]
                shifts = TopologyGraphVisualizer._resolve_label_parallel_shifts(
                    label_centers, edge_dirs, edge_lens, radii
                )
            else:
                shifts = []
            for rec, parallel_pts in zip(records, shifts, strict=True):
                display_name, midx, midy, angle, (perp_x, perp_y), (tan_x, tan_y), _elen, _pos, _rad = rec
                total_off_x = edge_offset_pts * perp_x + parallel_pts * tan_x
                total_off_y = edge_offset_pts * perp_y + parallel_pts * tan_y
                edge_trans = offset_copy(
                    ax.transData,
                    fig=ax.figure,
                    x=total_off_x,
                    y=total_off_y,
                    units="points",
                )
                ax.text(
                    midx,
                    midy,
                    display_name,
                    fontsize=font_size,
                    color=color,
                    ha="center",
                    # va="center" mirrors the primary labels exactly: the
                    # label center sits `edge_offset_pts` above the edge
                    # midpoint, so the inline name is the perpendicular
                    # reflection of the primary label across the edge line.
                    va="center",
                    rotation=angle,
                    rotation_mode="anchor",
                    transform=edge_trans,
                    clip_on=False,
                    bbox=edge_bbox,
                )

    @staticmethod
    def _format_name(name: str, max_chars: int, full_paths: bool) -> str:
        """Format ``name`` for display, clipping path-scoped names when too long.

        With ``full_paths=True`` (the explicit override), ``name`` is
        returned verbatim — long values may overflow tight cells visually
        but no information is lost. Otherwise the default flow applies:

        * Names whose total length is at most ``max_chars`` are returned
          unchanged.
        * Names containing ``/`` (USD-style scoped paths like
          ``"/DR_Legs/RigidBodies/ankle_bracket_b_r_o"``) drop their
          parent scope so just the leaf component survives, prefixed by
          ``…/`` — e.g. ``"…/ankle_bracket_b_r_o"``. When the leaf alone
          still exceeds ``max_chars`` it is mid-truncated as
          ``…/<leaf-prefix>…``.
        * Names without ``/`` are mid-truncated to ``<prefix>…``.

        Args:
            name: The original name to format.
            max_chars: Maximum visible characters in the returned string
                under default (``full_paths=False``) clipping. Ignored
                when ``full_paths=True``.
            full_paths: When ``True``, preserve the full name verbatim
                (no parent-side clipping or mid-truncation). When
                ``False``, clip path-scoped names from the parent side
                whenever they exceed ``max_chars``.

        Returns:
            The formatted name; bounded by ``max_chars`` only when
            ``full_paths`` is ``False``.
        """
        if full_paths:
            return name
        if len(name) <= max_chars:
            return name
        # Path-style clipping is the default for scoped names; the override
        # falls through to plain mid-truncation alongside the no-`/` branch.
        if "/" in name:
            leaf = name.rsplit("/", 1)[-1]
            if leaf:
                # `…/` is the 2-char prefix that signals a clipped scope.
                candidate = f"…/{leaf}"
                if len(candidate) <= max_chars:
                    return candidate
                # Even the leaf is too long; reserve room for the trailing
                # ellipsis (`…/<leaf-prefix>…`) and mid-truncate the leaf.
                leaf_budget = max(max_chars - 3, 1)
                return f"…/{leaf[:leaf_budget]}…"
            # `name` ends with `/`; fall through to plain mid-truncate.
        # Keep at least one source character before the ellipsis to avoid
        # producing a label that's just a single ``"…"`` glyph.
        keep = max(max_chars - 1, 1)
        return name[:keep] + "…"

    @staticmethod
    def _draw_name_tables(
        ax_joints,
        ax_bodies,
        joint_name_map: dict[int, str],
        node_name_map: dict[int, str],
        *,
        max_rows: int = 10,
        max_chars: int = 24,
        full_paths: bool = False,
        title_fontsize: int = 9,
        cell_fontsize: int = 7,
    ) -> None:
        """Render index→name reference tables for joints (left) and bodies (right).

        Each table is a multi-column wrap of ``index | name`` row pairs:
        for an input with ``N`` named entries, the table has
        ``min(N, max_rows)`` rows and ``2 * ceil(N / max_rows)``
        columns, filled column-major so reading top-to-bottom-then-left-
        to-right walks the entries in ascending index order. Names are
        formatted via :meth:`_format_name` so USD-style scoped paths
        (``/.../leaf``) collapse to ``…/leaf`` when they exceed
        ``max_chars``, with the existing tail-ellipsis fallback for
        non-path names. Pass ``full_paths=True`` to preserve full scopes.

        Sub-tables with no named entries are silently hidden — the
        caller can still pass both axes unconditionally and let this
        helper short-circuit.

        Args:
            ax_joints: Axis to host the joints table (drawn on the left).
            ax_bodies: Axis to host the bodies table (drawn on the right).
            joint_name_map: ``{joint_index: name}`` map (e.g. from
                :meth:`_build_joint_name_map`).
            node_name_map: ``{body_index: name}`` map (e.g. from
                :meth:`_build_node_name_map`).
            max_rows: Maximum rows per ``index | name`` column-pair.
            max_chars: Maximum characters retained for a name before
                clipping via :meth:`_format_name`.
            full_paths: When ``True``, preserve the full scoped name
                (mid-truncating only when still over ``max_chars``)
                instead of clipping the parent path.
            title_fontsize: Font size for the per-table title.
            cell_fontsize: Font size for each table cell.
        """
        try:
            import matplotlib.pyplot as _plt  # noqa: F401  - imported for side-effect imports
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for rendering name tables. Please install it with `pip install matplotlib`."
            ) from e

        TopologyGraphVisualizer._draw_single_name_table(
            ax_joints,
            joint_name_map,
            title="Joints",
            max_rows=max_rows,
            max_chars=max_chars,
            full_paths=full_paths,
            title_fontsize=title_fontsize,
            cell_fontsize=cell_fontsize,
        )
        TopologyGraphVisualizer._draw_single_name_table(
            ax_bodies,
            node_name_map,
            title="Bodies",
            max_rows=max_rows,
            max_chars=max_chars,
            full_paths=full_paths,
            title_fontsize=title_fontsize,
            cell_fontsize=cell_fontsize,
        )

    # Soft cap on the number of `index | name` column-pairs in a table,
    # past which we prefer growing the row count over packing more pairs.
    # Keeps the per-cell width wide enough to display ~24-char names at
    # typical figure widths (~12 inches per half-row).
    _MAX_TABLE_PAIRS: ClassVar[int] = 5

    @staticmethod
    def _draw_single_name_table(
        ax,
        name_map: dict[int, str],
        *,
        title: str,
        max_rows: int,
        max_chars: int,
        full_paths: bool,
        title_fontsize: int,
        cell_fontsize: int,
    ) -> None:
        """Render a single ``index | name`` reference table on ``ax``.

        Empty maps hide the axis entirely so a missing source (e.g.
        ``joints=None``) leaves no visual artefact; otherwise lays out
        the entries column-major so the table reads top-to-bottom-then-
        left-to-right in ascending index order. The natural pair count
        ``ceil(N / max_rows)`` is capped at
        :attr:`_MAX_TABLE_PAIRS` so very long lists grow the row count
        instead of squeezing each column past readability. Names are
        formatted via :meth:`_format_name` so scoped paths shorten to
        ``…/leaf`` when they overflow ``max_chars``.

        Args:
            ax: Axis to host the table.
            name_map: ``{index: name}`` map; empty hides the axis.
            title: Title text drawn above the table.
            max_rows: Soft maximum rows per ``index | name`` column-pair;
                effective row count grows past this when the natural
                pair count would exceed :attr:`_MAX_TABLE_PAIRS`.
            max_chars: Maximum characters retained for a name before
                clipping via :meth:`_format_name`.
            full_paths: When ``True``, preserve the full scoped name
                (mid-truncating only when still over ``max_chars``)
                instead of clipping the parent path.
            title_fontsize: Font size for the title.
            cell_fontsize: Font size for each table cell.
        """
        ax.set_axis_off()
        if not name_map:
            ax.set_visible(False)
            return

        items = sorted(name_map.items())
        n = len(items)
        natural_pairs = max(1, math.ceil(n / max_rows))
        if natural_pairs > TopologyGraphVisualizer._MAX_TABLE_PAIRS:
            n_pairs = TopologyGraphVisualizer._MAX_TABLE_PAIRS
            n_rows = math.ceil(n / n_pairs)
        else:
            n_pairs = natural_pairs
            n_rows = min(n, max_rows)
        n_cols = 2 * n_pairs

        cells: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for k, (idx, name) in enumerate(items):
            col_pair = k // n_rows
            row = k % n_rows
            cells[row][2 * col_pair] = str(idx)
            cells[row][2 * col_pair + 1] = TopologyGraphVisualizer._format_name(name, max_chars, full_paths)

        # Index columns are narrow; name columns get the lion's share. Allocate
        # `1` unit per index column and `8` units per name column, then normalize
        # so the table fills the axis width. The 1:8 ratio is generous enough
        # that ~24-character names render in full at typical figure sizes
        # (~12 inches wide) even when the table wraps into 4-6 column-pairs.
        col_weights: list[float] = []
        for c in range(n_cols):
            col_weights.append(1.0 if c % 2 == 0 else 8.0)
        total = sum(col_weights)
        col_widths = [w / total for w in col_weights]

        ax.set_title(title, fontsize=title_fontsize, loc="left", pad=2)
        table = ax.table(
            cellText=cells,
            colWidths=col_widths,
            cellLoc="left",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(cell_fontsize)
        # Soft cell borders that don't dominate the layout.
        for cell in table.get_celld().values():
            cell.set_linewidth(0.4)
            cell.set_edgecolor("0.7")
