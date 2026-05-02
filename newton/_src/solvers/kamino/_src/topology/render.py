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
from typing import ClassVar

from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    EdgeType,
    GraphEdge,
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
        joints: list[JointDescriptor] | None = None,
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
            joints: Optional joint descriptors used to enrich edge labels
                with joint names.
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

        # Set default figure size if not provided.
        if figsize is None:
            figsize = (12, 12)

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
        # entry is `(local_pos, local_radius, is_rooted, base_node)`. The trailing
        # `base_node` slot is normalized to `int | None` so the radial packer can
        # use it as a dict key alongside the int-keyed `local_pos`.
        comp_layouts: list[tuple[dict[int, tuple[float, float]], float, bool, int | None]] = []
        for comp in components:
            local_pos, local_radius, is_rooted = self._layout_component(comp, world_node)
            base_idx = int(comp.base_node) if comp.base_node is not None else None
            comp_layouts.append((local_pos, local_radius, is_rooted, base_idx))

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
            edge_label_map[uv] = self._build_edge_label(e.joint_type, e.joint_index, joints)

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

        # Begin drawing
        fig, ax = plt.subplots(figsize=figsize)

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

        # Edge labels with a small white bounding box so they remain legible
        # against the slightly grey internal-edge color.
        if edge_label_map:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_label_map,
                font_size=8,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
                ax=ax,
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
        joints: list[JointDescriptor] | None = None,
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
            joints: Optional joint descriptors for name-based edge labels.
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
        if figsize is None:
            fig_w = max(8.0, cols * per_w)
            fig_h = top_h + rows * per_h
            figsize = (fig_w, fig_h)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            nrows=rows + 1,
            ncols=cols,
            height_ratios=[top_h / per_h] + [1.0] * rows if rows > 0 else [1.0],
        )

        # Top panel: original component, classified as base / grounding / internal.
        ax_top = fig.add_subplot(gs[0, :] if rows > 0 else gs[0, 0])
        self._draw_component_on_axis(
            ax_top,
            component,
            pos,
            world_node,
            world_in_graph,
            joints,
            mode="graph",
            island_color=self._PALETTE[0],
            edge_label_font_size=edge_label_fs_top,
            node_label_font_size=node_label_fs_top,
            node_size_scale=size_scale,
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
                joints,
                mode="candidate",
                arc_joint_indices=arc_set,
                island_color=self._PALETTE[0],
                edge_label_font_size=edge_label_fs_cand,
                node_label_font_size=node_label_fs_cand,
                node_size_scale=size_scale,
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
        joints: list[JointDescriptor] | None = None,
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
            joints: Optional joint descriptors for name-based edge labels.
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
        if figsize is None:
            figsize = (8.0, top_h + bot_h)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[top_h, bot_h])

        ax_top = fig.add_subplot(gs[0, 0])
        self._draw_component_on_axis(
            ax_top,
            component,
            pos,
            world_node,
            world_in_graph,
            joints,
            mode="graph",
            island_color=self._PALETTE[0],
            edge_label_font_size=edge_label_fs_top,
            node_label_font_size=node_label_fs_top,
            node_size_scale=size_scale,
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
            joints,
            mode="candidate",
            arc_joint_indices=arc_set,
            island_color=self._PALETTE[0],
            edge_label_font_size=edge_label_fs_tree,
            node_label_font_size=node_label_fs_tree,
            node_size_scale=size_scale,
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
    ) -> tuple[dict[int, tuple[float, float]], float, bool]:
        """Compute a per-component layout in component-local coordinates.

        Returns ``(local_pos, local_radius, is_rooted)``. ``is_rooted`` is
        ``True`` when the layout grows along the local ``+x`` axis from
        the base node, enabling the radial packer to anchor the base
        toward the world. The position dict is keyed by raw int body
        index so it stays interoperable with NetworkX edge endpoints.

        Args:
            component: The component subgraph to lay out.
            world_node: The implicit world node index (skipped during layout).

        Returns:
            A ``(local_pos, local_radius, is_rooted)`` tuple.
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
            return {}, 0.0, False

        if len(comp_nodes) == 1:
            return {comp_nodes[0]: (0.0, 0.0)}, 0.0, False

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

        return local_pos, local_radius, is_rooted

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
        comp_layouts: list[tuple[dict[int, tuple[float, float]], float, bool, int | None]],
        world_node: int,
        world_in_graph: bool,
    ) -> dict[int, tuple[float, float]]:
        """Pack per-component local layouts radially around the world node.

        Args:
            comp_layouts: Per-component
                ``(local_pos, local_radius, is_rooted, base_node)`` tuples.
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

        # Sort components by descending local radius for stable packing — the largest
        # components claim their angular slots first, smaller ones fill the gaps.
        order = sorted(range(n_components), key=lambda i: -comp_layouts[i][1])
        radii = [comp_layouts[i][1] for i in order]

        # Choose an anchor ring radius `R` large enough that adjacent components do
        # not overlap. The angular footprint each component requires is approximately
        # `2 * arcsin(r / R)`. We pick the smallest R such that the sum of these
        # angular footprints fits inside `2 * pi`. Using a closed-form lower bound
        # `R >= sum(r) / pi` is conservative but always feasible; we then add a small
        # margin so labels and node radii do not collide visually.
        sum_r = sum(radii)
        # Floor on R so that even tiny graphs (e.g. a few orphans) get a sensible
        # ring; otherwise everything would collapse onto the world node.
        min_radius = max(radii) if radii else 1.0
        R = max(sum_r / math.pi, 2.5 * min_radius, 2.5)

        # Compute angular slot widths and running mid-angles
        slots: list[float] = []
        for r in radii:
            # Clamp the asin argument to [-1, 1] in case of edge cases (shouldn't
            # actually occur given the choice of R above, but defensive).
            ratio = min(max(r / R, -1.0), 1.0)
            slot = 2.0 * math.asin(ratio)
            # Minimum angular slot to keep small components readable
            slot = max(slot, 2.0 * math.pi / max(n_components * 2, 1))
            slots.append(slot)

        # Normalize slot widths so they sum to 2*pi (in case the floor above pushed
        # the total above 2*pi for many small components).
        total_slot = sum(slots)
        if total_slot > 2.0 * math.pi:
            scale = (2.0 * math.pi) / total_slot
            slots = [s * scale for s in slots]
            total_slot = 2.0 * math.pi
        # Distribute any leftover angular budget evenly as padding between slots
        padding = (2.0 * math.pi - total_slot) / max(n_components, 1)

        running = 0.0
        for sort_idx, comp_idx in enumerate(order):
            slot = slots[sort_idx]
            theta = running + slot / 2.0
            running += slot + padding

            local_pos, local_radius, is_rooted, base_node = comp_layouts[comp_idx]

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
        local_pos, _local_radius, is_rooted = TopologyGraphVisualizer._layout_component(component, world_node)

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
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_label_map,
                font_size=edge_label_font_size,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
                ax=ax,
            )
