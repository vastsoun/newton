# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Visualization back-end for the kamino topology subsystem.

This module ships the default :class:`TopologyGraphVisualizer` — a
``matplotlib``-based concrete :class:`TopologyGraphVisualizerBase` that
renders :class:`TopologyGraph` instances together with their parsed
components and (optionally) per-component spanning-tree candidates and
the selected spanning tree.

Layout, palette, and joint-type styling live in this module; downstream
back-ends are expected to honor the same overall figure conventions
(world node distinguished, grounding edges highlighted, base edge
emphasized, chord edges rendered with a distinct line style).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from enum import IntEnum
from typing import ClassVar

from .....sim import JointType
from ..core.joints import JointDescriptor, JointDoFType
from ..core.types import override
from .types import (
    DEFAULT_WORLD_NODE_INDEX,
    EdgeType,
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
    A default implementation of the `TopologyGraphVisualizerBase`
    that renders a topology graph using networkx and matplotlib.

    Edge labels show a short abbreviation of the joint type plus the joint index.
    Because the integer in :data:`EdgeType` ``[0]`` (the joint type) is ambiguous —
    Newton's :class:`JointType` and Kamino's :class:`JointDoFType` use overlapping
    integer values (e.g. ``0`` is ``JointType.PRISMATIC`` but ``JointDoFType.FREE``) —
    the visualizer must be told which enum the integer values refer to. The default
    is :class:`JointDoFType` because this module lives in the Kamino subpackage and
    edges built from :class:`ModelBuilderKamino` use ``joint.dof_type.value``.
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

    _JOINT_TYPE_ABBR: ClassVar[dict[int, str]] = {
        JointType.PRISMATIC: "PRISM",
        JointType.REVOLUTE: "REVL",
        JointType.BALL: "BAL",
        JointType.FIXED: "FIXD",
        JointType.FREE: "FREE",
        JointType.DISTANCE: "DIST",
        JointType.D6: "D6",
        JointType.CABLE: "CABL",
    }
    """Abbreviation table for Newton's :class:`JointType` integer values."""

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
    """Abbreviation table for Kamino's :class:`JointDoFType` integer values."""

    def __init__(self, joint_type_enum: type[IntEnum] = JointDoFType):
        """Initializes the visualizer with the chosen joint-type enum.

        Args:
            joint_type_enum:
                The enum used to interpret the integer joint type stored in :data:`EdgeType` ``[0]``.
                Must be either :class:`JointType` (Newton's joint type enum) or :class:`JointDoFType`
                (Kamino's joint DoF type enum). Defaults to :class:`JointDoFType` because edges
                produced by :class:`ModelBuilderKamino` use ``joint.dof_type.value``.

        Raises:
            ValueError: If ``joint_type_enum`` is neither :class:`JointType` nor :class:`JointDoFType`.
        """
        if joint_type_enum is JointType:
            self._abbr_table = self._JOINT_TYPE_ABBR
        elif joint_type_enum is JointDoFType:
            self._abbr_table = self._JOINT_DOF_TYPE_ABBR
        else:
            raise ValueError(
                f"Unsupported `joint_type_enum={joint_type_enum!r}`: must be either "
                f"`JointType` (Newton) or `JointDoFType` (Kamino)."
            )
        self._joint_type_enum = joint_type_enum

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
        """
        Renders the given topology graph using networkx and matplotlib.

        The world node is placed at the global origin and components are packed radially
        around it. Each component receives its own per-component sublayout: a rooted
        layered layout when a base node is available (so the base anchors closest to the
        world), or a deterministic Kamada-Kawai layout otherwise. Edges are styled by
        structural role (base, grounding, internal) and labeled with a joint-type
        abbreviation and joint index.

        Args:
            nodes: A list of `NodeType` instances representing the nodes in the topology graph.
            edges: A list of `EdgeType` instances representing the edges in the topology graph.
            components: A list of `TopologyComponent` instances representing the components in the topology graph.
            world_node: The index of the world node in the topology graph.
            joints:
                Optional list of joint descriptors used to look up joint names for edge labels.
                When provided, an edge label has the form ``f"{name}_{index}_{type}"``; otherwise
                it falls back to ``f"{index}_{type}"``. ``joints`` is expected to be indexable
                by the global joint index stored in :data:`EdgeType` ``[1]``: out-of-range
                indices and missing/empty names are tolerated and silently fall back to the
                index-only label.
            figsize: Optional tuple specifying the figure size for the plot.
            path: Optional string specifying the file path to save the plot.
            show: Boolean indicating whether to display the plot.

        Raises:
            ImportError: If :mod:`networkx` or :mod:`matplotlib` are not installed.
        """
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

        if figsize is None:
            figsize = (12, 12)

        # The world node is only drawn if it appears as an endpoint of any edge,
        # which mirrors the modelling convention that an unreferenced world node
        # should not visually clutter the graph.
        world_in_graph = any(world_node in pair for _, _, pair in edges)

        # Compute per-component local layouts and their bounding-circle radii. Each
        # entry is `(local_pos, local_radius, is_rooted, base_node)`.
        comp_layouts: list[tuple[dict[NodeType, tuple[float, float]], float, bool, NodeType | None]] = []
        for comp in components:
            local_pos, local_radius, is_rooted = self._layout_component(comp, world_node)
            comp_layouts.append((local_pos, local_radius, is_rooted, comp.base_node))

        # Pack components radially around the world node. Returns a global position
        # dict keyed by node index, including the world node when applicable.
        pos = self._pack_components(comp_layouts, world_node, world_in_graph)

        # Build a single nx.Graph for drawing. We use a plain Graph (not MultiGraph)
        # because parallel edges between the same body pair are rare in topology
        # graphs and would only slightly clutter the labels — by contrast, sticking
        # to Graph keeps the per-edge styling logic simple.
        G = nx.Graph()
        G.add_nodes_from(nodes)
        if world_in_graph:
            G.add_node(world_node)
        for _t, _j, (u, v) in edges:
            G.add_edge(u, v)

        # Classify each input edge into one of three role buckets via a per-component
        # scan. We compare on `(joint_type, joint_index)` rather than the full tuple
        # so the comparison is cheap and unambiguous regardless of edge ordering.
        base_keys: set[tuple[int, int]] = set()
        ground_keys: set[tuple[int, int]] = set()
        for comp in components:
            if comp.base_edge is not None:
                base_keys.add((comp.base_edge[0], comp.base_edge[1]))
            if comp.ground_edges is not None:
                for e in comp.ground_edges:
                    ground_keys.add((e[0], e[1]))

        base_edges_uv: list[tuple[NodeType, NodeType]] = []
        ground_edges_uv: list[tuple[NodeType, NodeType]] = []
        internal_edges_uv: list[tuple[NodeType, NodeType]] = []
        edge_label_map: dict[tuple[NodeType, NodeType], str] = {}
        for jt, jid, (u, v) in edges:
            uv = (u, v)
            key = (jt, jid)
            if key in base_keys:
                base_edges_uv.append(uv)
            elif key in ground_keys:
                ground_edges_uv.append(uv)
            else:
                internal_edges_uv.append(uv)
            edge_label_map[uv] = self._build_edge_label(jt, jid, joints)

        # Build per-node styling. Defaults are overwritten by component- and role-
        # specific styling further below; this ordering matters because the base
        # node should win over the generic island styling.
        node_color_map: dict[NodeType, str] = {}
        node_size_map: dict[NodeType, int] = {}
        node_edge_color_map: dict[NodeType, str] = {}
        node_linewidth_map: dict[NodeType, float] = {}

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
                    node_color_map[n] = color
                    node_size_map[n] = 700
                island_index += 1
            else:
                # Single-node component: connected vs isolated orphan
                n = comp.nodes[0]
                if comp.is_connected:
                    node_color_map[n] = "grey"
                else:
                    node_color_map[n] = "white"
                node_size_map[n] = 700

        # Base nodes get a thicker border to mark them as the local root, while
        # keeping their component fill so they remain visually grouped.
        for comp in components:
            if comp.base_node is not None:
                node_linewidth_map[comp.base_node] = 2.5

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

    @staticmethod
    def _layout_component(
        component: TopologyComponent,
        world_node: int,
    ) -> tuple[dict[NodeType, tuple[float, float]], float, bool]:
        """Computes a per-component layout in component-local coordinates.

        Args:
            component: The component subgraph to lay out.
            world_node: The index of the implicit world node, used to skip world endpoints.

        Returns:
            A tuple ``(local_pos, local_radius, is_rooted)`` where:

            - ``local_pos`` maps each node in the component to a local ``(x, y)`` position.
            - ``local_radius`` is the bounding-circle radius of the layout in local
              coordinates (with a small floor so single-node components still take a slot).
            - ``is_rooted`` is True when the layout grows along the local ``+x`` axis from
              the base node (``base_node`` first, at the local origin), enabling the radial
              packer to anchor the base toward the world. False when the layout has no
              preferred orientation.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for rendering the topology graph. Please install it with `pip install networkx`."
            ) from e

        comp_nodes = list(component.nodes) if component.nodes else []
        if not comp_nodes:
            return {}, 0.0, False

        # Orphan: a single-node component lays out trivially at the local origin.
        if len(comp_nodes) == 1:
            return {comp_nodes[0]: (0.0, 0.0)}, 0.0, False

        # Build the undirected internal subgraph (skip world endpoints).
        internal_pairs: list[tuple[NodeType, NodeType]] = []
        if component.edges:
            for _t, _j, (u, v) in component.edges:
                if u == world_node or v == world_node:
                    continue
                internal_pairs.append((u, v))

        if component.base_node is not None:
            local_pos = TopologyGraphVisualizer._rooted_layered_layout(
                comp_nodes, internal_pairs, root=component.base_node
            )
            is_rooted = True
        else:
            sub = nx.Graph()
            sub.add_nodes_from(comp_nodes)
            sub.add_edges_from(internal_pairs)
            try:
                local_pos = nx.kamada_kawai_layout(sub)
            except Exception:
                # Kamada-Kawai requires a connected graph and at least 2 nodes — fall
                # back to a deterministic spring layout if the heuristic fails.
                local_pos = nx.spring_layout(sub, seed=42)
            is_rooted = False

        # Bounding-circle radius (with a small floor so degenerate layouts still
        # take an angular slot during radial packing).
        max_r = 0.0
        for x, y in local_pos.values():
            max_r = max(max_r, (x * x + y * y) ** 0.5)
        local_radius = max(max_r, 0.5)

        return local_pos, local_radius, is_rooted

    @staticmethod
    def _rooted_layered_layout(
        nodes: list[NodeType],
        pairs: list[tuple[NodeType, NodeType]],
        root: NodeType,
    ) -> dict[NodeType, tuple[float, float]]:
        """Layered BFS layout rooted at ``root``, growing along the local ``+x`` axis.

        The root is placed at the local origin, children at depth ``d`` are placed at
        ``x = d`` and laterally distributed symmetrically around ``y = 0``. Nodes
        unreachable from the root (which should not occur for a well-formed component)
        are appended to the deepest layer to ensure every node receives a position.
        """
        adj: dict[NodeType, list[NodeType]] = {n: [] for n in nodes}
        for u, v in pairs:
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
        for n, neighbors in adj.items():
            adj[n] = sorted(set(neighbors))

        # BFS from the root to assign integer depths
        depth: dict[NodeType, int] = {root: 0}
        order: list[NodeType] = [root]
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
        layers: dict[int, list[NodeType]] = defaultdict(list)
        for n in order:
            layers[depth[n]].append(n)

        # Lateral spacing chosen so total layer width matches the layer count, which
        # keeps the layout's aspect ratio roughly square as depth grows.
        local_pos: dict[NodeType, tuple[float, float]] = {}
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
        comp_layouts: list[tuple[dict[NodeType, tuple[float, float]], float, bool, NodeType | None]],
        world_node: int,
        world_in_graph: bool,
    ) -> dict[NodeType, tuple[float, float]]:
        """Packs per-component local layouts radially around the world node.

        Args:
            comp_layouts: For each component, ``(local_pos, local_radius, is_rooted, base_node)``.
            world_node: The index of the implicit world node.
            world_in_graph: Whether the world node should be placed at the origin.

        Returns:
            A global position dict keyed by node index. The world node, when applicable, is at
            ``(0, 0)``.
        """
        pos: dict[NodeType, tuple[float, float]] = {}
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

    def _build_edge_label(
        self,
        joint_type: int,
        joint_index: int,
        joints: list[JointDescriptor] | None,
    ) -> str:
        """Builds an edge label of the form ``f"{name}_{index}_{type}"`` (or shorter variants).

        The joint name prefix is included only when ``joints`` is provided,
        ``joints[joint_index]`` exists, and the descriptor's ``name`` is a non-empty string.

        The joint-type suffix is included only when ``joint_type`` is a non-negative integer
        recognised by the active abbreviation table (selected at construction time via
        ``joint_type_enum``). The sentinel value ``-1`` is treated as "unspecified" and the
        suffix is omitted from the label. All other unrecognised joint-type integer values
        raise :class:`ValueError` so that incorrectly-categorised edges are surfaced loudly rather
        than silently rendered with an incorrect or generic label.

        Returns one of (in priority order, given the available pieces):

        - ``"{name}_{index}_{type}"``
        - ``"{index}_{type}"``
        - ``"{name}_{index}"``
        - ``"{index}"``

        Raises:
            ValueError: If ``joint_type`` is neither ``-1`` nor a value of the active
                joint-type enum (``JointType`` or ``JointDoFType``).
        """
        if joint_type == -1:
            abbr: str | None = None
        elif joint_type in self._abbr_table:
            abbr = self._abbr_table[joint_type]
        else:
            raise ValueError(
                f"Unsupported joint type `{joint_type}` for joint at index `{joint_index}`: "
                f"value is not a member of the active joint-type enum "
                f"`{self._joint_type_enum.__name__}` and is not the `-1` sentinel for "
                f"unspecified joint types."
            )

        name: str | None = None
        if joints is not None and 0 <= joint_index < len(joints):
            descriptor = joints[joint_index]
            candidate = getattr(descriptor, "name", None)
            if isinstance(candidate, str) and candidate:
                name = candidate

        prefix = f"{name}_" if name is not None else ""
        suffix = f"_{abbr}" if abbr is not None else ""
        return f"{prefix}{joint_index}{suffix}"

    @staticmethod
    def _pick_metadata_corner(
        pos: dict[NodeType, tuple[float, float]],
    ) -> tuple[float, float, str, str]:
        """Picks the panel corner with the most empty space for a metadata overlay.

        Returns the axes-fraction position and matplotlib alignment strings for
        the corner whose nearest node — in node-position-normalized
        coordinates — is the furthest. This minimises the chance that the
        candidate-panel metadata table overlaps with rendered nodes or edge
        labels. Top-right is preferred on ties (most natural reading position).

        Args:
            pos: The shared node-position map for the component, identical
                across all candidate panels.

        Returns:
            A tuple ``(x_frac, y_frac, ha, va)`` ready to be passed to
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

    @staticmethod
    def _compute_single_component_pos(
        component: TopologyComponent,
        world_node: int,
    ) -> tuple[dict[NodeType, tuple[float, float]], bool]:
        """Computes a node-position dict for a single-component figure.

        Reuses :meth:`_layout_component` to obtain the body-node positions in local
        component coordinates and then anchors the world node next to the layout
        when the component has any world-incident edge. The returned ``pos`` dict
        is shared by every panel of a candidate figure so candidate trees can be
        compared at a glance.

        Args:
            component: The component subgraph whose nodes need positions.
            world_node: The index of the implicit world node.

        Returns:
            A tuple ``(pos, world_in_graph)`` where:

            - ``pos`` maps each node (including the world node when applicable) to a
              ``(x, y)`` position.
            - ``world_in_graph`` is ``True`` when the component has at least one edge
              referencing the world node and the world should therefore be drawn.
        """
        local_pos, _local_radius, is_rooted = TopologyGraphVisualizer._layout_component(component, world_node)

        # The world is only drawn when the component actually references it via an
        # edge endpoint. This mirrors `render_graph`'s `world_in_graph` heuristic.
        world_in_graph = any(world_node in pair for _t, _j, pair in component.edges or [])

        pos: dict[NodeType, tuple[float, float]] = dict(local_pos)
        if not world_in_graph:
            return pos, False

        if is_rooted and component.base_node is not None and component.base_node in local_pos:
            # Rooted layout grows along local +x from the base node, so place the
            # world one unit to the left to mirror the radial-packer convention
            # ("base anchored toward the world") used by `_pack_components`.
            bx, by = local_pos[component.base_node]
            pos[world_node] = (bx - 1.0, by)
        else:
            # Unrooted: drop the world below the layout's bounding box. The 1.0
            # offset is in the same scale as `_rooted_layered_layout` (unit step
            # per BFS layer), so the world stays visually adjacent without
            # overlapping any body node.
            xs = [p[0] for p in local_pos.values()] if local_pos else [0.0]
            ys = [p[1] for p in local_pos.values()] if local_pos else [0.0]
            cx = (min(xs) + max(xs)) / 2.0
            min_y = min(ys)
            pos[world_node] = (cx, min_y - 1.0)

        return pos, True

    def _draw_component_on_axis(
        self,
        ax,
        component: TopologyComponent,
        pos: dict[NodeType, tuple[float, float]],
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
        """Draws a single component on a matplotlib axis using a shared ``pos`` dict.

        This helper is the single source of truth for the per-component drawing
        vocabulary used by the candidate-tree renderer. It mirrors the styling
        choices made by :meth:`render_graph` so that a candidate figure remains
        visually consistent with a graph figure of the same component.

        Args:
            ax: The matplotlib axis to draw on.
            component: The component to render.
            pos: A node-to-``(x, y)`` map covering every node referenced by ``component``
                (and the world node when ``world_in_graph`` is ``True``).
            world_node: Index of the implicit world node.
            world_in_graph: Whether the world node should be drawn on this axis.
            joints: Optional list of joint descriptors used to look up joint names
                for edge labels (forwarded verbatim to :meth:`_build_edge_label`).
            mode: Either ``"graph"`` (top-panel classification: base / grounding /
                internal) or ``"candidate"`` (sub-panel classification: base / arc /
                chord). In ``"candidate"`` mode, ``arc_joint_indices`` selects the
                set of joints that belong to the spanning tree.
            arc_joint_indices: Required in ``"candidate"`` mode; the set of global
                joint indices that are arcs of the candidate spanning tree (i.e.
                ``set(candidate.arcs) - {-1}``). Ignored in ``"graph"`` mode.
            island_color: Fill color used for island bodies. Orphans always use the
                ``grey`` (connected) / ``white`` (isolated) pair from ``render_graph``.
            show_edge_labels: Whether to draw per-edge text labels.
            edge_label_font_size: Font size for edge labels.
            node_label_font_size: Font size for node labels.
            node_size_scale: Multiplicative factor applied to all node sizes (default,
                world, island, orphan). Used by the candidate-grid renderer to keep
                node markers readable as the per-cell physical size grows with the
                component's complexity.

        Raises:
            ImportError: If :mod:`networkx` or :mod:`matplotlib` are not installed.
            ValueError: If ``mode`` is not one of ``"graph"`` or ``"candidate"``.
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

        # Build a plain `nx.Graph` from the component's nodes/edges. We deliberately
        # mirror `render_graph`'s choice of `Graph` (not `MultiGraph`) so the per-
        # edge styling logic stays simple — parallel edges between the same body
        # pair are rare in topology graphs.
        G = nx.Graph()
        if component.nodes:
            G.add_nodes_from(component.nodes)
        if world_in_graph:
            G.add_node(world_node)
        for _t, _j, (u, v) in component.edges or []:
            G.add_edge(u, v)

        base_key: tuple[int, int] | None = None
        if component.base_edge is not None:
            base_key = (component.base_edge[0], component.base_edge[1])

        ground_keys: set[tuple[int, int]] = set()
        if mode == "graph" and component.ground_edges is not None:
            for e in component.ground_edges:
                ground_keys.add((e[0], e[1]))

        # Classify each edge of the component into one of three buckets and
        # build a single edge-label map keyed by `(u, v)`.
        base_edges_uv: list[tuple[NodeType, NodeType]] = []
        secondary_edges_uv: list[tuple[NodeType, NodeType]] = []  # grounding (graph) or arc (candidate)
        tertiary_edges_uv: list[tuple[NodeType, NodeType]] = []  # internal (graph) or chord (candidate)
        edge_label_map: dict[tuple[NodeType, NodeType], str] = {}
        for jt, jid, (u, v) in component.edges or []:
            uv = (u, v)
            key = (jt, jid)
            if base_key is not None and key == base_key:
                base_edges_uv.append(uv)
            elif mode == "graph":
                if key in ground_keys:
                    secondary_edges_uv.append(uv)
                else:
                    tertiary_edges_uv.append(uv)
            else:
                # Candidate mode: arc vs chord. The base edge has already been
                # peeled off above, so a real `arc_joint_indices` membership check
                # is unambiguous for the remaining edges.
                if jid in arc_joint_indices:  # type: ignore[operator]
                    secondary_edges_uv.append(uv)
                else:
                    tertiary_edges_uv.append(uv)
            edge_label_map[uv] = self._build_edge_label(jt, jid, joints)

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

        node_color_map: dict[NodeType, str] = {}
        node_size_map: dict[NodeType, int] = {}
        node_edge_color_map: dict[NodeType, str] = {}
        node_linewidth_map: dict[NodeType, float] = {}
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
                node_color_map[n] = island_color
                node_size_map[n] = size_member
        elif component.nodes:
            n = component.nodes[0]
            if component.is_connected:
                node_color_map[n] = "grey"
            else:
                node_color_map[n] = "white"
            node_size_map[n] = size_member

        if component.base_node is not None:
            node_linewidth_map[component.base_node] = 2.5

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
        """
        Renders the candidate spanning trees of a component of the topology graph.

        The figure is composed of a top "Original Component" panel (rendered with
        the same ``base / grounding / internal`` edge classification used by
        :meth:`render_graph`) and a grid of candidate panels below it. All panels
        share the same node-position map produced by
        :meth:`_compute_single_component_pos` so candidate trees can be compared
        directly. In each candidate panel, edges are classified as ``base``
        (always solid black), ``arc`` (solid grey, in the spanning tree) or
        ``chord`` (dashed red, loop closure) per the candidate's ``arcs`` list.

        Per-cell physical size is held at a fixed default so the resulting
        figure (and any saved PDF / PNG) stays a reasonable page size. When the
        component has more than ten bodies, node markers, edge labels and node
        labels are shrunk so the layout still fits each cell; font sizes are
        floored at sensible minimums so labels remain legible.

        Each candidate panel additionally renders a small monospace metadata
        table at the top-right corner, listing the candidate's ``depth``,
        ``num_bodies``, ``num_tree_arcs`` and ``num_tree_chords``.

        Args:
            component: The :class:`TopologyComponent` instance whose spanning tree candidates are to be rendered.
            candidates: A list of :class:`TopologySpanningTree` instances representing the candidate spanning trees.
                When empty, only the top "Original Component" panel is drawn.
            world_node: The index of the world node in the topology graph.
            joints:
                Optional list of joint descriptors used to look up joint names for edge labels.
                When provided, an edge label has the form ``f"{name}_{index}_{type}"``; otherwise
                it falls back to ``f"{index}_{type}"``. Forwarded verbatim to :meth:`_build_edge_label`.
            skip_orphans:
                When ``True`` (default), this method returns immediately for orphan components
                (single-body subgraphs whose spanning tree is trivial), since their candidate
                figures carry no useful information beyond the original-graph render. Set to
                ``False`` to also render the trivial candidate of every orphan component.
            figsize: Optional tuple specifying the figure size for the plot. When ``None``, the
                figure size is derived from the candidate count alone (per-cell size is
                independent of component complexity to keep PDF page sizes manageable).
            path: Optional string specifying the file path to save the plot.
            show: Boolean indicating whether to display the plot.

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
        """
        Renders the selected spanning tree for a given component
        of the topology graph using networkx and matplotlib.

        Args:
            component: The :class:`TopologyComponent` instance whose selected spanning tree is to be rendered.
            tree: The :class:`TopologySpanningTree` instance representing the selected spanning tree.
            world_node: The index of the world node in the topology graph.
            joints:
                Optional list of joint descriptors used to look up joint names for edge labels.
                When provided, an edge label has the form ``f"{name}_{index}_{type}"``; otherwise
                it falls back to ``f"{index}_{type}"``.
            skip_orphans:
                When ``True`` (default), this method returns immediately for orphan components
                (single-body subgraphs whose spanning tree is trivial), since their figures carry
                no useful information beyond the original-graph render. Set to ``False`` to also
                render the trivial spanning tree of every orphan component.
            figsize: Optional tuple specifying the figure size for the plot.
            path: Optional string specifying the file path to save the plot.
            show: Boolean indicating whether to display the plot.

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
