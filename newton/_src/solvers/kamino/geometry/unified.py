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

"""
Provides a specialization of Newton's unified collision-detection pipeline for Kamino.

This module provides interfaces and data-conversion specializations for Kamino that wraps
the broad-phase and narrow-phase of Newton's CollisionPipelineUnified, writing generated
contacts data directly into Kamino's respective format.
"""

from typing import Literal

import numpy as np

# Warp imports
import warp as wp
from warp.context import Devicelike

# Newton imports
from ....geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ....geometry.broad_phase_sap import BroadPhaseSAP
from ....geometry.collision_core import compute_tight_aabb_from_support
from ....geometry.contact_data import ContactData
from ....geometry.flags import ShapeFlags
from ....geometry.narrow_phase import NarrowPhase
from ....geometry.sdf_utils import SDFData
from ....geometry.support_function import GenericShapeData, SupportMapDataProvider, pack_mesh_ptr
from ....geometry.types import GeoType

# Kamino imports
from ..core.builder import ModelBuilder
from ..core.materials import DEFAULT_FRICTION, DEFAULT_RESTITUTION, make_get_material_pair_properties
from ..core.model import Model, ModelData
from ..core.shapes import ShapeType
from ..core.types import float32, int32, quatf, transformf, uint32, uint64, vec2f, vec2i, vec3f, vec4f
from ..geometry.contacts import (
    DEFAULT_GEOM_PAIR_CONTACT_GAP,
    DEFAULT_GEOM_PAIR_MAX_CONTACTS,
    Contacts,
    make_contact_frame_znorm,
)
from ..geometry.keying import build_pair_key2

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class ContactWriterDataKamino:
    """Contact writer data for writing contacts directly in Kamino format."""

    # Contact limits
    model_max_contacts: int32
    world_max_contacts: wp.array(dtype=int32)

    # Geometry information arrays
    geom_wid: wp.array(dtype=int32)  # World ID for each geometry
    geom_bid: wp.array(dtype=int32)  # Body ID for each geometry
    geom_mid: wp.array(dtype=int32)  # Material ID for each geometry
    geom_gap: wp.array(dtype=float32)  # Detection gap for each geometry [m]

    # Material properties (indexed by material pair)
    material_restitution: wp.array(dtype=float32)
    material_static_friction: wp.array(dtype=float32)
    material_dynamic_friction: wp.array(dtype=float32)
    material_pair_restitution: wp.array(dtype=float32)
    material_pair_static_friction: wp.array(dtype=float32)
    material_pair_dynamic_friction: wp.array(dtype=float32)

    # Contact limit and active count (Newton interface)
    contact_max: int32
    contact_count: wp.array(dtype=int32)

    # Output arrays (Kamino Contacts format)
    contacts_model_num_active: wp.array(dtype=int32)
    contacts_world_num_active: wp.array(dtype=int32)
    contact_wid: wp.array(dtype=int32)
    contact_cid: wp.array(dtype=int32)
    contact_gid_AB: wp.array(dtype=vec2i)
    contact_bid_AB: wp.array(dtype=vec2i)
    contact_position_A: wp.array(dtype=vec3f)
    contact_position_B: wp.array(dtype=vec3f)
    contact_gapfunc: wp.array(dtype=vec4f)
    contact_frame: wp.array(dtype=quatf)
    contact_material: wp.array(dtype=vec2f)
    contact_key: wp.array(dtype=uint64)


###
# Functions
###


@wp.func
def convert_kamino_shape_to_newton_geo(sid: int32, params: vec4f) -> tuple[int32, vec3f]:
    """
    Converts Kamino :class:`ShapeType` and parameters to Newton :class:`GeoType` and scale.

    Shape parameter formats:
    - BOX:
        - Newton: half-extents as `scale := (x, y, z)`
        - Kamino: dimensions as `params := (depth, width, height, _)`

    - SPHERE:
        - Newton: radius as `scale := (radius, _, _)`
        - Kamino: radius as `params := (radius, _, _, _)`

    - CAPSULE:
        - Newton: radius and half-height as `scale := (radius, half_height, _)`
        - Kamino: radius and height as `params := (radius, height, _, _)`

    - CYLINDER:
        - Newton: radius and half-height as `scale := (radius, half_height, _)`
        - Kamino: radius and height as `params := (radius, height, _, _)`

    - CONE:
        - Newton: radius and half-height as `scale := (radius, half_height, _)`
        - Kamino: radius and height as `params := (radius, height, _, _)`

    - ELLIPSOID:
        - Newton: semi-axes as `scale := (x, y, z)`
        - Kamino: radii as `params := (a, b, c, _)`

    - PLANE:
        - Newton: half-width in x, half-length in y
        - Kamino: normal and distance as `params := (normal_x, normal_y, normal_z, distance)`

    See :class:`GenericShapeData` in :file:`support_function.py` for further details.

    Args:
        sid (int32):
            The Kamino ShapeType as :class:`int32`, i.e. the shape index.
        params(vec4f):
            Kamino shape parameters as :class:`vec4f`.

    Returns:
        (int32, vec3f):
        A tuple containing the corresponding Newton :class:`GeoType`
        as an :class:`int32`, and the shape scale as a :class:`vec3f`.
    """
    geo_type = int32(GeoType.NONE)
    scale = vec3f(0.0)

    if sid == ShapeType.SPHERE:
        # Kamino: (radius, 0, 0, 0) -> Newton: (radius, ?, ?)
        geo_type = GeoType.SPHERE
        scale = vec3f(params[0], 0.0, 0.0)

    elif sid == ShapeType.BOX:
        # Kamino: (depth, width, height) full size -> Newton: half-extents
        geo_type = GeoType.BOX
        scale = vec3f(params[0] * 0.5, params[1] * 0.5, params[2] * 0.5)

    elif sid == ShapeType.CAPSULE:
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = GeoType.CAPSULE
        scale = vec3f(params[0], params[1] * 0.5, 0.0)

    elif sid == ShapeType.CYLINDER:
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = GeoType.CYLINDER
        scale = vec3f(params[0], params[1] * 0.5, 0.0)

    elif sid == ShapeType.CONE:
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = GeoType.CONE
        scale = vec3f(params[0], params[1] * 0.5, 0.0)

    elif sid == ShapeType.ELLIPSOID:
        # Kamino: (a, b, c) semi-axes -> Newton: same
        geo_type = GeoType.ELLIPSOID
        scale = vec3f(params[0], params[1], params[2])

    elif sid == ShapeType.PLANE:
        # NOTE: For an infinite plane, we use (0, 0, _) to signal an infinite extents
        geo_type = GeoType.PLANE
        scale = vec3f(0.0, 0.0, 0.0)  # Infinite plane

    elif sid == ShapeType.MESH:
        geo_type = GeoType.MESH
        scale = vec3f(0.0, 0.0, 0.0)

    elif sid == ShapeType.CONVEX:
        geo_type = GeoType.CONVEX_MESH
        scale = vec3f(0.0, 0.0, 0.0)

    elif sid == ShapeType.HFIELD:
        geo_type = GeoType.HFIELD
        scale = vec3f(0.0, 0.0, 0.0)

    return geo_type, scale


@wp.func
def write_contact_unified_kamino(
    contact_data: ContactData,
    writer_data: ContactWriterDataKamino,
    output_index: int,
):
    """
    Write a contact to Kamino-compatible output arrays.

    This function is used as a custom contact writer for NarrowPhase.launch_custom_write().
    It converts ContactData from the narrow phase directly to Kamino's contact format,
    using the same distance computation as Newton core's ``write_contact``.

    Args:
        contact_data: ContactData struct from narrow phase containing contact information.
        writer_data: ContactWriterDataKamino struct containing output arrays.
        output_index: If < 0, apply gap-based filtering before writing.
            If >= 0, skip filtering (narrowphase already validated the contact).
            In both cases the model-level index is allocated from
            :attr:`ContactWriterDataKamino.contacts_model_num_active`.
    """
    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    # After narrow-phase post-processing (collision_core.py), contact_distance
    # is always the surface-to-surface signed distance regardless of kernel
    # (primitive or GJK), and contact_point_center is the midpoint between
    # the surface contact points on each shape.
    half_d = 0.5 * contact_data.contact_distance
    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * half_d
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * half_d

    # Margin-shifted signed distance (negative = penetrating beyond margin)
    d = contact_data.contact_distance - (contact_data.margin_a + contact_data.margin_b)

    # Both geoms share the same world (guaranteed by broadphase filtering)
    wid = writer_data.geom_wid[contact_data.shape_a]
    world_max_contacts = writer_data.world_max_contacts[wid]

    if output_index < 0:
        # Use per-shape detection gap (additive, matching Newton core)
        gap_a = writer_data.geom_gap[contact_data.shape_a]
        gap_b = writer_data.geom_gap[contact_data.shape_b]
        contact_gap = gap_a + gap_b

        if d > contact_gap:
            return

    # Always allocate from the model-level counter so the active count
    # stays accurate regardless of whether the narrowphase pre-allocated
    # an output_index (primitive kernel) or left it to the writer (-1).
    mcid = wp.atomic_add(writer_data.contacts_model_num_active, 0, 1)
    if mcid >= writer_data.model_max_contacts:
        wp.atomic_sub(writer_data.contacts_model_num_active, 0, 1)
        return

    # Atomically increment the world-specific contact counter and
    # roll-back the atomic add if the respective limit is exceeded
    wcid = wp.atomic_add(writer_data.contacts_world_num_active, wid, 1)
    if wcid >= world_max_contacts:
        wp.atomic_sub(writer_data.contacts_world_num_active, wid, 1)
        return

    # Retrieve the geom/body/material indices
    gid_a = contact_data.shape_a
    gid_b = contact_data.shape_b
    bid_a = writer_data.geom_bid[contact_data.shape_a]
    bid_b = writer_data.geom_bid[contact_data.shape_b]
    mid_a = writer_data.geom_mid[contact_data.shape_a]
    mid_b = writer_data.geom_mid[contact_data.shape_b]

    # Ensure the static body is always body A so that the normal
    # always points from A to B and bid_B is non-negative
    if bid_b < 0:
        gid_AB = vec2i(gid_b, gid_a)
        bid_AB = vec2i(bid_b, bid_a)
        normal = -contact_normal_a_to_b
        pos_A = b_contact_world
        pos_B = a_contact_world
    else:
        gid_AB = vec2i(gid_a, gid_b)
        bid_AB = vec2i(bid_a, bid_b)
        normal = contact_normal_a_to_b
        pos_A = a_contact_world
        pos_B = b_contact_world

    restitution_ab, _, mu_ab = wp.static(make_get_material_pair_properties())(
        mid_a,
        mid_b,
        writer_data.material_restitution,
        writer_data.material_static_friction,
        writer_data.material_dynamic_friction,
        writer_data.material_pair_restitution,
        writer_data.material_pair_static_friction,
        writer_data.material_pair_dynamic_friction,
    )
    material = vec2f(mu_ab, restitution_ab)

    gapfunc = vec4f(normal[0], normal[1], normal[2], d)
    q_frame = wp.quat_from_matrix(make_contact_frame_znorm(normal))
    key = build_pair_key2(uint32(gid_AB[0]), uint32(gid_AB[1]))

    # Store contact data in Kamino format
    writer_data.contact_wid[mcid] = wid
    writer_data.contact_cid[mcid] = wcid
    writer_data.contact_gid_AB[mcid] = gid_AB
    writer_data.contact_bid_AB[mcid] = bid_AB
    writer_data.contact_position_A[mcid] = pos_A
    writer_data.contact_position_B[mcid] = pos_B
    writer_data.contact_gapfunc[mcid] = gapfunc
    writer_data.contact_frame[mcid] = q_frame
    writer_data.contact_material[mcid] = material
    writer_data.contact_key[mcid] = key


###
# Kernels
###


@wp.func
def _compute_collision_radius(geo_type: int32, scale: vec3f) -> float32:
    """Compute the bounding-sphere radius for broadphase AABB fallback.

    Mirrors :func:`newton._src.geometry.utils.compute_shape_radius` for the
    primitive shape types that Kamino currently supports.
    """
    radius = float32(10.0)
    if geo_type == GeoType.SPHERE:
        radius = scale[0]
    elif geo_type == GeoType.BOX:
        radius = wp.length(scale)
    elif geo_type == GeoType.CAPSULE or geo_type == GeoType.CYLINDER or geo_type == GeoType.CONE:
        radius = scale[0] + scale[1]
    elif geo_type == GeoType.ELLIPSOID:
        radius = wp.max(wp.max(scale[0], scale[1]), scale[2])
    elif geo_type == GeoType.PLANE:
        if scale[0] > 0.0 and scale[1] > 0.0:
            radius = wp.length(scale)
        else:
            radius = float32(1.0e6)
    return radius


@wp.kernel
def _convert_geom_data_kamino_to_newton(
    # Inputs:
    default_gap: float32,
    geom_sid: wp.array(dtype=int32),
    geom_params: wp.array(dtype=vec4f),
    geom_margin: wp.array(dtype=float32),
    # Outputs:
    geom_gap: wp.array(dtype=float32),
    geom_type: wp.array(dtype=int32),
    geom_data: wp.array(dtype=vec4f),
    shape_collision_radius: wp.array(dtype=float32),
):
    """
    Converts Kamino geometry data to Newton-compatible format.

    Converts :class:`ShapeType` and parameters to :class:`GeoType` and scale,
    stores the per-geometry surface margin offset in ``geom_data.w``, applies
    a default floor to the per-geometry detection gap, and computes the
    bounding-sphere radius used for AABB fallback (planes, meshes, heightfields).
    """
    gid = wp.tid()

    sid = geom_sid[gid]
    params = geom_params[gid]
    margin = geom_margin[gid]
    gap = geom_gap[gid]

    geo_type, scale = convert_kamino_shape_to_newton_geo(sid, params)

    geom_type[gid] = geo_type
    geom_data[gid] = vec4f(scale[0], scale[1], scale[2], margin)
    geom_gap[gid] = wp.max(default_gap, gap)
    shape_collision_radius[gid] = _compute_collision_radius(geo_type, scale)


@wp.kernel
def _update_geom_poses_and_compute_aabbs(
    # Inputs:
    geom_type: wp.array(dtype=int32),
    geom_data: wp.array(dtype=vec4f),
    geom_bid: wp.array(dtype=int32),
    geom_ptr: wp.array(dtype=wp.uint64),
    geom_offset: wp.array(dtype=transformf),
    geom_margin: wp.array(dtype=float32),
    geom_gap: wp.array(dtype=float32),
    shape_collision_radius: wp.array(dtype=float32),
    body_pose: wp.array(dtype=transformf),
    # Outputs:
    geom_pose: wp.array(dtype=transformf),
    shape_aabb_lower: wp.array(dtype=vec3f),
    shape_aabb_upper: wp.array(dtype=vec3f),
):
    """
    Updates the pose of each Kamino geometry in world coordinates and computes its AABB.

    AABBs are enlarged by the per-shape ``margin + gap`` to ensure the broadphase
    catches all contacts within the detection threshold.
    """
    gid = wp.tid()

    geo_type = geom_type[gid]
    geo_data = geom_data[gid]
    bid = geom_bid[gid]
    margin = geom_margin[gid]
    gap = geom_gap[gid]
    X_bg = geom_offset[gid]

    X_b = wp.transform_identity(dtype=float32)
    if bid > -1:
        X_b = body_pose[bid]

    X_g = wp.transform_multiply(X_b, X_bg)

    r_g = wp.transform_get_translation(X_g)
    q_g = wp.transform_get_rotation(X_g)

    # Format is (vec3f scale, float32 margin_offset)
    scale = vec3f(geo_data[0], geo_data[1], geo_data[2])

    # Enlarge AABB by margin + gap per shape (matching Newton core convention)
    expansion = margin + gap
    margin_vec = wp.vec3(expansion, expansion, expansion)

    # Check if this is an infinite plane or mesh - use bounding sphere fallback
    is_infinite_plane = (geo_type == GeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == GeoType.MESH
    is_hfield = geo_type == GeoType.HFIELD

    # Compute the geometry AABB in world coordinates
    aabb_lower = wp.vec3(0.0)
    aabb_upper = wp.vec3(0.0)
    if is_infinite_plane or is_mesh or is_hfield:
        # Use conservative bounding sphere approach
        radius = shape_collision_radius[gid]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower = r_g - half_extents - margin_vec
        aabb_upper = r_g + half_extents + margin_vec
    else:
        # Use support function to compute tight AABB
        # Create generic shape data
        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        shape_data.scale = scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == GeoType.CONVEX_MESH:
            shape_data.auxiliary = pack_mesh_ptr(geom_ptr[gid])

        # Compute tight AABB using helper function
        data_provider = SupportMapDataProvider()
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, q_g, r_g, data_provider)
        aabb_lower = aabb_min_world - margin_vec
        aabb_upper = aabb_max_world + margin_vec

    # Store the updated geometry pose in world coordinates and computed AABB
    geom_pose[gid] = X_g
    shape_aabb_lower[gid] = aabb_lower
    shape_aabb_upper[gid] = aabb_upper


###
# Interfaces
###


class CollisionPipelineUnifiedKamino:
    """
    A specialization of the Newton's unified collision detection pipeline for Kamino.

    This pipeline uses the same broad phase algorithms (NXN, SAP, EXPLICIT) and narrow phase
    (NarrowPhase with GJK/MPR) as Newton's CollisionPipelineUnified, but writes contacts
    directly in Kamino's format using a custom contact writer.
    """

    def __init__(
        self,
        model: Model,
        builder: ModelBuilder,
        broadphase: Literal["nxn", "sap", "explicit"] = "explicit",
        max_contacts: int | None = None,
        max_contacts_per_pair: int = DEFAULT_GEOM_PAIR_MAX_CONTACTS,
        max_triangle_pairs: int = 1_000_000,
        default_gap: float = DEFAULT_GEOM_PAIR_CONTACT_GAP,
        default_friction: float = DEFAULT_FRICTION,
        default_restitution: float = DEFAULT_RESTITUTION,
        device: Devicelike = None,
    ):
        """
        Initialize an instance of Kamino's wrapper of the unified collision detection pipeline.

        Args:
            builder: Kamino ModelBuilder (used to extract collision pair information).
            broadphase: Broad-phase back-end to use (NXN, SAP, or EXPLICIT).
            max_contacts: Maximum contacts for the entire model (overrides computed value).
            max_contacts_per_pair: Maximum contacts per colliding geometry pair.
            max_triangle_pairs: Maximum triangle pairs for mesh/mesh and mesh/hfield collisions.
            default_gap: Default detection gap [m] applied as a floor to per-geometry gaps.
            default_friction: Default contact friction coefficient.
            default_restitution: Default impact restitution coefficient.
            device: Warp device used to allocate memory and operate on.
        """
        self._device: Devicelike = None
        if device is not None:
            self._device = device
        else:
            self._device = model.device

        # Cache pipeline settings
        self._broadphase: str = broadphase
        self._default_gap: float = default_gap
        self._default_friction: float = default_friction
        self._default_restitution: float = default_restitution
        self._max_contacts_per_pair: int = max_contacts_per_pair
        self._max_triangle_pairs: int = max_triangle_pairs

        # Get geometry count from builder
        self._num_geoms: int = builder.num_collision_geoms

        # Compute the maximum possible number of geom pairs (worst-case, needed for NXN/SAP)
        self._max_shape_pairs: int = (self._num_geoms * (self._num_geoms - 1)) // 2
        self._max_contacts: int = self._max_shape_pairs * self._max_contacts_per_pair

        # Override max contacts if specified explicitly
        if max_contacts is not None:
            self._max_contacts = max_contacts

        # Build shape pairs for EXPLICIT mode
        self.shape_pairs_filtered: wp.array | None = None
        if broadphase == "explicit":
            _, model_filtered_geom_pairs, _, _ = builder.make_collision_candidate_pairs()
            self.shape_pairs_filtered = wp.array(model_filtered_geom_pairs, dtype=vec2i, device=self._device)
            self._max_shape_pairs = len(model_filtered_geom_pairs)
            self._max_contacts = self._max_shape_pairs * self._max_contacts_per_pair

        # Build excluded pairs for NXN/SAP broadphase filtering.
        # Kamino uses a bitmask group/collides system that is more expressive than
        # Newton's integer collision groups. We keep all broadphase groups at 1
        # (same-group, all pairs pass group check) and instead supply an explicit
        # list of excluded pairs that encodes same-body, group/collides, and
        # neighbor-joint filtering.
        geom_collision_group_list = [1] * self._num_geoms
        self._excluded_pairs: wp.array | None = None
        self._num_excluded_pairs: int = 0
        if broadphase in ("nxn", "sap"):
            self._excluded_pairs, self._num_excluded_pairs = self._build_excluded_pairs(builder)

        # Capture a reference to per-geometry world indices already present in the model
        self.geom_wid: wp.array = model.cgeoms.wid

        # Define default shape flags for all geometries
        default_shape_flag: int = (
            ShapeFlags.VISIBLE  # Mark as visible for debugging/visualization
            | ShapeFlags.COLLIDE_SHAPES  # Enable shape-shape collision
            | ShapeFlags.COLLIDE_PARTICLES  # Enable shape-particle collision
        )

        # Allocate internal data needed by the pipeline that
        # the Kamino model and data do not yet provide
        with wp.ScopedDevice(self._device):
            self.geom_type = wp.zeros(self._num_geoms, dtype=int32)
            self.geom_data = wp.zeros(self._num_geoms, dtype=vec4f)
            self.geom_collision_group = wp.array(geom_collision_group_list, dtype=int32)
            self.shape_collision_radius = wp.zeros(self._num_geoms, dtype=float32)
            self.shape_flags = wp.full(self._num_geoms, default_shape_flag, dtype=int32)
            self.shape_aabb_lower = wp.zeros(self._num_geoms, dtype=wp.vec3)
            self.shape_aabb_upper = wp.zeros(self._num_geoms, dtype=wp.vec3)
            self.broad_phase_pairs = wp.zeros(self._max_shape_pairs, dtype=wp.vec2i)
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32)
            self.narrow_phase_contact_count = wp.zeros(1, dtype=int32)
            # TODO: These are currently left empty just to satisfy the narrow phase interface
            # but we need to implement SDF/mesh/heightfield support in Kamino to make use of them.
            # With has_meshes=False, these arrays are never accessed.
            self.shape_sdf_data = wp.empty(shape=(0,), dtype=SDFData)
            self.shape_sdf_index = wp.full_like(self.geom_type, -1)
            self.shape_collision_aabb_lower = wp.empty(shape=(0,), dtype=wp.vec3)
            self.shape_collision_aabb_upper = wp.empty(shape=(0,), dtype=wp.vec3)
            self.shape_voxel_resolution = wp.empty(shape=(0,), dtype=wp.vec3i)
            self.shape_heightfield_data = None  # TODO
            self.heightfield_elevation_data = None  # TODO

        # Initialize the broad-phase backend depending on the selected mode
        match self._broadphase:
            case "nxn":
                self.nxn_broadphase = BroadPhaseAllPairs(self.geom_wid, shape_flags=None, device=self._device)
            case "sap":
                self.sap_broadphase = BroadPhaseSAP(self.geom_wid, shape_flags=None, device=self._device)
            case "explicit":
                self.explicit_broadphase = BroadPhaseExplicit()
            case _:
                raise ValueError(f"Unsupported broad phase mode: {self._broadphase}")

        # Initialize narrow-phase backend with the contact writer customized for Kamino
        # Note: has_meshes=False since Kamino doesn't support mesh collisions yet
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=self._max_shape_pairs,
            max_triangle_pairs=self._max_triangle_pairs,
            device=self._device,
            shape_aabb_lower=self.shape_aabb_lower,
            shape_aabb_upper=self.shape_aabb_upper,
            contact_writer_warp_func=write_contact_unified_kamino,
            has_meshes=False,
        )

        # Convert geometry data from Kamino to Newton format
        self._convert_geometry_data(model)

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike:
        """Returns the Warp device the pipeline operates on."""
        return self._device

    ###
    # Operations
    ###

    def collide(self, model: Model, data: ModelData, contacts: Contacts):
        """
        Runs the unified collision detection pipeline to generate discrete contacts.

        Args:
            model (Model): The model container holding the time-invariant parameters of the simulation.
            data (ModelData): The data container holding the time-varying state of the simulation.
            contacts (Contacts): Output contacts container (will be cleared and populated)
        """
        # Check if contacts is allocated on the same device
        if contacts.device != self._device:
            raise ValueError(
                f"Contacts container device ({contacts.device}) does not match the pipeline device ({self._device})."
            )

        # Check if contacts can hold the maximum number of contacts
        if contacts.model_max_contacts_host < self._max_contacts:
            raise ValueError(
                f"Contacts container has insufficient capacity "
                f"({contacts.model_max_contacts_host} < {self._max_contacts}) "
                f"to hold all possible contacts generated by the pipeline."
            )

        # Clear contacts
        contacts.clear()

        # Clear internal contact counts
        self.narrow_phase_contact_count.zero_()

        # Update geometry poses from body states and compute respective AABBs
        self._update_geom_data(model, data)

        # Run broad-phase collision detection to get candidate shape pairs
        self._run_broadphase()

        # Run narrow-phase collision detection to generate contacts
        self._run_narrowphase(model, data, contacts)

    ###
    # Internals
    ###

    def _convert_geometry_data(self, model: Model):
        """
        Converts Kamino geometry data to the Newton format.

        This operation needs to be called only once during initialization.

        Args:
            model: The model container holding the time-invariant parameters of the simulation.
        """
        wp.launch(
            kernel=_convert_geom_data_kamino_to_newton,
            dim=self._num_geoms,
            inputs=[
                self._default_gap,
                model.cgeoms.sid,
                model.cgeoms.params,
                model.cgeoms.margin,
            ],
            outputs=[
                model.cgeoms.gap,
                self.geom_type,
                self.geom_data,
                self.shape_collision_radius,
            ],
            device=self._device,
        )

    @staticmethod
    def _build_excluded_pairs(builder: ModelBuilder) -> tuple[wp.array | None, int]:
        """Build a sorted array of shape pairs that the NXN/SAP broadphase should exclude.

        Encodes the same filtering rules as
        :meth:`ModelBuilder.make_collision_candidate_pairs` (same-body, group/collides
        bitmask, fixed-joint and DoF-joint neighbours) but returns the *complement*:
        pairs that should **not** collide.

        Returns:
            A tuple of ``(excluded_pairs_array, count)``.  When no exclusions are
            needed, returns ``(None, 0)``.
        """
        from ..core.joints import JointDoFType  # noqa: PLC0415

        geoms = builder.collision_geoms
        joints = builder.joints
        nw = builder.num_worlds
        worlds = builder._worlds

        # Pre-index joints per world for fast lookup
        joint_ranges: list[tuple[int, int]] = []
        for w in range(nw):
            lo = len(joints)
            hi = 0
            for i, j in enumerate(joints):
                if j.wid == w:
                    lo = min(lo, i)
                    hi = max(hi, i)
            joint_ranges.append((lo, hi))

        excluded: list[tuple[int, int]] = []
        ncg_offset = 0
        for wid in range(nw):
            ncg = worlds[wid].num_collision_geoms
            for idx1 in range(ncg):
                gid1 = idx1 + ncg_offset
                geom1 = geoms[gid1]
                for idx2 in range(idx1 + 1, ncg):
                    gid2 = idx2 + ncg_offset
                    geom2 = geoms[gid2]

                    # Same-body collision
                    if geom1.bid == geom2.bid:
                        excluded.append((gid1, gid2))
                        continue

                    # Group/collides bitmask check
                    if not ((geom1.group & geom2.collides) != 0 and (geom2.group & geom1.collides) != 0):
                        excluded.append((gid1, gid2))
                        continue

                    # Fixed-joint / DoF-joint neighbour check
                    jlo, jhi = joint_ranges[wid]
                    is_excluded_neighbour = False
                    for joint in joints[jlo : jhi + 1]:
                        is_pair = (joint.bid_B == geom1.bid and joint.bid_F == geom2.bid) or (
                            joint.bid_B == geom2.bid and joint.bid_F == geom1.bid
                        )
                        if is_pair:
                            if joint.dof_type == JointDoFType.FIXED:
                                is_excluded_neighbour = True
                            elif joint.bid_B >= 0:
                                is_excluded_neighbour = True
                            break
                    if is_excluded_neighbour:
                        excluded.append((gid1, gid2))

            ncg_offset += ncg

        if not excluded:
            return None, 0

        excluded.sort()
        return (
            wp.array(np.array(excluded, dtype=np.int32), dtype=wp.vec2i),
            len(excluded),
        )

    def _update_geom_data(self, model: Model, data: ModelData):
        """
        Updates geometry poses from corresponding body states and computes respective AABBs.

        Args:
            model: The model container holding the time-invariant parameters of the simulation.
            data: The data container holding the time-varying state of the simulation.
        """
        wp.launch(
            kernel=_update_geom_poses_and_compute_aabbs,
            dim=self._num_geoms,
            inputs=[
                self.geom_type,
                self.geom_data,
                model.cgeoms.bid,
                model.cgeoms.ptr,
                model.cgeoms.offset,
                model.cgeoms.margin,
                model.cgeoms.gap,
                self.shape_collision_radius,
                data.bodies.q_i,
            ],
            outputs=[
                data.cgeoms.pose,
                self.shape_aabb_lower,
                self.shape_aabb_upper,
            ],
            device=self._device,
        )

    def _run_broadphase(self):
        """
        Runs broad-phase collision detection to generate candidate geom/shape pairs.
        """
        # First clear broad phase counter
        self.broad_phase_pair_count.zero_()

        # Then launch the configured broad-phase collision detection
        match self._broadphase:
            case "nxn":
                self.nxn_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded
                    self.geom_collision_group,
                    self.geom_wid,
                    self._num_geoms,
                    self.broad_phase_pairs,
                    self.broad_phase_pair_count,
                    device=self._device,
                    filter_pairs=self._excluded_pairs,
                    num_filter_pairs=self._num_excluded_pairs,
                )
            case "sap":
                self.sap_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded
                    self.geom_collision_group,
                    self.geom_wid,
                    self._num_geoms,
                    self.broad_phase_pairs,
                    self.broad_phase_pair_count,
                    device=self._device,
                    filter_pairs=self._excluded_pairs,
                    num_filter_pairs=self._num_excluded_pairs,
                )
            case "explicit":
                self.explicit_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded
                    self.shape_pairs_filtered,
                    len(self.shape_pairs_filtered),
                    self.broad_phase_pairs,
                    self.broad_phase_pair_count,
                    device=self._device,
                )
            case _:
                raise ValueError(f"Unsupported broad phase mode: {self._broadphase}")

    def _run_narrowphase(self, model: Model, data: ModelData, contacts: Contacts):
        """
        Runs narrow-phase collision detection to generate contacts.

        Args:
            model (Model): The model container holding the time-invariant parameters of the simulation.
            data (ModelData): The data container holding the time-varying state of the simulation.
            contacts (Contacts): Output contacts container (will be populated by this function)
        """
        # Create a writer data struct to bundle all necessary input/output
        # arrays into a single object for the narrow phase custom writer
        # NOTE: Unfortunately, we need to do this on every call in python,
        # but graph-capture ensures this actually happens only once
        writer_data = ContactWriterDataKamino()
        writer_data.model_max_contacts = int32(contacts.model_max_contacts_host)
        writer_data.world_max_contacts = contacts.world_max_contacts
        writer_data.geom_bid = model.cgeoms.bid
        writer_data.geom_wid = model.cgeoms.wid
        writer_data.geom_mid = model.cgeoms.mid
        writer_data.geom_gap = model.cgeoms.gap
        writer_data.material_restitution = model.materials.restitution
        writer_data.material_static_friction = model.materials.static_friction
        writer_data.material_dynamic_friction = model.materials.dynamic_friction
        writer_data.material_pair_restitution = model.material_pairs.restitution
        writer_data.material_pair_static_friction = model.material_pairs.static_friction
        writer_data.material_pair_dynamic_friction = model.material_pairs.dynamic_friction
        writer_data.contact_max = int32(contacts.model_max_contacts_host)
        writer_data.contact_count = self.narrow_phase_contact_count
        writer_data.contacts_model_num_active = contacts.model_active_contacts
        writer_data.contacts_world_num_active = contacts.world_active_contacts
        writer_data.contact_wid = contacts.wid
        writer_data.contact_cid = contacts.cid
        writer_data.contact_gid_AB = contacts.gid_AB
        writer_data.contact_bid_AB = contacts.bid_AB
        writer_data.contact_position_A = contacts.position_A
        writer_data.contact_position_B = contacts.position_B
        writer_data.contact_gapfunc = contacts.gapfunc
        writer_data.contact_frame = contacts.frame
        writer_data.contact_material = contacts.material
        writer_data.contact_key = contacts.key

        # Run narrow phase with the custom Kamino contact writer
        self.narrow_phase.launch_custom_write(
            candidate_pair=self.broad_phase_pairs,
            candidate_pair_count=self.broad_phase_pair_count,
            shape_types=self.geom_type,
            shape_data=self.geom_data,
            shape_transform=data.cgeoms.pose,
            shape_source=model.cgeoms.ptr,
            sdf_data=self.shape_sdf_data,
            shape_sdf_index=self.shape_sdf_index,
            shape_gap=model.cgeoms.gap,
            shape_collision_radius=self.shape_collision_radius,
            shape_flags=self.shape_flags,
            shape_collision_aabb_lower=self.shape_collision_aabb_lower,
            shape_collision_aabb_upper=self.shape_collision_aabb_upper,
            shape_voxel_resolution=self.shape_voxel_resolution,
            shape_heightfield_data=self.shape_heightfield_data,
            heightfield_elevation_data=self.heightfield_elevation_data,
            writer_data=writer_data,
            device=self._device,
        )
