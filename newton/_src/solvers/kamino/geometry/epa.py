###########################################################################
# KAMINO: Collision Detection: EPA Operations
###########################################################################

from __future__ import annotations

import warp as wp

from newton._src.solvers.kamino.core.types import mat43f
from newton._src.solvers.kamino.geometry.types import (
    FLOAT_MAX, EPS_BEST_COUNT, TRIS_DIM,
    matc3, vecc3, mat2c3,
    VECI1, VECI2,
)
from newton._src.solvers.kamino.geometry.math import gjk_normalize
from newton._src.solvers.kamino.geometry.gjk import gjk_support


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###

@wp.func
def expand_polytope(count: int, prev_count: int, dists: vecc3, tris: mat2c3, p: matc3):
    # expand polytope greedily
    for j in range(count):
        best = int(0)
        dd = dists[0]
        for i in range(1, 3 * prev_count):
            if dists[i] < dd:
                dd = dists[i]
                best = i

        dists[best] = float(wp.static(2 * FLOAT_MAX))

        parent_index = best // 3
        child_index = best % 3

        # fill in the new triangle at the next index
        tris[TRIS_DIM + j * 3 + 0] = tris[parent_index * 3 + child_index]
        tris[TRIS_DIM + j * 3 + 1] = tris[parent_index * 3 + ((child_index + 1) % 3)]
        tris[TRIS_DIM + j * 3 + 2] = p[parent_index]

    for r in range(wp.static(EPS_BEST_COUNT * 3)):
        # swap triangles
        swap = tris[TRIS_DIM + r]
        tris[TRIS_DIM + r] = tris[r]
        tris[r] = swap

    return dists, tris


def get_epa(
  geomtype1: int,
  geomtype2: int,
  epa_iterations: int,
  epa_exact_neg_distance: bool,
  depth_extension: float,
):
    # compute contact normal and depth
    @wp.func
    def _epa(
        # Model:
        mesh_vert: wp.array(dtype=wp.vec3),
        # In:
        geom1: Geom,
        geom2: Geom,
        simplex: mat43f,
        normal: wp.vec3,
    ):
        # get the support, if depth < 0: objects do not intersect
        depth, _ = gjk_support(geom1, geom2, geomtype1, geomtype2, normal, mesh_vert)

        if depth < -depth_extension:
            # Objects are not intersecting, and we do not obtain the closest points as
            # specified by depth_extension.
            return wp.nan, wp.vec3(wp.nan, wp.nan, wp.nan)

        if wp.static(epa_exact_neg_distance):
            # Check closest points to all edges of the simplex, rather than just the
            # face normals. This gives the exact depth/normal for the non-intersecting
            # case.
            for i in range(6):
                i1 = VECI1[i]
                i2 = VECI2[i]

                si1 = simplex[i1]
                si2 = simplex[i2]

                if si1[0] != si2[0] or si1[1] != si2[1] or si1[2] != si2[2]:
                    v = si1 - si2
                    alpha = wp.dot(si1, v) / wp.dot(v, v)

                    # p0 is the closest segment point to the origin
                    p0 = wp.clamp(alpha, 0.0, 1.0) * v - si1
                    p0, pf = gjk_normalize(p0)

                    if pf:
                        depth2, _ = gjk_support(geom1, geom2, geomtype1, geomtype2, p0, mesh_vert)
                        if depth2 < depth:
                            depth = depth2
                            normal = p0

        # supporting points for each triangle
        p = matc3()

        # distance to the origin for candidate triangles
        dists = vecc3()

        tris = mat2c3()
        tris[0] = simplex[2]
        tris[1] = simplex[1]
        tris[2] = simplex[3]

        tris[3] = simplex[0]
        tris[4] = simplex[2]
        tris[5] = simplex[3]

        tris[6] = simplex[1]
        tris[7] = simplex[0]
        tris[8] = simplex[3]

        tris[9] = simplex[0]
        tris[10] = simplex[1]
        tris[11] = simplex[2]

        # Calculate the total number of iterations to avoid nested loop
        # This is a hack to reduce compile time
        count = int(4)
        it = int(0)
        for _ in range(wp.static(epa_iterations)):
            it += count
            count = wp.min(count * 3, EPS_BEST_COUNT)

        count = int(4)
        i = int(0)
        for _ in range(it):
            # Loop through all triangles, and obtain distances to the origin for each
            # new triangle candidate.
            ti = 3 * i
            n = wp.cross(tris[ti + 2] - tris[ti + 0], tris[ti + 1] - tris[ti + 0])
            n, nf = gjk_normalize(n)
            if not nf:
                for j in range(3):
                    dists[i * 3 + j] = wp.static(float(2 * FLOAT_MAX))
                continue

            dist, pi = gjk_support(geom1, geom2, geomtype1, geomtype2, n, mesh_vert)
            p[i] = pi

            if dist < depth:
                depth = dist
                normal = n

            # iterate over edges and get distance using support point
            for j in range(3):
                if wp.static(epa_exact_neg_distance):
                    # obtain closest point between new triangle edge and origin
                    tqj = tris[ti + j]

                    if (p[i, 0] != tqj[0]) or (p[i, 1] != tqj[1]) or (p[i, 2] != tqj[2]):
                        v = p[i] - tris[ti + j]
                        alpha = wp.dot(p[i], v) / wp.dot(v, v)
                        p0 = wp.clamp(alpha, 0.0, 1.0) * v - p[i]
                        p0, pf = gjk_normalize(p0)

                        if pf:
                            dist2, v = gjk_support(geom1, geom2, geomtype1, geomtype2, p0, mesh_vert)

                        if dist2 < depth:
                            depth = dist2
                            normal = p0

                plane = wp.cross(p[i] - tris[ti + j], tris[ti + ((j + 1) % 3)] - tris[ti + j])
                plane, pf = gjk_normalize(plane)

                if pf:
                    dd = wp.dot(plane, tris[ti + j])
                else:
                    dd = float(FLOAT_MAX)

                if (dd < 0 and depth >= 0) or (
                    tris[ti + ((j + 2) % 3)][0] == p[i][0]
                    and tris[ti + ((j + 2) % 3)][1] == p[i][1]
                    and tris[ti + ((j + 2) % 3)][2] == p[i][2]
                ):
                    dists[i * 3 + j] = float(FLOAT_MAX)
                else:
                    dists[i * 3 + j] = dd

            if i == count - 1:
                prev_count = count
                count = wp.min(count * 3, EPS_BEST_COUNT)
                dists, tris = expand_polytope(count, prev_count, dists, tris, p)
                i = int(0)
            else:
                i += 1

        return depth, normal

    return _epa
