import numpy as np
from numpy.linalg import norm, det
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from itertools import combinations


def compute_minor(arr, i, j):
    """
    Compute minor of a matrix
    """
    rows = set(range(arr.shape[0]))
    rows.remove(i)
    cols = set(range(arr.shape[1]))
    cols.remove(j)
    sub = arr[np.array(list(rows))[:, np.newaxis], np.array(list(cols))]
    return det(sub)


def circum(points):
    """
    Compute the radius of the circum circle or sphere to the 3 or 4 given points
    """
    if points.shape[1] == 2:
        n = 4
    else:
        n = 5
    M = np.ones((n, n))
    M[1:, :-1] = [[norm(p) ** 2, *p] for p in points]
    M11 = compute_minor(M, 0, 0)
    if M11 == 0:
        return np.inf, np.inf

    M12 = compute_minor(M, 0, 1)
    M13 = compute_minor(M, 0, 2)
    M14 = compute_minor(M, 0, 3)
    x0 = 0.5 * M12 / M11
    y0 = - 0.5 * M13 / M11
    if n == 4:
        center = np.hstack((x0, y0))
    else:
        z0 = 0.5 * M14 / M11
        center = np.hstack((x0, y0, z0))
    r = norm(points - center, axis=1)
    return r.mean(), center


def get_alpha_complex(simplices, points, alpha=.1, radii=None):
    if radii is None:
        radii = list(map(lambda s: circum(points[s])[0], simplices))
    return radii, [ix for ix, r in enumerate(radii) if r < alpha]


def vertex_to_simplices(vertices, dt):
    simplices = {v: [] for v in vertices}
    for v in vertices:
        spx = dt.vertex_to_simplex[v]
        simplices[v].append(spx)
        to_explore = [x for x in dt.neighbors[spx] if x != -1]
        ix = 0
        while ix < len(to_explore):
            n = to_explore[ix]
            ix += 1
            if v in dt.simplices[n]:
                simplices[v].append(n)
                to_explore.extend([x for x in dt.neighbors[n] if x != -1 and x not in to_explore])
    return simplices


def circles_from_p1p2r(p1, p2, r):
    """
    Code from here: https://rosettacode.org/wiki/Circles_of_given_radius_through_two_points#Python
    Following explanation at http://mathforum.org/library/drmath/view/53027.html
    """
    if r == 0.0:
        raise ValueError('radius of zero')
    (x1, y1), (x2, y2) = p1, p2
    if all(p1 == p2):
        raise ValueError('coincident points gives infinite number of Circles')
    # delta x, delta y between points
    dx, dy = x2 - x1, y2 - y1
    # halfway point
    x3, y3 = (x1 + x2) / 2, (y1 + y2) / 2
    # dist between points
    q = np.sqrt(dx ** 2 + dy ** 2)
    if q > 2.0 * r:
        # raise ValueError('separation of points > diameter')
        return (x3, y3), (x3, y3)
    # distance along the mirror line
    d = np.sqrt(r ** 2 - (q / 2) ** 2)
    # One answer
    c1 = (x3 - d * dy / q, y3 + d * dx / q)
    # The other answer
    c2 = (x3 + d * dy / q, y3 - d * dx / q)
    return c1, c2


def alpha_exposed_segments(simplex, dt, alpha):
    indices, indptr = dt.vertex_neighbor_vertices
    neigh = set(np.hstack([indptr[indices[p]:indices[p + 1]] for p in simplex]))
    segments = []
    for pair in combinations(simplex, 2):
        c_neigh = neigh - set(pair)
        neigh_coords = dt.points[list(c_neigh), :]
        centers = circles_from_p1p2r(dt.points[pair[0]], dt.points[pair[1]], alpha)
        dists = [cdist(np.atleast_2d(c), neigh_coords) for c in centers]
        exposed = [np.all(d > alpha) for d in dists]
        if exposed[0] ^ exposed[1]:
            segments.append(pair)
    return segments


def get_alpha_shape(spx_ix, dt, alpha):
    vert_in_ch = set(dt.convex_hull.reshape(-1))
    spx_in_cpx = set(spx_ix)
    vert_in_cpx = set(dt.simplices[spx_ix].reshape(-1))
    v_to_s = vertex_to_simplices(vert_in_cpx, dt)
    vert_in_shape = set()
    for v in vert_in_cpx:
        if v in vert_in_ch:
            vert_in_shape.add(v)
            continue
        if all(s in spx_in_cpx for s in v_to_s[v]):
            continue
        vert_in_shape.add(v)
    # spx_in_shape = [list(filter(lambda v: v in vert_in_shape, dt.simplices[s])) for s in spx_in_cpx]
    spx_in_shape = set(sum([v_to_s[v] for v in vert_in_shape], []))
    segments = [alpha_exposed_segments(dt.simplices[spx_ix], dt, alpha) for spx_ix in spx_in_shape]
    segments = list(set(sum(segments, [])))
    return vert_in_shape, spx_in_shape, segments


def switch_to_plane(points):
    """
    Given three points, change frame of reference to get coordinates of these points in the plane they define
    """
    points = np.asarray(points)
    c_points = points - points[0, :]
    p0, p1, p2 = c_points
    # Compute an orthogonal basis
    v0 = p1 - p0
    vs = p2 - p0
    if norm(v0) == 0 or norm(vs) == 0:
        return None, None, None
    e0 = v0 / norm(v0)
    v2 = -np.cross(e0, vs)
    e2 = v2 / norm(v2)
    v1 = np.cross(e0, e2)
    e1 = v1 / norm(v1)
    basis = np.vstack((e0, e1, e2)).T

    # proj = np.asarray([[np.dot(p, e) for e in basis] for p in c_points])
    proj = c_points @ np.linalg.inv(basis)
    return proj, basis, points[0, :]


def circum_circle_3d(points):
    """
    Find the circumcirclee through 3 points in 3D
    """
    p_points, basis, origin = switch_to_plane(points)
    if basis is None:
        return None, None, None
    radius, center = circum(p_points[:, :2])
    if np.isinf(radius):
        return None, None, None
    center_3d = np.array([*center, 0])
    center_3d = basis @ center_3d
    center_3d += origin
    return radius, center_3d, basis


def spheres_3points(points, radius):
    """
    Get the centers of the two spheres defined by three points and a radius
    """
    r, center_3d, basis = circum_circle_3d(points)
    # Pythagore for distance from center of circle to center of sphere
    if r is None or r > radius:
        return np.zeros((2, 3)) + np.inf
    dist_to_center = np.sqrt(radius ** 2 - r ** 2)
    s_centers = [basis[:, 2] * dist_to_center, -basis[:, 2] * dist_to_center]
    return s_centers


def alpha_exposed_faces(simplex, dt, alpha):
    indices, indptr = dt.vertex_neighbor_vertices
    neigh = set(np.hstack([indptr[indices[p]:indices[p + 1]] for p in simplex]))
    segments = []
    for tri in combinations(simplex, 3):
        c_neigh = neigh - set(tri)
        neigh_coords = dt.points[list(c_neigh), :]
        centers = spheres_3points(dt.points[tri, :], alpha)
        dists = [cdist(np.atleast_2d(c), neigh_coords) for c in centers]
        exposed = [np.all(d > alpha) for d in dists]
        if exposed[0] ^ exposed[1]:
            segments.append(tri)
    return segments


def get_alpha_shape_3d(spx_ix, dt, alpha):
    vert_in_ch = set(dt.convex_hull.reshape(-1))
    spx_in_cpx = set(spx_ix)
    vert_in_cpx = set(dt.simplices[spx_ix].reshape(-1))
    v_to_s = vertex_to_simplices(vert_in_cpx, dt)
    vert_in_shape = set()
    for v in vert_in_cpx:
        if v in vert_in_ch:
            vert_in_shape.add(v)
            continue
        if all(s in spx_in_cpx for s in v_to_s[v]):
            continue
        vert_in_shape.add(v)
    # spx_in_shape = [list(filter(lambda v: v in vert_in_shape, dt.simplices[s])) for s in spx_in_cpx]
    spx_in_shape = set(sum([v_to_s[v] for v in vert_in_shape], []))
    faces = [alpha_exposed_faces(dt.simplices[spx_ix], dt, alpha) for spx_ix in spx_in_shape]
    faces = list(set(sum(faces, [])))
    return vert_in_shape, spx_in_shape, faces


def fish_shape(coords, alpha=1):
    dt = Delaunay(coords)
    radii, spx_ix = get_alpha_complex(dt.simplices, coords, alpha=alpha)
    vert_in_shape, spx_in_shape, faces = get_alpha_shape_3d(spx_ix, dt, alpha)
    return dt.points, faces

def alpha_fish_new(points, alpha=.15):
    # points = rec.coords[:, :2]
    dt = Delaunay(points)
    radii, spx_ix = get_alpha_complex(dt.simplices, points, alpha=alpha)
    vert_shape, spx_in_shape, seg_shape = get_alpha_shape(spx_ix, dt, alpha=alpha)
    test = {}
    for iloop, seg in enumerate(seg_shape):
        pts = np.vstack([dt.points[s, :] for s in seg])
        test[iloop] = pts
    return test
