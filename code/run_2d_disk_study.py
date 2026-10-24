#!/usr/bin/env python3
"""Two-dimensional hp-CutFEM pilot on a translated disk.

The code intentionally has no external finite-element dependency.  It uses

* a fixed triangular background mesh,
* a hierarchical H1-conforming basis (vertices, integrated-Legendre edges,
  and element bubbles),
* exact circle geometry with polar and boundary quadrature split at all mesh
  intersections,
* complete-background-element patches,
* H1 polynomial projection on each patch,
* gradient and optional h^{-2} L2 projection-residual stabilization, and
* a symmetric overlapping patch Schwarz preconditioner.

It writes machine-readable CSV files.  It does not edit the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.polynomial import Legendre, Polynomial


TWO_PI = 2.0 * math.pi
Poly2 = Dict[Tuple[int, int], float]
_GEOMETRY_DIAGNOSTIC_CACHE: Dict[Tuple[object, ...], Dict[int, float]] = {}


def p_clean(a: Poly2, tol: float = 1.0e-15) -> Poly2:
    return {ij: float(c) for ij, c in a.items() if abs(c) > tol}


def p_add(a: Poly2, b: Poly2, alpha: float = 1.0) -> Poly2:
    out = dict(a)
    for ij, c in b.items():
        out[ij] = out.get(ij, 0.0) + alpha * c
    return p_clean(out)


def p_scale(a: Poly2, alpha: float) -> Poly2:
    return p_clean({ij: alpha * c for ij, c in a.items()})


def p_mul(a: Poly2, b: Poly2) -> Poly2:
    out: Poly2 = {}
    for (i, j), ca in a.items():
        for (k, ell), cb in b.items():
            key = (i + k, j + ell)
            out[key] = out.get(key, 0.0) + ca * cb
    return p_clean(out)


def p_pow(a: Poly2, n: int) -> Poly2:
    out: Poly2 = {(0, 0): 1.0}
    base = dict(a)
    m = n
    while m:
        if m & 1:
            out = p_mul(out, base)
        base = p_mul(base, base)
        m //= 2
    return out


def p_linear_combination(coeff: np.ndarray, polys: Sequence[Poly2]) -> Poly2:
    out: Poly2 = {}
    for alpha, poly in zip(coeff, polys):
        if abs(alpha) > 1.0e-15:
            out = p_add(out, poly, float(alpha))
    return p_clean(out)


def eval_polys(
    polys: Sequence[Poly2], r: np.ndarray, s: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(polys)
    vals = np.zeros((n, r.size))
    grads = np.zeros((n, 2, r.size))
    for a, poly in enumerate(polys):
        for (i, j), c in poly.items():
            vals[a] += c * (r**i) * (s**j)
            if i:
                grads[a, 0] += c * i * (r ** (i - 1)) * (s**j)
            if j:
                grads[a, 1] += c * j * (r**i) * (s ** (j - 1))
    return vals, grads


def gauss_legendre_interval(
    n: int, a: float, b: float
) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * ((b - a) * x + (a + b)), 0.5 * (b - a) * w


def triangle_quadrature(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tensor Gauss rule under r=x, s=(1-x)y for the reference triangle."""
    x, wx = gauss_legendre_interval(n, 0.0, 1.0)
    y, wy = gauss_legendre_interval(n, 0.0, 1.0)
    rr, yy = np.meshgrid(x, y, indexing="ij")
    r = rr.ravel()
    s = ((1.0 - rr) * yy).ravel()
    w = (wx[:, None] * wy[None, :] * (1.0 - rr)).ravel()
    return r, s, w


LAMBDAS: Tuple[Poly2, Poly2, Poly2] = (
    {(0, 0): 1.0, (1, 0): -1.0, (0, 1): -1.0},
    {(1, 0): 1.0},
    {(0, 1): 1.0},
)
EDGE_PAIRS = ((0, 1), (1, 2), (2, 0))


def integrated_legendre_edge_mode(
    degree: int, lam_lo: Poly2, lam_hi: Poly2
) -> Poly2:
    """Homogeneous extension S^n L_n(D/S), with L_n(-1)=L_n(1)=0."""
    integ = Legendre.basis(degree - 1).integ(lbnd=-1.0)
    power = integ.convert(kind=Polynomial)
    S = p_add(lam_lo, lam_hi)
    D = p_add(lam_hi, lam_lo, alpha=-1.0)
    out: Poly2 = {}
    for k, ck in enumerate(power.coef):
        out = p_add(out, p_scale(p_mul(p_pow(D, k), p_pow(S, degree - k)), ck))
    return p_clean(out)


def reference_bubbles(p: int) -> List[Poly2]:
    if p < 3:
        return []
    bubble = p_mul(p_mul(LAMBDAS[0], LAMBDAS[1]), LAMBDAS[2])
    candidates: List[Poly2] = []
    for total in range(p - 2):
        for i in range(total + 1):
            j = total - i
            candidates.append(p_mul(bubble, {(i, j): 1.0}))
    r, s, w = triangle_quadrature(max(7, p + 3))
    v, g = eval_polys(candidates, r, s)
    gram = (v * w) @ v.T
    gram += (g[:, 0] * w) @ g[:, 0].T
    gram += (g[:, 1] * w) @ g[:, 1].T
    L = np.linalg.cholesky(0.5 * (gram + gram.T))
    transform = np.linalg.inv(L)
    return [p_linear_combination(row, candidates) for row in transform]


def reference_basis(p: int, vertex_ids: Sequence[int]) -> List[Poly2]:
    polys: List[Poly2] = [dict(lam) for lam in LAMBDAS]
    for a, b in EDGE_PAIRS:
        if vertex_ids[a] < vertex_ids[b]:
            lam_lo, lam_hi = LAMBDAS[a], LAMBDAS[b]
        else:
            lam_lo, lam_hi = LAMBDAS[b], LAMBDAS[a]
        for degree in range(2, p + 1):
            polys.append(integrated_legendre_edge_mode(degree, lam_lo, lam_hi))
    polys.extend(reference_bubbles(p))
    expected = (p + 1) * (p + 2) // 2
    if len(polys) != expected:
        raise RuntimeError(f"basis size {len(polys)} != {expected}")
    return polys


@dataclass(frozen=True)
class Element:
    eid: int
    vids: Tuple[int, int, int]
    xy: np.ndarray
    J: np.ndarray
    detJ: float
    invJT: np.ndarray


@dataclass
class Mesh:
    N: int
    vertices: np.ndarray
    elements: List[Element]
    edges: np.ndarray
    h: float

    def locate(self, point: np.ndarray) -> int:
        x, y = float(point[0]), float(point[1])
        eps = 5.0e-13
        i = min(self.N - 1, max(0, int(math.floor((x + 1.0) / self.h + eps))))
        j = min(self.N - 1, max(0, int(math.floor((y + 1.0) / self.h + eps))))
        x0 = -1.0 + i * self.h
        y0 = -1.0 + j * self.h
        a = (x - x0) / self.h
        b = (y - y0) / self.h
        tri = 0 if b <= a + 2.0e-12 else 1
        return 2 * (j * self.N + i) + tri


def build_mesh(N: int) -> Mesh:
    grid = np.linspace(-1.0, 1.0, N + 1)
    vertices = np.array([(x, y) for y in grid for x in grid], dtype=float)

    def vid(i: int, j: int) -> int:
        return j * (N + 1) + i

    elements: List[Element] = []
    edge_keys = set()
    for j in range(N):
        for i in range(N):
            bl, br = vid(i, j), vid(i + 1, j)
            tl, tr = vid(i, j + 1), vid(i + 1, j + 1)
            for vids in ((bl, br, tr), (bl, tr, tl)):
                xy = vertices[np.array(vids)]
                J = np.column_stack((xy[1] - xy[0], xy[2] - xy[0]))
                eid = len(elements)
                elements.append(
                    Element(eid, vids, xy, J, float(np.linalg.det(J)), np.linalg.inv(J).T)
                )
                for a, b in EDGE_PAIRS:
                    edge_keys.add(tuple(sorted((vids[a], vids[b]))))
    edges = np.array(
        [(vertices[a], vertices[b]) for a, b in sorted(edge_keys)], dtype=float
    )
    return Mesh(N, vertices, elements, edges, 2.0 / N)


def point_segment_distance(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    t = float(np.dot(c - a, d) / np.dot(d, d))
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(c - (a + t * d)))


def point_in_triangle(point: np.ndarray, tri: np.ndarray, tol: float = 1.0e-13) -> bool:
    J = np.column_stack((tri[1] - tri[0], tri[2] - tri[0]))
    rs = np.linalg.solve(J, point - tri[0])
    return bool(rs[0] >= -tol and rs[1] >= -tol and rs.sum() <= 1.0 + tol)


def point_triangle_distance(c: np.ndarray, tri: np.ndarray) -> float:
    if point_in_triangle(c, tri):
        return 0.0
    return min(
        point_segment_distance(c, tri[0], tri[1]),
        point_segment_distance(c, tri[1], tri[2]),
        point_segment_distance(c, tri[2], tri[0]),
    )


def classify_elements(
    mesh: Mesh, center: np.ndarray, radius: float
) -> Tuple[List[int], List[int], List[int]]:
    active, interior, cut = [], [], []
    for e in mesh.elements:
        d_vertices = np.linalg.norm(e.xy - center[None, :], axis=1)
        is_interior = bool(np.max(d_vertices) < radius - 5.0e-13)
        is_active = point_triangle_distance(center, e.xy) < radius - 5.0e-13
        if is_active:
            active.append(e.eid)
            if is_interior:
                interior.append(e.eid)
            else:
                cut.append(e.eid)
    return active, interior, cut


def circle_mesh_intersection_angles(
    mesh: Mesh, center: np.ndarray, radius: float
) -> List[float]:
    """Directions where the circle meets mesh edges, including tangencies."""
    angles: List[float] = []
    for ab in mesh.edges:
        a, b = ab
        e = b - a
        q = a - center
        A = float(np.dot(e, e))
        B = 2.0 * float(np.dot(q, e))
        C = float(np.dot(q, q) - radius * radius)
        disc = B * B - 4.0 * A * C
        if disc < -1.0e-13:
            continue
        disc = max(0.0, disc)
        root = math.sqrt(disc)
        for t in ((-B - root) / (2.0 * A), (-B + root) / (2.0 * A)):
            if -1.0e-11 <= t <= 1.0 + 1.0e-11:
                x = a + min(1.0, max(0.0, t)) * e
                d = x - center
                angles.append(math.atan2(d[1], d[0]) % TWO_PI)
    return angles


def angular_partition(
    mesh: Mesh, center: np.ndarray, radius: float, refinement: int = 1
) -> np.ndarray:
    angles = list(
        np.linspace(0.0, TWO_PI, 12 * refinement * mesh.N + 1)[:-1]
    )
    delta = mesh.vertices - center[None, :]
    for x, y in delta:
        if x * x + y * y <= (1.6 * radius) ** 2:
            angles.append(math.atan2(y, x) % TWO_PI)
    # Radial limits switch between mesh edges and the circle at these angles.
    # Split there so each angular Gauss rule sees smooth integration limits.
    angles.extend(circle_mesh_intersection_angles(mesh, center, radius))
    angles = np.sort(np.mod(np.asarray(angles), TWO_PI))
    kept = [float(angles[0])]
    for a in angles[1:]:
        if float(a) - kept[-1] > 1.0e-11:
            kept.append(float(a))
    return np.asarray(kept)


def boundary_partition(
    mesh: Mesh, center: np.ndarray, radius: float, refinement: int = 1
) -> np.ndarray:
    angles = list(
        np.linspace(0.0, TWO_PI, 16 * refinement * mesh.N + 1)[:-1]
    )
    angles.extend(circle_mesh_intersection_angles(mesh, center, radius))
    angles = np.sort(np.mod(np.asarray(angles), TWO_PI))
    kept = [float(angles[0])]
    for a in angles[1:]:
        if float(a) - kept[-1] > 2.0e-11:
            kept.append(float(a))
    return np.asarray(kept)


def cyclic_intervals(angles: np.ndarray) -> Iterable[Tuple[float, float]]:
    for i, a in enumerate(angles):
        b = angles[i + 1] if i + 1 < angles.size else angles[0] + TWO_PI
        if b - a > 1.0e-12:
            yield float(a), float(b)


def ray_edge_radii(
    mesh: Mesh, center: np.ndarray, radius: float, direction: np.ndarray
) -> np.ndarray:
    a = mesh.edges[:, 0, :]
    e = mesh.edges[:, 1, :] - a
    q = a - center[None, :]
    den = direction[0] * e[:, 1] - direction[1] * e[:, 0]
    good = np.abs(den) > 1.0e-13
    rho = np.full(den.shape, np.nan)
    t = np.full(den.shape, np.nan)
    rho[good] = (q[good, 0] * e[good, 1] - q[good, 1] * e[good, 0]) / den[
        good
    ]
    t[good] = (q[good, 0] * direction[1] - q[good, 1] * direction[0]) / den[
        good
    ]
    mask = good & (rho > 1.0e-12) & (rho < radius - 1.0e-12)
    mask &= (t > -1.0e-11) & (t < 1.0 + 1.0e-11)
    vals = np.concatenate(([0.0], rho[mask], [radius]))
    vals = np.unique(np.round(vals, 13))
    return np.sort(vals)


def reference_coordinates(element: Element, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.linalg.solve(element.J, (points - element.xy[0]).T)
    return rs[0], rs[1]


def local_dof_keys(p: int, e: Element) -> List[Tuple]:
    keys: List[Tuple] = [("v", v) for v in e.vids]
    for a, b in EDGE_PAIRS:
        lo, hi = sorted((e.vids[a], e.vids[b]))
        for degree in range(2, p + 1):
            keys.append(("e", lo, hi, degree))
    for k in range((p - 1) * (p - 2) // 2):
        keys.append(("i", e.eid, k))
    return keys


def build_dofs(
    mesh: Mesh, active: Sequence[int], p: int
) -> Tuple[Dict[Tuple, int], Dict[int, np.ndarray], Dict[int, List[Poly2]]]:
    dof_map: Dict[Tuple, int] = {}
    element_dofs: Dict[int, np.ndarray] = {}
    basis_cache: Dict[Tuple[bool, bool, bool], List[Poly2]] = {}
    element_basis: Dict[int, List[Poly2]] = {}
    for eid in active:
        e = mesh.elements[eid]
        pattern = (
            e.vids[0] < e.vids[1],
            e.vids[1] < e.vids[2],
            e.vids[2] < e.vids[0],
        )
        if pattern not in basis_cache:
            basis_cache[pattern] = reference_basis(p, e.vids)
        element_basis[eid] = basis_cache[pattern]
        loc = []
        for key in local_dof_keys(p, e):
            if key not in dof_map:
                dof_map[key] = len(dof_map)
            loc.append(dof_map[key])
        element_dofs[eid] = np.asarray(loc, dtype=int)
    return dof_map, element_dofs, element_basis


def active_adjacency(mesh: Mesh, active: Sequence[int]) -> Dict[int, set]:
    edge_to_elements: Dict[Tuple[int, int], List[int]] = {}
    active_set = set(active)
    for eid in active:
        e = mesh.elements[eid]
        for a, b in EDGE_PAIRS:
            key = tuple(sorted((e.vids[a], e.vids[b])))
            edge_to_elements.setdefault(key, []).append(eid)
    adj = {eid: set() for eid in active}
    for elems in edge_to_elements.values():
        for a in elems:
            for b in elems:
                if a != b and b in active_set:
                    adj[a].add(b)
    return adj


def build_patches(
    mesh: Mesh, active: Sequence[int], cut: Sequence[int], layers: int = 2
) -> List[List[int]]:
    adj = active_adjacency(mesh, active)
    patches = []
    for seed in cut:
        patch = {seed}
        frontier = {seed}
        for _ in range(layers):
            new = set()
            for eid in frontier:
                new.update(adj[eid])
            new -= patch
            patch.update(new)
            frontier = new
        patches.append(sorted(patch))
    return patches


def physical_basis(
    element: Element, polys: Sequence[Poly2], points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r, s = reference_coordinates(element, points)
    vals, grads_ref = eval_polys(polys, r, s)
    grads = np.einsum("ab,ibq->iaq", element.invJT, grads_ref)
    return vals, grads


def volume_quadrature_groups(
    mesh: Mesh,
    center: np.ndarray,
    radius: float,
    p: int,
    active: Sequence[int],
    quadrature_boost: int = 0,
) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], Dict[int, float]]:
    active_set = set(active)
    edge_a = mesh.edges[:, 0, :]
    edge_b = mesh.edges[:, 1, :]
    _ = (edge_a, edge_b)  # retained for easy profiling hooks
    angular_refinement = max(1, 1 + quadrature_boost // 2)
    angles = angular_partition(mesh, center, radius, angular_refinement)
    qtheta = max(5, p + 3 + quadrature_boost)
    qrho = max(5, p + 3 + quadrature_boost)
    groups: List[Tuple[int, np.ndarray, np.ndarray]] = []
    area = {eid: 0.0 for eid in active}
    for aa, bb in cyclic_intervals(angles):
        thetas, wtheta = gauss_legendre_interval(qtheta, aa, bb)
        for theta, wt in zip(thetas, wtheta):
            direction = np.array([math.cos(theta), math.sin(theta)])
            radii = ray_edge_radii(mesh, center, radius, direction)
            for ra, rb in zip(radii[:-1], radii[1:]):
                if rb - ra < 2.0e-12:
                    continue
                midpoint = center + 0.5 * (ra + rb) * direction
                eid = mesh.locate(midpoint)
                if eid not in active_set:
                    continue
                rho, wr = gauss_legendre_interval(qrho, float(ra), float(rb))
                points = center[None, :] + rho[:, None] * direction[None, :]
                weights = wt * wr * rho
                groups.append((eid, points, weights))
                area[eid] += float(np.sum(weights))
    return groups, area


def circle_triangle_intersection_area(
    triangle: np.ndarray, center: np.ndarray, radius: float
) -> float:
    """Exact area of the intersection of a disk and an oriented triangle."""
    shifted = triangle - center[None, :]
    total = 0.0
    for a, b in zip(shifted, np.roll(shifted, -1, axis=0)):
        direction = b - a
        aa = float(np.dot(direction, direction))
        bb = 2.0 * float(np.dot(a, direction))
        cc = float(np.dot(a, a) - radius * radius)
        parameters = [0.0, 1.0]
        discriminant = bb * bb - 4.0 * aa * cc
        if discriminant > 0.0:
            root = math.sqrt(discriminant)
            for value in ((-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)):
                if 0.0 < value < 1.0:
                    parameters.append(float(value))
        parameters = sorted(parameters)
        for left, right in zip(parameters[:-1], parameters[1:]):
            p = a + left * direction
            q = a + right * direction
            midpoint = 0.5 * (p + q)
            cross = float(p[0] * q[1] - p[1] * q[0])
            if float(np.dot(midpoint, midpoint)) <= radius * radius:
                total += 0.5 * cross
            else:
                dot = float(np.dot(p, q))
                total += 0.5 * radius * radius * math.atan2(cross, dot)
    return abs(total)


def geometry_area_diagnostic(
    mesh: Mesh,
    center: np.ndarray,
    radius: float,
    active: Sequence[int],
) -> Dict[int, float]:
    """Return exact, p-independent physical element areas for the cut diagnostic."""
    key = (
        mesh.N,
        float(center[0]),
        float(center[1]),
        float(radius),
    )
    cached = _GEOMETRY_DIAGNOSTIC_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    area = {
        eid: circle_triangle_intersection_area(
            mesh.elements[eid].xy, center, radius
        )
        for eid in active
    }
    _GEOMETRY_DIAGNOSTIC_CACHE[key] = dict(area)
    return area


def manufactured_solution_values(
    points: np.ndarray,
    solution_kind: str,
    singularity_center: np.ndarray | None = None,
    singularity_alpha: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact value, exact gradient, and Poisson forcing at points."""
    if solution_kind == "exponential":
        value = np.exp(points[:, 0] + points[:, 1])
        gradient = np.column_stack((value, value))
        forcing = -2.0 * value
        return value, gradient, forcing
    if solution_kind == "radial_power":
        if singularity_center is None:
            raise ValueError("radial_power requires singularity_center")
        if singularity_alpha <= 1.0:
            raise ValueError("radial_power requires alpha > 1 so that f is in L2")
        delta = points - np.asarray(singularity_center, dtype=float)[None, :]
        radius = np.linalg.norm(delta, axis=1)
        value = radius**singularity_alpha
        gradient_factor = np.zeros_like(radius)
        positive = radius > 0.0
        gradient_factor[positive] = (
            singularity_alpha * radius[positive] ** (singularity_alpha - 2.0)
        )
        gradient = gradient_factor[:, None] * delta
        forcing = -(singularity_alpha**2) * radius ** (singularity_alpha - 2.0)
        return value, gradient, forcing
    raise ValueError(f"unknown manufactured solution {solution_kind!r}")


def boundary_quadrature_groups(
    mesh: Mesh,
    center: np.ndarray,
    radius: float,
    p: int,
    active: Sequence[int],
    quadrature_boost: int = 0,
) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    active_set = set(active)
    angular_refinement = max(1, 1 + quadrature_boost // 2)
    angles = boundary_partition(mesh, center, radius, angular_refinement)
    qtheta = max(6, p + 4 + quadrature_boost)
    groups = []
    for aa, bb in cyclic_intervals(angles):
        midpoint_theta = 0.5 * (aa + bb)
        midpoint = center + radius * np.array(
            [math.cos(midpoint_theta), math.sin(midpoint_theta)]
        )
        eid = mesh.locate(midpoint)
        if eid not in active_set:
            continue
        theta, wt = gauss_legendre_interval(qtheta, aa, bb)
        normals = np.column_stack((np.cos(theta), np.sin(theta)))
        points = center[None, :] + radius * normals
        weights = radius * wt
        groups.append((eid, points, normals, weights))
    return groups


def assemble_volume_and_boundary(
    mesh: Mesh,
    p: int,
    center: np.ndarray,
    radius: float,
    active: Sequence[int],
    element_dofs: Dict[int, np.ndarray],
    element_basis: Dict[int, List[Poly2]],
    quadrature_boost: int = 0,
    solution_kind: str = "exponential",
    singularity_center: np.ndarray | None = None,
    singularity_alpha: float = 2.5,
) -> Dict[str, object]:
    ndof = 1 + max(int(np.max(d)) for d in element_dofs.values())
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    Mb = np.zeros((ndof, ndof))
    Cn = np.zeros((ndof, ndof))
    fvol = np.zeros(ndof)
    fbc_cons = np.zeros(ndof)
    fbc_pen = np.zeros(ndof)

    volume_groups, area = volume_quadrature_groups(
        mesh, center, radius, p, active, quadrature_boost
    )
    for eid, points, weights in volume_groups:
        e = mesh.elements[eid]
        idx = element_dofs[eid]
        vals, grads = physical_basis(e, element_basis[eid], points)
        local_K = np.einsum("iaq,q,jaq->ij", grads, weights, grads)
        local_M = (vals * weights) @ vals.T
        _, _, f = manufactured_solution_values(
            points, solution_kind, singularity_center, singularity_alpha
        )
        K[np.ix_(idx, idx)] += local_K
        M[np.ix_(idx, idx)] += local_M
        fvol[idx] += vals @ (weights * f)

    boundary_groups = boundary_quadrature_groups(
        mesh, center, radius, p, active, quadrature_boost
    )
    boundary_elements = set()
    for eid, points, normals, weights in boundary_groups:
        boundary_elements.add(eid)
        e = mesh.elements[eid]
        idx = element_dofs[eid]
        vals, grads = physical_basis(e, element_basis[eid], points)
        dn = np.einsum("iaq,qa->iq", grads, normals)
        Mb[np.ix_(idx, idx)] += (vals * weights) @ vals.T
        Cn[np.ix_(idx, idx)] += (dn * weights) @ vals.T
        g, _, _ = manufactured_solution_values(
            points, solution_kind, singularity_center, singularity_alpha
        )
        fbc_cons[idx] += -dn @ (weights * g)
        fbc_pen[idx] += vals @ (weights * g)

    return {
        "K": K,
        "M": M,
        "Mb": Mb,
        "Cn": Cn,
        "fvol": fvol,
        "fbc_cons": fbc_cons,
        "fbc_pen": fbc_pen,
        "volume_groups": volume_groups,
        "boundary_groups": boundary_groups,
        "area": area,
        "boundary_elements": sorted(boundary_elements),
    }


def eval_patch_polynomials(
    indices: Sequence[Tuple[int, int]],
    points: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xx = (points[:, 0] - center[0]) / scale
    yy = (points[:, 1] - center[1]) / scale
    max_degree = max(max(i, j) for i, j in indices)
    vx, dx, vy, dy = [], [], [], []
    for degree in range(max_degree + 1):
        L = Legendre.basis(degree)
        vx.append(L(xx))
        vy.append(L(yy))
        dL = L.deriv()
        dx.append(dL(xx) / scale if degree else np.zeros_like(xx))
        dy.append(dL(yy) / scale if degree else np.zeros_like(yy))
    vals = np.zeros((len(indices), points.shape[0]))
    grads = np.zeros((len(indices), 2, points.shape[0]))
    for a, (i, j) in enumerate(indices):
        vals[a] = vx[i] * vy[j]
        grads[a, 0] = dx[i] * vy[j]
        grads[a, 1] = vx[i] * dy[j]
    return vals, grads


def psd_cleanup(A: np.ndarray, diagnostics: dict | None = None) -> np.ndarray:
    A = 0.5 * (A + A.T)
    eigenvalues, vectors = np.linalg.eigh(A)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if diagnostics is not None:
        discarded = eigenvalues[eigenvalues < 2.0e-12 * scale]
        diagnostics.update(
            raw_min_eigenvalue=float(eigenvalues[0]),
            raw_spectral_scale=scale,
            clip_threshold=2.0e-12 * scale,
            clipped_modes=int(discarded.size),
            discarded_absolute_trace=float(np.sum(np.abs(discarded))),
        )
    eigenvalues[eigenvalues < 2.0e-12 * scale] = 0.0
    return (vectors * eigenvalues) @ vectors.T


def patch_projection_coefficients(
    Gq: np.ndarray,
    mq: np.ndarray,
    B: np.ndarray,
    mv: np.ndarray,
    projection_basis: str,
    gradient_samples: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return the H1 projector in the original scaled-Legendre coordinates.

    Both implementations produce the same polynomial projector in exact
    arithmetic.  The ``h1_orthonormal`` implementation separates the constant
    mode, removes the mean from all higher modes, and H1-orthonormalizes their
    gradients before computing the projection coefficients.
    """
    nq = Gq.shape[0]
    area = float(mq[0])
    if area <= 0.0:
        raise ValueError("patch has non-positive area")

    saddle = np.zeros((nq + 1, nq + 1))
    saddle[:nq, :nq] = Gq
    saddle[:nq, nq] = mq
    saddle[nq, :nq] = mq
    raw_saddle_condition = float(np.linalg.cond(saddle))

    # The normalized mean term makes the constant and gradient blocks
    # dimensionally comparable in two dimensions.
    augmented = 0.5 * (Gq + Gq.T) + np.outer(mq, mq) / (area * area)
    raw_augmented_condition = float(np.linalg.cond(augmented))

    if projection_basis == "scaled_legendre":
        rhs = np.vstack((B, mv[None, :]))
        try:
            projection = np.linalg.solve(saddle, rhs)[:nq]
        except np.linalg.LinAlgError:
            projection = np.linalg.lstsq(saddle, rhs, rcond=1.0e-12)[0][:nq]
        effective_condition = raw_saddle_condition
        orthogonality_error = 0.0
    elif projection_basis == "h1_orthonormal":
        # q_0 = 1.  Subtract the mean from every higher mode; this does not
        # alter its gradient.  A weighted QR factorization of the sampled
        # gradients avoids forming and factoring the normal equations.
        if gradient_samples is None:
            raise ValueError("gradient samples are required for H1 orthonormalization")
        mean_free = np.zeros((nq - 1, nq))
        mean_free[:, 0] = -mq[1:] / area
        mean_free[:, 1:] = np.eye(nq - 1)
        _, upper = np.linalg.qr(gradient_samples, mode="reduced")
        higher_coordinate_transform = np.linalg.solve(
            upper.T, np.eye(nq - 1)
        )
        higher_transform = higher_coordinate_transform @ mean_free

        transform = np.zeros((nq, nq))
        transform[0, 0] = 1.0
        transform[1:, :] = higher_transform
        orthonormal_gram = transform @ augmented @ transform.T
        identity = np.eye(nq)
        orthogonality_error = float(
            np.linalg.norm(orthonormal_gram - identity, ord=2)
        )
        effective_condition = float(np.linalg.cond(orthonormal_gram))

        projected_rhs = np.vstack(
            (mv[None, :] / area, higher_transform @ B)
        )
        projection = transform.T @ projected_rhs
    else:
        raise ValueError(
            "projection_basis must be 'scaled_legendre' or 'h1_orthonormal'"
        )

    diagnostics = {
        "projection_condition": effective_condition,
        "raw_saddle_condition": raw_saddle_condition,
        "raw_augmented_condition": raw_augmented_condition,
        "orthogonality_error": orthogonality_error,
    }
    return projection, diagnostics


def assemble_patch_stabilization(
    mesh: Mesh,
    p: int,
    patches: Sequence[Sequence[int]],
    element_dofs: Dict[int, np.ndarray],
    element_basis: Dict[int, List[Poly2]],
    ndof: int,
    quadrature_boost: int = 0,
    projection_basis: str = "scaled_legendre",
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Dict[str, float]]]:
    S1 = np.zeros((ndof, ndof))
    S0 = np.zeros((ndof, ndof))
    patch_dofs_all: List[np.ndarray] = []
    diagnostics: List[Dict[str, float]] = []
    r, s, wref = triangle_quadrature(max(6, p + 3 + quadrature_boost))
    indices = [(i, total - i) for total in range(p + 1) for i in range(total + 1)]

    for patch in patches:
        patch_dofs = np.unique(np.concatenate([element_dofs[eid] for eid in patch]))
        patch_dofs_all.append(patch_dofs)
        local_pos = {int(g): i for i, g in enumerate(patch_dofs)}
        nd = patch_dofs.size
        nq = len(indices)
        Kv = np.zeros((nd, nd))
        Mv = np.zeros((nd, nd))
        Gq = np.zeros((nq, nq))
        Mq = np.zeros((nq, nq))
        B = np.zeros((nq, nd))
        C = np.zeros((nq, nd))
        mq = np.zeros(nq)
        mv = np.zeros(nd)
        gradient_sample_blocks: List[np.ndarray] = []

        patch_vertices = np.vstack([mesh.elements[eid].xy for eid in patch])
        pcenter = 0.5 * (patch_vertices.min(axis=0) + patch_vertices.max(axis=0))
        scale = 0.5 * float(np.max(patch_vertices.max(axis=0) - patch_vertices.min(axis=0)))
        scale = max(scale, 0.5 * mesh.h)

        for eid in patch:
            e = mesh.elements[eid]
            points = e.xy[0][None, :] + (e.J @ np.vstack((r, s))).T
            weights = e.detJ * wref
            vals, grads = physical_basis(e, element_basis[eid], points)
            qvals, qgrads = eval_patch_polynomials(indices, points, pcenter, scale)
            loc = np.asarray([local_pos[int(g)] for g in element_dofs[eid]], dtype=int)
            if projection_basis == "h1_orthonormal":
                sqrt_weights = np.sqrt(weights)
                gradient_sample_blocks.append(
                    (qgrads[1:, 0, :] * sqrt_weights).T
                )
                gradient_sample_blocks.append(
                    (qgrads[1:, 1, :] * sqrt_weights).T
                )

            Kv[np.ix_(loc, loc)] += np.einsum(
                "iaq,q,jaq->ij", grads, weights, grads
            )
            Mv[np.ix_(loc, loc)] += (vals * weights) @ vals.T
            Gq += np.einsum("iaq,q,jaq->ij", qgrads, weights, qgrads)
            Mq += (qvals * weights) @ qvals.T
            B[:, loc] += np.einsum("iaq,q,jaq->ij", qgrads, weights, grads)
            C[:, loc] += (qvals * weights) @ vals.T
            mq += qvals @ weights
            mv[loc] += vals @ weights

        gradient_samples = (
            np.vstack(gradient_sample_blocks)
            if gradient_sample_blocks
            else None
        )
        projection, projection_diagnostics = patch_projection_coefficients(
            Gq, mq, B, mv, projection_basis, gradient_samples
        )

        loc_S1 = Kv - projection.T @ B - B.T @ projection + projection.T @ Gq @ projection
        loc_S0 = Mv - projection.T @ C - C.T @ projection + projection.T @ Mq @ projection
        cleanup_S1, cleanup_S0 = {}, {}
        loc_S1 = psd_cleanup(loc_S1, cleanup_S1)
        loc_S0 = psd_cleanup(loc_S0, cleanup_S0) / (mesh.h * mesh.h)
        S1[np.ix_(patch_dofs, patch_dofs)] += loc_S1
        S0[np.ix_(patch_dofs, patch_dofs)] += loc_S0
        diagnostics.append(
            {
                "elements": float(len(patch)),
                "dofs": float(nd),
                "s1_trace": float(np.trace(loc_S1)),
                "s0_trace": float(np.trace(loc_S0)),
                **projection_diagnostics,
                **{"s1_" + key: value for key, value in cleanup_S1.items()},
                **{"s0_" + key: value for key, value in cleanup_S0.items()},
            }
        )
    return S1, S0, patch_dofs_all, diagnostics


def assemble_problem(
    N: int,
    p: int,
    epsilon: float,
    radius: float = 0.62,
    patch_layers: int = 2,
    center_override: np.ndarray | None = None,
    quadrature_boost: int = 0,
    projection_basis: str = "scaled_legendre",
    solution_kind: str = "exponential",
    singularity_center: np.ndarray | None = None,
    singularity_alpha: float = 2.5,
) -> Dict[str, object]:
    mesh = build_mesh(N)
    if center_override is None:
        # The circle is almost tangent to a background vertex from the right.
        x_candidates = np.unique(mesh.vertices[:, 0])
        admissible_x = [
            x for x in x_candidates if x >= 2.0 * radius - 1.0 + 5.0e-3
        ]
        vertex_x = min(admissible_x, key=lambda x: (abs(x - 1.0 / 3.0), -x))
        center = np.array([vertex_x - radius + epsilon * mesh.h, 0.0])
    else:
        center = np.asarray(center_override, dtype=float)
    if np.max(np.abs(center)) + radius > 0.999:
        raise ValueError("translated disk leaves the background square")
    active, interior, cut = classify_elements(mesh, center, radius)
    dof_map, element_dofs, element_basis = build_dofs(mesh, active, p)
    components = assemble_volume_and_boundary(
        mesh,
        p,
        center,
        radius,
        active,
        element_dofs,
        element_basis,
        quadrature_boost,
        solution_kind,
        singularity_center,
        singularity_alpha,
    )
    boundary_cut = components["boundary_elements"]
    patches = build_patches(mesh, active, boundary_cut, patch_layers)
    S1, S0, patch_dofs, patch_diag = assemble_patch_stabilization(
        mesh,
        p,
        patches,
        element_dofs,
        element_basis,
        len(dof_map),
        quadrature_boost,
        projection_basis,
    )

    diagnostic_area = geometry_area_diagnostic(mesh, center, radius, active)
    area_T = 0.5 * mesh.h * mesh.h
    physical_fractions = [
        diagnostic_area[eid] / area_T
        for eid in cut
        if diagnostic_area.get(eid, 0.0) > 1.0e-13 * area_T
    ]
    eta_min = min(physical_fractions)
    result = dict(components)
    result.update(
        {
            "mesh": mesh,
            "p": p,
            "epsilon": epsilon,
            "radius": radius,
            "center": center,
            "active": active,
            "interior": interior,
            "cut": cut,
            "dof_map": dof_map,
            "element_dofs": element_dofs,
            "element_basis": element_basis,
            "patches": patches,
            "patch_dofs": patch_dofs,
            "patch_diagnostics": patch_diag,
            "S1": S1,
            "S0": S0,
            "eta_min": eta_min,
            "ndof": len(dof_map),
            "quadrature_boost": quadrature_boost,
            "projection_basis": projection_basis,
            "solution_kind": solution_kind,
            "singularity_center": (
                None
                if singularity_center is None
                else np.asarray(singularity_center, dtype=float)
            ),
            "singularity_alpha": singularity_alpha,
            "eta_diagnostic": "exact_disk_triangle_intersection",
        }
    )
    return result


def matrices_for(
    problem: Dict[str, object], gamma0: float, beta0: float, gamma1: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = int(problem["p"])
    h = float(problem["mesh"].h)
    K = np.asarray(problem["K"])
    Mb = np.asarray(problem["Mb"])
    Cn = np.asarray(problem["Cn"])
    S1 = np.asarray(problem["S1"])
    S0 = np.asarray(problem["S0"])
    base = K + gamma1 * S1 + gamma0 * S0 - Cn - Cn.T
    penalty = (p * p / h) * Mb
    A = 0.5 * (base + base.T) + beta0 * penalty
    metric = K + gamma1 * S1 + gamma0 * S0 + penalty
    rhs = (
        np.asarray(problem["fvol"])
        + np.asarray(problem["fbc_cons"])
        + beta0 * (p * p / h) * np.asarray(problem["fbc_pen"])
    )
    return A, 0.5 * (metric + metric.T), rhs


def generalized_min_eigenvalue(A: np.ndarray, metric: np.ndarray) -> float:
    diag_scale = max(1.0, float(np.max(np.diag(metric))))
    metric_reg = metric + 1.0e-12 * diag_scale * np.eye(metric.shape[0])
    L = np.linalg.cholesky(metric_reg)
    X = np.linalg.solve(L, A)
    transformed = np.linalg.solve(L, X.T).T
    return float(np.linalg.eigvalsh(0.5 * (transformed + transformed.T))[0])


def coercivity_threshold(
    problem: Dict[str, object],
    gamma0: float,
    gamma1: float = 0.1,
    target: float = 1.0e-8,
    iterations: int = 22,
) -> float:
    lo, hi = 0.0, 1.0
    A, metric, _ = matrices_for(problem, gamma0, hi, gamma1)
    while generalized_min_eigenvalue(A, metric) <= target:
        hi *= 2.0
        if hi > 4096:
            return math.inf
        A, metric, _ = matrices_for(problem, gamma0, hi, gamma1)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        A, metric, _ = matrices_for(problem, gamma0, mid, gamma1)
        if generalized_min_eigenvalue(A, metric) > target:
            hi = mid
        else:
            lo = mid
    return hi


def symmetric_condition(A: np.ndarray) -> float:
    d = np.sqrt(np.maximum(np.diag(A), 1.0e-30))
    scaled = A / d[:, None] / d[None, :]
    eig = np.linalg.eigvalsh(0.5 * (scaled + scaled.T))
    positive = eig[eig > 1.0e-12 * max(1.0, float(np.max(eig)))]
    if positive.size != eig.size:
        return math.inf
    return float(positive[-1] / positive[0])


def build_patch_schwarz_inverse(
    A: np.ndarray, patch_dofs: Sequence[np.ndarray]
) -> np.ndarray:
    n = A.shape[0]
    covered = np.zeros(n, dtype=int)
    unique_patches: List[np.ndarray] = []
    seen = set()
    for idx in patch_dofs:
        key = tuple(int(i) for i in idx)
        if key not in seen:
            seen.add(key)
            unique_patches.append(idx)
            covered[idx] += 1
    interior = np.flatnonzero(covered == 0)
    subdomains = list(unique_patches)
    if interior.size:
        subdomains.append(interior)
        covered[interior] += 1
    if np.any(covered == 0):
        raise RuntimeError("Schwarz decomposition does not cover all dofs")

    B_inv = np.zeros_like(A)
    for idx in subdomains:
        local = 0.5 * (A[np.ix_(idx, idx)] + A[np.ix_(idx, idx)].T)
        regularization = 1.0e-12 * max(1.0, float(np.max(np.diag(local))))
        local_inv = np.linalg.inv(local + regularization * np.eye(idx.size))
        weights = 1.0 / np.sqrt(covered[idx])
        B_inv[np.ix_(idx, idx)] += weights[:, None] * local_inv * weights[None, :]
    return 0.5 * (B_inv + B_inv.T)


def preconditioned_condition(A: np.ndarray, B_inv: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.diag(B_inv))))
    L = np.linalg.cholesky(B_inv + 1.0e-14 * scale * np.eye(A.shape[0]))
    transformed = L.T @ A @ L
    eig = np.linalg.eigvalsh(0.5 * (transformed + transformed.T))
    positive = eig[eig > 1.0e-11 * max(1.0, float(np.max(eig)))]
    if positive.size != eig.size:
        return math.inf
    return float(positive[-1] / positive[0])


def pcg_iterations(
    A: np.ndarray,
    b: np.ndarray,
    B_inv: np.ndarray,
    tolerance: float = 1.0e-9,
    maxiter: int = 3000,
) -> Tuple[int, float]:
    x = np.zeros_like(b)
    r = b - A @ x
    norm0 = float(np.linalg.norm(r))
    if norm0 == 0:
        return 0, 0.0
    z = B_inv @ r
    direction = z.copy()
    rz = float(np.dot(r, z))
    for iteration in range(1, maxiter + 1):
        Ad = A @ direction
        alpha = rz / float(np.dot(direction, Ad))
        x += alpha * direction
        r -= alpha * Ad
        rel = float(np.linalg.norm(r) / norm0)
        if rel <= tolerance:
            return iteration, rel
        z = B_inv @ r
        rz_new = float(np.dot(r, z))
        direction = z + (rz_new / rz) * direction
        rz = rz_new
    return maxiter, float(np.linalg.norm(r) / norm0)


def solution_errors(problem: Dict[str, object], coeff: np.ndarray) -> Tuple[float, float]:
    l2 = 0.0
    h1 = 0.0
    mesh: Mesh = problem["mesh"]
    element_dofs = problem["element_dofs"]
    element_basis = problem["element_basis"]
    for eid, points, weights in problem["volume_groups"]:
        e = mesh.elements[eid]
        vals, grads = physical_basis(e, element_basis[eid], points)
        loc = coeff[element_dofs[eid]]
        uh = loc @ vals
        guh = np.einsum("i,iaq->qa", loc, grads)
        exact, gexact, _ = manufactured_solution_values(
            points,
            str(problem.get("solution_kind", "exponential")),
            problem.get("singularity_center"),
            float(problem.get("singularity_alpha", 2.5)),
        )
        l2 += float(np.dot(weights, (uh - exact) ** 2))
        h1 += float(np.dot(weights, np.sum((guh - gexact) ** 2, axis=1)))
    return math.sqrt(max(0.0, l2)), math.sqrt(max(0.0, h1))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_studies(output_dir: Path, quick: bool = False) -> Dict[str, object]:
    start = time.time()
    beta0 = 16.0
    gamma1 = 0.1
    gamma0_values = (0.0, 0.1)
    cut_ps = (1, 2, 3) if quick else (1, 2, 3, 4)
    cut_eps = (1.0e-1, 1.0e-2, 1.0e-3) if quick else (
        1.0e-1,
        3.0e-2,
        1.0e-2,
        3.0e-3,
        1.0e-3,
    )
    N_cut = 4 if quick else 6

    cut_rows: List[Dict[str, object]] = []
    for p in cut_ps:
        for epsilon in cut_eps:
            problem = assemble_problem(N_cut, p, epsilon)
            for gamma0 in gamma0_values:
                threshold = coercivity_threshold(
                    problem, gamma0, gamma1, iterations=16 if quick else 21
                )
                A, _, rhs = matrices_for(problem, gamma0, beta0, gamma1)
                condition_diag = symmetric_condition(A)
                row = {
                    "N": N_cut,
                    "p": p,
                    "epsilon": epsilon,
                    "eta_min": problem["eta_min"],
                    "gamma0": gamma0,
                    "gamma1": gamma1,
                    "beta0_threshold": threshold,
                    "beta0_run": beta0,
                    "ndof": problem["ndof"],
                    "cut_elements": len(problem["cut"]),
                    "patches": len(problem["patches"]),
                    "condition_diagonal_scaled": condition_diag,
                }
                # Schwarz is the expensive diagnostic; run it at the most severe cut.
                if epsilon == cut_eps[-1]:
                    B_inv = build_patch_schwarz_inverse(A, problem["patch_dofs"])
                    row["condition_patch_schwarz"] = preconditioned_condition(A, B_inv)
                    iterations, residual = pcg_iterations(A, rhs, B_inv)
                    row["pcg_iterations_patch"] = iterations
                    row["pcg_relative_residual"] = residual
                    jacobi = np.diag(1.0 / np.maximum(np.diag(A), 1.0e-30))
                    row["condition_jacobi"] = preconditioned_condition(A, jacobi)
                    jacobi_iterations, _ = pcg_iterations(A, rhs, jacobi)
                    row["pcg_iterations_jacobi"] = jacobi_iterations
                cut_rows.append(row)
                print(
                    f"cut N={N_cut} p={p} eps={epsilon:.0e} gamma0={gamma0:g} "
                    f"eta={problem['eta_min']:.2e} threshold={threshold:.3g} "
                    f"kdiag={condition_diag:.2e}",
                    flush=True,
                )
    write_csv(output_dir / "disk_cut_sweep.csv", cut_rows)

    convergence_rows: List[Dict[str, object]] = []
    conv_N = (4, 6) if quick else (4, 6, 8, 10)
    conv_ps = (1, 2, 3) if quick else (1, 2, 3, 4)
    convergence_center = np.array([0.071, -0.043])
    for p in conv_ps:
        for N in conv_N:
            problem = assemble_problem(
                N, p, 1.0e-3, center_override=convergence_center
            )
            A, _, rhs = matrices_for(problem, 0.1, beta0, gamma1)
            coeff = np.linalg.solve(A, rhs)
            l2, h1 = solution_errors(problem, coeff)
            convergence_rows.append(
                {
                    "N": N,
                    "h": problem["mesh"].h,
                    "p": p,
                    "center_x": convergence_center[0],
                    "center_y": convergence_center[1],
                    "eta_min": problem["eta_min"],
                    "gamma0": 0.1,
                    "gamma1": gamma1,
                    "beta0": beta0,
                    "ndof": problem["ndof"],
                    "l2_error": l2,
                    "h1_error": h1,
                }
            )
            print(
                f"conv N={N} p={p} eta={problem['eta_min']:.2e} "
                f"L2={l2:.3e} H1={h1:.3e}",
                flush=True,
            )
    write_csv(output_dir / "disk_convergence.csv", convergence_rows)

    summary = {
        "description": "2D translated-disk hp-CutFEM pilot",
        "basis": "hierarchical vertices + integrated-Legendre edges + H1-orthogonalized bubbles",
        "patches": "graph-distance-two unions of complete active background triangles",
        "physical_control": "projected polynomial is used in the physical-domain operator; residuals are integrated on full patches",
        "stabilization": "H1 residual plus optional h^-2 L2 residual",
        "beta0_run": beta0,
        "gamma1": gamma1,
        "gamma0_values": gamma0_values,
        "quick": quick,
        "elapsed_seconds": time.time() - start,
        "cut_sweep_file": "disk_cut_sweep.csv",
        "convergence_file": "disk_convergence.csv",
    }
    with (output_dir / "disk_study_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "numerics_2d",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    summary = run_studies(args.output_dir, quick=args.quick)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
