"""Road network of one trip, in local metres, with a spatial index for candidates.

Arcs are directed straight segments between two nodes. Most roads appear twice
(A->B and B->A, "reverse twins"); they share geometry and only differ in direction.
The spatial index is built on undirected roads, densified every DENSIFY_M metres,
and expanded back to the directed arcs when candidates are returned.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing_deg, point_segment

DENSIFY_M = 20.0


class Network:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection):
        self.proj = proj
        self.arcs = arcs
        self.n_nodes = len(nodes_lonlat)
        self.node_xy = proj.to_xy(nodes_lonlat)
        self.A = self.node_xy[arcs[:, 0]]
        self.B = self.node_xy[arcs[:, 1]]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing_deg(d[:, 0], d[:, 1])
        self.out_degree = np.bincount(arcs[:, 0], minlength=self.n_nodes)

        # Reverse twin of each arc (-1 if the road is one-way).
        key = {(int(a), int(b)): i for i, (a, b) in enumerate(arcs)}
        self.twin = np.array([key.get((int(b), int(a)), -1) for a, b in arcs], dtype=np.int64)

        # One representative per undirected road, with its directed arcs (second may be -1).
        rep_mask = (self.twin < 0) | (np.arange(len(arcs)) < self.twin)
        self.rep = np.flatnonzero(rep_mask)
        self.rep_arcs = np.column_stack([self.rep, np.where(self.twin[self.rep] >= 0, self.twin[self.rep], -1)])

        # Densify representatives for the KD-tree.
        n_pts = np.maximum(1, np.ceil(self.length[self.rep] / DENSIFY_M).astype(int)) + 1
        owner = np.repeat(np.arange(len(self.rep)), n_pts)
        frac = np.concatenate([np.linspace(0, 1, n) for n in n_pts])
        a, b = self.A[self.rep][owner], self.B[self.rep][owner]
        pts = a + (b - a) * frac[:, None]
        self._tree = cKDTree(pts)
        self._owner = owner

        # Directed node graph for shortest paths (weights = arc length).
        w = np.maximum(self.length, 1e-3)
        self.graph = csr_matrix((w, (arcs[:, 0], arcs[:, 1])), shape=(self.n_nodes, self.n_nodes))

    def candidates(self, pts_xy: np.ndarray, radius: float = 50.0, k: int = 12,
                   fallback_radius: float = 150.0, fallback_k: int = 3, q: int = 128):
        """Candidate arcs per point.

        Returns long-format arrays (point_idx, arc_id, dist, frac): up to k directed
        arcs within `radius`, or the nearest `fallback_k` within `fallback_radius`
        when nothing is inside `radius`.
        """
        n = len(pts_xy)
        q = min(q, len(self._owner))
        dd, ii = self._tree.query(pts_xy, k=q, distance_upper_bound=fallback_radius + DENSIFY_M)
        if q == 1:
            dd, ii = dd[:, None], ii[:, None]
        valid = np.isfinite(dd)
        rep_idx = np.where(valid, self._owner[np.minimum(ii, len(self._owner) - 1)], -1)

        # Remove duplicate roads within each row.
        rep_idx = np.sort(rep_idx, axis=1)
        dup = np.zeros_like(rep_idx, dtype=bool)
        dup[:, 1:] = rep_idx[:, 1:] == rep_idx[:, :-1]
        rep_idx[dup] = -1

        arc_rep = np.where(rep_idx >= 0, self.rep[np.maximum(rep_idx, 0)], 0)
        px, py = pts_xy[:, 0:1], pts_xy[:, 1:2]
        dist, frac = point_segment(px, py, self.A[arc_rep, 0], self.A[arc_rep, 1],
                                   self.B[arc_rep, 0], self.B[arc_rep, 1])
        dist = np.where(rep_idx >= 0, dist, np.inf)

        # Expand each road to its directed arcs (twin gets the mirrored fraction).
        pair = self.rep_arcs[np.maximum(rep_idx, 0)]                      # (n, q, 2)
        arc_id = np.concatenate([pair[:, :, 0], pair[:, :, 1]], axis=1)
        d2 = np.concatenate([dist, np.where(pair[:, :, 1] >= 0, dist, np.inf)], axis=1)
        f2 = np.concatenate([frac, 1.0 - frac], axis=1)

        order = np.argsort(d2, axis=1, kind="stable")
        arc_id = np.take_along_axis(arc_id, order, axis=1)
        d2 = np.take_along_axis(d2, order, axis=1)
        f2 = np.take_along_axis(f2, order, axis=1)

        rank = np.arange(d2.shape[1])[None, :]
        inside = (d2 <= radius) & (rank < k)
        none_inside = ~inside.any(axis=1)
        fallback = none_inside[:, None] & (d2 <= fallback_radius) & (rank < fallback_k)
        keep = inside | fallback
        pi, ci = np.nonzero(keep)
        return pi, arc_id[pi, ci], d2[pi, ci], f2[pi, ci]

    def connected(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where arc b starts at the end node of arc a (or a == b)."""
        return (a == b) | (self.arcs[a, 1] == self.arcs[b, 0])
