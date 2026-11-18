"""Per-observation labels derived from the reference route.

The .route file is an ordered list of arcs, not a label per GPS point. We align
the track to the route with a monotone dynamic programme: point i is assigned a
route position j(i) with j(i) >= j(i-1), minimising the summed point-to-arc
distance. Aligning positions (not arc ids) handles loops and repeated arcs.

  cost[i, j] = d(i, r_j) + min_{j' <= j} cost[i-1, j']

Distances are clipped at CLIP_M so a few far outliers cannot drag the alignment.
Points whose aligned distance exceeds MATCH_M are marked unmatched and are left
out of training and metrics.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment

CLIP_M = 100.0
MATCH_M = 50.0


def align_to_route(pts_xy: np.ndarray, route: np.ndarray, net) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (label_arc, aligned_distance, matched_mask) for every point."""
    n, m = len(pts_xy), len(route)
    A, B = net.A[route], net.B[route]
    back = np.empty((n, m), dtype=np.int16 if m < 32767 else np.int32)
    idx = np.arange(m)

    def dist_row(i):
        d, _ = point_segment(pts_xy[i, 0], pts_xy[i, 1], A[:, 0], A[:, 1], B[:, 0], B[:, 1])
        return np.minimum(d, CLIP_M)

    cost = dist_row(0)
    back[0] = idx
    for i in range(1, n):
        run_min = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost == run_min, idx, 0))
        cost = dist_row(i) + run_min
        back[i] = arg

    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]

    label = route[j]
    d, _ = point_segment(pts_xy[:, 0], pts_xy[:, 1], net.A[label, 0], net.A[label, 1],
                         net.B[label, 0], net.B[label, 1])
    return label, d, d <= MATCH_M
