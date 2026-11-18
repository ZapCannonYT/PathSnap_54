"""Features for each (GPS point, candidate arc) pair.

All features are computed from the GPS track and the road network only; the
reference route is never used here, so the same code runs at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff, bearing_deg

FEATURES = [
    "d_perp", "d_rank", "d_minus_min", "d_ratio",
    "heading_diff", "heading_valid", "speed", "dt_prev",
    "arc_len", "proj_frac", "at_endpoint", "n_cand",
    "in_prev_cands", "in_next_cands", "conn_prev", "conn_next", "end_out_degree",
]

MIN_MOVE_M = 2.0


def point_motion(xy: np.ndarray, t: np.ndarray):
    """Heading (deg), heading validity, speed (m/s) and dt to previous point."""
    n = len(xy)
    prv = np.r_[0, np.arange(n - 1)]
    nxt = np.r_[np.arange(1, n), n - 1]
    dx, dy = xy[nxt, 0] - xy[prv, 0], xy[nxt, 1] - xy[prv, 1]
    move = np.hypot(dx, dy)
    span = np.maximum(t[nxt] - t[prv], 1e-6)
    heading = bearing_deg(dx, dy)
    valid = move >= MIN_MOVE_M
    speed = move / span
    dt_prev = np.r_[0.0, np.diff(t)]
    return heading, valid, speed, dt_prev


def _pair_key(point_idx, value, n_values):
    return point_idx.astype(np.int64) * (n_values + 1) + value.astype(np.int64)


def build_features(net, xy: np.ndarray, t: np.ndarray, pi, arc, dist, frac) -> pd.DataFrame:
    heading, valid, speed, dt_prev = point_motion(xy, t)
    n_pts = len(xy)
    df = pd.DataFrame({"point_idx": pi, "arc_id": arc, "d_perp": dist, "proj_frac": frac})

    g = df.groupby("point_idx")["d_perp"]
    dmin = g.transform("min").to_numpy()
    df["d_rank"] = g.rank(method="dense").to_numpy()
    df["d_minus_min"] = dist - dmin
    df["d_ratio"] = (dist + 1.0) / (dmin + 1.0)
    df["n_cand"] = df.groupby("point_idx")["arc_id"].transform("size").to_numpy()

    hd = angle_diff(heading[pi], net.bearing[arc])
    df["heading_valid"] = valid[pi].astype(np.int8)
    df["heading_diff"] = np.where(valid[pi], hd, 90.0)
    df["speed"] = speed[pi]
    df["dt_prev"] = dt_prev[pi]
    df["arc_len"] = net.length[arc]
    df["at_endpoint"] = ((frac * net.length[arc] < 5.0) | ((1 - frac) * net.length[arc] < 5.0)).astype(np.int8)
    df["end_out_degree"] = net.out_degree[net.arcs[arc, 1]]

    # Is the same arc also a candidate of the neighbouring points?
    E = len(net.arcs)
    keys = _pair_key(pi, arc, E)
    df["in_prev_cands"] = np.isin(_pair_key(pi - 1, arc, E), keys).astype(np.int8)
    df["in_next_cands"] = np.isin(_pair_key(pi + 1, arc, E), keys).astype(np.int8)

    # Can this arc be reached from a candidate of the previous point (same arc or
    # directly connected), and can it continue into a candidate of the next point?
    V = net.n_nodes
    start, end = net.arcs[arc, 0], net.arcs[arc, 1]
    end_keys = _pair_key(pi + 1, end, V)          # this arc ends where point+1 may start
    start_keys = _pair_key(pi - 1, start, V)
    df["conn_prev"] = (np.isin(_pair_key(pi - 1, arc, E), keys)
                       | np.isin(_pair_key(pi, start, V), end_keys)).astype(np.int8)
    df["conn_next"] = (np.isin(_pair_key(pi + 1, arc, E), keys)
                       | np.isin(_pair_key(pi, end, V), start_keys)).astype(np.int8)
    df.loc[df["point_idx"] == 0, "conn_prev"] = 1
    df.loc[df["point_idx"] == n_pts - 1, "conn_next"] = 1
    return df
