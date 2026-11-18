"""Turn one trip into the candidate table used for training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features
from .geo import LocalProjection
from .io import Trip
from .labels import align_to_route
from .network import Network

RADIUS_M = 50.0
TOP_K = 12          # directed arcs, i.e. about 6 roads in both directions


def prepare_trip(trip: Trip):
    """Projection, network and projected track for a trip."""
    proj = LocalProjection.for_points(trip.track[:, :2])
    net = Network(trip.nodes, trip.arcs, proj)
    xy = proj.to_xy(trip.track[:, :2])
    return proj, net, xy


def trip_candidates(trip: Trip, net: Network, xy: np.ndarray, radius: float = RADIUS_M,
                    k: int = TOP_K) -> pd.DataFrame:
    pi, arc, dist, frac = net.candidates(xy, radius=radius, k=k)
    df = build_features(net, xy, trip.track[:, 2], pi, arc, dist, frac)
    df.insert(0, "trip_id", trip.trip_id)
    return df


def trip_points(trip: Trip, net: Network, xy: np.ndarray) -> pd.DataFrame:
    """Per-point table with the route-derived label (training/evaluation only)."""
    label, ldist, matched = align_to_route(xy, trip.route, net)
    return pd.DataFrame({"trip_id": trip.trip_id, "point_idx": np.arange(len(xy)),
                         "t": trip.track[:, 2], "label_arc": label,
                         "label_dist": ldist, "matched": matched})


def attach_labels(cands: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    lab = points.set_index("point_idx")
    cands = cands.copy()
    cands["matched"] = lab["matched"].to_numpy()[cands["point_idx"].to_numpy()]
    cands["y"] = (cands["arc_id"].to_numpy() == lab["label_arc"].to_numpy()[cands["point_idx"].to_numpy()]).astype(np.int8)
    return cands
