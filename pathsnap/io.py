"""Readers for the raw PathSnap trip files.

Each trip folder <id>/ holds four headerless, whitespace-separated files:
  <id>.track  lon, lat, t (seconds)      one row per GPS observation
  <id>.nodes  lon, lat                   row index = node id (0-based)
  <id>.arcs   from_node, to_node         row index = arc id (0-based)
  <id>.route  arc id                     ordered reference route (may be absent
                                         for evaluator trips on the client side)
Node and arc ids are local to the trip; they are never shared between trips.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DATA_DIR = os.environ.get("PATHSNAP_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "raw"))


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (V, 2) lon, lat
    arcs: np.ndarray             # (E, 2) from, to
    route: np.ndarray | None     # (M,) arc ids, or None


def _read(path: str, dtype=float, ndmin=2) -> np.ndarray:
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    trip_id = os.path.basename(os.path.normpath(trip_dir))
    base = os.path.join(trip_dir, trip_id)
    track = _read(base + ".track")
    nodes = _read(base + ".nodes")
    arcs = _read(base + ".arcs", dtype=np.int64)
    route = None
    if with_route and os.path.exists(base + ".route"):
        route = _read(base + ".route", dtype=np.int64, ndmin=1)

    # Clean the track: sort by time, drop repeated timestamps and NaN rows.
    track = track[np.isfinite(track).all(axis=1)]
    track = track[np.argsort(track[:, 2], kind="stable")]
    keep = np.ones(len(track), dtype=bool)
    keep[1:] = np.diff(track[:, 2]) > 0
    track = track[keep]
    return Trip(trip_id, track, nodes, arcs, route)


def list_trips(partition: str, data_dir: str = DATA_DIR) -> list[str]:
    """Trip folders of a partition ('train', 'student_test' or an evaluator folder)."""
    root = os.path.join(data_dir, partition)
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
