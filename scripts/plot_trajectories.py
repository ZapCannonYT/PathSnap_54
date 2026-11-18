"""Static GPS-trajectory and road-network plots for the report.

For each selected trip, draws the road network near the track, the raw GPS
observations and the reference route on lon/lat axes (aspect corrected for
latitude). Also writes a per-partition missing-value / data-gap summary.

Outputs:
  results/eda/trajectory_<partition>_<trip>.png
  results/eda/missing_values.csv
Usage:
  python scripts/plot_trajectories.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pathsnap.io import DATA_DIR, load_trip  # noqa: E402

OUT = os.path.join(ROOT, "results", "eda")
TRIPS = [("train", "00000001"), ("student_test", "00000052"), ("student_test", "00000077")]
PAD = 0.15  # fraction of the track extent added around it


def plot_trip(partition: str, trip_id: str) -> str:
    trip = load_trip(os.path.join(DATA_DIR, partition, trip_id))
    lon, lat = trip.track[:, 0], trip.track[:, 1]
    dx, dy = np.ptp(lon) * PAD + 1e-3, np.ptp(lat) * PAD + 1e-3
    box = (lon.min() - dx, lon.max() + dx, lat.min() - dy, lat.max() + dy)

    a, b = trip.nodes[trip.arcs[:, 0]], trip.nodes[trip.arcs[:, 1]]
    inside = ((a[:, 0] > box[0]) & (a[:, 0] < box[1]) & (a[:, 1] > box[2]) & (a[:, 1] < box[3])) | \
             ((b[:, 0] > box[0]) & (b[:, 0] < box[1]) & (b[:, 1] > box[2]) & (b[:, 1] < box[3]))
    segs = np.stack([a[inside], b[inside]], axis=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.add_collection(LineCollection(segs, colors="#9aa3ad", linewidths=0.6, label="Road network (near track)"))
    if trip.route is not None and len(trip.route):
        r = trip.arcs[trip.route]
        rsegs = np.stack([trip.nodes[r[:, 0]], trip.nodes[r[:, 1]]], axis=1)
        ax.add_collection(LineCollection(rsegs, colors="#e8780c", linewidths=2.4, label="Reference route"))
    ax.scatter(lon, lat, s=4, c="#1f5fbf", label="GPS observations", zorder=3)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{partition} trip {trip_id}: {len(lon)} GPS points, {int(inside.sum())} arcs shown")
    ax.legend(loc="best", fontsize=8)
    path = os.path.join(OUT, f"trajectory_{partition}_{trip_id}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def missing_summary() -> pd.DataFrame:
    inv = pd.read_csv(os.path.join(OUT, "inventory.csv"), dtype={"trip_id": str})
    files = ["track_present", "nodes_present", "arcs_present", "route_present"]
    rows = []
    for part, g in inv.groupby("partition"):
        rows.append({
            "partition": part,
            "trips": len(g),
            "trips_missing_any_file": int((~g[files].all(axis=1)).sum()),
            "gps_rows_raw": int(g.gps_rows_raw.sum()),
            "invalid_or_duplicate_rows": int(g.invalid_or_duplicate_rows.sum()),
            "trips_with_gap_over_120s": int((g.max_gap_s > 120).sum()),
            "gaps_over_10s_total": int(g.gaps_over_10s.sum()),
            "max_gap_s": float(g.max_gap_s.max()),
            "unmatched_point_share": float(1 - (g.matched_share * g.gps_rows_clean).sum() / g.gps_rows_clean.sum()),
            "isolated_nodes_total": int(g.isolated_nodes.sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for part, tid in TRIPS:
        print("wrote", plot_trip(part, tid))
    ms = missing_summary()
    ms.to_csv(os.path.join(OUT, "missing_values.csv"), index=False)
    print(ms.to_string(index=False))
