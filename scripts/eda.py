"""Exploratory data analysis of the raw trips (train and student_test only).

Outputs in results/eda/:
  inventory.csv          per trip: files, sizes, row counts, time span, gaps, extent,
                         network size, connectivity, route length and continuity
  partition_summary.csv  aggregated per partition
  data_quality.md        issues, assumptions and decisions found here
  *.png                  plots used in the report

Usage: python scripts/eda.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.geo import LocalProjection, haversine  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.labels import align_to_route  # noqa: E402
from pathsnap.network import Network  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "eda")


def raw_rows(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, all_dt, all_err = [], [], []
    for part in ["train", "student_test"]:
        for d in tqdm(list_trips(part), desc=f"EDA {part}", unit="trip"):
            tid = os.path.basename(d)
            files = {ext: os.path.join(d, f"{tid}.{ext}") for ext in ("track", "nodes", "arcs", "route")}
            trip = load_trip(d)
            tr = trip.track
            dt = np.diff(tr[:, 2])
            step_m = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
            proj = LocalProjection.for_points(tr[:, :2])
            net = Network(trip.nodes, trip.arcs, proj)
            xy = proj.to_xy(tr[:, :2])
            n_comp, comp = connected_components(net.graph, directed=True, connection="weak")
            deg = np.bincount(trip.arcs.ravel(), minlength=len(trip.nodes))
            r = trip.route
            label, ldist, matched = align_to_route(xy, r, net)
            all_dt.append(dt)
            all_err.append(ldist)
            rows.append({
                "partition": part, "trip_id": tid,
                **{f"{e}_present": os.path.exists(p) for e, p in files.items()},
                **{f"{e}_kb": round(os.path.getsize(p) / 1024, 1) for e, p in files.items() if os.path.exists(p)},
                "gps_rows_raw": raw_rows(files["track"]), "gps_rows_clean": len(tr),
                "invalid_or_duplicate_rows": raw_rows(files["track"]) - len(tr),
                "duration_s": float(tr[-1, 2] - tr[0, 2]),
                "share_1s_steps": float(np.mean(dt == 1)) if len(dt) else 1.0,
                "max_gap_s": float(dt.max()) if len(dt) else 0.0,
                "gaps_over_10s": int((dt > 10).sum()),
                "max_jump_m": float(step_m.max()) if len(step_m) else 0.0,
                "track_length_km": float(step_m.sum() / 1000),
                "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(),
                "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
                "nodes": len(trip.nodes), "arcs": len(trip.arcs),
                "two_way_share": float(np.mean(net.twin >= 0)),
                "weak_components": int(n_comp),
                "largest_component_share": float(np.bincount(comp).max() / len(comp)),
                "isolated_nodes": int((deg == 0).sum()),
                "route_arcs": len(r), "route_unique_arcs": len(np.unique(r)),
                "route_continuity": float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if len(r) > 1 else 1.0,
                "median_gps_to_route_m": float(np.median(ldist)),
                "p95_gps_to_route_m": float(np.percentile(ldist, 95)),
                "matched_share": float(matched.mean()),
            })

    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    summ = inv.groupby("partition").agg(
        trips=("trip_id", "count"), gps_rows=("gps_rows_clean", "sum"),
        gps_rows_median=("gps_rows_clean", "median"), max_gap_s=("max_gap_s", "max"),
        share_1s=("share_1s_steps", "mean"), nodes_median=("nodes", "median"), arcs_median=("arcs", "median"),
        route_continuity_min=("route_continuity", "min"), matched_share=("matched_share", "mean"),
        median_gps_err_m=("median_gps_to_route_m", "median"))
    summ.to_csv(os.path.join(OUT, "partition_summary.csv"))
    print(summ.T)

    # Plots
    dt = np.concatenate(all_dt)
    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(dt, 0, 30), bins=30)
    plt.yscale("log"); plt.xlabel("seconds between GPS points (clipped at 30)"); plt.ylabel("count (log)")
    plt.title("Sampling intervals, train + student_test")
    plt.savefig(os.path.join(OUT, "sampling_intervals.png"), dpi=120, bbox_inches="tight")

    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(np.concatenate(all_err), 0, 60), bins=60)
    plt.xlabel("GPS distance to aligned reference arc (m, clipped at 60)"); plt.ylabel("points")
    plt.title("GPS noise relative to the reference route")
    plt.savefig(os.path.join(OUT, "gps_noise.png"), dpi=120, bbox_inches="tight")

    plt.figure(figsize=(8, 4.5))
    for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        s = inv[inv.partition == part]
        plt.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, s=12, c=c, label=part)
    plt.xlabel("longitude"); plt.ylabel("latitude"); plt.legend(); plt.title("Trip locations")
    plt.savefig(os.path.join(OUT, "trip_locations.png"), dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    inv.boxplot(column="gps_rows_clean", by="partition", ax=ax[0])
    inv.boxplot(column="arcs", by="partition", ax=ax[1])
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].set_title("GPS points per trip"); ax[1].set_title("Road arcs per trip")
    plt.suptitle("")
    plt.savefig(os.path.join(OUT, "trip_sizes.png"), dpi=120, bbox_inches="tight")
    plt.close("all")

    worst_gap = inv.loc[inv.max_gap_s.idxmax()]
    with open(os.path.join(OUT, "data_quality.md"), "w") as f:
        f.write("# Data quality notes (generated by scripts/eda.py)\n\n")
        f.write(f"- Trips: {len(inv)} ({(inv.partition == 'train').sum()} train, "
                f"{(inv.partition == 'student_test').sum()} student_test); all four files present: "
                f"{bool(inv[[c for c in inv if c.endswith('_present')]].all().all())}.\n")
        f.write(f"- Invalid or duplicate GPS rows removed: {int(inv.invalid_or_duplicate_rows.sum())}.\n")
        f.write(f"- Largest time gap: {worst_gap.max_gap_s:.0f} s (trip {worst_gap.trip_id}); "
                f"{int((inv.max_gap_s > 120).sum())} trips have a gap over 120 s.\n")
        f.write(f"- Trips spread over lon {inv.lon_min.min():.1f} to {inv.lon_max.max():.1f}, "
                f"lat {inv.lat_min.min():.1f} to {inv.lat_max.max():.1f}: each trip is projected "
                "around its own centroid; node and arc ids are local to a trip.\n")
        f.write(f"- Reference routes are not always connected: minimum continuity "
                f"{inv.route_continuity.min():.3f}; treated as reference, not perfect truth.\n")
        f.write(f"- Two-way share of arcs (reverse twins): median {inv.two_way_share.median():.2f}; "
                "twins share geometry, so heading is needed to tell them apart.\n")
        f.write(f"- Median GPS distance to the aligned reference arc: "
                f"{inv.median_gps_to_route_m.median():.1f} m; share of points within 50 m: "
                f"{inv.matched_share.mean():.3f}.\n")


if __name__ == "__main__":
    main()
