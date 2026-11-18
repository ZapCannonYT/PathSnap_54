"""Build the candidate tables for the train and student_test partitions.

Outputs (data/processed/, not committed):
  <part>_cands.parquet   one row per (point, candidate arc) with features and label y
  <part>_points.parquet  one row per GPS point with the route-derived label
Also writes results/candidate_stats.csv (candidate miss rate per trip and radius).

Usage: python scripts/build_dataset.py [--radius 50] [--k 12]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.candidates import attach_labels, prepare_trip, trip_candidates, trip_points  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    stats = []
    for part in ["train", "student_test"]:
        all_c, all_p = [], []
        for d in tqdm(list_trips(part), desc=f"{part} trips", unit="trip"):
            t0 = time.time()
            trip = load_trip(d)
            _, net, xy = prepare_trip(trip)
            pts = trip_points(trip, net, xy)
            cands = attach_labels(trip_candidates(trip, net, xy, args.radius, args.k), pts)
            all_c.append(cands)
            all_p.append(pts)

            # Candidate miss rate at several radii (justifies the chosen radius).
            row = {"partition": part, "trip_id": trip.trip_id, "points": len(xy),
                   "matched_share": float(pts["matched"].mean())}
            matched_pts = pts.loc[pts["matched"], "point_idx"].to_numpy()
            for r in (30, 50, 80):
                c = cands[cands["d_perp"] <= r] if r <= args.radius else \
                    attach_labels(trip_candidates(trip, net, xy, r, args.k), pts)
                hit = np.zeros(len(xy), bool)
                hit[c.loc[c["y"] == 1, "point_idx"].to_numpy()] = True
                row[f"miss_rate_r{r}"] = float(1 - hit[matched_pts].mean()) if len(matched_pts) else 0.0
            row["mean_candidates"] = float(cands.groupby("point_idx").size().mean())
            stats.append(row)
            tqdm.write(f"{part} {trip.trip_id}: {len(xy)} pts, {len(cands)} cands, "
                  f"miss@{int(args.radius)}={row.get(f'miss_rate_r{int(args.radius)}', float('nan')):.4f} "
                  f"({time.time() - t0:.1f}s)")
        pd.concat(all_c, ignore_index=True).to_parquet(os.path.join(OUT, f"{part}_cands.parquet"))
        pd.concat(all_p, ignore_index=True).to_parquet(os.path.join(OUT, f"{part}_points.parquet"))

    s = pd.DataFrame(stats)
    s.to_csv(os.path.join(RES, "candidate_stats.csv"), index=False)
    print(s.groupby("partition")[["matched_share", "miss_rate_r30", "miss_rate_r50", "miss_rate_r80",
                                  "mean_candidates"]].mean())


if __name__ == "__main__":
    main()
