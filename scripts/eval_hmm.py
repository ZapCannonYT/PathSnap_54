"""Tune the HMM decoder and compare classifier-only vs classifier + HMM/Viterbi.

1. Decoder tuning (train data only): hold out one GroupKFold fold of train trips,
   fit the exported model configuration on the other train trips, and grid-search
   beta and tau on the held-out trips (observation Top-1).
2. Comparison on student_test with the exported classifier (fit on all train):
   Top-1, macro F1 of the chosen candidate, route continuity, chain breaks,
   route precision/recall against the reference arc set.

Outputs: results/hmm_grid.csv, results/hmm_comparison.csv, models/hmm_params.json,
results/plots/hmm_vs_classifier.png

Usage: python scripts/eval_hmm.py [--max-tune-trips 12]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import asdict

import joblib
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.candidates import prepare_trip  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.hmm import HMMParams, _Graph, fill_missing, reconstruct_route, viterbi_decode  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.metrics import route_continuity  # noqa: E402
from pathsnap import zoo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")


def classifier_arcs(c: pd.DataFrame, proba: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -1, dtype=np.int64)
    d = c[["point_idx", "arc_id"]].assign(p=proba)
    best = d.loc[d.groupby("point_idx")["p"].idxmax()]
    out[best["point_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return out


def score(arcs, pts, c, net, route_ref, decoded_route):
    m = pts["matched"].to_numpy()
    lab = pts["label_arc"].to_numpy()
    chosen = arcs[c["point_idx"].to_numpy()] == c["arc_id"].to_numpy()
    ref, pred = set(route_ref.tolist()), set(decoded_route)
    return {"top1": float((arcs[m] == lab[m]).mean()),
            "macro_f1": float(f1_score(c["y"].to_numpy(), chosen.astype(int), average="macro")),
            "point_continuity": route_continuity(arcs, net),
            "route_precision": len(ref & pred) / max(len(pred), 1),
            "route_recall": len(ref & pred) / max(len(ref), 1)}


BETAS = [1, 2, 5, 10, 20, 40]
TAUS = [0.5, 1.0, 2.0]
N_JOBS = max(1, (os.cpu_count() or 2) - 1)


def _grid_one_trip(trip_dir, c, proba, pts, settings):
    """Top-1 of one held-out train trip for every (beta, tau) setting."""
    trip = load_trip(trip_dir)
    _, net, xy = prepare_trip(trip)
    g = _Graph(net)
    mk = pts["matched"].to_numpy()
    lab = pts["label_arc"].to_numpy()
    out = []
    for beta, tau in settings:
        arcs, _ = viterbi_decode(net, xy, trip.track[:, 2], c, proba, HMMParams(beta=beta, tau=tau), g)
        arcs = fill_missing(arcs)
        out.append(float((arcs[mk] == lab[mk]).mean()))
    return out


def _compare_one_trip(trip_dir, c, proba, pts, params):
    """Classifier-only vs classifier + HMM rows for one student_test trip."""
    trip = load_trip(trip_dir)
    _, net, xy = prepare_trip(trip)
    g = _Graph(net)
    clf = fill_missing(classifier_arcs(c, proba, len(xy)))
    hmm, breaks = viterbi_decode(net, xy, trip.track[:, 2], c, proba, params, g)
    hmm = fill_missing(hmm)
    rows = []
    for name, arcs, br in [("classifier_only", clf, 0), ("classifier_hmm", hmm, breaks)]:
        r = score(arcs, pts, c, net, trip.route, reconstruct_route(net, arcs, g))
        rows.append({"trip_id": trip.trip_id, "method": name, "points": len(xy), "breaks": br, **r})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tune-trips", type=int, default=12)
    args = ap.parse_args()
    card = json.load(open(os.path.join(MODELS, "model_card.json")))

    # 1. Decoder tuning on held-out train trips.
    tr_c = pd.read_parquet(os.path.join(PROC, "train_cands.parquet"))
    tr_p = pd.read_parquet(os.path.join(PROC, "train_points.parquet"))
    trips = np.array(sorted(tr_c["trip_id"].unique()))
    fit_idx, hold_idx = next(GroupKFold(n_splits=5).split(trips, groups=trips))
    hold = trips[hold_idx][: args.max_tune_trips]
    fit_rows = tr_c[tr_c["trip_id"].isin(trips[fit_idx]) & tr_c["matched"] & (tr_c["point_idx"] % 4 == 0)]
    m_fold = zoo.get(card["model"]).build(card["params"]).fit(fit_rows[FEATURES].to_numpy(), fit_rows["y"].to_numpy())

    # Each held-out trip is decoded for every (beta, tau) setting in its own process.
    settings = list(itertools.product(BETAS, TAUS))
    jobs = []
    for tid in hold:
        c = tr_c[tr_c["trip_id"] == tid].reset_index(drop=True)
        jobs.append(delayed(_grid_one_trip)(
            os.path.join(ROOT, "data", "raw", "train", tid), c,
            m_fold.predict_proba(c[FEATURES].to_numpy())[:, 1],
            tr_p[tr_p["trip_id"] == tid].reset_index(drop=True), settings))
    per_trip = list(tqdm(Parallel(n_jobs=N_JOBS, return_as="generator")(jobs), total=len(jobs),
                         desc="HMM grid on held-out train trips", unit="trip"))
    acc = np.mean(np.array(per_trip), axis=0)
    grid = pd.DataFrame([{"beta": b, "tau": t, "heldout_train_top1": float(a)}
                         for (b, t), a in zip(settings, acc)])
    print(grid.to_string(index=False))
    grid.to_csv(os.path.join(RES, "hmm_grid.csv"), index=False)
    best = grid.loc[grid["heldout_train_top1"].idxmax()]
    params = HMMParams(beta=float(best["beta"]), tau=float(best["tau"]))
    json.dump(asdict(params), open(os.path.join(MODELS, "hmm_params.json"), "w"), indent=2)
    print("chosen", params, flush=True)

    # 2. Classifier-only vs HMM on student_test with the exported model.
    model = joblib.load(os.path.join(MODELS, "classifier.joblib"))
    te_c = pd.read_parquet(os.path.join(PROC, "student_test_cands.parquet"))
    te_p = pd.read_parquet(os.path.join(PROC, "student_test_points.parquet"))
    jobs = []
    for d in list_trips("student_test"):
        tid = os.path.basename(d)
        c = te_c[te_c["trip_id"] == tid].reset_index(drop=True)
        jobs.append(delayed(_compare_one_trip)(
            d, c, model.predict_proba(c[FEATURES].to_numpy())[:, 1],
            te_p[te_p["trip_id"] == tid].reset_index(drop=True), params))
    rows = []
    for r in tqdm(Parallel(n_jobs=N_JOBS, return_as="generator")(jobs), total=len(jobs),
                  desc="student_test HMM vs classifier", unit="trip"):
        rows += r
        tqdm.write("%s clf %.3f hmm %.3f" % (r[0]["trip_id"], r[0]["top1"], r[1]["top1"]))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "hmm_comparison.csv"), index=False)
    summary = res.groupby("method")[["top1", "macro_f1", "point_continuity", "route_precision",
                                     "route_recall", "breaks"]].mean()
    summary.to_csv(os.path.join(RES, "hmm_summary.csv"))
    print(summary)

    piv = res.pivot(index="trip_id", columns="method", values="top1")
    piv.plot.bar(figsize=(9, 4))
    plt.ylabel("observation Top-1"); plt.title("Classifier-only vs classifier + HMM (student_test)")
    plt.savefig(os.path.join(RES, "plots", "hmm_vs_classifier.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
