"""Fit the untuned and the tuned classifier on the train trips, compare them on
student_test with the same metrics, and export the tuned one for the API.

Outputs: results/tuned_vs_untuned.csv, models/classifier.joblib, models/model_card.json

Usage: python scripts/export_model.py [--sample-every 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.candidates import RADIUS_M, TOP_K  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import binary_metrics, top1_accuracy, top1_choice  # noqa: E402
from pathsnap.evaluation import cv_scores, training_rows  # noqa: E402
from pathsnap import zoo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-every", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True)
    best = json.load(open(os.path.join(RES, "best_params.json")))
    name = best["model"]

    cands = pd.read_parquet(os.path.join(PROC, "train_cands.parquet"))
    train = training_rows(cands, args.sample_every, zoo.get(name).MAX_ROWS)
    test = pd.read_parquet(os.path.join(PROC, "student_test_cands.parquet"))
    test_pts = pd.read_parquet(os.path.join(PROC, "student_test_points.parquet"))
    X, y = train[FEATURES].to_numpy(), train["y"].to_numpy()

    rows, fitted = [], {}
    for label, params in [("untuned", {}), ("tuned", best["best_params"])]:
        cv_f1, cv_top1 = cv_scores(name, params, train)
        t0 = time.time()
        m = zoo.get(name).build(params).fit(X, y)
        fit_s = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(test[FEATURES].to_numpy())[:, 1]
        inf_s = time.time() - t0
        bm = binary_metrics(test["y"].to_numpy(), p)
        rows.append({"model": f"{name} ({label})", "params": json.dumps(params) if params else "defaults",
                     "cv_macro_f1": cv_f1, "cv_top1": cv_top1,
                     "train_macro_f1": f1_score(y, (m.predict_proba(X)[:, 1] >= 0.5).astype(int), average="macro"),
                     "test_top1": top1_accuracy(top1_choice(test, p), test_pts),
                     "test_macro_f1": bm["macro_f1"], "test_log_loss": bm["log_loss"], "test_auc": bm["auc"],
                     "fit_seconds": fit_s, "inference_ms_per_point": 1000 * inf_s / len(test_pts)})
        fitted[label] = m
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)

    # Export the tuned model unless tuning made it worse on CV (then keep defaults and say so).
    pick = "tuned" if rows[1]["cv_top1"] >= rows[0]["cv_top1"] - 1e-4 else "untuned"
    joblib.dump(fitted[pick], os.path.join(MODELS, "classifier.joblib"), compress=3)
    card = {"model": name, "variant": pick, "params": best["best_params"] if pick == "tuned" else {},
            "features": FEATURES, "candidate_radius_m": RADIUS_M, "candidate_top_k": TOP_K,
            "trained_on": "train partition, every %d-th matched point" % args.sample_every,
            "student_test_top1": rows[1 if pick == "tuned" else 0]["test_top1"]}
    json.dump(card, open(os.path.join(MODELS, "model_card.json"), "w"), indent=2)
    print("exported", pick, "->", os.path.join(MODELS, "classifier.joblib"))


if __name__ == "__main__":
    main()
