"""Train and evaluate one or more candidate classifiers from the model zoo.

Protocol (same for every model, so results are comparable)
  - Train trips only: every 4th matched GPS point with all its candidates
    (models with MAX_ROWS use a further sample of whole points).
  - GroupKFold(5) grouped by trip on the train trips (folds run in parallel).
  - Final fit on all those rows, evaluation on the 10 student_test trips.
  - Evaluator trips are never read.

Outputs per model in results/models/: <name>.json (metrics + ROC curve points),
<name>_folds.csv; plots in results/plots/: cm_<name>.png, and
<name>_training_history.png for models with epoch-wise loss (MLP).

Usage: python scripts/train_model.py --models decision_tree,random_forest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_curve  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap import zoo  # noqa: E402
from pathsnap.candidates import prepare_trip  # noqa: E402
from pathsnap.evaluation import cv_folds, training_rows  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.metrics import binary_metrics, route_continuity, top1_accuracy, top1_choice  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES_M = os.path.join(ROOT, "results", "models")
PLOTS = os.path.join(ROOT, "results", "plots")
SAMPLE_EVERY = 4


def evaluate(name, train_all, test, test_pts, nets):
    spec = zoo.get(name)
    data = training_rows(train_all, SAMPLE_EVERY, spec.MAX_ROWS)
    X, y = data[FEATURES].to_numpy(), data["y"].to_numpy()

    folds = cv_folds(name, None, data, n_splits=5)
    t0 = time.time()
    m = spec.build().fit(X, y)
    fit_s = time.time() - t0
    train_f1 = f1_score(y, (m.predict_proba(X)[:, 1] >= 0.5).astype(int), average="macro")

    t0 = time.time()
    p = m.predict_proba(test[FEATURES].to_numpy())[:, 1]
    infer_s = time.time() - t0
    bm = binary_metrics(test["y"].to_numpy(), p)
    ch = top1_choice(test, p)
    cont = float(np.mean([route_continuity(ch[ch.trip_id == t].sort_values("point_idx")["arc_id"].to_numpy(), n)
                          for t, n in nets.items()]))
    fpr, tpr, _ = roc_curve(test["y"].to_numpy(), p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))

    row = {"model": name, "name": spec.NAME_TEXT, "family": spec.FAMILY, "config": spec.CONFIG,
           "train_rows": len(data), "train_macro_f1": float(train_f1),
           "cv_macro_f1_mean": float(np.mean([f[0] for f in folds])),
           "cv_macro_f1_std": float(np.std([f[0] for f in folds])),
           "cv_top1_mean": float(np.mean([f[1] for f in folds])),
           "test_top1": top1_accuracy(ch, test_pts), "test_accuracy": bm["accuracy"],
           "test_macro_f1": bm["macro_f1"], "test_precision": bm["precision"],
           "test_recall_sensitivity": bm["recall_sensitivity"], "test_specificity": bm["specificity"],
           "test_auc": bm["auc"], "test_log_loss": bm["log_loss"], "route_continuity": cont,
           "fit_seconds": fit_s, "inference_ms_per_point": 1000 * infer_s / len(test_pts),
           "tn": bm["tn"], "fp": bm["fp"], "fn": bm["fn"], "tp": bm["tp"]}
    with open(os.path.join(RES_M, f"{name}.json"), "w") as f:
        json.dump({**row, "roc_fpr": fpr[keep].tolist(), "roc_tpr": tpr[keep].tolist()}, f, indent=2)
    pd.DataFrame([{"model": name, "fold": k, "macro_f1": a, "top1": b} for k, (a, b) in enumerate(folds)]) \
        .to_csv(os.path.join(RES_M, f"{name}_folds.csv"), index=False)

    ConfusionMatrixDisplay(np.array([[bm["tn"], bm["fp"]], [bm["fn"], bm["tp"]]]),
                           display_labels=["not match", "match"]).plot(values_format="d")
    plt.title(f"{spec.NAME_TEXT} - student_test confusion matrix")
    plt.savefig(os.path.join(PLOTS, f"cm_{name}.png"), dpi=120, bbox_inches="tight")
    plt.close("all")

    est = m.steps[-1][1] if hasattr(m, "steps") else m
    if hasattr(est, "loss_curve_"):
        plt.figure()
        plt.plot(est.loss_curve_, label="training loss")
        if getattr(est, "validation_scores_", None) is not None:
            plt.plot(1 - np.array(est.validation_scores_), label="1 - validation accuracy")
        plt.xlabel("epoch"); plt.legend(); plt.title(f"{spec.NAME_TEXT} training history")
        plt.savefig(os.path.join(PLOTS, f"{name}_training_history.png"), dpi=120, bbox_inches="tight")
        plt.close("all")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated names from pathsnap/zoo, or 'all'")
    args = ap.parse_args()
    names = zoo.names() if args.models == "all" else args.models.split(",")
    os.makedirs(RES_M, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)

    train_all = pd.read_parquet(os.path.join(PROC, "train_cands.parquet"))
    test = pd.read_parquet(os.path.join(PROC, "student_test_cands.parquet"))
    test_pts = pd.read_parquet(os.path.join(PROC, "student_test_points.parquet"))
    nets = {}
    for d in tqdm(list_trips("student_test"), desc="loading test networks", unit="trip"):
        trip = load_trip(d)
        nets[trip.trip_id] = prepare_trip(trip)[1]

    for name in tqdm(names, desc="models", unit="model"):
        row = evaluate(name, train_all, test, test_pts, nets)
        tqdm.write(f"{name}: CV macro-F1 {row['cv_macro_f1_mean']:.4f}, CV Top-1 {row['cv_top1_mean']:.4f}, "
                   f"test Top-1 {row['test_top1']:.4f}, test macro-F1 {row['test_macro_f1']:.4f}, "
                   f"log loss {row['test_log_loss']:.4f}, fit {row['fit_seconds']:.1f}s")


if __name__ == "__main__":
    main()
