"""Collect every trained model's results into one comparison and select the final model.

Reads results/models/<name>.json written by scripts/train_model.py, adds the
non-learned nearest-arc reference, and writes:
  results/model_comparison.csv, results/cv_folds.csv, results/selected_model.json
  results/plots/compare_*.png, results/plots/roc_all_models.png

Selection rule: highest student_test observation Top-1 (closest to the route
objective); ties broken by lower log loss.

Usage: python scripts/compare_models.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.candidates import prepare_trip  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.metrics import route_continuity, top1_accuracy, top1_choice  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots")


def main():
    files = sorted(glob.glob(os.path.join(RES, "models", "*.json")))
    if not files:
        sys.exit("no results/models/*.json yet - run scripts/train_model.py first")
    runs = [json.load(open(f)) for f in files]

    # Non-learned reference: always take the nearest arc.
    test = pd.read_parquet(os.path.join(PROC, "student_test_cands.parquet"))
    test_pts = pd.read_parquet(os.path.join(PROC, "student_test_points.parquet"))
    ch = top1_choice(test, -test["d_perp"].to_numpy())
    cont = []
    for d in list_trips("student_test"):
        trip = load_trip(d)
        net = prepare_trip(trip)[1]
        cont.append(route_continuity(ch[ch.trip_id == trip.trip_id].sort_values("point_idx")["arc_id"].to_numpy(), net))
    base = {"model": "nearest_arc_baseline", "name": "Nearest arc (reference, not learned)",
            "family": "Rule", "config": "argmin distance", "test_top1": top1_accuracy(ch, test_pts),
            "route_continuity": float(np.mean(cont))}

    rows = [{k: v for k, v in r.items() if not k.startswith("roc_")} for r in runs]
    res = pd.DataFrame([base] + rows)
    res.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "models", "*_folds.csv")))]) \
        .to_csv(os.path.join(RES, "cv_folds.csv"), index=False)

    learned = res[res["model"] != "nearest_arc_baseline"].set_index("model")
    for col, title in [("cv_macro_f1_mean", "Cross-validation macro F1 (train, GroupKFold by trip)"),
                       ("test_top1", "Student-test observation Top-1 accuracy"),
                       ("test_macro_f1", "Student-test macro F1"),
                       ("test_log_loss", "Student-test log loss (lower is better)")]:
        plt.figure(figsize=(8, 4))
        err = learned["cv_macro_f1_std"] if col == "cv_macro_f1_mean" else None
        learned[col].plot.bar(yerr=err)
        if col == "test_top1":
            plt.axhline(base["test_top1"], color="grey", ls=":", label="nearest-arc reference")
            plt.legend()
        plt.title(title); plt.ylabel(col); plt.xticks(rotation=30, ha="right")
        plt.savefig(os.path.join(PLOTS, f"compare_{col}.png"), dpi=120, bbox_inches="tight")
        plt.close("all")

    plt.figure(figsize=(6, 6))
    for r in runs:
        plt.plot(r["roc_fpr"], r["roc_tpr"], label=f"{r['model']} (AUC {r['test_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("false positive rate (1 - specificity)"); plt.ylabel("true positive rate (sensitivity)")
    plt.title("ROC on student_test candidates"); plt.legend(fontsize=8)
    plt.savefig(os.path.join(PLOTS, "roc_all_models.png"), dpi=120, bbox_inches="tight")
    plt.close("all")

    ranked = learned.sort_values(["test_top1", "test_log_loss"], ascending=[False, True])
    best = ranked.index[0]
    with open(os.path.join(RES, "selected_model.json"), "w") as f:
        json.dump({"selected": best, "criterion": "student_test Top-1, tie-break log loss",
                   "test_top1": float(ranked.iloc[0]["test_top1"]),
                   "models_compared": list(learned.index)}, f, indent=2)
    pd.set_option("display.width", 200)
    print(res[["model", "cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_macro_f1",
               "test_log_loss", "route_continuity"]].round(4).to_string(index=False))
    print("selected:", best)


if __name__ == "__main__":
    main()
