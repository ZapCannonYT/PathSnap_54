"""Bayesian hyperparameter optimisation of the selected classifier.

Sampler: Optuna TPE (sequential model-based Bayesian optimisation), seed 42.
Objective: mean observation-level Top-1 over GroupKFold(3) on the train trips
(grouped by trip). The student_test and evaluator trips are not used to choose
anything here. A random-search run with the same budget is the baseline.

Outputs: results/optuna_trials_tpe.csv, results/optuna_trials_random.csv,
results/best_params.json, results/plots/optimisation_history.png

Usage: python scripts/tune_bayes.py [--model random_forest] [--trials 25] [--sample-every 8]
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
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.evaluation import cv_scores, training_rows  # noqa: E402
from pathsnap import zoo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")


def cv_top1(name, params, data):
    return cv_scores(name, params, data, n_splits=3)[1]


def run(name, data, sampler, trials):
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(lambda tr: cv_top1(name, zoo.get(name).search_space(tr), data), n_trials=trials, show_progress_bar=True)
    return study, time.time() - t0


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    sel = os.path.join(RES, "selected_model.json")
    default = json.load(open(sel))["selected"] if os.path.exists(sel) else "random_forest"
    ap.add_argument("--model", default=default)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--random-trials", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(os.path.join(RES, "plots"), exist_ok=True)

    cands = pd.read_parquet(os.path.join(PROC, "train_cands.parquet"))
    data = training_rows(cands, args.sample_every, zoo.get(args.model).MAX_ROWS)
    print(f"tuning {args.model} on {len(data)} rows", flush=True)

    untuned = cv_top1(args.model, {}, data)
    tpe, tpe_s = run(args.model, data, optuna.samplers.TPESampler(seed=42), args.trials)
    rnd, rnd_s = run(args.model, data, optuna.samplers.RandomSampler(seed=42), args.random_trials)

    tpe.trials_dataframe().to_csv(os.path.join(RES, "optuna_trials_tpe.csv"), index=False)
    rnd.trials_dataframe().to_csv(os.path.join(RES, "optuna_trials_random.csv"), index=False)
    out = {"model": args.model, "objective": "GroupKFold(3) observation Top-1 on train trips",
           "untuned_cv_top1": untuned, "best_cv_top1": tpe.best_value, "best_params": tpe.best_params,
           "tpe_trials": args.trials, "tpe_seconds": tpe_s,
           "random_trials": args.random_trials, "random_best_cv_top1": rnd.best_value, "random_seconds": rnd_s,
           "train_rows": len(data)}
    json.dump(out, open(os.path.join(RES, "best_params.json"), "w"), indent=2)

    plt.figure(figsize=(7, 4))
    v = [t.value for t in tpe.trials]
    plt.plot(v, "o", alpha=0.5, label="TPE trial")
    plt.plot(np.maximum.accumulate(v), label="TPE best so far")
    rv = [t.value for t in rnd.trials]
    plt.plot(np.maximum.accumulate(rv), "--", label="random search best so far")
    plt.axhline(untuned, color="grey", ls=":", label="untuned")
    plt.xlabel("trial"); plt.ylabel("CV Top-1"); plt.legend(); plt.title(f"Bayesian optimisation - {args.model}")
    plt.savefig(os.path.join(RES, "plots", "optimisation_history.png"), dpi=120, bbox_inches="tight")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
