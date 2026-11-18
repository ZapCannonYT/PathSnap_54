"""Metrics shared by model comparison, tuning, HMM evaluation and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)


def top1_choice(cands: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, point)."""
    df = cands[["trip_id", "point_idx", "arc_id"]].copy()
    df["score"] = score
    idx = df.groupby(["trip_id", "point_idx"])["score"].idxmax()
    return df.loc[idx, ["trip_id", "point_idx", "arc_id", "score"]].reset_index(drop=True)


def top1_accuracy(choice: pd.DataFrame, points: pd.DataFrame) -> float:
    """Share of matched points whose chosen arc equals the label.
    Points with no candidate count as wrong."""
    m = points[points["matched"]][["trip_id", "point_idx", "label_arc"]]
    j = m.merge(choice, on=["trip_id", "point_idx"], how="left")
    return float((j["arc_id"] == j["label_arc"]).mean())


def route_continuity(pred_arcs: np.ndarray, net) -> float:
    """Share of consecutive distinct predicted arcs that are connected in the network."""
    a = pred_arcs[pred_arcs >= 0]
    if len(a) < 2:
        return 1.0
    keep = np.r_[True, a[1:] != a[:-1]]
    a = a[keep]
    if len(a) < 2:
        return 1.0
    return float(np.mean(net.arcs[a[:-1], 1] == net.arcs[a[1:], 0]))


def binary_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
