"""Group-aware cross-validation (folds never split a trip), run in parallel."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from . import zoo
from .features import FEATURES
from .metrics import top1_choice

SEED = 42


def training_rows(cands: pd.DataFrame, every: int, max_rows: int | None = None) -> pd.DataFrame:
    """Matched points only, every `every`-th point of each trip (with all its candidates),
    then whole points sampled down to about `max_rows` rows."""
    df = cands[cands["matched"] & (cands["point_idx"] % every == 0)].reset_index(drop=True)
    if max_rows is None or len(df) <= max_rows:
        return df
    keys = df[["trip_id", "point_idx"]].drop_duplicates()
    keep = keys.sample(frac=max_rows / len(df), random_state=SEED)
    return df.merge(keep, on=["trip_id", "point_idx"]).reset_index(drop=True)


def _fold(name, params, X, y, data, tr, va, inner_jobs):
    m = zoo.get(name).build(params)
    inner = {k: inner_jobs for k in m.get_params() if k.endswith("n_jobs")}
    if inner:
        m.set_params(**inner)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[va])[:, 1]
    f1 = f1_score(y[va], (p >= 0.5).astype(int), average="macro")
    va_df = data.iloc[va]
    ch = top1_choice(va_df, p)
    truth = va_df[va_df["y"] == 1][["trip_id", "point_idx", "arc_id"]].rename(columns={"arc_id": "label"})
    j = va_df[["trip_id", "point_idx"]].drop_duplicates().merge(
        truth, on=["trip_id", "point_idx"], how="left").merge(ch, on=["trip_id", "point_idx"])
    return float(f1), float((j["arc_id"] == j["label"]).mean())


def cv_folds(name: str, params: dict | None, data: pd.DataFrame, n_splits: int = 5) -> list:
    """[(macro_f1, top1)] per GroupKFold fold. Folds run in parallel processes;
    multi-threaded models share the remaining cores."""
    X, y, groups = data[FEATURES].to_numpy(), data["y"].to_numpy(), data["trip_id"].to_numpy()
    n_jobs = min(n_splits, os.cpu_count() or 1)
    inner = max(1, (os.cpu_count() or 1) // n_jobs)
    return Parallel(n_jobs=n_jobs)(
        delayed(_fold)(name, params, X, y, data, tr, va, inner)
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups))


def cv_scores(name: str, params: dict | None, data: pd.DataFrame, n_splits: int = 3) -> tuple[float, float]:
    """Mean macro F1 (candidate level) and mean Top-1 (observation level)."""
    res = cv_folds(name, params, data, n_splits)
    return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))
