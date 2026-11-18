"""End-to-end inference for one trip: candidates -> classifier -> HMM/Viterbi.

The route file is never read here; ground truth is only used by the caller
(evaluation scripts, or the API's server-side scoring).
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from .candidates import RADIUS_M, TOP_K, prepare_trip, trip_candidates
from .features import FEATURES
from .hmm import HMMParams, _Graph, fill_missing, reconstruct_route, viterbi_decode
from .io import Trip

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts(models_dir: str = MODELS_DIR):
    model = joblib.load(os.path.join(models_dir, "classifier.joblib"))
    hp_path = os.path.join(models_dir, "hmm_params.json")
    hp = HMMParams(**json.load(open(hp_path))) if os.path.exists(hp_path) else HMMParams()
    return model, hp


def predict_trip(trip: Trip, model, hmm_params: HMMParams, use_hmm: bool = True,
                 radius: float = RADIUS_M, k: int = TOP_K) -> dict:
    t0 = time.time()
    proj, net, xy = prepare_trip(trip)
    cands = trip_candidates(trip, net, xy, radius, k)
    proba = model.predict_proba(cands[FEATURES].to_numpy())[:, 1]

    # Classifier-only: best candidate per point.
    clf = np.full(len(xy), -1, dtype=np.int64)
    order = np.lexsort((proba, cands["point_idx"].to_numpy()))
    last = np.r_[cands["point_idx"].to_numpy()[order][1:] != cands["point_idx"].to_numpy()[order][:-1], True]
    clf[cands["point_idx"].to_numpy()[order][last]] = cands["arc_id"].to_numpy()[order][last]

    breaks = 0
    if use_hmm:
        g = _Graph(net)
        arcs, breaks = viterbi_decode(net, xy, trip.track[:, 2], cands, proba, hmm_params, g)
        arcs = fill_missing(arcs)
        route = reconstruct_route(net, arcs, g)
    else:
        arcs = fill_missing(clf)
        route = reconstruct_route(net, arcs)
    return {"net": net, "proj": proj, "xy": xy, "cands": cands, "proba": proba,
            "classifier_arcs": clf, "point_arcs": arcs, "route": route, "breaks": breaks,
            "seconds": time.time() - t0}
