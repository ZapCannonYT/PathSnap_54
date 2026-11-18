"""PathSnap inference API (runs on laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR  folder holding the evaluator trip folders (default data/evaluator)
      PATHSNAP_MODELS    folder with classifier.joblib and hmm_params.json (default models)

Endpoints
  GET  /health              liveness and loaded model info
  GET  /trips               ids of the evaluation trips available on this server
  POST /predict/{trip_id}   full pipeline on the trip; the reference route is read only
                            after prediction, on the server, to compute accuracy. It is
                            never returned to the client.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pathsnap.geo import lonlat_to_latlon  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.labels import align_to_route  # noqa: E402
from pathsnap.metrics import route_continuity  # noqa: E402
from pathsnap.pipeline import load_artifacts, predict_trip  # noqa: E402

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "evaluator"))
MODELS_DIR = os.environ.get("PATHSNAP_MODELS", os.path.join(ROOT, "models"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL, HMM = load_artifacts(MODELS_DIR)
log.info("model loaded from %s, HMM params %s", MODELS_DIR, HMM)


def _trip_ids() -> list[str]:
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(d for d in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, d)))


@app.get("/health")
def health():
    return {"status": "ok", "model": type(MODEL).__name__, "hmm": HMM.__dict__, "trips": len(_trip_ids())}


@app.get("/trips")
def trips():
    ids = _trip_ids()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


def _arc_lines(net, proj, arcs) -> list:
    arcs = np.asarray(arcs, dtype=np.int64)
    if len(arcs) == 0:
        return []
    a = proj.to_lonlat(net.A[arcs])
    b = proj.to_lonlat(net.B[arcs])
    return [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(a, b)]


@app.post("/predict/{trip_id}")
def predict(trip_id: str, use_hmm: bool = True):
    if trip_id not in _trip_ids():
        raise HTTPException(status_code=404, detail=f"unknown trip {trip_id}")
    t0 = time.time()
    log.info("POST /predict/%s received (use_hmm=%s)", trip_id, use_hmm)
    trip = load_trip(os.path.join(EVAL_DIR, trip_id), with_route=False)
    out = predict_trip(trip, MODEL, HMM, use_hmm=use_hmm)
    net, proj, xy = out["net"], out["proj"], out["xy"]

    # Server-side scoring: the reference route is loaded only now, after prediction.
    metrics = {}
    ref = load_trip(os.path.join(EVAL_DIR, trip_id), with_route=True).route
    if ref is not None:
        label, _, matched = align_to_route(xy, ref, net)
        arcs = out["point_arcs"]
        pred_set, ref_set = set(out["route"]), set(ref.tolist())
        metrics = {
            "observation_top1_accuracy": float((arcs[matched] == label[matched]).mean()),
            "classifier_only_top1_accuracy": float((out["classifier_arcs"][matched] == label[matched]).mean()),
            "route_precision": len(pred_set & ref_set) / max(len(pred_set), 1),
            "route_recall": len(pred_set & ref_set) / max(len(ref_set), 1),
            "matched_points": int(matched.sum()),
        }
    metrics.update({"points": int(len(xy)), "route_arcs": len(out["route"]),
                    "route_continuity": route_continuity(out["point_arcs"], net),
                    "reconstructed_route_continuity": route_continuity(np.asarray(out["route"]), net),
                    "chain_breaks": int(out["breaks"]), "seconds": round(time.time() - t0, 3)})
    log.info("POST /predict/%s done: %s", trip_id, metrics)

    step = max(1, len(xy) // 1500)          # keep the payload light for the map
    nearby = np.unique(out["cands"]["arc_id"].to_numpy())
    return {
        "trip_id": trip_id,
        "metrics": metrics,
        "gps": lonlat_to_latlon(trip.track[::step, :2]),
        "predicted_route": _arc_lines(net, proj, out["route"]),
        "road_network": _arc_lines(net, proj, nearby[net.twin[nearby] < nearby] if len(nearby) else nearby),
    }
