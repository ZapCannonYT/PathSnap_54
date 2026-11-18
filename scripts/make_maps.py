"""Interactive Folium maps of selected trips.

Layers: raw GPS observations, nearby road network, reference route and (when a
trained model exists in models/) the predicted route. Saved to maps/<partition>_<trip>.html.
These are development maps on train/student_test trips; the reference route is
shown here only, never in the evaluator UI.

Usage: python scripts/make_maps.py [--partition student_test] [--trips 00000007,00000052]
"""
from __future__ import annotations

import argparse
import os
import sys

import folium
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.geo import LocalProjection, lonlat_to_latlon  # noqa: E402
from pathsnap.io import DATA_DIR, load_trip  # noqa: E402
from pathsnap.network import Network  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "maps")


def arc_lines(net, proj, arcs):
    a, b = proj.to_lonlat(net.A[arcs]), proj.to_lonlat(net.B[arcs])
    return [[[p[1], p[0]], [q[1], q[0]]] for p, q in zip(a, b)]


def make_map(partition: str, trip_id: str, with_prediction: bool = True) -> str:
    trip = load_trip(os.path.join(DATA_DIR, partition, trip_id))
    proj = LocalProjection.for_points(trip.track[:, :2])
    net = Network(trip.nodes, trip.arcs, proj)
    xy = proj.to_xy(trip.track[:, :2])
    gps = lonlat_to_latlon(trip.track[:, :2])
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=15, tiles="OpenStreetMap")

    # Road network within 150 m of the track.
    pi, arc, _, _ = net.candidates(xy, radius=150, k=40)
    near = np.unique(arc)
    near = near[(net.twin[near] < 0) | (near < net.twin[near])]
    layer = folium.FeatureGroup(name="Road network (150 m around track)")
    for seg in arc_lines(net, proj, near):
        folium.PolyLine(seg, color="#9e9e9e", weight=2, opacity=0.6).add_to(layer)
    layer.add_to(fmap)

    layer = folium.FeatureGroup(name="Reference route")
    for seg in arc_lines(net, proj, trip.route):
        folium.PolyLine(seg, color="#2ca02c", weight=6, opacity=0.7).add_to(layer)
    layer.add_to(fmap)

    if with_prediction and os.path.exists(os.path.join(ROOT, "models", "classifier.joblib")):
        from pathsnap.pipeline import load_artifacts, predict_trip
        model, hp = load_artifacts()
        out = predict_trip(trip, model, hp)
        layer = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
        for seg in arc_lines(net, proj, out["route"]):
            folium.PolyLine(seg, color="#d62728", weight=3, opacity=0.9, dash_array="6").add_to(layer)
        layer.add_to(fmap)

    layer = folium.FeatureGroup(name="GPS observations")
    for i, p in enumerate(gps):
        folium.CircleMarker(p, radius=2, color="#1f5fbf", fill=True, fill_opacity=0.8,
                            tooltip=f"t={trip.track[i, 2]:.0f}s").add_to(layer)
    layer.add_to(fmap)
    folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red")).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    lat = [p[0] for p in gps]
    lon = [p[1] for p in gps]
    fmap.fit_bounds([[min(lat), min(lon)], [max(lat), max(lon)]])
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;"
              "padding:6px 10px;border:1px solid #888;font-size:13px'><b>Trip %s</b><br>"
              "<span style='color:#1f5fbf'>&#9679;</span> GPS observations<br>"
              "<span style='color:#2ca02c'>&#9473;</span> Reference route<br>"
              "<span style='color:#d62728'>&#9473;</span> Predicted route<br>"
              "<span style='color:#9e9e9e'>&#9473;</span> Road network</div>") % trip_id
    fmap.get_root().html.add_child(folium.Element(legend))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{partition}_{trip_id}.html")
    fmap.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trips", default="00000007,00000052,00000077")
    ap.add_argument("--no-prediction", action="store_true")
    args = ap.parse_args()
    for t in args.trips.split(","):
        print("saved", make_map(args.partition, t, not args.no_prediction))


if __name__ == "__main__":
    main()
