"""PathSnap evaluator UI (runs on laptop 2).

Run:  streamlit run ui/app.py
Set the API address (laptop 1) in the sidebar, e.g. http://192.168.1.20:8000.
The UI only sends a trip id; all prediction and scoring happen in the API.
"""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - map-matching and route reconstruction")

api = st.sidebar.text_input("API address", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")
use_hmm = st.sidebar.checkbox("Use HMM / Viterbi decoding", value=True)

try:
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
except Exception as exc:  # network errors are shown, not raised
    st.error(f"Cannot reach the API at {api}: {exc}")
    st.stop()

if not trips:
    st.warning("The API has no evaluation trips loaded.")
    st.stop()

trip_id = st.selectbox("Evaluation trip", trips)
if st.button("Reconstruct route", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API..."):
        r = requests.post(f"{api}/predict/{trip_id}", params={"use_hmm": use_hmm}, timeout=300)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    cols = st.columns(5)
    if "observation_top1_accuracy" in m:
        cols[0].metric("Top-1 accuracy", f"{100 * m['observation_top1_accuracy']:.1f}%")
        cols[1].metric("Classifier only", f"{100 * m['classifier_only_top1_accuracy']:.1f}%")
        cols[2].metric("Route recall", f"{100 * m['route_recall']:.1f}%")
    cols[3].metric("Route continuity", f"{100 * m['route_continuity']:.1f}%")
    cols[4].metric("API time", f"{m['seconds']:.2f} s")
    st.caption(f"Trip {res['trip_id']}: {m['points']} GPS points, {m['route_arcs']} arcs in the "
               f"reconstructed route, {m['chain_breaks']} chain breaks.")

    gps = res["gps"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=15, tiles="OpenStreetMap")
    net_layer = folium.FeatureGroup(name="Road network (near track)")
    for seg in res["road_network"]:
        folium.PolyLine(seg, color="#9e9e9e", weight=2, opacity=0.6).add_to(net_layer)
    gps_layer = folium.FeatureGroup(name="GPS observations")
    for p in gps:
        folium.CircleMarker(p, radius=2, color="#1f5fbf", fill=True, fill_opacity=0.8).add_to(gps_layer)
    route_layer = folium.FeatureGroup(name="Predicted route")
    for seg in res["predicted_route"]:
        folium.PolyLine(seg, color="#d62728", weight=5, opacity=0.85).add_to(route_layer)
    for layer in (net_layer, gps_layer, route_layer):
        layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)],
                     [max(p[0] for p in gps), max(p[1] for p in gps)]])
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;"
              "padding:6px 10px;border:1px solid #888;font-size:13px'>"
              "<b>Legend</b><br><span style='color:#1f5fbf'>&#9679;</span> GPS points<br>"
              "<span style='color:#d62728'>&#9473;</span> Predicted route<br>"
              "<span style='color:#9e9e9e'>&#9473;</span> Road network</div>")
    fmap.get_root().html.add_child(folium.Element(legend))
    components.html(fmap._repr_html_(), height=650)
