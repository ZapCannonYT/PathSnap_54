# Two-laptop deployment checklist

## Laptop 1 (API)
1. git pull; pip install -r requirements.txt
2. Copy the 10 evaluator trip folders into data/evaluator/<trip_id>/ (never committed).
3. Start: uvicorn api.main:app --host 0.0.0.0 --port 8000
4. Find the LAN IP: ipconfig (Windows) / ifconfig (Mac). Allow port 8000 in the firewall.
5. Check from laptop 2's browser: http://<ip>:8000/health

## Laptop 2 (UI)
1. git pull; pip install -r requirements.txt
2. streamlit run ui/app.py
3. Enter http://<laptop1-ip>:8000 in the sidebar; the 10 trips appear.

## During evaluation
- Evaluator picks a trip, clicks Reconstruct route.
- Show the request log lines on laptop 1 (POST /predict/<trip> received / done with metrics).
- Both laptops on the same Wi-Fi or a phone hotspot; avoid networks with client isolation.

## Rehearsal results
Fill in: trip id, accuracy shown, API time, any issue found.
