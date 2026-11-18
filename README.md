# PathSnap - GPS map-matching and route reconstruction (Team 54)

## Overview

Noisy GPS points are matched to the road network. For each point, the system collects candidate directed arcs near the point. A binary classifier scores each candidate with the probability that it is the correct road. An HMM decoded with Viterbi then picks a connected route through the candidates, so the output is a continuous path rather than an independent choice per point. The system is served by a FastAPI backend (laptop 1, which holds the evaluator data and the ground truth) and a Streamlit UI (laptop 2), which only sends a trip id and displays the result.

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library: I/O, projection, road network, candidates, features, labels, models, HMM/Viterbi, metrics, inference pipeline |
| `scripts/` | EDA, dataset build, model comparison, Bayesian tuning, model export, HMM evaluation, map generation |
| `api/` | FastAPI inference backend (`api/main.py`) |
| `ui/` | Streamlit evaluator UI (`ui/app.py`) |
| `models/` | Exported classifier (`classifier.joblib`), HMM parameters (`hmm_params.json`), model card (`model_card.json`) |
| `results/` | CSV results and plots produced by the scripts |
| `maps/` | Folium HTML maps for development |
| `docs/` | Technical report (`PathSnap_Report_Team54.pdf`) and deployment checklist |
| `data/` | Raw and processed data, evaluator trips (not committed) |

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment (Windows: `.venv\Scripts\activate`; macOS/Linux: `source .venv/bin/activate`), then:

```bash
pip install -r requirements.txt
```

## Data

Unzip the provided "Student folder" so that the following folders exist:

- `data/raw/train/<trip>/`
- `data/raw/student_test/<trip>/`

Evaluator trips go in `data/evaluator/<trip>/` on the API laptop only. Data is never committed (see `.gitignore`).

## Reproducing the results

Run the following commands in order from the project root.

1. Exploratory data analysis. Writes `results/eda/` (inventory and partition CSVs, `data_quality.md`, plots).

```bash
python scripts/eda.py
```

Static GPS-trajectory and road-network plots and the missing-value summary (`results/eda/trajectory_*.png`, `results/eda/missing_values.csv`):

```bash
python scripts/plot_trajectories.py
```

2. Candidate dataset. Writes `data/processed/` (candidate and point parquet files) and `results/candidate_stats.csv`.

```bash
python scripts/build_dataset.py
```

3. Classifier comparison. Writes `results/model_comparison.csv`, `results/cv_folds.csv`, `results/selected_model.json` and plots in `results/plots/`.

```bash
python scripts/compare_models.py
```

4. Bayesian hyperparameter optimisation. Writes `results/optuna_trials_tpe.csv`, `results/optuna_trials_random.csv`, `results/best_params.json` and `results/plots/optimisation_history.png`.

```bash
python scripts/tune_bayes.py
```

5. Model export. Writes `results/tuned_vs_untuned.csv`, `models/classifier.joblib` and `models/model_card.json`.

```bash
python scripts/export_model.py
```

6. HMM/Viterbi evaluation. Writes `results/hmm_grid.csv`, `results/hmm_comparison.csv`, `models/hmm_params.json` and `results/plots/hmm_vs_classifier.png`.

```bash
python scripts/eval_hmm.py
```

7. Maps. Writes `maps/<partition>_<trip>.html` (Folium maps of selected trips).

```bash
python scripts/make_maps.py
```

## Deployment (two laptops)

**Laptop 1 (API)**

1. Put the evaluator trips in `data/evaluator/`.
2. Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

3. Find the laptop's IP address with `ipconfig`.
4. Allow port 8000 in the firewall.

The console shows a log line for every request.

**Laptop 2 (UI)**

1. Start the UI:

```bash
streamlit run ui/app.py
```

2. In the sidebar, enter `http://<laptop1-ip>:8000` as the API address.
3. Pick a trip and click **Reconstruct route**.

Ground truth stays on laptop 1. The UI only sends the trip id.

Shortcut scripts (Windows): on laptop 1 run `run_api.bat` (or `run_api.bat rehearsal` to serve the student_test trips); it prints the laptop's IP addresses. On laptop 2 run `run_ui.bat <laptop1-ip>`; it checks `/health` and opens the UI with the API address filled in.

## API endpoints

- `GET /health`: liveness check; returns the status, the loaded classifier type, the HMM parameters and the number of evaluator trips.
- `GET /trips`: returns the ids of the evaluator trips available on the server.
- `POST /predict/{trip_id}?use_hmm=true`: runs the full pipeline on one trip (`use_hmm=false` returns the classifier-only route). The response contains:
  - `metrics`: observation Top-1 accuracy, classifier-only Top-1 accuracy, route precision and recall, matched point count, point count, route arc count, route continuity, chain breaks and processing time in seconds. Accuracy and route metrics are computed on the server against the reference route.
  - `gps`: the GPS observations as latitude/longitude pairs (thinned for large trips).
  - `predicted_route`: the predicted route as a list of line segments.
  - `road_network`: the road segments near the track, for display.

The reference route is never returned to the client.

## Leakage rules followed

- Cross-validation uses GroupKFold by trip on the train trips, so no trip is split between folds.
- `student_test` is used only for comparison of the final models, not for choosing them.
- Evaluator data is never used in development.
- No predictions are hardcoded.
- The reference route is read by the API only after prediction, and only for scoring.

## Team

| Member | SRN | Area |
|---|---|---|
| Rahul Saran Sharma | PES1UG24AM218 | Project skeleton, preprocessing (projection, network, label alignment, EDA, features, candidates), MLP, Gradient Boosting, model comparison, Folium maps, README |
| Renetta Isabel Nathan | PES1UG24AM221 | Training harness, Decision Tree, Random Forest, HMM/Viterbi decoder and evaluation, inference pipeline |
| Ramalakshmi M N | PES1UG24AM219 | Logistic Regression, Gaussian Naive Bayes, Bayesian hyperparameter optimisation, FastAPI backend |
| Ramneek Gautum | PES1UG24AM220 | KNN, SVM, tuned model export, Streamlit UI, deployment checklist |

## AI assistance

AI tools were used to help write parts of the code; all experiments were run by the team and all results in `results/` come from these scripts.
