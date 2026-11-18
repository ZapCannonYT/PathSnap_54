# PathSnap - GPS map-matching and route reconstruction (Team 54)

Work in progress. Noisy GPS trajectories are matched to the road network with a
candidate classifier and an HMM decoded by Viterbi, served by a FastAPI backend
(laptop 1) and a Streamlit UI (laptop 2).

## Setup
    python -m venv .venv
    .venv\Scripts\activate        (Windows)  |  source .venv/bin/activate  (Mac/Linux)
    pip install -r requirements.txt

## Data
Unzip the provided "Student folder" into data/raw/ so that data/raw/train/<trip>/ and
data/raw/student_test/<trip>/ exist. Data is never committed (.gitignore).

## Layout
    pathsnap/   library code        scripts/  runnable steps
    api/        FastAPI backend     ui/       Streamlit UI
    results/    CSVs and plots      models/   exported artifacts
    maps/       Folium HTML maps    docs/     notes
