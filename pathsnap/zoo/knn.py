"""k-nearest neighbours on standardised features. Syllabus unit 1."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME_TEXT = "K-Nearest Neighbours"
FAMILY = "Instance-based"
DEFAULTS = dict(n_neighbors=25, weights="distance")
CONFIG = "k=25, distance-weighted votes, standardised features, 100k training rows"
MAX_ROWS = 100_000


def build(params=None):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1, **{**DEFAULTS, **(params or {})}))


def search_space(trial):
    return dict(n_neighbors=trial.suggest_int("n_neighbors", 3, 101, log=True),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]))
