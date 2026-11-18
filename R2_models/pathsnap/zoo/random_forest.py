"""Random forest: bagged decision trees with random feature subsets. Syllabus unit 3."""
from sklearn.ensemble import RandomForestClassifier

from . import SEED

NAME_TEXT = "Random Forest"
FAMILY = "Ensemble (bagging)"
DEFAULTS = dict(n_estimators=300, max_depth=None, min_samples_leaf=5, max_features="sqrt", criterion="gini")
CONFIG = "300 trees, min_samples_leaf 5, sqrt features, balanced classes"
MAX_ROWS = None


def build(params=None):
    return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED,
                                  **{**DEFAULTS, **(params or {})})


def search_space(trial):
    return dict(n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                max_depth=trial.suggest_int("max_depth", 6, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
                max_features=trial.suggest_float("max_features", 0.3, 1.0),
                criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]))
