"""Logistic regression on standardised features. Syllabus unit 1."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME_TEXT = "Logistic Regression"
FAMILY = "Linear"
DEFAULTS = dict(C=1.0)
CONFIG = "C=1, balanced classes, standardised features"
MAX_ROWS = None


def build(params=None):
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000,
                                                              **{**DEFAULTS, **(params or {})}))


def search_space(trial):
    return dict(C=trial.suggest_float("C", 1e-3, 1e2, log=True))
