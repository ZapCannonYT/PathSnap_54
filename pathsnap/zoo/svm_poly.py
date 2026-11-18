"""Soft-margin SVM with a degree-2 polynomial kernel; probabilities by Platt scaling. Syllabus unit 2."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import SEED

NAME_TEXT = "SVM (polynomial kernel)"
FAMILY = "Kernel (SVM)"
DEFAULTS = dict(kernel="poly", degree=2, C=1.0)
CONFIG = "soft margin C=1, polynomial kernel degree 2, Platt probabilities, 15k training rows"
MAX_ROWS = 15_000


def build(params=None):
    return make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", cache_size=1000,
                                               random_state=SEED, **{**DEFAULTS, **(params or {})}))


def search_space(trial):
    return dict(C=trial.suggest_float("C", 1e-2, 1e2, log=True),
                degree=trial.suggest_int("degree", 1, 3))
