"""Gradient boosting: shallow trees fitted one after another to the loss gradient. Syllabus unit 3."""
from sklearn.ensemble import GradientBoostingClassifier

from . import SEED

NAME_TEXT = "Gradient Boosting"
FAMILY = "Boosting"
DEFAULTS = dict(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8, min_samples_leaf=1)
CONFIG = "200 trees, learning rate 0.1, depth 3, subsample 0.8, 100k training rows"
MAX_ROWS = 100_000


def build(params=None):
    return GradientBoostingClassifier(random_state=SEED, **{**DEFAULTS, **(params or {})})


def search_space(trial):
    return dict(n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True))
