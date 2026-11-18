"""Decision tree, ID3-style: splits chosen by information gain (entropy). Syllabus unit 1."""
from sklearn.tree import DecisionTreeClassifier

from . import SEED

NAME_TEXT = "Decision Tree (ID3-style)"
FAMILY = "Tree"
DEFAULTS = dict(criterion="entropy", max_depth=12, min_samples_leaf=20)
CONFIG = "entropy, max_depth 12, min_samples_leaf 20 (depth and leaf limits control overfitting)"
MAX_ROWS = None


def build(params=None):
    return DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **{**DEFAULTS, **(params or {})})


def search_space(trial):
    return dict(criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                max_depth=trial.suggest_int("max_depth", 4, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 200, log=True))
