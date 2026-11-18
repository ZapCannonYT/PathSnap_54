"""Gaussian naive Bayes: independent per-class Gaussians combined with Bayes' rule. Syllabus unit 3."""
from sklearn.naive_bayes import GaussianNB

NAME_TEXT = "Gaussian Naive Bayes"
FAMILY = "Bayesian"
DEFAULTS = dict(var_smoothing=1e-9)
CONFIG = "var_smoothing 1e-9 (maximum-likelihood mean and variance per class)"
MAX_ROWS = None


def build(params=None):
    return GaussianNB(**{**DEFAULTS, **(params or {})})


def search_space(trial):
    return dict(var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True))
