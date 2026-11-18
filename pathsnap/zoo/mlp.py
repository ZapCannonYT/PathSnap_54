"""Multi-layer perceptron trained by backpropagation with the Adam optimiser. Syllabus unit 2."""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import SEED

NAME_TEXT = "MLP (neural network)"
FAMILY = "Neural network"
DEFAULTS = dict(hidden=(64, 32), alpha=1e-4, learning_rate_init=1e-3)
CONFIG = "hidden layers (64, 32), ReLU, Adam, early stopping on 10% validation"
MAX_ROWS = None


def build(params=None):
    p = dict(DEFAULTS)
    if params:
        p.update(params)
        if "n_layers" in params:
            p["hidden"] = tuple([int(params["n_units"])] * int(params["n_layers"]))
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=p["hidden"], activation="relu", solver="adam", alpha=p["alpha"],
        learning_rate_init=p["learning_rate_init"], early_stopping=True, max_iter=200, random_state=SEED))


def search_space(trial):
    return dict(n_layers=trial.suggest_int("n_layers", 1, 3),
                n_units=trial.suggest_int("n_units", 16, 128, log=True),
                alpha=trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True))
