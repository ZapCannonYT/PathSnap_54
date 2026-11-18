"""Model zoo: one module per classifier, discovered automatically.

Each module defines
  NAME_TEXT   readable name
  FAMILY      algorithm family
  CONFIG      short text of the default configuration
  MAX_ROWS    cap on training rows (None = all), for slow learners
  build(params=None) -> scikit-learn estimator with predict_proba
  search_space(trial) -> dict   (used by Bayesian tuning)
Adding a model means adding one file here; no other file changes.
"""
from __future__ import annotations

import importlib
import pkgutil

SEED = 42


def names() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))


def get(name: str):
    return importlib.import_module(f"{__name__}.{name}")
