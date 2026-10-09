"""Model A (L15.2): LightGBM - small decision trees, one after another - on the chart's numbers (features.py).

    from machine.eyes import model_a
    p = model_a.walk(X, y, folds)           # its chance of his yes for every guessed chart (NaN for the others)

Settings: config.EYES_LGBM_PARAMS, config.EYES_LGBM_ROUNDS - fixed before it saw his answers, no tuning. A missing
number (no candle body in the band) stays missing: LightGBM sends it down its own branch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import config


def fit(X: pd.DataFrame, y: np.ndarray, seed: int = 0):
    import lightgbm as lgb
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("model A needs both yes and no answers to learn from")
    params = {**config.EYES_LGBM_PARAMS, "seed": int(seed)}
    data = lgb.Dataset(X.to_numpy(float), label=y, feature_name=list(X.columns), free_raw_data=False)
    return lgb.train(params, data, num_boost_round=int(config.EYES_LGBM_ROUNDS))


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X.to_numpy(float)), dtype=float)


def walk(X: pd.DataFrame, y: np.ndarray, folds, seed: int = 0) -> tuple[np.ndarray, dict]:
    """The time split: each fold's model learns from its train rows and guesses its test rows. Returns the chances
    (NaN where not guessed) and how much each number was used (LightGBM's gain, summed over the folds' models)."""
    y = np.asarray(y)
    p = np.full(len(X), np.nan)
    gain = pd.Series(0.0, index=X.columns)
    for f in folds:
        m = fit(X.iloc[f.train], y[f.train], seed)
        p[f.test] = predict(m, X.iloc[f.test])
        gain += pd.Series(m.feature_importance("gain"), index=X.columns)
    total = float(gain.sum())
    used = (gain / total).sort_values(ascending=False).round(4).to_dict() if total > 0 else {}
    return p, used
