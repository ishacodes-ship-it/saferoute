"""Explanation faithfulness: deletion/insertion test (PRD §12, target >=90% direction-consistency).

For each whitelisted feature and each sampled row, perturb that feature in
the direction that should WORSEN safety (e.g. remove lighting, move CCTV/
businesses to zero, push distance-to-help to the sample's observed max) and
check whether the model's predicted risk moves in the expected direction
(up). This tests whether the model's own behavior is consistent with the
causal direction implied by the whitelist phrasing used in
template_nlg.py — if it isn't, the generated explanations would be
unfaithful to what the model actually does.

This module reports a rate; it never asserts the >=90% target is met
without running scikit-learn/xgboost-backed prediction and counting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from saferoute.models.feature_spec import FEATURE_COLUMNS

# feature -> (worsening_direction, ) matching template_nlg.py's FEATURE_PHRASES directions.
WORSENING_DIRECTION = {
    "lighting": "decrease",         # less light -> more risk
    "footfall_proxy": "decrease",   # fewer intersections/less activity -> more risk
    "open_businesses": "decrease",  # fewer open businesses -> more risk
    "cctv": "decrease",             # less CCTV -> more risk
    "distance_to_help": "increase", # farther from help -> more risk
}


def _perturb(X: pd.DataFrame, feature: str, direction: str) -> pd.DataFrame:
    """Push `feature` to a value guaranteed to differ from EVERY row's original
    value, in the worsening direction -- setting "increase" to exactly the
    column max would be a no-op for whichever row already sits at the max,
    silently reporting a false inconsistency for that row.
    """
    X_pert = X.copy()
    if direction == "decrease":
        col_min = X[feature].min()
        margin = max(abs(col_min), 1.0)
        X_pert[feature] = min(0.0, col_min - margin)
    else:
        col_max = X[feature].max()
        margin = max(abs(col_max), 1.0)
        X_pert[feature] = col_max + margin
    return X_pert


def evaluate_faithfulness(model, X: pd.DataFrame, predict_fn=None) -> dict:
    """
    Args:
        model: fitted regressor with .predict, OR ignored if predict_fn given.
        X: sampled feature rows (must contain FEATURE_COLUMNS).
        predict_fn: optional callable(X) -> np.ndarray, for wrapping e.g. a
            MapieRegressor whose .predict signature differs (returns a tuple
            when alpha is passed) — pass `lambda df: mapie_model.predict(df)`
            (no alpha) to get plain point predictions.
    """
    predict = predict_fn if predict_fn is not None else model.predict
    baseline_pred = np.asarray(predict(X[FEATURE_COLUMNS]))

    per_feature_results = {}
    all_consistent = []

    for feature, direction in WORSENING_DIRECTION.items():
        # A feature with zero variance in X (e.g. an OSM tag with no coverage
        # at all in a given bbox -- observed for `lighting` in the pilot run,
        # see results/logs/pilot_run_notes.md) gives the model nothing to
        # have learned a relationship FROM. Its perturbation necessarily has
        # delta==0 for every row, which is "the model correctly places no
        # weight on an uninformative feature," not "the model is unfaithful
        # to the explanation direction." Conflating the two by folding it
        # into the same aggregate rate as genuinely-tested features would
        # silently understate faithfulness for reasons that have nothing to
        # do with faithfulness -- so it's flagged and excluded from the
        # aggregate, not just averaged in.
        zero_variance = bool(X[feature].std() == 0)

        X_pert = _perturb(X, feature, direction)
        pert_pred = np.asarray(predict(X_pert[FEATURE_COLUMNS]))
        delta = pert_pred - baseline_pred
        consistent = delta > 0  # worsening the feature should raise predicted risk
        per_feature_results[feature] = {
            "direction_consistency_rate": float(consistent.mean()),
            "mean_delta": float(delta.mean()),
            "n_samples": int(len(X)),
            "zero_variance_in_sample": zero_variance,
        }
        if not zero_variance:
            all_consistent.append(consistent)

    overall_rate = float(np.concatenate(all_consistent).mean()) if all_consistent else None
    excluded = [f for f, r in per_feature_results.items() if r["zero_variance_in_sample"]]
    return {
        "overall_direction_consistency_rate": overall_rate,
        "per_feature": per_feature_results,
        "n_samples": int(len(X)),
        "features_excluded_from_aggregate": excluded,
    }
