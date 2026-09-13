"""Calibrated uncertainty via split conformal prediction (MAPIE).

Wraps the chosen point-estimate regressor (from train_risk_model.py) with
MAPIE's split-conformal regression, which is model-agnostic: it needs a
held-out calibration set disjoint from both training and the final
evaluation set (config: uncertainty.calibration_fraction carves this out of
the training split — never reuse validation or test rows for calibration,
that would leak into the coverage evaluation).

Returns, per row: point estimate (mean), lower bound, upper bound, and a
scalar `sigma` = (upper - lower) / 2 as the uncertainty measure the router
uses in `cost = time * (1 + lambda*risk + mu*sigma)`.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor
from sklearn.model_selection import train_test_split

from saferoute.config import Config, load_config
from saferoute.models.feature_spec import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

TARGET_COLUMN = "incident_count"


def fit_conformal_wrapper(base_model, train_df: pd.DataFrame, calibration_fraction: float, seed: int) -> MapieRegressor:
    """Fit a MAPIE split-conformal wrapper.

    `cv="split"` makes MAPIE internally carve out `test_size` (=
    calibration_fraction) of the rows passed to `.fit()` as a held-out
    calibration set (via ShuffleSplit) and fits `estimator` fresh on the
    complement — it does not reuse the already-trained model artifact from
    train_risk_model.py, since that model saw the full training set,
    including rows this method needs disjoint for calibration.
    """
    df = train_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN].clip(lower=0)

    mapie = MapieRegressor(estimator=base_model, cv="split", test_size=calibration_fraction, random_state=seed)
    mapie.fit(X, y)
    return mapie


def predict_with_interval(mapie_model: MapieRegressor, X: pd.DataFrame, nominal_coverage: float) -> pd.DataFrame:
    alpha = 1 - nominal_coverage
    y_pred, y_pis = mapie_model.predict(X, alpha=alpha)
    lower = np.clip(y_pis[:, 0, 0], a_min=0, a_max=None)
    upper = np.clip(y_pis[:, 1, 0], a_min=0, a_max=None)
    mean = np.clip(y_pred, a_min=0, a_max=None)
    sigma = (upper - lower) / 2.0
    return pd.DataFrame({"risk_mean": mean, "risk_lower": lower, "risk_upper": upper, "sigma": sigma})


def evaluate_coverage(mapie_model: MapieRegressor, X: pd.DataFrame, y_true: pd.Series, nominal_coverage: float) -> dict:
    """Empirical coverage: fraction of true values that actually fall in [lower, upper].

    This is the number reported against the PRD's >=90% target — it is
    computed here, never assumed.
    """
    intervals = predict_with_interval(mapie_model, X, nominal_coverage)
    covered = (y_true.values >= intervals["risk_lower"].values) & (y_true.values <= intervals["risk_upper"].values)
    return {
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": float(covered.mean()),
        "mean_interval_width": float((intervals["risk_upper"] - intervals["risk_lower"]).mean()),
        "n_samples": int(len(y_true)),
    }


def run(config_path: str | None = None, override_path: str | None = None) -> None:
    import json

    from saferoute.models.train_risk_model import _build_candidates

    logging.basicConfig(level=logging.INFO)
    from saferoute.config import DEFAULT_CONFIG_PATH

    cfg = load_config(config_path or DEFAULT_CONFIG_PATH, override_path)

    processed_dir = cfg.path("data_processed")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    meta = joblib.load(cfg.path("models") / "risk_model_meta.joblib")
    best_name = meta["best_model"]
    base_model = _build_candidates(cfg, cfg["seed"])[best_name]

    logger.info("Fitting conformal wrapper around %s", best_name)
    mapie_model = fit_conformal_wrapper(base_model, train_df, cfg["uncertainty"]["calibration_fraction"], cfg["seed"])
    joblib.dump(mapie_model, cfg.path("models") / "risk_model_conformal.joblib")

    X_test, y_test = test_df[FEATURE_COLUMNS].dropna(), None
    test_clean = test_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    coverage = evaluate_coverage(
        mapie_model,
        test_clean[FEATURE_COLUMNS],
        test_clean[TARGET_COLUMN].clip(lower=0),
        cfg["uncertainty"]["nominal_coverage"],
    )
    logger.info("Empirical coverage on test set: %.4f (nominal=%.2f)", coverage["empirical_coverage"], coverage["nominal_coverage"])

    out_path = cfg.path("tables") / "conformal_coverage.json"
    with open(out_path, "w") as f:
        json.dump(coverage, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()
    run(args.config, args.override)
