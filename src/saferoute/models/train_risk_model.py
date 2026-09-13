"""M1: train and compare candidate risk models.

Target: `incident_count`, the observed incident count for a (H3 cell, hourly
time bin) pair. Modeled as a regression target (expected incident rate),
not a binary classification probability — this is the standard framing in
spatiotemporal crime-rate prediction (analogous to a Poisson intensity) and
is what makes the conformal-prediction interval in uncertainty.py
interpretable as a rate interval rather than a class-probability interval.

Ground truth is heavily zero-inflated (most cell-hours have 0 incidents) —
this is reported, not hidden, in the evaluation output. Model comparison is
by held-out MAE / RMSE / Poisson deviance on the validation split; the
config's `model.primary` is the default but the actual best-performing
model per this comparison should be the one reported in the paper, not
assumed in advance.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from xgboost import XGBRegressor

from saferoute.config import Config, load_config
from saferoute.models.feature_spec import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

TARGET_COLUMN = "incident_count"


def _prep_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN].clip(lower=0)


def _build_candidates(cfg: Config, seed: int) -> dict:
    m = cfg["model"]
    candidates = {}
    if "xgboost" in m["candidates"]:
        p = m["xgboost"]
        candidates["xgboost"] = XGBRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            objective="count:poisson",
            random_state=seed,
            n_jobs=-1,
        )
    if "lightgbm" in m["candidates"]:
        p = m["lightgbm"]
        candidates["lightgbm"] = LGBMRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"],
            objective="poisson",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if "ebm" in m["candidates"]:
        p = m["ebm"]
        candidates["ebm"] = ExplainableBoostingRegressor(
            max_bins=p["max_bins"],
            interactions=p["interactions"],
            random_state=seed,
        )
    return candidates


def _evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = np.clip(model.predict(X), a_min=1e-6, a_max=None)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "poisson_deviance": float(mean_poisson_deviance(y, preds)),
        "mean_predicted": float(preds.mean()),
        "mean_actual": float(y.mean()),
        "n_samples": int(len(y)),
    }


def train_and_compare(cfg: Config) -> dict:
    processed_dir = cfg.path("data_processed")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    logger.info(
        "Train zero-inflation: %.4f%% of rows have incident_count==0",
        100.0 * (train_df[TARGET_COLUMN] == 0).mean(),
    )

    X_train, y_train = _prep_xy(train_df)
    X_val, y_val = _prep_xy(val_df)

    candidates = _build_candidates(cfg, cfg["seed"])
    results = {}
    fitted = {}
    for name, model in candidates.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        val_metrics = _evaluate(model, X_val, y_val)
        train_metrics = _evaluate(model, X_train, y_train)
        results[name] = {"val": val_metrics, "train": train_metrics}
        fitted[name] = model
        logger.info("%s val MAE=%.4f RMSE=%.4f poisson_dev=%.4f", name, val_metrics["mae"], val_metrics["rmse"], val_metrics["poisson_deviance"])

    best_name = min(results, key=lambda n: results[n]["val"]["mae"])
    logger.info("Best model by validation MAE: %s", best_name)

    models_dir = cfg.path("models")
    for name, model in fitted.items():
        joblib.dump(model, models_dir / f"risk_model_{name}.joblib")
    joblib.dump({"best_model": best_name, "feature_columns": FEATURE_COLUMNS}, models_dir / "risk_model_meta.joblib")

    metrics_path = cfg.path("tables") / "model_comparison.json"
    with open(metrics_path, "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)
    logger.info("Wrote %s", metrics_path)

    return {"results": results, "best_model": best_name, "models": fitted}


def get_feature_importance(model, model_name: str) -> pd.DataFrame:
    """Return a (feature, importance) table. Method differs by model family."""
    if model_name in ("xgboost", "lightgbm"):
        importances = model.feature_importances_
        return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances}).sort_values(
            "importance", ascending=False
        )
    if model_name == "ebm":
        explanation = model.explain_global()
        names = explanation.data()["names"]
        scores = explanation.data()["scores"]
        return pd.DataFrame({"feature": names, "importance": scores}).sort_values("importance", ascending=False)
    raise ValueError(f"Unknown model_name: {model_name}")


def run(config_path: str | None = None, override_path: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    from saferoute.config import DEFAULT_CONFIG_PATH

    cfg = load_config(config_path or DEFAULT_CONFIG_PATH, override_path)
    result = train_and_compare(cfg)

    importance_df = get_feature_importance(result["models"][result["best_model"]], result["best_model"])
    importance_path = cfg.path("tables") / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info("Wrote %s", importance_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()
    run(args.config, args.override)
