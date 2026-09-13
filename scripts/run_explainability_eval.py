#!/usr/bin/env python
"""M3 evaluation: whitelist-violation audit (FR-4, target 0/1000) and
explanation faithfulness (deletion/insertion test, target >=90% direction consistency).

Writes results/tables/whitelist_audit.json and results/tables/faithfulness.json.
"""
import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd

from saferoute.config import DEFAULT_CONFIG_PATH, load_config
from saferoute.explain.faithfulness import evaluate_faithfulness
from saferoute.explain.template_nlg import generate_explanation
from saferoute.explain.whitelist_audit import audit_explanation_batch
from saferoute.models.feature_spec import ENVIRONMENT_FEATURES, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH, args.override)

    test_df = pd.read_parquet(cfg.path("data_processed") / "test.parquet").dropna(subset=FEATURE_COLUMNS)
    rng = np.random.default_rng(cfg["seed"])

    n_audit = min(cfg["explainability"]["audit_sample_size"], len(test_df) // 2)
    sample = test_df.sample(n=2 * n_audit, random_state=cfg["seed"]).reset_index(drop=True)
    chosen_rows = sample.iloc[:n_audit]
    alt_rows = sample.iloc[n_audit:2 * n_audit].reset_index(drop=True)

    explanations = []
    for i in range(n_audit):
        chosen_feats = chosen_rows.iloc[i][ENVIRONMENT_FEATURES].to_dict()
        alt_feats = alt_rows.iloc[i][ENVIRONMENT_FEATURES].to_dict()
        risk_reduction = float(rng.uniform(5, 40))  # placeholder magnitude for the audit -- audit tests TEXT CONTENT, not this number
        explanations.append(generate_explanation(chosen_feats, alt_feats, risk_reduction))

    audit_result = audit_explanation_batch(explanations)
    audit_path = cfg.path("tables") / "whitelist_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit_result, f, indent=2)
    logger.info("Whitelist audit: %d/%d violations. Wrote %s", audit_result["n_violations"], audit_result["n_sampled"], audit_path)

    meta = joblib.load(cfg.path("models") / "risk_model_meta.joblib")
    best_name = meta["best_model"]
    model = joblib.load(cfg.path("models") / f"risk_model_{best_name}.joblib")

    faith_sample = test_df.sample(n=min(cfg["explainability"]["faithfulness_sample_size"], len(test_df)), random_state=cfg["seed"])
    faithfulness_result = evaluate_faithfulness(model, faith_sample)
    faith_path = cfg.path("tables") / "faithfulness.json"
    with open(faith_path, "w") as f:
        json.dump(faithfulness_result, f, indent=2)
    logger.info("Faithfulness overall direction-consistency: %.2f%%. Wrote %s", 100 * faithfulness_result["overall_direction_consistency_rate"], faith_path)


if __name__ == "__main__":
    main()
