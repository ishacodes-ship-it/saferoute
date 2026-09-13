#!/usr/bin/env python
"""Generate the paper's tables from results/tables/*.csv|json.

Same policy as generate_figures.py: a table is only written if its
underlying real-run output file exists. Missing inputs produce a
NOT_YET_EVALUATED placeholder, never fabricated numbers.
"""
import argparse
import json
import logging

import pandas as pd

from saferoute.config import DEFAULT_CONFIG_PATH, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _skip(out_path, reason):
    logger.warning("SKIPPED %s: %s", out_path.name, reason)
    out_path.with_suffix(".NOT_YET_EVALUATED.txt").write_text(f"Table not generated: {reason}\n")


def table_dataset_statistics(cfg, tables_dir):
    path = cfg.path("data_processed") / "feature_table_full.parquet"
    out = tables_dir / "table1_dataset_statistics.csv"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_parquet(path)
    stats = pd.DataFrame([{
        "n_rows": len(df),
        "n_cells": df["h3_cell"].nunique(),
        "n_time_bins": df["time_bin"].nunique(),
        "date_start": df["time_bin"].min(),
        "date_end": df["time_bin"].max(),
        "total_incidents": int(df["incident_count"].sum()),
        "pct_zero_incident_rows": float((df["incident_count"] == 0).mean()) * 100,
    }])
    stats.to_csv(out, index=False)
    logger.info("Wrote %s", out)


def table_feature_description(cfg, tables_dir):
    from saferoute.models.feature_spec import CONTEXTUAL_FEATURES, ENVIRONMENT_FEATURES, TEMPORAL_FEATURES

    out = tables_dir / "table2_feature_description.csv"
    rows = []
    for f in ENVIRONMENT_FEATURES:
        rows.append({"feature": f, "category": "environment (whitelisted)", "source": "OSM-derived proxy — see features/environment_features.py docstring"})
    for f in TEMPORAL_FEATURES:
        rows.append({"feature": f, "category": "temporal", "source": "derived from arrival/observation timestamp"})
    for f in CONTEXTUAL_FEATURES:
        rows.append({"feature": f, "category": "contextual", "source": "historical rate (causal) or Open-Meteo weather archive"})
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s (feature description is static metadata, always available)", out)


def table_model_comparison(cfg, tables_dir):
    path = tables_dir / "model_comparison.json"
    out = tables_dir / "table3_model_comparison.csv"
    if not path.exists():
        return _skip(out, f"missing {path}")

    with open(path) as f:
        results = json.load(f)["results"]
    rows = [{"model": name, **metrics["val"]} for name, metrics in results.items()]
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)


def table_calibration_results(cfg, tables_dir):
    path = tables_dir / "conformal_coverage.json"
    out = tables_dir / "table4_calibration_results.csv"
    if not path.exists():
        return _skip(out, f"missing {path}")

    with open(path) as f:
        cov = json.load(f)
    pd.DataFrame([cov]).to_csv(out, index=False)
    logger.info("Wrote %s", out)


def table_routing_comparison(cfg, tables_dir):
    path = tables_dir / "routing_ablation.csv"
    out = tables_dir / "table5_routing_comparison.csv"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_csv(path)
    summary = df.groupby("method").agg(
        mean_travel_time_s=("travel_time_s", "mean"),
        mean_accumulated_risk=("accumulated_risk", "mean"),
        mean_accumulated_sigma=("accumulated_sigma", "mean"),
        n_routes=("travel_time_s", "count"),
    ).reset_index()
    summary.to_csv(out, index=False)
    logger.info("Wrote %s", out)


def table_ablation_study(cfg, tables_dir):
    path = tables_dir / "routing_ablation.csv"
    out = tables_dir / "table6_ablation_study.csv"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_csv(path)
    summary = df.groupby(["method", "departure_time"]).agg(
        mean_travel_time_s=("travel_time_s", "mean"), mean_accumulated_risk=("accumulated_risk", "mean")
    ).reset_index()
    summary.to_csv(out, index=False)
    logger.info("Wrote %s", out)


def table_explanation_evaluation(cfg, tables_dir):
    audit_path = tables_dir / "whitelist_audit.json"
    faith_path = tables_dir / "faithfulness.json"
    out = tables_dir / "table7_explanation_evaluation.csv"
    if not audit_path.exists() or not faith_path.exists():
        return _skip(out, f"missing {audit_path if not audit_path.exists() else faith_path}")

    with open(audit_path) as f:
        audit = json.load(f)
    with open(faith_path) as f:
        faith = json.load(f)

    row = {
        "whitelist_violations": audit["n_violations"],
        "whitelist_n_sampled": audit["n_sampled"],
        "whitelist_violation_rate": audit["violation_rate"],
        "faithfulness_overall_rate": faith["overall_direction_consistency_rate"],
        "faithfulness_n_samples": faith["n_samples"],
    }
    pd.DataFrame([row]).to_csv(out, index=False)
    logger.info("Wrote %s", out)


def table_fairness_evaluation(cfg, tables_dir):
    path = tables_dir / "fairness_audit.json"
    out = tables_dir / "table8_fairness_evaluation.csv"
    if not path.exists():
        return _skip(out, f"missing {path}")

    with open(path) as f:
        data = json.load(f)
    pd.DataFrame([{k: v for k, v in data.items() if k != "group_means"}]).to_csv(out, index=False)
    logger.info("Wrote %s", out)


TABLES = [
    table_dataset_statistics, table_feature_description, table_model_comparison, table_calibration_results,
    table_routing_comparison, table_ablation_study, table_explanation_evaluation, table_fairness_evaluation,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH, args.override)
    tables_dir = cfg.path("tables")

    for fn in TABLES:
        fn(cfg, tables_dir)


if __name__ == "__main__":
    main()
