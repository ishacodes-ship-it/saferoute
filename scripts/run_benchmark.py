#!/usr/bin/env python
"""M4: counterfactual held-out-incident benchmark. Writes results/tables/benchmark_raw.csv
and results/tables/benchmark_bootstrap_ci.csv (avoidance rate vs detour budget per method,
with bootstrap CIs) -- the Pareto curve data required by FR-5.
"""
import argparse
import logging

import joblib
import pandas as pd

from saferoute.benchmark.counterfactual import bootstrap_avoidance_ci, generate_od_pairs_near_incidents, run_benchmark
from saferoute.config import DEFAULT_CONFIG_PATH, load_config
from saferoute.routing.baselines import build_static_risk_table
from saferoute.routing.graph import load_routing_graph
from saferoute.routing.risk_lookup import RiskLookup, build_cell_static_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    parser.add_argument("--n-incidents", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH, args.override)
    city = cfg["city"]["name"]

    graph = load_routing_graph(cfg.path("data_raw") / f"{city}_walk_network.graphml", cfg["spatial"]["h3_resolution"])
    mapie_model = joblib.load(cfg.path("models") / "risk_model_conformal.joblib")
    feature_table = pd.read_parquet(cfg.path("data_processed") / "feature_table_full.parquet")
    cell_static = build_cell_static_features(feature_table)
    risk_lookup = RiskLookup(mapie_model, cell_static, cfg["uncertainty"]["nominal_coverage"])

    static_table = build_static_risk_table(risk_lookup, cell_static["h3_cell"].tolist(), pd.Timestamp(cfg["split"]["test_end"]))

    test_df = pd.read_parquet(cfg.path("data_processed") / "test.parquet")
    held_out_incidents = test_df[test_df["has_incident"] == 1]
    logger.info("Held-out test-period incidents available: %d", len(held_out_incidents))

    od_pairs = generate_od_pairs_near_incidents(
        held_out_incidents, graph, args.n_incidents, walk_hops=5, seed=cfg["seed"],
        min_od_distance_m=cfg["benchmark"]["min_od_distance_m"],
    )
    if not od_pairs:
        raise RuntimeError(
            "generate_od_pairs_near_incidents produced 0 OD pairs -- the benchmark has nothing to run on. "
            "Check that held_out_incidents has usable h3_cell values and that the routing graph covers those cells."
        )

    results_df = run_benchmark(
        graph, od_pairs, risk_lookup, static_table,
        cfg["routing"]["risk_aversion_lambda"], cfg["routing"]["default_mu"], cfg["benchmark"]["detour_budget_fractions"],
    )
    raw_path = cfg.path("tables") / "benchmark_raw.csv"
    results_df.to_csv(raw_path, index=False)
    logger.info("Wrote %s (%d rows)", raw_path, len(results_df))

    ci_rows = []
    for method in results_df["method"].unique():
        for budget in cfg["benchmark"]["detour_budget_fractions"]:
            ci_rows.append(bootstrap_avoidance_ci(
                results_df, method, budget, cfg["benchmark"]["n_bootstrap"], cfg["benchmark"]["confidence_level"], cfg["seed"]
            ))
    ci_df = pd.DataFrame(ci_rows)
    ci_path = cfg.path("tables") / "benchmark_bootstrap_ci.csv"
    ci_df.to_csv(ci_path, index=False)
    logger.info("Wrote %s", ci_path)


if __name__ == "__main__":
    main()
