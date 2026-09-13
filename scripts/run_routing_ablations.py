#!/usr/bin/env python
"""FR-2 ablation: compare shortest-path / static-risk / ETA-aware / proposed
routing on a sample of OD pairs, and sweep lambda/mu/departure-time grids.

Writes results/tables/routing_ablation.csv (one row per OD pair x method x
config variant) — the raw material for the "route divergence from static
baseline" metric in PRD §12 and the ablation table in §18.
"""
import argparse
import logging

import joblib
import numpy as np
import pandas as pd

from saferoute.config import DEFAULT_CONFIG_PATH, load_config
from saferoute.routing.baselines import build_static_risk_table, route_all_variants
from saferoute.routing.graph import load_routing_graph
from saferoute.routing.risk_lookup import RiskLookup, build_cell_static_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH, args.override)

    city = cfg["city"]["name"]
    graph = load_routing_graph(cfg.path("data_raw") / f"{city}_walk_network.graphml", cfg["spatial"]["h3_resolution"])

    mapie_model = joblib.load(cfg.path("models") / "risk_model_conformal.joblib")
    feature_table = pd.read_parquet(cfg.path("data_processed") / "feature_table_full.parquet")
    cell_static = build_cell_static_features(feature_table)
    risk_lookup = RiskLookup(mapie_model, cell_static, cfg["uncertainty"]["nominal_coverage"])

    all_cells = cell_static["h3_cell"].tolist()
    reference_date = pd.Timestamp(cfg["split"]["test_end"])
    static_table = build_static_risk_table(risk_lookup, all_cells, reference_date)

    rng = np.random.default_rng(cfg["seed"])
    nodes = list(graph.nodes())
    n_pairs = min(cfg["routing"]["n_od_pairs_eval"], len(nodes) // 2)
    pairs = [tuple(rng.choice(nodes, size=2, replace=False)) for _ in range(n_pairs)]

    rows = []
    for departure_str in cfg["routing"]["departure_time_samples"]:
        departure_time = pd.Timestamp(f"{cfg['split']['test_end']} {departure_str}")
        for source, target in pairs:
            variants = route_all_variants(
                graph, source, target, departure_time, risk_lookup, static_table,
                cfg["routing"]["default_lambda"], cfg["routing"]["default_mu"],
            )
            for method, result in variants.items():
                if result is None:
                    continue
                rows.append({
                    "source": source, "target": target, "departure_time": departure_str, "method": method,
                    "travel_time_s": result.travel_time_s, "accumulated_risk": result.accumulated_risk,
                    "accumulated_sigma": result.accumulated_sigma, "cost": result.cost, "n_edges": len(result.edge_risks),
                })

    df = pd.DataFrame(rows)
    out_path = cfg.path("tables") / "routing_ablation.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))

    if len(df):
        pivot = df.pivot_table(index=["source", "target", "departure_time"], columns="method", values="travel_time_s")
        if "shortest_path" in pivot.columns and "proposed" in pivot.columns:
            divergence = (pivot["proposed"] != pivot["shortest_path"]).mean()
            logger.info("Fraction of OD pairs where proposed route differs in travel time from shortest path: %.2f%%", 100 * divergence)


if __name__ == "__main__":
    main()
