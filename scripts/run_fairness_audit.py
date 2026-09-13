#!/usr/bin/env python
"""Phase 2 / FR-6: demographic disparate-impact audit.

Writes results/tables/fairness_audit.json. No target ratio is assumed —
whatever four_fifths_audit computes is what gets reported (PRD §12/§14).
"""
import argparse
import json
import logging

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from saferoute.config import DEFAULT_CONFIG_PATH, load_config
from saferoute.fairness.disparate_impact import bootstrap_four_fifths_ci
from saferoute.routing.eta_cost import time_dependent_shortest_path
from saferoute.routing.graph import load_routing_graph
from saferoute.routing.risk_lookup import RiskLookup, build_cell_static_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ACS variable code for "White alone" (B02001_002E) vs total population (B01003_001E)
# -- used to derive pct_nonwhite, the demographic variable named in the audit methodology.
TOTAL_POP_VAR = "B01003_001E"
WHITE_ALONE_VAR = "B02001_002E"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    parser.add_argument("--n-routes", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH, args.override)
    city = cfg["city"]["name"]

    tracts = gpd.read_file(cfg.path("data_raw") / f"{city}_census_tracts.geojson")
    tracts["pct_nonwhite"] = 1 - (tracts[WHITE_ALONE_VAR] / tracts[TOTAL_POP_VAR].replace(0, np.nan))
    tracts = tracts.dropna(subset=["pct_nonwhite"])

    graph = load_routing_graph(cfg.path("data_raw") / f"{city}_walk_network.graphml", cfg["spatial"]["h3_resolution"])
    mapie_model = joblib.load(cfg.path("models") / "risk_model_conformal.joblib")
    feature_table = pd.read_parquet(cfg.path("data_processed") / "feature_table_full.parquet")
    cell_static = build_cell_static_features(feature_table)
    risk_lookup = RiskLookup(mapie_model, cell_static, cfg["uncertainty"]["nominal_coverage"])

    node_coords = {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}

    rng = np.random.default_rng(cfg["seed"])
    nodes = list(graph.nodes())
    departure_time = pd.Timestamp(f"{cfg['split']['test_end']} 20:00")

    from shapely.geometry import LineString

    linestrings = []
    for _ in range(args.n_routes):
        source, target = rng.choice(nodes, size=2, replace=False)
        result = time_dependent_shortest_path(
            graph, source, target, departure_time, risk_lookup, cfg["routing"]["default_lambda"], cfg["routing"]["default_mu"]
        )
        if result is None:
            continue
        coords = [node_coords[n] for n in result.path if n in node_coords]
        if len(coords) >= 2:
            linestrings.append(LineString(coords))

    logger.info("Built %d proposed routes for the fairness audit", len(linestrings))

    audit_result = bootstrap_four_fifths_ci(
        linestrings, tracts.rename(columns={"GEOID": "GEOID"}), "pct_nonwhite", cfg["benchmark"]["n_bootstrap"], cfg["seed"]
    )
    audit_result["four_fifths_lower"] = cfg["fairness"]["four_fifths_lower"]
    audit_result["four_fifths_upper"] = cfg["fairness"]["four_fifths_upper"]
    audit_result["n_routes"] = len(linestrings)

    out_path = cfg.path("tables") / "fairness_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit_result, f, indent=2, default=str)
    logger.info("Fairness audit ratio=%s flagged=%s. Wrote %s", audit_result.get("ratio"), audit_result.get("flagged"), out_path)


if __name__ == "__main__":
    main()
