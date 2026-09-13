"""M0 orchestrator: raw data -> H3 grid -> temporal bins -> joined feature table -> splits.

Run after the four ingest_*.py scripts have populated data/raw/. Produces
data/processed/{train,val,test,spatial_holdout}.parquet plus
data/processed/feature_table_full.parquet (pre-split, for inspection).

Grid-size note: to keep the dense (cell x hour) label table tractable, the
enumerated H3 grid is restricted to cells that intersect the OSM walk
network (drops cells over water/rail yards/etc. with no pedestrian
relevance) — see `_walkable_cells`. This is a documented scoping decision,
not silent data loss: the dropped cells are logged.
"""
from __future__ import annotations

import argparse
import logging

import networkx as nx
import osmnx as ox
import pandas as pd

from saferoute.config import Config, load_config
from saferoute.features.environment_features import build_environment_features
from saferoute.features.h3_grid import assign_h3_cells, enumerate_grid_cells
from saferoute.features.labels import add_historical_rate_features, build_dense_label_table
from saferoute.features.splitting import apply_spatial_holdout, temporal_block_split
from saferoute.features.temporal import assign_temporal_bin, full_time_bin_range

logger = logging.getLogger(__name__)


def _walkable_cells(graph, bbox: list[float], resolution: int) -> list[str]:
    all_cells = enumerate_grid_cells(bbox, resolution)
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False).reset_index()
    nodes = nodes.rename(columns={"y": "latitude", "x": "longitude"})
    nodes = assign_h3_cells(nodes, resolution)
    walkable = set(nodes["h3_cell"].unique())
    kept = [c for c in all_cells if c in walkable]
    logger.info(
        "Grid scoping: %d/%d cells intersect the walk network (%d dropped)",
        len(kept), len(all_cells), len(all_cells) - len(kept),
    )
    return kept


def build_feature_table(cfg: Config) -> pd.DataFrame:
    city = cfg["city"]["name"]
    raw_dir = cfg.path("data_raw")

    crime_df = pd.read_parquet(raw_dir / f"{city}_crime_raw.parquet")
    graph = ox.load_graphml(raw_dir / f"{city}_walk_network.graphml")
    weather_df = pd.read_parquet(raw_dir / f"{city}_weather_hourly.parquet")

    resolution = cfg["spatial"]["h3_resolution"]
    bin_size = cfg["temporal"]["bin_size_hours"]

    crime_df = assign_h3_cells(crime_df, resolution)
    crime_df = assign_temporal_bin(crime_df, "date", bin_size)

    cells = _walkable_cells(graph, cfg["city"]["bbox"], resolution)
    time_bins = full_time_bin_range(
        cfg["data_sources"]["crime"]["date_start"],
        cfg["data_sources"]["crime"]["date_end"],
        bin_size,
    )
    logger.info("Dense grid: %d cells x %d time bins = %d rows", len(cells), len(time_bins), len(cells) * len(time_bins))

    dense = build_dense_label_table(crime_df, cells, time_bins)
    dense = assign_temporal_bin(dense, "time_bin", bin_size)
    dense = add_historical_rate_features(dense, cfg["temporal"]["lookback_days_for_features"], bin_size)

    env_features = build_environment_features(graph, cfg["city"]["bbox"], cells, resolution)
    dense = dense.merge(env_features, on="h3_cell", how="left")

    weather_df = weather_df.rename(columns={"timestamp": "time_bin"})
    dense = dense.merge(weather_df, on="time_bin", how="left")

    return dense


def run(config_path: str | None = None, override_path: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    from saferoute.config import DEFAULT_CONFIG_PATH

    cfg = load_config(config_path or DEFAULT_CONFIG_PATH, override_path)

    feature_table = build_feature_table(cfg)
    processed_dir = cfg.path("data_processed")
    feature_table.to_parquet(processed_dir / "feature_table_full.parquet", index=False)
    logger.info("Wrote feature_table_full.parquet (%d rows)", len(feature_table))

    non_holdout, spatial_holdout = apply_spatial_holdout(
        feature_table, cfg["split"]["spatial_holdout_fraction"], cfg["seed"]
    )
    train, val, test = temporal_block_split(
        non_holdout, cfg["split"]["train_end"], cfg["split"]["val_end"], cfg["split"]["test_end"]
    )

    for name, part in [("train", train), ("val", val), ("test", test), ("spatial_holdout", spatial_holdout)]:
        path = processed_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(part))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()
    run(args.config, args.override)
