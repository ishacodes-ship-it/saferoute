"""Download the pedestrian road network for the configured city from OpenStreetMap.

Uses OSMnx, which wraps the Overpass API — free, no key. The resulting graph
is the basis for both the H3 feature grid's road-density features and the
routing layer (src/saferoute/routing).
"""
from __future__ import annotations

import logging
from pathlib import Path

import osmnx as ox

from saferoute.config import Config, load_config

logger = logging.getLogger(__name__)


def fetch_osm_graph(cfg: Config):
    """Download the walk network for the city's bounding box.

    Returns a networkx.MultiDiGraph in OSMnx's standard edge/node schema.
    """
    south, west, north, east = cfg["city"]["bbox"]
    network_type = cfg["data_sources"]["osm"]["network_type"]
    logger.info("Downloading OSM %s network for bbox=%s", network_type, cfg["city"]["bbox"])
    graph = ox.graph_from_bbox(north, south, east, west, network_type=network_type, simplify=True)
    logger.info("Downloaded graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def main(config_path: str | None = None) -> Path:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config_path) if config_path else load_config()
    graph = fetch_osm_graph(cfg)
    out_path = cfg.path("data_raw") / f"{cfg['city']['name']}_walk_network.graphml"
    ox.save_graphml(graph, out_path)
    logger.info("Wrote %s", out_path)
    return out_path


if __name__ == "__main__":
    main()
