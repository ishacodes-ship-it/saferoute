"""Routing graph construction: OSM walk network -> edges with travel time + H3 cell tags.

This is the "reproducible research routing layer" the PRD allows as a
substitute for OSRM/GraphHopper (PRD §7 component notes) — the engineering
decision to use OSMnx/NetworkX instead of a production routing engine is
documented in README.md §Routing Engine Decision. The tradeoff: slower
per-query search than a contraction-hierarchy engine, but the ETA-aligned,
per-edge, time-dependent risk cost this project needs is a first-class
citizen here rather than something bolted onto OSRM's static edge weights.
"""
from __future__ import annotations

import logging

import networkx as nx
import osmnx as ox

from saferoute.features.geo_utils import centroid_latlon
from saferoute.features.h3_grid import assign_h3_cells

logger = logging.getLogger(__name__)

DEFAULT_WALK_SPEED_KMH = 4.5  # standard pedestrian planning speed


def load_routing_graph(graphml_path, resolution: int, walk_speed_kmh: float = DEFAULT_WALK_SPEED_KMH) -> nx.MultiDiGraph:
    """Load the OSM walk graph and annotate every edge with travel_time_s and h3_cell.

    travel_time_s: length_m / walk_speed, used as the ETA increment during search.
    h3_cell: the H3 cell of the edge midpoint — this is the join key into the
    risk lookup, so a route's cost function can ask "what's the risk of THIS
    edge at the ETA I'd arrive at it."
    """
    graph = ox.load_graphml(graphml_path)
    speed_mps = walk_speed_kmh * 1000 / 3600

    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    mid_lat, mid_lon = centroid_latlon(edges_gdf)

    import pandas as pd

    mid_df = pd.DataFrame({"latitude": mid_lat, "longitude": mid_lon}, index=edges_gdf.index)
    mid_df = assign_h3_cells(mid_df.reset_index(), resolution).set_index(["u", "v", "key"])

    for u, v, k, data in graph.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        data["travel_time_s"] = length_m / speed_mps if speed_mps > 0 else 0.0
        try:
            data["h3_cell"] = mid_df.loc[(u, v, k), "h3_cell"]
        except KeyError:
            data["h3_cell"] = None

    n_missing = sum(1 for _, _, d in graph.edges(data=True) if d.get("h3_cell") is None)
    if n_missing:
        logger.warning("%d/%d edges could not be assigned an H3 cell", n_missing, graph.number_of_edges())

    return graph
