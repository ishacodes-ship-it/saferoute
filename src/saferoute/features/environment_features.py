"""Environmental features per H3 cell, derived from OSM tags.

These five features are the ONLY ones the explanation whitelist (FR-4,
config/base.yaml: explainability.feature_whitelist) is allowed to reference.
They double as risk-model inputs, so the whitelist constraint on the
explanation layer is enforced by construction: these are also the only
environment features the model itself is trained on (see
src/saferoute/features/build_features.py).

IMPORTANT — proxy documentation (do not remove; this is a research-honesty
requirement, not a style note):

- lighting: OSM `lit=yes/no/*` tag on ways. Real but sparse — most Chicago
  ways carry no `lit` tag at all. Untagged ways are treated as unknown
  (imputed as the cell's local mean), NOT as unlit. This is a genuine data
  coverage gap and must be reported as a limitation, not silently patched.
- footfall: no free, scriptable, real-time pedestrian-count feed exists for
  MVP scope. This is a STATIC PROXY = OSM street-intersection density per
  cell (count of graph nodes with degree >= 3), a standard pedestrian-activity
  proxy in urban-form literature (more intersections -> more through-traffic
  opportunity). It is NOT a measured footfall count and must never be
  described in output text as "measured footfall."
- open_businesses: static OSM POI density of commercial `shop=*` /
  `amenity=*` (restaurant, cafe, bar, retail, ...) tags. "Open" is a proxy —
  OSM `opening_hours` coverage is too sparse in Chicago to reliably compute
  live open/closed state, so this feature is commercial-POI DENSITY, not a
  real-time open/closed signal. Deliberately a different signal from the
  footfall proxy above (street form vs. commercial land use) even though
  both are OSM-derived. Documented, not hidden.
- cctv: OSM `man_made=surveillance` node count per cell. Real tag, very
  sparse coverage (crowdsourced, not a camera registry) — treat as a weak,
  noisy signal, not ground truth.
- distance_to_help: haversine distance (meters) from the cell centroid to
  the nearest OSM `amenity=police` or `amenity=hospital` node. Real
  distances, but "help" here means these two OSM categories only.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from saferoute.features.geo_utils import centroid_latlon
from saferoute.features.h3_grid import cell_centroid

logger = logging.getLogger(__name__)

BUSINESS_TAGS = {
    "shop": True,
    "amenity": ["restaurant", "cafe", "bar", "fast_food", "pub", "marketplace", "pharmacy", "convenience"],
}
CCTV_TAGS = {"man_made": "surveillance"}
HELP_TAGS = {"amenity": ["police", "hospital"]}

EARTH_RADIUS_M = 6_371_000


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def compute_lighting_by_cell(graph, cells: list[str], resolution: int) -> pd.Series:
    """Fraction of edge length tagged lit=yes among cells' incident OSM ways, per H3 cell."""
    from saferoute.features.h3_grid import assign_h3_cells

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    edges["mid_lat"], edges["mid_lon"] = centroid_latlon(edges)

    def lit_flag(v):
        if isinstance(v, list):
            v = v[0] if v else None
        return 1.0 if v == "yes" else (0.0 if v == "no" else np.nan)

    edges["lit"] = edges.get("lit", pd.Series(index=edges.index, dtype=object)).apply(lit_flag)
    edges = assign_h3_cells(edges.rename(columns={"mid_lat": "latitude", "mid_lon": "longitude"}), resolution)

    grouped = edges.groupby("h3_cell")["lit"].mean()
    result = pd.Series(index=cells, dtype=float)
    result.update(grouped)
    # grouped.mean() is itself NaN when NO edge in the whole graph carries a
    # `lit` tag (a real, observed case for small bboxes) -- .notna().any()
    # guards against silently propagating that NaN into every cell's value.
    global_mean = grouped.mean() if grouped.notna().any() else 0.5
    result = result.fillna(global_mean)
    result.name = "lighting"
    return result


def _poi_count_by_cell(bbox: list[float], tags: dict, cells: list[str], resolution: int) -> pd.Series:
    from saferoute.features.h3_grid import assign_h3_cells

    south, west, north, east = bbox
    try:
        pois = ox.features_from_bbox(north, south, east, west, tags=tags)
    except Exception as exc:  # OSM can return empty-result errors for rare tag combos
        logger.warning("POI fetch failed for tags=%s: %s", tags, exc)
        return pd.Series(0, index=cells, name="poi_count")

    pois = pois[pois.geometry.notnull()].copy()
    pois["latitude"], pois["longitude"] = centroid_latlon(pois)
    pois = assign_h3_cells(pois, resolution)
    counts = pois.groupby("h3_cell").size()
    result = pd.Series(0, index=cells, dtype=float)
    result.update(counts)
    result.name = "poi_count"
    return result


def compute_footfall_proxy_by_cell(graph, cells: list[str], resolution: int) -> pd.Series:
    """Street-intersection density per cell — proxy for pedestrian through-traffic (see module docstring)."""
    from saferoute.features.h3_grid import assign_h3_cells

    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False).reset_index()
    degrees = dict(graph.degree())
    nodes["degree"] = nodes["osmid"].map(degrees)
    intersections = nodes[nodes["degree"] >= 3].copy()
    intersections = intersections.rename(columns={"y": "latitude", "x": "longitude"})
    intersections = assign_h3_cells(intersections, resolution)

    counts = intersections.groupby("h3_cell").size()
    result = pd.Series(0, index=cells, dtype=float)
    result.update(counts)
    result.name = "footfall_proxy"
    return result


def compute_open_businesses_by_cell(bbox: list[float], cells: list[str], resolution: int) -> pd.Series:
    s = _poi_count_by_cell(bbox, BUSINESS_TAGS, cells, resolution)
    s.name = "open_businesses"
    return s


def compute_cctv_by_cell(bbox: list[float], cells: list[str], resolution: int) -> pd.Series:
    s = _poi_count_by_cell(bbox, CCTV_TAGS, cells, resolution)
    s.name = "cctv"
    return s


def compute_distance_to_help_by_cell(bbox: list[float], cells: list[str], resolution: int) -> pd.Series:
    south, west, north, east = bbox
    try:
        help_pois = ox.features_from_bbox(north, south, east, west, tags=HELP_TAGS)
    except Exception as exc:
        logger.warning("Help-POI fetch failed: %s", exc)
        return pd.Series(np.nan, index=cells, name="distance_to_help")

    help_pois = help_pois[help_pois.geometry.notnull()].copy()
    help_lats, help_lons = centroid_latlon(help_pois)

    dists = {}
    for cell in cells:
        lat, lon = cell_centroid(cell)
        if len(help_lats) == 0:
            dists[cell] = np.nan
            continue
        d = _haversine_m(lat, lon, help_lats, help_lons)
        dists[cell] = float(d.min())

    result = pd.Series(dists, name="distance_to_help")
    result = result.fillna(result.median())
    return result


def build_environment_features(graph, bbox: list[float], cells: list[str], resolution: int) -> pd.DataFrame:
    """Build the full whitelisted environment feature table, one row per H3 cell."""
    lighting = compute_lighting_by_cell(graph, cells, resolution)
    footfall = compute_footfall_proxy_by_cell(graph, cells, resolution)
    businesses = compute_open_businesses_by_cell(bbox, cells, resolution)
    cctv = compute_cctv_by_cell(bbox, cells, resolution)
    help_dist = compute_distance_to_help_by_cell(bbox, cells, resolution)

    df = pd.concat([lighting, footfall, businesses, cctv, help_dist], axis=1)
    df.index.name = "h3_cell"
    return df.reset_index()
