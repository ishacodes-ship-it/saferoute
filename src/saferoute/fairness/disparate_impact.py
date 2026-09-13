"""Phase 2 / FR-6: demographic disparate-impact audit of routing decisions.

Metric definition (documented — no target is set before running this):

For a sample of OD pairs, compute the PROPOSED route for each. For each
census tract t, define:

    selection_rate(t) = (# proposed routes whose path intersects tract t)
                         / (# total proposed routes)

i.e., how often SafeRoute's routing sends a pedestrian's path through tract
t, at the same OD-pair sample used throughout the benchmark. Group tracts
into quantile bins (default: quartiles) of a demographic variable (percent
non-white population, from the ACS ingest — the variable most directly
tied to the historical-redlining concern this audit exists to catch).
Compute the mean selection_rate per demographic group, then the four-fifths
ratio:

    ratio = min(group_selection_rate) / max(group_selection_rate)

per the standard EEOC four-fifths convention (a ratio < 0.8 is the
conventional adverse-impact threshold; here it is 0.8-1.25 per config,
i.e. checked symmetrically since routing "away from" a tract is the
directional concern named in PRD §15, not just "toward").

No target ratio is assumed correct or incorrect ahead of the computation
(PRD §12): this module returns whatever the data shows.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


def route_to_linestring(route_result, node_coords: dict) -> LineString:
    """node_coords: {node_id: (lon, lat)}."""
    coords = [node_coords[n] for n in route_result.path if n in node_coords]
    return LineString(coords) if len(coords) >= 2 else None


def compute_tract_selection_rates(route_linestrings: list, tracts_gdf: gpd.GeoDataFrame) -> pd.Series:
    """selection_rate(t) for every tract, given a list of route LineStrings (one per OD pair)."""
    counts = pd.Series(0, index=tracts_gdf["GEOID"], dtype=float)
    n_routes = len(route_linestrings)
    if n_routes == 0:
        return counts

    sindex = tracts_gdf.sindex
    for line in route_linestrings:
        if line is None:
            continue
        candidate_idx = list(sindex.query(line, predicate="intersects"))
        touched_geoids = tracts_gdf.iloc[candidate_idx]["GEOID"].unique()
        counts.loc[touched_geoids] += 1

    return counts / n_routes


def assign_demographic_quantiles(tracts_gdf: gpd.GeoDataFrame, demographic_col: str, n_groups: int = 4) -> pd.Series:
    """Quantile-bin tracts by a demographic variable. Returns a Series indexed by GEOID."""
    values = tracts_gdf.set_index("GEOID")[demographic_col]
    return pd.qcut(values, q=n_groups, labels=[f"q{i+1}" for i in range(n_groups)], duplicates="drop")


def four_fifths_audit(
    selection_rates: pd.Series, demographic_groups: pd.Series, lower: float = 0.8, upper: float = 1.25
) -> dict:
    df = pd.DataFrame({"selection_rate": selection_rates, "group": demographic_groups}).dropna()
    group_means = df.groupby("group", observed=True)["selection_rate"].mean()

    if len(group_means) < 2 or group_means.max() == 0:
        return {"ratio": None, "group_means": group_means.to_dict(), "flagged": None, "reason": "insufficient data"}

    ratio = float(group_means.min() / group_means.max())
    flagged = not (lower <= ratio <= upper)

    return {
        "ratio": ratio,
        "group_means": group_means.to_dict(),
        "lower_threshold": lower,
        "upper_threshold": upper,
        "flagged": flagged,
    }


def bootstrap_four_fifths_ci(
    route_linestrings: list, tracts_gdf: gpd.GeoDataFrame, demographic_col: str, n_bootstrap: int, seed: int, n_groups: int = 4
) -> dict:
    """Bootstrap CI on the four-fifths ratio by resampling OD-pair routes."""
    rng = np.random.default_rng(seed)
    groups = assign_demographic_quantiles(tracts_gdf, demographic_col, n_groups)

    point = four_fifths_audit(compute_tract_selection_rates(route_linestrings, tracts_gdf), groups)
    if point["ratio"] is None:
        return point

    n = len(route_linestrings)
    boot_ratios = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = [route_linestrings[i] for i in idx]
        rates = compute_tract_selection_rates(resampled, tracts_gdf)
        res = four_fifths_audit(rates, groups)
        if res["ratio"] is not None:
            boot_ratios.append(res["ratio"])

    if not boot_ratios:
        point["ci_low"], point["ci_high"] = None, None
        return point

    point["ci_low"] = float(np.quantile(boot_ratios, 0.025))
    point["ci_high"] = float(np.quantile(boot_ratios, 0.975))
    point["n_bootstrap"] = len(boot_ratios)
    return point
