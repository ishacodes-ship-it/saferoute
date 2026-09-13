"""M4: counterfactual held-out-incident benchmark (PRD §12/FR-5).

Answers: if a traveller had followed SafeRoute instead of a baseline route,
how much incident exposure would have been avoided, and at what detour
cost?

METHODOLOGY (documented — this is a research-engineering choice, not
fabricated data):

Real pedestrian trip origin/destination data does not exist in any open
dataset for Chicago. To evaluate against REAL held-out incidents (which do
exist — this uses the actual test-split crime records, never synthetic
labels), synthetic OD pairs are constructed around each held-out incident:
starting from the incident's nearest network node, a bounded random walk in
each of two roughly-opposite directions produces an origin and destination
such that a time-minimizing path is likely to pass near the incident. This
is a standard proxy technique in counterfactual routing-safety evaluation
(there being no ground-truth trip log to replay); it is NOT the same as
having real trip data, and this limitation belongs in the paper's
Limitations section.

Avoidance definition: a route "avoids" an incident if the route's edges do
not pass through the incident's H3 cell or its immediate 1-ring neighbors
(a small spatial buffer, since exact segment-level incident geocoding
precision is itself limited).

Detour-budget-constrained routing: for a target detour budget (fraction of
extra travel time over the shortest path), the lambda in
routing.eta_cost's cost function is swept (config: routing.risk_aversion_lambda)
and the highest-lambda route still within budget is selected — this
approximates a constrained shortest path via a weighted-sum scan, which can
miss the true constrained optimum in non-convex cases (documented
limitation, consistent with the Pareto-generation approximation in
routing/pareto.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import h3
import networkx as nx
import numpy as np
import pandas as pd

from saferoute.features.h3_grid import cell_centroid
from saferoute.routing.baselines import build_static_risk_table, route_shortest_path, route_static_risk, route_eta_aware_no_uncertainty, route_proposed
from saferoute.routing.eta_cost import RouteResult, time_dependent_shortest_path
from saferoute.routing.risk_lookup import RiskLookup

logger = logging.getLogger(__name__)


@dataclass
class ODPair:
    incident_cell: str
    incident_time: pd.Timestamp
    source: object
    target: object


def _nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float):
    import osmnx as ox

    return ox.nearest_nodes(graph, lon, lat)


DEFAULT_MIN_OD_DISTANCE_M = 900.0  # see module docstring: must be several times the incident
                                    # buffer's ~500m diameter, or even a 30% detour budget can't
                                    # geometrically route around it (a pilot-run finding, not a guess)
MAX_WALK_HOPS = 200  # hard cap so a dead-end/disconnected pocket can't loop forever


def _walk_until_far(start_node, min_distance_m, incident_lat, incident_lon, graph, rng):
    """Random-walk from `start_node` until at least `min_distance_m` from the
    incident, or MAX_WALK_HOPS is exhausted (best-effort node returned either way,
    checked by the caller)."""
    node = start_node
    for _ in range(MAX_WALK_HOPS):
        lat, lon = graph.nodes[node]["y"], graph.nodes[node]["x"]
        if _haversine_scalar_m(incident_lat, incident_lon, lat, lon) >= min_distance_m:
            return node
        nxt = list(graph.successors(node))
        if not nxt:
            break
        node = rng.choice(nxt)
    return node


def _haversine_scalar_m(lat1, lon1, lat2, lon2) -> float:
    from math import asin, cos, radians, sin, sqrt

    lat1, lon1, lat2, lon2 = radians(lat1), radians(lon1), radians(lat2), radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6_371_000 * asin(sqrt(a))


def generate_od_pairs_near_incidents(
    incidents_df: pd.DataFrame, graph: nx.MultiDiGraph, n_pairs: int, walk_hops: int, seed: int,
    min_od_distance_m: float = DEFAULT_MIN_OD_DISTANCE_M,
) -> list[ODPair]:
    """See module docstring's METHODOLOGY note for what this does and why.

    `incidents_df` is expected to be rows from the dense (h3_cell, time_bin)
    feature table (features/labels.py), which does NOT retain individual
    incidents' raw lat/lon -- multiple incidents in the same cell-hour are
    already collapsed into `incident_count` by that point. The H3 cell's
    centroid (`h3_grid.cell_centroid`) is therefore used as the incident
    location; this is the finest spatial granularity the dense table
    preserves, not a full-precision incident coordinate.

    `min_od_distance_m`: each of source and target is walked away from the
    incident until it is at least this far away (in a random direction),
    rather than walking a FIXED number of hops (`walk_hops` is accepted for
    API compatibility but no longer drives the walk -- see the pilot-run
    finding below). A fixed-hop walk was tried first and produced OD pairs
    only 30-110m apart in this project's dense downtown pilot bbox (street
    segments there run ~15-25m), which is smaller than an H3 res-9 cell
    (~174m edge length) -- every route search then had no way to route
    AROUND the incident's cell at all, and every method's avoidance rate was
    0% at every detour budget in the pilot run. That was a benchmark
    methodology bug, not a finding about the routing method, caught only by
    actually running the benchmark on real data (see
    results/logs/pilot_run_notes.md).

    A second pilot-run finding after that fix: raising min_od_distance_m to
    250m still produced 0% avoidance at every detour budget up to 30%. The
    incident's avoidance buffer (route_avoids_incident's 1-ring, ~500m
    diameter at H3 res 9) is a roughly FIXED absolute size, while the detour
    budget is a PERCENTAGE of a trip's own direct time -- for a 250-500m
    trip, routing around a 500m obstacle can easily need well over 30% extra
    distance, so no percentage-based budget in that range can ever succeed
    geometrically, regardless of how good the risk model or routing cost
    function are. Raising the default to 900m (several times the buffer
    diameter) is what makes a 0-30% detour budget a meaningful test of the
    ROUTING METHOD rather than an artifact of trip length. If the full-scale
    run still shows near-0% avoidance at low OD-pair distances, that
    distance-dependence should be reported explicitly, not folded into a
    single aggregate number.
    """
    rng = np.random.default_rng(seed)
    sample = incidents_df.sample(n=min(n_pairs, len(incidents_df)), random_state=seed)

    pairs = []
    for _, row in sample.iterrows():
        lat, lon = cell_centroid(row["h3_cell"])
        try:
            center_node = _nearest_node(graph, lat, lon)
        except Exception:
            continue

        neighbors = list(graph.successors(center_node)) + list(graph.predecessors(center_node))
        if not neighbors:
            continue

        source = _walk_until_far(rng.choice(neighbors), min_od_distance_m, lat, lon, graph, rng)
        target = _walk_until_far(rng.choice(neighbors), min_od_distance_m, lat, lon, graph, rng)

        source_ok = _haversine_scalar_m(lat, lon, graph.nodes[source]["y"], graph.nodes[source]["x"]) >= min_od_distance_m
        target_ok = _haversine_scalar_m(lat, lon, graph.nodes[target]["y"], graph.nodes[target]["x"]) >= min_od_distance_m
        if source == target or not (source_ok and target_ok):
            continue

        pairs.append(
            ODPair(
                incident_cell=row["h3_cell"],
                incident_time=pd.Timestamp(row["time_bin"]),
                source=source,
                target=target,
            )
        )
    logger.info("Generated %d/%d OD pairs near held-out incidents", len(pairs), len(sample))
    return pairs


def route_avoids_incident(route: RouteResult, incident_cell: str) -> bool:
    buffer_cells = set(h3.grid_disk(incident_cell, 1))
    route_cells = {e["cell"] for e in route.edge_risks if e["cell"]}
    return len(route_cells & buffer_cells) == 0


def route_within_detour_budget(route: RouteResult, shortest_time_s: float, budget_fraction: float) -> bool:
    if shortest_time_s <= 0:
        return True
    return route.travel_time_s <= shortest_time_s * (1 + budget_fraction)


def select_route_within_budget(
    graph, source, target, departure_time, risk_lookup, lambda_grid: list[float], mu: float, shortest_time_s: float, budget_fraction: float
) -> RouteResult | None:
    """Highest-lambda route (most risk-averse) still within the detour budget — see module docstring."""
    best = None
    for lam in sorted(lambda_grid):
        r = time_dependent_shortest_path(graph, source, target, departure_time, risk_lookup, lam=lam, mu=mu)
        if r is not None and route_within_detour_budget(r, shortest_time_s, budget_fraction):
            best = r  # keep overwriting; loop is ascending so the last valid one is highest-lambda
    return best


def run_benchmark(
    graph,
    od_pairs: list[ODPair],
    risk_lookup: RiskLookup,
    static_risk_table: dict,
    lambda_grid: list[float],
    default_mu: float,
    detour_budgets: list[float],
) -> pd.DataFrame:
    """Run all four methods across all OD pairs and detour budgets.

    Returns a long-form DataFrame: one row per (od_pair, method, detour_budget)
    with columns [avoided, travel_time_s, accumulated_risk, detour_pct].
    """
    rows = []
    for od in od_pairs:
        shortest = route_shortest_path(graph, od.source, od.target, od.incident_time, risk_lookup)
        if shortest is None:
            continue
        shortest_time = shortest.travel_time_s

        rows.append({
            "od_source": od.source, "od_target": od.target, "method": "shortest_path", "detour_budget": 0.0,
            "avoided": route_avoids_incident(shortest, od.incident_cell),
            "travel_time_s": shortest.travel_time_s, "accumulated_risk": shortest.accumulated_risk,
            "detour_pct": 0.0,
        })

        for budget in detour_budgets:
            # Static-risk variant sweeps lambda with route_static_risk directly (sigma is always 0 in static mode).
            best_static = None
            for lam in sorted(lambda_grid):
                r = route_static_risk(graph, od.source, od.target, od.incident_time, risk_lookup, static_risk_table, lam)
                if r is not None and route_within_detour_budget(r, shortest_time, budget):
                    best_static = r
            eta_route = select_route_within_budget(
                graph, od.source, od.target, od.incident_time, risk_lookup, lambda_grid, 0.0, shortest_time, budget
            )
            proposed_route = select_route_within_budget(
                graph, od.source, od.target, od.incident_time, risk_lookup, lambda_grid, default_mu, shortest_time, budget
            )

            for method_name, route in [
                ("static_risk", best_static),
                ("eta_aware_no_uncertainty", eta_route),
                ("proposed", proposed_route),
            ]:
                if route is None:
                    continue
                rows.append({
                    "od_source": od.source, "od_target": od.target, "method": method_name, "detour_budget": budget,
                    "avoided": route_avoids_incident(route, od.incident_cell),
                    "travel_time_s": route.travel_time_s, "accumulated_risk": route.accumulated_risk,
                    "detour_pct": (route.travel_time_s - shortest_time) / shortest_time if shortest_time > 0 else 0.0,
                })

    return pd.DataFrame(rows)


def bootstrap_avoidance_ci(results_df: pd.DataFrame, method: str, detour_budget: float, n_bootstrap: int, confidence_level: float, seed: int) -> dict:
    """Bootstrap CI for the avoidance rate of one (method, detour_budget) cell."""
    subset = results_df[(results_df["method"] == method) & (results_df["detour_budget"] == detour_budget)]
    if len(subset) == 0:
        return {"method": method, "detour_budget": detour_budget, "n": 0, "avoidance_rate": None, "ci_low": None, "ci_high": None}

    rng = np.random.default_rng(seed)
    avoided = subset["avoided"].to_numpy().astype(float)
    boot_means = np.array([rng.choice(avoided, size=len(avoided), replace=True).mean() for _ in range(n_bootstrap)])
    alpha = 1 - confidence_level

    return {
        "method": method,
        "detour_budget": detour_budget,
        "n": len(subset),
        "avoidance_rate": float(avoided.mean()),
        "ci_low": float(np.quantile(boot_means, alpha / 2)),
        "ci_high": float(np.quantile(boot_means, 1 - alpha / 2)),
    }
