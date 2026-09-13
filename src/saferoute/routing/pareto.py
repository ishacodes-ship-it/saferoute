"""FR-3: 2-3 Pareto-alternative routes (fast/balanced/cautious) with a
programmatic non-domination check.

Route generation: run the proposed cost function at several lambda values
(config: routing.risk_aversion_lambda) — low lambda biases toward "fast",
high lambda toward "cautious". This is a standard weighted-sum scan for
approximating a Pareto frontier over (time, risk); it can miss frontier
points in non-convex regions, which is a documented limitation, not a
silent gap (see README §Routing Engine Decision).

Non-domination: route A dominates route B if A.time <= B.time AND
A.accumulated_risk <= B.accumulated_risk with at least one strict
inequality. Only non-dominated routes are returned — this is checked
programmatically (tests/test_pareto.py), not asserted by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

from saferoute.routing.eta_cost import RouteResult, time_dependent_shortest_path
from saferoute.routing.risk_lookup import RiskLookup

LABELS_BY_LAMBDA_RANK = {0: "fast", 1: "balanced", 2: "cautious"}


@dataclass
class ParetoRoute:
    route_id: str
    label: str
    lam: float
    mu: float
    result: RouteResult


def dominates(a: RouteResult, b: RouteResult) -> bool:
    """True if route `a` dominates route `b` on (time, risk)."""
    time_le = a.travel_time_s <= b.travel_time_s
    risk_le = a.accumulated_risk <= b.accumulated_risk
    strictly_better = a.travel_time_s < b.travel_time_s or a.accumulated_risk < b.accumulated_risk
    return time_le and risk_le and strictly_better


def filter_non_dominated(routes: list[ParetoRoute]) -> list[ParetoRoute]:
    kept = []
    for i, r in enumerate(routes):
        dominated = any(dominates(other.result, r.result) for j, other in enumerate(routes) if j != i)
        if not dominated:
            kept.append(r)
    return kept


def generate_pareto_routes(
    graph, source, target, departure_time, risk_lookup: RiskLookup, lambda_grid: list[float], mu: float
) -> list[ParetoRoute]:
    """Generate one route per lambda in lambda_grid, then drop dominated ones.

    lambda_grid is sorted ascending; the lowest lambda tends toward "fast",
    the highest toward "cautious" — but the label is assigned by RANK among
    the SURVIVING non-dominated routes, not by lambda value directly, since
    a dominated route is dropped and rank shifts.
    """
    candidates = []
    for lam in sorted(lambda_grid):
        result = time_dependent_shortest_path(graph, source, target, departure_time, risk_lookup, lam=lam, mu=mu)
        if result is not None:
            candidates.append(ParetoRoute(route_id=f"lam_{lam}", label="unlabeled", lam=lam, mu=mu, result=result))

    non_dominated = filter_non_dominated(candidates)
    non_dominated.sort(key=lambda r: r.result.travel_time_s)

    for i, r in enumerate(non_dominated):
        r.label = LABELS_BY_LAMBDA_RANK.get(i, f"alt_{i}")

    return non_dominated
