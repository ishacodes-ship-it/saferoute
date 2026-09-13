import networkx as nx
import pandas as pd
import pytest

from saferoute.routing.eta_cost import edge_cost, time_dependent_shortest_path
from saferoute.routing.pareto import ParetoRoute, dominates, filter_non_dominated
from saferoute.routing.eta_cost import RouteResult


class StubRiskLookup:
    """Deterministic risk source for testing: risk depends only on (cell, hour)."""

    def __init__(self, risk_by_cell_hour: dict, sigma: float = 0.1):
        self.risk_by_cell_hour = risk_by_cell_hour
        self.sigma = sigma
        self.queried_hours = []

    def risk(self, cell, arrival_time, weather=None):
        self.queried_hours.append((cell, arrival_time.hour))
        return {"risk_mean": self.risk_by_cell_hour.get((cell, arrival_time.hour), 0.0), "sigma": self.sigma}


def _linear_graph():
    """0 -> 1 -> 2 (safe path) and 0 -> 3 -> 2 (risky-at-night path)."""
    g = nx.MultiDiGraph()
    g.add_edge(0, 1, key=0, travel_time_s=100, h3_cell="safe_a")
    g.add_edge(1, 2, key=0, travel_time_s=100, h3_cell="safe_b")
    g.add_edge(0, 3, key=0, travel_time_s=90, h3_cell="risky_a")
    g.add_edge(3, 2, key=0, travel_time_s=90, h3_cell="risky_b")
    return g


def test_edge_cost_formula():
    # cost = time * (1 + lambda*risk + mu*sigma)
    assert edge_cost(100, risk_mean=0.5, sigma=0.2, lam=1.0, mu=1.0) == pytest.approx(100 * (1 + 0.5 + 0.2))
    assert edge_cost(100, risk_mean=0.5, sigma=0.2, lam=0.0, mu=0.0) == 100.0


def test_shortest_path_ignores_risk_when_lambda_and_mu_zero():
    g = _linear_graph()
    lookup = StubRiskLookup({("risky_a", 8): 0.9, ("risky_b", 8): 0.9})
    result = time_dependent_shortest_path(g, 0, 2, pd.Timestamp("2023-01-01 08:00"), lookup, lam=0.0, mu=0.0)
    # risky path is shorter in time (180s vs 200s) -- with lambda=0 it must be chosen despite high risk
    assert result.path == [0, 3, 2]


def test_risk_averse_routing_avoids_high_risk_path():
    g = _linear_graph()
    lookup = StubRiskLookup({("risky_a", 8): 5.0, ("risky_b", 8): 5.0, ("safe_a", 8): 0.0, ("safe_b", 8): 0.0})
    result = time_dependent_shortest_path(g, 0, 2, pd.Timestamp("2023-01-01 08:00"), lookup, lam=1.0, mu=0.0)
    assert result.path == [0, 1, 2]


def test_eta_alignment_risk_queried_at_estimated_arrival_hour():
    """The risk lookup for the SECOND edge must be queried at departure_time + first edge's travel time
    (i.e. the ETA to node 1), not at the original departure time -- this is the ETA-alignment property
    FR-2 requires: risk is evaluated at when you'd actually be at that segment, not when you left home."""
    g = nx.MultiDiGraph()
    g.add_edge(0, 1, key=0, travel_time_s=3600, h3_cell="cell_a")  # exactly 1 hour
    g.add_edge(1, 2, key=0, travel_time_s=60, h3_cell="cell_b")

    lookup = StubRiskLookup({})
    time_dependent_shortest_path(g, 0, 2, pd.Timestamp("2023-01-01 08:00:00"), lookup, lam=1.0, mu=0.0)

    hours_queried_for_cell_a = [h for c, h in lookup.queried_hours if c == "cell_a"]
    hours_queried_for_cell_b = [h for c, h in lookup.queried_hours if c == "cell_b"]
    assert hours_queried_for_cell_a == [8]   # departure time
    assert hours_queried_for_cell_b == [9]   # ETA to node 1, one hour later -- NOT still 8


def test_static_risk_table_overrides_dynamic_lookup():
    g = _linear_graph()
    lookup = StubRiskLookup({("risky_a", 8): 0.0, ("risky_b", 8): 0.0})  # dynamic says risky path is safe
    static_table = {"risky_a": 10.0, "risky_b": 10.0, "safe_a": 0.0, "safe_b": 0.0}  # static table says opposite
    result = time_dependent_shortest_path(
        g, 0, 2, pd.Timestamp("2023-01-01 08:00"), lookup, lam=1.0, mu=0.0, static_risk_table=static_table
    )
    assert result.path == [0, 1, 2]  # must follow the static table, not the (ignored) dynamic lookup


def test_no_path_returns_none():
    g = nx.MultiDiGraph()
    g.add_node(0)
    g.add_node(1)
    lookup = StubRiskLookup({})
    result = time_dependent_shortest_path(g, 0, 1, pd.Timestamp("2023-01-01"), lookup, lam=0.0, mu=0.0)
    assert result is None


def _route(time_s, risk):
    return RouteResult(path=[], travel_time_s=time_s, accumulated_risk=risk, accumulated_sigma=0.0, cost=0.0)


def test_dominates_strict_improvement_required():
    a = ParetoRoute("a", "x", 0, 0, _route(100, 1.0))
    b = ParetoRoute("b", "x", 0, 0, _route(100, 1.0))
    assert not dominates(a.result, b.result)  # identical routes: neither dominates


def test_dominates_basic():
    fast = ParetoRoute("fast", "x", 0, 0, _route(100, 5.0))
    cautious = ParetoRoute("cautious", "x", 0, 0, _route(200, 1.0))
    assert not dominates(fast.result, cautious.result)
    assert not dominates(cautious.result, fast.result)


def test_filter_non_dominated_removes_strictly_worse_route():
    good = ParetoRoute("good", "x", 0, 0, _route(100, 1.0))
    bad = ParetoRoute("bad", "x", 0, 0, _route(150, 2.0))  # worse on both axes
    kept = filter_non_dominated([good, bad])
    assert [r.route_id for r in kept] == ["good"]


def test_filter_non_dominated_keeps_genuine_tradeoffs():
    fast = ParetoRoute("fast", "x", 0, 0, _route(100, 5.0))
    cautious = ParetoRoute("cautious", "x", 0, 0, _route(200, 1.0))
    kept = filter_non_dominated([fast, cautious])
    assert {r.route_id for r in kept} == {"fast", "cautious"}
