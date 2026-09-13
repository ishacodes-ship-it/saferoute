import pandas as pd

from saferoute.benchmark.counterfactual import bootstrap_avoidance_ci, route_avoids_incident, route_within_detour_budget
from saferoute.routing.eta_cost import RouteResult


def _route(cells, time_s):
    edge_risks = [{"cell": c, "travel_time_s": 10, "risk_mean": 0.0, "sigma": 0.0} for c in cells]
    return RouteResult(path=list(range(len(cells) + 1)), travel_time_s=time_s, accumulated_risk=0.0, accumulated_sigma=0.0, cost=0.0, edge_risks=edge_risks)


def test_route_avoids_incident_true_when_cell_not_visited():
    route = _route(["cell_far_1", "cell_far_2"], time_s=100)
    assert route_avoids_incident(route, incident_cell="892830828a7ffff")  # arbitrary real-looking cell id, won't collide


def test_route_does_not_avoid_when_it_passes_through_incident_cell():
    import h3

    incident_cell = h3.latlng_to_cell(41.8781, -87.6298, 9)
    route = _route([incident_cell], time_s=100)
    assert not route_avoids_incident(route, incident_cell)


def test_route_within_detour_budget():
    route = _route(["a"], time_s=110)
    assert route_within_detour_budget(route, shortest_time_s=100, budget_fraction=0.15)
    assert not route_within_detour_budget(route, shortest_time_s=100, budget_fraction=0.05)


def test_bootstrap_avoidance_ci_reasonable_bounds():
    df = pd.DataFrame({
        "method": ["proposed"] * 20,
        "detour_budget": [0.1] * 20,
        "avoided": [True] * 15 + [False] * 5,
    })
    result = bootstrap_avoidance_ci(df, method="proposed", detour_budget=0.1, n_bootstrap=500, confidence_level=0.95, seed=0)
    assert result["avoidance_rate"] == 0.75
    assert 0.0 <= result["ci_low"] <= result["avoidance_rate"] <= result["ci_high"] <= 1.0


def test_bootstrap_avoidance_ci_reproducible():
    df = pd.DataFrame({"method": ["m"] * 10, "detour_budget": [0.0] * 10, "avoided": [True] * 6 + [False] * 4})
    r1 = bootstrap_avoidance_ci(df, "m", 0.0, n_bootstrap=200, confidence_level=0.9, seed=5)
    r2 = bootstrap_avoidance_ci(df, "m", 0.0, n_bootstrap=200, confidence_level=0.9, seed=5)
    assert r1 == r2


def test_bootstrap_avoidance_ci_empty_subset_handled():
    df = pd.DataFrame({"method": ["other"], "detour_budget": [0.0], "avoided": [True]})
    result = bootstrap_avoidance_ci(df, method="proposed", detour_budget=0.1, n_bootstrap=100, confidence_level=0.9, seed=0)
    assert result["n"] == 0
    assert result["avoidance_rate"] is None
