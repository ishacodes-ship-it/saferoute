"""M2 core: ETA-aligned, uncertainty-aware time-dependent shortest path.

cost(edge) = travel_time_s * (1 + lambda*risk_mean + mu*sigma)

risk_mean and sigma are evaluated at the ESTIMATED ARRIVAL TIME at that
specific edge along the specific path being extended — not at the
departure time, and not a static per-location average. This is the
distinction FR-2 requires against a static-risk baseline (see
routing/baselines.py for the ablation that turns this off).

Correctness note: this is a label-setting (Dijkstra-style) search over a
FIFO time-dependent network. FIFO holds here because edge travel time
(travel_time_s) does not itself depend on departure time — only the cost
weight does — so arriving at a node earlier never arrives later downstream.
Given that and non-negative edge costs, standard Dijkstra's optimality
argument generalizes: track BOTH the accumulated cost (the priority-queue
key) and the accumulated real elapsed time (used only to evaluate risk at
each edge) as parallel path state.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from saferoute.routing.risk_lookup import RiskLookup


@dataclass
class RouteResult:
    path: list
    travel_time_s: float
    accumulated_risk: float
    accumulated_sigma: float
    cost: float
    arrival_times: list = field(default_factory=list)
    edge_risks: list = field(default_factory=list)


def edge_cost(travel_time_s: float, risk_mean: float, sigma: float, lam: float, mu: float) -> float:
    return travel_time_s * (1 + lam * risk_mean + mu * sigma)


def time_dependent_shortest_path(
    graph: nx.MultiDiGraph,
    source,
    target,
    departure_time: pd.Timestamp,
    risk_lookup: RiskLookup,
    lam: float,
    mu: float,
    static_risk_table: dict | None = None,
    weather: dict | None = None,
) -> RouteResult | None:
    """
    Args:
        static_risk_table: if provided, {h3_cell: risk_mean} used INSTEAD of
            the ETA-specific risk_lookup — this is what turns this same
            function into the "static-risk baseline" (Baseline B). sigma is
            treated as 0 in static mode (a static score has no time-indexed
            uncertainty concept in this formulation).
    """
    # dist[node] = best cost found so far; state[node] = (arrival_time, risk_sum, sigma_sum)
    dist: dict = {source: 0.0}
    state: dict = {source: (departure_time, 0.0, 0.0)}
    prev: dict = {}
    prev_edge_info: dict = {}

    pq = [(0.0, id(source), source)]
    visited = set()

    while pq:
        d, _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break

        arrival_u, risk_sum_u, sigma_sum_u = state[u]

        for v, edge_data_by_key in graph[u].items():
            for k, data in edge_data_by_key.items():
                travel_time_s = data.get("travel_time_s", 0.0)
                cell = data.get("h3_cell")
                arrival_v = arrival_u + pd.Timedelta(seconds=travel_time_s)

                if static_risk_table is not None:
                    r_mean = static_risk_table.get(cell, 0.0)
                    sigma = 0.0
                else:
                    r = risk_lookup.risk(cell, arrival_u, weather) if cell else {"risk_mean": 0.0, "sigma": 0.0}
                    r_mean, sigma = r["risk_mean"], r["sigma"]

                new_cost = d + edge_cost(travel_time_s, r_mean, sigma, lam, mu)

                if v not in dist or new_cost < dist[v] - 1e-12:
                    dist[v] = new_cost
                    state[v] = (arrival_v, risk_sum_u + r_mean, sigma_sum_u + sigma)
                    prev[v] = u
                    prev_edge_info[v] = {"cell": cell, "travel_time_s": travel_time_s, "risk_mean": r_mean, "sigma": sigma, "arrival": arrival_v}
                    heapq.heappush(pq, (new_cost, id(v), v))

    if target not in dist:
        return None

    path = [target]
    node = target
    edge_risks = []
    while node != source:
        edge_risks.append(prev_edge_info[node])
        node = prev[node]
        path.append(node)
    path.reverse()
    edge_risks.reverse()

    total_time = sum(e["travel_time_s"] for e in edge_risks)
    total_risk = sum(e["risk_mean"] for e in edge_risks)
    total_sigma = sum(e["sigma"] for e in edge_risks)
    arrival_times = [e["arrival"] for e in edge_risks]

    return RouteResult(
        path=path,
        travel_time_s=total_time,
        accumulated_risk=total_risk,
        accumulated_sigma=total_sigma,
        cost=dist[target],
        arrival_times=arrival_times,
        edge_risks=edge_risks,
    )
