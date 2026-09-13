"""The four routing variants compared in the paper (PRD §9).

All four share the exact same search implementation
(eta_cost.time_dependent_shortest_path) and differ only in which knobs are
turned on — this is deliberate: any difference in the resulting routes is
attributable to the risk/uncertainty formulation, not to different search
code paths.

  Baseline A — shortest_path:       lambda=0, mu=0 (time only)
  Baseline B — static_risk:         lambda>0, mu=0, risk from a
                                     time-AVERAGED per-cell table (ignores
                                     the queried arrival time)
  Baseline C — eta_aware_no_uncertainty: lambda>0, mu=0, risk evaluated AT
                                     the estimated arrival time per edge
  Proposed   — eta_aware_with_uncertainty: lambda>0, mu>0, ETA-aligned risk
                                     AND its conformal uncertainty (sigma)
"""
from __future__ import annotations

import pandas as pd

from saferoute.routing.eta_cost import RouteResult, time_dependent_shortest_path
from saferoute.routing.risk_lookup import RiskLookup

REPRESENTATIVE_HOURS = list(range(0, 24, 3))  # sampled every 3h to approximate a full-day average cheaply


def build_static_risk_table(risk_lookup: RiskLookup, cells: list[str], reference_date: pd.Timestamp) -> dict:
    """Time-averaged risk per cell — Baseline B's location-only risk score.

    Averages risk_mean over REPRESENTATIVE_HOURS on a single reference date,
    which folds in weekday temporal features but deliberately discards the
    query's actual arrival time — that discarding IS what makes this the
    static baseline.
    """
    table = {}
    for cell in cells:
        vals = [
            risk_lookup.risk(cell, reference_date.replace(hour=h))["risk_mean"]
            for h in REPRESENTATIVE_HOURS
        ]
        table[cell] = sum(vals) / len(vals)
    return table


def route_shortest_path(graph, source, target, departure_time, risk_lookup) -> RouteResult | None:
    return time_dependent_shortest_path(graph, source, target, departure_time, risk_lookup, lam=0.0, mu=0.0)


def route_static_risk(graph, source, target, departure_time, risk_lookup, static_risk_table, lam) -> RouteResult | None:
    return time_dependent_shortest_path(
        graph, source, target, departure_time, risk_lookup, lam=lam, mu=0.0, static_risk_table=static_risk_table
    )


def route_eta_aware_no_uncertainty(graph, source, target, departure_time, risk_lookup, lam) -> RouteResult | None:
    return time_dependent_shortest_path(graph, source, target, departure_time, risk_lookup, lam=lam, mu=0.0)


def route_proposed(graph, source, target, departure_time, risk_lookup, lam, mu) -> RouteResult | None:
    return time_dependent_shortest_path(graph, source, target, departure_time, risk_lookup, lam=lam, mu=mu)


def route_all_variants(graph, source, target, departure_time, risk_lookup, static_risk_table, lam, mu) -> dict:
    """Run all four variants for one OD pair — the unit of work for the ablation script."""
    return {
        "shortest_path": route_shortest_path(graph, source, target, departure_time, risk_lookup),
        "static_risk": route_static_risk(graph, source, target, departure_time, risk_lookup, static_risk_table, lam),
        "eta_aware_no_uncertainty": route_eta_aware_no_uncertainty(graph, source, target, departure_time, risk_lookup, lam),
        "proposed": route_proposed(graph, source, target, departure_time, risk_lookup, lam, mu),
    }
