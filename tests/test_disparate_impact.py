import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

from saferoute.fairness.disparate_impact import (
    assign_demographic_quantiles,
    compute_tract_selection_rates,
    four_fifths_audit,
)


def _synthetic_tracts():
    # 4 tracts in a row, each a 1x1 box, with an increasing demographic value.
    geoms = [box(i, 0, i + 1, 1) for i in range(4)]
    return gpd.GeoDataFrame({"GEOID": ["t0", "t1", "t2", "t3"], "pct_nonwhite": [0.1, 0.3, 0.6, 0.9]}, geometry=geoms)


def test_selection_rate_counts_routes_through_each_tract():
    tracts = _synthetic_tracts()
    routes = [LineString([(0.5, 0.5), (1.5, 0.5)]), LineString([(0.5, 0.5), (0.5, 0.5)])]
    # first route crosses t0 and t1; degenerate second route only touches t0
    rates = compute_tract_selection_rates(routes, tracts)
    assert rates["t0"] == 1.0  # touched by both
    assert rates["t3"] == 0.0  # touched by neither


def test_selection_rate_empty_routes_returns_zero():
    tracts = _synthetic_tracts()
    rates = compute_tract_selection_rates([], tracts)
    assert (rates == 0).all()


def test_four_fifths_ratio_computed_correctly_when_disparity_exists():
    selection_rates = pd.Series({"t0": 0.8, "t1": 0.8, "t2": 0.2, "t3": 0.2})
    groups = pd.Series({"t0": "low", "t1": "low", "t2": "high", "t3": "high"})
    result = four_fifths_audit(selection_rates, groups, lower=0.8, upper=1.25)
    assert abs(result["ratio"] - 0.25) < 1e-9
    assert result["flagged"] is True


def test_four_fifths_ratio_not_flagged_when_balanced():
    selection_rates = pd.Series({"t0": 0.5, "t1": 0.5, "t2": 0.45, "t3": 0.45})
    groups = pd.Series({"t0": "low", "t1": "low", "t2": "high", "t3": "high"})
    result = four_fifths_audit(selection_rates, groups)
    assert result["flagged"] is False


def test_assign_demographic_quantiles_returns_expected_bin_count():
    tracts = _synthetic_tracts()
    groups = assign_demographic_quantiles(tracts, "pct_nonwhite", n_groups=2)
    assert set(groups.unique().dropna().tolist()).issubset({"q1", "q2"})
