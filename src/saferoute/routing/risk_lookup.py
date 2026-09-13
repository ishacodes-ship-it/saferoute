"""risk(segment, arrival_time): the query interface FR-1 requires.

Combines:
  - static per-cell environment features (lighting, footfall_proxy, ...)
  - the cell's most recently observed historical_incident_rate (see note below)
  - temporal features derived FROM THE QUERIED ARRIVAL TIME, not from "now" —
    this is what makes the router genuinely ETA-aligned rather than a static
    risk score with a clock icon next to it.
  - a weather reading, defaulted to a fixed "typical" value if none supplied

into a feature row for the trained conformal model, returning
(risk_mean, risk_lower, risk_upper, sigma).

LIMITATION (documented, not hidden): `historical_incident_rate` is a
causally-computed rolling feature at TRAINING time (see features/labels.py).
At serving/routing time there is no live incident feed in this research
prototype, so the last available rolling rate per cell (as of the end of the
training data) is reused as a stand-in for "current" — this is a
documented simplification for a research routing layer, not a production
real-time system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor

from saferoute.features.temporal import assign_temporal_bin
from saferoute.models.feature_spec import FEATURE_COLUMNS

DEFAULT_WEATHER = {"temperature_2m": 15.0, "precipitation": 0.0, "is_day": 1}


class RiskLookup:
    def __init__(self, mapie_model: MapieRegressor, cell_static_features: pd.DataFrame, nominal_coverage: float):
        """
        Args:
            cell_static_features: one row per h3_cell with columns
                [h3_cell] + ENVIRONMENT_FEATURES + ['historical_incident_rate'].
        """
        self.model = mapie_model
        self.cell_features = cell_static_features.set_index("h3_cell")
        self.nominal_coverage = nominal_coverage
        self.default_row = cell_static_features.drop(columns=["h3_cell"]).mean()
        # Risk only varies with (cell, hour-of-day, weekend/weekday) given our
        # feature set (weather held fixed per lookup instance) — caching on
        # that key avoids re-predicting per edge relaxation during search,
        # which matters because a single route query touches O(edges) cells.
        self._cache: dict[tuple, dict] = {}

    def _feature_row(self, cell: str, arrival_time: pd.Timestamp, weather: dict | None) -> pd.DataFrame:
        static = self.cell_features.loc[cell] if cell in self.cell_features.index else self.default_row
        weather = weather or DEFAULT_WEATHER

        row = {**static.to_dict(), **weather}
        temporal_frame = assign_temporal_bin(pd.DataFrame({"ts": [arrival_time]}), "ts")
        for col in ["hour_sin", "hour_cos", "is_night", "is_weekend", "day_of_week", "month"]:
            row[col] = temporal_frame.iloc[0][col]

        return pd.DataFrame([row])[FEATURE_COLUMNS]

    def risk(self, cell: str, arrival_time: pd.Timestamp, weather: dict | None = None) -> dict:
        """risk(segment, arrival_time) -> {risk_mean, risk_lower, risk_upper, sigma}."""
        cache_key = (cell, arrival_time.hour, int(arrival_time.dayofweek >= 5))
        if cache_key in self._cache:
            return self._cache[cache_key]

        X = self._feature_row(cell, arrival_time, weather)
        alpha = 1 - self.nominal_coverage
        y_pred, y_pis = self.model.predict(X, alpha=alpha)
        lower = float(np.clip(y_pis[0, 0, 0], 0, None))
        upper = float(np.clip(y_pis[0, 1, 0], 0, None))
        mean = float(np.clip(y_pred[0], 0, None))
        result = {
            "risk_mean": mean,
            "risk_lower": lower,
            "risk_upper": upper,
            "sigma": (upper - lower) / 2.0,
        }
        self._cache[cache_key] = result
        return result

    def risk_batch(self, cells: list[str], arrival_times: list[pd.Timestamp], weather: dict | None = None) -> pd.DataFrame:
        rows = [self._feature_row(c, t, weather) for c, t in zip(cells, arrival_times)]
        X = pd.concat(rows, ignore_index=True)
        alpha = 1 - self.nominal_coverage
        y_pred, y_pis = self.model.predict(X, alpha=alpha)
        lower = np.clip(y_pis[:, 0, 0], 0, None)
        upper = np.clip(y_pis[:, 1, 0], 0, None)
        mean = np.clip(y_pred, 0, None)
        return pd.DataFrame({"risk_mean": mean, "risk_lower": lower, "risk_upper": upper, "sigma": (upper - lower) / 2.0})


def build_cell_static_features(feature_table_full: pd.DataFrame) -> pd.DataFrame:
    """Collapse the dense (cell, time_bin) feature table to one row per cell.

    Environment features are static already (take first). historical_incident_rate
    takes the LAST time-ordered value per cell (see module docstring's
    documented limitation).
    """
    from saferoute.models.feature_spec import ENVIRONMENT_FEATURES

    df = feature_table_full.sort_values("time_bin")
    static_env = df.groupby("h3_cell")[ENVIRONMENT_FEATURES].first()
    last_rate = df.groupby("h3_cell")["historical_incident_rate"].last()
    out = static_env.join(last_rate).reset_index()
    return out
