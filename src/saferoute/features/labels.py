"""Label construction: incident counts and binary risk labels per (H3 cell, time bin).

The label is built on a DENSE grid — every (cell, time_bin) combination in
the study period and area, not just cells/bins with at least one incident.
Skipping the dense join would implicitly label "no data" as "no risk",
which silently fabricates zero-risk ground truth for unobserved
cell-time pairs. Dense join makes the true class imbalance visible instead
of hiding it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_dense_label_table(
    crime_df: pd.DataFrame,
    all_cells: list[str],
    time_bins: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Cross-join all cells x all time bins, then attach observed incident counts.

    Args:
        crime_df: must have columns ['h3_cell', 'time_bin'] (already assigned
            by h3_grid.assign_h3_cells and temporal.assign_temporal_bin).
        all_cells: every H3 cell in the study bounding box (h3_grid.enumerate_grid_cells).
        time_bins: every hourly bin in the study period (temporal.full_time_bin_range).
    """
    counts = crime_df.groupby(["h3_cell", "time_bin"]).size().rename("incident_count")

    cell_index = pd.Index(all_cells, name="h3_cell")
    time_index = pd.Index(time_bins, name="time_bin")
    full_index = pd.MultiIndex.from_product([cell_index, time_index])

    dense = counts.reindex(full_index, fill_value=0).reset_index()
    dense["has_incident"] = (dense["incident_count"] > 0).astype(int)
    return dense


def add_historical_rate_features(
    dense_df: pd.DataFrame,
    lookback_days: int,
    bin_size_hours: int = 1,
) -> pd.DataFrame:
    """Rolling historical incident rate per cell, computed causally (no label leakage).

    For each (cell, time_bin) row, the rolling window covers
    [time_bin - lookback_days, time_bin) — strictly before the current bin —
    so the label for `time_bin` never contributes to its own feature.
    """
    df = dense_df.sort_values(["h3_cell", "time_bin"]).copy()
    window_bins = int(lookback_days * 24 / bin_size_hours)

    def _rolling_causal_mean(group: pd.Series) -> pd.Series:
        # shift(1) excludes the current bin, making the window strictly past.
        return group.shift(1).rolling(window=window_bins, min_periods=1).mean()

    df["historical_incident_rate"] = df.groupby("h3_cell")["has_incident"].transform(_rolling_causal_mean)
    df["historical_incident_rate"] = df["historical_incident_rate"].fillna(0.0)
    return df
