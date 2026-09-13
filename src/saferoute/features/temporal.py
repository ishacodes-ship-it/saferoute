"""Hourly temporal binning and time-of-day / seasonal feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_temporal_bin(df: pd.DataFrame, timestamp_col: str, bin_size_hours: int = 1) -> pd.DataFrame:
    """Floor each timestamp to the configured hourly bin and derive calendar features."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    freq = f"{bin_size_hours}h"
    df["time_bin"] = df[timestamp_col].dt.floor(freq)
    df["hour_of_day"] = df["time_bin"].dt.hour
    df["day_of_week"] = df["time_bin"].dt.dayofweek  # 0=Monday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["time_bin"].dt.month
    # Cyclical encodings so "23:00" and "00:00" are adjacent to the model.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h >= 20 or h < 6) else 0)
    return df


def full_time_bin_range(start: str, end: str, bin_size_hours: int = 1) -> pd.DatetimeIndex:
    """Enumerate every time bin in [start, end] — used to build the dense grid×time label table."""
    return pd.date_range(start=start, end=end, freq=f"{bin_size_hours}h")
