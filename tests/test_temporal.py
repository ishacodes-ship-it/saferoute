import pandas as pd

from saferoute.features.temporal import assign_temporal_bin, full_time_bin_range


def test_assign_temporal_bin_hour_floor():
    df = pd.DataFrame({"ts": [pd.Timestamp("2023-06-15 03:47:12")]})
    out = assign_temporal_bin(df, "ts", bin_size_hours=1)
    assert out["time_bin"].iloc[0] == pd.Timestamp("2023-06-15 03:00:00")
    assert out["hour_of_day"].iloc[0] == 3


def test_is_weekend_flag():
    # 2023-06-17 is a Saturday
    df = pd.DataFrame({"ts": [pd.Timestamp("2023-06-17 12:00:00"), pd.Timestamp("2023-06-14 12:00:00")]})
    out = assign_temporal_bin(df, "ts")
    assert out["is_weekend"].iloc[0] == 1
    assert out["is_weekend"].iloc[1] == 0


def test_is_night_flag_boundaries():
    df = pd.DataFrame({"ts": [pd.Timestamp("2023-06-15 21:00:00"), pd.Timestamp("2023-06-15 12:00:00"), pd.Timestamp("2023-06-15 05:00:00")]})
    out = assign_temporal_bin(df, "ts")
    assert out["is_night"].tolist() == [1, 0, 1]


def test_cyclical_hour_encoding_continuity():
    df = pd.DataFrame({"ts": [pd.Timestamp("2023-06-15 23:00:00"), pd.Timestamp("2023-06-15 00:00:00")]})
    out = assign_temporal_bin(df, "ts")
    # 23:00 and 00:00 should be close in (sin, cos) space, not maximally distant
    dist = ((out["hour_sin"].iloc[0] - out["hour_sin"].iloc[1]) ** 2 + (out["hour_cos"].iloc[0] - out["hour_cos"].iloc[1]) ** 2) ** 0.5
    assert dist < 0.3


def test_full_time_bin_range_length():
    bins = full_time_bin_range("2023-01-01", "2023-01-02", bin_size_hours=1)
    assert len(bins) == 25  # inclusive of both endpoints
