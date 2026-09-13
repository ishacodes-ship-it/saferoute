import pandas as pd

from saferoute.features.labels import add_historical_rate_features, build_dense_label_table


def test_dense_label_table_covers_every_cell_time_combo():
    crime_df = pd.DataFrame({
        "h3_cell": ["A", "A", "B"],
        "time_bin": [pd.Timestamp("2023-01-01 00:00"), pd.Timestamp("2023-01-01 02:00"), pd.Timestamp("2023-01-01 01:00")],
    })
    cells = ["A", "B", "C"]
    time_bins = pd.date_range("2023-01-01 00:00", "2023-01-01 03:00", freq="1h")

    dense = build_dense_label_table(crime_df, cells, time_bins)
    assert len(dense) == len(cells) * len(time_bins)

    row = dense[(dense["h3_cell"] == "A") & (dense["time_bin"] == pd.Timestamp("2023-01-01 00:00"))]
    assert row["incident_count"].iloc[0] == 1
    assert row["has_incident"].iloc[0] == 1

    row_c = dense[(dense["h3_cell"] == "C") & (dense["time_bin"] == pd.Timestamp("2023-01-01 00:00"))]
    assert row_c["incident_count"].iloc[0] == 0


def test_historical_rate_feature_has_no_same_bin_leakage():
    """The historical rate at time T must not include T's own label."""
    dense = pd.DataFrame({
        "h3_cell": ["A"] * 5,
        "time_bin": pd.date_range("2023-01-01", periods=5, freq="1h"),
        "has_incident": [0, 0, 0, 0, 1],
    })
    out = add_historical_rate_features(dense, lookback_days=1, bin_size_hours=1)
    # last row has_incident=1 but its OWN historical_incident_rate must be computed
    # from the 4 rows strictly before it (all zero), i.e. 0.0 -- not influenced by itself.
    assert out["historical_incident_rate"].iloc[-1] == 0.0


def test_historical_rate_reflects_past_incidents():
    dense = pd.DataFrame({
        "h3_cell": ["A"] * 4,
        "time_bin": pd.date_range("2023-01-01", periods=4, freq="1h"),
        "has_incident": [1, 0, 0, 0],
    })
    out = add_historical_rate_features(dense, lookback_days=1, bin_size_hours=1)
    # second row's rate should reflect the first row's incident
    assert out["historical_incident_rate"].iloc[1] > 0
