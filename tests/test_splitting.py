import pandas as pd

from saferoute.features.splitting import apply_spatial_holdout, temporal_block_split


def _sample_df():
    return pd.DataFrame({
        "h3_cell": ["A", "A", "B", "B", "C", "C"],
        "time_bin": [
            pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01"),
            pd.Timestamp("2023-07-01"), pd.Timestamp("2023-08-01"),
            pd.Timestamp("2023-09-01"), pd.Timestamp("2023-10-01"),
        ],
    })


def test_temporal_block_split_no_overlap_and_ordering():
    df = _sample_df()
    train, val, test = temporal_block_split(df, train_end="2023-06-30", val_end="2023-08-31", test_end="2023-12-31")

    assert train["time_bin"].max() <= pd.Timestamp("2023-06-30")
    assert val["time_bin"].min() > pd.Timestamp("2023-06-30")
    assert val["time_bin"].max() <= pd.Timestamp("2023-08-31")
    assert test["time_bin"].min() > pd.Timestamp("2023-08-31")
    assert len(train) + len(val) + len(test) == len(df)


def test_spatial_holdout_removes_entire_cells():
    df = _sample_df()
    non_holdout, holdout = apply_spatial_holdout(df, holdout_fraction=0.34, seed=0)

    holdout_cells = set(holdout["h3_cell"].unique())
    non_holdout_cells = set(non_holdout["h3_cell"].unique())
    # a held-out cell's rows must be ENTIRELY absent from non_holdout, at every time bin
    assert holdout_cells.isdisjoint(non_holdout_cells)
    assert len(holdout_cells) >= 1


def test_spatial_holdout_reproducible_with_same_seed():
    df = _sample_df()
    _, holdout_1 = apply_spatial_holdout(df, holdout_fraction=0.34, seed=7)
    _, holdout_2 = apply_spatial_holdout(df, holdout_fraction=0.34, seed=7)
    assert set(holdout_1["h3_cell"]) == set(holdout_2["h3_cell"])
