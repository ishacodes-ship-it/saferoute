"""Block/time-aware train/validation/test splitting.

Two leakage risks a naive random split would create, both closed here:

1. Temporal leakage: a randomly-split row from 2023-06-15 03:00 in train and
   2023-06-15 04:00 in test are adjacent in time and share almost all
   slow-moving context (season, nearby recent incidents). Splitting by
   contiguous time BLOCKS (train_end / val_end / test_end cutoffs) removes
   this.
2. Spatial leakage / optimistic generalization: even with a time split, a
   model can memorize a specific H3 cell's baseline risk level rather than
   learning transferable spatiotemporal structure. A fraction of cells
   (`spatial_holdout_fraction`) is removed from TRAIN ENTIRELY (never seen,
   any time bin) and evaluated separately as a spatial-generalization check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_block_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: str,
    time_col: str = "time_bin",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end, val_end, test_end = map(pd.Timestamp, (train_end, val_end, test_end))
    train = df[df[time_col] <= train_end]
    val = df[(df[time_col] > train_end) & (df[time_col] <= val_end)]
    test = df[(df[time_col] > val_end) & (df[time_col] <= test_end)]
    return train.copy(), val.copy(), test.copy()


def apply_spatial_holdout(
    df: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    cell_col: str = "h3_cell",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by cell identity: returns (non_holdout_rows, spatial_holdout_rows).

    Call this BEFORE temporal_block_split's train slice is used for fitting —
    remove holdout cells from all of train/val, and report test-set metrics
    separately for holdout vs. non-holdout cells.
    """
    rng = np.random.default_rng(seed)
    all_cells = np.sort(df[cell_col].unique())
    n_holdout = max(1, int(len(all_cells) * holdout_fraction))
    holdout_cells = set(rng.choice(all_cells, size=n_holdout, replace=False))

    is_holdout = df[cell_col].isin(holdout_cells)
    return df[~is_holdout].copy(), df[is_holdout].copy()
