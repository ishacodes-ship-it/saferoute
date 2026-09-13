# 9. Experimental Setup [DRAFTED]

## 9.1 Dataset

City: Chicago (see README.md "City choice" for the selection rationale
and the explicit caveat that this does not validate the method on the
target deployment city's data). Source: City of Chicago Data Portal,
Socrata dataset `ijzp-q8t2`, free and public, no API key required.

Full-scale configuration (`config/base.yaml`): 2018-01-01 to 2023-12-31,
full city bounding box. A small-scale pilot configuration
(`config/pilot.yaml`) covering a ~3km × 4km downtown-Chicago bounding box
and a 6-month window (2023-01 to 2023-06) is used to validate the pipeline
end-to-end before committing compute to the full run; pilot numbers are
NOT reported as paper results (see `results/logs/pilot_run_notes.md`).

## 9.2 Spatial and temporal resolution

H3 resolution 9 (~174m edge length), hourly bins. See README.md for why
this is the closest standard H3 resolution to the PRD's informal "~100m"
target rather than an exact match.

## 9.3 Splitting

Two leakage controls, both implemented in `src/saferoute/features/splitting.py`
and unit-tested (`tests/test_splitting.py`):

1. **Temporal block split** — train/val/test are contiguous, non-overlapping
   time windows (config: `split.train_end`, `val_end`, `test_end`), not a
   random row split.
2. **Spatial holdout** — a configured fraction of H3 cells
   (`split.spatial_holdout_fraction`) is excluded from train/val entirely
   and evaluated separately, to distinguish genuine spatiotemporal
   generalization from per-cell memorization.

## 9.4 Models compared

XGBoost, LightGBM (both with a Poisson objective, since the target is a
non-negative count), and InterpretML's `ExplainableBoostingRegressor`
(`src/saferoute/models/train_risk_model.py`). Model selection is by
held-out validation MAE — see §10 for which model actually won, once the
full run has been executed.

## 9.5 Uncertainty

MAPIE split-conformal regression (`cv="split"`), nominal coverage 90%,
calibration fraction 20% of the training split
(`config/base.yaml: uncertainty.*`).

## 9.6 Compute

All experiments run on a single local machine (no GPU, no paid cloud
compute) — consistent with the PRD's $0-direct-cost claim (PRD §14).
