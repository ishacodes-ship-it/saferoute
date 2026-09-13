"""Single source of truth for which columns feed the risk model.

Both training (train_risk_model.py) and serving (risk_service.py) import
FEATURE_COLUMNS from here, so a mismatch between train-time and predict-time
feature sets is impossible by construction.
"""
from __future__ import annotations

# Whitelisted environment features (also the explanation whitelist —
# config/base.yaml: explainability.feature_whitelist maps 1:1 to these,
# modulo the lighting/footfall_proxy/open_businesses/cctv/distance_to_help naming).
ENVIRONMENT_FEATURES = [
    "lighting",
    "footfall_proxy",
    "open_businesses",
    "cctv",
    "distance_to_help",
]

# Temporal features — these are what make risk(segment, arrival_time) genuinely
# time-aware rather than a static per-cell score with a time column bolted on.
TEMPORAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_weekend",
    "day_of_week",
    "month",
]

# Historical/contextual features.
CONTEXTUAL_FEATURES = [
    "historical_incident_rate",
    "temperature_2m",
    "precipitation",
    "is_day",
]

FEATURE_COLUMNS = ENVIRONMENT_FEATURES + TEMPORAL_FEATURES + CONTEXTUAL_FEATURES
LABEL_COLUMN = "has_incident"
