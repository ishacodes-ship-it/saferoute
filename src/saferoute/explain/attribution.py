"""SHAP attribution, restricted to the whitelisted feature set.

The model is trained on TEMPORAL_FEATURES and CONTEXTUAL_FEATURES too, and
those can legitimately dominate SHAP importance (e.g. hour-of-day usually
matters a lot for incident rate). This module computes SHAP values over ALL
features but then DROPS every non-whitelisted column before returning
anything — the explanation layer downstream never sees a non-whitelisted
attribution value, so it cannot leak one even by a coding mistake (FR-4).
"""
from __future__ import annotations

import pandas as pd
import shap

from saferoute.models.feature_spec import ENVIRONMENT_FEATURES, FEATURE_COLUMNS


def compute_whitelisted_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of SHAP values, columns restricted to ENVIRONMENT_FEATURES.

    `model` must be a tree model (XGBoost/LightGBM) or EBM's underlying
    estimator — shap.TreeExplainer is used for tree models; for EBM, its own
    native `explain_local` is used instead (see explain_ebm_local below).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[FEATURE_COLUMNS])
    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLUMNS, index=X.index)
    return shap_df[ENVIRONMENT_FEATURES]


def explain_ebm_local(ebm_model, X: pd.DataFrame) -> pd.DataFrame:
    """EBM's native local explanation, restricted to ENVIRONMENT_FEATURES."""
    explanation = ebm_model.explain_local(X[FEATURE_COLUMNS])
    rows = []
    for i in range(len(X)):
        data = explanation.data(i)
        contrib = dict(zip(data["names"], data["scores"]))
        rows.append({f: contrib.get(f, 0.0) for f in ENVIRONMENT_FEATURES})
    return pd.DataFrame(rows, index=X.index)
