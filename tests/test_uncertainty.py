import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from saferoute.models.uncertainty import evaluate_coverage, fit_conformal_wrapper, predict_with_interval


def _synthetic_regression_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    from saferoute.models.feature_spec import FEATURE_COLUMNS

    X = pd.DataFrame(rng.uniform(0, 1, size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = 2.0 * X["lighting"] + rng.normal(0, 0.1, size=n)
    X["incident_count"] = np.clip(y, 0, None)
    return X


def test_conformal_wrapper_produces_valid_ordered_intervals():
    df = _synthetic_regression_df()
    mapie_model = fit_conformal_wrapper(LinearRegression(), df, calibration_fraction=0.2, seed=42)

    from saferoute.models.feature_spec import FEATURE_COLUMNS

    intervals = predict_with_interval(mapie_model, df[FEATURE_COLUMNS].iloc[:20], nominal_coverage=0.9)
    assert (intervals["risk_lower"] <= intervals["risk_mean"]).all()
    assert (intervals["risk_mean"] <= intervals["risk_upper"]).all()
    assert (intervals["sigma"] >= 0).all()


def test_empirical_coverage_near_nominal_on_well_specified_model():
    df = _synthetic_regression_df(n=2000)
    mapie_model = fit_conformal_wrapper(LinearRegression(), df, calibration_fraction=0.2, seed=42)

    from saferoute.models.feature_spec import FEATURE_COLUMNS

    coverage = evaluate_coverage(mapie_model, df[FEATURE_COLUMNS], df["incident_count"], nominal_coverage=0.9)
    # split conformal has finite-sample slack; a well-specified linear model
    # should still land reasonably close to nominal, not wildly off (e.g. 40%).
    assert coverage["empirical_coverage"] > 0.7
    assert coverage["nominal_coverage"] == 0.9


def test_reproducibility_same_seed_same_intervals():
    df = _synthetic_regression_df()
    from saferoute.models.feature_spec import FEATURE_COLUMNS

    m1 = fit_conformal_wrapper(LinearRegression(), df, calibration_fraction=0.2, seed=123)
    m2 = fit_conformal_wrapper(LinearRegression(), df, calibration_fraction=0.2, seed=123)

    i1 = predict_with_interval(m1, df[FEATURE_COLUMNS].iloc[:10], 0.9)
    i2 = predict_with_interval(m2, df[FEATURE_COLUMNS].iloc[:10], 0.9)
    pd_testing_assert = np.allclose(i1["risk_mean"].to_numpy(), i2["risk_mean"].to_numpy())
    assert pd_testing_assert
