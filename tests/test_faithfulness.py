import numpy as np
import pandas as pd

from saferoute.explain.faithfulness import evaluate_faithfulness
from saferoute.models.feature_spec import FEATURE_COLUMNS


class _LinearRiskStub:
    """A hand-built model where the true risk-increasing direction of every
    whitelisted feature is known exactly, so we can check the faithfulness
    test correctly detects perfect consistency and correctly detects failure."""

    def __init__(self, signs: dict):
        self.signs = signs  # feature -> sign of its coefficient (positive coef = higher value -> higher predicted risk)

    def predict(self, X: pd.DataFrame):
        pred = np.zeros(len(X))
        for feat, sign in self.signs.items():
            pred += sign * X[feat].to_numpy()
        return pred


def _sample_X(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(1, 5, size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)


def test_faithfulness_perfect_consistency_when_model_matches_expected_directions():
    # WORSENING_DIRECTION expects: lighting/footfall/businesses/cctv DOWN -> risk UP (negative coef),
    # distance_to_help UP -> risk UP (positive coef).
    signs = {"lighting": -1, "footfall_proxy": -1, "open_businesses": -1, "cctv": -1, "distance_to_help": 1}
    model = _LinearRiskStub(signs)
    X = _sample_X()
    result = evaluate_faithfulness(model, X)
    assert result["overall_direction_consistency_rate"] == 1.0
    for feat, stats in result["per_feature"].items():
        assert stats["direction_consistency_rate"] == 1.0


def test_faithfulness_detects_inconsistent_model():
    # Flip every sign -- the model now behaves opposite to what the explanation template claims.
    signs = {"lighting": 1, "footfall_proxy": 1, "open_businesses": 1, "cctv": 1, "distance_to_help": -1}
    model = _LinearRiskStub(signs)
    X = _sample_X()
    result = evaluate_faithfulness(model, X)
    assert result["overall_direction_consistency_rate"] == 0.0


def test_faithfulness_excludes_zero_variance_feature_from_aggregate():
    """Regression test for a real pilot-run finding: `lighting` had zero
    variance in a small bbox (no OSM `lit` tags at all), so its perturbation
    delta was always 0 -- correctly reflecting "model learned nothing from
    an uninformative feature," not "model is unfaithful." Before this fix,
    that 0% rate was silently averaged into the overall score, understating
    faithfulness for a reason unrelated to faithfulness."""
    signs = {"lighting": -1, "footfall_proxy": -1, "open_businesses": -1, "cctv": -1, "distance_to_help": 1}
    model = _LinearRiskStub(signs)
    X = _sample_X()
    X["lighting"] = 0.5  # constant -- zero variance

    result = evaluate_faithfulness(model, X)
    assert result["per_feature"]["lighting"]["zero_variance_in_sample"] is True
    assert "lighting" in result["features_excluded_from_aggregate"]
    # the other 4 features are still perfectly consistent and must still drive the aggregate to 1.0,
    # not be dragged down by lighting's untestable 0%.
    assert result["overall_direction_consistency_rate"] == 1.0


def test_faithfulness_reports_sample_size():
    model = _LinearRiskStub({"lighting": -1, "footfall_proxy": -1, "open_businesses": -1, "cctv": -1, "distance_to_help": 1})
    X = _sample_X(n=37)
    result = evaluate_faithfulness(model, X)
    assert result["n_samples"] == 37
