"""Template-based natural-language route explanation (MVP default, no LLM — PRD §7).

Generates one sentence naming ONLY whitelisted features, contrasting the
chosen route against a rejected alternative (typically the shortest-path
baseline). Direction-aware: a feature only gets mentioned if it moved in
the direction that REDUCES risk along the chosen route relative to the
alternative — this keeps every generated sentence causally consistent with
"why this route was preferred," rather than listing feature values that
happen to be true but didn't drive the preference.
"""
from __future__ import annotations

from saferoute.models.feature_spec import ENVIRONMENT_FEATURES

# feature -> (phrase when chosen route is BETTER, direction of "better")
# direction: "higher_is_better" or "lower_is_better"
FEATURE_PHRASES = {
    "lighting": ("better lighting", "higher_is_better"),
    "footfall_proxy": ("busier, more walkable streets", "higher_is_better"),
    "open_businesses": ("more open businesses nearby", "higher_is_better"),
    "cctv": ("greater CCTV coverage", "higher_is_better"),
    "distance_to_help": ("closer proximity to help", "lower_is_better"),
}

MAX_FEATURES_IN_EXPLANATION = 2


def _improvement(chosen_val: float, alt_val: float, direction: str) -> float:
    """Positive = chosen route is better on this feature; magnitude = size of the improvement."""
    diff = chosen_val - alt_val
    return diff if direction == "higher_is_better" else -diff


def generate_explanation(chosen_avg_features: dict, alternative_avg_features: dict, risk_reduction_pct: float) -> str:
    """
    Args:
        chosen_avg_features / alternative_avg_features: {feature_name: route-average value}
            for exactly ENVIRONMENT_FEATURES.
        risk_reduction_pct: (alt_risk - chosen_risk) / alt_risk * 100, precomputed by the caller
            from the actual accumulated_risk values — this function does not compute it,
            it only phrases a number it's given, so it can never invent a magnitude.
    """
    improvements = []
    for feat in ENVIRONMENT_FEATURES:
        phrase, direction = FEATURE_PHRASES[feat]
        imp = _improvement(chosen_avg_features.get(feat, 0.0), alternative_avg_features.get(feat, 0.0), direction)
        if imp > 0:
            improvements.append((imp, phrase))

    if not improvements:
        return (
            f"This route was preferred because it has a predicted risk reduction of "
            f"{risk_reduction_pct:.0f}% during your arrival period."
        )

    improvements.sort(key=lambda x: x[0], reverse=True)
    top_phrases = [p for _, p in improvements[:MAX_FEATURES_IN_EXPLANATION]]

    if len(top_phrases) == 1:
        reason_clause = top_phrases[0]
    else:
        reason_clause = " and ".join([", ".join(top_phrases[:-1]), top_phrases[-1]]) if len(top_phrases) > 2 else " and ".join(top_phrases)

    return (
        f"This route was preferred because it has lower predicted risk during your arrival period, "
        f"with {reason_clause}."
    )
