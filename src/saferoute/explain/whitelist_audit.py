"""FR-4 acceptance test infrastructure: 0 whitelist violations in an N-route sample.

Checks generated explanation strings for two kinds of leakage:
  1. Direct: a non-whitelisted feature/column name appears in the text
     (e.g. "historical_incident_rate", "hour_of_day", "temperature").
  2. Proxy: a small set of terms that could let a reader infer a
     non-whitelisted attribute (e.g. weather words, day-of-week words,
     demographic words) — this is the PRD's named proxy-leakage concern
     (§15 risks table: "Explanation whitelist could still leak identity via
     proxy features").

This module does NOT claim the 0-violation result — it only computes it.
Whether the actual run in results/tables/whitelist_audit.json shows 0
violations is an empirical fact to report, not to assume.
"""
from __future__ import annotations

import re

from saferoute.models.feature_spec import CONTEXTUAL_FEATURES, TEMPORAL_FEATURES

# Non-whitelisted model feature names — must never appear verbatim.
BANNED_TERMS = set(TEMPORAL_FEATURES + CONTEXTUAL_FEATURES) | {
    "historical_incident_rate",
    "incident_count",
    "has_incident",
}

# Proxy terms: words that reference weather, calendar, or demographic
# attributes even if phrased in plain English rather than a column name.
BANNED_PROXY_PATTERNS = [
    r"\brain\w*\b", r"\bweather\b", r"\btemperature\b", r"\bsnow\w*\b",
    r"\bweekday\b", r"\bweekend\b", r"\bmonday\b", r"\btuesday\b", r"\bwednesday\b",
    r"\bthursday\b", r"\bfriday\b", r"\bsaturday\b", r"\bsunday\b",
    r"\brace\b", r"\bincome\b", r"\bdemographic\w*\b", r"\bethnicity\b",
    r"\bpolice\b", r"\barrest\w*\b", r"\bcrime rate\b",
]

_PROXY_REGEX = re.compile("|".join(BANNED_PROXY_PATTERNS), flags=re.IGNORECASE)


def audit_explanation(text: str) -> list[str]:
    """Return a list of violation reasons for one explanation string (empty = clean)."""
    violations = []
    lowered = text.lower()

    for term in BANNED_TERMS:
        if term.replace("_", " ") in lowered or term in lowered:
            violations.append(f"banned_feature_term:{term}")

    proxy_match = _PROXY_REGEX.search(lowered)
    if proxy_match:
        violations.append(f"proxy_leak:{proxy_match.group(0)}")

    return violations


def audit_explanation_batch(explanations: list[str]) -> dict:
    """Run the audit over a sample of explanations and summarize.

    Returns a dict with total violations, violation rate, and the list of
    (index, violations) for any explanation that failed — this is what
    gets dumped to results/tables/whitelist_audit.json.
    """
    results = []
    for i, text in enumerate(explanations):
        v = audit_explanation(text)
        if v:
            results.append({"index": i, "text": text, "violations": v})

    return {
        "n_sampled": len(explanations),
        "n_violations": len(results),
        "violation_rate": len(results) / len(explanations) if explanations else 0.0,
        "violations": results,
    }
