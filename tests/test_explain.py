from saferoute.explain.template_nlg import generate_explanation
from saferoute.explain.whitelist_audit import audit_explanation, audit_explanation_batch


def test_generate_explanation_names_only_improved_features():
    chosen = {"lighting": 0.9, "footfall_proxy": 5, "open_businesses": 3, "cctv": 1, "distance_to_help": 50}
    alt = {"lighting": 0.3, "footfall_proxy": 5, "open_businesses": 3, "cctv": 1, "distance_to_help": 500}
    text = generate_explanation(chosen, alt, risk_reduction_pct=20.0)
    assert "lighting" in text or "better lighting" in text
    assert "help" in text  # distance_to_help improved too (lower is better, chosen is closer)


def test_generate_explanation_falls_back_when_no_feature_improved():
    chosen = {"lighting": 0.3, "footfall_proxy": 1, "open_businesses": 1, "cctv": 0, "distance_to_help": 500}
    alt = {"lighting": 0.9, "footfall_proxy": 5, "open_businesses": 5, "cctv": 3, "distance_to_help": 10}
    text = generate_explanation(chosen, alt, risk_reduction_pct=5.0)
    assert "5%" in text or "5 %" in text or "5" in text


def test_audit_clean_explanation_has_no_violations():
    text = "This route was preferred because it has lower predicted risk, with better lighting and more open businesses."
    assert audit_explanation(text) == []


def test_audit_flags_banned_feature_term():
    text = "This route was preferred due to a lower historical_incident_rate."
    violations = audit_explanation(text)
    assert any("banned_feature_term" in v for v in violations)


def test_audit_flags_proxy_terms():
    text = "This route avoids areas with higher rain and is preferred on weekends."
    violations = audit_explanation(text)
    assert any("proxy_leak" in v for v in violations)


def test_audit_batch_summary_counts_correctly():
    texts = [
        "Better lighting and more open businesses nearby.",
        "Lower risk due to temperature and weekend patterns.",
        "Closer proximity to help and greater CCTV coverage.",
    ]
    summary = audit_explanation_batch(texts)
    assert summary["n_sampled"] == 3
    assert summary["n_violations"] == 1
    assert summary["violation_rate"] == 1 / 3
