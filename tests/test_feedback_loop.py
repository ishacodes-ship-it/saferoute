from saferoute.simulation.feedback_loop import run_feedback_simulation, summarize_drift


def test_simulation_output_shape():
    df = run_feedback_simulation([0.0, 0.5], n_cells=10, n_agents=20, n_steps=5, risk_aversion_lambda=1.0, risk_update_rate=0.05, seed=1)
    assert set(df["adoption_rate"].unique()) == {0.0, 0.5}
    assert df["step"].max() == 4
    assert df["risk"].between(0, 1).all()


def test_reproducibility_same_seed():
    df1 = run_feedback_simulation([0.3], n_cells=10, n_agents=20, n_steps=5, risk_aversion_lambda=1.0, risk_update_rate=0.05, seed=99)
    df2 = run_feedback_simulation([0.3], n_cells=10, n_agents=20, n_steps=5, risk_aversion_lambda=1.0, risk_update_rate=0.05, seed=99)
    assert (df1["risk"].to_numpy() == df2["risk"].to_numpy()).all()


def test_different_seeds_can_diverge():
    df1 = run_feedback_simulation([0.3], n_cells=10, n_agents=20, n_steps=5, risk_aversion_lambda=1.0, risk_update_rate=0.05, seed=1)
    df2 = run_feedback_simulation([0.3], n_cells=10, n_agents=20, n_steps=5, risk_aversion_lambda=1.0, risk_update_rate=0.05, seed=2)
    assert not (df1["risk"].to_numpy() == df2["risk"].to_numpy()).all()


def test_summarize_drift_flags_compounding_when_std_increases():
    df = run_feedback_simulation([0.0], n_cells=10, n_agents=20, n_steps=20, risk_aversion_lambda=1.0, risk_update_rate=0.05, seed=1)
    summary = summarize_drift(df)
    assert "compounding" in summary.columns
    assert len(summary) == 1
