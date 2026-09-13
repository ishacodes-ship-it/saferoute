#!/usr/bin/env python
"""Phase 2 / FR-7: Mesa footfall feedback-loop simulation across adoption rates.

Writes results/tables/feedback_loop_raw.csv and feedback_loop_summary.csv.
"""
import argparse
import logging

from saferoute.config import DEFAULT_CONFIG_PATH, load_config
from saferoute.simulation.feedback_loop import run_feedback_simulation, summarize_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=500)
    parser.add_argument("--n-steps", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH, args.override)

    sim_df = run_feedback_simulation(
        adoption_rates=cfg["fairness"]["adoption_rates"],
        n_cells=args.n_cells,
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        risk_aversion_lambda=cfg["routing"]["default_lambda"],
        risk_update_rate=0.05,
        seed=cfg["seed"],
    )
    raw_path = cfg.path("tables") / "feedback_loop_raw.csv"
    sim_df.to_csv(raw_path, index=False)
    logger.info("Wrote %s (%d rows)", raw_path, len(sim_df))

    summary = summarize_drift(sim_df)
    summary_path = cfg.path("tables") / "feedback_loop_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)
    for _, row in summary.iterrows():
        logger.info("adoption_rate=%.1f compounding=%s", row["adoption_rate"], row["compounding"])


if __name__ == "__main__":
    main()
