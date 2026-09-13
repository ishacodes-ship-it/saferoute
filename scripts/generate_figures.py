#!/usr/bin/env python
"""Generate the paper's figures from results/tables/*.csv|json.

Each figure function reads ONLY from files already written by an earlier
pipeline stage (build_features/train_model/calibrate_uncertainty/
run_routing_ablations/run_benchmark/run_explainability_eval/
run_fairness_audit/run_feedback_simulation). If the required input file is
missing, the figure is SKIPPED with a logged reason and a placeholder text
file is written instead of a fabricated plot — per project policy, no
figure is drawn from numbers that were not actually computed by a real run.
"""
import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from saferoute.config import DEFAULT_CONFIG_PATH, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _skip(fig_path, reason):
    logger.warning("SKIPPED %s: %s", fig_path.name, reason)
    fig_path.with_suffix(".NOT_YET_EVALUATED.txt").write_text(
        f"Figure not generated: {reason}\nRun the pipeline stage that produces the required input, then re-run this script.\n"
    )


def fig_dataset_distribution(cfg, tables_dir, figures_dir):
    path = cfg.path("data_processed") / "feature_table_full.parquet"
    out = figures_dir / "fig1_dataset_distribution.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_parquet(path, columns=["hour_of_day", "incident_count"])
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("hour_of_day")["incident_count"].sum().plot(kind="bar", ax=ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Total incidents (train+val+test period)")
    ax.set_title("Spatiotemporal distribution: incidents by hour of day")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_risk_by_time_of_day(cfg, tables_dir, figures_dir):
    path = cfg.path("data_processed") / "feature_table_full.parquet"
    out = figures_dir / "fig2_risk_by_time_of_day.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_parquet(path, columns=["hour_of_day", "is_weekend", "incident_count"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for weekend_flag, label in [(0, "Weekday"), (1, "Weekend")]:
        subset = df[df["is_weekend"] == weekend_flag]
        subset.groupby("hour_of_day")["incident_count"].mean().plot(ax=ax, label=label)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean incident count per cell-hour")
    ax.set_title("Predicted-relevant risk signal by time of day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_model_comparison(cfg, tables_dir, figures_dir):
    path = tables_dir / "model_comparison.json"
    out = figures_dir / "fig3_model_comparison.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    with open(path) as f:
        data = json.load(f)["results"]

    models = list(data.keys())
    maes = [data[m]["val"]["mae"] for m in models]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(models, maes)
    ax.set_ylabel("Validation MAE (incident count)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_conformal_coverage(cfg, tables_dir, figures_dir):
    path = tables_dir / "conformal_coverage.json"
    out = figures_dir / "fig4_conformal_coverage.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    with open(path) as f:
        cov = json.load(f)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Nominal", "Empirical"], [cov["nominal_coverage"], cov["empirical_coverage"]], color=["gray", "tab:blue"])
    ax.axhline(cov["nominal_coverage"], linestyle="--", color="red")
    ax.set_ylim(0, 1)
    ax.set_title(f"Conformal coverage (n={cov['n_samples']})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_routing_comparison(cfg, tables_dir, figures_dir):
    path = tables_dir / "routing_ablation.csv"
    out = figures_dir / "fig5_static_vs_eta_aware.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5, 4))
    df.boxplot(column="accumulated_risk", by="method", ax=ax, rot=30)
    ax.set_title("Accumulated route risk by method")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_pareto_curve(cfg, tables_dir, figures_dir):
    path = tables_dir / "benchmark_bootstrap_ci.csv"
    out = figures_dir / "fig6_avoidance_vs_detour_pareto.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_csv(path).dropna(subset=["avoidance_rate"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("detour_budget")
        ax.plot(grp["detour_budget"], grp["avoidance_rate"], marker="o", label=method)
        ax.fill_between(grp["detour_budget"], grp["ci_low"], grp["ci_high"], alpha=0.15)
    ax.set_xlabel("Detour budget (fraction of shortest-path time)")
    ax.set_ylabel("Incident avoidance rate")
    ax.set_title("Avoidance rate vs. detour cost (counterfactual benchmark)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_ablation_results(cfg, tables_dir, figures_dir):
    path = tables_dir / "routing_ablation.csv"
    out = figures_dir / "fig7_ablation_results.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("method")["accumulated_sigma"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean accumulated uncertainty (sigma)")
    ax.set_title("Ablation: uncertainty contribution by method")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_explanation_faithfulness(cfg, tables_dir, figures_dir):
    path = tables_dir / "faithfulness.json"
    out = figures_dir / "fig8_explanation_faithfulness.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    with open(path) as f:
        data = json.load(f)["per_feature"]

    features = list(data.keys())
    rates = [data[f]["direction_consistency_rate"] for f in features]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(features, rates)
    ax.axhline(0.9, linestyle="--", color="red", label="0.90 target")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Direction-consistency rate")
    ax.set_title("Explanation faithfulness by feature")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_fairness_results(cfg, tables_dir, figures_dir):
    path = tables_dir / "fairness_audit.json"
    out = figures_dir / "fig9_fairness_disparate_impact.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    with open(path) as f:
        data = json.load(f)
    if data.get("ratio") is None:
        return _skip(out, "fairness audit ran but returned no ratio (insufficient data)")

    group_means = data["group_means"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(group_means.keys()), list(group_means.values()))
    ax.set_ylabel("Mean route selection rate")
    ax.set_title(f"Disparate impact by demographic quantile (ratio={data['ratio']:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def fig_feedback_drift(cfg, tables_dir, figures_dir):
    path = tables_dir / "feedback_loop_raw.csv"
    out = figures_dir / "fig10_feedback_loop_risk_drift.png"
    if not path.exists():
        return _skip(out, f"missing {path}")

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for p, grp in df.groupby("adoption_rate"):
        mean_by_step = grp.groupby("step")["risk"].mean()
        ax.plot(mean_by_step.index, mean_by_step.values, label=f"p={p}")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean cell risk")
    ax.set_title("Feedback-loop risk drift by adoption rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


FIGURES = [
    fig_dataset_distribution, fig_risk_by_time_of_day, fig_model_comparison, fig_conformal_coverage,
    fig_routing_comparison, fig_pareto_curve, fig_ablation_results, fig_explanation_faithfulness,
    fig_fairness_results, fig_feedback_drift,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH, args.override)
    tables_dir = cfg.path("tables")
    figures_dir = cfg.path("figures")

    for fn in FIGURES:
        fn(cfg, tables_dir, figures_dir)


if __name__ == "__main__":
    main()
