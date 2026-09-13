# SafeRoute — Research Implementation

Research-grade implementation of ETA-aligned, uncertainty-aware pedestrian
safety routing, built to support an IEEE TKDE submission. This is **not** a
polished app — see `PRD.md` for the full requirements this code implements.

**Status:** M0-M4 pipeline code complete and unit-tested. A small-scale
pilot run (see below) validates that the pipeline executes end-to-end on
real data. Paper-scale experimental results are `NOT YET EVALUATED` until
the full pipeline is run on `config/base.yaml` — nothing in `results/` from
a full run should be treated as final until that run has actually happened.
See `paper/` for the results/discussion placeholders this rule keeps empty.

## What this is (and isn't)

This repository implements the research pipeline: data ingestion, spatial
grid construction, a time-aware risk model, calibrated uncertainty
(conformal prediction), ETA-aligned risk-and-uncertainty-aware routing,
Pareto route alternatives, whitelisted template explanations, a
counterfactual held-out-incident benchmark, and (Phase 2) a fairness audit
and footfall feedback-loop simulation. There is no frontend, no auth, no
deployment config — those are explicitly out of scope for this stage
(PRD §2).

## Repository layout

```
config/            experiment configuration (base.yaml = full city; pilot.yaml = small validation run)
src/saferoute/
  data/            ingestion: Chicago crime (Socrata), OSM (OSMnx), weather (Open-Meteo), census (ACS)
  features/        H3 grid, temporal binning, environment features, label construction, splitting
  models/          risk model training (XGBoost/LightGBM/EBM), conformal uncertainty (MAPIE)
  routing/         graph construction, ETA-aligned cost function, baselines, Pareto filtering
  explain/         SHAP/EBM attribution, whitelist audit, template NLG, faithfulness eval
  benchmark/       counterfactual held-out-incident evaluation
  fairness/        demographic disparate-impact audit (Phase 2)
  simulation/      Mesa footfall feedback-loop simulation (Phase 2)
scripts/           thin CLI entrypoints, one per pipeline stage (see below)
tests/             pytest suite — data, features, models, routing, explain, benchmark, fairness, sim
results/           tables/ figures/ logs/ — all generated, none committed except .gitkeep
paper/             section skeleton; Results/Discussion are placeholders until real numbers exist
```

## Setup

Requires Python 3.12 (the repo's system Python may be older — this project
was built against Homebrew's `python@3.12`). LightGBM needs OpenMP:

```bash
brew install libomp   # macOS only, needed for the lightgbm wheel build
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest -q
```

## Pipeline

```
scripts/download_data.py         -> data/raw/*
scripts/build_features.py        -> data/processed/{train,val,test,spatial_holdout}.parquet
scripts/train_model.py           -> results/models/risk_model_*.joblib, results/tables/model_comparison.json
scripts/calibrate_uncertainty.py -> results/models/risk_model_conformal.joblib, results/tables/conformal_coverage.json
scripts/run_routing_ablations.py -> results/tables/routing_ablation.csv
scripts/run_benchmark.py         -> results/tables/benchmark_raw.csv, benchmark_bootstrap_ci.csv
scripts/run_explainability_eval.py -> results/tables/whitelist_audit.json, faithfulness.json
scripts/run_fairness_audit.py    -> results/tables/fairness_audit.json      (needs CENSUS_API_KEY, see below)
scripts/run_feedback_simulation.py -> results/tables/feedback_loop_{raw,summary}.csv
scripts/generate_figures.py      -> results/figures/*.png   (skips + placeholders for missing inputs)
scripts/generate_tables.py       -> results/tables/table*.csv (same skip policy)
```

Every script takes `--config <path>` (default `config/base.yaml`) and
`--override <path>` (deep-merged on top) — never hardcode an experiment
parameter in a script; add it to `config/base.yaml` and read it via
`saferoute.config.load_config`.

To reproduce the full paper-scale run:

```bash
for s in download_data build_features train_model calibrate_uncertainty \
         run_routing_ablations run_benchmark run_explainability_eval \
         run_fairness_audit run_feedback_simulation generate_figures generate_tables; do
  python scripts/$s.py --config config/base.yaml
done
```

To run the small pilot instead (validates the pipeline runs correctly,
NOT a paper-scale result):

```bash
python scripts/download_data.py --override config/pilot.yaml
python scripts/build_features.py --override config/pilot.yaml
# ... same pattern for the remaining stages
```

### Census API key

The Census Bureau's ACS5 endpoint now requires a free API key (this
changed after the PRD was written, which assumed anonymous low-volume
access still worked — it no longer does, confirmed while building this
pipeline). Get one at https://api.census.gov/data/key_signup.html and:

```bash
export CENSUS_API_KEY=your_key_here
```

Only `scripts/download_data.py` (the census step) and
`scripts/run_fairness_audit.py` need this. The M0-M4 core pipeline (risk
model, uncertainty, routing, explainability, benchmark) does not.

## Key research-engineering decisions (documented, not hidden)

### City choice: Chicago
Chicago's open crime data (Socrata `ijzp-q8t2`, City of Chicago Data
Portal) is free, requires no key, and includes lat/long + timestamp back to
2001 — the richest ready-to-use open crime feed found for MVP scope (PRD
§9's "US-heavy" caveat). The eventual target deployment city (per the PRD,
implied to be India) is a separate, later data-acquisition task — this
codebase validates the *method*, not a India-specific deployment, and the
risk model has NOT been validated on any Indian data.

### Routing engine: OSMnx/NetworkX, not OSRM/GraphHopper
The PRD (§7) prefers OSRM/GraphHopper but explicitly allows a custom
routing layer if a production engine "makes custom time-dependent
experimentation impractical" — documented here as the deciding factor:
this machine has no Docker, and OSRM's contraction hierarchies are built
around static edge weights, making a genuinely time-dependent,
per-query-arrival-time risk cost awkward to inject without rebuilding the
hierarchy per query. `src/saferoute/routing/eta_cost.py` implements a
label-setting (Dijkstra-style) search directly over an OSMnx-loaded
NetworkX graph, evaluating risk at each edge's *estimated arrival time*
rather than at departure time. Tradeoff: slower per-query search than a
contraction-hierarchy engine — acceptable for single-city research-scale
graphs, not for national-scale production routing.

### Spatial resolution: H3 res 9 (~174m), not exactly "~100m"
H3 does not offer an exact 100m resolution. Res 9 (~174m edge) is used
instead of res 10 (~66m edge) because Chicago's crime density at hourly
bins is too sparse per-cell at res 10 to support reliable label
estimation — this is a documented approximation of the PRD's "~100m"
target, not a silent substitution.

### Whitelisted feature proxies (read before citing these as measurements)
The five explanation-whitelist features (`lighting`, `footfall_proxy`,
`open_businesses`, `cctv`, `distance_to_help`) are OSM-tag-derived
proxies, not ground-truth measurements. Full detail and caveats are in
`src/saferoute/features/environment_features.py`'s module docstring —
read it before writing any paper text that references these features, so
the paper doesn't overstate what they actually are (e.g. `footfall_proxy`
is street-intersection density, not a measured pedestrian count).

### Risk target: incident count regression, not classification
The risk model predicts `incident_count` (a rate-like continuous target)
via Poisson-objective regression, not P(incident) via classification. This
is what makes the MAPIE split-conformal interval interpretable as a rate
interval `[risk_lower, risk_upper]` around `risk_mean`, per FR-1. Ground
truth is heavily zero-inflated — this is reported in
`results/tables/table1_dataset_statistics.csv`, not hidden.

### Counterfactual benchmark OD pairs are synthetic; the incidents are real
No open pedestrian-trip dataset exists for Chicago. `benchmark/counterfactual.py`
constructs synthetic origin/destination pairs via a bounded random walk from
each REAL held-out incident's location, so a shortest path is likely to
pass near it. The incidents themselves are real test-split crime records —
only the trip endpoints around them are synthetic. This belongs in the
paper's Limitations section, not glossed over as real trip-log evaluation.

### Pareto route generation is a weighted-sum scan, not an exact algorithm
`routing/pareto.py` generates alternatives by sweeping lambda and keeping
non-dominated results — a standard approximation that can miss frontier
points in non-convex regions of (time, risk) space. The non-domination
filter itself is exact and tested (`tests/test_routing.py`); the frontier
*coverage* is approximate.

### Mesa feedback-loop simulation is a stylized mechanism demonstration
`simulation/feedback_loop.py` is explicitly NOT calibrated to real
Chicago footfall/risk dynamics — no such dataset exists for this project
to calibrate against. It demonstrates whether the PRD's named feedback
mechanism (adoption reduces footfall on risk-avoided cells, which a
footfall-risk update rule can compound) appears, stays bounded, or decays
under this project's own cost function. Report it as a mechanism
demonstration in the paper, not as a calibrated forecast.

## Ethics / terminology (PRD §23 — enforced by code, not just convention)

Never write, log, or generate (including in template_nlg.py output) a
claim that: the system prevents crime; a route is objectively "safe"; a
location is inherently dangerous; a demographic group causes risk; or the
model is unbiased without the fairness audit's evidence. Use: predicted
risk, estimated exposure, uncertainty, historical incident patterns, model
output. `explain/whitelist_audit.py` checks generated explanation text
programmatically; it does not by itself catch every possible overclaim in
hand-written paper prose — that's still a human review responsibility.

## Known limitations already surfaced by the pilot run

See `results/logs/` and the per-module docstrings above; this section is
kept in sync with what the pilot run actually found (not written in
advance of running it).
