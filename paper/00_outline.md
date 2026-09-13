# SafeRoute — Paper Outline (IEEE TKDE submission)

Status legend used throughout this directory:
- `[DRAFTED]` — prose exists and reflects the actual implementation.
- `[PLACEHOLDER]` — structure/section exists, content must not be written
  until the corresponding experiment has actually been run. Filling this in
  before the run is exactly the fabrication PRD.md §24 prohibits.
- `[NOT YET EVALUATED]` — a specific number/claim that requires a real
  experiment run. Never replace this with a number that wasn't produced by
  `results/tables/*` or `results/figures/*` from an actual run.

## Section files

1. `01_introduction.md` — [PLACEHOLDER]
2. `02_related_work.md` — [PLACEHOLDER] (source material: `literature survey.md`, not yet in this repo)
3. `03_problem_formulation.md` — [DRAFTED] (formalizes risk(segment, arrival_time), cost function — these are implementation facts, not results)
4. `04_proposed_method.md` — [DRAFTED] (architecture description — implementation fact)
5. `05_spatiotemporal_risk_modeling.md` — [PLACEHOLDER] (needs model comparison results)
6. `06_uncertainty_estimation.md` — [PLACEHOLDER] (needs conformal coverage results)
7. `07_risk_aware_routing.md` — [PLACEHOLDER] (needs routing ablation results)
8. `08_explainability.md` — [PLACEHOLDER] (needs whitelist audit + faithfulness results)
9. `09_experimental_setup.md` — [DRAFTED] (dataset, city, splits — implementation facts)
10. `10_results.md` — [NOT YET EVALUATED] (empty until scripts/run_*.py have produced results/tables/*)
11. `11_fairness_and_feedback_loop.md` — [NOT YET EVALUATED] (needs CENSUS_API_KEY + fairness/feedback runs)
12. `12_discussion.md` — [PLACEHOLDER] (cannot be written before Results)
13. `13_limitations.md` — [DRAFTED] (known upfront: proxy features, synthetic OD pairs, US-only validation, Pareto approximation — see README.md)
14. `14_conclusion.md` — [PLACEHOLDER]

## Hard rule for anyone (human or model) editing this directory

Do not write a number into any section file unless it was copied from a
file under `results/tables/` or `results/figures/` that was produced by an
actual script run in this repository. If you are tempted to write "the
model achieved X% accuracy" before running `scripts/train_model.py`,
stop — that is the exact failure mode PRD.md §24 names explicitly
(citing the literature survey's R²=0.9987-on-self-generated-labels
example).
