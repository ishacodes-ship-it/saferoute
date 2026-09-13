# Pilot run notes (config/pilot.yaml)

Purpose: validate the M0-M4 pipeline executes end-to-end on real data
before committing compute to the full `config/base.yaml` run. Numbers here
are from a small bbox (~3km x 4km, downtown Chicago) and a 6-month window
(2023-01 to 2023-06) — **not paper-scale results**. Do not cite these in
`paper/10_results.md`.

## Real bugs this pilot run surfaced (fixed in the same session)

1. **Centroid computed in geographic CRS.** `environment_features.py` and
   `routing/graph.py` computed `.geometry.centroid` directly on lat/lon
   degree geometries instead of a projected CRS — geopandas warns about
   this for a reason (degree-based centroid math distorts). Fixed via a
   shared `features/geo_utils.centroid_latlon` helper that projects to
   the geometry's local UTM zone first. Numeric impact at this project's
   scale (H3 res 9, ~174m cells) is negligible, but it was fixed properly
   rather than suppressed, since a paper's methodology section should not
   describe an approach the code doesn't actually implement.

2. **`lighting` feature was 100% NaN, which silently emptied the entire
   training set.** `compute_lighting_by_cell`'s fallback-to-default logic
   (`grouped.mean() if len(grouped) else 0.5`) didn't handle the case
   where the `lit` OSM tag is present on ZERO edges in the graph — a real
   condition for a small bbox extract, not a hypothetical. `grouped.mean()`
   of an all-NaN group is NaN (not caught by `len(grouped)`, which was
   still > 0), so every cell's lighting value was NaN, which propagated
   through `dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])` in
   `train_risk_model.py` and reduced the training set to 0 rows —
   XGBoost's `count:poisson` objective then raised
   `Check failed: info.labels.Size() != 0U`. Fixed by checking
   `grouped.notna().any()` before using `grouped.mean()` as the fallback.
   Regression test: `tests/test_environment_features.py::test_lighting_falls_back_to_default_when_no_edge_has_lit_tag`.

3. **Census API requires a key now.** The PRD's assumption (anonymous
   ACS5 access at low volume) no longer holds — the endpoint returns an
   HTML "Missing Key" page instead of JSON. Confirmed live during this
   run, not assumed. See README.md "Census API key" for the fix (a free
   key, user must obtain it — account creation is not something this
   session does on the user's behalf).

4. **Mesa `seed` reproducibility footgun.** `mesa.Model.__new__` only
   reads a `seed` KEYWORD argument (positional is silently ignored,
   falling back to an unseeded random value) — `run_feedback_simulation`
   was calling `FeedbackLoopModel(..., seed)` positionally. Caught by
   `tests/test_feedback_loop.py::test_reproducibility_same_seed` failing.
   Fixed by making `seed` keyword-only in `FeedbackLoopModel.__init__` so
   this class of mistake becomes a `TypeError`, not a silent
   reproducibility break. Also replaced `mesa.time.RandomActivation`
   (whose `AgentSet.shuffle` depends on Python weak-reference dict
   ordering, itself dependent on object `id()` / memory layout — NOT
   fully determined by the seed) with an explicit seeded list-shuffle.

5. **Several scripts passed `None` as `config_path` into `load_config`
   when only `--override` was given**, since `load_config`'s positional
   default is `DEFAULT_CONFIG_PATH`, not `None` — `Path(None)` raises
   `TypeError`. Fixed across all `scripts/*.py` (see git history / diff
   for the full list) by defaulting to `DEFAULT_CONFIG_PATH` explicitly
   at the call site.

6. **Counterfactual OD-pair generation referenced columns that don't
   exist on the dense feature table.** `generate_od_pairs_near_incidents`
   read `row["latitude"]`, `row["longitude"]`, `row["date"]` from
   `incidents_df` — but that function is called with rows from the dense
   `(h3_cell, time_bin)` table (`features/labels.py`), which never carries
   raw per-incident coordinates (multiple incidents in the same cell-hour
   are already collapsed into `incident_count`). Every row raised inside a
   broad `except: continue`, so it failed completely silently — "Generated
   0/20 OD pairs," no traceback, benchmark then crashed further downstream
   on an empty DataFrame with a confusing `KeyError: 'method'`. Fixed by
   using the H3 cell's centroid (`h3_grid.cell_centroid`) as the incident
   location and `row["time_bin"]` for the time. Regression test:
   `tests/test_counterfactual_od_generation.py::test_generate_od_pairs_uses_h3_cell_and_time_bin_not_lat_lon`.
   Also added a fail-fast check in `scripts/run_benchmark.py` so "0 OD
   pairs" is a clear `RuntimeError`, not a `KeyError` three lines later.

7. **OD pairs from a fixed-hop-count random walk were geometrically too
   close for ANY route to avoid the incident.** With `walk_hops=5` on
   this pilot bbox's dense street grid (~15-25m segments), source and
   target ended up only 30-110m apart — smaller than the incident's own
   avoidance buffer (H3 res-9 1-ring, ~500m diameter). Every method
   (including the proposed one) scored exactly 0% avoidance at every
   detour budget, because there was no way to route around the buffer
   geometrically, regardless of routing quality. Fixed by walking until a
   MINIMUM DISTANCE from the incident is reached (`_walk_until_far`)
   rather than a fixed hop count, with the minimum itself configurable
   (`benchmark.min_od_distance_m`) since it needs to be city/graph-density
   dependent. Regression test:
   `tests/test_counterfactual_od_generation.py::test_generate_od_pairs_respects_minimum_distance`.

## Real (non-bug) findings from the pilot run — report honestly, do not force

- **Conformal coverage: 88.1% empirical vs. 90% nominal**
  (`results/tables/conformal_coverage.json`). Close but short of the PRD's
  target at pilot scale (6 months, ~90 walkable cells). Needs re-checking
  at full scale — do not assume it clears 90% just because it's close.

- **Counterfactual benchmark shows ~0% avoidance at every detour budget,
  even after fixing the OD-generation bugs above.** With
  `min_od_distance_m=400` (as large as this small pilot bbox reliably
  supports — 900m, the value planned for the full run, produced 0 usable
  OD pairs entirely; see `config/pilot.yaml`'s comment), most OD pairs are
  still not far enough from the incident relative to the buffer size for a
  0-30% detour budget to matter. This is a SCALE artifact of the pilot
  bbox, not a claim about the method — the full-scale run (city-wide
  graph, `min_od_distance_m=900`) needs to be checked before concluding
  anything about real avoidance-vs-detour behavior.

- **Explanation faithfulness: 24.75% overall (after excluding `lighting`,
  which has zero variance in this bbox and is mathematically untestable —
  see `explain/faithfulness.py`'s `zero_variance_in_sample` flag), well
  below the PRD's 90% target.** Per-feature
  (`results/tables/faithfulness.json`): `distance_to_help` 62% (best,
  correct sign), `cctv` 18%, `open_businesses` 12%, `footfall_proxy` 7%
  (worst) — and `footfall_proxy`/`open_businesses` have NEGATIVE mean
  delta, meaning the trained model associates MORE foot traffic and MORE
  open businesses with HIGHER predicted risk, the opposite of the
  guardianship-theory assumption ("more eyes on the street = safer")
  baked into `template_nlg.py`'s phrasing and `faithfulness.py`'s
  `WORSENING_DIRECTION` table. This could be a genuine opportunity-theory
  signal in Chicago's actual incident data (busy commercial areas have
  more targets, not just more guardians), a small-sample artifact (6
  months, zero-inflated labels), or both — it needs investigation at full
  scale before the paper's Explainability/Discussion sections take a
  position. **Do not tune the model or the whitelist phrasing to force
  this number up before understanding WHY it's low** — that would be
  exactly the result-shaping PRD.md §24 prohibits.

- **Feedback-loop simulation shows "compounding" risk-drift at EVERY
  adoption rate, including p=0.0** (`results/tables/feedback_loop_summary.csv`).
  Since p=0.0 means no agent ever uses risk-avoidant routing, this
  suggests the simulation's own risk-update mechanism (a bounded random
  walk with a footfall-deficit rule) is inherently unstable at these
  pilot parameters (`n_cells=20, n_agents=100, risk_update_rate=0.05, 20
  steps`) independent of SafeRoute adoption — meaning the current
  "compounding" flag doesn't yet distinguish a genuine adoption-driven
  effect from baseline mechanism noise. A more informative comparison
  would look at the DIFFERENCE in drift between p=0 and p>0, not each
  one's absolute compounding flag in isolation. Left as-is rather than
  "fixed," since PRD.md explicitly asks for the mechanism to be shown
  honestly, not adjusted until it produces a clean story.

- **Routing divergence from shortest path: 57.5% of sampled OD pairs**
  (`results/tables/routing_ablation.csv`, logged by
  `run_routing_ablations.py`) show the proposed ETA+uncertainty route
  taking measurably different travel time than the pure shortest path —
  real evidence the router isn't degenerating to shortest-path, at least
  at pilot scale.

- **Whitelist audit: 0/200 violations** — clean, matches the PRD's target
  shape (it specifies 1000; this is a 200-sample pilot check).

## Pipeline stages actually executed against real data this session

- [x] `download_data.py` (crime, OSM, weather) — census blocked on API key, see above
- [x] `build_features.py`
- [x] `train_model.py` — xgboost/lightgbm/ebm all trained; lightgbm won on val MAE
- [x] `calibrate_uncertainty.py`
- [x] `run_routing_ablations.py`
- [x] `run_benchmark.py`
- [x] `run_explainability_eval.py`
- [ ] `run_fairness_audit.py` (blocked on CENSUS_API_KEY — get one at https://api.census.gov/data/key_signup.html)
- [x] `run_feedback_simulation.py`
- [x] `generate_figures.py` (9/10 figures produced; fig9 fairness skipped, correctly, per the missing-input policy)
- [x] `generate_tables.py` (7/8 tables produced; table8 fairness skipped, correctly)

Every number above came from an actual script run against real Chicago
crime/OSM/weather data in this small pilot bbox. None of it is a
paper-scale claim — see the top of this file.
