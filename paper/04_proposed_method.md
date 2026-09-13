# 4. Proposed Method [DRAFTED]

## 4.1 Architecture

```
Chicago crime (Socrata) ─┐
OSM walk network ────────┼─▶ H3 grid (res 9, ~174m) × hourly bins ─▶ dense (cell, time) label table
Open-Meteo weather ───────┘         │
                                     ▼
                    causal historical-rate + environment features
                                     │
                                     ▼
                  risk model (XGBoost / LightGBM / EBM, compared)
                                     │
                                     ▼
                    MAPIE split-conformal calibration (90% nominal)
                                     │
                                     ▼
        risk(segment, arrival_time) = {mean, lower, upper, sigma}
                                     │
                                     ▼
     OSMnx/NetworkX routing graph + time-dependent label-setting search
       cost = travel_time · (1 + λ·risk + μ·σ), evaluated at each edge's ETA
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            Pareto routes    SHAP/EBM + template    Counterfactual
          (fast/balanced/     explanation (whitelist  benchmark against
             cautious)         features only)         held-out incidents
```

Every component above is implemented in `src/saferoute/` under the
directory named in the diagram; see README.md for the file-level mapping
and the documented engineering decisions (routing engine choice, H3
resolution, feature proxies) that this architecture depends on.

## 4.2 What distinguishes this from a static-risk router

Three baselines are implemented alongside the proposed method
(`src/saferoute/routing/baselines.py`), sharing the identical search code
so that any measured difference is attributable to the risk/uncertainty
formulation and not to different search implementations:

1. **Shortest path** — $\lambda = \mu = 0$.
2. **Static risk** — $\lambda > 0$, but risk is a time-AVERAGED per-cell
   score, discarding the query's arrival time.
3. **ETA-aware, no uncertainty** — $\lambda > 0$, risk evaluated at each
   edge's estimated arrival time, $\mu = 0$.
4. **Proposed** — ETA-aware risk AND its conformal uncertainty, $\mu > 0$.

Ablations sweep $\lambda$, $\mu$, and departure time
(`config/base.yaml: routing.*`) to establish which component contributes
measurable value — that establishment is an empirical claim to be made
in §10 (Results) from `results/tables/table6_ablation_study.csv`, not
here.
