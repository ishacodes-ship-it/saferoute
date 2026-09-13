# 3. Problem Formulation [DRAFTED]

## 3.1 Risk prediction

For a road segment (represented by its H3 cell $c$) and an arrival time
$t$, define the target risk function:

$$\text{risk}(c, t) = \mathbb{E}[\text{incidents in } c \text{ during the hour containing } t]$$

modeled as a regression target (`incident_count` in
`src/saferoute/models/train_risk_model.py`), not a binary
classification probability. This is what makes the conformal interval
below interpretable as a rate interval.

## 3.2 Calibrated uncertainty

A split-conformal wrapper (MAPIE) around the point estimator produces,
for a nominal coverage level $1-\alpha$:

$$[\text{risk}_\text{lower}(c,t), \text{risk}_\text{upper}(c,t)], \quad \sigma(c,t) = \frac{\text{risk}_\text{upper} - \text{risk}_\text{lower}}{2}$$

with the guarantee (subject to exchangeability of calibration/test data)
that $\Pr[\text{risk}(c,t) \in [\text{risk}_\text{lower}, \text{risk}_\text{upper}]] \geq 1-\alpha$
asymptotically. Empirical coverage is measured, not assumed —
`results/tables/table4_calibration_results.csv` reports the actual number.

## 3.3 Routing cost

For a path $P = (e_1, \ldots, e_n)$ departing at $t_0$, define the arrival
time at edge $e_i$ recursively as $t_i = t_{i-1} + \text{travel\_time}(e_i)$,
and the total cost:

$$\text{cost}(P, t_0) = \sum_{i=1}^{n} \text{travel\_time}(e_i) \cdot \left(1 + \lambda \cdot \text{risk}(c_{e_i}, t_{i-1}) + \mu \cdot \sigma(c_{e_i}, t_{i-1})\right)$$

where $\lambda$ (risk aversion) and $\mu$ (uncertainty aversion) are
configured, not hardcoded (`config/base.yaml: routing.default_lambda`,
`routing.default_mu`). Risk and uncertainty for edge $e_i$ are evaluated at
$t_{i-1}$ — the estimated arrival time AT that edge — not at $t_0$. This is
the ETA-alignment property distinguishing the proposed method from a
static-risk baseline (`tests/test_routing.py::test_eta_alignment_risk_queried_at_estimated_arrival_hour`
tests this directly).

## 3.4 Pareto route alternatives

A route $A$ dominates route $B$ if $A.\text{time} \leq B.\text{time}$ and
$A.\text{risk} \leq B.\text{risk}$ with at least one strict inequality
(`src/saferoute/routing/pareto.py`). Returned alternatives are the
non-dominated set from a $\lambda$-sweep — an approximation of the true
Pareto frontier (see README.md's documented limitation on this point).

## 3.5 Counterfactual evaluation

For a held-out incident at cell $c^*$ and time $t^*$, a route "avoids" the
incident if none of its edges' cells intersect the 1-ring neighborhood of
$c^*$. Avoidance rate is reported as a function of detour budget (fraction
of extra travel time over the shortest path), not collapsed to one number
(PRD §12).
