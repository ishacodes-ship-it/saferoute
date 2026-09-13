"""Phase 2 / FR-7: Mesa agent-based footfall feedback-loop simulation.

THIS IS A STYLIZED MECHANISM-DEMONSTRATION MODEL, not a calibrated
prediction of real footfall/risk dynamics in Chicago — that calibration
would require a real pedestrian-flow dataset this project does not have.
Its purpose is narrow and honest: show whether the qualitative mechanism
the PRD is worried about (adoption of risk-avoidant routing reduces
footfall on already-labeled-risky segments, which a footfall-risk feedback
rule can compound into a redistribution spiral) appears, stays bounded, or
decays under this project's own routing cost function, at adoption rates
p in {0, 0.1, 0.3, 0.5} (FR-7's required grid).

Mechanism per simulation step:
  1. Each of N pedestrian agents has a fixed origin/destination cell pair.
  2. With probability p (the adoption rate), an agent is a SafeRoute
     "adopter" this run: it chooses its next-cell transition weighted by
     1 / (1 + lambda * current_risk[cell]). Non-adopters choose uniformly
     at random among neighbor cells toward the destination.
  3. Footfall[cell] = count of agents passing through cell this step.
  4. Risk update (the feedback rule): risk[cell] moves toward a higher
     value when footfall is below the cell's own historical baseline
     footfall, and toward a lower value when footfall exceeds it — a
     simple "eyes on the street" proxy — bounded to [0, 1] and damped by
     `risk_update_rate` so a single step cannot swing risk from 0 to 1.

Reported output: risk trajectory per cell (or its mean/spread across
cells) across steps, per adoption rate — a drift plot, not a single number
(PRD FR-7: "flagged if drift compounds rather than stabilizes").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mesa import Agent, Model


class PedestrianAgent(Agent):
    def __init__(self, unique_id, model, origin: int, destination: int, is_adopter: bool):
        super().__init__(unique_id, model)
        self.current = origin
        self.destination = destination
        self.is_adopter = is_adopter

    def step(self):
        if self.current == self.destination:
            return
        neighbors = self.model.grid_neighbors[self.current]
        if not neighbors:
            return

        if self.is_adopter:
            risks = np.array([self.model.risk[n] for n in neighbors])
            weights = 1.0 / (1.0 + self.model.risk_aversion_lambda * risks)
            # Bias toward the neighbor that reduces graph distance to destination.
            dist_bias = np.array([1.0 if self.model.distance_to(n, self.destination) < self.model.distance_to(self.current, self.destination) else 0.3 for n in neighbors])
            weights = weights * dist_bias
        else:
            weights = np.array([1.0 if self.model.distance_to(n, self.destination) < self.model.distance_to(self.current, self.destination) else 0.3 for n in neighbors])

        weights = weights / weights.sum()
        self.current = self.random.choices(neighbors, weights=weights, k=1)[0]
        self.model.footfall[self.current] += 1


class FeedbackLoopModel(Model):
    def __init__(self, n_cells: int, n_agents: int, adoption_rate: float, risk_aversion_lambda: float, risk_update_rate: float, *, seed: int):
        # `seed` is keyword-only on purpose: mesa.Model.__new__ only reads a
        # `seed` KEYWORD argument to set up self.random (it ignores a
        # positionally-passed one and silently falls back to an unseeded
        # random value) -- making it keyword-only here turns that footgun
        # into a TypeError at the call site instead of a silent
        # reproducibility break.
        super().__init__(seed=seed)
        self.n_cells = n_cells
        self.risk_aversion_lambda = risk_aversion_lambda
        self.risk_update_rate = risk_update_rate
        # Deliberately NOT using mesa.time.RandomActivation: its AgentSet
        # shuffles via Python weak-reference dict ordering, which depends on
        # object id() / memory layout and is NOT fully determined by
        # self.random's seed alone -- it breaks the exact reproducibility
        # this research pipeline requires (see tests/test_feedback_loop.py).
        # A plain list + explicit seeded shuffle each step sidesteps that.
        self.agent_list: list[PedestrianAgent] = []

        rng = np.random.default_rng(seed)
        # Abstract ring-lattice grid over n_cells — stands in for the H3 road
        # network topology without requiring a live graph for this stylized model.
        self.grid_neighbors = {c: [(c - 1) % n_cells, (c + 1) % n_cells] for c in range(n_cells)}
        self.risk = rng.uniform(0.05, 0.3, size=n_cells)
        self.baseline_footfall = rng.uniform(5, 20, size=n_cells)
        self.footfall = np.zeros(n_cells)

        for i in range(n_agents):
            origin, destination = rng.choice(n_cells, size=2, replace=False)
            is_adopter = rng.random() < adoption_rate
            agent = PedestrianAgent(i, self, int(origin), int(destination), bool(is_adopter))
            self.agent_list.append(agent)

        self.n_agents = n_agents
        self.history = []

    def distance_to(self, a: int, b: int) -> int:
        direct = abs(a - b)
        return min(direct, self.n_cells - direct)

    def step(self):
        self.footfall[:] = 0
        order = list(range(len(self.agent_list)))
        self.random.shuffle(order)  # seeded via self.random (set from `seed` in Model.__new__)
        for idx in order:
            self.agent_list[idx].step()

        deficit = (self.baseline_footfall - self.footfall) / np.maximum(self.baseline_footfall, 1e-6)
        self.risk = np.clip(self.risk + self.risk_update_rate * deficit, 0.0, 1.0)

        self.history.append(self.risk.copy())


def run_feedback_simulation(
    adoption_rates: list[float], n_cells: int, n_agents: int, n_steps: int, risk_aversion_lambda: float, risk_update_rate: float, seed: int
) -> pd.DataFrame:
    """Run the simulation for each adoption rate; return a long-form DataFrame
    of [adoption_rate, step, cell, risk] for drift-plot generation.
    """
    rows = []
    for p in adoption_rates:
        model = FeedbackLoopModel(n_cells, n_agents, p, risk_aversion_lambda, risk_update_rate, seed=seed)
        for step in range(n_steps):
            model.step()
            for cell, risk_val in enumerate(model.risk):
                rows.append({"adoption_rate": p, "step": step, "cell": cell, "risk": risk_val})

    return pd.DataFrame(rows)


def summarize_drift(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Per adoption rate: mean risk and cross-cell std at first vs. last step, and whether drift compounds.

    "Compounding" is flagged if the cross-cell standard deviation of risk
    (a proxy for redistribution inequality across cells) is higher at the
    final step than the first — i.e. risk is spreading apart across cells,
    not stabilizing.
    """
    results = []
    for p, grp in sim_df.groupby("adoption_rate"):
        first_step = grp[grp["step"] == grp["step"].min()]
        last_step = grp[grp["step"] == grp["step"].max()]
        results.append({
            "adoption_rate": p,
            "mean_risk_first": first_step["risk"].mean(),
            "mean_risk_last": last_step["risk"].mean(),
            "std_risk_first": first_step["risk"].std(),
            "std_risk_last": last_step["risk"].std(),
            "compounding": last_step["risk"].std() > first_step["risk"].std(),
        })
    return pd.DataFrame(results)
