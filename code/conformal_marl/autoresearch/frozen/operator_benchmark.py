"""Exact operator-estimation benchmark for perceived MARL non-stationarity.

For one fixed state and ego action, the random Bellman target ``Z`` has a stationary
conditional mean for each teammate action. Replay actions are drawn from a behaviour
policy while the desired perceived Bellman backup marginalises those means under a
different target policy. The benchmark knows the exact backup but exposes only sampled
replay and the two policies to a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from conformal_marl.autoresearch.api import OperatorBatch, OperatorEstimator


@dataclass(frozen=True)
class OperatorScenario:
    name: str
    conditional_means: tuple[float, ...]
    behavior_probs: tuple[float, ...]
    target_probs: tuple[float, ...]
    noise_std: float
    sample_size: int

    def validate(self) -> None:
        means = np.asarray(self.conditional_means, dtype=np.float64)
        behavior = np.asarray(self.behavior_probs, dtype=np.float64)
        target = np.asarray(self.target_probs, dtype=np.float64)
        if means.ndim != 1 or behavior.shape != means.shape or target.shape != means.shape:
            raise ValueError(f"invalid aligned vectors in scenario {self.name!r}")
        if np.any(behavior <= 0.0) or np.any(target < 0.0):
            raise ValueError(f"scenario {self.name!r} violates the support contract")
        if not np.isclose(behavior.sum(), 1.0) or not np.isclose(target.sum(), 1.0):
            raise ValueError(f"scenario {self.name!r} policies must sum to one")
        if self.noise_std < 0.0 or self.sample_size <= 0:
            raise ValueError(f"scenario {self.name!r} has an invalid sampling budget")

    @property
    def true_backup(self) -> float:
        return float(np.dot(self.target_probs, self.conditional_means))


DEFAULT_SCENARIOS = (
    OperatorScenario(
        name="matched",
        conditional_means=(-1.0, 1.0),
        behavior_probs=(0.5, 0.5),
        target_probs=(0.5, 0.5),
        noise_std=0.5,
        sample_size=512,
    ),
    OperatorScenario(
        name="moderate_shift",
        conditional_means=(-1.0, 1.0),
        behavior_probs=(0.8, 0.2),
        target_probs=(0.3, 0.7),
        noise_std=0.5,
        sample_size=512,
    ),
    OperatorScenario(
        name="heavy_shift",
        conditional_means=(-1.0, 1.0),
        behavior_probs=(0.97, 0.03),
        target_probs=(0.2, 0.8),
        noise_std=0.5,
        sample_size=512,
    ),
    OperatorScenario(
        name="four_actions",
        conditional_means=(-1.5, -0.25, 0.75, 2.0),
        behavior_probs=(0.7, 0.2, 0.08, 0.02),
        target_probs=(0.1, 0.2, 0.3, 0.4),
        noise_std=0.75,
        sample_size=1024,
    ),
)


def sample_batch(scenario: OperatorScenario, rng: np.random.Generator) -> OperatorBatch:
    """Draw one replay batch without exposing the exact conditional means."""

    scenario.validate()
    behavior = np.asarray(scenario.behavior_probs, dtype=np.float64)
    actions = rng.choice(len(behavior), size=scenario.sample_size, p=behavior)
    means = np.asarray(scenario.conditional_means, dtype=np.float64)
    outcomes = means[actions] + rng.normal(0.0, scenario.noise_std, scenario.sample_size)
    batch = OperatorBatch(
        teammate_actions=actions.astype(np.int32),
        outcomes=outcomes,
        behavior_probs=behavior,
        target_probs=np.asarray(scenario.target_probs, dtype=np.float64),
    )
    batch.validate()
    return batch.readonly()


def run_operator_benchmark(
    estimator: OperatorEstimator,
    *,
    seed: int,
    trials: int,
    scenarios: Iterable[OperatorScenario] = DEFAULT_SCENARIOS,
) -> dict:
    """Evaluate an estimator repeatedly and return JSON-compatible exact errors."""

    if trials <= 1:
        raise ValueError("trials must exceed one to estimate variance")

    root = np.random.SeedSequence(seed)
    scenario_list = tuple(scenarios)
    child_sequences = root.spawn(len(scenario_list) * trials)
    cursor = 0
    results: dict[str, dict[str, float | int]] = {}

    for scenario in scenario_list:
        scenario.validate()
        estimates = []
        for _ in range(trials):
            data_rng, candidate_rng = (
                np.random.default_rng(sequence)
                for sequence in child_sequences[cursor].spawn(2)
            )
            cursor += 1
            batch = sample_batch(scenario, data_rng)
            estimate = estimator(batch, candidate_rng)
            estimate.validate()
            estimates.append(float(estimate.value))

        values = np.asarray(estimates, dtype=np.float64)
        errors = values - scenario.true_backup
        results[scenario.name] = {
            "true_backup": scenario.true_backup,
            "mean_estimate": float(values.mean()),
            "bias": float(errors.mean()),
            "variance": float(values.var(ddof=1)),
            "mse": float(np.mean(errors**2)),
            "sample_size": scenario.sample_size,
            "trials": trials,
        }

    aggregate_mse = float(np.mean([result["mse"] for result in results.values()]))
    return {
        "status": "ok",
        "metric_direction": "maximize",
        "primary_score": -aggregate_mse,
        "aggregate_mse": aggregate_mse,
        "scenarios": results,
    }
