"""Frozen paired-seed learning screen for replay candidates.

The screen asks two separate scientific questions. Cooperative STORM measures whether a
candidate improves learning under policy-induced replay mismatch. Simultaneous-Attack is
a support-preservation gate: replay selection is known to starve its rare optimal joint
action, so a candidate may not trade that failure for a STORM gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import jax
import numpy as np

from conformal_marl.envs.registry import make_env
from conformal_marl.iql import make_train


@dataclass(frozen=True)
class LearningScenario:
    """One immutable environment/configuration in the learning screen."""

    name: str
    env_name: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    return_scale: float = 1.0
    support_gate: bool = False

    def validate(self) -> None:
        if not self.name or self.return_scale <= 0.0:
            raise ValueError("learning scenarios need a name and positive return scale")
        if self.config_overrides.get("eval_during_training") is not True:
            raise ValueError("learning scenarios must record an evaluation curve")


DEFAULT_LEARNING_SCENARIOS = (
    LearningScenario(
        name="simultaneous_attack_support",
        env_name="simu",
        env_kwargs={"num_agents": 3},
        config_overrides={
            "total_timesteps": 8_192,
            "num_envs": 8,
            "num_steps": 8,
            "num_grad_steps": 4,
            "learning_starts": 128,
            "hidden_dim": 32,
            "batch_size": 64,
            "buffer_size": 8_192,
            "cal_size": 32,
            "epsilon_anneal_time": 4_096,
            "eval_episodes": 32,
            "eval_steps": 2,
            "eval_during_training": True,
        },
        return_scale=4.0,
        support_gate=True,
    ),
    LearningScenario(
        name="storm_cooperative",
        env_name="storm",
        env_kwargs={"num_inner_steps": 48},
        config_overrides={
            "total_timesteps": 16_384,
            "num_envs": 8,
            "num_steps": 32,
            "num_grad_steps": 4,
            "learning_starts": 256,
            "hidden_dim": 64,
            "batch_size": 64,
            "buffer_size": 16_384,
            "cal_size": 64,
            "epsilon_anneal_time": 10_000,
            "eval_episodes": 8,
            "eval_steps": 48,
            "eval_during_training": True,
        },
        return_scale=16.0,
    ),
)


def _run_sampler(scenario: LearningScenario, sampler_name: str, seeds: np.ndarray) -> dict:
    overrides = dict(scenario.config_overrides)
    overrides["sampler_name"] = sampler_name
    env, spec = make_env(
        scenario.env_name,
        config_overrides=overrides,
        **scenario.env_kwargs,
    )
    train = jax.jit(jax.vmap(make_train(spec.default_config, env)))
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    return jax.device_get(train(keys))


def _sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


def run_learning_benchmark(
    candidate_sampler: str,
    *,
    seed: int,
    num_seeds: int,
    scenarios: Iterable[LearningScenario] = DEFAULT_LEARNING_SCENARIOS,
    confidence_z: float = 1.96,
    support_tolerance: float = 0.05,
) -> dict:
    """Compare candidate and uniform replay on identical random seeds.

    The primary utility is ``0.6 * normalized AUC + 0.4 * normalized final return``.
    Scenario scores are lower confidence bounds on the paired candidate-minus-uniform
    utility. If a support-gate scenario falls below ``-support_tolerance``, the primary
    score is capped below zero regardless of gains elsewhere.
    """

    if num_seeds < 2:
        raise ValueError("the paired learning screen requires at least two seeds")
    scenario_list = tuple(scenarios)
    if not scenario_list:
        raise ValueError("at least one learning scenario is required")
    for scenario in scenario_list:
        scenario.validate()

    root = np.random.SeedSequence(seed)
    seed_values = root.generate_state(num_seeds, dtype=np.uint32)
    results: dict[str, dict[str, Any]] = {}
    lower_bounds = []
    gate_failures = []

    for scenario in scenario_list:
        baseline = _run_sampler(scenario, "uniform", seed_values)
        candidate = _run_sampler(scenario, candidate_sampler, seed_values)
        baseline_curve = np.asarray(baseline["returns"], dtype=np.float64)
        candidate_curve = np.asarray(candidate["returns"], dtype=np.float64)
        baseline_final = np.asarray(baseline["end_greedy_return"], dtype=np.float64)
        candidate_final = np.asarray(candidate["end_greedy_return"], dtype=np.float64)

        baseline_auc = baseline_curve.mean(axis=1) / scenario.return_scale
        candidate_auc = candidate_curve.mean(axis=1) / scenario.return_scale
        final_delta = (candidate_final - baseline_final) / scenario.return_scale
        paired_delta = 0.6 * (candidate_auc - baseline_auc) + 0.4 * final_delta
        mean_delta = float(paired_delta.mean())
        lower_bound = mean_delta - confidence_z * _sem(paired_delta)
        lower_bounds.append(lower_bound)

        if scenario.support_gate and lower_bound < -support_tolerance:
            gate_failures.append(scenario.name)

        results[scenario.name] = {
            "support_gate": scenario.support_gate,
            "paired_utility_delta": paired_delta.tolist(),
            "mean_utility_delta": mean_delta,
            "utility_sem": _sem(paired_delta),
            "lower_confidence_bound": lower_bound,
            "baseline_final_return": baseline_final.tolist(),
            "candidate_final_return": candidate_final.tolist(),
            "baseline_auc": baseline_auc.tolist(),
            "candidate_auc": candidate_auc.tolist(),
        }

    raw_score = float(np.mean(lower_bounds))
    primary_score = raw_score if not gate_failures else min(raw_score, -support_tolerance)
    return {
        "status": "ok" if not gate_failures else "support_regression",
        "metric_direction": "maximize",
        "primary_score": primary_score,
        "raw_score": raw_score,
        "candidate_sampler": candidate_sampler,
        "num_seeds": num_seeds,
        "paired_seeds": seed_values.astype(np.uint64).tolist(),
        "support_gate_failures": gate_failures,
        "scenarios": results,
    }
