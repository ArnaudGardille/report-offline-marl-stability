"""Validity gates for the constrained operator-estimation research tier."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_marl.autoresearch.api import Estimate, OperatorBatch
from conformal_marl.autoresearch.candidate import estimate_operator, sample_replay  # noqa: F401
from conformal_marl.autoresearch.frozen.operator_benchmark import (
    DEFAULT_SCENARIOS,
    OperatorScenario,
    run_operator_benchmark,
)


def test_operator_batch_requires_full_behavior_support():
    batch = OperatorBatch(
        teammate_actions=np.array([0, 1], dtype=np.int32),
        outcomes=np.array([0.0, 1.0]),
        behavior_probs=np.array([1.0, 0.0]),
        target_probs=np.array([0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="full support"):
        batch.validate()


def test_candidate_cannot_mutate_frozen_batch():
    batch = OperatorBatch(
        teammate_actions=np.array([0, 1], dtype=np.int32),
        outcomes=np.array([0.0, 1.0]),
        behavior_probs=np.array([0.5, 0.5]),
        target_probs=np.array([0.5, 0.5]),
    ).readonly()
    with pytest.raises(ValueError, match="read-only"):
        batch.outcomes[0] = 2.0


def test_stratified_estimator_recovers_deterministic_backup():
    batch = OperatorBatch(
        teammate_actions=np.array([0, 0, 1, 1], dtype=np.int32),
        outcomes=np.array([-1.0, -1.0, 2.0, 2.0]),
        behavior_probs=np.array([0.5, 0.5]),
        target_probs=np.array([0.25, 0.75]),
    )

    def stratified(observed: OperatorBatch, rng: np.random.Generator) -> Estimate:
        del rng
        means = np.array(
            [
                observed.outcomes[observed.teammate_actions == action].mean()
                for action in range(len(observed.target_probs))
            ]
        )
        return Estimate(float(np.dot(observed.target_probs, means)))

    estimate = stratified(batch, np.random.default_rng(0))
    assert estimate.value == pytest.approx(1.25)


def test_operator_benchmark_is_deterministic_and_detects_uniform_shift_bias():
    first = run_operator_benchmark(estimate_operator, seed=7, trials=32)
    second = run_operator_benchmark(estimate_operator, seed=7, trials=32)
    assert first == second
    assert first["status"] == "ok"
    assert first["scenarios"]["matched"]["mse"] < 0.02
    assert first["scenarios"]["heavy_shift"]["mse"] > 1.0


def test_default_scenarios_are_valid_and_cover_policy_mismatch():
    for scenario in DEFAULT_SCENARIOS:
        scenario.validate()
    assert any(s.behavior_probs != s.target_probs for s in DEFAULT_SCENARIOS)


def test_scenario_rejects_zero_support_even_when_target_is_zero():
    scenario = OperatorScenario(
        name="bad_support",
        conditional_means=(0.0, 1.0),
        behavior_probs=(1.0, 0.0),
        target_probs=(1.0, 0.0),
        noise_std=0.0,
        sample_size=10,
    )
    with pytest.raises(ValueError, match="support"):
        scenario.validate()
