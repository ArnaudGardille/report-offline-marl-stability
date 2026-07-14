"""Reliability and budget gates for the configurable mechanism benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_marl.autoresearch.api import Estimate, OperatorBatch
from conformal_marl.autoresearch.candidate import estimate_operator
from conformal_marl.autoresearch.evaluate import evaluate_candidate
from conformal_marl.autoresearch.frozen.mechanism_benchmark import (
    MechanismSweepConfig,
    generate_scenarios,
    run_mechanism_benchmark,
)


TINY = MechanismSweepConfig(
    action_counts=(2, 4),
    sample_sizes=(128,),
    shift_strengths=(0.0, 0.9),
    noise_levels=(0.1,),
    trials=8,
)


def test_mechanism_sweep_is_deterministic_and_covers_shift_buckets():
    first = run_mechanism_benchmark(estimate_operator, seed=7, config=TINY)
    second = run_mechanism_benchmark(estimate_operator, seed=7, config=TINY)
    assert first == second
    assert first["num_cells"] == 4
    assert set(first["buckets"]) == {"matched", "heavy"}
    assert first["robust_risk"] >= first["aggregate_mse"]


def test_exact_importance_estimator_reduces_uniform_drift_error():
    def exact_is(batch: OperatorBatch, rng: np.random.Generator) -> Estimate:
        del rng
        weights = (
            batch.target_probs[batch.teammate_actions]
            / batch.behavior_probs[batch.teammate_actions]
        )
        return Estimate(float(np.mean(weights * batch.outcomes)))

    uniform = run_mechanism_benchmark(estimate_operator, seed=11, config=TINY)
    corrected = run_mechanism_benchmark(exact_is, seed=11, config=TINY)
    assert corrected["buckets"]["heavy"]["mean_mse"] < uniform["buckets"]["heavy"]["mean_mse"]


def test_generated_scenarios_have_full_support_and_valid_probabilities():
    for scenario in generate_scenarios(TINY):
        scenario.validate()
        assert min(scenario.behavior_probs) > 0.0
        assert sum(scenario.target_probs) == pytest.approx(1.0)


def test_config_rejects_unknown_fields_and_excessive_sweep():
    with pytest.raises(ValueError, match="unknown"):
        MechanismSweepConfig.from_dict({"agent_score": 1.0})
    with pytest.raises(ValueError, match="128-cell"):
        MechanismSweepConfig(
            action_counts=tuple(range(2, 20)),
            sample_sizes=(32, 64, 128),
            shift_strengths=(0.0, 0.5, 1.0),
            noise_levels=(0.0, 1.0),
        ).validate()


def test_cli_trial_budget_overrides_agent_mechanism_config():
    result = evaluate_candidate(
        tier="mechanism",
        seed=5,
        trials=7,
        mechanism_config={
            "action_counts": [2],
            "sample_sizes": [64],
            "shift_strengths": [0.9],
            "noise_levels": [0.1],
            "coverage_floor": 0.02,
            "scenario_seed": 9,
            "trials": 4,
        },
    )
    assert result["num_cells"] == 1
    only_scenario = next(iter(result["scenarios"].values()))
    assert only_scenario["trials"] == 7
