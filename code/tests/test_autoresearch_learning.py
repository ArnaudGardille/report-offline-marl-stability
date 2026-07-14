"""Validity gates for the frozen paired learning tier."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_marl.autoresearch.candidate import sample_replay  # noqa: F401
from conformal_marl.autoresearch.frozen import learning_benchmark
from conformal_marl.autoresearch.frozen.learning_benchmark import (
    LearningScenario,
    run_learning_benchmark,
)


def _tiny_config() -> dict:
    return {
        "total_timesteps": 128,
        "num_envs": 2,
        "num_steps": 4,
        "num_grad_steps": 2,
        "learning_starts": 8,
        "batch_size": 8,
        "buffer_size": 256,
        "cal_size": 8,
        "epsilon_anneal_time": 64,
        "eval_episodes": 4,
        "eval_steps": 2,
        "eval_during_training": True,
    }


TINY_SCORED = LearningScenario(
    name="tiny_scored",
    env_name="simu",
    env_kwargs={"num_agents": 3},
    config_overrides=_tiny_config(),
    return_scale=4.0,
    support_gate=False,
)

TINY_GATE = LearningScenario(
    name="tiny_support",
    env_name="simu",
    env_kwargs={"num_agents": 3},
    config_overrides=_tiny_config(),
    return_scale=4.0,
    support_gate=True,
)

TINY_SCENARIOS = (TINY_SCORED, TINY_GATE)


def test_uniform_candidate_matches_paired_uniform_baseline():
    result = run_learning_benchmark(
        "autoresearch_candidate", seed=7, num_seeds=2, scenarios=(TINY_SCORED,)
    )
    assert result["status"] == "ok"
    assert result["primary_score"] == pytest.approx(0.0)
    scenario = result["scenarios"]["tiny_scored"]
    assert scenario["paired_utility_delta"] == pytest.approx([0.0, 0.0])
    assert scenario["baseline_final_return"] == pytest.approx(
        scenario["candidate_final_return"]
    )


def test_learning_screen_requires_multiple_seeds_and_eval_curve():
    with pytest.raises(ValueError, match="at least two seeds"):
        run_learning_benchmark(
            "autoresearch_candidate", seed=0, num_seeds=1, scenarios=TINY_SCENARIOS
        )

    invalid = LearningScenario(
        name="invalid", env_name="simu", config_overrides={}, support_gate=True
    )
    with pytest.raises(ValueError, match="evaluation curve"):
        run_learning_benchmark(
            "autoresearch_candidate", seed=0, num_seeds=2, scenarios=(invalid,)
        )


def test_all_gate_scenarios_are_rejected_as_unscored():
    """A run made only of support gates has nothing to score."""
    with pytest.raises(ValueError, match="non-gate"):
        run_learning_benchmark(
            "autoresearch_candidate", seed=0, num_seeds=2, scenarios=(TINY_GATE,)
        )


def test_support_gate_improvement_cannot_lift_the_score(monkeypatch):
    """The Simultaneous-Attack gate must never be a rewarded objective.

    A candidate that leaves the scored scenario flat but sharply *improves* the
    support-gate scenario must not raise ``primary_score`` above the scored
    result (support-starvation theorem). Before the fix the gate scenario was
    averaged into ``raw_score`` and this run would score strictly positive.
    """

    def fake_run_sampler(scenario, sampler_name, seeds):
        n = len(seeds)
        # baseline is always uniform (flat); the candidate only "wins" on the
        # gate scenario, and is flat on the genuinely scored scenario.
        if sampler_name == "uniform" or not scenario.support_gate:
            value = 0.0
        else:
            value = 100.0
        return {
            "returns": np.full((n, 2), value, dtype=np.float64),
            "end_greedy_return": np.full(n, value, dtype=np.float64),
        }

    monkeypatch.setattr(learning_benchmark, "_run_sampler", fake_run_sampler)

    result = run_learning_benchmark(
        "some_candidate", seed=0, num_seeds=4, scenarios=TINY_SCENARIOS
    )

    # Gate is present and would have a large positive delta if it were scored...
    gate = result["scenarios"]["tiny_support"]
    assert gate["lower_confidence_bound"] > 0.0
    # ...but the score is taken from the scored scenario alone.
    assert result["raw_score"] == pytest.approx(0.0)
    assert result["primary_score"] == pytest.approx(0.0)
    assert result["support_gate_failures"] == []
