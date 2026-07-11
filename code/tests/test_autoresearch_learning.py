"""Validity gates for the frozen paired learning tier."""

from __future__ import annotations

import pytest

from conformal_marl.autoresearch.candidate import sample_replay  # noqa: F401
from conformal_marl.autoresearch.frozen.learning_benchmark import (
    LearningScenario,
    run_learning_benchmark,
)


TINY_SCENARIOS = (
    LearningScenario(
        name="tiny_support",
        env_name="simu",
        env_kwargs={"num_agents": 3},
        config_overrides={
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
        },
        return_scale=4.0,
        support_gate=True,
    ),
)


def test_uniform_candidate_matches_paired_uniform_baseline():
    result = run_learning_benchmark(
        "autoresearch_candidate", seed=7, num_seeds=2, scenarios=TINY_SCENARIOS
    )
    assert result["status"] == "ok"
    assert result["primary_score"] == pytest.approx(0.0)
    scenario = result["scenarios"]["tiny_support"]
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
