"""Validity gates for the frozen paired learning tier."""

from __future__ import annotations

import json

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


def _constant_run_sampler(gate_value: float, scored_value: float):
    """``_run_sampler`` stub: flat curves at a per-scenario, per-arm constant."""

    def fake_run_sampler(scenario, sampler_name, seeds):
        n = len(seeds)
        if sampler_name == "uniform":
            value = 0.0
        else:
            value = gate_value if scenario.support_gate else scored_value
        return {
            "returns": np.full((n, 2), value, dtype=np.float64),
            "end_greedy_return": np.full(n, value, dtype=np.float64),
        }

    return fake_run_sampler


def test_support_regression_hard_rejects_the_candidate(monkeypatch):
    """A candidate that starves the gate is rejected however well it screens.

    This is the safety mechanism the whole tier rests on: the candidate below buys
    a large STORM-side gain with a large Simultaneous-Attack regression.
    """
    monkeypatch.setattr(
        learning_benchmark,
        "_run_sampler",
        _constant_run_sampler(gate_value=-100.0, scored_value=100.0),
    )

    result = run_learning_benchmark(
        "greedy_candidate", seed=0, num_seeds=4, scenarios=TINY_SCENARIOS
    )

    assert result["status"] == "support_regression"
    assert result["support_gate_failures"] == ["tiny_support"]
    # The uncapped score is kept for diagnosis, but must not be the ranking key.
    assert result["raw_score"] > 0.0
    assert result["primary_score"] == learning_benchmark.REJECTED_SCORE


def test_gate_failure_ranks_below_every_passing_candidate(monkeypatch):
    """Rejection must be -inf, not a finite cap.

    A finite cap of ``-support_tolerance`` let a gate-failing candidate outrank a
    gate-*passing* one that merely scored worse than the cap, so ranking on
    ``primary_score`` alone silently readmitted rejected candidates.
    """
    monkeypatch.setattr(
        learning_benchmark,
        "_run_sampler",
        _constant_run_sampler(gate_value=-100.0, scored_value=100.0),
    )
    rejected = run_learning_benchmark(
        "gate_breaker", seed=0, num_seeds=4, scenarios=TINY_SCENARIOS
    )

    # A candidate that preserves support but screens badly -- far worse than the
    # old -support_tolerance cap of -0.05.
    monkeypatch.setattr(
        learning_benchmark,
        "_run_sampler",
        _constant_run_sampler(gate_value=0.0, scored_value=-100.0),
    )
    passing = run_learning_benchmark(
        "weak_but_safe", seed=0, num_seeds=4, scenarios=TINY_SCENARIOS
    )

    assert passing["status"] == "ok"
    assert passing["primary_score"] < -0.05
    assert rejected["primary_score"] < passing["primary_score"]
    assert max(rejected, passing, key=lambda r: r["primary_score"]) is passing


def test_rejected_score_survives_the_json_round_trip(monkeypatch):
    """``evaluate.py`` persists results as JSON; -inf must come back as -inf."""
    monkeypatch.setattr(
        learning_benchmark,
        "_run_sampler",
        _constant_run_sampler(gate_value=-100.0, scored_value=100.0),
    )
    result = run_learning_benchmark(
        "gate_breaker", seed=0, num_seeds=4, scenarios=TINY_SCENARIOS
    )
    restored = json.loads(json.dumps(result, sort_keys=True))
    assert restored["primary_score"] == learning_benchmark.REJECTED_SCORE
    assert restored["status"] == "support_regression"


def test_interval_uses_a_t_quantile_not_the_normal_approximation(monkeypatch):
    """The lower bound must widen as seeds get scarcer.

    With an estimated standard error the statistic is t-distributed. Pinning 1.96
    (the normal quantile) understates the interval by 6.5x at 2 seeds, which makes
    the non-inferiority gate stop rejecting real regressions.
    """
    assert learning_benchmark._critical_value(8, 0.975) == pytest.approx(2.3646, abs=1e-3)
    assert learning_benchmark._critical_value(2, 0.975) == pytest.approx(12.706, abs=1e-2)

    def noisy_run_sampler(scenario, sampler_name, seeds):
        n = len(seeds)
        base = np.zeros(n, dtype=np.float64)
        # Candidate has mean 0 but real seed-to-seed spread, so the lower bound is
        # driven entirely by the critical value.
        cand = np.linspace(-1.0, 1.0, n)
        value = base if sampler_name == "uniform" else cand
        return {
            "returns": np.repeat(value[:, None], 2, axis=1),
            "end_greedy_return": value,
        }

    monkeypatch.setattr(learning_benchmark, "_run_sampler", noisy_run_sampler)
    result = run_learning_benchmark(
        "noisy", seed=0, num_seeds=4, scenarios=(TINY_SCORED,)
    )
    scenario = result["scenarios"]["tiny_scored"]
    expected = -learning_benchmark._critical_value(4, 0.975) * scenario["utility_sem"]
    assert scenario["mean_utility_delta"] == pytest.approx(0.0)
    assert scenario["lower_confidence_bound"] == pytest.approx(expected)
    # Strictly wider than the normal approximation it replaced.
    assert scenario["lower_confidence_bound"] < -1.96 * scenario["utility_sem"]


def test_frozen_scenarios_reject_in_place_mutation():
    """Candidate code shares this process, so pinned budgets must be read-only."""
    scenario = learning_benchmark.DEFAULT_LEARNING_SCENARIOS[0]
    with pytest.raises(TypeError):
        scenario.config_overrides["total_timesteps"] = 1
    with pytest.raises(TypeError):
        scenario.env_kwargs["num_agents"] = 99


def test_scenario_fingerprint_tracks_the_measured_budget():
    """A reported score is tied to the exact budget that produced it."""
    base = LearningScenario(
        name="fp", env_name="simu", config_overrides=_tiny_config(), return_scale=4.0
    )
    same = LearningScenario(
        name="fp", env_name="simu", config_overrides=_tiny_config(), return_scale=4.0
    )
    retuned = dict(_tiny_config())
    retuned["total_timesteps"] = 999_999
    changed = LearningScenario(
        name="fp", env_name="simu", config_overrides=retuned, return_scale=4.0
    )
    assert base.fingerprint() == same.fingerprint()
    assert base.fingerprint() != changed.fingerprint()


def test_baseline_integrity_rejects_a_displaced_control(monkeypatch):
    """A hostile candidate, not the benign shipped one.

    The previous version of this guard asserted ``get_sampler("uniform") is
    uniform_sampler`` against the shipped candidate, which never touches the
    registry -- so it passed with the entire production change reverted.
    """
    from conformal_marl import samplers

    def impostor(*args, **kwargs):  # pragma: no cover - never invoked
        raise AssertionError("the impostor control must never run")

    monkeypatch.setitem(samplers._SAMPLER_REGISTRY, "uniform", impostor)
    with pytest.raises(RuntimeError, match="no longer the frozen implementation"):
        samplers.assert_baseline_integrity("uniform")


def test_baseline_integrity_covers_every_sealed_control():
    """Not just ``uniform`` -- any sampler used as a control arm must be sealed."""
    from conformal_marl import samplers

    sealed = samplers.trusted_baselines()
    assert {"uniform", "exact_ratio_floor"} <= set(sealed)
    samplers.assert_baseline_integrity()  # all of them, unmodified


def test_learning_screen_refuses_to_score_against_a_modified_control(monkeypatch):
    """The check must bite where the control is *used*, not only at import."""
    from conformal_marl import samplers

    monkeypatch.setitem(
        samplers._SAMPLER_REGISTRY, "uniform", lambda *a, **k: None
    )
    with pytest.raises(RuntimeError, match="no longer the frozen implementation"):
        run_learning_benchmark(
            "autoresearch_candidate", seed=7, num_seeds=2, scenarios=(TINY_SCORED,)
        )


def test_frozen_scorers_are_imported_before_candidate_code():
    """Pin the load-bearing import order.

    Nothing else enforces it: ruff's isort rule is not enabled for this project, so a
    future ``ruff --select I --fix`` (or any import sorter) would silently reorder
    these and reopen the pre-import ``sys.modules`` window.
    """
    import ast
    import pathlib

    import conformal_marl.autoresearch.evaluate as evaluate_module

    source = pathlib.Path(evaluate_module.__file__).read_text(encoding="utf-8")
    frozen_at = None
    candidate_at = None
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module.startswith("conformal_marl.autoresearch.frozen"):
            frozen_at = node.lineno if frozen_at is None else min(frozen_at, node.lineno)
        if node.module.startswith("conformal_marl.autoresearch.candidate"):
            candidate_at = node.lineno

    assert frozen_at is not None and candidate_at is not None
    assert frozen_at < candidate_at, (
        "the frozen scorers must be imported before the generated candidate module, "
        "so the baseline snapshot is sealed before untrusted code runs"
    )
