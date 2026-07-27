"""Validity gates for the frozen teammate-policy drift intervention."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from conformal_marl.autoresearch import evaluate  # noqa: F401
from conformal_marl.autoresearch.frozen import causal_drift_benchmark as benchmark
from conformal_marl.autoresearch.frozen.causal_drift_benchmark import (
    CausalDriftConfig,
    HELDOUT_CAUSAL_DRIFT_CONFIG,
    SEARCH_CAUSAL_DRIFT_CONFIG,
    paired_streams,
    run_causal_drift_benchmark,
)


TINY = CausalDriftConfig(
    pre_size=32,
    post_size=32,
    calibration_size=16,
    draw_batch_size=32,
    train_batch_size=8,
    updates=8,
)


def test_paired_streams_match_before_intervention_and_only_partner_changes_after():
    stationary, drift, stationary_cal, drift_cal = paired_streams(7, TINY)
    pre = TINY.pre_size
    stationary_np = jax.tree_util.tree_map(np.asarray, stationary)
    drift_np = jax.tree_util.tree_map(np.asarray, drift)

    for left, right in zip(
        jax.tree_util.tree_leaves(stationary_np),
        jax.tree_util.tree_leaves(drift_np),
        strict=True,
    ):
        np.testing.assert_array_equal(left[:pre], right[:pre])

    np.testing.assert_array_equal(
        stationary_np.actions[pre:, 0], drift_np.actions[pre:, 0]
    )
    assert np.any(stationary_np.actions[pre:, 1] != drift_np.actions[pre:, 1])
    np.testing.assert_array_equal(
        np.asarray(stationary_cal.actions[:, 0]),
        np.asarray(drift_cal.actions[:, 0]),
    )


def test_intervention_switches_the_analytic_best_response():
    pre = benchmark._expected_action_values(TINY.pre_attack_prob)
    post = benchmark._expected_action_values(TINY.post_attack_prob)
    assert int(np.argmax(pre)) == 1
    assert int(np.argmax(post)) == 0


def test_identical_candidate_and_control_have_zero_causal_interaction():
    result = run_causal_drift_benchmark(
        "autoresearch_candidate",
        seed=9,
        num_seeds=2,
        config=TINY,
        control_samplers=("uniform",),
    )
    assert result["status"] == "ok"
    assert result["primary_score"] == pytest.approx(0.0)
    control = result["controls"]["uniform"]
    assert control["causal_interaction"] == pytest.approx([0.0, 0.0])


def test_exact_ratio_control_detects_uniform_drift_failure():
    result = run_causal_drift_benchmark(
        "autoresearch_candidate",
        seed=9,
        num_seeds=2,
        config=TINY,
        control_samplers=("uniform", "exact_ratio_floor"),
    )
    assert result["controls"]["uniform"]["interaction_screening_bound"] == pytest.approx(0.0)
    assert result["controls"]["exact_ratio_floor"]["interaction_screening_bound"] < 0.0
    assert result["primary_score"] < 0.0


def test_difference_in_differences_rejects_generic_gain(monkeypatch):
    utilities = {
        "candidate": {"no_drift": 0.8, "drift": 0.8},
        "uniform": {"no_drift": 0.5, "drift": 0.5},
    }

    def fake_run(method, replay, calibration, partner_attack_prob, **kwargs):
        del replay, calibration, kwargs
        regime = "no_drift" if partner_attack_prob > 0.25 else "drift"
        return utilities[method][regime]

    monkeypatch.setattr(benchmark, "_run_regime", fake_run)
    result = run_causal_drift_benchmark(
        "candidate",
        seed=0,
        num_seeds=4,
        config=TINY,
        control_samplers=("uniform",),
    )
    assert result["controls"]["uniform"]["drift_screening_bound"] > 0.0
    assert result["primary_score"] == pytest.approx(0.0)


def test_difference_in_differences_rewards_drift_specific_gain(monkeypatch):
    utilities = {
        "candidate": {"no_drift": 0.5, "drift": 0.9},
        "uniform": {"no_drift": 0.5, "drift": 0.5},
    }

    def fake_run(method, replay, calibration, partner_attack_prob, **kwargs):
        del replay, calibration, kwargs
        regime = "no_drift" if partner_attack_prob > 0.25 else "drift"
        return utilities[method][regime]

    monkeypatch.setattr(benchmark, "_run_regime", fake_run)
    result = run_causal_drift_benchmark(
        "candidate",
        seed=0,
        num_seeds=4,
        config=TINY,
        control_samplers=("uniform",),
    )
    assert result["status"] == "ok"
    assert result["primary_score"] == pytest.approx(0.4)


def test_no_drift_regression_is_a_hard_gate(monkeypatch):
    utilities = {
        "candidate": {"no_drift": 0.4, "drift": 0.9},
        "uniform": {"no_drift": 0.6, "drift": 0.5},
    }

    def fake_run(method, replay, calibration, partner_attack_prob, **kwargs):
        del replay, calibration, kwargs
        regime = "no_drift" if partner_attack_prob > 0.25 else "drift"
        return utilities[method][regime]

    monkeypatch.setattr(benchmark, "_run_regime", fake_run)
    result = run_causal_drift_benchmark(
        "candidate",
        seed=0,
        num_seeds=4,
        config=TINY,
        control_samplers=("uniform",),
    )
    assert result["status"] == "no_drift_regression"
    assert result["primary_score"] < 0.0


def test_heldout_causal_profile_changes_intervention_and_budget():
    assert HELDOUT_CAUSAL_DRIFT_CONFIG.pre_attack_prob != (
        SEARCH_CAUSAL_DRIFT_CONFIG.pre_attack_prob
    )
    assert HELDOUT_CAUSAL_DRIFT_CONFIG.post_attack_prob != (
        SEARCH_CAUSAL_DRIFT_CONFIG.post_attack_prob
    )
    assert HELDOUT_CAUSAL_DRIFT_CONFIG.updates != SEARCH_CAUSAL_DRIFT_CONFIG.updates
