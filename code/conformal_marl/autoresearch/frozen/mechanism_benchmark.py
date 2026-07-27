"""Deterministic factorial mechanism benchmark for target-policy replay estimators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from conformal_marl.autoresearch.api import OperatorEstimator
from conformal_marl.autoresearch.frozen.operator_benchmark import (
    OperatorScenario,
    run_operator_benchmark,
)


@dataclass(frozen=True)
class MechanismSweepConfig:
    action_counts: tuple[int, ...] = (2, 4, 8)
    sample_sizes: tuple[int, ...] = (128, 512)
    shift_strengths: tuple[float, ...] = (0.0, 0.5, 0.9)
    noise_levels: tuple[float, ...] = (0.1, 0.75)
    coverage_floor: float = 0.02
    scenario_seed: int = 20260712
    trials: int = 32

    def validate(self) -> None:
        if not self.action_counts or any(value < 2 or value > 32 for value in self.action_counts):
            raise ValueError("action_counts must be in [2, 32]")
        if not self.sample_sizes or any(value < 32 or value > 8192 for value in self.sample_sizes):
            raise ValueError("sample_sizes must be in [32, 8192]")
        if not self.shift_strengths or any(
            value < 0.0 or value > 1.0 for value in self.shift_strengths
        ):
            raise ValueError("shift_strengths must be in [0, 1]")
        if not self.noise_levels or any(value < 0.0 or value > 5.0 for value in self.noise_levels):
            raise ValueError("noise_levels must be in [0, 5]")
        if not 0.001 <= self.coverage_floor <= 0.25:
            raise ValueError("coverage_floor must be in [0.001, 0.25]")
        if self.trials < 4 or self.trials > 512:
            raise ValueError("trials must be in [4, 512]")
        cells = (
            len(self.action_counts)
            * len(self.sample_sizes)
            * len(self.shift_strengths)
            * len(self.noise_levels)
        )
        if cells > 128:
            raise ValueError("mechanism sweep exceeds the 128-cell budget")

    @classmethod
    def from_dict(cls, payload: dict) -> "MechanismSweepConfig":
        allowed = {
            "action_counts", "sample_sizes", "shift_strengths", "noise_levels",
            "coverage_floor", "scenario_seed", "trials",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown mechanism config fields: {sorted(unknown)}")
        converted = dict(payload)
        for key in ("action_counts", "sample_sizes", "shift_strengths", "noise_levels"):
            if key in converted:
                converted[key] = tuple(converted[key])
        config = cls(**converted)
        config.validate()
        return config


HELDOUT_MECHANISM_CONFIG = MechanismSweepConfig(
    action_counts=(3, 6, 12),
    sample_sizes=(96, 384),
    shift_strengths=(0.25, 0.75, 1.0),
    noise_levels=(0.25, 1.25),
    coverage_floor=0.01,
    scenario_seed=20260713,
    trials=128,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def generate_scenarios(config: MechanismSweepConfig) -> tuple[OperatorScenario, ...]:
    """Generate a frozen factorial sweep without exposing conditional means to candidates."""
    config.validate()
    root = np.random.SeedSequence(config.scenario_seed)
    combinations = tuple(product(
        config.action_counts,
        config.sample_sizes,
        config.shift_strengths,
        config.noise_levels,
    ))
    scenarios = []
    for index, ((num_actions, sample_size, shift, noise), sequence) in enumerate(
        zip(combinations, root.spawn(len(combinations)), strict=True)
    ):
        rng = np.random.default_rng(sequence)
        means = rng.normal(0.0, 1.0, num_actions)
        behavior_raw = _softmax(rng.normal(0.0, 1.5, num_actions))
        uniform = np.full(num_actions, 1.0 / num_actions)
        floor_mix = min(1.0, config.coverage_floor * num_actions)
        behavior = (1.0 - floor_mix) * behavior_raw + floor_mix * uniform
        target_anchor = _softmax(-means + rng.normal(0.0, 0.25, num_actions))
        target = (1.0 - shift) * behavior + shift * target_anchor
        scenarios.append(OperatorScenario(
            name=(
                f"m{index:03d}_a{num_actions}_n{sample_size}_"
                f"d{shift:.2f}_z{noise:.2f}"
            ),
            conditional_means=tuple(float(value) for value in means),
            behavior_probs=tuple(float(value) for value in behavior),
            target_probs=tuple(float(value) for value in target),
            noise_std=float(noise),
            sample_size=int(sample_size),
        ))
    return tuple(scenarios)


def _bucket(scenarios: Iterable[OperatorScenario], results: dict) -> dict:
    buckets: dict[str, list[float]] = {"matched": [], "moderate": [], "heavy": []}
    for scenario in scenarios:
        shift = float(scenario.name.split("_d", 1)[1].split("_", 1)[0])
        key = "matched" if shift == 0.0 else "moderate" if shift < 0.8 else "heavy"
        buckets[key].append(float(results[scenario.name]["mse"]))
    return {
        name: {"mean_mse": float(np.mean(values)), "cells": len(values)}
        for name, values in buckets.items()
        if values
    }


def run_mechanism_benchmark(
    estimator: OperatorEstimator,
    *,
    seed: int,
    config: MechanismSweepConfig = MechanismSweepConfig(),
) -> dict:
    scenarios = generate_scenarios(config)
    raw = run_operator_benchmark(
        estimator, seed=seed, trials=config.trials, scenarios=scenarios
    )
    mses = np.asarray([value["mse"] for value in raw["scenarios"].values()])
    worst_mse = float(np.max(mses))
    robust_risk = float(np.mean(mses) + 0.25 * worst_mse)
    return {
        **raw,
        "primary_score": -robust_risk,
        "robust_risk": robust_risk,
        "worst_mse": worst_mse,
        "num_cells": len(scenarios),
        "buckets": _bucket(scenarios, raw["scenarios"]),
    }
