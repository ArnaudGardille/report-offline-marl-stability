"""CLI entry point for frozen candidate evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conformal_marl.autoresearch.frozen.causal_drift_benchmark import (
    HELDOUT_CAUSAL_DRIFT_CONFIG,
    SEARCH_CAUSAL_DRIFT_CONFIG,
    run_causal_drift_benchmark,
)
from conformal_marl.autoresearch.frozen.learning_benchmark import (
    HELDOUT_LEARNING_SCENARIOS,
    run_learning_benchmark,
)
from conformal_marl.autoresearch.frozen.mechanism_benchmark import (
    HELDOUT_MECHANISM_CONFIG,
    MechanismSweepConfig,
    run_mechanism_benchmark,
)
from conformal_marl.autoresearch.frozen.operator_benchmark import run_operator_benchmark
from conformal_marl.samplers import get_sampler, uniform_sampler

# Bind the trusted scorer and baseline modules before candidate import-time code runs.
from conformal_marl.autoresearch.candidate import estimate_operator, sample_replay  # noqa: E402,F401

if get_sampler("uniform") is not uniform_sampler:
    raise RuntimeError("candidate import changed the frozen uniform replay baseline")


def evaluate_candidate(
    *,
    tier: str,
    seed: int,
    trials: int,
    num_seeds: int = 8,
    mechanism_config: dict | None = None,
    mechanism_profile: str = "search",
    learning_profile: str = "search",
    control_samplers: tuple[str, ...] = ("uniform",),
) -> dict:
    if tier == "operator":
        return run_operator_benchmark(estimate_operator, seed=seed, trials=trials)
    if tier == "mechanism":
        # The experiment designer may choose bounded sweep axes, but never the trial
        # count supplied by the frozen evaluator command.
        if mechanism_config is not None and mechanism_profile != "search":
            raise ValueError("custom mechanism config is search-only")
        payload = (
            dict(mechanism_config)
            if mechanism_config is not None
            else {
                "action_counts": list(HELDOUT_MECHANISM_CONFIG.action_counts),
                "sample_sizes": list(HELDOUT_MECHANISM_CONFIG.sample_sizes),
                "shift_strengths": list(HELDOUT_MECHANISM_CONFIG.shift_strengths),
                "noise_levels": list(HELDOUT_MECHANISM_CONFIG.noise_levels),
                "coverage_floor": HELDOUT_MECHANISM_CONFIG.coverage_floor,
                "scenario_seed": HELDOUT_MECHANISM_CONFIG.scenario_seed,
            }
            if mechanism_profile == "heldout"
            else {}
        )
        payload["trials"] = trials
        config = MechanismSweepConfig.from_dict(payload)
        return run_mechanism_benchmark(estimate_operator, seed=seed, config=config)
    if tier == "learning":
        scenarios = (
            HELDOUT_LEARNING_SCENARIOS
            if learning_profile == "heldout"
            else None
        )
        return run_learning_benchmark(
            "autoresearch_candidate",
            seed=seed,
            num_seeds=num_seeds,
            control_samplers=control_samplers,
            **({"scenarios": scenarios} if scenarios is not None else {}),
        )
    if tier == "causal_drift":
        config = (
            HELDOUT_CAUSAL_DRIFT_CONFIG
            if learning_profile == "heldout"
            else SEARCH_CAUSAL_DRIFT_CONFIG
        )
        return run_causal_drift_benchmark(
            "autoresearch_candidate",
            seed=seed,
            num_seeds=num_seeds,
            config=config,
            control_samplers=control_samplers,
        )
    raise ValueError(f"unknown evaluation tier {tier!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one constrained replay candidate")
    parser.add_argument(
        "--tier",
        choices=("operator", "mechanism", "causal_drift", "learning"),
        default="operator",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--mechanism-config", type=Path)
    parser.add_argument(
        "--mechanism-profile", choices=("search", "heldout"), default="search"
    )
    parser.add_argument(
        "--learning-profile", choices=("search", "heldout"), default="search"
    )
    parser.add_argument(
        "--control-sampler",
        action="append",
        dest="control_samplers",
        help="Repeat to require improvement over multiple fixed controls.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mechanism_config = (
        json.loads(args.mechanism_config.read_text(encoding="utf-8"))
        if args.mechanism_config
        else None
    )
    result = evaluate_candidate(
        tier=args.tier,
        seed=args.seed,
        trials=args.trials,
        num_seeds=args.num_seeds,
        mechanism_config=mechanism_config,
        mechanism_profile=args.mechanism_profile,
        learning_profile=args.learning_profile,
        control_samplers=tuple(args.control_samplers or ("uniform",)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
