"""CLI entry point for frozen candidate evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Importing the module runs its ``@register_sampler`` decorator, which is how the
# learning tier resolves ``"autoresearch_candidate"`` by name. Keep this import even
# if ``estimate_operator`` stops being used here -- dropping it would leave the
# learning tier unable to find the candidate's sampler.
from conformal_marl.autoresearch.candidate import estimate_operator
from conformal_marl.autoresearch.frozen.learning_benchmark import run_learning_benchmark
from conformal_marl.autoresearch.frozen.mechanism_benchmark import (
    MechanismSweepConfig,
    run_mechanism_benchmark,
)
from conformal_marl.autoresearch.frozen.operator_benchmark import run_operator_benchmark

DEFAULT_NUM_SEEDS = 8
DEFAULT_TRIALS = 128


def evaluate_candidate(
    *,
    tier: str,
    seed: int,
    trials: int,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    mechanism_config: dict | None = None,
) -> dict:
    if tier == "operator":
        return run_operator_benchmark(estimate_operator, seed=seed, trials=trials)
    if tier == "mechanism":
        # The experiment designer may choose bounded sweep axes, but never the trial
        # count supplied by the frozen evaluator command.
        payload = dict(mechanism_config or {})
        payload["trials"] = trials
        config = MechanismSweepConfig.from_dict(payload)
        return run_mechanism_benchmark(estimate_operator, seed=seed, config=config)
    if tier == "learning":
        return run_learning_benchmark(
            "autoresearch_candidate", seed=seed, num_seeds=num_seeds
        )
    raise ValueError(f"unknown evaluation tier {tier!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one constrained replay candidate")
    parser.add_argument(
        "--tier", choices=("operator", "mechanism", "learning"), default="operator"
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--mechanism-config", type=Path)
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
