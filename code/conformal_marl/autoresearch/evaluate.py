"""CLI entry point for frozen candidate evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conformal_marl.autoresearch.candidate import estimate_operator
from conformal_marl.autoresearch.frozen.operator_benchmark import run_operator_benchmark


def evaluate_candidate(*, tier: str, seed: int, trials: int) -> dict:
    if tier != "operator":
        raise ValueError(f"unknown evaluation tier {tier!r}")
    return run_operator_benchmark(estimate_operator, seed=seed, trials=trials)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one constrained replay candidate")
    parser.add_argument("--tier", choices=("operator",), default="operator")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_candidate(tier=args.tier, seed=args.seed, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
