"""Prepare a constrained AI Scientist search without invoking an LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from conformal_marl.autoresearch.orchestration import build_search_config, write_search_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrator", type=Path, required=True)
    parser.add_argument("--base-ref", default="research-ai-scientist-integration")
    parser.add_argument("--output", type=Path, default=Path("outputs/ai_scientist/config.json"))
    parser.add_argument("--results", type=Path, default=Path("outputs/ai_scientist/search"))
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--operator-trials", type=int, default=128)
    parser.add_argument("--learning-seeds", type=int, default=8)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    context = repository / "code/research/ai_scientist_task.md"
    config = build_search_config(
        repository=repository,
        orchestrator=args.orchestrator,
        base_ref=args.base_ref,
        output_dir=args.results,
        task_context_file=context,
        max_nodes=args.max_nodes,
        branching_factor=args.branching_factor,
        operator_trials=args.operator_trials,
        learning_seeds=args.learning_seeds,
    )
    write_search_config(config, args.output)
    launcher = args.orchestrator.resolve() / "launch_constrained_bfts.py"
    print(f"Wrote {args.output.resolve()}")
    print(
        f"Launch with: python3 {launcher} --config {args.output.resolve()} "
        "--model claude-code/sonnet"
    )


if __name__ == "__main__":
    main()
