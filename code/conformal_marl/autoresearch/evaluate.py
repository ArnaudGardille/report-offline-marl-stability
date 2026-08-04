"""CLI entry point for frozen candidate evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Import order is load-bearing and must not be sorted.
#
# The frozen scorers and the sampler registry are imported FIRST, so that the
# baseline snapshot in ``conformal_marl.samplers`` is sealed before any generated
# candidate module executes. With the candidate imported first, its module-level code
# would run while ``conformal_marl.samplers`` was still absent from ``sys.modules``,
# and could install a substitute that the frozen benchmarks would then import as
# their own dependency.
#
# ``ruff``'s isort rule (I001) is not enabled for this project; if it is ever turned
# on, this block needs an explicit ``# isort: skip_file`` rather than a reordering.
# ``test_frozen_scorers_are_imported_before_candidate_code`` pins the order.
# ---------------------------------------------------------------------------
from conformal_marl.samplers import assert_baseline_integrity  # isort: skip
from conformal_marl.autoresearch.frozen.learning_benchmark import (  # isort: skip
    DEFAULT_LEARNING_SCENARIOS,
    HELDOUT_LEARNING_SCENARIOS,
    run_learning_benchmark,
)
from conformal_marl.autoresearch.frozen.mechanism_benchmark import (  # isort: skip
    MechanismSweepConfig,
    run_mechanism_benchmark,
)
from conformal_marl.autoresearch.frozen.operator_benchmark import (  # isort: skip
    run_operator_benchmark,
)

# Only now the untrusted module. Importing it runs its ``@register_sampler``
# decorator, which is how the learning tier resolves ``"autoresearch_candidate"`` by
# name. Keep this import even if ``estimate_operator`` stops being used here --
# dropping it would leave the learning tier unable to find the candidate's sampler.
from conformal_marl.autoresearch.candidate import estimate_operator  # isort: skip

# Fail fast if importing the candidate already displaced a control sampler. This is a
# snapshot, not a guarantee -- the benchmarks re-check before each control arm,
# because the registry is consulted at call time.
assert_baseline_integrity()

DEFAULT_NUM_SEEDS = 8
DEFAULT_TRIALS = 128

# The search profile screens candidates; the held-out profile confirms the single
# frozen finalist on a shifted task distribution. Selecting by name through this table
# (rather than by truth-testing a string) means an unrecognised profile raises instead
# of silently falling back to the search benchmark and being recorded as a held-out
# result.
LEARNING_PROFILES = {
    "search": DEFAULT_LEARNING_SCENARIOS,
    "heldout": HELDOUT_LEARNING_SCENARIOS,
}


def evaluate_candidate(
    *,
    tier: str,
    seed: int,
    trials: int,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    mechanism_config: dict | None = None,
    learning_profile: str = "search",
    reference_controls: tuple[str, ...] = (),
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
        try:
            scenarios = LEARNING_PROFILES[learning_profile]
        except KeyError:
            raise ValueError(
                f"unknown learning profile {learning_profile!r}; "
                f"expected one of {sorted(LEARNING_PROFILES)}"
            ) from None
        return run_learning_benchmark(
            "autoresearch_candidate",
            seed=seed,
            num_seeds=num_seeds,
            scenarios=scenarios,
            reference_controls=reference_controls,
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
    parser.add_argument(
        "--learning-profile", choices=sorted(LEARNING_PROFILES), default="search"
    )
    parser.add_argument(
        "--reference-control",
        action="append",
        default=[],
        dest="reference_controls",
        metavar="SAMPLER",
        help=(
            "Additional control to measure the candidate against on the same paired "
            "seeds. Reported only -- it never gates promotion."
        ),
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
        learning_profile=args.learning_profile,
        reference_controls=tuple(args.reference_controls),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
