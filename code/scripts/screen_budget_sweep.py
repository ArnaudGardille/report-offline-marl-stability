"""How much training budget the scored learning scenario needs to carry signal.

`screen_power_check.py` shows the paired screen cannot resolve even a maximally
degraded sampler at the pinned STORM budget. That is a signal problem, not a
statistics problem: the greedy-return curve is almost everywhere zero, so there is
nothing for a paired test to compare. This script measures where the signal starts,
by running the *uniform* baseline alone at a range of `total_timesteps` and
reporting how much of the evaluation curve is non-zero and how large the achieved
return actually gets.

The output is what a recalibration decision needs: the cheapest budget at which the
scored task is learnable, and the true scale of its return (which the scenario's
`return_scale` divisor should match -- at the pinned budget it is off by ~25x, which
is why `minimum_score` and `support_tolerance` are meaningless there).

Run from `code/`:

    uv run python scripts/screen_budget_sweep.py --num-seeds 4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from conformal_marl.autoresearch.frozen.learning_benchmark import (
    DEFAULT_LEARNING_SCENARIOS,
    _run_sampler,
)

DEFAULT_BUDGETS = (16_384, 65_536, 262_144)


def _scored_scenario():
    for scenario in DEFAULT_LEARNING_SCENARIOS:
        if not scenario.support_gate:
            return scenario
    raise ValueError("no scored (non-gate) learning scenario found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="override eval_episodes (more episodes = less evaluation noise)")
    parser.add_argument("--output", type=Path, default=Path("outputs/screen_budget_sweep.json"))
    args = parser.parse_args()

    base = _scored_scenario()
    seed_values = np.random.SeedSequence(args.seed).generate_state(
        args.num_seeds, dtype=np.uint32
    )

    print(f"scored scenario: {base.name} (env {base.env_name})")
    print(f"pinned budget:   {base.config_overrides['total_timesteps']:,}")
    print(f"pinned return_scale: {base.return_scale}\n")
    print(
        f"{'timesteps':>10s} {'nonzero':>8s} {'max':>7s} {'final mean':>11s} "
        f"{'final sd':>9s} {'seeds>0':>8s} {'wall s':>8s}"
    )

    rows = []
    for budget in args.budgets:
        overrides = dict(base.config_overrides)
        overrides["total_timesteps"] = budget
        overrides["buffer_size"] = max(overrides["buffer_size"], budget)
        overrides["epsilon_anneal_time"] = min(
            overrides["epsilon_anneal_time"], max(budget // 2, 1)
        )
        if args.eval_episodes is not None:
            overrides["eval_episodes"] = args.eval_episodes
        scenario = replace(base, config_overrides=overrides)

        started = time.time()
        out = _run_sampler(scenario, "uniform", seed_values)
        elapsed = time.time() - started

        curve = np.asarray(out["returns"], dtype=np.float64)
        final = np.asarray(out["end_greedy_return"], dtype=np.float64)
        row = {
            "total_timesteps": budget,
            "eval_episodes": overrides["eval_episodes"],
            "nonzero_eval_fraction": float((curve != 0.0).mean()),
            "curve_max": float(curve.max()),
            "final_return_mean": float(final.mean()),
            "final_return_std": float(final.std(ddof=1)) if final.size > 1 else 0.0,
            "seeds_with_positive_final": int((final > 0.0).sum()),
            "num_seeds": args.num_seeds,
            "wall_seconds": elapsed,
        }
        rows.append(row)
        print(
            f"{budget:>10,} {row['nonzero_eval_fraction']:>7.1%} "
            f"{row['curve_max']:>7.3f} {row['final_return_mean']:>11.4f} "
            f"{row['final_return_std']:>9.4f} "
            f"{row['seeds_with_positive_final']:>4d}/{args.num_seeds:<3d} {elapsed:>8.1f}"
        )

    best = max(rows, key=lambda r: r["final_return_mean"])
    print(
        f"\nreturn_scale should track the achievable return: observed max "
        f"{best['curve_max']:.3f} at {best['total_timesteps']:,} steps, "
        f"against the pinned divisor {base.return_scale}."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scenario": base.name,
                "pinned_total_timesteps": base.config_overrides["total_timesteps"],
                "pinned_return_scale": base.return_scale,
                "num_seeds": args.num_seeds,
                "seed": args.seed,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
