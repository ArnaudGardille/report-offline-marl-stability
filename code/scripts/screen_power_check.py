"""Fixed-design sensitivity check for the paired learning screen.

`autoresearch/frozen/learning_benchmark.py` decides whether an automatically
generated replay candidate improved sample efficiency. PR #34 shipped it with an
explicit caveat: at the pinned STORM budget the reward is so sparse that the screen
may not be able to separate candidates at all. A screen that cannot see a *bad*
sampler cannot be trusted when it says a sampler is good, so this script measures
that directly instead of assuming it.

Two families of probe sampler are registered here (never in the frozen package --
they are measurement instruments, not candidates):

* ``probe_uniform_reseed`` is distributionally identical to ``uniform`` but draws on
  an independently folded key. Its paired delta measures the empirical variability
  between two equal-law replay draws on this fixed seed set.
* ``probe_collapse_k`` draws the usual uniform batch and then keeps only ``k``
  distinct rows, tiling them back to the trained batch size. ``k = batch_size`` is
  the identity; smaller ``k`` monotonically destroys the diversity of the replay
  batch without touching environments, rewards, seeds or budgets. ``k = 1`` trains
  every gradient step on a single repeated transition and is about as degraded as a
  sampler can be while still running.

The check is informative only if degraded probes separate from the equal-law probe on
the paired seeds. It is a stress test, not a general power guarantee. Run from ``code/``:

    uv run python scripts/screen_power_check.py --num-seeds 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.autoresearch.frozen.learning_benchmark import (
    DEFAULT_LEARNING_SCENARIOS,
    _run_sampler,
    run_learning_benchmark,
)
from conformal_marl.samplers import register_sampler, uniform_sampler

COLLAPSE_LEVELS = (1, 4, 16)


@register_sampler("probe_uniform_reseed")
def _uniform_reseed(buffer, buffer_state, rng, ctx, batch_size):
    """Uniform replay on an independently folded key: the same law, a different draw."""
    return uniform_sampler(
        buffer, buffer_state, jax.random.fold_in(rng, 0x9E37), ctx, batch_size
    )


def _make_collapse_sampler(k: int):
    def _collapse(buffer, buffer_state, rng, ctx, batch_size):
        batch, weights, _ = uniform_sampler(buffer, buffer_state, rng, ctx, batch_size)
        size = batch.reward.shape[0]
        # Keep the first k rows of an otherwise uniform draw and tile them back up to
        # the trained batch size. The batch keeps its shape (so the loss and every
        # downstream shape check are untouched) and loses its diversity.
        idx = jnp.arange(size) % min(k, size)
        collapsed = jax.tree_util.tree_map(lambda leaf: leaf[idx], batch)
        return collapsed, weights, {}

    return _collapse


for _k in COLLAPSE_LEVELS:
    register_sampler(f"probe_collapse_{_k}")(_make_collapse_sampler(_k))


def _scenarios(total_timesteps: int | None, return_scale: float | None,
               eval_episodes: int | None) -> tuple:
    """The frozen scenarios, optionally re-budgeted for a recalibration trial.

    Passing no override reproduces the pinned benchmark exactly. The overrides exist
    so a candidate recalibration can be *tested* before it is written into the frozen
    package -- changing the constants and hoping is not a power check.
    """
    if total_timesteps is None and return_scale is None and eval_episodes is None:
        return DEFAULT_LEARNING_SCENARIOS
    out = []
    for scenario in DEFAULT_LEARNING_SCENARIOS:
        if scenario.support_gate:
            out.append(scenario)
            continue
        overrides = dict(scenario.config_overrides)
        if total_timesteps is not None:
            overrides["total_timesteps"] = total_timesteps
            overrides["buffer_size"] = max(overrides["buffer_size"], total_timesteps)
            overrides["epsilon_anneal_time"] = min(
                overrides["epsilon_anneal_time"], max(total_timesteps // 2, 1)
            )
        if eval_episodes is not None:
            overrides["eval_episodes"] = eval_episodes
        out.append(
            replace(
                scenario,
                config_overrides=overrides,
                return_scale=scenario.return_scale if return_scale is None else return_scale,
            )
        )
    return tuple(out)


def _sparsity_report(scenarios: tuple, num_seeds: int, seed: int) -> dict:
    """How much signal the scored curves carry at all, before any comparison."""
    seed_values = np.random.SeedSequence(seed).generate_state(num_seeds, dtype=np.uint32)
    report = {}
    for scenario in scenarios:
        out = _run_sampler(scenario, "uniform", seed_values)
        curve = np.asarray(out["returns"], dtype=np.float64)
        final = np.asarray(out["end_greedy_return"], dtype=np.float64)
        report[scenario.name] = {
            "eval_points_per_seed": int(curve.shape[1]),
            "nonzero_eval_fraction": float((curve != 0.0).mean()),
            "curve_max": float(curve.max()),
            "return_scale": scenario.return_scale,
            "scaled_curve_max": float(curve.max() / scenario.return_scale),
            "final_return_mean": float(final.mean()),
            "final_return_std": float(final.std(ddof=1)) if final.size > 1 else 0.0,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="re-budget the scored scenario (recalibration trial)")
    parser.add_argument("--return-scale", type=float, default=None,
                        help="override the scored scenario's return_scale divisor")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="override eval_episodes on the scored scenario")
    parser.add_argument("--output", type=Path, default=Path("outputs/screen_power_check.json"))
    args = parser.parse_args()

    scenarios = _scenarios(args.total_timesteps, args.return_scale, args.eval_episodes)
    probes = ["probe_uniform_reseed"] + [f"probe_collapse_{k}" for k in COLLAPSE_LEVELS]

    print(f"jax backend: {jax.default_backend()}  seeds: {args.num_seeds}")
    if scenarios is not DEFAULT_LEARNING_SCENARIOS:
        scored = [s for s in scenarios if not s.support_gate][0]
        print(
            f"RECALIBRATION TRIAL: {scored.name} at "
            f"{scored.config_overrides['total_timesteps']:,} steps, "
            f"return_scale={scored.return_scale}, "
            f"eval_episodes={scored.config_overrides['eval_episodes']}"
        )
    print()
    print("== signal available in the scored curves (uniform baseline) ==")
    sparsity = _sparsity_report(scenarios, args.num_seeds, args.seed)
    for name, stats in sparsity.items():
        print(
            f"  {name:34s} nonzero {stats['nonzero_eval_fraction']:6.1%} of "
            f"{stats['eval_points_per_seed']:3d} pts/seed | max {stats['curve_max']:.3f} "
            f"(scaled {stats['scaled_curve_max']:.4f}) | final "
            f"{stats['final_return_mean']:.3f} +/- {stats['final_return_std']:.3f}"
        )

    results = {}
    print("\n== paired screen vs uniform ==")
    for probe in probes:
        out = run_learning_benchmark(
            probe, seed=args.seed, num_seeds=args.num_seeds, scenarios=scenarios
        )
        results[probe] = out
        scored = {
            name: payload
            for name, payload in out["scenarios"].items()
            if not payload["support_gate"]
        }
        detail = "  ".join(
            f"{name}: d={payload['mean_utility_delta']:+.5f} "
            f"sem={payload['utility_sem']:.5f} "
            f"lcb={payload['lower_screening_bound']:+.5f}"
            for name, payload in scored.items()
        )
        print(
            f"  {probe:24s} score={out['primary_score']:+.5f} "
            f"status={out['status']:18s} {detail}"
        )

    # Compare deliberately degraded replay with an equal-law draw on each seed. Their
    # shared uniform control then cancels before estimating the standard error.
    null = results["probe_uniform_reseed"]
    verdict: dict[str, dict] = {}
    print("\n== verdict ==")

    gate_minimum = 0.001  # the learning stage's minimum_score in orchestration.py
    null_passes = null["primary_score"] >= gate_minimum
    # A one-sided lower bound on a zero-mean delta is negative by construction, so a
    # equal-law sampler *should* fail the gate. The approximation below shows how large
    # an improvement must be before the bound clears the threshold at this variance.
    scored_null = [p for p in null["scenarios"].values() if not p["support_gate"]]
    mde = gate_minimum + null["critical_value"] * float(
        np.mean([p["utility_sem"] for p in scored_null])
    )
    print(
        f"  gate calibration: a statistically null sampler scores "
        f"{null['primary_score']:+.5f} vs minimum_score={gate_minimum} -> "
        f"{'passes' if null_passes else 'rejected (expected: the bound is one-sided)'}"
    )
    print(
        f"  approximate promotion threshold under observed equal-law variance: "
        f"{mde:+.5f} scaled utility at {args.num_seeds} seeds"
    )

    ordering = [(p, results[p]["primary_score"]) for p in probes]
    ranked = [name for name, _ in sorted(ordering, key=lambda kv: kv[1], reverse=True)]
    expected = ["probe_uniform_reseed"] + [
        f"probe_collapse_{k}" for k in sorted(COLLAPSE_LEVELS, reverse=True)
    ]
    print(f"  ranking by score (best first): {ranked}")
    print(f"  expected diagnostic ordering:  {expected}")
    print(f"  -> monotone in degradation: {'yes' if ranked == expected else 'NO'}")

    for scenario_name, null_payload in null["scenarios"].items():
        floor = null_payload["utility_sem"]
        null_delta = null_payload["mean_utility_delta"]
        crit = null["critical_value"]
        print(
            f"\n  {scenario_name}: null delta {null_delta:+.5f}, "
            f"noise floor (SEM) {floor:.5f}"
        )
        verdict[scenario_name] = {"null_delta": null_delta, "noise_sem": floor, "probes": {}}
        for probe in probes[1:]:
            payload = results[probe]["scenarios"][scenario_name]
            delta = payload["mean_utility_delta"]
            sem = payload["utility_sem"]
            null_paired = np.asarray(null_payload["paired_utility_delta"], dtype=np.float64)
            probe_paired = np.asarray(payload["paired_utility_delta"], dtype=np.float64)
            paired_separation = null_paired - probe_paired
            separation = float(paired_separation.mean())
            separation_sem = float(
                paired_separation.std(ddof=1) / np.sqrt(paired_separation.size)
            )
            spread = crit * separation_sem
            resolved = separation > spread
            ratio = abs(delta) / floor if floor > 0 else float("inf")
            verdict[scenario_name]["probes"][probe] = {
                "delta": delta,
                "sem": sem,
                "delta_vs_noise_floor": ratio,
                "separation_from_null": separation,
                "separation_sem": separation_sem,
                "required_separation": spread,
                "resolved": bool(resolved),
            }
            print(
                f"    {probe:22s} delta={delta:+.5f} ({ratio:5.2f} x floor) "
                f"gap-to-null={separation:+.5f} needs>{spread:.5f} -> "
                f"{'RESOLVED' if resolved else 'not resolvable'}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "num_seeds": args.num_seeds,
                "seed": args.seed,
                "sparsity": sparsity,
                "probes": results,
                "verdict": verdict,
                "gate_minimum": gate_minimum,
                "null_passes_gate": bool(null_passes),
                "approximate_promotion_threshold": mde,
                "score_ranking": ranked,
                "score_ranking_expected": expected,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
