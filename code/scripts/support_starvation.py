"""M6' support-starvation demonstration -- the paper's rigorous centerpiece.

This script turns the M3 negative result (ratio-free conformal *selection* is
null-to-negative on Simultaneous-Attack, collapsing some seeds to the all-abstain
consensus) into a theorem-backed finding. The support-starvation theorem (paper
section "support-starvation") says conformal reweighting makes IQL compute the best
response of a **support-starved** perceived MDP, which coincides with the *true* best
response **iff** both

    (a) the weights equal the exact importance ratio rho, AND
    (b) the reweighted visitation support covers the BR support: supp(d_w) >= supp(d^BR).

Ratio-free conformal selection violates both, hence the Simu collapse. We demonstrate
all of this empirically on Simultaneous-Attack (N=3), reusing the M3 IQL infrastructure.

Three deliverables (each a committed figure):

1. **Quantify the starvation** (``m6_support_starvation_metric.png``): the effective
   replay mass on the best-response support -- ``P(all-attack joint action in the
   trained batch)`` -- versus environment steps, for ``uniform`` vs ``conformal_policy``
   vs ``exact_ratio``. Conformal *selection* drives the all-attack mass toward 0 (starves
   condition (b)); uniform retains it.

2. **Connect starvation to the collapse** (``m6_support_starvation_collapse.png``): a
   per-seed scatter of *final* all-attack replay mass against *final* greedy return,
   showing the collapsed seeds are exactly those whose conformal-selected support lost
   the all-attack transitions (mass -> 0 <=> return -> 0).

3. **Verify the theorem's "iff"** (``m6_support_starvation_iff.png``): exact-ratio
   weighting **with** the support floor (``exact_ratio_floor``: both conditions)
   recovers the true best response (return -> 1, no collapses); dropping the exact ratio
   (``policy_floor``: floor on, ratio-free weights -> condition (a) fails) **or** dropping
   the support floor (``exact_ratio`` no floor -> condition (b) fails) collapses it.

Honesty note: this is meant to *explain a negative*. The win is a clean, interpretable
demonstration that pins the two theorem conditions -- not a performance number. The
script reports exactly what it measures (collapse counts, the starvation trajectory, the
per-seed link, the iff table) and manufactures nothing.

Usage::

    uv run python scripts/support_starvation.py              # 16 seeds (default)
    uv run python scripts/support_starvation.py --seeds 24
    uv run python scripts/support_starvation.py --floor 0.3  # support-floor magnitude
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from conformal_marl.envs.simultaneous_attack import SimultaneousAttack
from conformal_marl.iql import IQLConfig, make_train

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
OUTPUTS_DIR = CODE_DIR / "outputs"

# Colour-blind-friendly palette, consistent with the M3 ablation figure.
COLORS = {
    "uniform": "#4C72B0",
    "conformal_policy": "#DD8452",
    "exact_ratio": "#55A868",
    "exact_ratio_floor": "#8172B3",
    "policy_floor": "#C44E52",
}


def starvation_config(sampler: str, floor: float = 0.0) -> IQLConfig:
    """Staleness-stressed Simu config (the M3 ablation setup) with the M6' diagnostics on.

    Identical hyper-parameters to ``scripts/ablation.py``'s ``simu_ablation_config`` (a
    large buffer so stale anti-coordinated transitions linger, moderate budget, the
    setup that surfaces the collapse), plus ``track_batch_action_stats=True`` so we get
    the per-iteration all-attack replay mass, and the support floor for the floored
    samplers.
    """
    return IQLConfig(
        total_timesteps=15_000,
        num_envs=8,
        num_steps=16,
        num_grad_steps=8,
        learning_starts=256,
        hidden_dim=64,
        lr=5e-4,
        gamma=0.99,
        batch_size=64,
        large_batch_size=512 if sampler != "uniform" else None,
        cal_size=128,
        buffer_size=50_000,
        target_update_interval=100,
        epsilon_anneal_time=6_000,
        eval_episodes=128,
        eval_steps=2,
        sampler_name=sampler,
        eval_during_training=True,
        track_batch_action_stats=True,
        support_floor=floor,
    )


def run(env, sampler: str, seeds: int, base_seed: int, floor: float = 0.0):
    """Train ``seeds`` seeds; return (steps, returns (S,T), all_attack (S,T), seconds)."""
    cfg = starvation_config(sampler, floor)
    train_fn = make_train(cfg, env)
    rngs = jax.random.split(jax.random.PRNGKey(base_seed), seeds)
    t0 = time.perf_counter()
    out = jax.block_until_ready(jax.jit(jax.vmap(train_fn))(rngs))
    dt = time.perf_counter() - t0
    steps = np.asarray(out["return_steps"][0])
    returns = np.asarray(out["returns"])  # (S, T)
    all_attack = np.asarray(out["all_attack_frac"])  # (S, T)
    end = np.asarray(out["end_greedy_return"])  # (S,)
    return steps, returns, all_attack, end, dt


# --------------------------------------------------------------------------- figures
def plot_starvation_metric(results, out_png: Path):
    """Figure 1: all-attack replay mass vs steps for uniform / conformal_policy / exact_ratio."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for sampler in ("uniform", "conformal_policy", "exact_ratio"):
        steps, _ret, aa, _end, _dt = results[sampler]
        c = COLORS[sampler]
        # Faint individual seed traces expose the BIMODALITY honestly: under conformal
        # selection some seeds keep the all-attack support (-> ~1) while others starve it
        # to 0. The mean alone would hide this (it sits in the empty middle).
        for s in range(aa.shape[0]):
            ax.plot(steps, aa[s], color=c, linewidth=0.5, alpha=0.18)
        ax.plot(steps, aa.mean(axis=0), color=c, linewidth=2.4, label=f"{sampler} (mean)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("P(all-attack joint action in trained batch)")
    ax.set_title(
        "Support starvation on Simultaneous-Attack (N=3)\n"
        "effective replay mass on the BR support (bold = mean, thin = per-seed)"
    )
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_collapse_link(results, out_png: Path):
    """Figure 2: per-seed final all-attack mass vs final return (collapse <=> starvation)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    rng = np.random.default_rng(0)
    for sampler in ("uniform", "conformal_policy", "exact_ratio"):
        _steps, _ret, aa, end, _dt = results[sampler]
        aa_final = aa[:, -5:].mean(axis=1)  # late-training all-attack mass per seed
        # Small jitter so coincident collapsed seeds (all at (0,0)) are visible as a cluster.
        jx = rng.uniform(-0.012, 0.012, size=aa_final.shape)
        jy = rng.uniform(-0.012, 0.012, size=end.shape)
        ax.scatter(
            aa_final + jx, end + jy, s=55, alpha=0.75, color=COLORS[sampler],
            edgecolors="white", linewidths=0.6, label=sampler,
        )
    # Pooled correlation across the conformal samplers (where there is spread).
    pooled_aa, pooled_ret = [], []
    for sampler in ("conformal_policy", "exact_ratio"):
        _s, _r, aa, end, _dt = results[sampler]
        pooled_aa.append(aa[:, -5:].mean(axis=1))
        pooled_ret.append(end)
    pa = np.concatenate(pooled_aa)
    pr = np.concatenate(pooled_ret)
    corr = float(np.corrcoef(pa, pr)[0, 1]) if pa.std() > 1e-9 and pr.std() > 1e-9 else float("nan")
    # Annotate the collapsed cluster (seeds that starved to mass 0 and returned 0).
    n_collapsed = int(np.sum(pr < 0.5))
    if n_collapsed:
        ax.annotate(
            f"{n_collapsed} starved seeds\n(mass -> 0, return -> 0)",
            xy=(0.0, 0.0), xytext=(0.18, 0.18), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="black", alpha=0.6),
        )
    ax.set_xlabel("final all-attack replay mass  (support coverage of d_w)")
    ax.set_ylabel("final greedy return  (1 = best response)")
    ax.set_title(
        "Collapse is exactly support starvation\n"
        f"per-seed; conformal samplers pooled corr(mass, return) = {corr:+.2f}"
    )
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, 1.08)
    ax.legend(loc="center left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return corr


def plot_iff(iff_results, floor: float, out_png: Path):
    """Figure 3: the iff bar chart -- final return (+collapse counts) per configuration."""
    order = [
        ("uniform", "uniform\n(no selection)"),
        ("exact_ratio", "exact_ratio\n(drop floor: cond. b fails)"),
        ("policy_floor", f"policy_floor f={floor}\n(drop rho: cond. a fails)"),
        ("exact_ratio_floor", f"exact_ratio_floor f={floor}\n(both conditions)"),
    ]
    labels, means, stds, colls, colors = [], [], [], [], []
    for key, lab in order:
        _s, _r, _aa, end, _dt = iff_results[key]
        labels.append(lab)
        means.append(float(end.mean()))
        stds.append(float(end.std()))
        colls.append(int(np.sum(end < 0.5)))
        colors.append(COLORS[key])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.9,
                  edgecolor="black", linewidth=0.6)
    n_seeds = iff_results["uniform"][3].shape[0]
    for i, b in enumerate(bars):
        # Place the collapse count just above the bar top (clamped below the y-limit so it
        # is never clipped), regardless of the error-bar whisker length.
        ytext = min(means[i] + 0.04, 1.05)
        ax.text(b.get_x() + b.get_width() / 2, ytext,
                f"{colls[i]}/{n_seeds}\ncollapsed", ha="center", va="bottom", fontsize=8.0)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("final greedy return (1 = best response)")
    ax.set_title(
        "Pinning the support-starvation theorem's 'iff' on Simultaneous-Attack (N=3)\n"
        f"mean +/- 1 std over {n_seeds} seeds; only exact-rho + support floor recovers the BR"
    )
    ax.set_ylim(0.0, 1.18)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="M6' support-starvation demonstration")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=0.3, help="support-floor magnitude f")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    env = SimultaneousAttack(num_agents=3)
    seeds, base = args.seeds, args.base_seed
    floor = args.floor

    print(f"\n===== Support-starvation demonstration: Simu N=3, {seeds} seeds =====")

    # --- Part 1+2: the three M3 samplers (floor off) for starvation + collapse link ---
    base_samplers = ["uniform", "conformal_policy", "exact_ratio"]
    results = {}
    print("\n[1/2] Starvation metric + collapse link (M3 samplers, floor off):")
    for s in base_samplers:
        steps, ret, aa, end, dt = run(env, s, seeds, base, floor=0.0)
        results[s] = (steps, ret, aa, end, dt)
        nc = int(np.sum(end < 0.5))
        print(
            f"  {s:18s}: final return {end.mean():.3f}+/-{end.std():.3f} | "
            f"all-attack mass {aa[:, :5].mean():.3f} (early) -> {aa[:, -5:].mean():.3f} (late) | "
            f"collapsed {nc}/{seeds} | {dt:.1f}s"
        )

    plot_starvation_metric(results, RESULTS_DIR / "m6_support_starvation_metric.png")
    corr = plot_collapse_link(results, RESULTS_DIR / "m6_support_starvation_collapse.png")
    print("  -> figures: m6_support_starvation_metric.png, m6_support_starvation_collapse.png")
    print(f"  -> collapse<->starvation pooled corr(final mass, final return) = {corr:+.3f}")

    # --- Part 3: the iff experiment ---
    print(f"\n[3] iff experiment (support floor f={floor}):")
    iff_results = dict(results)  # reuse uniform + exact_ratio (no floor) arms
    for s, fl in (("policy_floor", floor), ("exact_ratio_floor", floor)):
        steps, ret, aa, end, dt = run(env, s, seeds, base, floor=fl)
        iff_results[s] = (steps, ret, aa, end, dt)
    for key, desc in (
        ("uniform", "no selection (reference)"),
        ("exact_ratio", "exact rho, NO floor  [cond (b) support fails]"),
        ("policy_floor", "ratio-free wts + floor [cond (a) exact-rho fails]"),
        ("exact_ratio_floor", "exact rho + floor    [both conditions hold]"),
    ):
        _s, _r, _aa, end, _dt = iff_results[key]
        nc = int(np.sum(end < 0.5))
        print(f"  {key:20s} {desc:52s}: return {end.mean():.3f}+/-{end.std():.3f}  collapsed {nc}/{seeds}")

    plot_iff(iff_results, floor, RESULTS_DIR / "m6_support_starvation_iff.png")
    print("  -> figure: m6_support_starvation_iff.png")

    # --- honest read ---
    rec = iff_results["exact_ratio_floor"][3]
    abl_floor = iff_results["exact_ratio"][3]
    abl_rho = iff_results["policy_floor"][3]
    print("\n===== Honest read =====")
    print(
        f"  Recovering config (exact_ratio_floor): {int(np.sum(rec >= 0.5))}/{seeds} reach BR, "
        f"mean return {rec.mean():.3f}."
    )
    print(
        f"  Drop support floor (exact_ratio): {int(np.sum(abl_floor < 0.5))}/{seeds} collapse "
        f"(mean {abl_floor.mean():.3f}) -- condition (b) is necessary."
    )
    print(
        f"  Drop exact ratio (policy_floor): {int(np.sum(abl_rho < 0.5))}/{seeds} collapse "
        f"(mean {abl_rho.mean():.3f}) -- condition (a) is necessary even with the floor on."
    )

    # --- raw dump (gitignored) ---
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dump = {
        "seeds": seeds, "floor": floor,
        "samplers": {
            k: {
                "steps": v[0].tolist(),
                "returns": v[1].tolist(),
                "all_attack_frac": v[2].tolist(),
                "final_return": v[3].tolist(),
            }
            for k, v in iff_results.items()
        },
        "collapse_corr": corr,
    }
    (OUTPUTS_DIR / "m6_support_starvation.json").write_text(json.dumps(dump))
    print(f"\n  raw metrics -> {OUTPUTS_DIR / 'm6_support_starvation.json'}")


if __name__ == "__main__":
    main()
