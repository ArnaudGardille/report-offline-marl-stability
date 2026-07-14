"""DECISIVE EXPERIMENT: does detection-driven replay control improve *return* recovery?

This is the test the closed-loop demo could not give. The closed-loop measured a held-out
**TD error** after a one-shot re-fit on a frozen buffer; here we measure **greedy team
return** recovered over *online* training after a co-learner drift, under four replay regimes
(see :mod:`conformal_marl.recovery`):

* ``static``       -- uniform buffer, never forgets (control);
* ``detector``     -- our method (frozen-calibration conformal martingale: Channel-A alarm
  flushes the buffer, Channel-B drift rate sets the staleness half-life);
* ``oracle_reset`` -- flush at the *known* swap (upper bound of perfect detection);
* ``always_short`` -- permanent aggressive forgetting (is *detecting* better than always
  forgetting?).

Protocol: pretrain two good cooperative policies ``A`` (pre-swap) and ``B`` (post-swap) once,
shared across seeds (seeds vary only the recovery-phase RNG, so the inter-seed bands measure
how *reliably* each regime helps). For each seed and arm, the ego starts as ``A``, the
partners drift ``A -> B`` at a known step, and the ego re-learns. We report per-arm recovery
curves with inter-seed bands and a **pre-registered** paired verdict:

    detector beats static on final return / AUC with non-overlapping (>2 SEM) paired bands
        => POSITIVE; within bands => NULL; worse => NEGATIVE.
    detector vs oracle_reset = cost of detection delay; detector vs always_short = value of
    detecting over blindly forgetting.

We also print the **drop magnitude** at the swap: if it is negligible the env was too
forgiving and the test had no teeth (MPE simple_spread is symmetric; Overcooked is the
headline env -- see ``code/RUN_ON_GPU.md``). Honest by construction: whatever we find is
reported.

Usage::

    uv run python scripts/recovery_benchmark.py --smoke              # CPU/CI-time sanity
    uv run python scripts/recovery_benchmark.py --full --seeds 8     # decisive MPE run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

from conformal_marl.envs.registry import make_env
from conformal_marl.iql import IQLConfig, get_env_info, make_train
from conformal_marl.recovery import ARMS, RecoveryConfig, build_network, run_arm

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
OUTPUTS_DIR = CODE_DIR / "outputs"

ARM_STYLE = {
    "static": ("#4C72B0", "o", "static (uniform, never forgets)"),
    "detector": ("#C44E52", "s", "detector (alarm-reset + Channel-B half-life)"),
    "oracle_reset": ("#55A868", "^", "oracle reset (flush at known swap)"),
    "always_short": ("#8172B3", "D", "always-short (constant aggressive forgetting)"),
}


def pretrain(env, seed: int, hidden_dim: int, timesteps: int):
    """Train one parameter-shared IQL policy on the env; return (params, start_ret, end_ret)."""
    cfg = IQLConfig(
        total_timesteps=timesteps,
        num_envs=8,
        num_steps=32,
        num_grad_steps=8,
        learning_starts=512,
        net_type="ff",
        hidden_dim=hidden_dim,
        lr=5e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=50_000,
        epsilon_anneal_time=max(1, timesteps // 4),
        eval_episodes=16,
        eval_steps=64,
    )
    train = jax.jit(make_train(cfg, env))
    out = jax.block_until_ready(train(jax.random.PRNGKey(seed)))
    return (
        out["final_params"],
        float(out["start_greedy_return"]),
        float(out["end_greedy_return"]),
    )


def _curve_stats(curves: np.ndarray, pre_chunks: int):
    """Per-arm scalar summaries from a ``(seeds, chunks)`` curve array.

    Returns dict of per-seed scalars (so the caller can do paired stats):
      R0          -- pre-swap plateau (mean over pre-swap chunks), per seed;
      drop        -- R0 - return at the first post-swap chunk (>0 == a real drop);
      final       -- mean return over the last 5 post-swap chunks;
      auc         -- mean post-swap return (area under the recovery curve).
    """
    pre = curves[:, :pre_chunks]
    post = curves[:, pre_chunks:]
    R0 = pre.mean(axis=1)
    drop = R0 - post[:, 0]
    final = post[:, -5:].mean(axis=1)
    auc = post.mean(axis=1)
    return {"R0": R0, "drop": drop, "final": final, "auc": auc}


def _paired(a: np.ndarray, b: np.ndarray):
    """Paired delta a-b: (mean, sem, n_favour_a)."""
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return float(d.mean()), float(sem), int(np.sum(d > 0))


def _verdict(delta_mean: float, sem: float) -> str:
    if delta_mean > 2 * (sem + 1e-12):
        return "POSITIVE (detector beats static, >2 SEM)"
    if delta_mean < -2 * (sem + 1e-12):
        return "NEGATIVE (detector worse than static, >2 SEM)"
    return "NULL / inconclusive (within 2 SEM)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Recovery benchmark: detection-driven replay control vs static")
    ap.add_argument("--env", default="MPE_simple_spread_v3")
    ap.add_argument("--drift", choices=["policy", "scripted"], default="scripted",
                    help="post-swap drift: a different trained policy (mild on symmetric envs) "
                         "or a scripted broken partner (guaranteed real drop -> teeth)")
    ap.add_argument("--seeds", type=int, default=None, help="recovery seeds (default: scale-dependent)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--smoke", action="store_const", dest="scale", const="smoke")
    grp.add_argument("--full", action="store_const", dest="scale", const="full")
    ap.set_defaults(scale="smoke")
    ap.add_argument("--cpu", action="store_true", help="force JAX onto CPU")
    args = ap.parse_args()

    if args.scale == "smoke" or args.cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== recovery_benchmark: env={args.env} scale={args.scale} "
          f"| jax backend={jax.default_backend()} ===")

    if args.scale == "smoke":
        hidden, pre_ts = 64, 12_000
        rcfg = RecoveryConfig(
            hidden_dim=hidden, chunk_len=64, pre_chunks=3, post_chunks=6, n_cal=128,
            grad_steps=8, batch_size=32, buffer_size=2_000, eval_episodes=8,
            drift_mode=args.drift,
        )
        n_seeds = args.seeds or 2
    else:
        hidden, pre_ts = 128, 200_000
        rcfg = RecoveryConfig(
            hidden_dim=hidden, chunk_len=256, pre_chunks=12, post_chunks=40, n_cal=512,
            grad_steps=32, batch_size=64, buffer_size=20_000, eval_episodes=32,
            drift_mode=args.drift,
        )
        n_seeds = args.seeds or 8

    env, _ = make_env(args.env)
    info = get_env_info(env)
    _network, apply = build_network(info, rcfg)

    # Pretrain the two shared policies (A = pre-swap regime, B = post-swap drift).
    print(f"\n[pretrain] two policies on {args.env} ({pre_ts} steps each)...")
    t0 = time.perf_counter()
    params_A, a0, a1 = pretrain(env, 0, hidden, pre_ts)
    params_B, b0, b1 = pretrain(env, 1, hidden, pre_ts)
    print(f"  A: greedy return {a0:.2f} -> {a1:.2f}")
    print(f"  B: greedy return {b0:.2f} -> {b1:.2f}   ({time.perf_counter()-t0:.1f}s)")

    # Run all arms x seeds.
    results = {arm: [] for arm in ARMS}
    for seed in range(n_seeds):
        for arm in ARMS:
            ts = time.perf_counter()
            r = run_arm(arm, env, info, apply, params_A, params_B, rcfg, seed)
            results[arm].append(r)
            tail = (f" alarm@{r['alarm_step']}" if arm == "detector" else "")
            print(f"  seed {seed} {arm:13s}: final={np.mean(r['curve'][-5:]):.2f} "
                  f"({time.perf_counter()-ts:.1f}s){tail}")

    # Aggregate curves -> (seeds, chunks).
    curves = {arm: np.array([r["curve"] for r in results[arm]]) for arm in ARMS}
    steps = np.array(results["static"][0]["steps"])
    swap_step = results["static"][0]["swap_step"]
    stats = {arm: _curve_stats(curves[arm], rcfg.pre_chunks) for arm in ARMS}

    # ---- summary + pre-registered verdict ----
    print("\n================= RECOVERY SUMMARY =================")
    print(f"  swap at env step {swap_step}; {n_seeds} seeds; env {args.env}")
    drop = stats["static"]["drop"]
    print(f"  drop at swap (static):    {drop.mean():+.3f} +/- {drop.std():.3f}  "
          f"(R0={stats['static']['R0'].mean():.2f}); "
          f"{'TEETH' if drop.mean() > 0.5 else 'WEAK DROP -- env may be too forgiving'}")
    for arm in ARMS:
        f = stats[arm]["final"]
        a = stats[arm]["auc"]
        print(f"  {arm:13s}: final {f.mean():.3f} +/- {f.std():.3f} | AUC {a.mean():.3f} +/- {a.std():.3f}")

    print("\n  paired contrasts (positive => first arm better):")
    for label, x, y in [
        ("detector - static    (final)", stats["detector"]["final"], stats["static"]["final"]),
        ("detector - static    (AUC)  ", stats["detector"]["auc"], stats["static"]["auc"]),
        ("oracle   - detector  (final)", stats["oracle_reset"]["final"], stats["detector"]["final"]),
        ("detector - always    (final)", stats["detector"]["final"], stats["always_short"]["final"]),
    ]:
        dm, sem, nfav = _paired(x, y)
        print(f"    {label}: {dm:+.3f} +/- {sem:.3f}  ({nfav}/{n_seeds} seeds)")

    det_alarms = [r["alarm_step"] for r in results["detector"] if r["alarm_step"] >= 0]
    if det_alarms:
        delays = np.array(det_alarms) - swap_step
        print(f"\n  detector fired on {len(det_alarms)}/{n_seeds} seeds; "
              f"detection delay {delays.mean():.0f} +/- {delays.std():.0f} env steps")

    dm_f, sem_f, _ = _paired(stats["detector"]["final"], stats["static"]["final"])
    dm_a, sem_a, _ = _paired(stats["detector"]["auc"], stats["static"]["auc"])
    print(f"\n  VERDICT (final): {_verdict(dm_f, sem_f)}")
    print(f"  VERDICT (AUC)  : {_verdict(dm_a, sem_a)}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for arm in ARMS:
        color, marker, label = ARM_STYLE[arm]
        m = curves[arm].mean(axis=0)
        sem = curves[arm].std(axis=0) / np.sqrt(max(n_seeds, 1))
        ax.plot(steps, m, marker=marker, ms=3, color=color, label=label)
        ax.fill_between(steps, m - sem, m + sem, color=color, alpha=0.18)
    ax.axvline(swap_step, color="k", ls="--", lw=1, alpha=0.7, label="partner swap")
    # alarm_step is already an absolute env step (one p-value per step from step 0); the
    # detection *delay* is alarm_step - swap_step.
    det_alarms = [r["alarm_step"] for r in results["detector"] if r["alarm_step"] >= 0]
    if det_alarms:
        mean_alarm = float(np.mean(det_alarms))
        ax.axvline(mean_alarm, color="#C44E52", ls=":", lw=1.2,
                   label=f"mean detector alarm (delay +{mean_alarm - swap_step:.0f})")
    ax.set_xlabel("env step")
    ax.set_ylabel("greedy team return (vs current partner)")
    ax.set_title(f"Recovery after co-learner drift -- {args.env} ({n_seeds} seeds)\n"
                 f"detector vs static (final): {_verdict(dm_f, sem_f)}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    tag = f"{args.env}_{args.drift}_{args.scale}"
    out = RESULTS_DIR / f"recovery_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    payload = {
        "env": args.env, "drift": args.drift, "scale": args.scale,
        "n_seeds": n_seeds, "swap_step": swap_step,
        "steps": steps.tolist(),
        "curves": {arm: curves[arm].tolist() for arm in ARMS},
        "alarms": {arm: [r["alarm_step"] for r in results[arm]] for arm in ARMS},
        "verdict_final": _verdict(dm_f, sem_f), "verdict_auc": _verdict(dm_a, sem_a),
        "pretrain": {"A": [a0, a1], "B": [b0, b1]},
    }
    (OUTPUTS_DIR / f"recovery_{tag}.json").write_text(json.dumps(payload))
    print(f"\nfigure -> {out}")
    print(f"raw numbers -> {OUTPUTS_DIR / f'recovery_{tag}.json'}")


if __name__ == "__main__":
    sys.exit(main())
