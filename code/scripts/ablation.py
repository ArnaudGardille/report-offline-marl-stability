"""M3 ablation: uniform vs conformal replay selection on Simultaneous-Attack.

Compares the M2 ``uniform`` sampler against M3's ``conformal_policy`` (ratio-free)
and ``exact_ratio`` samplers on the Simultaneous-Attack coordination bandit, set up
to *surface the staleness phenomenon*: a large replay buffer (so old, high-epsilon
anti-coordinated transitions accumulate and linger) and a moderate budget, averaged
over many seeds. We plot **average optimality (normalised greedy return) vs
environment steps** with inter-seed bands, save the figure to
``code/results/m3_ablation_simu.png`` (committed) and the raw metrics to a gitignored
``outputs/`` directory.

Optionally also runs MPE simple_spread (``--spread``) if you want the spread numbers
(slower; off by default).

Honesty note (read the printed summary): on Simultaneous-Attack -- a *fully
observable*, easily-learned bandit whose all-attack optimum has a strong, consistent
reward gradient -- buffer staleness does **not** gate greedy performance: plain
uniform replay already solves it robustly, and conformal subsampling, by reducing the
effective diversity of the trained batch, tends to be *neutral-to-slightly-slower*
here. That is the expected result at this scale (the strong differentiator is the
masked-action / partially-observable SMAX setting at M5). The script reports what it
sees -- AUC, steps-to-threshold, final return -- it does not manufacture an effect.

Usage::

    uv run python scripts/ablation.py                 # Simu, 12 seeds (default)
    uv run python scripts/ablation.py --seeds 16
    uv run python scripts/ablation.py --spread        # also run simple_spread
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.envs.registry import make_env
from conformal_marl.iql import IQLConfig, get_env_info, make_train

# matplotlib: headless backend (no display on CI / servers).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Repo layout: this file is code/scripts/ablation.py.
CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"  # committed figures
OUTPUTS_DIR = CODE_DIR / "outputs"  # gitignored raw metrics

SAMPLERS = ["uniform", "conformal_policy", "exact_ratio"]
# Stable, colour-blind-friendly palette per sampler.
COLORS = {
    "uniform": "#4C72B0",
    "conformal_policy": "#DD8452",
    "exact_ratio": "#55A868",
}
LABELS = {
    "uniform": "uniform (M2 baseline)",
    "conformal_policy": "conformal_policy (ratio-free)",
    "exact_ratio": "exact_ratio (importance score)",
}


def simu_ablation_config(sampler: str) -> IQLConfig:
    """Staleness-stressed Simultaneous-Attack config (shared across samplers).

    Large buffer (stale transitions linger) + moderate budget + per-iteration greedy
    eval so we get a return-vs-steps curve. The conformal samplers draw a large
    LaBER batch ``L`` and subsample to ``B``; ``uniform`` draws ``B`` directly
    (``large_batch_size`` is ignored for it via ``draw_batch_size()``).
    """
    return IQLConfig(
        total_timesteps=15_000,
        num_envs=8,
        num_steps=16,  # chunk = 16*8 = 128 transitions/iter >= cal_size (no shrink)
        num_grad_steps=8,
        learning_starts=256,
        hidden_dim=64,
        lr=5e-4,
        gamma=0.99,
        batch_size=64,
        large_batch_size=512 if sampler != "uniform" else None,
        cal_size=128,
        buffer_size=50_000,  # >> transitions collected: old data dominates the buffer
        target_update_interval=100,
        epsilon_anneal_time=6_000,
        eval_episodes=128,
        eval_steps=2,  # 1-step env
        sampler_name=sampler,
        eval_during_training=True,
    )


def spread_ablation_config(sampler: str) -> IQLConfig:
    """A (slower) MPE simple_spread ablation config; reported only with --spread."""
    return IQLConfig(
        total_timesteps=200_000,
        num_envs=16,
        num_steps=16,
        num_grad_steps=8,
        learning_starts=2_000,
        hidden_dim=128,
        lr=5e-4,
        gamma=0.95,
        batch_size=128,
        large_batch_size=512 if sampler != "uniform" else None,
        cal_size=256,
        buffer_size=200_000,
        target_update_interval=200,
        epsilon_anneal_time=100_000,
        eval_episodes=32,
        eval_steps=26,  # horizon 25
        sampler_name=sampler,
        eval_during_training=True,
    )


def normalise(returns: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map raw return to [0, 1] optimality given a random-policy ``lo`` and optimal ``hi``."""
    return (returns - lo) / (hi - lo + 1e-8)


def run_sampler(env, config_fn, sampler: str, seeds: int, base_seed: int):
    """Train ``seeds`` seeds for one sampler; return (steps, returns_raw (seeds, T), seconds)."""
    cfg = config_fn(sampler)
    train_fn = make_train(cfg, env)
    rngs = jax.random.split(jax.random.PRNGKey(base_seed), seeds)
    t0 = time.perf_counter()
    out = jax.block_until_ready(jax.jit(jax.vmap(train_fn))(rngs))
    dt = time.perf_counter() - t0
    steps = np.asarray(out["return_steps"][0])  # identical across seeds
    returns = np.asarray(out["returns"])  # (seeds, T)
    return steps, returns, dt


def random_baseline(env, episodes: int, max_steps: int, seed: int) -> float:
    """Mean greedy-equivalent team return of a uniformly random policy (for normalisation)."""
    info = get_env_info(env)
    agents = info.agents

    def _one(rng_e):
        rng_e, rng_reset = jax.random.split(rng_e)
        _, env_state = env.reset(rng_reset)

        def _step(carry, _):
            env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            akeys = jax.random.split(rng_a, len(agents))
            acts = {a: env.action_space(a).sample(akeys[i]) for i, a in enumerate(agents)}
            _, env_state, rewards, dones, _ = env.step(rng_s, env_state, acts)
            team_r = jnp.mean(jnp.stack([rewards[a] for a in agents]))
            return (env_state, rng), (team_r, dones["__all__"])

        _, (rews, dones) = jax.lax.scan(_step, (env_state, rng_e), None, max_steps)
        first_done = jnp.argmax(dones)
        last_idx = jnp.where(jnp.any(dones), first_done, max_steps - 1)
        mask = jnp.arange(max_steps) <= last_idx
        return jnp.sum(jnp.where(mask, rews, 0.0))

    return float(jnp.mean(jax.vmap(_one)(jax.random.split(jax.random.PRNGKey(seed), episodes))))


def summarise(steps, returns_raw, lo, hi, thresh=0.95):
    """Per-sampler summary: final optimality (mean+/-std), AUC of mean curve, steps-to-thresh."""
    opt = normalise(returns_raw, lo, hi)  # (seeds, T)
    mean_curve = opt.mean(axis=0)
    final_mean = float(opt[:, -1].mean())
    final_std = float(opt[:, -1].std())
    auc = float(mean_curve.mean())  # mean optimality over training (area / span)
    hit = np.where(mean_curve >= thresh)[0]
    steps_to = int(steps[hit[0]]) if hit.size else -1
    return {
        "final_opt_mean": final_mean,
        "final_opt_std": final_std,
        "auc": auc,
        "steps_to_thresh": steps_to,
    }


def plot(env_name, results, lo, hi, out_png: Path):
    """Plot mean optimality vs env steps with +/-1 std inter-seed bands per sampler."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for sampler in SAMPLERS:
        steps, returns_raw, _dt = results[sampler]
        opt = normalise(returns_raw, lo, hi)
        mean_c = opt.mean(axis=0)
        std_c = opt.std(axis=0)
        c = COLORS[sampler]
        ax.plot(steps, mean_c, color=c, label=LABELS[sampler], linewidth=2.0)
        ax.fill_between(steps, mean_c - std_c, mean_c + std_c, color=c, alpha=0.18)

    n_seeds = next(iter(results.values()))[1].shape[0]
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("average optimality (normalised return)")
    ax.set_title(
        f"{env_name}: replay-selection ablation\n"
        f"mean +/- 1 std over {n_seeds} seeds (optimal = 1.0, random = 0.0)"
    )
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_env(env_name, env, config_fn, opt_return, seeds, base_seed, fig_path, tag):
    """Full ablation for one env: run all samplers, summarise, plot, dump raw metrics."""
    print(f"\n========== {env_name} ({seeds} seeds) ==========")
    eval_steps = config_fn("uniform").eval_steps
    lo = random_baseline(env, 256, eval_steps, base_seed)
    hi = opt_return
    print(f"random baseline return = {lo:+.4f}  | optimal (normaliser hi) = {hi:+.4f}")

    results = {}
    summaries = {}
    for sampler in SAMPLERS:
        steps, returns_raw, dt = run_sampler(env, config_fn, sampler, seeds, base_seed)
        results[sampler] = (steps, returns_raw, dt)
        s = summarise(steps, returns_raw, lo, hi)
        summaries[sampler] = s
        print(
            f"  {sampler:18s}: final opt {s['final_opt_mean']:.3f} +/- {s['final_opt_std']:.3f} "
            f"| AUC {s['auc']:.3f} | steps->0.95 {s['steps_to_thresh']:>6} | {dt:.1f}s"
        )

    plot(env_name, results, lo, hi, fig_path)
    print(f"  figure -> {fig_path}")

    # Raw metrics dump (gitignored outputs/).
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = {
        "env": env_name,
        "seeds": seeds,
        "random_baseline": lo,
        "optimal": hi,
        "samplers": {
            s: {
                "steps": results[s][0].tolist(),
                "returns_raw": results[s][1].tolist(),  # (seeds, T)
                "summary": summaries[s],
            }
            for s in SAMPLERS
        },
    }
    raw_path = OUTPUTS_DIR / f"m3_ablation_{tag}.json"
    raw_path.write_text(json.dumps(raw))
    print(f"  raw metrics -> {raw_path}")
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 conformal-selection ablation")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--spread", action="store_true", help="also run MPE simple_spread (slow)")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # Simultaneous-Attack (the primary deliverable).
    run_env(
        "Simultaneous-Attack (N=3)",
        make_env("simu", num_agents=3)[0],
        simu_ablation_config,
        opt_return=1.0,  # all-attack pays exactly +1
        seeds=args.seeds,
        base_seed=args.base_seed,
        fig_path=RESULTS_DIR / "m3_ablation_simu.png",
        tag="simu",
    )

    if args.spread:
        # simple_spread normaliser: optimal ~ 0 (agents perfectly cover landmarks is
        # near-0 cost in this reward scale); we just use 0.0 as an upper anchor and the
        # random baseline as the lower one, so "optimality" is relative progress.
        run_env(
            "MPE simple_spread",
            make_env("MPE_simple_spread_v3")[0],
            spread_ablation_config,
            opt_return=0.0,
            seeds=max(3, args.seeds // 4),
            base_seed=args.base_seed,
            fig_path=RESULTS_DIR / "m3_ablation_spread.png",
            tag="spread",
        )


if __name__ == "__main__":
    main()
