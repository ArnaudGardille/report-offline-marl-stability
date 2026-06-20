"""CLI entry point for the M2 IQL backbone.

Runs IQL on Simultaneous-Attack and/or MPE simple_spread, over a batch of seeds in
a single vmapped+jitted call, and prints start-vs-end greedy returns plus a random
baseline. This is the "real run" config (bigger than the fast committed test).

Examples
--------
    uv run python scripts/train.py --env simu --seeds 4
    uv run python scripts/train.py --env spread --seeds 4 --timesteps 120000
    uv run python scripts/train.py --env both
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from conformal_marl.envs.registry import make_env
from conformal_marl.iql import IQLConfig, get_env_info, make_train
from conformal_marl.networks import QNetwork


def random_baseline(env, num_episodes: int, max_steps: int, rng) -> float:
    """Mean greedy-equivalent team return of a uniformly random policy."""
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

    return float(jnp.mean(jax.vmap(_one)(jax.random.split(rng, num_episodes))))


def run_env(name: str, env, config: IQLConfig, seeds: int, base_seed: int = 0):
    """Train ``seeds`` seeds in one vmapped call; return a results dict."""
    print(f"\n=== {name} : IQL ({seeds} seeds, sampler={config.sampler_name}) ===")
    print(f"    config: total_timesteps={config.total_timesteps} batch={config.batch_size} "
          f"hidden={config.hidden_dim} target_every={config.target_update_interval}")

    rng = jax.random.PRNGKey(base_seed)
    rng, rng_base = jax.random.split(rng)
    baseline = random_baseline(env, config.eval_episodes, config.eval_steps, rng_base)

    train_fn = make_train(config, env)
    rngs = jax.random.split(rng, seeds)

    t0 = time.perf_counter()
    out = jax.block_until_ready(jax.jit(jax.vmap(train_fn))(rngs))
    dt = time.perf_counter() - t0

    start = out["start_greedy_return"]
    end = out["end_greedy_return"]
    print(f"    random baseline return : {baseline:+.4f}")
    print(f"    start greedy return    : {float(jnp.mean(start)):+.4f} "
          f"(per-seed {[round(float(x), 3) for x in start]})")
    print(f"    end   greedy return    : {float(jnp.mean(end)):+.4f} +/- "
          f"{float(jnp.std(end)):.4f} (per-seed {[round(float(x), 3) for x in end]})")
    print(f"    wall-clock (compile+run): {dt:.1f}s")
    return {"baseline": baseline, "start": start, "end": end, "seconds": dt}


def simu_config() -> IQLConfig:
    """A solid config for Simultaneous-Attack (1-step coordination bandit)."""
    return IQLConfig(
        total_timesteps=40_000,
        num_envs=8,
        num_steps=8,
        num_grad_steps=8,
        learning_starts=200,
        hidden_dim=64,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=10_000,
        target_update_interval=100,
        epsilon_anneal_time=15_000,
        eval_episodes=64,
        eval_steps=2,  # 1-step env; 2 steps is plenty to capture one episode
    )


def spread_config() -> IQLConfig:
    """A config for MPE simple_spread (fixed-horizon, 25 steps, 5 actions).

    simple_spread is a genuinely harder credit-assignment problem for *plain* IQL
    (no value-mixing as in QMIX/VDN), so it needs a healthy budget and replay ratio
    to clearly beat the random baseline. With these settings IQL reaches ~-20.5 vs a
    ~-27.4 random baseline (clear improvement) in ~2 min for 3 seeds on CPU.
    """
    return IQLConfig(
        total_timesteps=500_000,
        num_envs=16,
        num_steps=16,
        num_grad_steps=16,  # high replay ratio: 16 gradient steps / collection
        learning_starts=5_000,
        hidden_dim=128,
        lr=5e-4,
        gamma=0.95,
        batch_size=256,
        buffer_size=200_000,
        target_update_interval=200,
        epsilon_anneal_time=200_000,
        eval_episodes=32,
        eval_steps=26,  # env horizon is 25
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 IQL backbone trainer")
    ap.add_argument("--env", choices=["simu", "spread", "both"], default="simu")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    ap.add_argument("--sampler", type=str, default="uniform")
    args = ap.parse_args()

    if args.env in ("simu", "both"):
        cfg = simu_config()
        cfg.sampler_name = args.sampler
        if args.timesteps:
            cfg.total_timesteps = args.timesteps
        env, _ = make_env("simu", num_agents=2)
        run_env("Simultaneous-Attack (N=2)", env, cfg, args.seeds)

    if args.env in ("spread", "both"):
        cfg = spread_config()
        cfg.sampler_name = args.sampler
        if args.timesteps:
            cfg.total_timesteps = args.timesteps
        env, _ = make_env("MPE_simple_spread_v3")
        run_env("MPE simple_spread", env, cfg, args.seeds)


if __name__ == "__main__":
    # Touch QNetwork so static analyzers keep the import (used via make_train).
    _ = QNetwork
    main()
