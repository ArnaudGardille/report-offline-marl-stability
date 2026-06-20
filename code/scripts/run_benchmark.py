"""Config-driven benchmark harness (M8 P3): one entry point, CPU-smoke or GPU-full.

The heavy JaxMARL benchmarks (SMAX, Overcooked) need a GPU to train to a meaningful
policy; the cheap ones (Simultaneous-Attack, MPE ``simple_spread``) run fine on CPU.
This script is the single, parameterised driver the user runs on Colab/GPU to produce
camera-ready figures, *and* the thing CI can run end-to-end at ``--smoke`` scale on CPU
to prove the pipeline is wired correctly.

It does **not** re-implement the experiments: it dispatches to the existing, tested
machinery --

* ``detection``  -> :mod:`detection_benchmark` (conformal martingale vs AES at matched
  false-alarm; calibrated on the cheap Simu lab, detection delay measured after a
  controlled partner swap on the target env);
* ``selection``  -> :mod:`ablation` (``uniform`` vs ``conformal_policy`` / ``exact_ratio``
  replay selection; optimality-vs-steps with inter-seed bands).

-- only swapping in **scale-appropriate configs** via :data:`SCALE`. The ``--smoke``
column is sized to finish in CI-time on CPU (tiny nets / budgets / few seeds); the
``--full`` column is sized for a GPU run (the real budgets that train SMAX/Overcooked to
a non-trivial policy). Nothing about the scale changes the *code path* exercised, only the
numbers, so a green ``--smoke`` is genuine evidence the ``--full`` run is wired right
(though of course ``--full`` is GPU-untested here -- see ``code/RUN_ON_GPU.md``).

Usage::

    # CPU, CI-time (verifiable locally):
    uv run python scripts/run_benchmark.py --env mpe   --experiment detection --smoke
    uv run python scripts/run_benchmark.py --env simu  --experiment selection --smoke
    uv run python scripts/run_benchmark.py --env smax  --experiment detection --smoke
    uv run python scripts/run_benchmark.py --env overcooked --experiment detection --smoke

    # GPU, camera-ready (Colab):
    uv run python scripts/run_benchmark.py --env smax --experiment detection --full \
        --map-name 3m
    uv run python scripts/run_benchmark.py --env overcooked --experiment detection --full

Figures land in ``code/results/`` (committed), raw numbers in ``code/outputs/`` (gitignored).
The figure/json names encode env+experiment+scale so smoke and full runs don't clobber.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ``scripts/`` is not a package (run as ``uv run python scripts/foo.py``); import the
# sibling experiment modules directly off this directory, exactly as detection_benchmark
# imports drift_demo.
sys.path.insert(0, str(Path(__file__).resolve().parent))

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
OUTPUTS_DIR = CODE_DIR / "outputs"


# --------------------------------------------------------------------------- env map
# Harness env key -> (factory name, default env_kwargs, default IQL hidden_dim, human label).
# ``simu`` and ``mpe`` are CPU-friendly; ``smax`` and ``overcooked`` are the GPU targets.
ENV_TABLE: dict[str, dict] = {
    "simu": {"factory": "simu", "kwargs": {}, "hidden": 32, "label": "Simultaneous-Attack"},
    "mpe": {"factory": "MPE_simple_spread_v3", "kwargs": {}, "hidden": 128,
            "label": "MPE simple_spread"},
    "smax": {"factory": "SMAX", "kwargs": {"map_name": "3m"}, "hidden": 128, "label": "SMAX"},
    "overcooked": {"factory": "overcooked", "kwargs": {"layout": "cramped_room"},
                   "hidden": 64, "label": "Overcooked"},
}


# --------------------------------------------------------------------------- scale knobs
# Per (experiment, scale) tuning. ``--smoke`` is sized for CPU CI-time; ``--full`` for a
# GPU run. The *code path* is identical across scales -- only budgets/seeds/sizes differ.
#
# Detection knobs:
#   train_steps : IQL budget per seed for the target-env policy (the thing a GPU buys).
#   seeds       : number of partner-swap seeds for the delay distribution.
#   pre_len/post_len : stream lengths around the swap (post is the anytime-valid window).
#   n_h0/h0_len : Simu false-alarm lab size (cheap; CPU even in --full).
# Selection knobs:
#   train_steps : IQL budget per seed (per sampler).
#   seeds       : seeds per sampler.
SCALE: dict[str, dict] = {
    "detection": {
        "smoke": dict(train_steps=4_000, seeds=2, pre_len=24, post_len=48,
                      n_h0=24, h0_len=60),
        "full": dict(train_steps=None, seeds=8, pre_len=150, post_len=400,
                     n_h0=200, h0_len=240),
    },
    "selection": {
        "smoke": dict(train_steps=3_000, seeds=3),
        "full": dict(train_steps=None, seeds=12),
    },
}
# A smoke-scale target-env training budget. ``--full`` uses ``train_steps=None`` -> the
# env's tuned default (or the script-level short config); smoke shrinks it hard so the
# heavy envs (SMAX/Overcooked) still complete one compile + a few iterations on CPU.
SMOKE_TARGET_STEPS = {"simu": 4_000, "mpe": 8_000, "smax": 4_000, "overcooked": 6_000}


# --------------------------------------------------------------------------- detection
def run_detection(env_key: str, scale: str, args) -> dict:
    """Detection-delay benchmark on ``env_key`` at the requested scale.

    Reuses :mod:`detection_benchmark`: the Simu false-alarm lab fixes the matched AES
    operating point, then per-seed detection delays are measured after a controlled
    partner swap on the target env (conformal martingale vs AES, env-native AES calibrated
    too). For the cheap envs this runs on CPU; for SMAX/Overcooked the GPU is what makes
    ``--full`` train to a non-trivial policy.
    """
    import detection_benchmark as db
    from drift_demo import short_iql_config

    knobs = SCALE["detection"][scale]
    meta = ENV_TABLE[env_key]
    factory, kwargs, hidden = meta["factory"], dict(meta["kwargs"]), meta["hidden"]
    if args.map_name and env_key == "smax":
        kwargs["map_name"] = args.map_name
    if args.layout and env_key == "overcooked":
        kwargs["layout"] = args.layout

    # 1) Simu false-alarm lab (always cheap; matched-budget operating point for AES).
    calib = db.calibrate_on_simu(args.delta, knobs["n_h0"], knobs["h0_len"], args.seed)

    # 2) Detection delay on the target env. ``simu`` has no swap-detection variant of its
    #    own here (it *is* the false-alarm lab); for it we just report the calibration.
    heavy = None
    if env_key != "simu":
        seeds = list(range(args.seed, args.seed + knobs["seeds"]))
        # Decide the target-env training budget for this scale by monkeypatching the
        # config factory's budget. ``short_iql_config`` is the single source of per-env
        # budgets used by detection_benchmark; we shrink it for smoke, leave it for full.
        train_steps = knobs["train_steps"]
        if scale == "smoke":
            train_steps = SMOKE_TARGET_STEPS[env_key]
        _patch_short_config_steps(short_iql_config, factory, train_steps)

        heavy = db.detection_delay_env(
            calib, args.delta, seeds,
            pre_len=knobs["pre_len"], post_len=knobs["post_len"],
            env_name=factory, hidden_dim=hidden, label=meta["label"], **kwargs,
        )

    tag = f"{env_key}_detection_{scale}"
    fig = RESULTS_DIR / f"m8_{tag}.png"
    db.plot_benchmark(calib, heavy, args.delta, fig)
    payload = {"experiment": "detection", "env": env_key, "scale": scale,
               "calibration": calib, "heavy": heavy, "delta": args.delta}
    (OUTPUTS_DIR / f"m8_{tag}.json").write_text(json.dumps(payload))
    print(f"\nfigure -> {fig}")
    print(f"raw numbers -> {OUTPUTS_DIR / f'm8_{tag}.json'}")

    # Console summary (honest read).
    print("\n================= SUMMARY =================")
    print(f"[{meta['label']} | detection | {scale}] delta={args.delta}")
    print(f"  Simu conformal achieved false-alarm: {calib['fa_conf']:.4f} "
          f"(Ville bound <= {args.delta})")
    if heavy is not None:
        nc, mc, ac = db._delay_stats(heavy["conf_delays"])
        nn, mn, an = db._delay_stats(heavy["aes_delays_oc_native"])
        ns = len(heavy["seeds"])
        print(f"  {meta['label']} H0 false-alarm: conformal {heavy['conf_fa_overcooked']:.4f}")
        print(f"  detection delay (median / mean, #detected over {ns} seeds):")
        print(f"    conformal           : {mc:.1f} / {ac:.1f}  ({nc}/{ns})")
        print(f"    AES (env-native cal): {mn:.1f} / {an:.1f}  ({nn}/{ns})")
    return payload


def _patch_short_config_steps(short_iql_config, factory_name: str, train_steps):
    """Best-effort: wrap ``short_iql_config`` so its returned config uses ``train_steps``.

    ``detection_benchmark`` calls ``short_iql_config(env_name)`` internally to budget the
    target-env policy. To control that budget by scale without threading a new argument
    through the whole module, we replace the function in ``detection_benchmark``'s
    namespace with a thin wrapper that overrides ``total_timesteps`` when a value is given
    (``None`` -> leave the tuned default untouched, the ``--full`` behaviour).
    """
    import detection_benchmark as db

    if train_steps is None:
        return  # full scale: keep the env's tuned default budget.

    original = short_iql_config

    def _wrapped(env_name, sampler="uniform"):
        cfg = original(env_name, sampler)
        cfg.total_timesteps = int(train_steps)
        # Keep learning_starts below the (possibly tiny) budget so a smoke run still learns
        # at least a few gradient steps rather than skipping learning entirely.
        cfg.learning_starts = min(cfg.learning_starts, max(64, train_steps // 8))
        cfg.epsilon_anneal_time = min(cfg.epsilon_anneal_time, max(1, train_steps))
        return cfg

    db.short_iql_config = _wrapped


# --------------------------------------------------------------------------- selection
def run_selection(env_key: str, scale: str, args) -> dict:
    """Replay-selection ablation (uniform vs conformal samplers) on ``env_key``.

    Reuses :mod:`ablation`: trains each sampler for ``seeds`` seeds and plots optimality
    (normalised greedy return) vs env steps with inter-seed bands. Honest by construction
    -- it reports AUC / steps-to-threshold / final optimality, and the ablation module's
    own docstring documents that on the easily-learned envs selection is neutral-to-slower.
    """
    import ablation as ab
    import numpy as np
    from conformal_marl.envs.registry import make_env

    knobs = SCALE["selection"][scale]
    meta = ENV_TABLE[env_key]
    factory, kwargs = meta["factory"], dict(meta["kwargs"])
    if args.map_name and env_key == "smax":
        kwargs["map_name"] = args.map_name
    if args.layout and env_key == "overcooked":
        kwargs["layout"] = args.layout

    env, _spec = make_env(factory, **kwargs)

    # Choose the per-env ablation config factory, then override its budget for this scale.
    base_config_fn = ab.simu_ablation_config if env_key == "simu" else ab.spread_ablation_config
    train_steps = knobs["train_steps"]

    def config_fn(sampler: str):
        cfg = base_config_fn(sampler)
        if train_steps is not None:
            cfg.total_timesteps = int(train_steps)
            cfg.learning_starts = min(cfg.learning_starts, max(64, train_steps // 8))
            cfg.epsilon_anneal_time = min(cfg.epsilon_anneal_time, max(1, train_steps))
        return cfg

    # Random + (heuristic) optimal normalisers, mirroring ablation.main.
    probe = config_fn("uniform")
    lo = ab.random_baseline(env, episodes=min(64, probe.eval_episodes),
                            max_steps=probe.eval_steps, seed=args.seed)
    hi = 0.0 if env_key != "simu" else _simu_optimal(env, probe)

    results = {}
    for sampler in ab.SAMPLERS:
        steps, returns_raw, dt = ab.run_sampler(env, config_fn, sampler, knobs["seeds"], args.seed)
        results[sampler] = (steps, returns_raw, dt)
        s = ab.summarise(steps, returns_raw, lo, hi)
        print(f"  {sampler:18s}: final_opt={s['final_opt_mean']:.3f}+/-{s['final_opt_std']:.3f} "
              f"auc={s['auc']:.3f} steps_to_0.95={s['steps_to_thresh']} ({dt:.1f}s)")

    tag = f"{env_key}_selection_{scale}"
    fig = RESULTS_DIR / f"m8_{tag}.png"
    ab.plot(meta["label"], results, lo, hi, fig)
    summary = {s: ab.summarise(*results[s][:2], lo, hi) for s in ab.SAMPLERS}
    payload = {"experiment": "selection", "env": env_key, "scale": scale,
               "lo": lo, "hi": hi, "summary": summary,
               "curves": {s: {"steps": np.asarray(results[s][0]).tolist(),
                              "returns": np.asarray(results[s][1]).tolist()}
                          for s in ab.SAMPLERS}}
    (OUTPUTS_DIR / f"m8_{tag}.json").write_text(json.dumps(payload))
    print(f"\nfigure -> {fig}")
    print(f"raw numbers -> {OUTPUTS_DIR / f'm8_{tag}.json'}")
    return payload


def _simu_optimal(env, cfg) -> float:
    """Greedy return of the all-attack optimum on Simultaneous-Attack (the ``hi`` normaliser).

    Simu's cooperative optimum is every agent attacking; its per-episode team return is the
    env's max reward. We read it off a short rollout of the *scripted* all-attack policy.
    """
    import jax
    import jax.numpy as jnp
    from conformal_marl.iql import get_env_info

    info = get_env_info(env)
    agents = info.agents

    def _one(rng):
        rng, rr = jax.random.split(rng)
        _, st = env.reset(rr)
        acts = {a: jnp.array(1) for a in agents}  # action 1 == ATTACK
        _, _, rewards, _, _ = env.step(rng, st, acts)
        return jnp.mean(jnp.stack([rewards[a] for a in agents]))

    return float(jnp.mean(jax.vmap(_one)(jax.random.split(jax.random.PRNGKey(0), 64))))


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="M8 config-driven benchmark harness (CPU-smoke / GPU-full)")
    ap.add_argument("--env", choices=list(ENV_TABLE), required=True)
    ap.add_argument("--experiment", choices=["detection", "selection"], default="detection")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--smoke", action="store_const", dest="scale", const="smoke",
                     help="tiny CPU/CI-time config (the locally-verifiable path)")
    grp.add_argument("--full", action="store_const", dest="scale", const="full",
                     help="full GPU config (camera-ready; GPU-untested locally)")
    ap.set_defaults(scale="smoke")
    ap.add_argument("--delta", type=float, default=0.05, help="Ville level / target false-alarm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-name", type=str, default=None, help="SMAX map (e.g. 3m, 2s3z)")
    ap.add_argument("--layout", type=str, default=None, help="Overcooked layout")
    ap.add_argument("--cpu", action="store_true", help="force JAX onto CPU (smoke default)")
    args = ap.parse_args()

    # Smoke runs default to CPU; full runs leave the platform to JAX (so a GPU box uses it).
    if args.scale == "smoke" or args.cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    import jax
    print(f"=== run_benchmark: env={args.env} experiment={args.experiment} scale={args.scale} "
          f"| jax backend={jax.default_backend()} ===")
    t0 = time.perf_counter()
    if args.experiment == "detection":
        run_detection(args.env, args.scale, args)
    else:
        run_selection(args.env, args.scale, args)
    print(f"\n[done] {args.env}/{args.experiment}/{args.scale} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
