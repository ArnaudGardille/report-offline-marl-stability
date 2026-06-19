"""M5 P3 (stretch): close the loop -- drive the replay knob with Channel B's drift rate.

This is the *partially-evaluated* closed-loop component. It tests, in a controlled replay
experiment, whether wiring Channel B's drift-rate signal to the **buffer half-life**
(staleness discount) helps a learner **recover after a partner swap** faster than a static
(uniform) buffer. We report whatever we find -- this may be neutral or negative; no effect
is manufactured.

Why a controlled replay experiment (not a live training rewrite)
----------------------------------------------------------------
The M2 training loop is a single jit'd ``lax.scan`` with a swappable sampler. A faithful
closed-loop study only needs the **replay knob** path -- "does down-weighting stale
transitions by a drift-driven half-life speed post-drift recovery?" -- so we isolate that
path in a self-contained experiment rather than threading a live controller through the
jit boundary (which would risk the validated training loop for a stretch goal). The
experiment:

1. Train a short Overcooked policy A (the pre-drift learner).
2. Build a replay buffer that is **half pre-drift** (partner = A) and **half post-drift**
   (partner = a scripted swap), with each transition tagged by its *age* (post-drift
   transitions are "new", pre-drift ones are "old / stale").
3. Compute the Channel-B drift rate from the conformal p-value stream of the buffer
   (frozen pre-drift calibration), map it to a buffer half-life ``H`` via the controller.
4. Re-fit the Q-network on this buffer toward the **post-drift** regime under two replay
   weightings, for a fixed number of gradient steps:
   * **static**: uniform weights (the baseline buffer);
   * **adaptive**: staleness weights ``2^{-age/H}`` with ``H`` from Channel B (stale
     pre-drift transitions down-weighted in proportion to the detected drift).
5. Measure **post-drift recovery** = the mean TD error on a held-out set of *post-drift*
   transitions after the re-fit (lower == the learner has adapted to the new regime). We
   compare static vs adaptive over seeds and report the distribution + a paired delta.

Honest read: a positive result is "adaptive recovers to a lower post-drift TD error (or
gets there in fewer steps)"; a null/negative is "no significant difference / static is as
good". Both are reported. This is the documented, partially-evaluated P3 component.

Usage::

    uv run python scripts/closed_loop_demo.py                # 4 seeds (default)
    uv run python scripts/closed_loop_demo.py --seeds 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts

from conformal_marl.conformal import conformal_p_values, policy_scores
from conformal_marl.controller import (
    drift_rate_to_half_life,
    ewma_logwealth_slope,
    staleness_weight,
)
from conformal_marl.iql import get_env_info, make_train
from conformal_marl.martingale import run_martingale
from conformal_marl.networks import QNetwork

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_benchmark import td_errors_under_policy  # noqa: E402
from drift_demo import (  # noqa: E402
    BEHAVIOUR_EPS,
    SCORE_EPS,
    FlatObs,
    collect_stream,
    least_likely_action,
    make_constant_partner_params,
    short_iql_config,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
OUTPUTS_DIR = CODE_DIR / "outputs"


def _refit_td(params0, net, hidden_dim, buffer, weights_T, target_batch, gamma,
              lr=5e-4, steps=300, key=None):
    """Re-fit Q on ``buffer`` with per-transition ``weights_T`` for ``steps`` SGD steps.

    Weighted TD(0) regression (single-network bootstrap off ``params0``-initialised online
    net; target uses a stop-gradient on the same net, the simplest self-contained fit).
    Returns the **post-fit mean |TD error| on ``target_batch``** (the held-out post-drift
    set), the recovery metric. Lower == better adaptation to the post-drift regime.
    """
    opt = optax.adam(lr)
    opt_state = opt.init(params0)
    w = jnp.asarray(weights_T, dtype=jnp.float32)
    w = w / (jnp.sum(w) + 1e-8) * w.shape[0]  # normalise to mean 1 (keep LR scale)

    def loss_fn(params):
        td = td_errors_under_policy(buffer, params, net.apply, hidden_dim, gamma)
        return jnp.sum(w * td**2) / (jnp.sum(w) + 1e-8)

    @jax.jit
    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), _ = jax.lax.scan(step, (params0, opt_state), None, length=steps)
    post_td = td_errors_under_policy(target_batch, params, net.apply, hidden_dim, gamma)
    return float(jnp.mean(post_td))


def run_seed(seed, layout="cramped_room", pre_len=150, post_len=150, refit_steps=300):
    """One closed-loop seed: build the mixed buffer, get H from Channel B, compare fits."""
    base = jaxmarl.make("overcooked", layout=overcooked_layouts[layout])
    env = FlatObs(base)
    info = get_env_info(env)
    hd = 64
    net = QNetwork(action_dim=info.num_actions, hidden_dim=hd)
    cfg = short_iql_config("overcooked")
    train = jax.jit(make_train(cfg, env))

    params_self = jax.block_until_ready(train(jax.random.PRNGKey(seed)))["final_params"]
    swap_action = least_likely_action(env, info, net, params_self, hd,
                                      jax.random.PRNGKey(seed + 5))
    partner_post = make_constant_partner_params(
        net, hd, info.num_actions, info.obs_dim, info.num_agents, action=swap_action
    )

    k = jax.random.PRNGKey(seed + 1)
    k_cal, k_pre, k_post, k_held = jax.random.split(k, 4)

    # Frozen pre-drift calibration for the conformal p-values.
    cal = collect_stream(env, info, net, params_self, params_self, k_cal,
                         4 * pre_len, hd, BEHAVIOUR_EPS)
    cal_scores = policy_scores(cal, params_self, net.apply, SCORE_EPS, hd)

    pre = collect_stream(env, info, net, params_self, params_self, k_pre,
                        pre_len, hd, BEHAVIOUR_EPS)
    post = collect_stream(env, info, net, params_self, partner_post, k_post,
                         post_len, hd, BEHAVIOUR_EPS)
    buffer = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], 0), pre, post)

    # Held-out post-drift set (fresh, same swapped partner) -> the recovery target.
    held = collect_stream(env, info, net, params_self, partner_post, k_held,
                         post_len, hd, BEHAVIOUR_EPS)

    # Channel B: drift rate from the buffer's conformal p-value stream (frozen cal).
    stream_scores = policy_scores(buffer, params_self, net.apply, SCORE_EPS, hd)
    p0 = conformal_p_values(stream_scores, cal_scores)[:, 0]
    raw = run_martingale(jnp.asarray(p0), delta=0.05, reset_on_alarm=False)
    drift_rate = ewma_logwealth_slope(raw.log_m, beta=0.1)
    # Use the steady-state post-drift rate to set a single half-life for the buffer.
    post_rate = float(jnp.mean(drift_rate[pre_len + post_len // 2:]))
    half_life = float(drift_rate_to_half_life(jnp.array(post_rate)))

    # Ages: oldest transition (index 0) is the most stale; newest (end) age 0.
    n = buffer.reward.shape[0]
    ages = jnp.arange(n - 1, -1, -1, dtype=jnp.float32)  # index 0 -> age n-1 (oldest)

    static_w = jnp.ones((n,), dtype=jnp.float32)
    adaptive_w = staleness_weight(ages, half_life)

    gamma = cfg.gamma
    static_rec = _refit_td(params_self, net, hd, buffer, static_w, held, gamma,
                           steps=refit_steps)
    adaptive_rec = _refit_td(params_self, net, hd, buffer, adaptive_w, held, gamma,
                             steps=refit_steps)

    return {
        "seed": seed,
        "post_drift_rate": post_rate,
        "half_life": half_life,
        "static_recovery_td": static_rec,
        "adaptive_recovery_td": adaptive_rec,
        "delta_td": static_rec - adaptive_rec,  # >0 => adaptive recovered to lower TD
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 P3 closed-loop replay-knob demo")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit-steps", type=int, default=300)
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("========== M5 P3: closed-loop replay-knob (Overcooked swap) ==========")
    rows = []
    for s in range(args.seed, args.seed + args.seeds):
        t0 = time.perf_counter()
        r = run_seed(s, refit_steps=args.refit_steps)
        rows.append(r)
        print(f"  seed {s} ({time.perf_counter() - t0:.1f}s): drift_rate={r['post_drift_rate']:.3f} "
              f"H={r['half_life']:.0f} | static_recovery_TD={r['static_recovery_td']:.4f} "
              f"adaptive_recovery_TD={r['adaptive_recovery_td']:.4f} "
              f"delta={r['delta_td']:+.4f}")

    static = np.array([r["static_recovery_td"] for r in rows])
    adaptive = np.array([r["adaptive_recovery_td"] for r in rows])
    deltas = static - adaptive  # >0 => adaptive better (lower post-drift TD)

    print("\n================= P3 SUMMARY =================")
    print(f"  static   recovery TD: mean {static.mean():.4f} +/- {static.std():.4f}")
    print(f"  adaptive recovery TD: mean {adaptive.mean():.4f} +/- {adaptive.std():.4f}")
    print(f"  paired delta (static - adaptive): mean {deltas.mean():+.4f} "
          f"+/- {deltas.std():.4f}  ({int(np.sum(deltas > 0))}/{len(deltas)} seeds favour adaptive)")
    verdict = ("adaptive helps" if deltas.mean() > 2 * (deltas.std() / np.sqrt(len(deltas)) + 1e-9)
               else "neutral/inconclusive" if abs(deltas.mean()) <= 2 * (deltas.std() / np.sqrt(len(deltas)) + 1e-9)
               else "adaptive hurts")
    print(f"  read: {verdict} (paired mean vs 2 SE)")

    # Figure: paired static-vs-adaptive recovery TD.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(rows))
    ax.plot(x, static, "o-", color="#4C72B0", label="static (uniform buffer)")
    ax.plot(x, adaptive, "s-", color="#DD8452", label="adaptive (Channel-B half-life)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {r['seed']}" for r in rows])
    ax.set_ylabel("post-drift recovery mean |TD error| (lower=better)")
    ax.set_title("M5 P3: closed-loop replay knob vs static buffer\n"
                 f"(Overcooked swap; paired delta {deltas.mean():+.4f} +/- {deltas.std():.4f})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = RESULTS_DIR / "m5_closed_loop.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  figure -> {out}")

    (OUTPUTS_DIR / "m5_closed_loop.json").write_text(json.dumps(rows))
    print(f"  raw -> {OUTPUTS_DIR / 'm5_closed_loop.json'}")


if __name__ == "__main__":
    main()
