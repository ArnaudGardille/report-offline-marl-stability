"""M4 controlled-drift demo: the conformal test martingale fires *after* a known swap.

This is the empirical showcase for the drift detector (``conformal_marl.martingale``):
we engineer a **known drift event** -- at a fixed step we swap one agent's policy -- and
show the martingale ``M_t`` stays controlled (well below the Ville threshold ``1/delta``)
*before* the swap and crosses it *after*, using the **real conformal p-values** from
:mod:`conformal_marl.conformal` (the ratio-free policy score), with the calibration set
**fixed at the pre-drift period** (not continuously refreshed -- otherwise the detector
would adapt to the drift and never fire).

Pipeline (per env)::

    1. train a short IQL policy A (and a second, independent policy B for the swap);
    2. roll out a *pre-drift* calibration set under (self=A, partner=A) and freeze its
       non-conformity scores as the conformal calibration set C;
    3. stream transitions: PRE the swap the partner is A (exchangeable with C -> p-values
       ~ Uniform -> M controlled); at the swap step the partner becomes a *different*
       policy (Simu: flip abstain<->attack; Overcooked: swap to params B), so agent 0's
       view of its partner's actions becomes non-conforming -> small p-values -> M grows;
    4. feed agent 0's p-value stream to ``run_martingale`` (reset-on-alarm) and plot
       ``M_t`` (log axis) vs step with the swap point and ``1/delta`` threshold marked.

Two environments are demonstrated:

* **Simultaneous-Attack** (primary): a 1-step coordination bandit where the controlled
  swap is a clean partner-policy flip (the learned partner attacks; post-swap it
  abstains, which the learner deems unlikely). Cheap and a very clean signal.
* **Overcooked** (``cramped_room``, the standard JaxMARL benchmark): partner is swapped
  to an independently trained policy. The detector also fires here; note the *learned*
  task reward is ~0 at this tiny training budget (Overcooked's soup reward is sparse),
  but the drift signal does **not** depend on reward -- only on how (un)likely the
  partner's actions are under the learner's policy -- so the demo is valid regardless.
  Overcooked obs are fed through a thin ``FlatObs`` wrapper so the M2 IQL backbone (MLP
  over a flat obs vector) trains on them without any change to ``iql.py``.

Figures are saved to ``code/results/m4_drift_demo_*.png`` (committed); raw traces to the
gitignored ``code/outputs/``. The printed summary reports, per env, the swap step and the
detection delay (alarm step minus swap step).

Usage::

    uv run python scripts/drift_demo.py                # Simu + Overcooked (default)
    uv run python scripts/drift_demo.py --no-overcooked # Simu only (fastest)
    uv run python scripts/drift_demo.py --delta 0.01
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

from conformal_marl.conformal import conformal_p_values, policy_scores
from conformal_marl.envs.registry import FlatObs, make_env
from conformal_marl.envs.simultaneous_attack import SimultaneousAttack
from conformal_marl.iql import IQLConfig, get_env_info, make_train
from conformal_marl.martingale import detection_delay, run_martingale
from conformal_marl.networks import QNetwork
from conformal_marl.replay import Transition

# ``FlatObs`` now lives in the package (conformal_marl.envs.registry) -- it used to be
# defined in this file and is imported by the sibling closed_loop_demo / detection_benchmark
# scripts. Re-export it here so their ``from drift_demo import FlatObs`` keeps working.
__all__ = ["FlatObs"]

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"  # committed figures
OUTPUTS_DIR = CODE_DIR / "outputs"  # gitignored raw traces

# Fixed scoring epsilon: the "current policy" pi_cur in the conformal score is the
# greedy-ish learner (small eps), so its non-conformity score sharply distinguishes a
# partner that plays the learner's expected actions (conforming) from one that does not.
SCORE_EPS = 0.05
# Behaviour epsilon while collecting the streamed/calibration rollouts (a little
# exploration so the calibration set has spread; the swap is the dominant effect).
BEHAVIOUR_EPS = 0.10


# --------------------------------------------------------------------------- scoring
def _eps_greedy_actions(params, net, hidden_dim, num_agents, num_actions, obs_NA, key, eps):
    """Per-agent eps-greedy actions ``(N,)`` under ``params`` for a single obs ``(N, D)``."""
    carry = QNetwork.initialize_carry(hidden_dim, num_agents)
    _, q_NA = net.apply(params, carry, (obs_NA[jnp.newaxis], jnp.zeros((1, num_agents))))
    q_NA = q_NA[0]
    greedy_N = jnp.argmax(q_NA, axis=-1)
    rand_N = jax.random.randint(key, (num_agents,), 0, num_actions)
    pick_N = jax.random.uniform(jax.random.fold_in(key, 7), (num_agents,)) < eps
    return jnp.where(pick_N, rand_N, greedy_N)


def collect_stream(env, info, net, p_self, p_partner, key, n_steps, hidden_dim, eps):
    """Roll out ``n_steps`` with agent 0 acting under ``p_self`` and **all other agents**
    under ``p_partner`` (the controllable "partner"). Returns a batched :class:`Transition`.

    The stored joint action is agent 0 from ``p_self`` and agents ``1..N-1`` from
    ``p_partner`` -- swapping ``p_partner`` mid-stream is exactly the controlled drift that
    agent 0's conformal score (which aggregates over the *other* agents ``j != 0``) sees.
    For the 2-agent envs (Simu / Overcooked) this is ``[a_self_0, a_partner_1]`` exactly as
    before; for the 3-agent MPE ``simple_spread`` it drives agents 1 and 2 together, so a
    single scripted swap produces a strong, observable drift in agent 0's p-values.

    Precondition: an **unmasked / fixed-action-set** env. The stored ``avail_actions`` /
    ``next_avail_actions`` are all-ones and the eps-greedy selection is unmasked, so on an
    action-masked env (SMAX) this would pick/score illegal actions. Callers building the
    detection benchmark reject masked envs upstream (see ``detection_delay_env``).
    """
    agents = info.agents
    n_agents = info.num_agents
    n_act = info.num_actions
    # All agents except index 0 are driven by the (swappable) partner policy.
    is_partner_N = jnp.arange(n_agents) != 0
    obs0, state0 = env.reset(key)

    def _step(carry, k):
        state, obs, rng = carry
        rng, k_act, k_step = jax.random.split(rng, 3)
        obs_NA = jnp.stack([obs[a] for a in agents], axis=0)
        a_self = _eps_greedy_actions(
            p_self, net, hidden_dim, n_agents, n_act, obs_NA, jax.random.fold_in(k_act, 0), eps
        )
        a_part = _eps_greedy_actions(
            p_partner, net, hidden_dim, n_agents, n_act, obs_NA, jax.random.fold_in(k_act, 1), eps
        )
        # Agent 0 from self; every other agent (the partner block) from partner params.
        acts_N = jnp.where(is_partner_N, a_part, a_self)
        acts = {a: acts_N[i] for i, a in enumerate(agents)}
        nobs, nstate, rewards, dones, _ = env.step(k_step, state, acts)
        nobs_NA = jnp.stack([nobs[a] for a in agents], axis=0)
        tr = Transition(
            obs=obs_NA,
            actions=acts_N.astype(jnp.int32),
            reward=jnp.stack([rewards[a] for a in agents]).astype(jnp.float32),
            done=jnp.stack([dones[a] for a in agents]),
            next_obs=nobs_NA,
            avail_actions=jnp.ones((n_agents, n_act), dtype=jnp.float32),
            # Detection streams run on unmasked envs (Simu / MPE / Overcooked-via-FlatObs),
            # so the next-state mask is all-ones (== avail_actions); kept explicit for the
            # M8 schema.
            next_avail_actions=jnp.ones((n_agents, n_act), dtype=jnp.float32),
            behavior_log_prob=jnp.zeros((n_agents,), dtype=jnp.float32),
            # Feed-forward stream: zero-width recurrent carry placeholders (M7 schema).
            hidden=jnp.zeros((n_agents, 0), dtype=jnp.float32),
            next_hidden=jnp.zeros((n_agents, 0), dtype=jnp.float32),
        )
        return (nstate, nobs, rng), tr

    (_, _, _), traj = jax.lax.scan(_step, (state0, obs0, key), jax.random.split(key, n_steps))
    return traj


def _concat_streams(a: Transition, b: Transition) -> Transition:
    return jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), a, b)


def drift_pvalue_stream(env, info, net, params_self, params_partner_post, hidden_dim,
                        pre_len, post_len, key):
    """Build the conformal p-value stream for agent 0 across a controlled partner swap.

    * Calibration set C: a pre-drift rollout under (self, self), scored once and frozen.
    * Stream: ``pre_len`` steps under (self, self) then ``post_len`` under (self,
      partner_post). p-values are agent 0's score ranked against the *fixed* C.

    Returns ``(p0, change_point, mean_p_pre, mean_p_post)`` where ``p0`` is the ``(T,)``
    agent-0 p-value stream and ``change_point == pre_len``.
    """
    k_cal, k_pre, k_post = jax.random.split(key, 3)

    cal = collect_stream(env, info, net, params_self, params_self, k_cal,
                         4 * pre_len, hidden_dim, BEHAVIOUR_EPS)
    cal_scores = policy_scores(cal, params_self, net.apply, SCORE_EPS, hidden_dim)

    pre = collect_stream(env, info, net, params_self, params_self, k_pre,
                        pre_len, hidden_dim, BEHAVIOUR_EPS)
    post = collect_stream(env, info, net, params_self, params_partner_post, k_post,
                         post_len, hidden_dim, BEHAVIOUR_EPS)
    stream = _concat_streams(pre, post)

    stream_scores = policy_scores(stream, params_self, net.apply, SCORE_EPS, hidden_dim)
    p_TN = conformal_p_values(stream_scores, cal_scores)  # (T, N)
    p0 = p_TN[:, 0]  # agent 0's view of the (other) partner

    mean_pre = float(jnp.mean(p0[:pre_len]))
    mean_post = float(jnp.mean(p0[pre_len:]))
    return np.asarray(p0), pre_len, mean_pre, mean_post


# ----------------------------------------------------------------------- partner swap
def make_constant_partner_params(net, hidden_dim, num_actions, obs_dim, num_agents, action):
    """Q-params whose greedy action is *always* a fixed ``action`` (a scripted partner).

    A clean controlled swap for Simultaneous-Attack: we want the post-swap partner to
    deterministically play ``action`` (e.g. ABSTAIN) regardless of obs. We initialise a
    net and overwrite its final Dense bias so ``action``'s logit dominates; the kernels
    are zeroed so the output is the (constant) bias -> argmax is always ``action``.
    """
    carry = QNetwork.initialize_carry(hidden_dim, 1)
    params = net.init(
        jax.random.PRNGKey(0), carry, (jnp.zeros((1, 1, obs_dim)), jnp.zeros((1, 1)))
    )
    # Set every output-Dense bias (the action_dim-wide vector) to a one-hot on ``action``
    # and zero all kernels/other biases, so the network output is the constant one-hot
    # logit -> argmax is always ``action``, for any observation.
    def _override(leaf):
        if leaf.ndim == 1 and leaf.shape[0] == num_actions:  # an output bias
            return jnp.zeros_like(leaf).at[action].set(10.0)
        return jnp.zeros_like(leaf)  # zero every kernel / non-output bias

    return jax.tree_util.tree_map(_override, params)


def least_likely_action(env, info, net, params, hidden_dim, key, n_steps=200):
    """The action the learner's greedy policy takes *least* often (a maximally
    non-conforming scripted-partner target).

    Rolls the learner out for ``n_steps``, tallies its greedy action histogram, and
    returns the rarest action. Using it as the post-swap scripted partner guarantees a
    strong drift signal (the partner now plays what the learner deems most unlikely),
    independent of the random seed / which action the learner happened to favour.
    """
    n_agents, n_act = info.num_agents, info.num_actions
    agents = info.agents
    obs0, state0 = env.reset(key)

    def _step(carry, k):
        state, obs, rng = carry
        rng, k_step = jax.random.split(rng)
        obs_NA = jnp.stack([obs[a] for a in agents], axis=0)
        carry_q = QNetwork.initialize_carry(hidden_dim, n_agents)
        _, q_NA = net.apply(params, carry_q, (obs_NA[jnp.newaxis], jnp.zeros((1, n_agents))))
        greedy_N = jnp.argmax(q_NA[0], axis=-1)
        acts = {a: greedy_N[i] for i, a in enumerate(agents)}
        nobs, nstate, _, _, _ = env.step(k_step, state, acts)
        return (nstate, nobs, rng), greedy_N

    (_, _, _), greedy_TN = jax.lax.scan(_step, (state0, obs0, key), jax.random.split(key, n_steps))
    counts = jnp.bincount(greedy_TN.reshape(-1), length=n_act)
    return int(jnp.argmin(counts))


# --------------------------------------------------------------------------- plotting
def plot_demo(env_name, p0, change_point, delta, result, out_png: Path):
    """Plot ``M_t`` (log axis) vs step with the swap point and ``1/delta`` marked."""
    log_m = np.asarray(result.log_m)
    m = np.exp(log_m)  # may be large; log axis handles it
    thresh = 1.0 / delta
    steps = np.arange(log_m.size)

    fig, (ax_m, ax_p) = plt.subplots(
        2, 1, figsize=(8.0, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot the genuine M_t. Pre-swap the mixture decays toward ~0 (p-values ~ 1), reaching
    # values like 1e-25 that a log axis would stretch to -inf and that would compress the
    # interesting post-swap region. Fix a readable lower limit (1e-4) and let matplotlib
    # clip the deeply-decayed segment off the bottom -- an honest "well below threshold,
    # off the bottom of view" rather than a misleading flat floor line.
    y_floor = 1e-4
    ax_m.plot(steps, m, color="#4C72B0", linewidth=1.8, label="$M_t$ (mixture)")
    ax_m.axhline(thresh, color="#C44E52", linestyle="--", linewidth=1.5,
                 label=rf"Ville threshold $1/\delta={thresh:g}$")
    ax_m.axvline(change_point, color="black", linestyle=":", linewidth=1.5,
                 label=f"policy swap (step {change_point})")
    first = int(result.first_alarm)
    if first >= 0:
        ax_m.scatter([first], [m[first]], color="#C44E52", zorder=5, s=45,
                     label=f"first alarm (step {first})")
    ax_m.set_yscale("log")
    ax_m.set_ylim(y_floor, thresh * 3)
    ax_m.set_ylabel("martingale $M_t$  (log scale)")
    ax_m.set_title(
        f"{env_name}: conformal drift martingale\n"
        f"controlled partner swap at step {change_point}; "
        f"detection delay = {int(detection_delay(result, change_point))} steps"
    )
    ax_m.grid(True, which="both", alpha=0.25)
    ax_m.legend(loc="upper left", fontsize=8, framealpha=0.95)

    ax_p.plot(steps, p0, color="#55A868", linewidth=0.9, alpha=0.85)
    ax_p.axvline(change_point, color="black", linestyle=":", linewidth=1.5)
    ax_p.set_ylabel("p-value")
    ax_p.set_xlabel("stream step (incoming transition)")
    ax_p.set_ylim(-0.03, 1.03)
    ax_p.grid(True, alpha=0.25)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- runners
def short_iql_config(env_name: str, sampler="uniform") -> IQLConfig:
    """A short training budget that yields a usable (not necessarily expert) policy."""
    if env_name == "simu":
        return IQLConfig(
            total_timesteps=8_000, num_envs=8, num_steps=8, num_grad_steps=8,
            learning_starts=200, hidden_dim=32, lr=1e-3, gamma=0.99, batch_size=64,
            epsilon_anneal_time=4_000, eval_episodes=64, eval_steps=2, sampler_name=sampler,
        )
    if env_name.startswith("MPE"):
        # MPE simple_spread trains fast and *meaningfully* on CPU: 250k steps takes ~25s
        # and lifts greedy return from ~-31 (random) to ~-23 (agents move to cover
        # landmarks). A non-trivial policy is the whole point of the MPE detection
        # showcase -- unlike Overcooked, the post-swap drift is a real behaviour change,
        # not noise around a near-random policy.
        return IQLConfig(
            total_timesteps=250_000, num_envs=16, num_steps=16, num_grad_steps=16,
            learning_starts=5_000, hidden_dim=128, lr=5e-4, gamma=0.95, batch_size=256,
            buffer_size=200_000, target_update_interval=200, epsilon_anneal_time=150_000,
            eval_episodes=32, eval_steps=26, sampler_name=sampler,
        )
    # overcooked: brief budget; reward stays ~0 (sparse) but policies become distinct.
    return IQLConfig(
        total_timesteps=50_000, num_envs=16, num_steps=64, num_grad_steps=4,
        learning_starts=512, hidden_dim=64, lr=5e-4, gamma=0.99, batch_size=64,
        buffer_size=50_000, target_update_interval=100, epsilon_anneal_time=35_000,
        eval_episodes=4, eval_steps=400, sampler_name=sampler,
    )


def run_simu(delta: float, seed: int) -> dict:
    """Simultaneous-Attack controlled-swap demo. Returns a summary dict."""
    print("\n========== Simultaneous-Attack (controlled partner-policy swap) ==========")
    env = SimultaneousAttack(num_agents=2)
    info = get_env_info(env)
    hd = 32
    net = QNetwork(action_dim=info.num_actions, hidden_dim=hd)

    # Train a short policy whose greedy is all-attack (the cooperative optimum).
    cfg = short_iql_config("simu")
    t0 = time.perf_counter()
    out = jax.block_until_ready(jax.jit(make_train(cfg, env))(jax.random.PRNGKey(seed)))
    params_self = out["final_params"]
    print(f"  trained IQL (end greedy return {float(out['end_greedy_return']):.3f}) "
          f"in {time.perf_counter() - t0:.1f}s")

    # Post-swap partner: a scripted ABSTAIN policy (the learner expects ATTACK, so abstain
    # is non-conforming -> the drift the detector should catch).
    partner_post = make_constant_partner_params(
        net, hd, info.num_actions, info.obs_dim, info.num_agents, action=0  # ABSTAIN
    )

    pre_len, post_len = 120, 120
    p0, cp, mpre, mpost = drift_pvalue_stream(
        env, info, net, params_self, partner_post, hd, pre_len, post_len,
        jax.random.PRNGKey(seed + 1),
    )
    result = run_martingale(jnp.asarray(p0), delta=delta, reset_on_alarm=True)
    delay = int(detection_delay(result, cp))
    print(f"  mean p  pre-swap={mpre:.3f}  post-swap={mpost:.3f}")
    print(f"  swap at step {cp}; crossed 1/delta={1 / delta:g}: {bool(result.crossed)}; "
          f"first alarm step {int(result.first_alarm)}; detection delay {delay} steps; "
          f"#alarms {int(np.sum(np.asarray(result.alarms)))}")

    fig = RESULTS_DIR / "m4_drift_demo_simu.png"
    plot_demo("Simultaneous-Attack", p0, cp, delta, result, fig)
    print(f"  figure -> {fig}")
    return _summary("simu", cp, delta, result, mpre, mpost, p0)


def run_overcooked(delta: float, seed: int, layout_name: str = "cramped_room") -> dict:
    """Overcooked controlled-swap demo: partner swapped to a scripted non-conforming policy.

    Parallels the Simu demo: the partner is swapped at a known step to a *scripted*
    constant-action policy whose action is the one the learner takes **least** often
    (chosen data-driven via :func:`least_likely_action`, so the swap is maximally
    non-conforming regardless of seed). This gives a clean, reproducible crossing on the
    real Overcooked benchmark. (We also tried swapping to an independently *trained*
    policy B; it works but the signal strength is seed-dependent because two
    briefly-trained policies can be similar -- the scripted swap is the controlled,
    reproducible version, and the detector is the same either way.)
    """
    print(f"\n========== Overcooked ({layout_name}) controlled partner swap ==========")
    # Env via the central factory (flattens Overcooked's image obs through the shared
    # FlatObs wrapper). We keep this script's own short config below, so behaviour /
    # figures are unchanged; only the env construction is de-duplicated.
    env, _ = make_env("overcooked", layout=layout_name)
    info = get_env_info(env)
    hd = 64
    net = QNetwork(action_dim=info.num_actions, hidden_dim=hd)
    cfg = short_iql_config("overcooked")

    t0 = time.perf_counter()
    train = jax.jit(make_train(cfg, env))
    params_self = jax.block_until_ready(train(jax.random.PRNGKey(seed)))["final_params"]
    print(f"  trained IQL policy in {time.perf_counter() - t0:.1f}s "
          f"(task reward ~0 at this budget: Overcooked's soup reward is sparse; the drift "
          f"signal does not depend on it)")

    # Scripted post-swap partner: the learner's least-likely action -> strong drift.
    swap_action = least_likely_action(env, info, net, params_self, hd, jax.random.PRNGKey(seed + 5))
    partner_post = make_constant_partner_params(
        net, hd, info.num_actions, info.obs_dim, info.num_agents, action=swap_action
    )
    print(f"  scripted partner plays the learner's least-likely action: {swap_action}")

    pre_len, post_len = 150, 150
    p0, cp, mpre, mpost = drift_pvalue_stream(
        env, info, net, params_self, partner_post, hd, pre_len, post_len,
        jax.random.PRNGKey(seed + 1),
    )
    result = run_martingale(jnp.asarray(p0), delta=delta, reset_on_alarm=True)
    delay = int(detection_delay(result, cp))
    print(f"  mean p  pre-swap={mpre:.3f}  post-swap={mpost:.3f}")
    print(f"  swap at step {cp}; crossed 1/delta={1 / delta:g}: {bool(result.crossed)}; "
          f"first alarm step {int(result.first_alarm)}; detection delay {delay} steps; "
          f"#alarms {int(np.sum(np.asarray(result.alarms)))}")

    fig = RESULTS_DIR / "m4_drift_demo_overcooked.png"
    plot_demo(f"Overcooked ({layout_name})", p0, cp, delta, result, fig)
    print(f"  figure -> {fig}")
    return _summary("overcooked", cp, delta, result, mpre, mpost, p0)


def _summary(tag, cp, delta, result, mpre, mpost, p0) -> dict:
    return {
        "env": tag,
        "delta": delta,
        "swap_step": int(cp),
        "crossed": bool(result.crossed),
        "first_alarm": int(result.first_alarm),
        "detection_delay": int(detection_delay(result, cp)),
        "num_alarms": int(np.sum(np.asarray(result.alarms))),
        "mean_p_pre": mpre,
        "mean_p_post": mpost,
        "log_m": np.asarray(result.log_m).tolist(),
        "p_values": np.asarray(p0).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 conformal drift-detection demo")
    ap.add_argument("--delta", type=float, default=0.01, help="Ville level (threshold 1/delta)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-overcooked", action="store_true", help="skip the Overcooked demo")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {"simu": run_simu(args.delta, args.seed)}
    if not args.no_overcooked:
        summaries["overcooked"] = run_overcooked(args.delta, args.seed)

    (OUTPUTS_DIR / "m4_drift_demo.json").write_text(json.dumps(summaries))
    print(f"\nraw traces -> {OUTPUTS_DIR / 'm4_drift_demo.json'}")
    print("\nSUMMARY")
    for tag, s in summaries.items():
        print(f"  {tag:11s}: swap@{s['swap_step']}  crossed={s['crossed']}  "
              f"delay={s['detection_delay']}  mean_p {s['mean_p_pre']:.2f}->{s['mean_p_post']:.2f}")


if __name__ == "__main__":
    main()
