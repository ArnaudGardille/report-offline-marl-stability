"""Recovery experiment -- does detection-driven replay control improve *return*?

The decisive test the closed-loop demo (``scripts/closed_loop_demo.py``) could not give:
that script measured a held-out **TD error** after a one-shot re-fit on a frozen buffer.
This module measures the thing the paper actually claims to help -- **greedy team return**
recovered over *online* training after a co-learner drift -- under four replay regimes that
differ only in how (and whether) they forget the pre-drift data.

Set-up (one cooperative env, e.g. MPE ``simple_spread``):

* A single **ego** agent (index 0) learns by IQL from a replay buffer.
* The remaining agents are **partners** following a *fixed* policy. Before the swap the
  partners play policy ``A`` (the same policy the ego starts from, so the team is coherent
  and the return ``R0`` is high). At a known ``swap_step`` the partners switch to a
  different fixed policy ``B`` -- a co-learner drift the ego must adapt to.
* The ego keeps learning post-swap; we evaluate greedy team return throughout -> a
  **recovery curve** (drop at the swap, then re-climb).

Four arms, identical drift and data-generating process, differing only in the replay knob:

* ``static``       -- uniform buffer, never forgets (the control).
* ``detector``     -- our method: a **frozen-calibration conformal martingale**
  (:func:`conformal_marl.controller.two_channel_controller`) run on the ego's perceived
  p-value stream. **Channel A** alarm flushes the buffer once (reset-on-alarm);
  **Channel B**'s drift rate sets the staleness half-life ``2^{-age/H}`` on the replay
  weights. (The two knobs the user selected.)
* ``oracle_reset`` -- flush exactly at the *known* swap step, then forget aggressively
  (``H = HALF_LIFE_MIN``). Upper bound on what perfect, instant detection buys.
* ``always_short`` -- permanent aggressive forgetting (``H = HALF_LIFE_MIN``), no alarm.
  Isolates whether *detecting* matters vs always forgetting.

Contrasts: ``detector`` vs ``static`` = does detection-driven control help at all;
``detector`` vs ``oracle_reset`` = cost of detection delay; ``detector`` vs ``always_short``
= value of detection over blind forgetting.

Design notes / why a dedicated loop
-----------------------------------
``iql.make_train`` is parameter-shared (one net for all agents), has no notion of a fixed
partner or a mid-training swap, and runs its whole life in a single jit'd ``lax.scan`` with
its own buffer. Threading a swap + a between-chunk controller + a buffer flush through that
scan would risk the validated training loop. So this is a **separate, smaller** online loop
with a **Python outer loop** (one jit'd collect-scan + one jit'd grad-scan per chunk), which
makes the swap, the buffer reset, and the controller call trivial and side-effect-free. It
**reuses the validated primitives** (``iql._forward_q`` / ``double_dqn_target`` /
``_epsilon_greedy_actions``, ``conformal.policy_scores`` / ``conformal_p_values``,
``controller.two_channel_controller``) so the science is identical to the rest of the repo.

The replay buffer is a plain host-side **ring buffer that also stores each transition's
insertion step**, so the staleness weight ``2^{-(cur_step - insert_step)/H}`` is exact
without changing the shared :class:`~conformal_marl.replay.Transition` schema. The buffer is
sized so a full run never evicts -- ``static`` therefore keeps *all* the stale pre-swap data
(the maximal staleness handicap), and the forgetting arms differ only by their weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

from conformal_marl.conformal import conformal_p_values, policy_scores
from conformal_marl.controller import (
    HALF_LIFE_MAX,
    HALF_LIFE_MIN,
    staleness_weight,
    two_channel_controller,
)
from conformal_marl.iql import (
    EnvInfo,
    _avail_actions,
    _epsilon_greedy_actions,
    _forward_q,
    _greedy_actions,
    double_dqn_target,
)
from conformal_marl.networks import get_network_cls
from conformal_marl.replay import Transition

ARMS = ("static", "detector", "oracle_reset", "always_short")


@dataclass
class RecoveryConfig:
    """Hyper-parameters for one recovery run (all four arms share these)."""

    # --- env / network ---
    hidden_dim: int = 128
    net_type: str = "ff"

    # --- phase structure (single-env collection, Python outer loop) ---
    chunk_len: int = 256          # env steps collected per outer iteration
    pre_chunks: int = 12          # pre-swap chunks (fill buffer with stale (A,A) data)
    post_chunks: int = 40         # post-swap chunks (the recovery phase; ego learns)
    n_cal: int = 512              # frozen pre-swap calibration-set size (conformal)

    # --- optimisation (ego only) ---
    lr: float = 5e-4
    max_grad_norm: float = 10.0
    gamma: float = 0.99
    batch_size: int = 64
    grad_steps: int = 32          # ego gradient updates per post-swap chunk
    buffer_size: int = 20_000     # >= (pre+post)*chunk_len so a full run never evicts

    # --- exploration / scoring epsilons ---
    ego_eps: float = 0.1          # ego eps-greedy during the recovery phase
    cal_eps: float = 0.05         # behaviour eps while filling the calibration set
    score_eps: float = 0.05       # eps of pi_cur used by the conformal score (frozen A)

    # --- drift type ---
    # "policy"   -- post-swap partners follow a different *trained* policy B (a realistic but
    #               possibly mild co-learner drift; on symmetric envs B may be ~ as good as A,
    #               giving no performance drop and hence a toothless test).
    # "scripted" -- post-swap partners play a fixed ``scripted_action`` (a broken / non-
    #               cooperative partner). Guarantees a real performance drop *and* a strongly
    #               non-conforming action stream, so the recovery test has teeth on CPU envs.
    drift_mode: str = "policy"
    scripted_action: int = 0      # action partners repeat post-swap when drift_mode=="scripted"

    # --- detector ---
    delta: float = 0.05           # Ville level for Channel A
    half_life_min: float = HALF_LIFE_MIN
    half_life_max: float = HALF_LIFE_MAX

    # --- evaluation ---
    eval_episodes: int = 32
    eval_steps: int = 64

    def total_steps(self) -> int:
        return (self.pre_chunks + self.post_chunks) * self.chunk_len


# --------------------------------------------------------------------------- ring buffer
@dataclass
class RingBuffer:
    """Host-side ring buffer of joint transitions that also tracks insertion step.

    Stored as numpy arrays (the outer loop is Python), so a buffer flush is just
    ``fill = ptr = 0`` and circular writes are plain slicing. ``insert_step`` lets the
    staleness weight ``2^{-(cur_step - insert_step)/H}`` be exact.
    """

    obs: np.ndarray
    actions: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    next_obs: np.ndarray
    avail_actions: np.ndarray
    next_avail_actions: np.ndarray
    behavior_log_prob: np.ndarray
    insert_step: np.ndarray
    cap: int
    fill: int = 0
    ptr: int = 0

    @classmethod
    def create(cls, cap: int, n_agents: int, obs_dim: int, num_actions: int) -> "RingBuffer":
        z = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
        return cls(
            obs=z(cap, n_agents, obs_dim),
            actions=np.zeros((cap, n_agents), dtype=np.int32),
            reward=z(cap, n_agents),
            done=np.zeros((cap, n_agents), dtype=bool),
            next_obs=z(cap, n_agents, obs_dim),
            avail_actions=z(cap, n_agents, num_actions),
            next_avail_actions=z(cap, n_agents, num_actions),
            behavior_log_prob=z(cap, n_agents),
            insert_step=np.zeros((cap,), dtype=np.int64),
            cap=cap,
        )

    def reset(self) -> None:
        """Flush: forget everything (the reset-on-alarm / oracle action)."""
        self.fill = 0
        self.ptr = 0

    def add_chunk(self, traj: dict, base_step: int) -> None:
        """Append a chunk of ``T`` transitions (numpy arrays, leading axis ``T``).

        ``insert_step`` is ``base_step + t`` for the ``t``-th transition in the chunk.
        Circular: overwrites the oldest entries once full (sized so a full run does not).
        """
        t = traj["actions"].shape[0]
        steps = base_step + np.arange(t, dtype=np.int64)
        for i in range(t):
            j = (self.ptr + i) % self.cap
            self.obs[j] = traj["obs"][i]
            self.actions[j] = traj["actions"][i]
            self.reward[j] = traj["reward"][i]
            self.done[j] = traj["done"][i]
            self.next_obs[j] = traj["next_obs"][i]
            self.avail_actions[j] = traj["avail_actions"][i]
            self.next_avail_actions[j] = traj["next_avail_actions"][i]
            self.behavior_log_prob[j] = traj["behavior_log_prob"][i]
            self.insert_step[j] = steps[i]
        self.ptr = (self.ptr + t) % self.cap
        self.fill = min(self.fill + t, self.cap)

    def sample_batch(self, rng: np.random.Generator, batch: int, grad_steps: int):
        """Draw ``(grad_steps, batch)`` uniform indices and gather them.

        Returns ``(batch_g, ages_g)`` where ``batch_g`` is a dict of jnp arrays with shape
        ``(grad_steps, batch, ...)`` and ``ages_g`` is ``(grad_steps, batch)`` of insertion
        *indices* (age is filled in by the caller, which knows ``cur_step``).
        """
        idx = rng.integers(0, self.fill, size=(grad_steps, batch))
        gather = lambda a: jnp.asarray(a[idx])  # noqa: E731
        batch_g = dict(
            obs=gather(self.obs),
            actions=gather(self.actions),
            reward=gather(self.reward),
            done=gather(self.done),
            next_obs=gather(self.next_obs),
            avail_actions=gather(self.avail_actions),
            next_avail_actions=gather(self.next_avail_actions),
            behavior_log_prob=gather(self.behavior_log_prob),
        )
        insert_g = self.insert_step[idx]  # (grad_steps, batch) int
        return batch_g, insert_g


# --------------------------------------------------------------------------- jitted parts
def _zero_carry(n_rows: int) -> chex.Array:
    """Zero-width FF carry of shape ``(n_rows, 0)`` (the feed-forward placeholder)."""
    return jnp.zeros((n_rows, 0), dtype=jnp.float32)


def make_collect(env: Any, info: EnvInfo, apply_fn: Callable, chunk_len: int) -> Callable:
    """Build a jitted single-env collector with a heterogeneous ego/partner policy.

    Agent 0 (the ego) acts eps-greedy under ``ego_params``. Partners (agents >= 1) act greedy
    under ``partner_params`` when ``partner_action < 0``, or play the fixed integer
    ``partner_action`` when it is ``>= 0`` (the scripted broken-partner drift). Returns
    ``(end_state, traj)`` with ``traj`` fields ``(chunk_len, N, ...)``; auto-reset is handled
    by the JaxMARL base ``env.step``.
    """
    agents = info.agents
    n = info.num_agents
    is_ego = jnp.arange(n) == 0

    def collect(ego_params, partner_params, eps, partner_action, state0):
        def body(carry, _):
            env_state, last_obs, rng = carry
            rng, k_act, k_step = jax.random.split(rng, 3)
            obs_NA = jnp.stack([last_obs[a] for a in agents], axis=0)  # (N, obs)
            avail_NA = _avail_actions(env, env_state, info)            # (N, A)

            zc = _zero_carry(n)
            _, q_ego_NA = _forward_q(apply_fn, ego_params, zc, obs_NA, jnp.zeros((n,)))
            _, q_par_NA = _forward_q(apply_fn, partner_params, zc, obs_NA, jnp.zeros((n,)))

            a_ego_N, logp_ego_N = _epsilon_greedy_actions(q_ego_NA, avail_NA, eps, k_act)
            a_par_net_N = _greedy_actions(q_par_NA, avail_NA)
            # Scripted broken partner: repeat ``partner_action`` (>=0) instead of the net.
            a_par_N = jnp.where(partner_action >= 0, partner_action, a_par_net_N)
            actions_N = jnp.where(is_ego, a_ego_N, a_par_N)
            # Behaviour log-prob: ego's eps-greedy density; partners are greedy (log p = 0).
            logp_N = jnp.where(is_ego, logp_ego_N, 0.0)

            acts = {a: actions_N[i] for i, a in enumerate(agents)}
            next_obs, next_env_state, rewards, dones, _ = env.step(k_step, env_state, acts)
            next_obs_NA = jnp.stack([next_obs[a] for a in agents], axis=0)
            reward_N = jnp.stack([rewards[a] for a in agents], axis=0)
            done_N = jnp.stack([dones[a] for a in agents], axis=0)
            next_avail_NA = _avail_actions(env, next_env_state, info)

            zc_store = _zero_carry(n)
            tr = Transition(
                obs=obs_NA,
                actions=actions_N.astype(jnp.int32),
                reward=reward_N.astype(jnp.float32),
                done=done_N,
                next_obs=next_obs_NA,
                avail_actions=avail_NA,
                next_avail_actions=next_avail_NA,
                behavior_log_prob=logp_N.astype(jnp.float32),
                hidden=zc_store,
                next_hidden=zc_store,
            )
            return (next_env_state, next_obs, rng), tr

        end, traj = jax.lax.scan(body, state0, None, length=chunk_len)
        return end, traj

    return jax.jit(collect)


def make_ego_learn(apply_fn: Callable, tx: optax.GradientTransformation, gamma: float) -> Callable:
    """Build a jitted scan that runs ``grad_steps`` weighted-TD updates on the ego (agent 0).

    ``batch_g`` fields are ``(G, B, N, ...)``; only the ego column (index 0) drives the
    single-agent DQN loss. ``weights_g`` is ``(G, B)`` staleness weights.
    """

    def ego_loss(params, target_params, batch_b, w_b):
        obs_B = batch_b["obs"][:, 0]            # (B, obs)
        next_obs_B = batch_b["next_obs"][:, 0]
        act_B = batch_b["actions"][:, 0]        # (B,)
        r_B = batch_b["reward"][:, 0]
        done_B = batch_b["done"][:, 0].astype(jnp.float32)
        next_avail_B = batch_b["next_avail_actions"][:, 0]  # (B, A)

        b = obs_B.shape[0]
        zc = _zero_carry(b)
        resets = jnp.zeros((b,))
        _, q_BA = _forward_q(apply_fn, params, zc, obs_B, resets)
        _, qn_on_BA = _forward_q(apply_fn, params, zc, next_obs_B, resets)
        _, qn_tg_BA = _forward_q(apply_fn, target_params, zc, next_obs_B, resets)

        chosen_B = jnp.take_along_axis(q_BA, act_B[:, jnp.newaxis], axis=-1)[:, 0]
        target_B = double_dqn_target(qn_on_BA, qn_tg_BA, next_avail_B, r_B, done_B, gamma)
        td_B = chosen_B - jax.lax.stop_gradient(target_B)
        return jnp.sum(w_b * td_B**2) / (jnp.sum(w_b) + 1e-8)

    def learn(params, opt_state, target_params, batch_g, weights_g):
        def step(carry, xs):
            params, opt_state = carry
            batch_b, w_b = xs
            loss, grads = jax.value_and_grad(ego_loss)(params, target_params, batch_b, w_b)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            step, (params, opt_state), (batch_g, weights_g)
        )
        return params, opt_state, jnp.mean(losses)

    return jax.jit(learn)


def make_eval(env: Any, info: EnvInfo, apply_fn: Callable, eval_steps: int) -> Callable:
    """Build a jitted, vmapped greedy team-return evaluator (ego 0 + fixed partners)."""
    agents = info.agents
    n = info.num_agents
    is_ego = jnp.arange(n) == 0

    def team_return(ego_params, partner_params, partner_action, rng, n_eps):
        def one_env(rng_e):
            rng_e, rng_reset = jax.random.split(rng_e)
            obs0, state0 = env.reset(rng_reset)

            def step(carry, _):
                env_state, last_obs, rng = carry
                rng, k_step = jax.random.split(rng)
                obs_NA = jnp.stack([last_obs[a] for a in agents], axis=0)
                avail_NA = _avail_actions(env, env_state, info)
                zc = _zero_carry(n)
                _, q_ego = _forward_q(apply_fn, ego_params, zc, obs_NA, jnp.zeros((n,)))
                _, q_par = _forward_q(apply_fn, partner_params, zc, obs_NA, jnp.zeros((n,)))
                a_ego = _greedy_actions(q_ego, avail_NA)
                a_par_net = _greedy_actions(q_par, avail_NA)
                a_par = jnp.where(partner_action >= 0, partner_action, a_par_net)
                act_N = jnp.where(is_ego, a_ego, a_par)
                acts = {a: act_N[i] for i, a in enumerate(agents)}
                next_obs, next_state, rewards, dones, _ = env.step(k_step, env_state, acts)
                team_r = jnp.mean(jnp.stack([rewards[a] for a in agents]))
                return (next_state, next_obs, rng), (team_r, dones["__all__"])

            _, (rews_T, dones_T) = jax.lax.scan(
                step, (state0, obs0, rng_e), None, eval_steps
            )
            first_done = jnp.argmax(dones_T)
            any_done = jnp.any(dones_T)
            last_idx = jnp.where(any_done, first_done, eval_steps - 1)
            mask_T = jnp.arange(eval_steps) <= last_idx
            return jnp.sum(jnp.where(mask_T, rews_T, 0.0))

        rngs = jax.random.split(rng, n_eps)
        return jnp.mean(jax.vmap(one_env)(rngs))

    return jax.jit(team_return, static_argnums=(4,))


# --------------------------------------------------------------------------- arm runner
def _traj_to_numpy(traj: Transition) -> dict:
    """Flatten a collected ``Transition`` (T, N, ...) into a numpy dict for the ring."""
    return {
        "obs": np.asarray(traj.obs),
        "actions": np.asarray(traj.actions),
        "reward": np.asarray(traj.reward),
        "done": np.asarray(traj.done),
        "next_obs": np.asarray(traj.next_obs),
        "avail_actions": np.asarray(traj.avail_actions),
        "next_avail_actions": np.asarray(traj.next_avail_actions),
        "behavior_log_prob": np.asarray(traj.behavior_log_prob),
    }


def _perceived_pvalues(traj: Transition, score_params, apply_fn, cal_scores_nN, eps, hidden_dim):
    """Agent-0 conformal p-values of a chunk, scored under the frozen policy ``A``.

    ``traj`` fields are ``(T, N, ...)`` (T is the conformal batch axis). Returns ``(T,)``.
    """
    scores_TN = policy_scores(traj, score_params, apply_fn, eps, hidden_dim)
    p_TN = conformal_p_values(scores_TN, cal_scores_nN)
    return np.asarray(p_TN[:, 0])


def run_arm(
    arm: str,
    env: Any,
    info: EnvInfo,
    network_apply: Callable,
    params_A: Any,           # pre-swap partner policy == ego init (coherent team)
    params_B: Any,           # post-swap partner policy (the drift)
    cfg: RecoveryConfig,
    seed: int,
) -> dict:
    """Run one arm; return its recovery curve and detector diagnostics.

    The ego starts as ``A``. Pre-swap the partners play ``A`` (no ego learning; the buffer
    fills with stale (A,A) data and we freeze the conformal calibration set). At the swap the
    partners switch to ``B`` and the ego learns for ``post_chunks``. Greedy team return
    (ego + *current* partner) is evaluated after every chunk -> the recovery curve.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")

    rng_np = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    collect = make_collect(env, info, network_apply, cfg.chunk_len)
    cal_collect = make_collect(env, info, network_apply, cfg.n_cal)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))
    learn = make_ego_learn(network_apply, tx, cfg.gamma)
    evaluate = make_eval(env, info, network_apply, cfg.eval_steps)

    buf = RingBuffer.create(cfg.buffer_size, info.num_agents, info.obs_dim, info.num_actions)

    # Ego train-state (starts at A).
    ego_params = jax.tree_util.tree_map(jnp.array, params_A)
    opt_state = tx.init(ego_params)
    target_params = jax.tree_util.tree_map(jnp.copy, ego_params)

    # Frozen conformal calibration: a pre-swap (A, A) stream scored under frozen A.
    no_script = jnp.asarray(-1, dtype=jnp.int32)  # partners follow their net (not scripted)
    key, k_cal = jax.random.split(key)
    _, cal_traj = cal_collect(
        params_A, params_A, jnp.asarray(cfg.cal_eps), no_script, _init_state(env, k_cal)
    )
    cal_scores_nN = policy_scores(
        cal_traj, params_A, network_apply, jnp.asarray(cfg.score_eps), cfg.hidden_dim
    )

    # Rolling collection state (single env).
    key, k_state = jax.random.split(key)
    state = _init_state(env, k_state)

    curve, curve_steps = [], []
    p_stream: list[float] = []
    swap_stream_idx = None
    alarm_step = -1           # env step at first Channel-A alarm (detector arm)
    did_reset = False
    cur_step = 0
    total_chunks = cfg.pre_chunks + cfg.post_chunks

    scripted = cfg.drift_mode == "scripted"
    for c in range(total_chunks):
        post = c >= cfg.pre_chunks
        eps = jnp.asarray(cfg.ego_eps if post else cfg.cal_eps)
        # Post-swap drift: either a different trained policy B, or a scripted broken partner.
        if post and scripted:
            partner, partner_action = params_A, jnp.asarray(cfg.scripted_action, dtype=jnp.int32)
        elif post:
            partner, partner_action = params_B, no_script
        else:
            partner, partner_action = params_A, no_script

        # --- COLLECT one chunk ---
        key, k_c = jax.random.split(key)
        state = (state[0], state[1], k_c)
        state, traj = collect(ego_params, partner, eps, partner_action, state)
        buf.add_chunk(_traj_to_numpy(traj), cur_step)

        # --- DETECTOR: perceived p-values (scored under frozen A) -> controller ---
        p_chunk = _perceived_pvalues(
            traj, params_A, network_apply, cal_scores_nN,
            jnp.asarray(cfg.score_eps), cfg.hidden_dim,
        )
        if post and swap_stream_idx is None:
            swap_stream_idx = len(p_stream)
        p_stream.extend(p_chunk.tolist())

        half_life = float(cfg.half_life_max)  # default: ~uniform (static / pre-swap)
        if arm == "detector":
            # One p-value per collected env step, so the stream index == env step. Re-running
            # on the full (growing) stream each chunk is redundant but correct; first_alarm is
            # stable once it fires.
            ctrl = two_channel_controller(
                jnp.asarray(p_stream), delta=cfg.delta,
                half_life_max=cfg.half_life_max, half_life_min=cfg.half_life_min,
            )
            half_life = float(ctrl.half_life[-1])
            if bool(ctrl.alarm) and not did_reset:
                # Flush the (now-stale) buffer on the first Channel-A alarm. A pre-swap H0
                # false alarm (prob <= delta) would flush near-empty data -- harmless.
                buf.reset()
                did_reset = True
                alarm_step = int(ctrl.first_alarm)  # absolute env step (1 p-value/step)
        elif arm == "oracle_reset":
            if post and not did_reset:
                buf.reset()
                did_reset = True
            half_life = float(cfg.half_life_min) if post else float(cfg.half_life_max)
        elif arm == "always_short":
            half_life = float(cfg.half_life_min)
        # static: half_life stays half_life_max (uniform), no reset.

        cur_step += cfg.chunk_len

        # --- LEARN (only post-swap, and only once enough samples are buffered) ---
        if post and buf.fill >= cfg.batch_size:
            key, k_s = jax.random.split(key)
            batch_g, insert_g = buf.sample_batch(rng_np, cfg.batch_size, cfg.grad_steps)
            ages_g = jnp.asarray(np.maximum(cur_step - insert_g, 0).astype(np.float32))
            weights_g = staleness_weight(ages_g, jnp.asarray(half_life))
            ego_params, opt_state, _loss = learn(
                ego_params, opt_state, target_params, batch_g, weights_g
            )
            target_params = jax.tree_util.tree_map(jnp.copy, ego_params)  # hard target / chunk

        # --- EVAL greedy team return against the *current* partner (or scripted drift) ---
        key, k_e = jax.random.split(key)
        ret = float(evaluate(ego_params, partner, partner_action, k_e, cfg.eval_episodes))
        curve.append(ret)
        curve_steps.append(cur_step)

    return {
        "arm": arm,
        "seed": seed,
        "curve": curve,
        "steps": curve_steps,
        "swap_step": cfg.pre_chunks * cfg.chunk_len,
        "alarm_step": alarm_step,
        "did_reset": did_reset,
    }


def _init_state(env: Any, key: chex.PRNGKey):
    """Initial single-env collection state ``(env_state, obs, rng)``."""
    key, k_reset = jax.random.split(key)
    obs0, state0 = env.reset(k_reset)
    return (state0, obs0, key)


def build_network(info: EnvInfo, cfg: RecoveryConfig):
    """Instantiate the Q-network and return ``(network, apply_fn)``."""
    network_cls = get_network_cls(cfg.net_type)
    network = network_cls(action_dim=info.num_actions, hidden_dim=cfg.hidden_dim)
    return network, network.apply
