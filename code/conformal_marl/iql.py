"""Independent Q-Learning (IQL) backbone -- M2.

A clean, PureJaxRL-style IQL written from scratch against the installed JaxMARL
(the v0.0.2 ``baselines/QLearning/iql.py`` was used only as a *reference* for the
network / epsilon / double-Q patterns; this is not a fork -- it is transition-based
with a flashbax buffer and a swappable-sampler seam, where the reference was
trajectory/RNN-based with a bespoke buffer).

Architecture (see milestone M2 must-haves):
* **IQL with parameter sharing across agents** conditioned on the per-agent obs
  (each env already hands every agent a distinct observation -- e.g. a one-hot id in
  Simultaneous-Attack -- so a shared net *is* an independent per-agent learner).
  The seam to fully-independent params is a one-line change (vmap ``agent.init``);
  shared is simpler and learns both target envs.
* **Target network**, hard-updated every ``target_update_interval`` gradient steps
  (Polyak is available via ``tau`` < 1).
* **Epsilon-greedy** annealed linearly ``epsilon_start`` -> ``epsilon_finish`` over
  ``epsilon_anneal_time`` environment steps.
* **TD(0)** loss with a **double-DQN** target (online net picks the argmax action,
  target net evaluates it); available-action masks are respected. Per-sample
  weights from the sampler multiply the squared error (uniform -> 1).
* **Pure functional**: ``make_train(config) -> train(rng) -> metrics`` with the
  whole run in a ``jax.lax.scan``, ``jax.jit``-compiled and ``jax.vmap``-able over a
  batch of seeds.
* **RNN-ready**: the collection and eval scans thread a hidden-state ``carry`` (the
  network's ``initialize_carry`` / ``(carry, x) -> (carry, q)`` API), so M5 can drop
  in :class:`~conformal_marl.networks.GRUQNetwork` without touching the loop.
* **Swappable sampler**: the learn phase calls ``get_sampler(config.sampler_name)``
  with a :class:`SamplerContext`; M3 registers new samplers without editing this
  file.

The collection model is single-env (``lax.scan`` over steps); auto-reset is handled
by JaxMARL's base ``env.step``. Each scan step stores **one** joint transition into
the buffer (add-batch axis of 1). This keeps the per-transition unit clean for M3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

import optax
from jaxmarl.wrappers.baselines import LogWrapper

from conformal_marl.networks import QNetwork
from conformal_marl.replay import Transition, empty_transition, make_replay_buffer
from conformal_marl.samplers import SamplerContext, get_sampler

# Large negative used to mask unavailable actions before an argmax/max.
_MASKED_Q = -1e9


@dataclass
class IQLConfig:
    """Hyper-parameters for one IQL run. Plain dataclass (static, not traced)."""

    # --- run length / collection ---
    # Collection follows the PureJaxRL pattern: each outer iteration steps
    # ``num_envs`` parallel envs for ``num_steps`` steps (filling the buffer with
    # ``num_envs * num_steps`` transitions), then does ``num_grad_steps`` gradient
    # updates. The outer loop scans ``num_updates`` times. Compile cost scales with
    # ``num_steps`` (the inner scan length), NOT ``total_timesteps`` -- so large
    # budgets stay tractable, unlike a single-env scan over every step.
    total_timesteps: int = 200_000  # total env steps (across all parallel envs)
    num_envs: int = 8  # parallel envs stepped per inner collection step
    num_steps: int = 16  # env steps collected per env, per outer iteration
    num_grad_steps: int = 8  # gradient updates per outer iteration
    learning_starts: int = 256  # transitions before the first gradient update

    # --- network ---
    hidden_dim: int = 64
    init_scale: float = 1.0

    # --- optimisation ---
    lr: float = 5e-4
    max_grad_norm: float = 10.0
    gamma: float = 0.99
    batch_size: int = 64  # trained batch B (== L drawn for the uniform sampler)
    # LaBER large-batch L: the buffer draws this many items per sample; conformal
    # samplers subsample it down to ``batch_size`` (B). ``None`` -> L = B (no
    # subsampling, the ``uniform`` default). A typical LaBER setting is L = m * B
    # with m in {2..8}; larger L gives the conformal filter more to choose from.
    large_batch_size: int | None = None
    buffer_size: int = 20_000

    # --- target network ---
    target_update_interval: int = 200  # gradient steps between hard updates
    tau: float = 1.0  # 1.0 -> hard copy; <1 -> Polyak

    # --- exploration ---
    epsilon_start: float = 0.99
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 25_000  # env steps to anneal over

    # --- sampler seam ---
    sampler_name: str = "uniform"

    # --- conformal selection (M3) ---
    # Size of the fresh calibration set C threaded to conformal samplers. It is the
    # most-recent ``cal_size`` collected (on-policy) transitions, held out from the
    # buffer's uniform draw -- the inductive-conformal calibration set against which
    # buffer scores are ranked into p-values. Ignored by the ``uniform`` sampler.
    # Must be <= steps_per_update() (one collection chunk).
    cal_size: int = 64

    # --- evaluation ---
    eval_episodes: int = 32  # parallel greedy eval episodes
    eval_steps: int = 64  # steps per greedy eval rollout (>= env horizon)
    # When True, greedy return is evaluated *every outer iteration* and returned as a
    # per-iteration time series ``returns`` of shape ``(num_updates,)``, plus the env
    # step at each eval ``return_steps``. This is what the M3 ablation plots
    # (return-vs-steps with inter-seed bands). Off by default to keep the fast
    # learning test (which only needs start/end) cheap.
    eval_during_training: bool = False

    def draw_batch_size(self) -> int:
        """Number of items the buffer draws per sample (LaBER's ``L``).

        ``large_batch_size`` if set, else ``batch_size`` (so ``uniform`` draws exactly
        the trained batch and does no subsampling).
        """
        return self.large_batch_size if self.large_batch_size is not None else self.batch_size

    def steps_per_update(self) -> int:
        """Env steps collected per outer iteration."""
        return self.num_envs * self.num_steps

    def num_updates(self) -> int:
        """Number of outer (collect + learn) iterations."""
        return self.total_timesteps // self.steps_per_update()


@dataclass
class EnvInfo:
    """Static env metadata extracted once, outside the traced ``train``."""

    agents: list[str]
    num_agents: int
    num_actions: int
    obs_dim: int
    has_avail_actions: bool = field(default=False)


def get_env_info(env: Any) -> EnvInfo:
    """Pull static shapes/metadata off a (wrapped) JaxMARL env."""
    agents = list(env.agents)
    a0 = agents[0]
    num_actions = int(env.action_space(a0).n)
    obs_dim = int(np.prod(env.observation_space(a0).shape))
    has_avail = hasattr(env, "get_avail_actions")
    return EnvInfo(
        agents=agents,
        num_agents=len(agents),
        num_actions=num_actions,
        obs_dim=obs_dim,
        has_avail_actions=has_avail,
    )


# --------------------------------------------------------------------------- helpers
def _epsilon_by_step(step: chex.Array, cfg: IQLConfig) -> chex.Array:
    """Linearly annealed epsilon as a function of the env-step counter."""
    frac = jnp.clip(step / cfg.epsilon_anneal_time, 0.0, 1.0)
    return cfg.epsilon_start + frac * (cfg.epsilon_finish - cfg.epsilon_start)


def _stack_agents(d: dict, agents: list[str]) -> chex.Array:
    """Stack a per-agent dict into ``(N, ...)`` in canonical agent order."""
    return jnp.stack([d[a] for a in agents], axis=0)


def _avail_actions(env: Any, env_state: Any, info: EnvInfo) -> chex.Array:
    """Per-agent available-action mask ``(N, A)``; all ones if env has no masking."""
    if info.has_avail_actions:
        avail = env.get_avail_actions(env_state.env_state)
        return _stack_agents(avail, info.agents).astype(jnp.float32)
    return jnp.ones((info.num_agents, info.num_actions), dtype=jnp.float32)


def _forward_q(
    apply_fn: Callable,
    params: Any,
    carry: chex.Array,
    obs_NA: chex.Array,
    resets_N: chex.Array,
) -> tuple[chex.Array, chex.Array]:
    """One forward pass of the shared net over the agent batch.

    ``obs_NA`` is ``(N, obs_dim)``; we add a unit time axis (the recurrent API is
    ``(time, batch, ...)``) and treat the ``N`` agents as the batch. Returns the new
    carry and ``q`` of shape ``(N, A)``.
    """
    obs_TNB = obs_NA[jnp.newaxis, :, :]  # (1, N, obs_dim) : time=1, batch=N
    resets_TN = resets_N[jnp.newaxis, :]  # (1, N)
    carry, q_TNA = apply_fn(params, carry, (obs_TNB, resets_TN))
    return carry, q_TNA[0]  # drop the time axis -> (N, A)


def _greedy_actions(q_NA: chex.Array, avail_NA: chex.Array) -> chex.Array:
    """Masked greedy action per agent ``(N,)``."""
    masked = jnp.where(avail_NA > 0, q_NA, _MASKED_Q)
    return jnp.argmax(masked, axis=-1)


def _epsilon_greedy_actions(
    q_NA: chex.Array, avail_NA: chex.Array, eps: chex.Array, rng: chex.PRNGKey
) -> tuple[chex.Array, chex.Array]:
    """Masked epsilon-greedy actions and their behaviour log-probs, per agent.

    Returns ``(actions_N, log_prob_N)`` where ``log_prob`` is the log of the
    epsilon-greedy probability of the chosen action (uniform over available actions
    with prob eps, plus 1-eps on the greedy action). This is the cheap behaviour
    log-prob M3 needs; it is exact for the epsilon-greedy collection policy.
    """
    n_agents = q_NA.shape[0]
    key_pick, key_rand = jax.random.split(rng)

    greedy_N = _greedy_actions(q_NA, avail_NA)

    # Random action sampled *uniformly over available actions* per agent.
    logits = jnp.where(avail_NA > 0, 0.0, _MASKED_Q)
    rand_keys = jax.random.split(key_rand, n_agents)
    random_N = jax.vmap(lambda lg, k: jax.random.categorical(k, lg))(logits, rand_keys)

    pick_random_N = jax.random.uniform(key_pick, (n_agents,)) < eps
    actions_N = jnp.where(pick_random_N, random_N, greedy_N)

    # Behaviour probability of the *taken* action under epsilon-greedy:
    #   p(a) = (1-eps) * [a == greedy] + eps * (1/|avail|)   [a available]
    n_avail_N = jnp.sum(avail_NA > 0, axis=-1).astype(jnp.float32)
    is_greedy_N = (actions_N == greedy_N).astype(jnp.float32)
    p_taken_N = (1.0 - eps) * is_greedy_N + eps * (1.0 / n_avail_N)
    log_prob_N = jnp.log(jnp.clip(p_taken_N, 1e-8, 1.0))
    return actions_N, log_prob_N


def _soft_update(target: Any, online: Any, tau: float) -> Any:
    """Polyak update; ``tau == 1`` is a hard copy."""
    return jax.tree_util.tree_map(lambda t, o: (1.0 - tau) * t + tau * o, target, online)


# ---------------------------------------------------------------------- make_train
def make_train(config: IQLConfig, env: Any) -> Callable[[chex.PRNGKey], dict]:
    """Build a pure ``train(rng) -> metrics`` closure for the given env/config.

    ``env`` is a raw JaxMARL env; it is wrapped in ``LogWrapper`` here. The returned
    ``train`` is ``jax.jit``-able and ``jax.vmap``-able over a batch of ``rng`` seeds.
    """
    env = LogWrapper(env)
    info = get_env_info(env)
    agents = info.agents
    cfg = config

    network = QNetwork(
        action_dim=info.num_actions,
        hidden_dim=cfg.hidden_dim,
        init_scale=cfg.init_scale,
    )
    buffer = make_replay_buffer(
        max_size=cfg.buffer_size,
        min_size=cfg.learning_starts,
        # Draw L (>= B) so conformal samplers can subsample to B (LaBER seam). For
        # ``uniform`` L == B, so no subsampling happens.
        sample_batch_size=cfg.draw_batch_size(),
    )
    sampler = get_sampler(cfg.sampler_name)

    # The calibration set is sliced from one collection chunk (steps_per_update
    # transitions). Asking for more than that would silently shrink it, so guard the
    # static config up front (cheap, outside the traced train).
    if cfg.cal_size > cfg.steps_per_update():
        raise ValueError(
            f"cal_size ({cfg.cal_size}) > steps_per_update ({cfg.steps_per_update()}): "
            f"the calibration set is taken from one collection chunk, so it cannot "
            f"exceed num_envs * num_steps. Lower cal_size or raise num_steps/num_envs."
        )

    def _reset_carry(batch: int) -> chex.Array:
        return QNetwork.initialize_carry(cfg.hidden_dim, batch)

    # vmapped (over the parallel-env axis) env reset/step.
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    def _avail_batched(env_states) -> chex.Array:
        """Available-action masks ``(num_envs, N, A)`` across the parallel envs."""
        return jax.vmap(lambda s: _avail_actions(env, s, info))(env_states)

    def train(rng: chex.PRNGKey) -> dict:
        # -------------------------------------------------------------- init
        rng, rng_net, rng_reset = jax.random.split(rng, 3)

        init_obs = jnp.zeros((1, info.num_agents, info.obs_dim))  # (T=1, N, obs)
        init_resets = jnp.zeros((1, info.num_agents))
        init_carry = _reset_carry(info.num_agents)
        params = network.init(rng_net, init_carry, (init_obs, init_resets))

        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.lr),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        target_params = jax.tree_util.tree_map(jnp.copy, params)

        buffer_state = buffer.init(
            empty_transition(info.num_agents, info.obs_dim, info.num_actions)
        )

        # Reset ``num_envs`` parallel envs.
        reset_keys = jax.random.split(rng_reset, cfg.num_envs)
        obs0, env_state0 = v_reset(reset_keys)  # obs: dict agent -> (num_envs, obs)

        runner = {
            "train_state": train_state,
            "target_params": target_params,
            "buffer_state": buffer_state,
            "env_state": env_state0,
            "last_obs": obs0,
            "step": jnp.array(0, dtype=jnp.int32),  # total env steps collected
            "updates": jnp.array(0, dtype=jnp.int32),  # total gradient updates
            "rng": rng,
        }

        # The network params used during collection are fixed to the start-of-outer
        # iteration online params (standard for off-policy collection chunks). A
        # factory closes over those params so the inner scan body is parameter-free.
        def make_collect(params):
            def _collect_step_p(cstate, _):
                env_state, last_obs, step, rng = cstate
                rng, rng_act, rng_step = jax.random.split(rng, 3)
                obs_ENA = jnp.stack([last_obs[a] for a in agents], axis=1)
                avail_ENA = _avail_batched(env_state)
                eps = _epsilon_by_step(step, cfg)

                def _act_one(obs_NA, avail_NA, key):
                    carry = _reset_carry(info.num_agents)
                    resets_N = jnp.zeros((info.num_agents,))
                    _, q_NA = _forward_q(network.apply, params, carry, obs_NA, resets_N)
                    return _epsilon_greedy_actions(q_NA, avail_NA, eps, key)

                act_keys = jax.random.split(rng_act, cfg.num_envs)
                actions_EN, logp_EN = jax.vmap(_act_one)(obs_ENA, avail_ENA, act_keys)
                actions = {a: actions_EN[:, i] for i, a in enumerate(agents)}

                step_keys = jax.random.split(rng_step, cfg.num_envs)
                next_obs, next_env_state, rewards, dones, _ = v_step(
                    step_keys, env_state, actions
                )
                next_obs_ENA = jnp.stack([next_obs[a] for a in agents], axis=1)
                reward_EN = jnp.stack([rewards[a] for a in agents], axis=1)
                done_EN = jnp.stack([dones[a] for a in agents], axis=1)
                transitions = Transition(
                    obs=obs_ENA,
                    actions=actions_EN.astype(jnp.int32),
                    reward=reward_EN.astype(jnp.float32),
                    done=done_EN,
                    next_obs=next_obs_ENA,
                    avail_actions=avail_ENA,
                    behavior_log_prob=logp_EN.astype(jnp.float32),
                )
                return (next_env_state, next_obs, step + cfg.num_envs, rng), transitions

            return _collect_step_p

        # --------------------------------- one outer (collect + learn) iteration
        def _step(runner: dict, _: Any) -> tuple[dict, dict]:
            ts = runner["train_state"]
            rng = runner["rng"]

            # --- COLLECT num_steps across num_envs parallel envs ---
            rng, rng_collect = jax.random.split(rng)
            (env_state, last_obs, step, _), traj = jax.lax.scan(
                make_collect(ts.params),
                (runner["env_state"], runner["last_obs"], runner["step"], rng_collect),
                None,
                length=cfg.num_steps,
            )
            # traj fields are (num_steps, num_envs, N, ...). Flatten the (num_steps,
            # num_envs) lead dims into a single add-batch axis for the buffer.
            def _to_add_batch(x):
                return x.reshape((cfg.num_steps * cfg.num_envs,) + x.shape[2:])

            add_batch = jax.tree_util.tree_map(_to_add_batch, traj)
            buffer_state = buffer.add(runner["buffer_state"], add_batch)

            # --- conformal calibration set: the freshest on-policy transitions ---
            # Take the last ``cal_size`` rows of the just-collected chunk. These are
            # drawn under the current (collection) policy and are held out from the
            # buffer's uniform draw, so they are a valid inductive-conformal
            # calibration set C for scoring the buffer. (Re-collected every outer
            # iteration -> a fresh FIFO that tracks the moving policy.) The whole
            # chunk lives in ``add_batch``; the uniform sampler ignores ``cal_batch``,
            # so this slice is pure overhead only for the conformal samplers.
            cal_batch = jax.tree_util.tree_map(
                lambda x: x[-cfg.cal_size:], add_batch
            )

            # --- LEARN: num_grad_steps gradient updates on sampled batches ---
            ctx = SamplerContext(
                online_params=ts.params,
                target_params=runner["target_params"],
                apply_fn=ts.apply_fn,
                num_agents=info.num_agents,
                num_actions=info.num_actions,
                hidden_dim=cfg.hidden_dim,
                gamma=cfg.gamma,
                epsilon=_epsilon_by_step(step, cfg),
                cal_batch=cal_batch,
            )

            def _grad_step(lstate, _):
                ts, target_params, updates, rng = lstate
                rng, rng_sample = jax.random.split(rng)

                def _do_learn(_):
                    batch, weights, _info = sampler(
                        buffer, buffer_state, rng_sample, ctx, cfg.batch_size
                    )
                    loss, grads = jax.value_and_grad(_loss_fn)(
                        ts.params, target_params, batch, weights
                    )
                    return ts.apply_gradients(grads=grads), loss

                def _skip_learn(_):
                    return ts, jnp.array(0.0)

                can_learn = buffer.can_sample(buffer_state)
                new_ts, loss = jax.lax.cond(can_learn, _do_learn, _skip_learn, None)
                new_updates = updates + can_learn.astype(jnp.int32)

                # Hard/Polyak target update on the gradient-step cadence.
                do_target = jnp.logical_and(
                    can_learn, (new_updates % cfg.target_update_interval) == 0
                )
                new_target = jax.lax.cond(
                    do_target,
                    lambda _: _soft_update(target_params, new_ts.params, cfg.tau),
                    lambda _: target_params,
                    operand=None,
                )
                return (new_ts, new_target, new_updates, rng), loss

            (ts, new_target, updates, rng), losses = jax.lax.scan(
                _grad_step,
                (ts, runner["target_params"], runner["updates"], rng),
                None,
                length=cfg.num_grad_steps,
            )

            runner = {
                "train_state": ts,
                "target_params": new_target,
                "buffer_state": buffer_state,
                "env_state": env_state,
                "last_obs": last_obs,
                "step": step,
                "updates": updates,
                "rng": rng,
            }
            # Optional per-iteration greedy eval -> a return-vs-steps curve for the
            # ablation. Skipped (returns 0.0) when eval_during_training is False so the
            # default fast path pays nothing.
            if cfg.eval_during_training:
                rng_eval = jax.random.fold_in(runner["rng"], 0)
                iter_return = greedy_return(ts.params, rng_eval)
            else:
                iter_return = jnp.array(0.0)

            metrics = {
                "loss": jnp.mean(losses),
                "epsilon": _epsilon_by_step(step, cfg),
                "greedy_return": iter_return,
                "step": step,
            }
            return runner, metrics

        # ----------------------------------------------- weighted TD(0) loss
        def _loss_fn(
            params: Any, target_params: Any, batch: Transition, weights: chex.Array
        ) -> chex.Array:
            # batch fields are (B, N, ...). Flatten (B, N) -> (B*N) agent-rows so the
            # shared net scores every (sample, agent) pair in one pass.
            b, n = batch.actions.shape
            flat = lambda x: x.reshape((b * n,) + x.shape[2:])  # noqa: E731

            obs_BN = flat(batch.obs)  # (B*N, obs)
            next_obs_BN = flat(batch.next_obs)
            actions_BN = flat(batch.actions)
            reward_BN = flat(batch.reward)
            done_BN = flat(batch.done).astype(jnp.float32)
            avail_BN = flat(batch.avail_actions)

            zeros_resets = jnp.zeros((b * n,))
            carry = _reset_carry(b * n)

            _, q_BNA = _forward_q(network.apply, params, carry, obs_BN, zeros_resets)
            _, q_next_online_BNA = _forward_q(
                network.apply, params, carry, next_obs_BN, zeros_resets
            )
            _, q_next_target_BNA = _forward_q(
                network.apply, target_params, carry, next_obs_BN, zeros_resets
            )

            # Q of the taken action.
            chosen_q_BN = jnp.take_along_axis(
                q_BNA, actions_BN[:, jnp.newaxis], axis=-1
            )[:, 0]

            # Double-DQN target: online net argmax (masked), target net evaluates.
            # NOTE next-state avail mask: in these envs every step is a fresh reset
            # (Simu) or fixed action set (MPE), so reuse the stored mask. Stored mask
            # is for `obs`; for masked next-state envs (SMAX, M5) store next-avail.
            next_greedy_BN = _greedy_actions(
                jax.lax.stop_gradient(q_next_online_BNA), avail_BN
            )
            target_next_q_BN = jnp.take_along_axis(
                q_next_target_BNA, next_greedy_BN[:, jnp.newaxis], axis=-1
            )[:, 0]
            target_BN = reward_BN + cfg.gamma * (1.0 - done_BN) * target_next_q_BN

            td_err_BN = chosen_q_BN - jax.lax.stop_gradient(target_BN)
            se_BN = td_err_BN**2
            # Per-sample weights. Two accepted shapes:
            #   * (B,)   per-transition weight, broadcast across agents (uniform / IS);
            #   * (B, N) per-agent weight -- each agent weights its own action-column
            #            independently (M3's conformal samplers: agent a down-weights
            #            transitions whose other-agent actions are unlikely now).
            # Both reduce to (B*N,) aligned with the flattened (B, N) agent-rows.
            if weights.ndim == 1:
                w_flat = jnp.repeat(weights, n)
            else:
                w_flat = weights.reshape((b * n,))
            return jnp.sum(w_flat * se_BN) / (jnp.sum(w_flat) + 1e-8)

        # ----------------------------------------------- greedy evaluation
        def greedy_return(params: Any, rng: chex.PRNGKey) -> chex.Array:
            """Mean per-episode greedy return over ``eval_episodes`` parallel envs.

            Runs the deterministic (eps=0) policy for ``eval_steps`` and sums the
            team reward of the *first* episode each parallel env completes. The team
            return is the per-agent reward averaged over agents (cooperative).
            """

            def _eval_step(state, _):
                env_state, last_obs, carry, rng = state
                rng, rng_step = jax.random.split(rng)
                obs_NA = _stack_agents(last_obs, agents)
                avail_NA = _avail_actions(env, env_state, info)
                resets_N = jnp.zeros((info.num_agents,))
                carry, q_NA = _forward_q(network.apply, params, carry, obs_NA, resets_N)
                act_N = _greedy_actions(q_NA, avail_NA)
                acts = {a: act_N[i] for i, a in enumerate(agents)}
                next_obs, next_env_state, rewards, dones, _ = env.step(
                    rng_step, env_state, acts
                )
                team_r = jnp.mean(_stack_agents(rewards, agents))
                return (next_env_state, next_obs, carry, rng), (team_r, dones["__all__"])

            def _one_env(rng_e):
                rng_e, rng_reset = jax.random.split(rng_e)
                obs0, env_state0 = env.reset(rng_reset)
                carry0 = _reset_carry(info.num_agents)
                _, (rews_T, dones_T) = jax.lax.scan(
                    _eval_step,
                    (env_state0, obs0, carry0, rng_e),
                    None,
                    cfg.eval_steps,
                )
                # Sum reward up to and including the first episode termination.
                first_done = jnp.argmax(dones_T)
                # If no done in the window, count all steps.
                any_done = jnp.any(dones_T)
                last_idx = jnp.where(any_done, first_done, cfg.eval_steps - 1)
                mask_T = jnp.arange(cfg.eval_steps) <= last_idx
                return jnp.sum(jnp.where(mask_T, rews_T, 0.0))

            rngs = jax.random.split(rng, cfg.eval_episodes)
            returns = jax.vmap(_one_env)(rngs)
            return jnp.mean(returns)

        # --------------------------------------------------- run the scan
        rng, rng_eval0 = jax.random.split(rng)
        start_return = greedy_return(train_state.params, rng_eval0)

        runner, step_metrics = jax.lax.scan(
            _step, runner, None, length=cfg.num_updates()
        )

        rng_final = runner["rng"]
        rng_final, rng_evalN = jax.random.split(rng_final)
        end_return = greedy_return(runner["train_state"].params, rng_evalN)

        return {
            "start_greedy_return": start_return,
            "end_greedy_return": end_return,
            "final_params": runner["train_state"].params,
            "loss": step_metrics["loss"],
            "epsilon": step_metrics["epsilon"],
            # Per-iteration greedy-return curve and the env-step at each eval. The
            # curve is all-zeros unless eval_during_training was set (the ablation
            # sets it); return_steps is always meaningful (cumulative env steps).
            "returns": step_metrics["greedy_return"],
            "return_steps": step_metrics["step"],
        }

    return train
