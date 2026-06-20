"""Fast, committed learning test for the M2 IQL backbone.

The real acceptance gate is **it must learn**. This module runs a *small* IQL
config (tiny net, a handful of seeds, a few thousand env steps) on the
Simultaneous-Attack coordination bandit and asserts **non-vacuous learning**: the
mean final greedy return clears a robust threshold *and* improves clearly over the
random-policy baseline. It is calibrated to be fast (well under ~2 min on CPU; in
practice a couple of seconds) and robust across seeds, not flaky.

simple_spread is intentionally *not* exercised here (its 25-step horizon makes a
meaningful run too slow for a unit test); its learning numbers are reported in the
PR instead, per the milestone.

Also included:
* a vmap-over-seeds shape/execution check (``train`` must be jit+vmap compatible and
  return per-seed metrics of the right shape);
* a sampler-seam check (the ``uniform`` sampler returns unit weights and a batch of
  the configured size, and the registry rejects unknown names);
* a transition-schema check (the stored joint action lets ``u_{-a}`` be recovered).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conformal_marl.envs.simultaneous_attack import SimultaneousAttack
from conformal_marl.iql import IQLConfig, get_env_info, make_train
from conformal_marl.replay import empty_transition, make_replay_buffer
from conformal_marl.samplers import (
    SamplerContext,
    get_sampler,
    list_samplers,
    uniform_sampler,
)


def _fast_config() -> IQLConfig:
    """A deliberately small config that still learns Simu reliably and fast."""
    return IQLConfig(
        total_timesteps=8_000,
        num_envs=8,
        num_steps=8,
        num_grad_steps=8,
        learning_starts=200,
        hidden_dim=32,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=5_000,
        target_update_interval=50,
        epsilon_anneal_time=4_000,
        eval_episodes=64,
        eval_steps=2,  # 1-step env
    )


def _random_baseline(env, num_episodes: int, rng) -> float:
    """Mean team return of a uniformly random policy on a 1-step env."""
    info = get_env_info(env)
    agents = info.agents

    def _one(rng_e):
        rng_e, rng_reset, rng_a, rng_s = jax.random.split(rng_e, 4)
        _, env_state = env.reset(rng_reset)
        akeys = jax.random.split(rng_a, len(agents))
        acts = {a: env.action_space(a).sample(akeys[i]) for i, a in enumerate(agents)}
        _, _, rewards, _, _ = env.step(rng_s, env_state, acts)
        return jnp.mean(jnp.stack([rewards[a] for a in agents]))

    return float(jnp.mean(jax.vmap(_one)(jax.random.split(rng, num_episodes))))


def test_iql_learns_simultaneous_attack():
    """IQL with the uniform sampler learns to coordinate on all-attack.

    Asserts (mean over seeds): final greedy return >= 0.5 (a robust threshold --
    perfect coordination scores 1.0, so 0.5 is comfortably non-vacuous yet not
    flaky) AND improvement over the random baseline >= 0.3.
    """
    cfg = _fast_config()
    env = SimultaneousAttack(num_agents=2)

    rng = jax.random.PRNGKey(0)
    rng, rng_base = jax.random.split(rng)
    baseline = _random_baseline(env, 256, rng_base)
    # The env is calibrated so a uniform policy scores ~0; sanity-check that.
    assert abs(baseline) < 0.2, f"random baseline should be ~0, got {baseline}"

    train = make_train(cfg, env)
    seeds = 4
    rngs = jax.random.split(rng, seeds)
    out = jax.block_until_ready(jax.jit(jax.vmap(train))(rngs))

    end = np.asarray(out["end_greedy_return"])
    mean_end = float(np.mean(end))

    assert mean_end >= 0.5, (
        f"IQL failed to learn Simu: mean final greedy return {mean_end:.3f} < 0.5 "
        f"(per-seed {np.round(end, 3).tolist()})"
    )
    assert mean_end - baseline >= 0.3, (
        f"IQL did not improve clearly over random: end {mean_end:.3f} vs baseline "
        f"{baseline:.3f} (delta {mean_end - baseline:.3f} < 0.3)"
    )


def test_train_is_vmappable_over_seeds():
    """``train`` is jit+vmap compatible and returns per-seed metrics of right shape."""
    cfg = _fast_config()
    cfg.total_timesteps = 1_500  # keep this check snappy; shapes are the point
    env = SimultaneousAttack(num_agents=2)
    train = make_train(cfg, env)

    seeds = 5
    rngs = jax.random.split(jax.random.PRNGKey(1), seeds)
    out = jax.block_until_ready(jax.jit(jax.vmap(train))(rngs))

    # Per-seed scalar returns.
    assert out["start_greedy_return"].shape == (seeds,)
    assert out["end_greedy_return"].shape == (seeds,)
    # Per-seed, per-step loss/epsilon time series.
    assert out["loss"].shape == (seeds, cfg.num_updates())
    assert out["epsilon"].shape == (seeds, cfg.num_updates())
    assert bool(jnp.all(jnp.isfinite(out["loss"])))
    # Epsilon is annealed within [finish, start].
    assert float(out["epsilon"].min()) >= cfg.epsilon_finish - 1e-6
    assert float(out["epsilon"].max()) <= cfg.epsilon_start + 1e-6


def test_uniform_sampler_contract():
    """The uniform sampler returns a batch of the configured size with unit weights."""
    num_agents, obs_dim, num_actions = 2, 2, 2
    batch_size = 16
    buffer = make_replay_buffer(max_size=128, min_size=8, sample_batch_size=batch_size)
    state = buffer.init(empty_transition(num_agents, obs_dim, num_actions))

    # Fill past min_size with a batch of distinct transitions.
    base = empty_transition(num_agents, obs_dim, num_actions)
    batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (32,) + x.shape), base)
    state = buffer.add(state, batch)
    assert bool(buffer.can_sample(state))

    ctx = SamplerContext(
        online_params={}, target_params={}, apply_fn=lambda *a: a,
        num_agents=num_agents, num_actions=num_actions, hidden_dim=8,
    )
    out_batch, weights, info = uniform_sampler(
        buffer, state, jax.random.PRNGKey(0), ctx, batch_size
    )
    assert weights.shape == (batch_size,)
    np.testing.assert_allclose(np.asarray(weights), np.ones(batch_size))
    assert out_batch.actions.shape == (batch_size, num_agents)
    assert isinstance(info, dict)


def test_sampler_registry():
    """``uniform`` is registered; unknown names raise a clear error."""
    assert "uniform" in list_samplers()
    assert get_sampler("uniform") is uniform_sampler
    with pytest.raises(KeyError):
        get_sampler("definitely_not_a_sampler")


def test_transition_stores_joint_action():
    """The stored joint action lets every agent recover the others' actions u_{-a}."""
    num_agents, obs_dim, num_actions = 3, 3, 2
    t = empty_transition(num_agents, obs_dim, num_actions)
    # actions has a per-agent leading axis: agent a's own action is actions[a],
    # and u_{-a} = actions[[j for j != a]] -- recoverable from one stored field.
    assert t.actions.shape == (num_agents,)
    assert t.behavior_log_prob.shape == (num_agents,)
    assert t.avail_actions.shape == (num_agents, num_actions)
    # The next-state mask (M8) carries the same per-agent (N, A) shape as avail_actions.
    assert t.next_avail_actions.shape == (num_agents, num_actions)
    # u_{-0} for agent 0 is just the other two entries.
    others = t.actions[jnp.array([1, 2])]
    assert others.shape == (2,)


# ===================================================== P1: next-state target masking
def test_double_dqn_target_uses_next_state_mask():
    """The double-DQN target masks Q(next_obs) with ``next_avail`` -- a changed next-state
    legal set picks a different bootstrap action (the SMAX correctness fix).

    Hand-built two-row example: at ``s'`` the highest Q is action 2, but if ``next_avail``
    forbids action 2 the masked argmax falls to the next-best *available* action, so the
    target changes. With an all-ones mask the unrestricted argmax (action 2) is used.
    """
    from conformal_marl.iql import double_dqn_target

    # online == target Q at s' (so the target is exactly Q at the chosen action); one row.
    q_next = jnp.array([[0.1, 0.5, 0.9, 0.2]])  # (1, 4): argmax is action 2 (Q=0.9)
    reward = jnp.array([0.0])
    done = jnp.array([0.0])
    gamma = 1.0

    # All actions available -> unrestricted argmax (action 2) -> target == 0.9.
    all_avail = jnp.ones((1, 4))
    t_all = double_dqn_target(q_next, q_next, all_avail, reward, done, gamma)
    np.testing.assert_allclose(np.asarray(t_all), [0.9], atol=1e-6)

    # Action 2 NOT available at s' -> masked argmax is action 1 (Q=0.5) -> target == 0.5.
    # This is the value the *obs*-mask (which still allowed action 2) would have gotten
    # wrong; using the next-state mask fixes it.
    next_avail = jnp.array([[1.0, 1.0, 0.0, 1.0]])
    t_masked = double_dqn_target(q_next, q_next, next_avail, reward, done, gamma)
    np.testing.assert_allclose(np.asarray(t_masked), [0.5], atol=1e-6)
    assert float(t_all[0]) != float(t_masked[0])

    # Terminal transition: the bootstrap is zeroed regardless of the mask.
    t_term = double_dqn_target(q_next, q_next, all_avail, reward, jnp.array([1.0]), gamma)
    np.testing.assert_allclose(np.asarray(t_term), [0.0], atol=1e-6)


@pytest.mark.slow
def test_smax_next_avail_mask_changes_between_steps():
    """On SMAX (a real action-masked env) the available-action set at ``next_obs`` differs
    from the set at ``obs`` -- the case the M8 next-state-mask fix exists for.

    Without the fix the double-DQN target would mask Q(next_obs) with the *obs* mask. Here
    we confirm the two masks genuinely diverge over a short rollout (so storing the
    next-state mask is load-bearing, not cosmetic). Marked ``slow`` because building SMAX
    is heavier than the bandit tests.
    """
    from conformal_marl.envs.registry import make_env

    env, spec = make_env("SMAX", map_name="3m")
    assert spec.masks_actions  # SMAX masks; precondition for the fix to matter.
    agents = list(env.agents)

    def avail(state):  # per-agent (N, A) mask, as the IQL loop builds it
        m = env.get_avail_actions(state)
        return jnp.stack([m[a] for a in agents], axis=0)

    # Roll a short masked-random episode; record (avail_at_obs, avail_at_next) each step.
    obs, state = env.reset(jax.random.PRNGKey(0))

    def _step(carry, k):
        state, rng = carry
        rng, k_act, k_step = jax.random.split(rng, 3)
        cur = avail(state)  # (N, A) at current state
        # A legal random action per agent (categorical over the mask) so steps progress.
        akeys = jax.random.split(k_act, len(agents))
        acts_N = jax.vmap(lambda lg, kk: jax.random.categorical(kk, jnp.log(lg + 1e-9)))(
            cur, akeys
        )
        acts = {a: acts_N[i] for i, a in enumerate(agents)}
        _, nstate, _, _, _ = env.step(k_step, state, acts)
        nxt = avail(nstate)  # (N, A) at next state
        return (nstate, rng), (cur, nxt)

    _, (cur_T, nxt_T) = jax.lax.scan(
        _step, (state, jax.random.PRNGKey(1)), jax.random.split(jax.random.PRNGKey(2), 32)
    )
    # The fix exists because these can differ; assert they actually do somewhere.
    assert bool(jnp.any(cur_T != nxt_T)), (
        "expected SMAX's available-action set to change between steps"
    )
