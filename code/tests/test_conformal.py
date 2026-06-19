"""Rigorous correctness tests for the M3 conformal selection machinery.

This is the correctness gate for the scientific core of the paper, so the tests are
written to be *non-vacuous*: they hand-derive the score algebra on a controlled
example, validate the conformal p-value construction against the uniform null both
in the clean continuous case (precise) and end-to-end on real transition scores
(honest super-uniformity), check the Gumbel-top-k subsample is distinct / jittable /
correctly biased, and confirm both new samplers plug into ``make_train`` with no edit
to the sampler seam and still *learn* Simultaneous-Attack.

All tests are calibrated to run fast on CPU (well under the ~2 min committed budget).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.conformal import (
    conformal_p_values,
    current_log_probs,
    exact_scores,
    gumbel_top_k,
    policy_scores,
)
from conformal_marl.envs.simultaneous_attack import SimultaneousAttack
from conformal_marl.iql import IQLConfig, make_train
from conformal_marl.networks import QNetwork
from conformal_marl.replay import empty_transition
from conformal_marl.samplers import get_sampler, list_samplers


# --------------------------------------------------------------------------- helpers
def _tiny_net(num_actions: int, hidden: int, obs_dim: int, seed: int):
    """A small initialised QNetwork + params for scoring tests."""
    net = QNetwork(action_dim=num_actions, hidden_dim=hidden)
    carry = QNetwork.initialize_carry(hidden, 1)
    params = net.init(
        jax.random.PRNGKey(seed), carry, (jnp.zeros((1, 1, obs_dim)), jnp.zeros((1, 1)))
    )
    return net, params


def _make_batch(key, m, num_agents, obs_dim, num_actions, past_p=0.5):
    """A batch of ``m`` transitions with random obs/actions and a fixed behaviour prob."""
    base = empty_transition(num_agents, obs_dim, num_actions)
    k_obs, k_act = jax.random.split(key)
    obs = jax.random.normal(k_obs, (m, num_agents, obs_dim))
    actions = jax.random.randint(k_act, (m, num_agents), 0, num_actions).astype(jnp.int32)
    # Feed-forward batch: the recurrent carry fields are zero-width but must still carry
    # the (m, N) leading axes so the Transition pytree is batch-consistent (M7 schema).
    zero_h = jnp.zeros((m, num_agents, 0), dtype=jnp.float32)
    batch = base.replace(
        obs=obs,
        actions=actions,
        behavior_log_prob=jnp.full((m, num_agents), jnp.log(past_p)),
        avail_actions=jnp.ones((m, num_agents, num_actions)),
        reward=jnp.zeros((m, num_agents)),
        done=jnp.zeros((m, num_agents), dtype=jnp.bool_),
        next_obs=obs,
        hidden=zero_h,
        next_hidden=zero_h,
    )
    return batch


# ============================================================ 1. score correctness
def test_score_correctness_hand_derivation():
    """``A^exact``/``A^policy`` match a from-scratch hand computation, others-only.

    On a controlled batch (fixed net, fixed eps, known stored actions) we recompute
    ``log pi_cur``, the per-agent greedy choice, and the "leave-one-out over j != a"
    aggregation by hand, and assert the module reproduces them to tolerance, including
    that ``A^exact = -log(pi_cur / pi_past)`` and ``A^policy = -log pi_cur`` summed
    over the *other* agents only.
    """
    num_agents, obs_dim, num_actions, hidden = 3, 4, 2, 8
    net, params = _tiny_net(num_actions, hidden, obs_dim, seed=7)
    eps = 0.3

    b = 5
    key = jax.random.PRNGKey(11)
    k_obs, k_act, k_past = jax.random.split(key, 3)
    obs = jax.random.normal(k_obs, (b, num_agents, obs_dim))
    actions = jax.random.randint(k_act, (b, num_agents), 0, num_actions).astype(jnp.int32)
    # Distinct, non-trivial behaviour probabilities so pi_past actually matters.
    past_p = jax.random.uniform(k_past, (b, num_agents), minval=0.2, maxval=0.8)
    past_lp = jnp.log(past_p)
    avail = jnp.ones((b, num_agents, num_actions))
    base = empty_transition(num_agents, obs_dim, num_actions)
    zero_h = jnp.zeros((b, num_agents, 0), dtype=jnp.float32)
    batch = base.replace(
        obs=obs, actions=actions, behavior_log_prob=past_lp, avail_actions=avail,
        reward=jnp.zeros((b, num_agents)), done=jnp.zeros((b, num_agents), dtype=jnp.bool_),
        next_obs=obs, hidden=zero_h, next_hidden=zero_h,
    )

    # --- hand computation of log pi_cur ---
    flat_obs = obs.reshape(b * num_agents, obs_dim)
    carry = QNetwork.initialize_carry(hidden, b * num_agents)
    _, q = net.apply(params, carry, (flat_obs[jnp.newaxis], jnp.zeros((1, b * num_agents))))
    q = q[0].reshape(b, num_agents, num_actions)
    greedy = jnp.argmax(q, axis=-1)  # all actions available -> plain argmax
    # eps-greedy prob of the taken action; |avail| = num_actions here.
    p_cur = jnp.where(
        actions == greedy, (1.0 - eps) + eps / num_actions, eps / num_actions
    )
    logp_cur_manual = jnp.log(p_cur)

    logp_cur_code = current_log_probs(batch, params, net.apply, eps, hidden)
    np.testing.assert_allclose(
        np.asarray(logp_cur_manual), np.asarray(logp_cur_code), atol=1e-5
    )

    # --- A^policy_a = -sum_{j != a} log pi_cur_j ---
    total = jnp.sum(logp_cur_manual, axis=-1, keepdims=True)
    a_policy_manual = -(total - logp_cur_manual)  # leave-one-out
    a_policy_code = policy_scores(batch, params, net.apply, eps, hidden)
    np.testing.assert_allclose(
        np.asarray(a_policy_manual), np.asarray(a_policy_code), atol=1e-5
    )

    # --- A^exact_a = sum_{j != a} (log pi_past_j - log pi_cur_j) = -log rho_a ---
    diff = past_lp - logp_cur_manual
    diff_total = jnp.sum(diff, axis=-1, keepdims=True)
    a_exact_manual = diff_total - diff
    a_exact_code = exact_scores(batch, params, net.apply, eps, hidden)
    np.testing.assert_allclose(
        np.asarray(a_exact_manual), np.asarray(a_exact_code), atol=1e-5
    )

    # --- explicit others-only check for agent 0: must exclude its own column ---
    a0_policy_explicit = -jnp.sum(logp_cur_manual[:, 1:], axis=-1)  # agents 1,2 only
    np.testing.assert_allclose(
        np.asarray(a0_policy_explicit), np.asarray(a_policy_code[:, 0]), atol=1e-5
    )
    # And the exact score for agent 0 excludes its own (cur, past) too.
    a0_exact_explicit = jnp.sum(diff[:, 1:], axis=-1)
    np.testing.assert_allclose(
        np.asarray(a0_exact_explicit), np.asarray(a_exact_code[:, 0]), atol=1e-5
    )


def test_exact_score_is_minus_log_ratio():
    """``A^exact`` equals ``-log(prod_{j!=a} pi_cur_j / pi_past_j)`` exactly.

    A second, independent phrasing of score 1: assemble the ratio directly from
    ``current_log_probs`` and the stored ``behavior_log_prob`` and check
    ``A^exact_a == -log rho_a`` per agent.
    """
    num_agents, obs_dim, num_actions, hidden = 4, 3, 2, 8
    net, params = _tiny_net(num_actions, hidden, obs_dim, seed=3)
    eps = 0.25
    batch = _make_batch(jax.random.PRNGKey(21), 6, num_agents, obs_dim, num_actions, 0.4)

    logp_cur = current_log_probs(batch, params, net.apply, eps, hidden)  # (B, N)
    logp_past = batch.behavior_log_prob  # (B, N)
    log_ratio = logp_cur - logp_past  # log(pi_cur/pi_past), per agent
    # -log rho_a = -sum_{j!=a} log_ratio_j
    total = jnp.sum(log_ratio, axis=-1, keepdims=True)
    a_exact_expected = -(total - log_ratio)

    a_exact = exact_scores(batch, params, net.apply, eps, hidden)
    np.testing.assert_allclose(np.asarray(a_exact_expected), np.asarray(a_exact), atol=1e-5)


# ===================================================== 2. p-value super-uniformity
def test_pvalues_uniform_under_h0_continuous():
    """KEY TEST (precise): p-values are uniform when scores are exchangeable.

    With continuous i.i.d. scores for calibration and test (the cleanest H0, no
    ties), the inductive-conformal p-value is *exactly* uniform on the grid
    ``{1/(n+1), ..., 1}``. We pool many trials and assert: mean p ~ 0.5, the KS
    statistic vs Uniform(0, 1) is below its 0.05 critical value, and the empirical
    CDF matches the uniform CDF (and never exceeds it beyond slack) at several alphas.
    This directly validates the rank/p-value construction in ``conformal_p_values``.
    """
    n_cal, b, num_agents, trials = 500, 200, 2, 40
    key = jax.random.PRNGKey(0)
    chunks = []
    for _ in range(trials):
        key, k1, k2 = jax.random.split(key, 3)
        cal = jax.random.normal(k1, (n_cal, num_agents))
        test = jax.random.normal(k2, (b, num_agents))
        p = conformal_p_values(test, cal)
        chunks.append(np.asarray(p).reshape(-1))
    ps = np.concatenate(chunks)
    n = ps.size

    # All p-values live in (0, 1].
    assert ps.min() > 0.0 and ps.max() <= 1.0 + 1e-6

    # Mean ~ 0.5 (uniform). With ~16k samples this is tight.
    assert abs(ps.mean() - 0.5) < 0.02, f"mean p = {ps.mean():.4f}, expected ~0.5"

    # Kolmogorov-Smirnov vs Uniform(0, 1): below the 0.05 critical value 1.36/sqrt(n).
    xs = np.sort(ps)
    ecdf = np.arange(1, n + 1) / n
    ks = float(np.max(np.abs(ecdf - xs)))
    ks_crit = 1.36 / np.sqrt(n)
    assert ks < ks_crit, f"KS {ks:.4f} >= critical {ks_crit:.4f}: p-values not uniform"

    # CDF tracks uniform at several alphas (uniform => P(p<=a) ~ a).
    for a in (0.1, 0.25, 0.5, 0.75, 0.9):
        emp = float(np.mean(ps <= a))
        assert abs(emp - a) < 0.03, f"P(p<={a})={emp:.3f} deviates from uniform"


def test_pvalues_superuniform_under_h0_real_scores():
    """End-to-end (honest): real transition scores give *super*-uniform p-values.

    Here calibration and test transitions are drawn from the SAME generating
    distribution (exchangeable) and scored by the actual ``policy_scores`` under a
    fixed policy. Because the discrete env (2 actions, fixed eps) makes the score take
    only a few distinct values, the conformal guarantee holds in its conservative
    form: ``P(p <= alpha) <= alpha`` (super-uniform), with mean p >= 0.5. We assert
    the one-sided domination at several alphas -- which is exactly the validity
    property conformal selection relies on -- and document that ties make it
    conservative rather than exactly uniform.
    """
    num_agents, obs_dim, num_actions, hidden = 3, 4, 2, 16
    net, params = _tiny_net(num_actions, hidden, obs_dim, seed=0)
    eps = 0.2
    n_cal, trials = 200, 1500

    key = jax.random.PRNGKey(1)
    ps = []
    for _ in range(trials):
        key, kc, kt = jax.random.split(key, 3)
        cal = _make_batch(kc, n_cal, num_agents, obs_dim, num_actions)
        test = _make_batch(kt, 1, num_agents, obs_dim, num_actions)
        cs = policy_scores(cal, params, net.apply, eps, hidden)
        ts = policy_scores(test, params, net.apply, eps, hidden)
        p = conformal_p_values(ts, cs)  # (1, N)
        ps.append(float(p[0, 0]))  # agent 0
    ps = np.array(ps)

    assert ps.min() > 0.0 and ps.max() <= 1.0 + 1e-6
    # Super-uniform => conservative => mean at or above 0.5.
    assert ps.mean() >= 0.5 - 0.02, f"mean p = {ps.mean():.4f} (expected >= 0.5)"
    # One-sided domination P(p<=a) <= a (+ Monte-Carlo slack) -- the validity property.
    for a in (0.1, 0.25, 0.5, 0.75):
        emp = float(np.mean(ps <= a))
        assert emp <= a + 0.05, f"super-uniformity violated: P(p<={a})={emp:.3f} > {a}"


# ============================================================== 3. Gumbel-top-k
def test_gumbel_top_k_distinct_and_jittable():
    """Gumbel-top-k returns ``k`` distinct in-range indices and is jittable."""
    b, k = 64, 16
    log_w = jax.random.normal(jax.random.PRNGKey(0), (b,))

    idx = gumbel_top_k(log_w, k, jax.random.PRNGKey(1))
    idx_np = np.asarray(idx)
    assert idx_np.shape == (k,)
    assert len(set(idx_np.tolist())) == k, "indices must be distinct (no replacement)"
    assert idx_np.min() >= 0 and idx_np.max() < b

    # Jittable with static k.
    jitted = jax.jit(lambda lw, key: gumbel_top_k(lw, k, key))
    idx2 = jitted(log_w, jax.random.PRNGKey(1))
    np.testing.assert_array_equal(np.asarray(idx2), idx_np)


def test_gumbel_top_k_biased_toward_high_weight():
    """Higher-weight items are selected with higher empirical frequency.

    A statistical assertion: with weights increasing in the index, the top half of
    indices should be selected far more often than the bottom half across many draws.
    """
    b, k, draws = 20, 5, 4000
    # Weights increasing in index: item i has weight proportional to exp(i * 0.4).
    log_w = jnp.arange(b, dtype=jnp.float32) * 0.4

    keys = jax.random.split(jax.random.PRNGKey(0), draws)
    all_idx = jax.vmap(lambda key: gumbel_top_k(log_w, k, key))(keys)  # (draws, k)
    counts = np.bincount(np.asarray(all_idx).reshape(-1), minlength=b)
    freq = counts / counts.sum()

    top_half = freq[b // 2:].sum()
    bottom_half = freq[: b // 2].sum()
    assert top_half > 0.8, f"high-weight half selected only {top_half:.2f} of the time"
    assert top_half > 5 * bottom_half, (
        f"selection not strongly biased: top {top_half:.3f} vs bottom {bottom_half:.3f}"
    )
    # Monotone-ish: the single largest-weight item beats the single smallest.
    assert freq[-1] > freq[0]


# ============================================================== 4. integration
def _fast_conformal_config(sampler_name: str) -> IQLConfig:
    """Small config that still learns Simu and exercises the LaBER (L>B) subsample."""
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
        large_batch_size=256,  # draw L=256, subsample to B=64 (conformal selection)
        cal_size=64,
        buffer_size=5_000,
        target_update_interval=50,
        epsilon_anneal_time=4_000,
        eval_episodes=64,
        eval_steps=2,
        sampler_name=sampler_name,
    )


def test_conformal_samplers_registered():
    """Both M3 samplers are in the registry and resolvable by name."""
    names = list_samplers()
    assert "conformal_policy" in names
    assert "exact_ratio" in names
    assert get_sampler("conformal_policy") is not None
    assert get_sampler("exact_ratio") is not None


def test_conformal_policy_learns_simu():
    """``conformal_policy`` plugs into make_train (no seam edit), jit+vmap, and learns.

    Conformal selection must not break learning: a short run reaches a clearly
    positive mean greedy return on Simultaneous-Attack.
    """
    cfg = _fast_conformal_config("conformal_policy")
    env = SimultaneousAttack(num_agents=2)
    train = make_train(cfg, env)

    rngs = jax.random.split(jax.random.PRNGKey(0), 4)
    out = jax.block_until_ready(jax.jit(jax.vmap(train))(rngs))

    end = np.asarray(out["end_greedy_return"])
    mean_end = float(np.mean(end))
    assert bool(np.all(np.isfinite(np.asarray(out["loss"]))))
    assert mean_end >= 0.5, (
        f"conformal_policy failed to learn Simu: mean final return {mean_end:.3f} "
        f"< 0.5 (per-seed {np.round(end, 3).tolist()})"
    )


def test_exact_ratio_learns_simu():
    """``exact_ratio`` plugs in the same way (no seam edit), jit+vmap, and learns."""
    cfg = _fast_conformal_config("exact_ratio")
    env = SimultaneousAttack(num_agents=2)
    train = make_train(cfg, env)

    rngs = jax.random.split(jax.random.PRNGKey(0), 4)
    out = jax.block_until_ready(jax.jit(jax.vmap(train))(rngs))

    end = np.asarray(out["end_greedy_return"])
    mean_end = float(np.mean(end))
    assert bool(np.all(np.isfinite(np.asarray(out["loss"]))))
    assert mean_end >= 0.5, (
        f"exact_ratio failed to learn Simu: mean final return {mean_end:.3f} < 0.5 "
        f"(per-seed {np.round(end, 3).tolist()})"
    )


def test_conformal_sampler_returns_per_agent_weights():
    """The conformal samplers return per-agent ``(B, N)`` weights in (0, 1].

    Documents/locks the per-agent design: each agent weights its own action-column by
    its own p-value. Built directly via the sampler API (the same path make_train
    uses), with a fresh calibration batch in the context.
    """
    from conformal_marl.replay import make_replay_buffer
    from conformal_marl.samplers import SamplerContext

    num_agents, obs_dim, num_actions, hidden = 2, 2, 2, 16
    big_l, trained_b = 128, 32
    net, params = _tiny_net(num_actions, hidden, obs_dim, seed=0)
    buffer = make_replay_buffer(max_size=512, min_size=8, sample_batch_size=big_l)
    state = buffer.init(empty_transition(num_agents, obs_dim, num_actions))

    fill = _make_batch(jax.random.PRNGKey(5), 256, num_agents, obs_dim, num_actions)
    state = buffer.add(state, fill)
    cal = _make_batch(jax.random.PRNGKey(6), 64, num_agents, obs_dim, num_actions)

    ctx = SamplerContext(
        online_params=params, target_params=params, apply_fn=net.apply,
        num_agents=num_agents, num_actions=num_actions, hidden_dim=hidden,
        epsilon=jnp.array(0.2), cal_batch=cal,
    )
    sampler = get_sampler("conformal_policy")
    batch, weights, info = sampler(buffer, state, jax.random.PRNGKey(7), ctx, trained_b)

    assert weights.shape == (trained_b, num_agents), "weights must be per-agent (B, N)"
    assert float(weights.min()) > 0.0 and float(weights.max()) <= 1.0 + 1e-6
    assert batch.actions.shape == (trained_b, num_agents)
    assert "p_values" in info
