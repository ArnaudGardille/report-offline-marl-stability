"""Fast committed tests for the M6' support-starvation machinery (D2).

The heavy multi-seed demonstration lives in ``scripts/support_starvation.py`` (and its
committed figures); these tests only lock the *mechanism* it relies on, fast enough for
the committed budget:

1. The support-floored samplers (``exact_ratio_floor``, ``policy_floor``) are registered,
   plug into ``make_train`` with no seam edit, are jit+vmap-able, and learn Simu.
2. The **support floor genuinely floors the selection distribution**: with a floor, the
   Gumbel-top-k selection keeps a controlled all-attack (BR-support) mass in the batch
   even when the conformal score would otherwise starve it; with no floor (the M3
   ``exact_ratio``) the same construction can drive that mass to ~0. This is the
   constructive ``supp(d_w) >= supp(d^BR)`` guarantee (condition (b) of the theorem),
   tested directly on a buffer whose rare all-attack transitions the policy scores as
   non-conforming.
3. The ``all_attack_frac`` diagnostic threaded through ``make_train`` is in ``[0, 1]`` and
   is 0 when tracking is off (so the default path pays nothing).

The exact-ratio-vs-ratio-free *return* separation (the full "iff") is a multi-seed,
higher-variance result and is intentionally left to the script, not asserted here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.envs.simultaneous_attack import SimultaneousAttack
from conformal_marl.iql import IQLConfig, make_train
from conformal_marl.samplers import (
    _floored_log_select_weights,
    get_sampler,
    list_samplers,
)


def _fast_floor_config(sampler_name: str, floor: float) -> IQLConfig:
    """Small Simu config exercising the floored LaBER subsample + the diagnostic."""
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
        large_batch_size=256,
        cal_size=64,
        buffer_size=5_000,
        target_update_interval=50,
        epsilon_anneal_time=4_000,
        eval_episodes=64,
        eval_steps=2,
        sampler_name=sampler_name,
        support_floor=floor,
        track_batch_action_stats=True,
    )


# ============================================================ 1. registration + learning
def test_floor_samplers_registered():
    """Both support-floored samplers are in the registry and resolvable by name."""
    names = list_samplers()
    assert "exact_ratio_floor" in names
    assert "policy_floor" in names
    assert get_sampler("exact_ratio_floor") is not None
    assert get_sampler("policy_floor") is not None


def test_exact_ratio_floor_learns_simu():
    """``exact_ratio_floor`` (both theorem conditions) plugs in, jit+vmap, and learns Simu.

    The recovering configuration must of course still *learn*; a short run reaches a
    clearly positive mean greedy return. (The strong multi-seed "recovers BR with no
    collapses" claim is the script's job; here we only guard that the new sampler trains.)
    """
    cfg = _fast_floor_config("exact_ratio_floor", floor=0.3)
    env = SimultaneousAttack(num_agents=3)
    train = make_train(cfg, env)

    rngs = jax.random.split(jax.random.PRNGKey(0), 4)
    out = jax.block_until_ready(jax.jit(jax.vmap(train))(rngs))

    assert bool(np.all(np.isfinite(np.asarray(out["loss"]))))
    mean_end = float(np.mean(np.asarray(out["end_greedy_return"])))
    assert mean_end >= 0.5, f"exact_ratio_floor failed to learn Simu: mean {mean_end:.3f}"


def test_policy_floor_runs_and_is_finite():
    """``policy_floor`` (floor on, ratio-free weights) plugs in and stays finite under jit+vmap."""
    cfg = _fast_floor_config("policy_floor", floor=0.3)
    env = SimultaneousAttack(num_agents=3)
    train = make_train(cfg, env)

    rngs = jax.random.split(jax.random.PRNGKey(1), 4)
    out = jax.block_until_ready(jax.jit(jax.vmap(train))(rngs))
    assert bool(np.all(np.isfinite(np.asarray(out["loss"]))))
    # The diagnostic is populated and in range.
    aa = np.asarray(out["all_attack_frac"])
    assert aa.min() >= 0.0 and aa.max() <= 1.0 + 1e-6


# ============================================================ 2. the floor floors selection
def test_floored_log_select_weights_mixes_in_uniform_floor():
    """``_floored_log_select_weights`` is exactly ``log((1-f) softmax + f/L)``; floor=0 is identity.

    A direct algebraic check of the support-floor primitive: with ``f = 0`` the returned
    log-weights equal the input (the M3 pure-top-p behaviour); with ``f > 0`` the implied
    selection probabilities equal ``(1 - f) * softmax(log_p) + f / L`` elementwise and sum
    to 1, so every item carries probability ``>= f / L`` (no item can be starved out).
    """
    log_p = jnp.array([0.0, -2.0, -10.0, -20.0, 1.0])  # 3rd/4th would be starved by top-p
    length = log_p.shape[0]

    # floor = 0 -> identity (Gumbel-top-k is invariant to the softmax-normalising shift).
    np.testing.assert_allclose(
        np.asarray(_floored_log_select_weights(log_p, 0.0)), np.asarray(log_p), atol=1e-6
    )

    f = 0.25
    out = _floored_log_select_weights(log_p, f)
    probs = np.asarray(jnp.exp(out))  # these are the (normalised) selection probs
    expected = (1.0 - f) * np.asarray(jax.nn.softmax(log_p)) + f / length
    np.testing.assert_allclose(probs, expected, atol=1e-6)
    assert abs(probs.sum() - 1.0) < 1e-5
    # Every item keeps at least f/L mass -- the support guarantee.
    assert probs.min() >= f / length - 1e-6


def _gumbel_top_k_mass(log_sel, k, is_target, draws, seed):
    """Mean fraction of selected items that are 'target' under Gumbel-top-k(log_sel)."""
    from conformal_marl.conformal import gumbel_top_k

    keys = jax.random.split(jax.random.PRNGKey(seed), draws)
    sel = jax.vmap(lambda key: gumbel_top_k(log_sel, k, key))(keys)  # (draws, k)
    chosen_is_target = jnp.take_along_axis(
        is_target[jnp.newaxis, :].repeat(draws, axis=0), sel, axis=1
    )  # (draws, k)
    return float(jnp.mean(chosen_is_target))


def test_support_floor_keeps_br_support_in_selected_batch():
    """KEY (D2 mechanism): the floor keeps starved (low-score) BR-support items in the batch.

    The starvation regime, isolated from the network so the score ordering is controlled
    exactly: among ``L`` candidate transitions a small ``n_target`` fraction is the
    best-response support (the all-attack transitions) and is assigned a **strongly
    non-conforming** conformal score (tiny p -> very negative ``log p`` -> bottom of the
    selection order); the rest score as conforming. We then subsample ``B`` via the same
    Gumbel-top-k the samplers use:

    * **floor = 0** (M3 pure top-p): the target items sit at the bottom of the order and
      are essentially never selected -> the BR support is *starved* out of the batch.
    * **floor = 0.5** (the support-floored selection): the uniform-mixed distribution
      keeps target items at ~their candidate frequency -> the support is preserved.

    This pins the constructive ``supp(d_w) >= supp(d^BR)`` guarantee (condition (b)) on the
    exact ``_floored_log_select_weights`` -> Gumbel-top-k path the floored samplers use.
    """
    length, trained_b = 512, 128
    n_target = 32  # rare BR-support items
    target_freq = n_target / length  # ~0.0625

    is_target = jnp.zeros((length,), dtype=jnp.float32).at[:n_target].set(1.0)
    # Conformal log-selection score sum_a log p: target items get tiny p (very negative
    # log p), conforming items get p~1 (log p ~ 0). This is exactly the starvation order.
    log_p = jnp.where(is_target > 0, jnp.log(1e-4), jnp.log(0.95))

    log_sel_no_floor = _floored_log_select_weights(log_p, 0.0)
    log_sel_floor = _floored_log_select_weights(log_p, 0.5)

    mass_no_floor = _gumbel_top_k_mass(log_sel_no_floor, trained_b, is_target, 64, seed=10)
    mass_floor = _gumbel_top_k_mass(log_sel_floor, trained_b, is_target, 64, seed=10)

    # No-floor selection starves the BR support far below its candidate frequency.
    assert mass_no_floor < 0.25 * target_freq, (
        f"no-floor BR-support mass {mass_no_floor:.4f} not starved below "
        f"0.25*freq={0.25 * target_freq:.4f} (the starvation premise must hold)"
    )
    # The floor restores it toward the candidate frequency (support preserved).
    assert mass_floor > mass_no_floor, "floor failed to increase BR-support mass"
    assert mass_floor > 0.5 * target_freq, (
        f"floored BR-support mass {mass_floor:.4f} should approach candidate frequency "
        f"{target_freq:.4f} (support preserved by the floor)"
    )


# ============================================================ 3. diagnostic plumbing
def test_all_attack_frac_zero_when_tracking_off():
    """``all_attack_frac`` is identically 0 when ``track_batch_action_stats`` is off.

    Guards that the diagnostic is truly opt-in (the default fast/learning path pays
    nothing and reports a clean zero, not stale numbers).
    """
    cfg = _fast_floor_config("uniform", floor=0.0)
    cfg.track_batch_action_stats = False
    env = SimultaneousAttack(num_agents=3)
    train = make_train(cfg, env)
    out = jax.block_until_ready(jax.jit(train)(jax.random.PRNGKey(0)))
    aa = np.asarray(out["all_attack_frac"])
    assert np.all(aa == 0.0), "all_attack_frac must be 0 when tracking is disabled"
