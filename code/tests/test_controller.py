"""Validity gate for the M5 two-channel drift controller (``conformal_marl.controller``).

Three properties the paper's "two-channel controller" section claims, each tested on fast
**synthetic p-value streams** (no environment), mirroring the M4 martingale tests' style:

1. **Channel A keeps its guarantee.** The frozen alarm channel's anytime false-alarm rate
   is still ``<= delta`` under ``H0`` (Ville), because Channel A *is* the unmodified M4
   martingale on a frozen calibration set -- adding Channel B must not change that. We
   re-run the M4-style Monte-Carlo Ville check through the controller's Channel A.

2. **Channel B tracks the injected KL rate.** Feeding post-change streams whose p-values
   are stochastically smaller (larger conformal score == more drift == larger KL between
   the post-change p-law and the uniform null), the estimated drift rate (EWMA log-wealth
   slope) increases **monotonically** with the injected severity. We sweep a severity
   knob that provably orders the post-change distributions by KL and assert the estimated
   slope is monotone in it, near-zero at no drift, and strictly positive under drift.

3. **The separation holds.** Running Channel A with *any* Channel-B-driven adaptation
   applied downstream leaves Channel A's alarm decisions **bit-identical** -- the
   adaptation cannot perturb the guaranteed channel. We compare the alarm mask / first
   alarm across wildly different adaptation functions and against the plain frozen
   martingale.

Plus a small contract test (shapes, jit/vmap) and the knob-mapping monotonicity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.controller import (
    drift_rate_to_half_life,
    ewma_logwealth_slope,
    run_with_adaptive_calibration,
    staleness_weight,
    two_channel_controller,
    windowed_logwealth_slope,
)
from conformal_marl.martingale import run_martingale


# --------------------------------------------------------------------------- helpers
def _uniform_streams(key, n_streams: int, length: int):
    """``(n_streams, length)`` i.i.d. Uniform(0, 1] p-values (the H0 stream)."""
    u = jax.random.uniform(key, (n_streams, length), dtype=jnp.float32)
    return 1.0 - u  # in (0, 1]


def _drift_streams(key, n_streams, pre_len, post_len, ceil):
    """H0 prefix then post-change p-values ~ Uniform(0, ceil].

    Smaller ``ceil`` => p-values concentrated nearer 0 => the post-change law is
    stochastically smaller => strictly larger KL to Uniform(0,1] (the null). So ``ceil``
    is an inverse severity knob: decreasing ``ceil`` increases the injected KL rate.
    """
    k_pre, k_post = jax.random.split(key)
    pre = _uniform_streams(k_pre, n_streams, pre_len)
    post = ceil * (1.0 - jax.random.uniform(k_post, (n_streams, post_len)))
    return jnp.concatenate([pre, post], axis=1)


# ============================================== 1. Channel A keeps the Ville guarantee
def test_channel_a_false_alarm_still_controlled():
    """KEY: the frozen alarm channel's ever-cross fraction is ``<= delta`` under H0.

    Channel A is the M4 grid-mixture martingale on a frozen calibration set; wrapping it
    in the two-channel controller (which additionally computes Channel B) must not touch
    its alarm. We run the controller on many i.i.d. uniform streams and check the
    fraction that *ever* alarm is at or below ``delta`` within Monte-Carlo error, for two
    deltas -- exactly the Ville false-alarm guarantee, verified end-to-end through the
    controller.
    """
    n_streams, length = 20_000, 200
    ps = _uniform_streams(jax.random.PRNGKey(11), n_streams, length)

    for delta in (0.1, 0.05):
        run = jax.jit(jax.vmap(lambda p: two_channel_controller(p, delta=delta)))
        res = run(ps)
        frac = float(np.asarray(res.alarm).mean())
        se = float(np.sqrt(delta * (1 - delta) / n_streams))
        assert frac <= delta + 3 * se, (
            f"Channel A Ville violated: delta={delta} empirical ever-alarm={frac:.4f} "
            f"> delta + 3*SE = {delta + 3 * se:.4f}"
        )


def test_channel_a_matches_plain_martingale_exactly():
    """Channel A's alarm trace is identical to ``run_martingale`` (reset-on-alarm).

    The controller must not alter the alarm channel: its ``alarms`` / ``first_alarm`` /
    ``log_m`` should equal the bare M4 martingale run with the same delta and grid. A
    deterministic equality check on a drift stream (so there is an actual alarm to match).
    """
    p = _drift_streams(jax.random.PRNGKey(3), 1, 80, 80, ceil=0.05)[0]
    ctrl = two_channel_controller(p, delta=0.01)
    mart = run_martingale(p, delta=0.01, reset_on_alarm=True)

    np.testing.assert_array_equal(np.asarray(ctrl.alarms), np.asarray(mart.alarms))
    assert int(ctrl.first_alarm) == int(mart.first_alarm)
    assert bool(ctrl.alarm) == bool(mart.crossed)
    np.testing.assert_allclose(np.asarray(ctrl.log_m), np.asarray(mart.log_m), rtol=1e-5)


# ============================================== 2. Channel B tracks the injected KL rate
def test_drift_rate_monotone_in_injected_kl():
    """KEY: the estimated drift rate increases monotonically with the injected KL rate.

    We inject post-change p-value distributions ``Uniform(0, ceil]`` of increasing
    severity (decreasing ``ceil`` => stochastically smaller p => strictly larger KL to the
    uniform null). For each severity we measure the controller's Channel-B drift rate,
    averaged over streams and over the post-change window (steady-state slope). The
    sequence of mean post-change drift rates must be **strictly increasing** in severity,
    near zero for the no-drift control, and strictly positive once drift is present. This
    is the "the adaptive estimate tracks an injected KL rate" requirement.
    """
    n_streams, pre_len, post_len = 600, 120, 120
    # ceil = 1.0 is the no-drift control (post == uniform); decreasing ceil raises KL.
    ceils = [1.0, 0.5, 0.2, 0.1, 0.05]

    mean_rates = []
    for i, ceil in enumerate(ceils):
        streams = _drift_streams(
            jax.random.PRNGKey(100 + i), n_streams, pre_len, post_len, ceil
        )
        run = jax.jit(jax.vmap(lambda p: two_channel_controller(p, delta=0.01)))
        res = run(streams)
        # Steady-state post-change slope: average the drift-rate over the *second half*
        # of the post-change window (after the EWMA has caught up), across streams.
        rate = np.asarray(res.drift_rate)  # (n_streams, T)
        tail = rate[:, pre_len + post_len // 2:]
        mean_rates.append(float(tail.mean()))

    # No-drift control sits near zero (the mixture gently decays => slope <~ 0).
    assert mean_rates[0] < 0.05, (
        f"no-drift drift-rate not ~0: {mean_rates[0]:.4f}"
    )
    # Strictly increasing in severity (each more-severe drift gives a larger slope).
    for a, b in zip(mean_rates, mean_rates[1:]):
        assert b > a + 1e-3, (
            f"drift rate not monotone in injected KL: {mean_rates} "
            f"(expected strictly increasing)"
        )
    # The strongest drift is clearly positive (a real, large KL rate).
    assert mean_rates[-1] > 0.3, (
        f"strongest-drift slope too small: {mean_rates[-1]:.4f}; sequence {mean_rates}"
    )


def test_drift_rate_near_zero_under_pure_h0():
    """Under a pure ``H0`` stream the drift rate stays near zero throughout (no false drift).

    A no-change uniform stream should give a Channel-B drift rate that hovers around zero
    (slightly negative on average, as the betting mixture loses wealth under the null).
    We assert the mean is small in magnitude and not systematically positive.
    """
    streams = _uniform_streams(jax.random.PRNGKey(21), 500, 200)
    run = jax.jit(jax.vmap(lambda p: two_channel_controller(p, delta=0.01)))
    res = run(streams)
    rate = np.asarray(res.drift_rate)
    # Average over the back half (steady state).
    mean_rate = float(rate[:, 100:].mean())
    assert abs(mean_rate) < 0.05, f"H0 drift-rate drifted from 0: {mean_rate:.4f}"
    assert mean_rate <= 0.02, f"H0 drift-rate systematically positive: {mean_rate:.4f}"


def test_windowed_slope_also_tracks_drift():
    """The windowed OLS slope estimator agrees in sign/order with the EWMA on drift.

    A secondary estimator sanity check: on a strong-drift stream the trailing-window OLS
    slope of ``log M_t`` is clearly positive post-change and ~0 pre-change, like the EWMA.
    """
    p = _drift_streams(jax.random.PRNGKey(4), 1, 120, 120, ceil=0.05)[0]
    raw = run_martingale(p, delta=0.01, reset_on_alarm=False)
    slope_w = np.asarray(windowed_logwealth_slope(raw.log_m, window=20))
    slope_e = np.asarray(ewma_logwealth_slope(raw.log_m, beta=0.1))
    # Pre-change near zero; post-change clearly positive, for both estimators.
    assert abs(slope_w[60]) < 0.2 and abs(slope_e[60]) < 0.2
    assert slope_w[-1] > 0.3 and slope_e[-1] > 0.3


# ============================================================ 3. The separation property
def test_separation_adaptation_does_not_change_alarm():
    """KEY: any Channel-B-driven adaptation leaves Channel A's alarm bit-identical.

    The whole point of the two-channel split: the adaptive channel may compute *any*
    control action, but it must not perturb the guaranteed alarm. We run the frozen
    alarm channel through ``run_with_adaptive_calibration`` with three wildly different
    adaptation functions (a no-op, an aggressive half-life schedule, and a nonsense
    function that returns huge values) and assert the returned alarm mask / first alarm
    are identical to each other and to the plain frozen martingale.
    """
    p = _drift_streams(jax.random.PRNGKey(5), 1, 100, 100, ceil=0.05)[0]

    adapt_fns = [
        lambda s: s,  # identity
        lambda s: drift_rate_to_half_life(s),  # the real knob
        lambda s: jnp.full_like(s, 1e9) * jnp.cumsum(s),  # pathological
    ]
    results = [run_with_adaptive_calibration(p, fn, delta=0.01) for fn in adapt_fns]
    baseline = run_martingale(p, delta=0.01, reset_on_alarm=True)

    for r in results:
        np.testing.assert_array_equal(np.asarray(r.alarms), np.asarray(baseline.alarms))
        assert int(r.first_alarm) == int(baseline.first_alarm)
        assert bool(r.crossed) == bool(baseline.crossed)
    # And all adaptations agree with each other (transitively, but assert explicitly).
    for r in results[1:]:
        np.testing.assert_array_equal(
            np.asarray(r.alarms), np.asarray(results[0].alarms)
        )


def test_channel_b_does_not_feed_channel_a_in_controller():
    """In the controller, Channel A's alarm is invariant to Channel B's knob parameters.

    Changing Channel B's knob mapping (``kappa``, half-life bounds) or its EWMA ``beta``
    must not change Channel A's alarm trace -- they live on different channels. We vary
    every Channel-B parameter and assert Channel A's ``alarms`` / ``first_alarm`` are
    unchanged.
    """
    p = _drift_streams(jax.random.PRNGKey(6), 1, 100, 100, ceil=0.05)[0]
    base = two_channel_controller(p, delta=0.01)
    varied = two_channel_controller(
        p, delta=0.01, beta=0.5, kappa=100.0, half_life_max=10.0, half_life_min=1.0
    )
    np.testing.assert_array_equal(np.asarray(base.alarms), np.asarray(varied.alarms))
    assert int(base.first_alarm) == int(varied.first_alarm)
    # But Channel B *did* change (sanity: the knob parameters had an effect somewhere).
    assert not np.allclose(np.asarray(base.half_life), np.asarray(varied.half_life))


# ====================================================== 4. Knob mapping + API contract
def test_drift_rate_to_half_life_is_monotone_decreasing():
    """The knob map ``rate -> half_life`` is non-increasing, clamped to [min, max].

    Higher drift rate => shorter (or equal) half-life, bounded by the configured min/max,
    and a zero/negative rate maps to the max half-life (no forgetting boost).
    """
    rates = jnp.linspace(-0.5, 3.0, 50)
    h = np.asarray(drift_rate_to_half_life(rates))
    # Non-increasing.
    assert np.all(np.diff(h) <= 1e-3), "half-life must be non-increasing in drift rate"
    # Clamped to bounds.
    assert h.max() <= 50_000.0 + 1e-3 and h.min() >= 1_000.0 - 1e-3
    # Zero / negative rate -> max half-life (no boost).
    assert abs(float(drift_rate_to_half_life(jnp.array(0.0))) - 50_000.0) < 1.0
    assert abs(float(drift_rate_to_half_life(jnp.array(-1.0))) - 50_000.0) < 1.0
    # Large rate -> floored at min.
    assert abs(float(drift_rate_to_half_life(jnp.array(10.0))) - 1_000.0) < 1.0


def test_staleness_weight_halves_at_half_life():
    """``staleness_weight(age=H, H) == 1/2`` and decays monotonically in age."""
    h = 1000.0
    assert abs(float(staleness_weight(jnp.array(0.0), h)) - 1.0) < 1e-6
    assert abs(float(staleness_weight(jnp.array(h), h)) - 0.5) < 1e-6
    assert abs(float(staleness_weight(jnp.array(2 * h), h)) - 0.25) < 1e-6
    ages = jnp.linspace(0, 5000, 50)
    w = np.asarray(staleness_weight(ages, h))
    assert np.all(np.diff(w) <= 0.0)


def test_controller_jit_and_vmap_shapes():
    """``two_channel_controller`` is jittable / vmap-able with the documented shapes."""
    streams = _uniform_streams(jax.random.PRNGKey(9), 8, 50)

    single = jax.jit(lambda p: two_channel_controller(p, delta=0.05))(streams[0])
    assert single.log_m.shape == (50,)
    assert single.alarms.shape == (50,)
    assert single.drift_rate.shape == (50,)
    assert single.half_life.shape == (50,)
    assert single.alarm.shape == () and single.first_alarm.shape == ()

    batched = jax.jit(jax.vmap(lambda p: two_channel_controller(p, delta=0.05)))(streams)
    assert batched.log_m.shape == (8, 50)
    assert batched.drift_rate.shape == (8, 50)
    assert batched.alarm.shape == (8,)
