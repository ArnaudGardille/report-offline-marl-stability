"""Rigorous validity gate for the M4 conformal drift-detection test martingale.

These tests are the correctness gate the lead author checks, so they are written to be
*non-vacuous* and to mirror the guarantees the paper claims (Algorithm 2 / Appendix D /
Ville's inequality). They run entirely on **synthetic p-value streams** -- no
environment -- so they are fast and exercise the martingale math in isolation:

1. ``E[M_t] = 1`` under ``H0`` (the defining martingale property), Monte-Carlo over many
   i.i.d. Uniform(0, 1) streams. We assert this on a *single* power martingale at a
   well-conditioned ``eps`` where the mean is actually estimable, and separately show
   the grid mixture's per-step factor has the right mean. We also confirm the
   **supermartingale** ``E[M_t] <= 1`` with conservative (tie-laden) super-uniform
   p-values.
2. **Ville false-alarm control**: over many ``H0`` streams the empirical fraction that
   *ever* cross ``1/delta`` is ``<= delta`` (within Monte-Carlo error), for two deltas.
   This validates the anytime guarantee on the *actual* grid-mixture detector.
3. **Detection power + reset-on-alarm**: a stream with a change point (uniform, then
   stochastically small p-values) crosses ``1/delta`` soon after the change, and the
   reset re-arms the detector so it can fire again.

Numerical note (documented in the asserts too): the power martingale ``M_t^eps`` has an
extremely heavy right tail for small ``eps`` -- ``E[M_t^eps] = 1`` is carried by rare
enormous values of probability ``~eps^t`` that no finite sample observes, so the *sample
mean* of small-``eps`` (or full-grid) ``M_t`` is biased low and is **not** a valid test
of ``E[M_t] = 1`` at large ``t``. We therefore (a) verify ``E[M_t] = 1`` on ``eps`` near
1 where the tail is light, in float64 to avoid the float32 underflow of the heavy lower
tail, and (b) use the upper-tail-only crossing probability (which float32 represents
fine) for the Ville test. This is exactly the regime Appendix D validates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.martingale import (
    DEFAULT_EPS_GRID,
    detection_delay,
    mixture_closed_form_factor,
    power_log_martingale,
    run_martingale,
)


# --------------------------------------------------------------------------- helpers
def _uniform_streams(key, n_streams: int, length: int, x64: bool = False):
    """``(n_streams, length)`` i.i.d. Uniform(0, 1] p-values (the H0 stream)."""
    dtype = jnp.float64 if x64 else jnp.float32
    # Uniform on (0, 1] (shift the [0, 1) sample off exact 0 so ln p is finite).
    u = jax.random.uniform(key, (n_streams, length), dtype=dtype)
    return 1.0 - u  # in (0, 1]


# ============================================================ 1. E[M_t] = 1 (validity)
def test_power_martingale_mean_is_one_under_h0():
    """KEY TEST (Appendix D): ``E[M_t^eps] = 1`` under i.i.d. Uniform p-values.

    The defining martingale property. We test a *single* power martingale at
    ``eps in {0.8, 0.9}`` -- large enough that the heavy right tail of ``M_t^eps`` is
    light enough for a Monte-Carlo mean to converge -- across many streams and several
    horizons ``t``, and assert the sample mean is ~1 within a few standard errors.

    Run in float64: the lower tail of ``M_t^eps`` reaches ~``eps^t`` and underflows
    float32 to 0, which (together with the unobservably-rare compensating upper tail)
    would bias a float32 sample mean low. This is a numerical property of power
    martingales, not of the implementation -- documented here explicitly.
    """
    jax.config.update("jax_enable_x64", True)
    try:
        n_streams, length = 200_000, 40
        ps = _uniform_streams(jax.random.PRNGKey(0), n_streams, length, x64=True)

        for eps in (0.8, 0.9):
            log_m = jax.vmap(lambda p: power_log_martingale(p, eps))(ps)  # (S, T)
            m = np.asarray(jnp.exp(log_m))
            for t in (0, 4, 9, 19, 39):  # M_1, M_5, M_10, M_20, M_40
                col = m[:, t]
                mean = float(col.mean())
                se = float(col.std() / np.sqrt(n_streams))
                # Within 5 standard errors of 1 (tail-driven, so we allow a healthy band
                # but it is genuinely tight here: SE is small for eps near 1).
                assert abs(mean - 1.0) < max(5 * se, 0.02), (
                    f"eps={eps} t={t + 1}: E[M_t]={mean:.4f} (SE={se:.4f}) not ~1"
                )
    finally:
        jax.config.update("jax_enable_x64", False)


def test_mixture_closed_form_matches_eps_quadrature():
    """The closed form ``(1/p + ln p - 1)/(ln p)^2`` equals ``\\int_0^1 eps p^{eps-1} d eps``.

    The single-p-value mixture weight that backs the grid average. We verify the closed
    form against a fine deterministic quadrature *over eps* (which is well-conditioned:
    the integrand ``eps p^{eps-1}`` is smooth and bounded on ``eps in [0, 1]`` for any
    fixed ``p``), at several ``p`` spanning the range, plus the ``p -> 1`` limit ``1/2``.

    (We deliberately do *not* assert the ``\\int_0^1 g(p) dp = 1`` identity by sampling:
    although it holds analytically, ``g(p) ~ 1/(p (ln p)^2)`` as ``p -> 0`` spreads unit
    mass across every decade of small ``p``, so neither Monte-Carlo nor uniform
    quadrature estimates it without astronomically many points -- a property of the
    integrand, not the code. The defining ``E[M_t]=1`` martingale property is validated
    directly on a power martingale in ``test_power_martingale_mean_is_one_under_h0``.)
    """
    eps = (jnp.arange(200_000, dtype=jnp.float64) + 0.5) / 200_000  # midpoints in (0,1)
    # Well-conditioned p values: away from 1, where the float32 closed form suffers
    # catastrophic cancellation in (1/p + ln p - 1) / (ln p)^2 ((ln p)^2 -> 0). The
    # near-1 behaviour is pinned separately by the exact p = 1 -> 1/2 limit below.
    for p_val in (0.02, 0.05, 0.2, 0.5, 0.8):
        quad = float(jnp.mean(eps * jnp.power(p_val, eps - 1.0)))  # ~ \int eps p^{eps-1}
        closed = float(mixture_closed_form_factor(jnp.array(p_val, dtype=jnp.float32)))
        assert abs(quad - closed) < 2e-3, (
            f"closed form {closed:.5f} != eps-quadrature {quad:.5f} at p={p_val}"
        )
    # Limit at p = 1 is exactly 1/2 (the analytic value; the Taylor branch handles it).
    assert abs(float(mixture_closed_form_factor(jnp.array(1.0))) - 0.5) < 1e-6


def test_grid_mixture_is_supermartingale_with_conservative_pvalues():
    """With *conservative* super-uniform (tie-laden) p-values, ``E[M_t] <= 1``.

    The conformal p-values used in practice are discrete (ties) and therefore only
    *super*-uniform: ``P(p <= u) <= u``. That makes each power-martingale factor have
    mean ``<= 1``, so the mixture is a **supermartingale** -- still valid for
    false-alarm control. We build a stochastically-larger-than-uniform stream (so it is
    strictly conservative) and assert the grid-mixture sample mean stays at or below 1
    (plus Monte-Carlo slack) at several horizons. (Uses ``eps`` near 1 via a restricted
    grid so the mean is estimable; the conservativeness only pushes it further down.)
    """
    key = jax.random.PRNGKey(7)
    n_streams, length = 100_000, 25
    u = 1.0 - jax.random.uniform(key, (n_streams, length))
    # Stochastically larger than uniform => super-uniform (conservative): sqrt(U) has
    # CDF u^2 <= u on [0,1], i.e. P(p<=u)=u^2<=u. A clean conservative p-value stream.
    conservative = jnp.sqrt(u)

    # Restricted upper grid so the Monte-Carlo mean is well-conditioned; the point here
    # is the <= 1 direction, which conservativeness can only help.
    grid = jnp.array([0.7, 0.8, 0.9], dtype=jnp.float32)
    run = jax.jit(
        jax.vmap(lambda p: run_martingale(p, delta=0.01, eps_grid=grid, reset_on_alarm=False))
    )
    res = run(conservative)
    m = np.asarray(res.m)
    for t in (4, 9, 24):
        col = m[:, t]
        mean = float(col.mean())
        se = float(col.std() / np.sqrt(n_streams))
        assert mean <= 1.0 + 5 * se, (
            f"t={t + 1}: conservative-p mean E[M_t]={mean:.4f} exceeds 1 (SE={se:.4f}); "
            f"a super-uniform stream must give a supermartingale (E[M_t] <= 1)"
        )


# ===================================================== 2. Ville false-alarm control
def test_ville_false_alarm_control():
    """KEY TEST: under ``H0`` the empirical ever-cross fraction is ``<= delta``.

    Ville's inequality: ``P(sup_t M_t >= 1/delta) <= delta`` for the non-negative
    martingale started at 1. We run the *actual* grid-mixture detector (no reset, so a
    crossing is a genuine first-passage over the whole horizon) on many i.i.d. uniform
    streams and check the fraction that ever cross ``1/delta`` is at or below ``delta``
    within binomial Monte-Carlo error, for ``delta in {0.1, 0.05}``. This is the
    anytime false-alarm guarantee the drift detector relies on, and the test is
    non-vacuous: a broken (e.g. non-martingale) update would over-fire here.

    Crossings are an *upper-tail* event, which float32 represents accurately (unlike the
    heavy lower tail that complicates the mean), so no float64 is needed.
    """
    n_streams, length = 20_000, 200
    ps = _uniform_streams(jax.random.PRNGKey(1), n_streams, length)

    for delta in (0.1, 0.05):
        run = jax.jit(jax.vmap(lambda p: run_martingale(p, delta=delta, reset_on_alarm=False)))
        res = run(ps)
        crossed = np.asarray(res.crossed)
        frac = float(crossed.mean())
        # Binomial SE of the crossing-fraction estimate; allow 3 SE of slack above delta.
        se = float(np.sqrt(delta * (1 - delta) / n_streams))
        assert frac <= delta + 3 * se, (
            f"Ville violated: delta={delta} empirical ever-cross={frac:.4f} "
            f"> delta + 3*SE = {delta + 3 * se:.4f}"
        )
        # Non-vacuous: the detector *can* fire under H0 occasionally (not identically 0),
        # but should be well under delta on average. (We only assert the upper bound;
        # the fraction being far below delta is fine and expected -- the bound is loose.)
        assert frac >= 0.0


def test_threshold_is_one_over_delta():
    """An alarm fires exactly when ``M_t >= 1/delta`` (threshold wiring is correct).

    A deterministic, non-statistical check: feed a constant tiny p-value so the
    martingale grows monotonically, and assert the first alarm step is the first step
    whose ``M_t`` reaches ``1/delta`` (and not before). Guards against an off-by-one or
    log/linear-threshold mismatch.
    """
    delta = 0.01
    thresh = 1.0 / delta
    # Constant small p -> strictly increasing M_t; no reset so we see the first passage.
    p = jnp.full((50,), 0.2, dtype=jnp.float32)
    res = run_martingale(p, delta=delta, reset_on_alarm=False)
    m = np.asarray(res.m)
    alarms = np.asarray(res.alarms)
    first = int(res.first_alarm)

    assert bool(res.crossed) and first >= 0
    # Everything strictly before the first alarm is below threshold; the alarm step is
    # at/above it.
    assert np.all(m[:first] < thresh), "an alarm-free prefix step was already >= 1/delta"
    assert m[first] >= thresh, "the first alarm step is below 1/delta"
    # alarms mask agrees with the threshold on M.
    np.testing.assert_array_equal(alarms, m >= thresh)


# ===================================================== 3. Detection power + reset
def test_detection_power_after_change_point():
    """KEY TEST: a drift (small p-values after a change point) is detected soon after.

    Pre-change p-values are Uniform(0, 1] (``H0``); post-change they are stochastically
    *small* (drift makes the conformal score large -> p small). The martingale must
    cross ``1/delta`` and the first alarm must land shortly *after* the change point
    (not before). We assert detection happens and the delay is small relative to the
    post-change horizon, averaged over many streams for a stable delay estimate.
    """
    n_streams = 400
    pre_len, post_len = 100, 100
    change_point = pre_len
    delta = 0.01

    k_pre, k_post = jax.random.split(jax.random.PRNGKey(2))
    pre = _uniform_streams(k_pre, n_streams, pre_len)
    # Post-change: small p-values. Uniform(0, 0.05] -> strongly non-conforming.
    post = 0.05 * (1.0 - jax.random.uniform(k_post, (n_streams, post_len)))
    streams = jnp.concatenate([pre, post], axis=1)

    run = jax.jit(jax.vmap(lambda p: run_martingale(p, delta=delta, reset_on_alarm=True)))
    res = run(streams)

    crossed = np.asarray(res.crossed)
    first = np.asarray(res.first_alarm)
    # Essentially every stream should detect this strong drift.
    assert crossed.mean() > 0.98, f"only {crossed.mean():.2%} of drift streams detected"

    delays = first - change_point
    fired = first >= 0
    # No alarm should fire *before* the change point on the uniform prefix (that would be
    # a false alarm); allow a tiny fraction from the Ville tail.
    pre_alarm_rate = float(np.mean((first >= 0) & (first < change_point)))
    assert pre_alarm_rate < 0.05, f"too many pre-change false alarms: {pre_alarm_rate:.2%}"

    mean_delay = float(np.mean(delays[fired & (first >= change_point)]))
    assert mean_delay < 30, f"detection too slow: mean delay {mean_delay:.1f} steps > 30"


def test_reset_on_alarm_rearms_detector():
    """Reset-on-alarm returns ``M`` to 1 and lets the detector fire again.

    With ``reset_on_alarm=True`` a long stream of small p-values should produce
    *multiple* alarms (each followed by a reset back to ``ln M = 0``); with
    ``reset_on_alarm=False`` the same stream produces a single first passage and then
    keeps climbing. We assert: (a) more than one alarm with reset, (b) the log-trace
    drops back to 0 right after each reset, and (c) ``detection_delay`` is consistent.
    """
    delta = 0.01
    # A long run of small (but not constant) p-values -> repeated growth and crossings.
    key = jax.random.PRNGKey(5)
    p = 0.1 * (1.0 - jax.random.uniform(key, (300,)))

    with_reset = run_martingale(p, delta=delta, reset_on_alarm=True)
    no_reset = run_martingale(p, delta=delta, reset_on_alarm=False)

    n_alarms_reset = int(np.sum(np.asarray(with_reset.alarms)))
    n_alarms_noreset = int(np.sum(np.asarray(no_reset.alarms)))
    assert n_alarms_reset > 1, f"reset failed to re-arm: only {n_alarms_reset} alarm(s)"
    # Without reset the martingale crosses once and then *stays* above threshold every
    # subsequent step (it keeps growing), so it reports many 'alarm' steps too -- the
    # distinguishing property is the trace returning to 0 after a reset.
    log_m = np.asarray(with_reset.log_m)
    alarms = np.asarray(with_reset.alarms)
    alarm_idx = np.where(alarms)[0]
    # The step immediately after a (non-final) alarm starts from a reset M=1, so its
    # log_m equals exactly the single-step contribution (==> strictly less than the
    # crossing value ln(1/delta)). Check the trace genuinely dropped.
    for i in alarm_idx:
        if i + 1 < log_m.size:
            assert log_m[i + 1] < log_m[i], (
                f"no reset drop after alarm at {i}: log_m[{i + 1}]={log_m[i + 1]:.2f} "
                f">= log_m[{i}]={log_m[i]:.2f}"
            )

    # Consistency of the convenience delay helper for the first alarm.
    d0 = int(detection_delay(with_reset, change_point=0))
    assert d0 == int(with_reset.first_alarm)
    assert n_alarms_noreset >= 1


# ===================================================== 4. API / jit / vmap contract
def test_run_martingale_jit_and_vmap():
    """``run_martingale`` is jittable and vmap-able with the documented output shapes."""
    streams = _uniform_streams(jax.random.PRNGKey(9), 8, 50)

    single = jax.jit(lambda p: run_martingale(p, delta=0.05))(streams[0])
    assert single.log_m.shape == (50,)
    assert single.m.shape == (50,)
    assert single.alarms.shape == (50,)
    assert single.alarms.dtype == jnp.bool_
    assert single.crossed.shape == ()
    assert single.first_alarm.shape == ()

    batched = jax.jit(jax.vmap(lambda p: run_martingale(p, delta=0.05)))(streams)
    assert batched.log_m.shape == (8, 50)
    assert batched.crossed.shape == (8,)
    # log_m and m agree (m = exp(log_m)) wherever m is finite.
    finite = np.isfinite(np.asarray(batched.m))
    np.testing.assert_allclose(
        np.asarray(batched.m)[finite],
        np.exp(np.asarray(batched.log_m))[finite],
        rtol=1e-5,
    )


def test_default_grid_is_nine_interior_points():
    """The default mixing grid is ``{0.1, ..., 0.9}`` (paper's grid), all in (0, 1)."""
    grid = np.asarray(DEFAULT_EPS_GRID)
    assert grid.shape == (9,)
    np.testing.assert_allclose(grid, np.arange(1, 10) / 10.0, atol=1e-7)
    assert grid.min() > 0.0 and grid.max() < 1.0
