"""Rigorous validity gate for the M6' certified-replay constructions.

Two finite-sample guarantees, each tested on a **synthetic, controlled covariate
shift** (no environment -- fast, and the math is exercised in isolation):

1. **ρ-weighted conformal restores validity under known covariate shift**
   (Tibshirani et al. 2019). Calibration is drawn from a source law ``pi_past`` and the
   test point from a *shifted* target law ``pi_cur`` with the exact, known ratio ``rho``.
   We assert the **ρ-weighted** p-values are super-uniform *w.r.t. the shifted law*
   (mean ~ 0.5, ``P(p<=a) ~ a``, KS below its critical value -- the clean continuous
   case is exactly uniform), while the **unweighted** inductive p-values on the *same*
   data are demonstrably miscalibrated (mean p far below 0.5, ``P(p<=a) >> a``). This is
   the "missing middle": the exact-ratio score's first finite-sample validity statement.

2. **NexCP time-decay keeps coverage near nominal under drift** and respects the
   coverage-gap **bound** (Barber, Candes, Ramdas & Tibshirani 2023). On a single-change
   -point stream (pre-change calibration from ``P0``, recent calibration + test from
   ``P1``) with the *exact* ``d_TV(P0, P1)``, ordinary exchangeable conformal
   (uniform weights) badly *under*-covers, while time-decay weights restore coverage
   toward ``1 - alpha``. We assert (a) decay's coverage is much closer to nominal than
   uniform's, and (b) the NexCP one-sided guarantee ``miscoverage <= alpha + bound``
   holds, with the bound non-vacuous and shrinking as the decay down-weights stale data.

Honesty notes baked into the asserts:
* The NexCP guarantee bounds **under-coverage** (miscoverage), so an *over*-conservative
  predictor (coverage > 1 - alpha from aggressive decay) does NOT violate it -- the test
  checks the correct one-sided inequality, not an absolute gap.
* Both constructions reduce to the plain inductive p-value when weights are uniform; a
  regression test pins that exact equivalence so the generalisation is faithful.
"""

from __future__ import annotations

from math import erf, sqrt

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.conformal import conformal_p_values
from conformal_marl.weighted_conformal import (
    empirical_coverage,
    importance_p_values,
    nexcp_coverage_bound,
    nexcp_p_values,
    time_decay_weights,
    weighted_conformal_p_values,
)


# --------------------------------------------------------------------------- helpers
def _tv_gaussian_unit(shift: float) -> float:
    """Exact total variation ``d_TV(N(0,1), N(shift,1)) = 2*Phi(|shift|/2) - 1``."""
    phi = 0.5 * (1.0 + erf((abs(shift) / 2.0) / sqrt(2.0)))
    return 2.0 * phi - 1.0


def _gaussian_ratio(x, mu: float):
    """Known likelihood ratio ``N(mu,1)/N(0,1) = exp(mu*x - mu^2/2)`` (target/source)."""
    return jnp.exp(mu * x - 0.5 * mu**2)


# ============================================ 1. ρ-weighted validity under covariate shift
def test_weighted_conformal_superuniform_under_shift():
    """KEY TEST: ρ-weighted p-values are uniform w.r.t. the shifted (current) law.

    Calibration ``~ N(0,1)`` (``pi_past``), test ``~ N(mu,1)`` (``pi_cur``), score
    ``A = X`` (continuous -> tie-free, so the conformal p-value is *exactly* uniform
    when valid), weight ``w = rho = pi_cur/pi_past``. We pool many trials and assert the
    weighted p-values look Uniform(0,1): mean ~ 0.5, ``P(p<=a) ~ a`` at several alphas,
    and KS vs Uniform below its 0.05 critical value. This is the validity Tibshirani et
    al. (2019) prove and the exact-ratio replay score inherits.
    """
    mu = 1.0
    n_cal, trials = 500, 5000
    key = jax.random.PRNGKey(0)
    ps = []
    for _ in range(trials):
        key, k1, k2 = jax.random.split(key, 3)
        cal = jax.random.normal(k1, (n_cal,))  # source N(0,1)
        test = mu + jax.random.normal(k2, (1,))  # target N(mu,1)
        w_cal = _gaussian_ratio(cal, mu)
        w_test = _gaussian_ratio(test, mu)
        p = weighted_conformal_p_values(test, cal, w_cal, w_test)
        ps.append(float(p[0]))
    ps = np.array(ps)
    n = ps.size

    assert ps.min() > 0.0 and ps.max() <= 1.0 + 1e-6
    assert abs(ps.mean() - 0.5) < 0.02, f"weighted mean p={ps.mean():.4f} (expected ~0.5)"

    xs = np.sort(ps)
    ecdf = np.arange(1, n + 1) / n
    ks = float(np.max(np.abs(ecdf - xs)))
    ks_crit = 1.36 / np.sqrt(n)
    assert ks < ks_crit, f"weighted KS {ks:.4f} >= crit {ks_crit:.4f}: not uniform under shift"

    for a in (0.05, 0.1, 0.25, 0.5, 0.75):
        emp = float(np.mean(ps <= a))
        assert abs(emp - a) < 0.03, f"weighted P(p<={a})={emp:.3f} deviates from uniform"


def test_unweighted_pvalues_miscalibrated_under_same_shift():
    """COMPANION: the *unweighted* inductive p-value is miscalibrated under the shift.

    The point of weighting: on the *same* shifted data the plain inductive p-value
    (which assumes test ~ calibration) is invalid. The test point is stochastically
    larger than calibration (positive shift -> larger scores -> systematically small
    p-values), so ``P(p<=a) >> a`` and mean p is pulled well below 0.5. We assert the
    *failure* explicitly so the weighting's restoration of validity is non-vacuous --
    the two tests together are the "weighted fixes what unweighted breaks" statement.
    """
    mu = 1.0
    n_cal, trials = 500, 5000
    key = jax.random.PRNGKey(1)
    ps = []
    for _ in range(trials):
        key, k1, k2 = jax.random.split(key, 3)
        cal = jax.random.normal(k1, (n_cal,))
        test = mu + jax.random.normal(k2, (1,))
        p = importance_p_values(test, cal)  # NO weighting
        ps.append(float(p[0]))
    ps = np.array(ps)

    # Miscalibrated low: mean well under 0.5.
    assert ps.mean() < 0.40, (
        f"unweighted mean p={ps.mean():.4f}; expected clear downward bias under shift "
        f"(if this is ~0.5 the shift/score is too weak to be a meaningful test)"
    )
    # Tail badly inflated: small-p mass far exceeds the nominal level.
    emp05 = float(np.mean(ps <= 0.05))
    assert emp05 > 0.12, (
        f"unweighted P(p<=0.05)={emp05:.3f}; a valid p-value would give ~0.05, so this "
        f"must be visibly inflated to demonstrate the miscalibration"
    )


def test_weighted_equals_unweighted_when_weights_uniform():
    """Regression: uniform weights reproduce the plain inductive p-value exactly.

    With ``w_i = w* = 1`` every ``w~ = 1/(n+1)`` and the weighted formula collapses to
    ``(1 + #{A_i >= A*}) / (n + 1)`` -- so ρ-weighted conformal is a faithful strict
    generalisation. Checked against both the local ``importance_p_values`` and the
    project's agent-axis ``conformal_p_values`` (single agent) to rule out an off-by-one
    in either rank convention.
    """
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    cal = jax.random.normal(k1, (137,))
    test = jax.random.normal(k2, (40,))
    ones_cal = jnp.ones_like(cal)
    ones_test = jnp.ones_like(test)

    p_weighted = weighted_conformal_p_values(test, cal, ones_cal, ones_test)
    p_plain = importance_p_values(test, cal)
    np.testing.assert_allclose(np.asarray(p_weighted), np.asarray(p_plain), atol=1e-6)

    # And against conformal_p_values (agent axis = 1): shapes (B,1)/(n,1).
    p_agent = conformal_p_values(test[:, None], cal[:, None])[:, 0]
    np.testing.assert_allclose(np.asarray(p_weighted), np.asarray(p_agent), atol=1e-6)


def test_weighted_pvalues_in_unit_interval_and_monotone():
    """``p in (0, 1]`` always, and decreasing in the test score (rank monotonicity).

    Sanity invariants independent of the law: a larger test score is at least as
    non-conforming, so it can only have *fewer* calibration scores at/above it -> a
    smaller (or equal) p. We feed a sorted sweep of test scores against a fixed weighted
    calibration set and assert the p-values are non-increasing and in range.
    """
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    cal = jax.random.normal(k1, (200,))
    w_cal = jnp.exp(0.5 * jax.random.normal(k2, (200,)))  # arbitrary positive weights
    test = jnp.linspace(-4.0, 4.0, 50)  # increasing scores
    w_test = jnp.ones_like(test)

    p = np.asarray(weighted_conformal_p_values(test, cal, w_cal, w_test))
    assert p.min() > 0.0 and p.max() <= 1.0 + 1e-6
    # Non-increasing in the (increasing) test score.
    assert np.all(np.diff(p) <= 1e-6), "weighted p-values must be monotone in the score"


# ============================================ 2. NexCP time-decay coverage under drift
def test_nexcp_time_decay_weights_shape_and_orientation():
    """``time_decay_weights`` are positive, recent-heavy, and uniform at ``rho=1``.

    Locks the orientation the coverage math depends on: index ``0`` is oldest and gets
    the *smallest* weight ``rho^n``; index ``n-1`` is newest and gets the *largest*
    ``rho^1``. At ``rho_decay = 1`` all weights equal 1 (ordinary exchangeable conformal).
    """
    n = 10
    w_uniform = np.asarray(time_decay_weights(n, 1.0))
    np.testing.assert_allclose(w_uniform, np.ones(n), atol=1e-6)

    rho = 0.8
    w = np.asarray(time_decay_weights(n, rho))
    assert np.all(w > 0)
    # Strictly increasing oldest -> newest.
    assert np.all(np.diff(w) > 0), "recent points must carry more weight"
    # Exact geometric values: w[i] = rho^{n-i}.
    expected = rho ** (n - np.arange(n))
    np.testing.assert_allclose(w, expected, rtol=1e-5)


def test_nexcp_restores_coverage_and_respects_bound_under_drift():
    """KEY TEST: under drift, time-decay coverage beats uniform and obeys the NexCP bound.

    Single change point: the first ``n - recent`` calibration points are drawn from
    ``P0 = N(0,1)``, the last ``recent`` and the test point from ``P1 = N(shift,1)``
    (the *current* law). The per-point TV drift is the **exact** ``d_TV(P0,P1)`` for the
    pre-change points and ``0`` for the post-change points.

    We assert, over many trials at level ``alpha``:

    * **Uniform** conformal (``rho_decay = 1``) badly *under*-covers (the stale ``P0``
      points inflate the rank -> p-values too large -> the test is wrongly excluded);
    * **Time-decay** (``rho_decay = 0.9``) restores coverage close to ``1 - alpha`` and
      strictly better than uniform;
    * the **NexCP one-sided guarantee** holds for both: ``miscoverage = P(p<=alpha)
      <= alpha + bound`` where ``bound = sum_i w~_i TV_i``; and the bound is *non-vacuous*
      (uniform's bound is well below 1 yet its miscoverage saturates it, decay's bound is
      much smaller). Over-coverage from decay is allowed and not flagged -- the assertion
      is the correct one-sided inequality, never an absolute two-sided gap.
    """
    alpha = 0.2
    n_cal, recent, shift = 200, 40, 2.0
    trials = 6000
    tv = _tv_gaussian_unit(shift)
    # per-point TV (oldest first): pre-change = tv, post-change = 0.
    tv_vec = jnp.concatenate(
        [jnp.full((n_cal - recent,), tv), jnp.zeros((recent,))]
    )

    def coverage_for(rho_decay: float, seed: int) -> float:
        key = jax.random.PRNGKey(seed)
        hits = []
        for _ in range(trials):
            key, k1, k2 = jax.random.split(key, 3)
            pre = jax.random.normal(k1, (n_cal - recent,))  # P0
            post = shift + jax.random.normal(jax.random.fold_in(k1, 1), (recent,))  # P1
            cal = jnp.concatenate([pre, post])  # chronological, oldest first
            test = shift + jax.random.normal(k2, (1,))  # P1 (current)
            p = nexcp_p_values(test, cal, rho_decay)
            hits.append(float(p[0] > alpha))  # covered iff p > alpha
        return float(np.mean(hits))

    cov_uniform = coverage_for(1.0, seed=10)
    cov_decay = coverage_for(0.9, seed=11)

    bound_uniform = float(nexcp_coverage_bound(1.0, n_cal, tv_vec))
    bound_decay = float(nexcp_coverage_bound(0.9, n_cal, tv_vec))

    nominal = 1.0 - alpha

    # (a) Uniform under-covers badly; decay is much closer to nominal and clearly better.
    assert cov_uniform < nominal - 0.2, (
        f"uniform coverage {cov_uniform:.3f} should be far below nominal {nominal:.2f} "
        f"under this drift (otherwise the drift is too mild to need NexCP)"
    )
    assert cov_decay > cov_uniform + 0.2, (
        f"time-decay coverage {cov_decay:.3f} not meaningfully better than uniform "
        f"{cov_uniform:.3f}"
    )
    assert abs(cov_decay - nominal) < 0.1, (
        f"time-decay coverage {cov_decay:.3f} should be within 0.1 of nominal {nominal:.2f}"
    )

    # (b) NexCP one-sided guarantee: miscoverage <= alpha + bound (allow small MC slack).
    mc_slack = 0.02
    miscov_uniform = 1.0 - cov_uniform
    miscov_decay = 1.0 - cov_decay
    assert miscov_uniform <= alpha + bound_uniform + mc_slack, (
        f"NexCP bound violated (uniform): miscoverage {miscov_uniform:.3f} > "
        f"alpha+bound = {alpha + bound_uniform:.3f}"
    )
    assert miscov_decay <= alpha + bound_decay + mc_slack, (
        f"NexCP bound violated (decay): miscoverage {miscov_decay:.3f} > "
        f"alpha+bound = {alpha + bound_decay:.3f}"
    )

    # (c) Bound is non-vacuous and shrinks with decay (the formal reason decay helps).
    assert bound_uniform < 1.0, "a coverage-gap bound >= 1 would be vacuous"
    assert bound_decay < bound_uniform, (
        f"decay should shrink the bound: {bound_decay:.4f} !< {bound_uniform:.4f}"
    )
    # Uniform's bound is genuinely informative here: it is (nearly) saturated by the
    # observed miscoverage, i.e. not loose by an order of magnitude.
    assert miscov_uniform >= alpha + 0.5 * bound_uniform, (
        f"uniform bound looks vacuous: miscoverage {miscov_uniform:.3f} far below "
        f"alpha + bound = {alpha + bound_uniform:.3f}"
    )


def test_nexcp_bound_to_zero_recovers_exact_coverage():
    """As drift -> 0 the NexCP bound -> 0 and coverage -> nominal (sanity / continuity).

    With no shift the stream is exchangeable, every ``TV_i = 0``, so the bound is exactly
    0 and time-decay conformal recovers the ordinary ``1 - alpha`` coverage guarantee. A
    short Monte-Carlo confirms coverage is at/above nominal (valid, mildly conservative
    from discreteness) and the bound is identically zero.
    """
    alpha = 0.2
    n_cal, trials = 150, 4000
    tv_vec = jnp.zeros((n_cal,))  # no drift -> all TV are 0
    assert float(nexcp_coverage_bound(0.9, n_cal, tv_vec)) == 0.0

    key = jax.random.PRNGKey(20)
    hits = []
    for _ in range(trials):
        key, k1, k2 = jax.random.split(key, 3)
        cal = jax.random.normal(k1, (n_cal,))  # single law, exchangeable
        test = jax.random.normal(k2, (1,))
        p = nexcp_p_values(test, cal, 0.9)
        hits.append(float(p[0] > alpha))
    cov = float(np.mean(hits))
    # Valid (>= 1 - alpha) within MC slack; conformal is conservative so over-coverage ok.
    assert cov >= (1.0 - alpha) - 0.03, (
        f"no-drift coverage {cov:.3f} fell below nominal {1 - alpha:.2f}: time-decay "
        f"conformal must stay valid under exchangeability"
    )


def test_empirical_coverage_helper():
    """``empirical_coverage`` is just ``mean(p > alpha)`` (wiring check)."""
    p = jnp.array([0.01, 0.2, 0.5, 0.9, 0.05])
    # p > 0.1 -> [F, T, T, T, F] -> 3/5.
    assert abs(float(empirical_coverage(p, 0.1)) - 0.6) < 1e-6


def test_weighted_conformal_jit_and_vmap():
    """The weighted/NexCP p-values are jittable and vmap-able with stable shapes."""
    cal = jax.random.normal(jax.random.PRNGKey(0), (64,))
    w_cal = jnp.exp(0.1 * cal)
    test = jax.random.normal(jax.random.PRNGKey(1), (16,))
    w_test = jnp.ones_like(test)

    jit_w = jax.jit(weighted_conformal_p_values)
    p = jit_w(test, cal, w_cal, w_test)
    assert p.shape == (16,)

    # vmap over a batch of independent (test, cal) problems.
    tests = jax.random.normal(jax.random.PRNGKey(2), (5, 16))
    cals = jax.random.normal(jax.random.PRNGKey(3), (5, 64))
    wcs = jnp.exp(0.1 * cals)
    wts = jnp.ones_like(tests)
    batched = jax.vmap(weighted_conformal_p_values)(tests, cals, wcs, wts)
    assert batched.shape == (5, 16)

    jit_nex = jax.jit(lambda t, c: nexcp_p_values(t, c, 0.9))
    pn = jit_nex(test, cal)
    assert pn.shape == (16,) and float(pn.min()) > 0.0 and float(pn.max()) <= 1.0 + 1e-6
