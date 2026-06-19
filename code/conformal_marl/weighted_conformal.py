"""Certified replay -- weighted & non-exchangeable conformal p-values (M6').

This module supplies the **finite-sample validity machinery** that the exact-ratio
replay score was missing. M3 gave two non-conformity scores -- the ratio-free
``A^policy`` and the exact ``A^exact = -log rho`` -- and ranked them with the *plain*
inductive conformal p-value :func:`conformal_marl.conformal.conformal_p_values`. That
p-value is only valid when the scored transition is **exchangeable** with the
calibration set. Replay breaks exactly that assumption: a stored transition's
other-agent actions ``u_{-a}`` were drawn under the *past* policy ``pi_past`` but are
being judged under the *current* policy ``pi_cur`` -- a **covariate shift on**
``u_{-a}`` with the *known* likelihood ratio ``rho = pi_cur / pi_past``.

Two certified constructions handle the two failure modes:

1. :func:`weighted_conformal_p_values` -- **weighted conformal under known covariate
   shift** (Tibshirani, Foygel Barber, Candes & Ramdas, NeurIPS 2019). Reweighting
   the calibration empirical measure by ``w_i propto rho(x_i)`` (and the test point by
   ``rho(x*)``) restores validity *with respect to the shifted (current) law*. This is
   the **missing middle**: it gives the exact-rho score its first finite-sample
   guarantee -- between the ratio-free heuristic (valid but for a different, shifted
   target) and raw importance sampling (unbiased mean but no coverage statement).

2. :func:`nexcp_p_values` / :func:`nexcp_coverage` -- **non-exchangeable conformal**
   (Barber, Candes, Ramdas & Tibshirani, Annals of Statistics 2023) with fixed
   **time-decay** weights ``w_i = rho_decay^{(t - i)}``. This is a *certified* version
   of Foerster et al.'s heuristic time-discount of stale replay data: recent
   calibration points count more, and the coverage gap is **bounded** by a weighted
   total-variation term ``sum_i w_i * ||R_i - R_{n+1}||_TV`` that shrinks to 0 as the
   stream stops drifting (so under exchangeability it reduces to the usual guarantee).

Everything is a pure, batched, ``jax.jit`` / ``jax.vmap``-friendly function over a
leading axis, mirroring :mod:`conformal_marl.conformal`. Small score == more
non-conforming == staler (same orientation as the rest of the codebase), so a small
p-value flags a transition the present policy is unlikely to have produced.

Notation
--------
``A_i`` : calibration non-conformity scores (``i = 1..n``).
``A*``  : test non-conformity score (one or a batch).
``rho`` : known likelihood ratio ``pi_cur / pi_past`` evaluated at the covariate; the
          calibration weight ``w_i = rho_i`` and test weight ``w* = rho*`` (Tibshirani).
The conservative ``>=`` rank convention and the ``+w*`` test-point inclusion are kept
throughout, so every returned p-value lies in ``(0, 1]`` and is super-uniform under
the stated law (exactly uniform in the continuous, weight-tie-free case).
"""

from __future__ import annotations

import chex
import jax.numpy as jnp

# Floor for normalising denominators / weights, so divisions stay finite even if a
# whole weight vector underflows (defensive; real rho / decay weights are positive).
_W_EPS = 1e-12


def weighted_conformal_p_values(
    test_scores_B: chex.Array,
    cal_scores_n: chex.Array,
    cal_weights_n: chex.Array,
    test_weights_B: chex.Array,
) -> chex.Array:
    r"""ρ-weighted conformal p-values (Tibshirani et al., 2019), batched over tests.

    For each test point ``b`` with score ``A* = test_scores_B[b]`` and weight
    ``w* = test_weights_B[b]``, calibration scores ``A_1..A_n`` and weights
    ``w_1..w_n`` (here ``w_i = rho_i = pi_cur(x_i)/pi_past(x_i)``)::

        w~_i = w_i / (sum_{j=1..n} w_j + w*)        (normalised calibration weights)
        w~*  = w*  / (sum_{j=1..n} w_j + w*)        (normalised test-point weight)
        p    = sum_{i=1..n} w~_i * 1{A_i >= A*}  +  w~*

    The ``+ w~*`` includes the test point in its own rank (the standard conformal
    self-inclusion), which both guarantees ``p in (0, 1]`` and is what makes the
    *weighted* empirical distribution exchangeable-after-reweighting, hence valid.

    Validity (Theorem 2 of Tibshirani et al. 2019): if calibration ``~ pi_past`` and
    the test point ``~ pi_cur`` with ``pi_cur << pi_past`` and known ratio ``rho``,
    then ``p`` is **super-uniform w.r.t. the shifted (current) law** -- exactly the
    guarantee the exact-ratio replay score needs but does not get from the unweighted
    inductive p-value (which assumes test and calibration share a law).

    Parameters
    ----------
    test_scores_B:
        ``(B,)`` test non-conformity scores ``A*``.
    cal_scores_n:
        ``(n,)`` calibration non-conformity scores ``A_i``.
    cal_weights_n:
        ``(n,)`` non-negative calibration weights ``w_i`` (unnormalised; the function
        normalises). For covariate-shift replay, ``w_i = rho_i``.
    test_weights_B:
        ``(B,)`` non-negative test weights ``w*`` (for replay, ``rho*``). Each test
        point's own weight enters its denominator, so this is per-test, not shared.

    Returns
    -------
    ``(B,)`` weighted conformal p-values in ``(0, 1]``.

    Notes
    -----
    Recovers the *unweighted* inductive p-value ``(1 + #{A_i >= A*})/(n + 1)`` exactly
    when all weights are equal (every ``w~`` becomes ``1/(n+1)``), so this is a strict
    generalisation of :func:`conformal_marl.conformal.conformal_p_values`.
    """
    cal_scores_n = jnp.asarray(cal_scores_n)
    cal_weights_n = jnp.asarray(cal_weights_n)
    test_scores_B = jnp.asarray(test_scores_B)
    test_weights_B = jnp.asarray(test_weights_B)

    sum_w_cal = jnp.sum(cal_weights_n)  # scalar: sum_j w_j over calibration
    # Per-test denominator includes that test point's own weight w*.
    denom_B = sum_w_cal + test_weights_B + _W_EPS  # (B,)

    # 1{A_i >= A*} for every (test b, cal i): (B, n).
    ge_Bn = cal_scores_n[jnp.newaxis, :] >= test_scores_B[:, jnp.newaxis]
    # Weighted count of calibration scores at/above the test score: sum_i w_i 1{...}.
    weighted_ge_B = jnp.sum(cal_weights_n[jnp.newaxis, :] * ge_Bn, axis=1)  # (B,)

    # p = (sum_i w_i 1{A_i >= A*} + w*) / (sum_j w_j + w*).
    return (weighted_ge_B + test_weights_B) / denom_B


def importance_p_values(
    test_scores_B: chex.Array,
    cal_scores_n: chex.Array,
) -> chex.Array:
    """Plain (unweighted) inductive p-value for a 1-D score stream.

    ``p = (1 + #{ i : A_i >= A* }) / (n + 1)``. Identical in spirit to
    :func:`conformal_marl.conformal.conformal_p_values` but for *flat* ``(n,)`` /
    ``(B,)`` score arrays (no agent axis), so the weighted/unweighted comparison in the
    validity tests is apples-to-apples. Provided here (rather than reused) only to keep
    this module self-contained for the synthetic covariate-shift experiment; it is the
    ``cal_weights = test_weights = 1`` special case of
    :func:`weighted_conformal_p_values`.
    """
    cal_scores_n = jnp.asarray(cal_scores_n)
    test_scores_B = jnp.asarray(test_scores_B)
    n = cal_scores_n.shape[0]
    ge_Bn = cal_scores_n[jnp.newaxis, :] >= test_scores_B[:, jnp.newaxis]
    count_B = jnp.sum(ge_Bn, axis=1)
    return (1.0 + count_B.astype(jnp.float32)) / (n + 1.0)


def time_decay_weights(n: int, rho_decay: float) -> chex.Array:
    r"""NexCP fixed time-decay weights ``w_i = rho_decay^{(t - i)}`` for ``i = 1..n``.

    The calibration points are assumed in **chronological order** (index ``0`` oldest,
    index ``n-1`` newest) and the test point sits at time ``t = n`` (just after the most
    recent calibration point). With ``geom = rho_decay``::

        w_i = rho_decay^{(n - i)}   for i = 0..n-1   (so w_{n-1} = rho_decay^1, ... )

    i.e. the **most recent** calibration point gets the **largest** weight and older
    points decay geometrically. ``rho_decay in (0, 1]``; ``rho_decay = 1`` recovers
    uniform (ordinary exchangeable conformal). These are *fixed* (data-independent)
    weights -- the regime in which Barber et al. (2023, Thm 2a / Cor 1) give the clean
    coverage-gap bound used by :func:`nexcp_coverage`.

    Returns ``(n,)`` weights, **unnormalised** (callers that want a convex combination
    normalise; the coverage bound uses the normalised version including the test point).
    """
    # exponent (n - i) for i = 0..n-1  ->  [n, n-1, ..., 1].
    idx = jnp.arange(n, dtype=jnp.float32)
    exponent = n - idx  # (n,) : oldest gets n, newest gets 1
    return jnp.asarray(rho_decay, dtype=jnp.float32) ** exponent


def nexcp_p_values(
    test_scores_B: chex.Array,
    cal_scores_n: chex.Array,
    rho_decay: float,
) -> chex.Array:
    r"""Non-exchangeable conformal p-values with time-decay weights (Barber et al. 2023).

    A thin specialisation of :func:`weighted_conformal_p_values` whose weights are the
    **fixed time-decay** weights ``w_i = rho_decay^{(n - i)}`` (recent calibration
    weighted more), with the test point given weight ``w* = rho_decay^0 = 1`` (it sits
    at the current time, the reference). Concretely::

        w~_i = w_i / (sum_j w_j + 1),   w~* = 1 / (sum_j w_j + 1)
        p    = sum_i w~_i 1{A_i >= A*} + w~*

    This is the certified analogue of a heuristic "discount stale replay data" rule:
    the discount ``rho_decay`` is exactly the NexCP weight, and the resulting coverage
    gap is bounded (see :func:`nexcp_coverage`). Calibration scores must be in
    chronological order (oldest first).

    Returns ``(B,)`` p-values in ``(0, 1]``.
    """
    n = jnp.asarray(cal_scores_n).shape[0]
    w_n = time_decay_weights(n, rho_decay)  # (n,)
    test_w_B = jnp.ones_like(jnp.asarray(test_scores_B), dtype=jnp.float32)  # w* = 1
    return weighted_conformal_p_values(test_scores_B, cal_scores_n, w_n, test_w_B)


def nexcp_coverage_bound(rho_decay: float, n: int, tv_drift: chex.Array) -> chex.Array:
    r"""NexCP coverage-gap **upper bound** ``sum_i w~_i * TV_i`` (Barber et al. 2023).

    Barber, Candes, Ramdas & Tibshirani (2023, Thm 2 / Eq. for fixed weights) bound the
    coverage gap of weighted conformal with fixed weights by a **weighted sum of total-
    variation distances** between the law of the data and the law obtained by swapping
    the test point with calibration point ``i``::

        | P(coverage) - (1 - alpha) |  <=  sum_{i=1..n} w~_i * d_TV( Z , Z^{(i)} )

    where ``w~_i = w_i / (sum_j w_j + w*)`` are the normalised weights (test point
    ``w* = 1``) and ``d_TV(Z, Z^{(i)})`` measures how far point ``i`` is from being
    exchangeable with the test point -- i.e. how much the distribution **drifted**
    between time ``i`` and now. As drift -> 0 every ``TV_i -> 0`` and the bound -> 0
    (recovering exact coverage). Down-weighting old (high-``TV``) points with
    ``rho_decay < 1`` *shrinks the bound* versus uniform weights -- the formal reason
    time-decay helps under drift.

    Parameters
    ----------
    rho_decay:
        The time-decay base (``in (0, 1]``); selects the weights.
    n:
        Number of calibration points.
    tv_drift:
        ``(n,)`` per-point total-variation drift proxies ``TV_i in [0, 1]`` (oldest
        first, matching :func:`time_decay_weights`). In the test we use the known TV
        between the pre- and post-drift sampling laws for points on each side of the
        change point.

    Returns
    -------
    Scalar bound ``sum_i w~_i * TV_i``.
    """
    w_n = time_decay_weights(n, rho_decay)  # (n,) unnormalised
    denom = jnp.sum(w_n) + 1.0 + _W_EPS  # + w* (=1) for the test point
    w_tilde = w_n / denom  # (n,) normalised calibration weights
    return jnp.sum(w_tilde * jnp.asarray(tv_drift))


def empirical_coverage(
    p_values_B: chex.Array,
    alpha: float,
) -> chex.Array:
    r"""Empirical coverage ``P(p > alpha)`` of a p-value stream at level ``alpha``.

    A conformal predictor that *excludes* the test label when its p-value ``<= alpha``
    covers (includes the truth) exactly when ``p > alpha``; under valid (super-uniform)
    p-values ``P(p <= alpha) <= alpha`` so coverage ``= P(p > alpha) >= 1 - alpha``. The
    NexCP test measures how close this empirical coverage stays to the nominal
    ``1 - alpha`` under drift. Returns the scalar fraction ``mean(p_values > alpha)``.
    """
    return jnp.mean(jnp.asarray(p_values_B) > alpha)
