"""Conformal drift-detection **test martingale** -- the M4 scientific core.

This module implements Algorithm 2 / Appendix D of the paper: an *anytime-valid*
sequential test that watches a stream of conformal p-values and raises an alarm when
the perceived environment has **drifted** away from the null ``H0 = "no drift"``.

It is deliberately **score-agnostic**: it consumes a 1-D array of p-values
``p_1, p_2, ...`` and knows nothing about how they were produced. In the paper those
p-values come from :mod:`conformal_marl.conformal` (a per-agent non-conformity score
ranked against a *fixed* pre-drift calibration set), but any conditionally
super-uniform p-value stream is a valid input. This keeps the detector reusable and
makes its validity testable on purely synthetic streams (see ``tests/test_martingale.py``).

The mathematics
---------------
Let ``p_1, p_2, ...`` be the p-value stream. Under ``H0`` each ``p_t`` is
conditionally **super-uniform** given the past: ``P(p_t <= u | F_{t-1}) <= u`` for all
``u in [0, 1]`` (exactly uniform in the continuous, no-ties case).

* **Power martingale** for a fixed mixing parameter ``eps in (0, 1)``::

      M_t^eps = prod_{j <= t} eps * p_j^{eps - 1}

  Each factor has mean 1 under a Uniform(0, 1) p-value, because
  ``E[eps * p^{eps-1}] = eps * \\int_0^1 p^{eps-1} dp = eps * (1/eps) = 1``. A product
  of independent unit-mean factors is a **non-negative martingale** with
  ``E[M_t^eps] = 1`` (a supermartingale, ``E[M_t^eps] <= 1``, if the p-values are
  only super-uniform, e.g. with ties). A small ``eps`` bets on *small* p-values
  (drift makes the score large and the p-value small), a large ``eps`` bets on large.

* **Mixture martingale** -- to avoid choosing ``eps``, average the power
  martingales over a prior on ``eps``. With the Uniform(0, 1) prior::

      M_t = \\int_0^1 prod_{j <= t} eps * p_j^{eps - 1} d eps.

  The closed form of one factor's mixture is, for reference only::

      \\int_0^1 eps * p^{eps - 1} d eps = (1/p + ln p - 1) / (ln p)^2   ( -> 1/2 as p -> 1 )

  but the *product-then-integrate* has no elementary closed form and the integrand is
  singular as ``p -> 1`` (``p^{eps-1} -> infinity`` for ``eps < 1``... actually
  ``p^{eps-1}`` is bounded on ``p in (0,1]``; the genuine numerical hazard is the
  opposite tail, very small ``p`` with the running product, and catastrophic
  cancellation if done in linear space). We therefore use a **discrete grid mixture**
  over ``eps in {0.1, 0.2, ..., 0.9}`` and average, computed entirely in **log space**
  for stability (see below). The grid average is itself a convex combination of
  unit-mean martingales, so it is *exactly* a martingale with ``E[M_t] = 1`` (the grid
  vs. continuous-integral distinction changes only the mixing weights / power of the
  test, never the ``E[M_t] = 1`` validity).

* **Ville's inequality** -- the anytime guarantee. For any non-negative
  (super)martingale with ``M_0 = 1`` and any ``delta in (0, 1)``::

      P( sup_t M_t >= 1 / delta )  <=  delta.

  So thresholding at ``1 / delta`` controls the probability of *ever* raising a false
  alarm at level ``delta``, uniformly over time (no multiple-testing correction needed,
  unlike a fixed-horizon test). We **alarm** the first time ``M_t >= 1 / delta``,
  declare drift, (the caller refreshes its calibration set), and **reset** ``M <- 1``
  so the detector can fire again on the next change point.

Numerical stability (log space)
-------------------------------
For each grid point ``eps`` we carry the **running log-martingale**::

    L_t^eps = sum_{j <= t} [ ln eps + (eps - 1) * ln p_j ]   ( = ln M_t^eps )

which is a plain cumulative sum -- no products, no overflow. The mixture is then::

    M_t = mean_eps exp(L_t^eps) = exp( logsumexp_eps(L_t^eps) - ln(n_grid) )

evaluated with :func:`jax.scipy.special.logsumexp`, so even when individual
``M_t^eps`` over/underflow the float range the mixture stays accurate. We also return
``log M_t`` directly (``logsumexp - ln n_grid``) because under strong drift ``M_t``
itself overflows float32 while ``log M_t`` remains informative.

The update is a single :func:`jax.lax.scan` over the p-value stream with the per-grid
log-sums as carry; reset-on-alarm is a :func:`jax.lax.cond` inside the scan. Everything
is pure and ``jax.jit`` / ``jax.vmap``-friendly (vmap over independent streams gives
the Monte-Carlo validity tests for free).
"""

from __future__ import annotations

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# Default mixing grid: eps in {0.1, ..., 0.9}. Excludes the endpoints 0 and 1 where
# the power-martingale factor degenerates (eps = 0 -> 0; eps = 1 -> the trivial
# martingale M_t = prod 1 = 1, which never alarms). Nine interior points evenly
# spaced are the paper's grid.
DEFAULT_EPS_GRID = jnp.arange(1, 10, dtype=jnp.float32) / 10.0  # 0.1 .. 0.9

# Floor / ceil applied to incoming p-values before ``ln p`` so the log is finite and
# the bets are well-defined. Conformal p-values are already in (0, 1], but a defensive
# clip lets the module accept any [0, 1] stream (e.g. exact-0 from a degenerate score).
_P_FLOOR = 1e-12
_P_CEIL = 1.0


def _ln_p(p: chex.Array) -> chex.Array:
    """``ln p`` with a defensive clip to keep it finite on ``[0, 1]`` inputs."""
    return jnp.log(jnp.clip(p, _P_FLOOR, _P_CEIL))


def power_log_martingale(p_values: chex.Array, eps: float) -> chex.Array:
    """``ln M_t^eps`` for a **single** power martingale, the whole ``(T,)`` trace.

    ``M_t^eps = prod_{j<=t} eps * p_j^{eps-1}``, returned in log space as the cumulative
    sum ``L_t^eps = sum_{j<=t} [ln eps + (eps-1) ln p_j]``. Pure ``cumsum`` -- no scan,
    no reset (a single power martingale is the cleanest object on which to verify the
    ``E[M_t^eps] = 1`` martingale property of Appendix D, since the grid mixture's small-
    ``eps`` terms have astronomically heavy tails that no finite Monte-Carlo sample can
    estimate; see ``tests/test_martingale.py``). ``exp`` of the result is ``M_t^eps``.
    """
    ln_term = jnp.log(eps) + (eps - 1.0) * _ln_p(jnp.asarray(p_values))
    return jnp.cumsum(ln_term, axis=-1)


def mixture_closed_form_factor(p: chex.Array) -> chex.Array:
    """Closed-form *single-factor* mixture ``\\int_0^1 eps * p^{eps-1} d eps``.

    Equals ``(1/p + ln p - 1) / (ln p)^2`` and tends to ``1/2`` as ``p -> 1``. Provided
    for reference / cross-checking the grid mixture on a single p-value (the per-step
    "instantaneous" mixture weight); the sequential detector uses the log-space grid
    mixture in :func:`run_martingale`, not this, because the *product* over the stream
    has no elementary closed form. Implemented with a Taylor fallback near ``p = 1``
    where ``(ln p)^2`` underflows.
    """
    lp = jnp.log(jnp.clip(p, _P_FLOOR, _P_CEIL))
    # Switch to the Taylor expansion well before (ln p)^2 loses all float precision in
    # the cancellation of (1/p + ln p - 1). 1e-2 is comfortably inside float32's safe
    # range for the main branch and inside the Taylor remainder's accuracy.
    safe = jnp.abs(lp) > 1e-2
    pc = jnp.clip(p, _P_FLOOR, _P_CEIL)
    # Main branch: (1/p + ln p - 1) / (ln p)^2.
    main = (1.0 / pc + lp - 1.0) / (lp**2)
    # Near p = 1 (lp -> 0): 1/p = e^{-lp} = 1 - lp + lp^2/2 - lp^3/6 + ..., so
    # (1/p + lp - 1) = lp^2/2 - lp^3/6 + ..., and dividing by lp^2 gives
    # 1/2 - lp/6 + O(lp^2). Use this two-term expansion (-> exactly 1/2 at lp = 0).
    taylor = 0.5 - lp / 6.0
    return jnp.where(safe, main, taylor)


class MartingaleResult(NamedTuple):
    """Output of :func:`run_martingale`.

    Attributes
    ----------
    log_m:
        ``(T,)`` array of ``ln M_t`` *after* processing each p-value (and after any
        reset triggered at that step). Carried in log space because ``M_t`` overflows
        float under strong drift; ``log_m`` stays finite and is what the demo plots on
        a log axis. ``M_t = exp(log_m)``.
    m:
        ``(T,)`` array of ``M_t`` (the mixture martingale value), i.e. ``exp(log_m)``.
        May be ``inf`` under extreme drift -- prefer ``log_m`` for anything sensitive.
    alarms:
        ``(T,)`` boolean array; ``True`` at steps where ``M_t`` crossed the threshold
        ``1/delta`` (an alarm fired). With ``reset_on_alarm`` the martingale is reset to
        1 immediately after, so multiple alarms can appear in one stream.
    crossed:
        Scalar bool -- did the martingale *ever* cross the threshold? (``any(alarms)``).
        This is the quantity Ville's inequality bounds: under ``H0``,
        ``P(crossed) <= delta``.
    first_alarm:
        Scalar int -- index (0-based) of the first alarm, or ``-1`` if none. The demo's
        "detection delay" is ``first_alarm - change_point``.
    """

    log_m: chex.Array
    m: chex.Array
    alarms: chex.Array
    crossed: chex.Array
    first_alarm: chex.Array


def run_martingale(
    p_values: chex.Array,
    delta: float = 0.05,
    eps_grid: chex.Array = DEFAULT_EPS_GRID,
    reset_on_alarm: bool = True,
) -> MartingaleResult:
    """Run the conformal mixture test martingale over a p-value stream.

    Pure, jittable, and ``vmap``-able over a leading stream axis (feed
    ``(num_streams, T)`` under ``jax.vmap`` for Monte-Carlo validity experiments).

    Parameters
    ----------
    p_values:
        ``(T,)`` stream of conformal p-values in ``(0, 1]`` (conditionally
        super-uniform under ``H0``). Clipped defensively to ``[1e-12, 1]``.
    delta:
        Ville level. The alarm threshold is ``1/delta`` and, under ``H0``, the
        probability of *ever* alarming is ``<= delta``. Static (controls the threshold,
        not traced).
    eps_grid:
        1-D grid of mixing parameters ``eps in (0, 1)`` to average over. Defaults to
        ``{0.1, ..., 0.9}``. The mixture is the **mean** over this grid (a uniform-prior
        Riemann/Monte-Carlo approximation of ``\\int_0^1 ... d eps``).
    reset_on_alarm:
        If ``True`` (default, the paper's Algorithm 2), reset ``M <- 1`` immediately
        after each alarm so the detector re-arms for the next change point. If
        ``False``, the martingale runs unreset -- used by the ``E[M_t] = 1`` validity
        test, which needs the raw (unreset) martingale.

    Returns
    -------
    :class:`MartingaleResult` with the per-step ``log_m`` / ``m`` traces, the per-step
    ``alarms`` mask, the scalar ``crossed`` flag, and the ``first_alarm`` index.

    Notes
    -----
    The carry is the per-grid running log-sum ``L^eps`` (shape ``(n_grid,)``), all zero
    at ``t = 0`` (so ``M_0 = 1``). At each step ``t``:

    1. ``L^eps <- L^eps + ln eps + (eps - 1) * ln p_t``  (extend every power martingale);
    2. ``log M_t <- logsumexp(L^eps) - ln(n_grid)``       (mix in log space);
    3. ``alarm <- (log M_t >= ln(1/delta))``;
    4. if ``reset_on_alarm and alarm``: ``L^eps <- 0`` (so the *next* step starts from
       ``M = 1``); the reported ``log_m`` for this step is still the crossing value.

    Step 4 resets *after* recording the crossing value, so the trace shows the spike
    that triggered the alarm and then returns to ``ln 1 = 0`` on the following step.
    """
    eps_grid = jnp.asarray(eps_grid, dtype=jnp.float32)
    n_grid = eps_grid.shape[0]
    ln_n_grid = jnp.log(jnp.asarray(n_grid, dtype=jnp.float32))
    ln_thresh = jnp.log(1.0 / delta)
    ln_eps = jnp.log(eps_grid)  # (n_grid,) ; ln of each mixing weight's eps factor

    p_values = jnp.asarray(p_values, dtype=jnp.float32)

    def _step(carry_L, p_t):
        # carry_L: (n_grid,) running log-martingale per grid point (== ln M_t^eps).
        ln_p = _ln_p(p_t)
        # Extend every power martingale: ln M^eps += ln eps + (eps - 1) ln p.
        new_L = carry_L + ln_eps + (eps_grid - 1.0) * ln_p
        # Mixture in log space: ln M_t = logsumexp_eps(L^eps) - ln(n_grid).
        log_m_t = logsumexp(new_L) - ln_n_grid
        alarm_t = log_m_t >= ln_thresh

        # Reset-on-alarm: zero the per-grid log-sums after recording the crossing.
        def _do_reset(_):
            return jnp.zeros_like(new_L)

        def _no_reset(_):
            return new_L

        do_reset = jnp.logical_and(alarm_t, reset_on_alarm)
        carry_out = jax.lax.cond(do_reset, _do_reset, _no_reset, operand=None)
        return carry_out, (log_m_t, alarm_t)

    L0 = jnp.zeros((n_grid,), dtype=jnp.float32)  # M_0 = 1 for every grid point
    _, (log_m, alarms) = jax.lax.scan(_step, L0, p_values)

    crossed = jnp.any(alarms)
    # Index of the first alarm, or -1 if none (argmax of a bool gives the first True).
    first_alarm = jnp.where(crossed, jnp.argmax(alarms), -1).astype(jnp.int32)

    return MartingaleResult(
        log_m=log_m,
        m=jnp.exp(log_m),
        alarms=alarms,
        crossed=crossed,
        first_alarm=first_alarm,
    )


def detection_delay(result: MartingaleResult, change_point: int) -> chex.Array:
    """Detection delay ``first_alarm - change_point`` (``-1`` if never alarmed).

    A convenience for the drift demo: how many p-values *after* the known change point
    were needed before the martingale crossed the threshold. Negative-but-not-``-1``
    values would indicate a (false) alarm *before* the change point.
    """
    return jnp.where(result.first_alarm >= 0, result.first_alarm - change_point, -1)
