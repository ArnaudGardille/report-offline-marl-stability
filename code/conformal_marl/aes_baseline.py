"""AES baseline -- the heuristic TD-error-entropy drift detector we compare against.

This is the **honest reconstruction** of the AES drift signal from

    Wang, Xia, Chen, Liu, "Tracking Drift: Variation-Aware Entropy Scheduling for
    Non-Stationary Reinforcement Learning", arXiv:2601.19624 (Jan 2026),

used as the heuristic counterpart to our conformal test martingale in the headline
detection experiment (``scripts/detection_benchmark.py``).

What AES actually is (and is not)
---------------------------------
AES is **not a drift detector** -- it is an *entropy scheduler*. Its paper makes the
point explicitly: AES performs "continuous entropy modulation" and "requires **no
explicit change-point detection**". Concretely it (i) reads a scalar **drift proxy** off
the critic at each update, (ii) smooths it, and (iii) feeds it into a square-root
scheduling rule for the entropy weight. The proxy (their Eq. 31) is the upper quantile
of the absolute TD errors on the current batch::

    v_hat_t = Quantile_q(|delta|),   q = 0.9                       (AES Eq. 31)

smoothed over time "via a window or EMA" (the exact EMA constant is deferred to their
Appendix C; we use a standard EMA, see ``AES_BETA``). The scheduler is::

    lambda_t = clip[lam_min, lam_max]( sqrt(C1/C2) * sqrt(A_hat_t / t) )   (AES Eq. 24)

with ``A_hat_t = sum_{s<=t} xi_hat_s`` the accumulated proxy budget.

Because AES ships no detector, to put it on the **same footing** as our martingale in a
detection-delay comparison we must *construct* a detector from its proxy. We do the most
natural and charitable thing: standardize the smoothed proxy against a **frozen pre-drift
reference** (mean/std of the proxy over a no-drift calibration window) and **raise an
alarm when the standardized proxy exceeds a threshold** that we calibrate to a target
false-alarm rate on ``H0`` streams (exactly as we calibrate our own detector). This is a
one-sided shift detector on the AES signal -- "the TD-error 0.9-quantile has risen well
above its no-drift level", which is precisely the drift indication AES's scheduler reacts
to. We document this as *our* reconstruction: the AES paper does not specify a threshold
or a false-alarm notion, so the threshold is something we add to make a detector out of
the scheduler's proxy.

The scientific contrast (the paper's claim)
-------------------------------------------
* Our conformal martingale fires at ``M_t >= 1/delta`` and Ville's inequality gives an
  **anytime** false-alarm guarantee ``P(ever alarm | H0) <= delta`` -- the threshold is
  ``1/delta``, known a priori, distribution-free, and valid for any stopping time.
* The AES-proxy detector has **no such guarantee**. Its threshold is a free knob with no
  analytic false-alarm control; we *must* calibrate it empirically on H0 streams, and
  that calibration is distribution-specific (a threshold tuned on one environment's
  TD-error scale does not transfer). The benchmark holds the achieved H0 false-alarm
  *equal* for both detectors and then compares detection delay -- so any delay difference
  is a like-for-like comparison, and the martingale additionally carries the guarantee
  the heuristic lacks.

This module is deliberately p-value-free and martingale-free: it consumes a stream of
**TD errors** (the quantity AES uses) and produces the AES proxy + the reconstructed
detector. It is pure / jittable so the calibration lab and the benchmark can ``vmap`` it
over many streams.
"""

from __future__ import annotations

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

# AES proxy quantile: their Eq. 31 uses q = 0.9 (the 90th percentile of |TD error|).
AES_Q = 0.9
# EMA smoothing constant for the proxy. The AES paper defers the exact value to its
# Appendix C ("smoothed via a window or EMA"); we use a standard slow EMA (~20-step
# memory), matching the spirit of their temporal smoothing. Documented as our choice.
AES_BETA = 0.05
# Trailing window (number of recent TD errors) over which the batch-quantile proxy is
# computed each step -- the streaming analogue of AES's "current update batch" quantile.
AES_WINDOW = 64


def batch_quantile_proxy(
    abs_td: chex.Array, q: float = AES_Q, window: int = AES_WINDOW
) -> chex.Array:
    """Streaming AES proxy ``v_hat_t = Quantile_q(|delta|)`` over a trailing window ``(T,)``.

    AES computes the ``q``-quantile of the absolute TD errors *on the current update
    batch*. In a per-transition stream there is no batch, so we take the quantile over a
    trailing window of the last ``window`` absolute TD errors -- the streaming analogue of
    a batch. Steps before a full window use the available prefix.

    Pure ``lax.scan`` keeping a fixed-length ring of the most recent ``window`` values;
    jit/vmap-able. (The quantile over a masked ring is computed by pushing the invalid
    prefix to ``+inf`` so it never lands at or below the 0.9-quantile.)
    """
    abs_td = jnp.asarray(abs_td, dtype=jnp.float32)
    w = int(window)

    def _step(carry, x_t):
        ring, count = carry  # ring: (w,) most-recent at the end; count: filled length
        ring = jnp.concatenate([ring[1:], x_t[jnp.newaxis]])
        count = jnp.minimum(count + 1, w)
        # Mask invalid (unfilled) prefix to +inf so it sits above the upper quantile.
        valid = jnp.arange(w) >= (w - count)
        masked = jnp.where(valid, ring, jnp.inf)
        # Quantile over the valid suffix: use the 'lower' interpolation on the finite
        # part. jnp.quantile with +inf padding would be polluted, so compute over a
        # sorted finite slice by counting. Simplest jittable route: sort the masked ring
        # (inf sinks to the top) and index the q-quantile position within ``count``.
        srt = jnp.sort(masked)
        # position of the q-quantile among ``count`` valid entries (0-based, clamped).
        pos = jnp.floor(q * (count.astype(jnp.float32) - 1.0) + 0.5).astype(jnp.int32)
        pos = jnp.clip(pos, 0, w - 1)
        v_t = srt[pos]
        return (ring, count), v_t

    ring0 = jnp.zeros((w,), dtype=jnp.float32)
    count0 = jnp.zeros((), dtype=jnp.int32)
    _, v = jax.lax.scan(_step, (ring0, count0), abs_td)
    return v


def smoothed_proxy(abs_td: chex.Array, q: float = AES_Q, window: int = AES_WINDOW,
                   beta: float = AES_BETA) -> chex.Array:
    """AES proxy after EMA smoothing: ``xi_t = beta * v_hat_t + (1-beta) * xi_{t-1}`` ``(T,)``.

    The full AES drift signal: the trailing-window 0.9-quantile of ``|TD error|``
    (:func:`batch_quantile_proxy`) passed through a causal EMA (``xi_{-1} = 0``). This is
    the scalar AES feeds its scheduler; we feed it our reconstructed detector. Pure /
    jittable.
    """
    v = batch_quantile_proxy(abs_td, q=q, window=window)
    b = jnp.asarray(beta, dtype=jnp.float32)

    def _step(xi_prev, v_t):
        xi = b * v_t + (1.0 - b) * xi_prev
        return xi, xi

    _, xi = jax.lax.scan(_step, jnp.zeros((), dtype=jnp.float32), v)
    return xi


class AESDetectorResult(NamedTuple):
    """Output of :func:`aes_detector`.

    Attributes
    ----------
    proxy:
        ``(T,)`` smoothed AES drift proxy (the EMA'd 0.9-quantile of ``|TD error|``).
    statistic:
        ``(T,)`` standardized proxy ``(proxy - ref_mean) / ref_std`` -- the detector's
        running statistic (a z-score of the AES signal against its no-drift reference).
    alarms:
        ``(T,)`` boolean: ``statistic >= threshold`` (an alarm fired this step).
    crossed:
        Scalar bool -- did the statistic ever reach the threshold?
    first_alarm:
        Scalar int -- first alarm index, or ``-1`` (detection delay vs a known change
        point is ``first_alarm - change_point``).
    """

    proxy: chex.Array
    statistic: chex.Array
    alarms: chex.Array
    crossed: chex.Array
    first_alarm: chex.Array


def aes_detector(
    abs_td: chex.Array,
    ref_mean: chex.Array,
    ref_std: chex.Array,
    threshold: float,
    q: float = AES_Q,
    window: int = AES_WINDOW,
    beta: float = AES_BETA,
) -> AESDetectorResult:
    """Reconstructed AES drift **detector**: threshold the standardized AES proxy.

    Builds the smoothed AES proxy (:func:`smoothed_proxy`), standardizes it against a
    **frozen** no-drift reference (``ref_mean`` / ``ref_std`` estimated on a pre-drift
    calibration window of the proxy), and alarms the first time the standardized proxy
    (z-score) reaches ``threshold``. One-sided (drift raises the TD-error quantile, so we
    only test the upper tail).

    The reference and threshold are *frozen* (not adapted) so that, like Channel A of our
    controller, the detector's decisions are well-defined; but unlike the martingale the
    threshold has **no analytic false-alarm meaning** -- it is calibrated empirically on
    H0 streams in the benchmark to match our detector's achieved false-alarm rate.

    Parameters
    ----------
    abs_td:
        ``(T,)`` stream of **absolute** TD errors (the AES proxy operand).
    ref_mean, ref_std:
        Scalar no-drift reference mean/std of the *smoothed proxy* (computed once on a
        frozen pre-drift window). Used to standardize the running proxy.
    threshold:
        Alarm level on the standardized proxy (calibrated to a target false-alarm rate).
    q, window, beta:
        AES proxy hyper-parameters (quantile level, trailing-window length, EMA decay).

    Returns
    -------
    :class:`AESDetectorResult`. Pure / jittable / vmap-able over a leading stream axis.
    """
    proxy = smoothed_proxy(abs_td, q=q, window=window, beta=beta)
    std = jnp.maximum(jnp.asarray(ref_std, dtype=jnp.float32), 1e-6)
    statistic = (proxy - jnp.asarray(ref_mean, dtype=jnp.float32)) / std
    alarms = statistic >= jnp.asarray(threshold, dtype=jnp.float32)
    crossed = jnp.any(alarms)
    first_alarm = jnp.where(crossed, jnp.argmax(alarms), -1).astype(jnp.int32)
    return AESDetectorResult(
        proxy=proxy,
        statistic=statistic,
        alarms=alarms,
        crossed=crossed,
        first_alarm=first_alarm,
    )


def reference_stats(abs_td_ref: chex.Array, q: float = AES_Q, window: int = AES_WINDOW,
                    beta: float = AES_BETA) -> tuple[chex.Array, chex.Array]:
    """Frozen no-drift reference ``(mean, std)`` of the smoothed AES proxy.

    Run on a pre-drift (``H0``) stream of absolute TD errors to get the standardization
    constants for :func:`aes_detector`. We discard the first ``window`` steps (the proxy
    is still filling its trailing window there) before taking the mean/std, so the
    reference reflects the steady-state no-drift proxy level. Returns Python-friendly
    scalar arrays.
    """
    proxy = smoothed_proxy(abs_td_ref, q=q, window=window, beta=beta)
    w = int(window)
    tail = proxy[w:] if proxy.shape[0] > w else proxy
    return jnp.mean(tail), jnp.std(tail)
