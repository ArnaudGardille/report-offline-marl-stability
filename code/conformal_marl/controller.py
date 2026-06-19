"""Two-channel conformal **drift controller** -- the M5 centerpiece.

This module wires the M4 conformal test martingale (:mod:`conformal_marl.martingale`)
into the *replay-control* loop of the pivoted paper. The design principle (paper
section "two-channel controller") is a clean **separation of concerns** that lets the
detector adapt replay *without* sacrificing the anytime-valid drift guarantee:

* **Channel A -- the FROZEN alarm channel.** The M4 grid-mixture martingale run on the
  stream of conformal p-values scored against a **frozen** pre-drift calibration set.
  Its Ville alarm (``M_t >= 1/delta``) is the *valid* "the perceived environment has
  drifted" signal: because the calibration set is never adapted, the p-value stream is
  exchangeable under ``H0`` and Ville's inequality gives ``P(ever alarm | H0) <= delta``
  -- an anytime false-alarm guarantee that holds no matter how long the stream runs.
  This channel's output (alarm / first-alarm index) is what the paper reports as the
  certified change detector.

* **Channel B -- the ADAPTIVE drift-rate channel.** A point estimate of the **drift
  rate** -- the speed at which the perceived environment is moving away from
  calibration -- read off the martingale's **log-wealth slope** ``d(log M_t)/dt``. Under
  a sustained post-change distribution the running ``log M_t`` of a power/mixture test
  martingale grows *linearly* with a slope equal to the per-step expected log-wealth
  increment, which is a Kullback-Leibler-type divergence between the current p-value
  distribution and the (uniform) null -- i.e. a **KL rate**. A steeper slope means a
  larger distribution shift. We estimate that slope online and map it monotonically to a
  **replay control knob** (the chosen knob is the *buffer half-life* / staleness
  discount; see :func:`drift_rate_to_half_life`).

Why the separation matters (the validity argument)
--------------------------------------------------
If the adaptation in Channel B were allowed to feed back into Channel A's calibration
set -- e.g. "drift is high, so refresh the calibration set" -- the p-value stream feeding
the alarm would no longer be exchangeable with a fixed reference, and Ville's inequality
(which assumes a martingale started at 1 under a *fixed* null) would no longer bound the
false-alarm probability. By keeping the *adaptation* strictly **downstream** of the
frozen martingale (Channel B only *reads* ``log M_t``; it never changes the calibration
set, the threshold, or the p-values that Channel A consumes), the alarm channel's
guarantee is preserved exactly while the replay knob still tracks the drift severity.
:func:`two_channel_controller` runs both channels off the *same* frozen p-value stream
and returns them side by side; :func:`run_with_adaptive_calibration` demonstrates that
even a *pathological* Channel-B-driven adaptation leaves Channel A's decisions untouched
(used by the separation test).

Drift-rate estimator (Channel B internals)
------------------------------------------
The raw signal is the **non-reset** log-martingale ``L_t = log M_t`` from
:func:`run_martingale` with ``reset_on_alarm=False`` (a reset would zero ``L`` and
corrupt the slope just when drift is strongest). Two estimators are provided:

* :func:`ewma_logwealth_slope` -- an exponentially weighted moving average of the
  one-step increments ``dL_t = L_t - L_{t-1}`` with decay ``beta`` (default 0.1, i.e. a
  ~10-step memory). Cheap, causal, jittable; the per-step control signal.
* :func:`windowed_logwealth_slope` -- an ordinary-least-squares slope of ``L`` over a
  trailing window of ``w`` steps (a smoother estimate for plotting / the closed-loop
  knob). Also jittable (fixed window via a strided convolution-free running fit).

Both are *rate* estimates (log-wealth per step); under the null they fluctuate around a
small value (the mixture martingale gently decays, so the slope is ``<= 0`` in
expectation), and under drift they become strongly positive in proportion to the KL
rate. The mapping :func:`drift_rate_to_half_life` clamps the rate at 0 (no negative
"anti-drift" forgetting) and converts it to a buffer half-life.

Everything here is pure and ``jax.jit`` / ``jax.vmap``-friendly so it can run inside the
training scan or be Monte-Carlo'd over synthetic streams in the tests.
"""

from __future__ import annotations

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from conformal_marl.martingale import (
    DEFAULT_EPS_GRID,
    MartingaleResult,
    run_martingale,
)

# ----------------------------------------------------------------- Channel B: estimator


def logwealth_increments(log_m: chex.Array) -> chex.Array:
    """One-step log-wealth increments ``dL_t = log M_t - log M_{t-1}`` (``(T,)``).

    The first increment is taken against ``log M_0 = 0`` (the martingale starts at
    ``M_0 = 1``), so ``dL_0 = log M_0_actual``. Under ``H0`` these average to a small
    non-positive number (the mixture gently decays as it bets and loses); under drift
    they average to the positive KL rate. The drift-rate estimators below are just
    causal smoothers of this sequence.
    """
    log_m = jnp.asarray(log_m, dtype=jnp.float32)
    prev = jnp.concatenate([jnp.zeros((1,), dtype=log_m.dtype), log_m[:-1]])
    return log_m - prev


def ewma_logwealth_slope(log_m: chex.Array, beta: float = 0.1) -> chex.Array:
    """Causal EWMA of the log-wealth increments -> per-step drift-rate estimate ``(T,)``.

    ``s_t = beta * dL_t + (1 - beta) * s_{t-1}`` with ``s_{-1} = 0`` and
    ``dL_t = log M_t - log M_{t-1}``. This is the Channel-B drift-rate signal: an
    exponentially weighted estimate of ``d(log M_t)/dt`` with an effective memory of
    ``~1/beta`` steps. Larger ``beta`` reacts faster but is noisier.

    The estimate is a **rate** (log-wealth per step). Because the per-step expected
    increment of the mixture martingale equals a KL-type divergence between the current
    p-value law and the uniform null, ``s_t`` tracks that **KL rate**: ~0 (slightly
    negative) under ``H0``, strongly positive under a sustained drift, and monotone in
    the drift severity (validated in ``tests/test_controller.py``).

    Pure ``lax.scan``; jit/vmap-able over a leading stream axis.
    """
    dL = logwealth_increments(log_m)
    b = jnp.asarray(beta, dtype=jnp.float32)

    def _step(s_prev, dl):
        s = b * dl + (1.0 - b) * s_prev
        return s, s

    _, s = jax.lax.scan(_step, jnp.zeros((), dtype=jnp.float32), dL)
    return s


def windowed_logwealth_slope(log_m: chex.Array, window: int = 20) -> chex.Array:
    """Trailing ordinary-least-squares slope of ``log M_t`` over ``window`` steps ``(T,)``.

    At each ``t`` fits a line to ``(t - window + 1, ..., t)`` vs ``L`` and returns its
    slope (least-squares ``d L / dt``). Steps before a full window are fit on the
    available prefix (so the estimate is defined from ``t = 1``). Smoother and less
    biased than the EWMA for a *constant* drift rate, at the cost of a ``window``-step
    lag; used for the plotted drift-rate curve and the closed-loop knob.

    Implementation is a fixed-size sliding-window OLS computed with running sums (no
    Python loop over the window), so it stays jittable. The slope of ``y`` on
    ``x = 0..k-1`` is ``(k * sum(xy) - sum(x) sum(y)) / (k * sum(x^2) - sum(x)^2)``.
    """
    log_m = jnp.asarray(log_m, dtype=jnp.float32)
    t_total = log_m.shape[0]
    w = int(window)

    # Per-end-index running window stats via a scan that keeps the last ``w`` values.
    # We carry a fixed-length ring of the most recent ``w`` L-values and recompute the
    # OLS slope each step over the *valid* suffix (count grows from 1 to w).
    idx = jnp.arange(w, dtype=jnp.float32)  # local x = 0..w-1 (newest gets largest x)

    def _step(carry, l_t):
        ring, count = carry  # ring: (w,) newest at the end; count: filled length
        ring = jnp.concatenate([ring[1:], l_t[jnp.newaxis]])
        count = jnp.minimum(count + 1, w)
        # Valid mask: the last ``count`` entries of the ring are real data.
        valid = jnp.arange(w) >= (w - count)
        x = idx  # 0..w-1; only masked entries count
        n = count.astype(jnp.float32)
        # Masked sums (zero out the invalid prefix).
        xm = jnp.where(valid, x, 0.0)
        ym = jnp.where(valid, ring, 0.0)
        sx = jnp.sum(xm)
        sy = jnp.sum(ym)
        sxx = jnp.sum(xm * xm)
        sxy = jnp.sum(xm * ym)
        denom = n * sxx - sx * sx
        slope = jnp.where(denom > 1e-8, (n * sxy - sx * sy) / denom, 0.0)
        return (ring, count), slope

    ring0 = jnp.zeros((w,), dtype=jnp.float32)
    count0 = jnp.zeros((), dtype=jnp.int32)
    _, slopes = jax.lax.scan(_step, (ring0, count0), log_m)
    # If the whole stream is shorter than 2 points the slope is undefined -> 0.
    return jnp.where(t_total >= 2, slopes, jnp.zeros_like(slopes))


# ----------------------------------------------------------- Channel B: knob mapping


# Default control-knob bounds for the buffer half-life (in *replay steps*). When the
# estimated drift rate is ~0 (no drift) the half-life sits at ``HALF_LIFE_MAX`` (the
# buffer keeps a long memory, == a near-uniform replay); as the drift rate grows the
# half-life shrinks toward ``HALF_LIFE_MIN`` (the buffer aggressively forgets stale
# transitions, biasing replay toward the post-drift regime). The shrink is exponential
# in the rate with gain ``KAPPA`` (drift-rate units of log-wealth per step).
HALF_LIFE_MAX = 50_000.0
HALF_LIFE_MIN = 1_000.0
KAPPA = 4.0


def drift_rate_to_half_life(
    drift_rate: chex.Array,
    half_life_max: float = HALF_LIFE_MAX,
    half_life_min: float = HALF_LIFE_MIN,
    kappa: float = KAPPA,
) -> chex.Array:
    """Map an estimated drift rate to a **buffer half-life** (the chosen replay knob).

    The control law (documented in the paper's two-channel section)::

        r_+   = max(drift_rate, 0)                        # ignore "anti-drift"
        H(r)  = clip( H_max * exp(-kappa * r_+),  H_min,  H_max )

    i.e. an exponentially decreasing half-life in the (clamped) drift rate. With no
    drift (``r_+ = 0``) the half-life is ``H_max`` -- a long memory, replay ~ uniform.
    As the drift rate climbs, ``H`` falls toward ``H_min`` so the staleness discount
    ``2^{-age/H}`` decays old transitions faster and replay concentrates on the current
    (post-drift) distribution. The choice of *half-life* as the knob is deliberate: it
    is the most direct lever on replay non-stationarity (it sets how fast the buffer
    forgets) and it leaves the **alarm** channel untouched, preserving the guarantee.

    ``kappa`` sets the sensitivity: a drift rate of ``ln(2)/kappa`` halves the half-life
    once, ``2 ln(2)/kappa`` halves it twice, etc. Pure / elementwise; jit/vmap-able.
    """
    r = jnp.maximum(jnp.asarray(drift_rate, dtype=jnp.float32), 0.0)
    h = half_life_max * jnp.exp(-jnp.asarray(kappa, dtype=jnp.float32) * r)
    return jnp.clip(h, half_life_min, half_life_max)


def staleness_weight(age: chex.Array, half_life: chex.Array) -> chex.Array:
    """Staleness discount ``2^{-age/half_life}`` for a transition of the given ``age``.

    The replay weight Channel B ultimately drives: a transition stored ``age`` replay
    steps ago is down-weighted by ``2^{-age/H}``, so with a short half-life (high drift)
    old transitions are exponentially suppressed. Provided so the closed-loop script
    (P3) and tests can turn the controller's ``half_life`` into actual per-sample replay
    weights. Pure / elementwise.
    """
    age = jnp.asarray(age, dtype=jnp.float32)
    half_life = jnp.asarray(half_life, dtype=jnp.float32)
    return jnp.power(2.0, -age / jnp.maximum(half_life, 1e-6))


# --------------------------------------------------------------- the two-channel API


class ControllerResult(NamedTuple):
    """Output of :func:`two_channel_controller`.

    Attributes
    ----------
    alarm:
        Channel A. Scalar bool -- did the frozen martingale ever cross ``1/delta``?
        (``MartingaleResult.crossed``.) This is the certified drift alarm.
    first_alarm:
        Channel A. Scalar int -- index of the first alarm, or ``-1`` (the detection
        delay relative to a known change point is ``first_alarm - change_point``).
    log_m:
        Channel A. ``(T,)`` running ``log M_t`` of the **frozen** martingale (the
        anytime wealth process). Carried in log space (overflows otherwise).
    alarms:
        Channel A. ``(T,)`` boolean per-step alarm mask.
    drift_rate:
        Channel B. ``(T,)`` per-step drift-rate estimate (EWMA log-wealth slope) -- the
        adaptive signal. ~0 under ``H0``, positive and increasing in the KL rate under
        drift.
    half_life:
        Channel B. ``(T,)`` buffer half-life implied by ``drift_rate`` via
        :func:`drift_rate_to_half_life` -- the actual replay knob value at each step.
    """

    alarm: chex.Array
    first_alarm: chex.Array
    log_m: chex.Array
    alarms: chex.Array
    drift_rate: chex.Array
    half_life: chex.Array


def two_channel_controller(
    p_values: chex.Array,
    delta: float = 0.05,
    eps_grid: chex.Array = DEFAULT_EPS_GRID,
    beta: float = 0.1,
    half_life_max: float = HALF_LIFE_MAX,
    half_life_min: float = HALF_LIFE_MIN,
    kappa: float = KAPPA,
) -> ControllerResult:
    """Run both channels off a single **frozen** conformal p-value stream.

    Channel A is :func:`run_martingale` with ``reset_on_alarm=True`` (the paper's
    Algorithm 2 alarm, re-arming after each change point) -- its ``crossed`` /
    ``first_alarm`` / ``alarms`` are the certified detector output. Channel B reads the
    log-wealth of an *un-reset* martingale (``reset_on_alarm=False``, so the slope is
    not corrupted by resets), estimates the drift rate with an EWMA of the log-wealth
    increments, and maps it to a buffer half-life.

    Both channels consume the *same* frozen p-values; Channel B never modifies the
    calibration, the threshold, or the p-values feeding Channel A -- so Channel A's Ville
    guarantee is untouched by the adaptation (the separation property; see module
    docstring and the separation test).

    Parameters
    ----------
    p_values:
        ``(T,)`` conformal p-value stream scored against a **frozen** calibration set
        (conditionally super-uniform under ``H0``).
    delta:
        Ville level for Channel A (threshold ``1/delta``; anytime false-alarm ``<=
        delta`` under ``H0``).
    eps_grid:
        Mixing grid for the martingale (defaults to ``{0.1, ..., 0.9}``).
    beta:
        EWMA decay for the Channel-B drift-rate estimator (~``1/beta`` step memory).
    half_life_max, half_life_min, kappa:
        Channel-B knob-mapping parameters (see :func:`drift_rate_to_half_life`).

    Returns
    -------
    :class:`ControllerResult` with both channels' per-step traces and Channel A's alarm
    summary. Pure / jittable / vmap-able over a leading stream axis.
    """
    # Channel A: frozen alarm martingale (reset-on-alarm so it re-arms per change point).
    alarm_res: MartingaleResult = run_martingale(
        p_values, delta=delta, eps_grid=eps_grid, reset_on_alarm=True
    )

    # Channel B: drift-rate from the *un-reset* log-wealth slope (reset would zero L).
    raw_res: MartingaleResult = run_martingale(
        p_values, delta=delta, eps_grid=eps_grid, reset_on_alarm=False
    )
    drift_rate = ewma_logwealth_slope(raw_res.log_m, beta=beta)
    half_life = drift_rate_to_half_life(
        drift_rate,
        half_life_max=half_life_max,
        half_life_min=half_life_min,
        kappa=kappa,
    )

    return ControllerResult(
        alarm=alarm_res.crossed,
        first_alarm=alarm_res.first_alarm,
        log_m=alarm_res.log_m,
        alarms=alarm_res.alarms,
        drift_rate=drift_rate,
        half_life=half_life,
    )


def run_with_adaptive_calibration(
    p_values: chex.Array,
    adapt_fn,
    delta: float = 0.05,
    eps_grid: chex.Array = DEFAULT_EPS_GRID,
) -> MartingaleResult:
    """Channel A with a Channel-B-style adaptation applied **only downstream**.

    A demonstrator (and the separation test's foil): it runs the *frozen* Channel-A
    martingale on ``p_values`` exactly as :func:`two_channel_controller` does, then
    calls ``adapt_fn(controller_signal)`` to compute a control action -- but that action
    is **not** fed back into the martingale. The returned :class:`MartingaleResult` is
    bit-for-bit the frozen-channel result regardless of ``adapt_fn``: this is the whole
    point of the separation. The test asserts the alarm decisions are identical across
    wildly different ``adapt_fn`` choices.

    ``adapt_fn`` takes the ``(T,)`` drift-rate signal and returns anything (e.g. a
    half-life schedule); its return value is intentionally discarded here -- the function
    exists to prove that *invoking* the adaptive channel cannot perturb the alarm.
    """
    res = run_martingale(p_values, delta=delta, eps_grid=eps_grid, reset_on_alarm=True)
    raw = run_martingale(p_values, delta=delta, eps_grid=eps_grid, reset_on_alarm=False)
    # Exercise the adaptive channel (its output is deliberately unused: separation).
    _ = adapt_fn(ewma_logwealth_slope(raw.log_m))
    return res
