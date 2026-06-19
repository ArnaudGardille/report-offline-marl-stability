"""Fast synthetic tests for the reconstructed AES drift detector (``conformal_marl.aes_baseline``).

The AES baseline is the heuristic counterpart in the headline detection benchmark, so its
proxy + detector are exercised on synthetic **TD-error** streams (no environment, fast):

1. The smoothed proxy (EMA of the trailing 0.9-quantile of ``|TD error|``) **rises after a
   drift** that inflates the TD errors, and stays flat under no drift.
2. The reconstructed detector, with a frozen no-drift reference and a threshold, **fires
   after a known change point** on a drift stream and (at a sensibly calibrated threshold)
   rarely fires on H0 -- the property the benchmark calibrates to a matched false-alarm.
3. Shapes / jit / vmap contract.

These do *not* re-derive AES's entropy scheduler (that is a training-time knob, out of
scope for a detector comparison); they validate the drift-proxy-as-detector we build.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.aes_baseline import (
    aes_detector,
    batch_quantile_proxy,
    reference_stats,
    smoothed_proxy,
)


def _td_streams(key, n_streams, pre_len, post_len, pre_scale=0.2, post_scale=1.5):
    """``|TD error|`` streams: small (scale ``pre_scale``) then large (``post_scale``).

    Half-normal magnitudes; the post-change block has a larger scale, mimicking the
    TD-error inflation a drift causes (the critic is suddenly wrong about the new regime).
    """
    k_pre, k_post = jax.random.split(key)
    pre = pre_scale * jnp.abs(jax.random.normal(k_pre, (n_streams, pre_len)))
    post = post_scale * jnp.abs(jax.random.normal(k_post, (n_streams, post_len)))
    return jnp.concatenate([pre, post], axis=1)


def test_proxy_rises_after_td_inflation():
    """The smoothed AES proxy increases after the TD errors inflate, flat under no drift."""
    pre_len, post_len = 150, 150
    drift = _td_streams(jax.random.PRNGKey(0), 200, pre_len, post_len,
                        pre_scale=0.2, post_scale=1.5)
    proxy = jax.vmap(lambda x: smoothed_proxy(x))(drift)  # (S, T)
    proxy = np.asarray(proxy)
    pre_level = proxy[:, pre_len - 20:pre_len].mean()
    post_level = proxy[:, -20:].mean()
    assert post_level > 3 * pre_level, (
        f"proxy did not rise after drift: pre={pre_level:.3f} post={post_level:.3f}"
    )

    # No-drift control: same scale throughout -> proxy roughly flat.
    flat = _td_streams(jax.random.PRNGKey(1), 200, pre_len, post_len,
                       pre_scale=0.5, post_scale=0.5)
    proxy_flat = np.asarray(jax.vmap(lambda x: smoothed_proxy(x))(flat))
    a = proxy_flat[:, pre_len - 20:pre_len].mean()
    b = proxy_flat[:, -20:].mean()
    assert abs(b - a) < 0.3 * a, f"no-drift proxy not flat: {a:.3f} -> {b:.3f}"


def test_detector_fires_after_change_and_is_quiet_on_h0():
    """Detector fires soon after the change on drift, rarely on H0 (frozen ref+threshold).

    Calibrate the reference on a no-drift stream, pick a threshold, and check (a) on drift
    streams the detector crosses *after* the change point with high probability, and (b)
    on fresh H0 streams the ever-cross fraction is small. (The benchmark tunes the
    threshold to a *target* false-alarm; here we just assert the detector is discriminative
    at a reasonable threshold.)
    """
    pre_len, post_len = 150, 150
    cp = pre_len

    # Frozen reference from a no-drift stream.
    ref_stream = _td_streams(jax.random.PRNGKey(10), 1, 600, 0,
                             pre_scale=0.3, post_scale=0.3)[0]
    ref_mean, ref_std = reference_stats(ref_stream)

    threshold = 5.0  # z-score; calibrated empirically in the real benchmark

    drift = _td_streams(jax.random.PRNGKey(11), 300, pre_len, post_len,
                        pre_scale=0.3, post_scale=1.6)
    det = jax.vmap(
        lambda x: aes_detector(x, ref_mean, ref_std, threshold)
    )(drift)
    crossed = np.asarray(det.crossed)
    first = np.asarray(det.first_alarm)
    assert crossed.mean() > 0.9, f"detector missed drift: only {crossed.mean():.2%} fired"
    # Alarms should land after the change point (allow a few pre-change from window fill).
    pre_alarm = float(np.mean((first >= 0) & (first < cp)))
    assert pre_alarm < 0.1, f"too many pre-change AES alarms: {pre_alarm:.2%}"

    # H0 control: fresh no-drift streams, same ref/threshold -> few crossings.
    h0 = _td_streams(jax.random.PRNGKey(12), 300, pre_len, post_len,
                     pre_scale=0.3, post_scale=0.3)
    det0 = jax.vmap(lambda x: aes_detector(x, ref_mean, ref_std, threshold))(h0)
    fa = float(np.asarray(det0.crossed).mean())
    assert fa < 0.25, f"AES detector false-alarm too high at threshold {threshold}: {fa:.2%}"


def test_batch_quantile_tracks_upper_tail():
    """The trailing-window proxy approximates the 0.9-quantile of recent |TD error|.

    On a constant-scale stream the windowed 0.9-quantile should sit near the true 0.9
    quantile of the (half-normal) magnitudes once the window has filled.
    """
    key = jax.random.PRNGKey(2)
    scale = 1.0
    x = scale * jnp.abs(jax.random.normal(key, (2000,)))
    v = np.asarray(batch_quantile_proxy(x, q=0.9, window=128))
    true_q = float(np.quantile(np.asarray(x), 0.9))
    # Steady-state windowed quantile near the true quantile (loose: small window noise).
    est = v[256:].mean()
    assert abs(est - true_q) < 0.25 * true_q, (
        f"windowed 0.9-quantile {est:.3f} far from true {true_q:.3f}"
    )


def test_aes_detector_jit_and_vmap_shapes():
    """``aes_detector`` is jittable / vmap-able with the documented shapes."""
    streams = jnp.abs(jax.random.normal(jax.random.PRNGKey(9), (8, 60)))
    single = jax.jit(
        lambda x: aes_detector(x, 0.5, 0.1, 4.0)
    )(streams[0])
    assert single.proxy.shape == (60,)
    assert single.statistic.shape == (60,)
    assert single.alarms.shape == (60,) and single.alarms.dtype == jnp.bool_
    assert single.crossed.shape == () and single.first_alarm.shape == ()

    batched = jax.jit(jax.vmap(lambda x: aes_detector(x, 0.5, 0.1, 4.0)))(streams)
    assert batched.proxy.shape == (8, 60)
    assert batched.crossed.shape == (8,)
