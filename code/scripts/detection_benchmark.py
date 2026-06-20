"""HEADLINE EXPERIMENT (M5 P1): detection delay vs the AES baseline at matched false-alarm.

The pivoted paper's centrepiece. We compare two drift detectors on a controlled
partner-swap, **calibrated to the same false-alarm rate**, and report detection delay:

* **Conformal test martingale** (ours, ``conformal_marl.martingale`` / ``controller`` Channel
  A): watches the stream of conformal p-values (incoming transitions scored against a
  *frozen* pre-swap calibration set) and alarms at ``M_t >= 1/delta``. Ville's inequality
  gives an **anytime** false-alarm guarantee ``P(ever alarm | H0) <= delta`` -- the
  threshold is fixed a priori, distribution-free.
* **AES baseline** (``conformal_marl.aes_baseline``): our honest *reconstruction* of the
  drift signal from arXiv:2601.19624 ("Tracking Drift: Variation-Aware Entropy
  Scheduling"). AES itself is an entropy *scheduler* with **no change-point detection**;
  its drift proxy is the EMA-smoothed 0.9-quantile of ``|TD error|`` (their Eq. 31). We
  turn that proxy into a detector by standardizing it against a frozen no-drift reference
  and thresholding -- a detector AES does not itself provide, with **no analytic
  false-alarm guarantee**, so its threshold must be calibrated empirically.

Experimental protocol
----------------------
1. **False-alarm lab on Simultaneous-Attack (cheap).** Generate many *no-swap* (H0)
   streams. The conformal detector's threshold is fixed by Ville (``1/delta``); we measure
   its **achieved** H0 false-alarm rate. We then **sweep the AES detector's z-threshold and
   pick the one whose H0 false-alarm rate matches the conformal detector's achieved rate**
   -- so the two detectors are compared at *exactly* the same empirical false-alarm, not
   just the same nominal level. (We also report the AES threshold for the nominal delta.)

2. **Detection delay on Overcooked (the real benchmark).** For each seed, build a
   partner-swap stream (reusing the M4 swap infrastructure), compute *both* the conformal
   p-values and the per-transition ``|TD error|`` under the frozen policy, run both
   detectors at their matched thresholds, and record the detection delay
   (``first_alarm - swap_step``). We report the delay distribution over seeds for each
   detector, plus the per-detector miss rate.

Honesty
-------
Both detectors see the *same* incoming transitions (so the comparison is like-for-like):
the conformal detector reads their p-values, the AES detector reads their TD errors. We
report whichever wins on delay. The conformal detector additionally carries the Ville
guarantee the AES heuristic lacks -- which is the paper's real differentiator even if the
delays are close. Any miss, tie, or AES win is reported as-is.

Outputs
-------
* ``code/results/m5_detection_vs_aes.png`` (committed figure: matched-false-alarm bar +
  detection-delay distributions).
* ``code/outputs/m5_detection_benchmark.json`` (gitignored raw numbers).
* A printed summary with the achieved Simu false-alarm rates (matched?) and the Overcooked
  detection delays for both detectors.

M8 update
---------
The detection-delay measurement is now **env-agnostic** (:func:`detection_delay_env`):
the original Overcooked path is one call into it (kept as :func:`detection_delay_overcooked`),
and ``--env mpe`` runs the same experiment on MPE ``simple_spread`` -- a cheaper *and*
cleaner showcase, because that policy is non-trivial (its post-swap behaviour change is a
real drift, not noise around a near-random Overcooked policy). The partner-swap stream
collector (:func:`drift_demo.collect_stream`) drives **all** non-zero agents with the
swappable partner policy, so it works for the 3-agent MPE as well as the 2-agent envs.

Usage::

    uv run python scripts/detection_benchmark.py                 # full (Simu lab + Overcooked)
    uv run python scripts/detection_benchmark.py --env mpe       # Simu lab + MPE simple_spread
    uv run python scripts/detection_benchmark.py --no-heavy      # Simu lab only (fastest)
    uv run python scripts/detection_benchmark.py --env mpe --seeds 8 --delta 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.aes_baseline import aes_detector, reference_stats
from conformal_marl.conformal import conformal_p_values, policy_scores
from conformal_marl.envs.registry import make_env
from conformal_marl.iql import get_env_info, make_train
from conformal_marl.martingale import run_martingale
from conformal_marl.networks import QNetwork
from conformal_marl.replay import Transition

# Reuse the M4 swap infrastructure verbatim (no duplication). ``scripts/`` is not a
# package (scripts run as ``uv run python scripts/foo.py``), so add this dir to the path
# and import the sibling module directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_demo import (  # noqa: E402
    BEHAVIOUR_EPS,
    SCORE_EPS,
    collect_stream,
    least_likely_action,
    make_constant_partner_params,
    short_iql_config,
)

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
OUTPUTS_DIR = CODE_DIR / "outputs"


# --------------------------------------------------------------------- TD-error signal
def td_errors_under_policy(batch: Transition, params, apply_fn, hidden_dim, gamma):
    """Per-transition **absolute** double-DQN TD residual under a frozen policy ``(T,)``.

    The AES proxy operand. Mirrors ``iql._loss_fn``'s residual exactly (double-DQN: online
    net picks the masked argmax at ``next_obs``, the *same frozen* params evaluate it --
    here online == target == the frozen policy, the natural choice for a fixed detector),
    then averages the absolute residual over agents to a per-transition scalar.

    ``td = chosen_q - [r + gamma (1 - done) * max_a' Q(next_obs, a')]``; returns
    ``mean_agents |td|`` per transition.
    """
    b, n = batch.actions.shape
    flat = lambda x: x.reshape((b * n,) + x.shape[2:])  # noqa: E731
    obs_BN = flat(batch.obs)
    next_obs_BN = flat(batch.next_obs)
    actions_BN = flat(batch.actions)
    reward_BN = flat(batch.reward)
    done_BN = flat(batch.done).astype(jnp.float32)
    avail_BN = flat(batch.avail_actions)

    carry = QNetwork.initialize_carry(hidden_dim, b * n)
    resets = jnp.zeros((1, b * n))

    def _q(obs):
        _, q = apply_fn(params, carry, (obs[jnp.newaxis], resets))
        return q[0]  # (B*N, A)

    q_BNA = _q(obs_BN)
    q_next_BNA = _q(next_obs_BN)

    chosen_q = jnp.take_along_axis(q_BNA, actions_BN[:, jnp.newaxis], axis=-1)[:, 0]
    masked_next = jnp.where(avail_BN > 0, q_next_BNA, -1e9)
    max_next = jnp.max(masked_next, axis=-1)
    target = reward_BN + gamma * (1.0 - done_BN) * max_next
    td_BN = (chosen_q - target).reshape((b, n))
    return jnp.mean(jnp.abs(td_BN), axis=-1)  # (T,)


# ------------------------------------------------------------ build a dual-signal stream
def build_signals_jax(env, info, net, params_self, params_partner_post, hidden_dim,
                      pre_len, post_len, key, gamma=0.99):
    """Dual detector signals ``(p0, abs_td)`` as **jax arrays** (vmappable over ``key``).

    Rolls ``pre_len`` steps under (self, self) then ``post_len`` under (self,
    ``params_partner_post``); if ``params_partner_post`` *is* ``params_self`` this is a
    no-swap H0 stream (used for the Overcooked-native AES calibration). Returns the
    agent-0 conformal p-values (vs a frozen pre-swap calibration set) and the
    per-transition ``|TD error|`` under the frozen policy -- both on the *same* stream.
    Pure jax so the H0 lab can ``jax.vmap`` it over keys.
    """
    k_cal, k_pre, k_post = jax.random.split(key, 3)
    cal = collect_stream(env, info, net, params_self, params_self, k_cal,
                         4 * pre_len, hidden_dim, BEHAVIOUR_EPS)
    cal_scores = policy_scores(cal, params_self, net.apply, SCORE_EPS, hidden_dim)
    pre = collect_stream(env, info, net, params_self, params_self, k_pre,
                        pre_len, hidden_dim, BEHAVIOUR_EPS)
    post = collect_stream(env, info, net, params_self, params_partner_post, k_post,
                         post_len, hidden_dim, BEHAVIOUR_EPS)
    stream = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate([a, b], axis=0), pre, post
    )
    stream_scores = policy_scores(stream, params_self, net.apply, SCORE_EPS, hidden_dim)
    p0 = conformal_p_values(stream_scores, cal_scores)[:, 0]
    abs_td = td_errors_under_policy(stream, params_self, net.apply, hidden_dim, gamma)
    return p0, abs_td


def build_stream_signals(env, info, net, params_self, params_partner_post, hidden_dim,
                         pre_len, post_len, key, gamma=0.99):
    """``build_signals_jax`` returning numpy arrays + the change point (per-seed use)."""
    p0, abs_td = build_signals_jax(env, info, net, params_self, params_partner_post,
                                   hidden_dim, pre_len, post_len, key, gamma)
    return np.asarray(p0), np.asarray(abs_td), pre_len


def h0_signals_jax(env, info, net, params_self, hidden_dim, length, key, gamma=0.99):
    """A no-swap (H0) stream's dual signals ``(p0, abs_td)`` as **jax arrays** (vmappable).

    Identical pipeline to :func:`build_stream_signals` but the partner is *never* swapped
    (partner == self throughout), so any alarm is a false alarm. Pure jax (no ``np``
    conversion) so the false-alarm lab can ``jax.vmap`` it over many keys at once -- 200
    H0 streams cost ~one stream's compile + a batched execution instead of 200 Python
    iterations. ``cal_size = 4 * length`` calibration steps per stream.
    """
    k_cal, k_stream = jax.random.split(key)
    cal = collect_stream(env, info, net, params_self, params_self, k_cal,
                         4 * length, hidden_dim, BEHAVIOUR_EPS)
    cal_scores = policy_scores(cal, params_self, net.apply, SCORE_EPS, hidden_dim)
    stream = collect_stream(env, info, net, params_self, params_self, k_stream,
                           length, hidden_dim, BEHAVIOUR_EPS)
    stream_scores = policy_scores(stream, params_self, net.apply, SCORE_EPS, hidden_dim)
    p0 = conformal_p_values(stream_scores, cal_scores)[:, 0]
    abs_td = td_errors_under_policy(stream, params_self, net.apply, hidden_dim, gamma)
    return p0, abs_td


# ------------------------------------------------------------------ calibration (Simu)
def calibrate_on_simu(delta, n_h0, h0_len, seed):
    """Simultaneous-Attack false-alarm lab: match AES's threshold to conformal's rate.

    Trains a short Simu policy, generates ``n_h0`` no-swap H0 streams, and:

    * runs the conformal martingale (threshold ``1/delta``) on each -> measures its
      **achieved** H0 false-alarm rate ``fa_conf``;
    * builds the frozen AES reference (mean/std of the proxy) from a long H0 stream;
    * sweeps the AES z-threshold and picks the smallest threshold whose H0 false-alarm
      rate is ``<= fa_conf`` (matching the conformal detector's achieved rate, the
      like-for-like operating point).

    Returns a dict with the achieved rates, the chosen AES threshold, the AES reference
    stats, and the trained policy bundle (reused for the Overcooked run is *not* done --
    Overcooked retrains; this policy is Simu-only).
    """
    print("\n========== Simultaneous-Attack false-alarm lab ==========")
    env, _ = make_env("simu", num_agents=2)
    info = get_env_info(env)
    hd = 32
    net = QNetwork(action_dim=info.num_actions, hidden_dim=hd)
    cfg = short_iql_config("simu")

    t0 = time.perf_counter()
    out = jax.block_until_ready(jax.jit(make_train(cfg, env))(jax.random.PRNGKey(seed)))
    params_self = out["final_params"]
    print(f"  trained Simu IQL (greedy return {float(out['end_greedy_return']):.3f}) "
          f"in {time.perf_counter() - t0:.1f}s")

    # AES frozen reference from a long no-swap stream.
    _, td_ref = h0_signals_jax(
        env, info, net, params_self, hd, 4 * h0_len, jax.random.PRNGKey(seed + 99)
    )
    ref_mean, ref_std = reference_stats(td_ref)
    ref_mean, ref_std = float(ref_mean), float(ref_std)

    # Generate n_h0 independent H0 streams (vary the rollout key) -- vmapped over keys so
    # all streams compute in one batched pass (no Python loop over n_h0).
    t1 = time.perf_counter()
    h0_keys = jax.random.split(jax.random.PRNGKey(seed + 1000), n_h0)
    p0_arr, td_arr = jax.block_until_ready(
        jax.jit(jax.vmap(
            lambda k: h0_signals_jax(env, info, net, params_self, hd, h0_len, k)
        ))(h0_keys)
    )  # (n_h0, T) each
    print(f"  generated {n_h0} H0 streams (len {h0_len}) in {time.perf_counter() - t1:.1f}s")

    # Conformal achieved false-alarm (reset OFF so a crossing is a genuine first passage
    # over the whole H0 horizon -- the quantity Ville bounds).
    conf = jax.jit(jax.vmap(lambda p: run_martingale(p, delta=delta, reset_on_alarm=False)))(
        p0_arr
    )
    fa_conf = float(np.asarray(conf.crossed).mean())

    # Sweep AES threshold; measure H0 ever-cross fraction at each. The grid starts low
    # (z=0.3) so the curve captures AES's over-firing region -- on a clean H0 stream AES
    # false-alarms heavily at small z and only becomes safe past a (data-dependent) knee,
    # which is exactly the point: its safe threshold has to be *calibrated*, unlike the
    # conformal 1/delta which is fixed a priori.
    thr_grid = np.linspace(0.3, 10.0, 50)
    aes_fa = []
    for thr in thr_grid:
        det = jax.jit(jax.vmap(
            lambda x: aes_detector(x, ref_mean, ref_std, float(thr))  # noqa: B023
        ))(td_arr)
        aes_fa.append(float(np.asarray(det.crossed).mean()))
    aes_fa = np.array(aes_fa)

    # Matched operating point: share the same false-alarm BUDGET delta. Both detectors
    # are held to H0 false-alarm <= delta; we give AES the *most sensitive* threshold
    # consistent with that budget -- the smallest z whose H0 false-alarm is <= delta
    # (generous to the baseline: any larger z only slows it down). We target the nominal
    # delta rather than conformal's *achieved* rate, because Ville's bound is loose so the
    # conformal achieved rate is often ~0 at finite n, which would force AES to an
    # arbitrarily conservative threshold -- targeting the shared budget delta is the fair,
    # well-posed operating point. ``fa_conf`` is reported as a diagnostic (it confirms the
    # Ville bound holds: fa_conf <= delta).
    feasible = np.where(aes_fa <= delta + 1e-9)[0]
    if feasible.size > 0:
        matched_idx = int(feasible[0])  # smallest z meeting the budget -> most sensitive
    else:
        # Even the largest z can't get AES's H0 false-alarm down to delta -> use the
        # largest (most conservative) and report that AES could not meet the budget.
        matched_idx = int(len(thr_grid) - 1)
    aes_threshold = float(thr_grid[matched_idx])
    fa_aes = float(aes_fa[matched_idx])
    aes_threshold_nominal = aes_threshold  # kept for back-compat in the payload

    print(f"  conformal achieved H0 false-alarm (delta={delta}): {fa_conf:.4f} "
          f"(<= delta={delta}: {fa_conf <= delta})")
    print(f"  AES matched threshold z={aes_threshold:.2f} -> H0 false-alarm {fa_aes:.4f} "
          f"(shared budget delta={delta})")

    return {
        "delta": delta,
        "fa_conf": fa_conf,
        "fa_aes_matched": fa_aes,
        "aes_threshold": aes_threshold,
        "aes_threshold_nominal": aes_threshold_nominal,
        "aes_ref_mean": ref_mean,
        "aes_ref_std": ref_std,
        "thr_grid": thr_grid.tolist(),
        "aes_fa_curve": aes_fa.tolist(),
        "n_h0": n_h0,
        "h0_len": h0_len,
    }


# ------------------------------------------------------------ detection delay (per env)
def detection_delay_env(calib, delta, seeds, pre_len, post_len, env_name,
                        hidden_dim=None, label=None, **env_kwargs):
    """Per-seed detection delays for both detectors on a controlled partner swap.

    Env-agnostic core of the headline experiment (used for Overcooked *and* MPE
    ``simple_spread``). Uses the matched AES threshold/reference from the Simu calibration.
    For each seed: train a short policy on ``env_name``, swap the partner block at
    ``pre_len`` to the learner's least-likely action (reusing the M4 swap infra), build
    both detector signals on the *same* stream, and run both detectors at matched
    false-alarm. Also computes an env-native AES calibration (the fairest baseline).

    Returns a dict with per-seed delays (``-1`` == missed) and the raw signals of seed 0
    (for the figure). The ``layout``/``map_name`` etc. are forwarded via ``env_kwargs``.
    """
    label = label or env_name
    print(f"\n========== {label} detection delay over {len(seeds)} seeds ==========")
    ref_mean = calib["aes_ref_mean"]
    ref_std = calib["aes_ref_std"]
    aes_thr = calib["aes_threshold"]

    env, _ = make_env(env_name, **env_kwargs)
    info = get_env_info(env)
    cfg = short_iql_config(env_name)
    # The detection stream collector + scorer use the feed-forward ``QNetwork`` (single-step
    # forwards, no recurrent carry), so the trained policy must be FF too, and the scoring
    # net's hidden_dim MUST equal the trained config's. We derive ``hd`` from ``cfg`` (the
    # single source of truth) rather than a separately-passed value, so they can never
    # disagree -- the bug that bites when a caller's hidden_dim != the env's training config.
    cfg.net_type = "ff"
    if hidden_dim is not None:
        cfg.hidden_dim = hidden_dim  # let the caller pick the (shared) width if it wants
    hd = cfg.hidden_dim
    net = QNetwork(action_dim=info.num_actions, hidden_dim=hd)
    train = jax.jit(make_train(cfg, env))

    # --- env-native AES calibration (gives AES its BEST shot on this env) ---
    # Train one calibration policy, generate no-swap (H0) streams *on this env*, set the AES
    # reference + threshold to hit the shared budget delta on the env itself (not
    # transferred from Simu). Also measure the conformal detector's achieved H0 false-alarm
    # here, so the matched-budget claim is checked on this env too. This is the fairest
    # possible AES baseline: if it still loses on delay when natively calibrated, the
    # conformal advantage is robust; if it wins, the honest differentiator is the
    # guarantee/transfer, not raw delay.
    cal_seed = seeds[0]
    cal_params = jax.block_until_ready(train(jax.random.PRNGKey(cal_seed)))["final_params"]
    n_oc_h0 = 60
    h0_keys = jax.random.split(jax.random.PRNGKey(cal_seed + 7000), n_oc_h0)

    def _oc_h0(k):  # no-swap (partner == self) stream signals on this env
        return build_signals_jax(env, info, net, cal_params, cal_params, hd,
                                 pre_len, post_len, k)

    p0_h0, td_h0 = jax.block_until_ready(jax.jit(jax.vmap(_oc_h0))(h0_keys))
    oc_ref_mean, oc_ref_std = reference_stats(td_h0.reshape(-1))
    oc_ref_mean, oc_ref_std = float(oc_ref_mean), float(oc_ref_std)
    conf_h0 = jax.jit(jax.vmap(
        lambda p: run_martingale(p, delta=delta, reset_on_alarm=False)
    ))(p0_h0)
    conf_fa_oc = float(np.asarray(conf_h0.crossed).mean())
    # Sweep AES threshold on this env's H0 -> smallest z with FA <= delta (most sensitive).
    oc_thr_grid = np.linspace(0.3, 10.0, 50)
    oc_aes_fa = []
    for thr in oc_thr_grid:
        d = jax.jit(jax.vmap(
            lambda x: aes_detector(x, oc_ref_mean, oc_ref_std, float(thr))  # noqa: B023
        ))(td_h0)
        oc_aes_fa.append(float(np.asarray(d.crossed).mean()))
    oc_aes_fa = np.array(oc_aes_fa)
    feas = np.where(oc_aes_fa <= delta + 1e-9)[0]
    oc_aes_thr = float(oc_thr_grid[int(feas[0])] if feas.size else oc_thr_grid[-1])
    print(f"  {label}-native calibration: conformal H0 FA={conf_fa_oc:.4f} "
          f"(<= delta={delta}: {conf_fa_oc <= delta}); AES native z={oc_aes_thr:.2f} "
          f"(ref mean={oc_ref_mean:.3f} std={oc_ref_std:.3f})")

    conf_delays, aes_delays = [], []
    example = None
    for si, seed in enumerate(seeds):
        t0 = time.perf_counter()
        params_self = jax.block_until_ready(train(jax.random.PRNGKey(seed)))["final_params"]
        swap_action = least_likely_action(
            env, info, net, params_self, hd, jax.random.PRNGKey(seed + 5)
        )
        partner_post = make_constant_partner_params(
            net, hd, info.num_actions, info.obs_dim, info.num_agents, action=swap_action
        )
        p0, abs_td, cp = build_stream_signals(
            env, info, net, params_self, partner_post, hd, pre_len, post_len,
            jax.random.PRNGKey(seed + 1),
        )

        conf = run_martingale(jnp.asarray(p0), delta=delta, reset_on_alarm=True)
        conf_first = int(conf.first_alarm)
        conf_delay = conf_first - cp if conf_first >= cp else -1

        # AES: reference's frozen stats from Simu; building the AES reference from this
        # stream's *pre-swap* block would leak the env; we keep the Simu-calibrated frozen
        # reference (a detector tuned on the cheap lab, applied unchanged -- the honest
        # deployment story). But the TD scale differs across envs, so we ALSO compute an
        # env-local frozen reference from the pre-swap block and report both.
        det = aes_detector(jnp.asarray(abs_td), ref_mean, ref_std, aes_thr)
        aes_first = int(det.first_alarm)
        aes_delay = aes_first - cp if aes_first >= cp else -1

        # Env-local AES reference (pre-swap block) -> a *favourable* AES calibration (gives
        # AES the benefit of an in-distribution reference). Simu threshold reused.
        loc_mean, loc_std = reference_stats(jnp.asarray(abs_td[:pre_len]))
        det_loc = aes_detector(jnp.asarray(abs_td), float(loc_mean), float(loc_std), aes_thr)
        aes_first_loc = int(det_loc.first_alarm)
        aes_delay_loc = aes_first_loc - cp if aes_first_loc >= cp else -1

        # Env-NATIVE AES: reference + threshold both calibrated on this env's H0 (AES's best
        # possible operating point on this env -- the fairest baseline).
        det_nat = aes_detector(jnp.asarray(abs_td), oc_ref_mean, oc_ref_std, oc_aes_thr)
        aes_first_nat = int(det_nat.first_alarm)
        aes_delay_nat = aes_first_nat - cp if aes_first_nat >= cp else -1

        conf_delays.append(conf_delay)
        aes_delays.append((aes_delay, aes_delay_loc, aes_delay_nat))
        print(f"  seed {seed}: trained {time.perf_counter() - t0:.1f}s | swap@{cp} "
              f"swap_action={swap_action} | conformal delay={conf_delay} "
              f"AES(simu-ref) delay={aes_delay} AES(local-ref) delay={aes_delay_loc} "
              f"AES({label}-native) delay={aes_delay_nat}")

        if si == 0:
            example = {
                "p0": p0.tolist(),
                "abs_td": abs_td.tolist(),
                "cp": int(cp),
                "conf_log_m": np.asarray(conf.log_m).tolist(),
                "aes_stat_simu": np.asarray(det.statistic).tolist(),
                "aes_stat_local": np.asarray(det_loc.statistic).tolist(),
                "aes_stat_native": np.asarray(det_nat.statistic).tolist(),
            }

    return {
        "env": env_name,
        "label": label,
        "env_kwargs": {k: str(v) for k, v in env_kwargs.items()},
        "seeds": list(seeds),
        "conf_delays": conf_delays,
        "aes_delays_simu_ref": [d[0] for d in aes_delays],
        "aes_delays_local_ref": [d[1] for d in aes_delays],
        # Key kept as ``*_oc_native`` for the plotting code (env-native AES calibration);
        # despite the name it is whichever env this run targeted (Overcooked or MPE).
        "aes_delays_oc_native": [d[2] for d in aes_delays],
        "aes_threshold_simu": aes_thr,
        "aes_threshold_oc_native": oc_aes_thr,
        "conf_fa_overcooked": conf_fa_oc,
        "example": example,
    }


def detection_delay_overcooked(calib, delta, seeds, pre_len, post_len, layout="cramped_room"):
    """Back-compat wrapper: Overcooked partner-swap detection delay (see
    :func:`detection_delay_env`)."""
    return detection_delay_env(
        calib, delta, seeds, pre_len, post_len, env_name="overcooked",
        hidden_dim=64, label=f"Overcooked ({layout})", layout=layout,
    )


# --------------------------------------------------------------------------- plotting
def _delay_stats(delays):
    """(#detected, median, mean) over the detected (>=0) delays; (0, nan, nan) if none."""
    arr = np.array([d for d in delays if d >= 0], dtype=float)
    if arr.size == 0:
        return 0, float("nan"), float("nan")
    return arr.size, float(np.median(arr)), float(np.mean(arr))


def plot_benchmark(calib, oc, delta, out_png: Path):
    """Three-panel figure: AES false-alarm calibration, delay distributions, example traces."""
    fig = plt.figure(figsize=(11.0, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # (a) AES false-alarm vs threshold, with the matched operating point.
    ax = fig.add_subplot(gs[0, 0])
    thr = np.array(calib["thr_grid"])
    ax.plot(thr, calib["aes_fa_curve"], color="#C44E52", lw=1.8, label="AES H0 false-alarm")
    ax.axhline(calib["fa_conf"], color="#4C72B0", ls="--", lw=1.5,
               label=f"conformal achieved ({calib['fa_conf']:.3f})")
    ax.axhline(delta, color="gray", ls=":", lw=1.2, label=rf"nominal $\delta={delta}$")
    ax.axvline(calib["aes_threshold"], color="black", ls="-.", lw=1.2,
               label=f"matched z={calib['aes_threshold']:.1f}")
    ax.set_xlabel("AES detector z-threshold")
    ax.set_ylabel("H0 false-alarm rate (Simu)")
    ax.set_title("(a) Matched-false-alarm calibration\n(Simultaneous-Attack H0 lab)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)

    # (b) Detection-delay distributions on the heavy/standard env.
    env_label = oc["label"] if (oc is not None and "label" in oc) else "Overcooked"
    ax = fig.add_subplot(gs[0, 1])
    if oc is not None:
        conf_d = [d for d in oc["conf_delays"] if d >= 0]
        aes_d = [d for d in oc["aes_delays_simu_ref"] if d >= 0]
        aes_dl = [d for d in oc["aes_delays_local_ref"] if d >= 0]
        aes_dn = [d for d in oc["aes_delays_oc_native"] if d >= 0]
        n_seeds = len(oc["seeds"])
        data, labels, colors = [], [], []
        for series, lab, col in [
            (conf_d, f"conformal\n({len(conf_d)}/{n_seeds})", "#4C72B0"),
            (aes_dn, f"AES native\n({len(aes_dn)}/{n_seeds})", "#55A868"),
            (aes_dl, f"AES local\n({len(aes_dl)}/{n_seeds})", "#DD8452"),
            (aes_d, f"AES simu\n({len(aes_d)}/{n_seeds})", "#C44E52"),
        ]:
            # Always reserve a slot per detector (even with no detections) so the
            # miss rate is visible in the x-tick count; plot an empty box if none.
            data.append(series if series else [])
            labels.append(lab)
            colors.append(col)
        positions = np.arange(1, len(data) + 1)
        nonempty = [(p, d, c) for p, d, c in zip(positions, data, colors) if d]
        if nonempty:
            pos = [p for p, _, _ in nonempty]
            dd_ = [d for _, d, _ in nonempty]
            cc = [c for _, _, c in nonempty]
            try:
                bp = ax.boxplot(dd_, positions=pos, tick_labels=None,
                                patch_artist=True, showmeans=True)
            except TypeError:
                bp = ax.boxplot(dd_, positions=pos, patch_artist=True, showmeans=True)
            for patch, c in zip(bp["boxes"], cc):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("detection delay (steps after swap)  [#detected/seeds]")
        ax.set_title(f"(b) Detection delay, {env_label}\n(matched false-alarm, "
                     f"{len(oc['seeds'])} seeds)")
        ax.grid(alpha=0.25, axis="y")
    else:
        ax.text(0.5, 0.5, "Overcooked skipped", ha="center", va="center")
        ax.set_title("(b) Detection delay (skipped)")

    # (c) Example traces (seed 0): conformal log M_t and AES statistic vs step.
    ax = fig.add_subplot(gs[1, :])
    if oc is not None and oc["example"] is not None:
        ex = oc["example"]
        cp = ex["cp"]
        steps = np.arange(len(ex["conf_log_m"]))
        ax.plot(steps, np.array(ex["conf_log_m"]), color="#4C72B0", lw=1.6,
                label=r"conformal $\log M_t$")
        ax.axhline(np.log(1.0 / delta), color="#4C72B0", ls="--", lw=1.0,
                   label=r"conformal threshold $\log(1/\delta)$")
        ax2 = ax.twinx()
        ax2.plot(steps, np.array(ex["aes_stat_native"]), color="#55A868", lw=1.4, alpha=0.9,
                 label="AES statistic (oc-native)")
        ax2.axhline(oc["aes_threshold_oc_native"], color="#55A868", ls="--", lw=1.0,
                    label=f"AES native threshold z={oc['aes_threshold_oc_native']:.1f}")
        ax.axvline(cp, color="black", ls=":", lw=1.5, label=f"swap (step {cp})")
        ax.set_xlabel("stream step (incoming transition)")
        ax.set_ylabel(r"conformal $\log M_t$", color="#4C72B0")
        ax2.set_ylabel("AES z-statistic", color="#DD8452")
        ax.set_title("(c) Example seed-0 traces: conformal wealth vs AES proxy "
                     "(both at matched false-alarm)")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper left")
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "Overcooked skipped", ha="center", va="center")

    fig.suptitle(f"Anytime-valid conformal drift detection vs AES heuristic at matched "
                 f"false-alarm ({env_label})", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="detection-vs-AES benchmark (Simu lab + heavy env)")
    ap.add_argument("--delta", type=float, default=0.05, help="Ville level / target false-alarm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8, help="number of heavy-env seeds")
    ap.add_argument("--n-h0", type=int, default=200, help="H0 streams for the Simu lab")
    ap.add_argument("--h0-len", type=int, default=240, help="length of each H0 stream")
    ap.add_argument(
        "--env", choices=["overcooked", "mpe"], default="overcooked",
        help="heavy/standard env for the detection-delay panel (mpe == simple_spread)",
    )
    ap.add_argument("--no-heavy", "--no-overcooked", dest="no_heavy", action="store_true",
                    help="Simu false-alarm lab only (skip the heavy-env detection run)")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    calib = calibrate_on_simu(args.delta, args.n_h0, args.h0_len, args.seed)

    oc = None
    if not args.no_heavy:
        seeds = list(range(args.seed, args.seed + args.seeds))
        # post_len: an anytime-valid test legitimately accumulates evidence over a long
        # post-change window (its defining strength). Overcooked's weak per-step KL (near-
        # random co-players) needs a long window; MPE's non-trivial policy gives a much
        # stronger per-step signal, so a shorter window suffices there.
        if args.env == "mpe":
            oc = detection_delay_env(
                calib, args.delta, seeds, pre_len=80, post_len=200,
                env_name="MPE_simple_spread_v3", hidden_dim=128, label="MPE simple_spread",
            )
        else:
            oc = detection_delay_overcooked(
                calib, args.delta, seeds, pre_len=150, post_len=300
            )

    fig_name = "m8_detection_mpe.png" if args.env == "mpe" else "m5_detection_vs_aes.png"
    fig = RESULTS_DIR / fig_name
    plot_benchmark(calib, oc, args.delta, fig)
    print(f"\nfigure -> {fig}")

    json_name = "m8_detection_mpe.json" if args.env == "mpe" else "m5_detection_benchmark.json"
    payload = {"calibration": calib, "heavy": oc, "env": args.env, "delta": args.delta}
    (OUTPUTS_DIR / json_name).write_text(json.dumps(payload))
    print(f"raw numbers -> {OUTPUTS_DIR / json_name}")

    # Printed summary.
    print("\n================= SUMMARY =================")
    print(f"Simu false-alarm budget (delta={args.delta}):")
    print(f"  conformal achieved : {calib['fa_conf']:.4f}  (Ville bound delta={args.delta})")
    print(f"  AES matched (z={calib['aes_threshold']:.2f}) : {calib['fa_aes_matched']:.4f}")
    if oc is not None:
        nc, mc, ac = _delay_stats(oc["conf_delays"])
        na, ma, aa = _delay_stats(oc["aes_delays_simu_ref"])
        nl, ml, al = _delay_stats(oc["aes_delays_local_ref"])
        nn, mn, an = _delay_stats(oc["aes_delays_oc_native"])
        ns = len(oc["seeds"])
        lab = oc.get("label", "heavy env")
        print(f"{lab} H0 false-alarm: conformal {oc['conf_fa_overcooked']:.4f} "
              f"(<= delta={args.delta}); AES native threshold z={oc['aes_threshold_oc_native']:.2f}")
        print(f"{lab} detection delay over {ns} seeds (median / mean, #detected):")
        print(f"  conformal             : {mc:.1f} / {ac:.1f}  ({nc}/{ns} detected)")
        print(f"  AES (env-native cal)  : {mn:.1f} / {an:.1f}  ({nn}/{ns} detected)")
        print(f"  AES (local ref)       : {ml:.1f} / {al:.1f}  ({nl}/{ns} detected)")
        print(f"  AES (simu ref)        : {ma:.1f} / {aa:.1f}  ({na}/{ns} detected)")


if __name__ == "__main__":
    main()
