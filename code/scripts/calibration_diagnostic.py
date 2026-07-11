"""Conformal p-value calibration diagnostic (reviewer request).

Are the conformal p-values actually (super-)uniform in the nominal regime, and do they
degrade under drift? This is the calibration check that backs the validity claims of the
detector (Section 4.4) and the selection rule (Section 4.3): under H0 the inductive
p-values of Eq. (12), ranked against a *frozen* fresh calibration set, should be ~Uniform;
under a partner-policy swap they should concentrate near 0 (a drifted transition is
non-conforming -> large score -> small p).

We reuse the *validated* M4/M5 pipeline verbatim:
  * ``h0_signals_jax``    -- no-swap (H0) stream -> agent-0 conformal p-values;
  * ``build_signals_jax`` -- pre-then-post-swap stream -> agent-0 conformal p-values,
    with the post-swap block the controlled drift.
No new scoring/calibration code is introduced, so the numbers here are exactly the
p-values the detector consumes.

Outputs (gitignored ``code/`` dirs):
  * ``results/m_calibration_pvalues.png`` -- (a) QQ-plot under H0 vs Uniform,
    (b) histograms H0 vs post-swap;
  * ``outputs/m_calibration_pvalues.json`` -- the calibration statistics.

Run: ``uv run python scripts/calibration_diagnostic.py``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conformal_marl.envs.registry import make_env  # noqa: E402
from conformal_marl.iql import get_env_info, make_train  # noqa: E402
from conformal_marl.networks import QNetwork  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_demo import (  # noqa: E402
    least_likely_action,
    make_constant_partner_params,
    short_iql_config,
)
from detection_benchmark import build_signals_jax, h0_signals_jax  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
OUTPUTS_DIR = CODE_DIR / "outputs"


def _frac_le(x: np.ndarray, a: float) -> float:
    return float((x <= a).mean())


def _ks_uniform(x: np.ndarray) -> float:
    """One-sample KS distance of ``x`` to Uniform(0,1): ``sup_t |F_n(t) - t|``."""
    xs = np.sort(x)
    n = xs.size
    cdf_hi = np.arange(1, n + 1) / n
    cdf_lo = np.arange(0, n) / n
    return float(max(np.max(np.abs(cdf_hi - xs)), np.max(np.abs(cdf_lo - xs))))


def main() -> None:
    env_name = sys.argv[1] if len(sys.argv) > 1 else "simu"
    if env_name == "mpe":  # friendly alias, matching detection_benchmark
        env_name = "MPE_simple_spread_v3"
    delta = 0.05
    n_h0, h0_len = 200, 60
    n_swap, pre_len, post_len = 200, 60, 60
    seed = 0

    env, _ = make_env("simu", num_agents=2) if env_name == "simu" else make_env(env_name)
    info = get_env_info(env)
    cfg = short_iql_config(env_name)
    cfg.net_type = "ff"  # the stream collector/scorer use single-step FF forwards
    hd = cfg.hidden_dim  # single source of truth (32 for simu, 128 for MPE)
    net = QNetwork(action_dim=info.num_actions, hidden_dim=hd)
    print(f"=== calibration diagnostic on '{env_name}' "
          f"({info.num_agents} agents, {info.num_actions} actions, hd={hd}) ===")

    t0 = time.perf_counter()
    out = jax.block_until_ready(jax.jit(make_train(cfg, env))(jax.random.PRNGKey(seed)))
    params_self = out["final_params"]
    print(f"trained Simu IQL (greedy return {float(out['end_greedy_return']):.3f}) "
          f"in {time.perf_counter() - t0:.1f}s")

    # ---- H0 (no-swap) p-values: should be ~Uniform ----
    t1 = time.perf_counter()
    h0_keys = jax.random.split(jax.random.PRNGKey(seed + 1000), n_h0)
    p0_h0, _ = jax.block_until_ready(
        jax.jit(jax.vmap(lambda k: h0_signals_jax(env, info, net, params_self, hd, h0_len, k)))(h0_keys)
    )
    p_h0 = np.asarray(p0_h0).reshape(-1)
    print(f"generated {n_h0} H0 streams in {time.perf_counter() - t1:.1f}s")

    # ---- Drift (post-swap) p-values: should concentrate near 0 ----
    t2 = time.perf_counter()
    swap_action = least_likely_action(env, info, net, params_self, hd, jax.random.PRNGKey(seed + 5))
    partner_post = make_constant_partner_params(
        net, hd, info.num_actions, info.obs_dim, info.num_agents, action=swap_action
    )
    swap_keys = jax.random.split(jax.random.PRNGKey(seed + 2000), n_swap)
    p0_sw, _ = jax.block_until_ready(
        jax.jit(jax.vmap(
            lambda k: build_signals_jax(env, info, net, params_self, partner_post, hd,
                                        pre_len, post_len, k)
        ))(swap_keys)
    )
    p0_sw = np.asarray(p0_sw)  # (n_swap, pre_len + post_len)
    p_pre = p0_sw[:, :pre_len].reshape(-1)      # pre-swap block (exchangeable -> sanity ~Uniform)
    p_drift = p0_sw[:, pre_len:].reshape(-1)    # post-swap block (the drift)
    print(f"generated {n_swap} swap streams in {time.perf_counter() - t2:.1f}s")

    stats = {
        "delta": delta,
        "H0": {"mean": float(p_h0.mean()), "frac_le_0.05": _frac_le(p_h0, 0.05),
               "ks_uniform": _ks_uniform(p_h0), "n": int(p_h0.size)},
        "pre_swap": {"mean": float(p_pre.mean()), "frac_le_0.05": _frac_le(p_pre, 0.05),
                     "ks_uniform": _ks_uniform(p_pre), "n": int(p_pre.size)},
        "drift": {"mean": float(p_drift.mean()), "frac_le_0.05": _frac_le(p_drift, 0.05),
                  "n": int(p_drift.size)},
    }
    print(json.dumps(stats, indent=2))

    # ---- figure: (a) PP-plot (empirical CDF vs Uniform) -- "below diagonal == valid
    # super-uniform"; (b) histograms H0 vs drift. ----
    def _ecdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
        xs = np.sort(x)
        return np.searchsorted(xs, grid, side="right") / xs.size

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    u = np.linspace(0, 1, 200)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Uniform ($H_0$ target)")
    ax1.plot(u, _ecdf(p_h0, u), color="C0", lw=1.8,
             label=f"$H_0$  ($P(p\\leq 0.05)={stats['H0']['frac_le_0.05']:.3f}$)")
    ax1.plot(u, _ecdf(p_drift, u), color="C3", lw=1.8,
             label=f"drift ($P(p\\leq 0.05)={stats['drift']['frac_le_0.05']:.3f}$)")
    ax1.set_xlabel("level $u$")
    ax1.set_ylabel(r"empirical $P(p\leq u)$")
    ax1.set_title("(a) Validity: $H_0$ on/below diagonal")
    ax1.legend(loc="lower right", fontsize=7.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    bins = np.linspace(0, 1, 21)
    ax2.hist(p_h0, bins=bins, density=True, alpha=0.6, color="C0",
             label=f"$H_0$ (mean {stats['H0']['mean']:.2f})")
    ax2.hist(p_drift, bins=bins, density=True, alpha=0.6, color="C3",
             label=f"post-swap (mean {stats['drift']['mean']:.2f})")
    ax2.axhline(1.0, color="k", ls="--", lw=1, label="Uniform density")
    ax2.set_xlabel("p-value")
    ax2.set_ylabel("density")
    ax2.set_title("(b) $H_0$ piles at 1, drift at 0")
    ax2.legend(fontsize=7.5)
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_png = RESULTS_DIR / f"m_calibration_pvalues_{env_name}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    (OUTPUTS_DIR / f"m_calibration_pvalues_{env_name}.json").write_text(json.dumps(stats, indent=2))
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
