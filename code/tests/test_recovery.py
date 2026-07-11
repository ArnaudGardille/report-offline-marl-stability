"""Tests for the recovery experiment engine (conformal_marl/recovery.py).

Covers the schema-free age-tracking ring buffer, the staleness weighting, and a smoke run
of every arm end-to-end on the tiny Simultaneous-Attack env (no pretraining needed -- the
loop only needs *valid* params, not good ones, to exercise the machinery).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.controller import staleness_weight
from conformal_marl.envs.registry import make_env
from conformal_marl.iql import get_env_info
from conformal_marl.networks import get_network_cls
from conformal_marl.recovery import ARMS, RecoveryConfig, RingBuffer, build_network, run_arm


# --------------------------------------------------------------------------- ring buffer
def _chunk(t: int, n: int, obs_dim: int, num_actions: int) -> dict:
    return {
        "obs": np.ones((t, n, obs_dim), np.float32),
        "actions": np.zeros((t, n), np.int32),
        "reward": np.zeros((t, n), np.float32),
        "done": np.zeros((t, n), bool),
        "next_obs": np.ones((t, n, obs_dim), np.float32),
        "avail_actions": np.ones((t, n, num_actions), np.float32),
        "next_avail_actions": np.ones((t, n, num_actions), np.float32),
        "behavior_log_prob": np.zeros((t, n), np.float32),
    }


def test_ring_buffer_add_fill_and_insert_step():
    buf = RingBuffer.create(cap=10, n_agents=2, obs_dim=3, num_actions=2)
    assert buf.fill == 0 and buf.ptr == 0

    buf.add_chunk(_chunk(4, 2, 3, 2), base_step=0)
    assert buf.fill == 4 and buf.ptr == 4
    assert list(buf.insert_step[:4]) == [0, 1, 2, 3]

    # Overflow the ring: 4 + 8 = 12 into cap 10 -> wraps, fill clamps at cap, ptr = 2.
    buf.add_chunk(_chunk(8, 2, 3, 2), base_step=4)
    assert buf.fill == 10
    assert buf.ptr == 2
    # The two wrapped slots (indices 0,1) hold the newest insert steps (10, 11).
    assert buf.insert_step[0] == 10 and buf.insert_step[1] == 11


def test_ring_buffer_reset_flushes():
    buf = RingBuffer.create(cap=8, n_agents=2, obs_dim=3, num_actions=2)
    buf.add_chunk(_chunk(5, 2, 3, 2), base_step=0)
    assert buf.fill == 5
    buf.reset()
    assert buf.fill == 0 and buf.ptr == 0


def test_ring_buffer_sample_shapes_and_age():
    buf = RingBuffer.create(cap=64, n_agents=2, obs_dim=3, num_actions=2)
    buf.add_chunk(_chunk(40, 2, 3, 2), base_step=0)
    rng = np.random.default_rng(0)
    batch_g, insert_g = buf.sample_batch(rng, batch=8, grad_steps=3)
    assert batch_g["obs"].shape == (3, 8, 2, 3)
    assert batch_g["actions"].shape == (3, 8, 2)
    assert insert_g.shape == (3, 8)
    # Ages w.r.t. a current step are non-negative and bounded by the run length.
    ages = 40 - insert_g
    assert np.all(ages >= 0) and np.all(ages <= 40)


def test_staleness_weight_monotone():
    H = jnp.asarray(100.0)
    assert float(staleness_weight(jnp.asarray(0.0), H)) == 1.0
    assert abs(float(staleness_weight(jnp.asarray(100.0), H)) - 0.5) < 1e-5
    w = np.array(staleness_weight(jnp.asarray([0.0, 50.0, 100.0, 200.0]), H))
    assert np.all(np.diff(w) < 0)  # strictly decreasing in age


# --------------------------------------------------------------------------- smoke run
def _tiny_params(env, info, cfg):
    """A valid (untrained) FF param set, mirroring iql's init."""
    net_cls = get_network_cls("ff")
    network, _apply = build_network(info, cfg)
    init_obs = jnp.zeros((1, info.num_agents, info.obs_dim))
    init_resets = jnp.zeros((1, info.num_agents))
    init_carry = net_cls.initialize_carry(cfg.hidden_dim, info.num_agents)
    return network.init(jax.random.PRNGKey(0), init_carry, (init_obs, init_resets))


def test_run_arm_smoke_all_arms():
    env, _ = make_env("simu")
    info = get_env_info(env)
    cfg = RecoveryConfig(
        hidden_dim=16, chunk_len=8, pre_chunks=2, post_chunks=3, n_cal=16,
        grad_steps=2, batch_size=8, buffer_size=200, eval_episodes=4, eval_steps=8,
    )
    _net, apply = build_network(info, cfg)
    params_A = _tiny_params(env, info, cfg)
    params_B = _tiny_params(env, info, cfg)  # a *different* init -> a real "drift"

    n_chunks = cfg.pre_chunks + cfg.post_chunks
    for arm in ARMS:
        r = run_arm(arm, env, info, apply, params_A, params_B, cfg, seed=0)
        assert len(r["curve"]) == n_chunks
        assert np.all(np.isfinite(r["curve"]))
        assert r["swap_step"] == cfg.pre_chunks * cfg.chunk_len

    # oracle_reset always flushes (it knows the swap step).
    r_oracle = run_arm("oracle_reset", env, info, apply, params_A, params_B, cfg, seed=0)
    assert r_oracle["did_reset"] is True
