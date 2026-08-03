"""Acceptance tests for the central JaxMARL env factory (M7, P1).

Covers, for every *in-scope* discrete env the factory exposes:

* **instantiation** -- ``make_env(name)`` builds a wrapped env and an :class:`EnvSpec`
  whose shapes agree with the IQL loop's own ``get_env_info`` (so the net is sized
  right); and
* **the IQL loop runs on it** -- a few jitted ``make_train`` steps on a tiny CI budget,
  asserting a finite loss. This is the "every env instantiates and the loop runs" gate.

The lighter envs run ``make_train`` directly (each a few seconds). The two heavyweight
SMAX scenarios (large obs, slowest to compile) are always *instantiated and
shape-checked* in CI; their short ``make_train`` run is marked ``slow`` (opt in with
``pytest -m slow``) so the default suite stays snappy -- the milestone explicitly allows
slow-skipping the heavy ones as long as they still instantiate.

The known-incompatible envs (Coin-Game and the N-player STORM: non-per-agent
reward/obs APIs) are asserted to raise a clear ``NotImplementedError`` rather than fail
deep inside training.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from conformal_marl.envs.registry import (
    INCOMPATIBLE_ENVS,
    AvailActionsWrapper,
    EnvSpec,
    FlatObs,
    StormAdapter,
    available_envs,
    make_env,
)
from conformal_marl.iql import get_env_info, make_train

# In-scope envs split by cost. The light ones run make_train in the default suite; the
# SMAX scenarios are instantiate-checked always and run only under ``-m slow``.
LIGHT_ENVS = [
    "simu",
    "MPE_simple_spread_v3",
    "MPE_simple_reference_v3",
    "overcooked",
    "hanabi",
    "switch_riddle",
    "storm_2p",
]
SMAX_ENVS = ["SMAX", "HeuristicEnemySMAX"]


def _tiny_overrides() -> dict:
    """A minimal CI budget: a handful of jitted steps, just to prove the loop runs."""
    return dict(
        total_timesteps=64,
        num_envs=2,
        num_steps=4,
        num_grad_steps=2,
        learning_starts=8,
        batch_size=8,
        buffer_size=256,
        cal_size=8,
        eval_episodes=2,
        eval_steps=8,
        epsilon_anneal_time=32,
    )


def _env_kwargs(name: str) -> dict:
    return {"map_name": "3m"} if "SMAX" in name else {}


def _assert_runs(name: str, env, spec: EnvSpec) -> None:
    """A tiny jitted make_train run on ``env`` asserts a finite loss + spec/info agreement."""
    info = get_env_info(env)
    assert info.num_agents == spec.num_agents
    assert info.num_actions == spec.num_actions
    assert info.obs_dim == spec.obs_dim, (
        f"{name}: EnvSpec.obs_dim {spec.obs_dim} disagrees with get_env_info {info.obs_dim}"
    )
    # The default config's net type matches the partial-observability flag.
    expected_net = "gru" if spec.partially_observable else "ff"
    assert spec.default_config.net_type == expected_net

    out = jax.block_until_ready(jax.jit(make_train(spec.default_config, env))(jax.random.PRNGKey(0)))
    loss = np.asarray(out["loss"])
    assert loss.shape[0] == spec.default_config.num_updates()
    assert np.isfinite(loss).all(), f"{name}: make_train produced a non-finite loss"
    # Greedy-return scalars are finite too (the eval path ran on this env).
    assert np.isfinite(float(out["start_greedy_return"]))
    assert np.isfinite(float(out["end_greedy_return"]))


@pytest.mark.parametrize("name", LIGHT_ENVS)
def test_make_env_instantiates_and_runs(name: str):
    """Each light in-scope env instantiates and the jitted IQL loop runs on it."""
    env, spec = make_env(name, config_overrides=_tiny_overrides(), **_env_kwargs(name))
    _assert_runs(name, env, spec)


@pytest.mark.parametrize("name", SMAX_ENVS)
def test_smax_instantiates(name: str):
    """SMAX scenarios instantiate and expose the masked, GRU-defaulted spec (always run).

    Cheap: builds the env (with the ``scenario=map_name_to_scenario(...)`` path the
    factory discovered) and checks shapes/flags without compiling a training run.
    """
    env, spec = make_env(name, **_env_kwargs(name))
    info = get_env_info(env)
    assert info.num_agents == spec.num_agents > 0
    assert info.obs_dim == spec.obs_dim > 0
    assert spec.partially_observable and spec.default_config.net_type == "gru"
    assert spec.masks_actions, "SMAX should expose action masking (get_avail_actions)"


@pytest.mark.slow
@pytest.mark.parametrize("name", SMAX_ENVS)
def test_smax_make_train_runs(name: str):
    """The IQL loop runs on SMAX too (slow: large obs / compile). Opt in with -m slow."""
    env, spec = make_env(name, config_overrides=_tiny_overrides(), **_env_kwargs(name))
    _assert_runs(name, env, spec)


@pytest.mark.parametrize("name", INCOMPATIBLE_ENVS)
def test_incompatible_envs_raise(name: str):
    """Known-incompatible envs fail at the factory boundary, not in the IQL loop."""
    with pytest.raises(NotImplementedError):
        make_env(name)


def test_unknown_env_raises():
    """An unregistered name raises a clear KeyError listing the available envs."""
    with pytest.raises(KeyError):
        make_env("not_a_real_env_v9")


def test_available_envs_lists_targets():
    """The registry advertises the targeted discrete catalogue."""
    envs = available_envs()
    for expected in [
        "simu", "MPE_simple_spread_v3", "SMAX", "overcooked", "hanabi",
        "switch_riddle", "storm_2p",
    ]:
        assert expected in envs


def test_storm_adapter_exposes_per_agent_contract():
    """STORM's tuples become flat per-agent dicts keyed by string agent name."""
    env, spec = make_env("storm_2p", num_inner_steps=8)
    assert isinstance(env, StormAdapter)
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert set(obs) == set(env.agents)
    assert all(value.shape == (spec.obs_dim,) for value in obs.values())
    assert all(value.dtype == np.float32 for value in obs.values())

    actions = {agent: np.int32(0) for agent in env.agents}
    next_obs, _, rewards, dones, _ = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )
    assert set(next_obs) == set(env.agents)
    assert set(rewards) == set(env.agents)
    assert set(dones) == {*env.agents, "__all__"}
    # String keys are load-bearing: JaxMARL's auto-reset tree-flattens the done dict,
    # and STORM's native int ids cannot be sorted alongside the "__all__" key.
    assert all(isinstance(agent, str) for agent in env.agents)


def test_storm_payoff_is_symmetric_and_common():
    """The factory's payoff matrix really does make the game common-payoff.

    Stepping from ``reset`` cannot show this: both inventories are empty, so STORM
    masks the reward to 0.0 for *any* payoff matrix (including the general-sum
    Prisoner's Dilemma default) and an equality assertion there is vacuous. Score
    non-empty inventories directly instead.
    """
    env, _ = make_env("storm_2p", num_inner_steps=8)
    _, state = env.reset(jax.random.PRNGKey(0))

    for red, blue in [((2, 0), (2, 0)), ((0, 2), (0, 2)), ((2, 0), (0, 2))]:
        scored = state.replace(
            red_inventory=jnp.asarray(red, dtype=jnp.float32),
            blue_inventory=jnp.asarray(blue, dtype=jnp.float32),
        )
        r_red, r_blue = env.get_reward(scored)
        assert float(r_red) == pytest.approx(float(r_blue)), (red, blue)

    def payoff(red, blue) -> float:
        scored = state.replace(
            red_inventory=jnp.asarray(red, dtype=jnp.float32),
            blue_inventory=jnp.asarray(blue, dtype=jnp.float32),
        )
        return float(env.get_reward(scored)[0])

    # Stag-Hunt shape: coordinating on the first coin is optimal, the second is a
    # worse equilibrium, and miscoordination pays less than either.
    assert payoff((1, 0), (1, 0)) == pytest.approx(4.0)
    assert payoff((0, 1), (0, 1)) == pytest.approx(2.0)
    assert payoff((1, 0), (0, 1)) == pytest.approx(0.0)


def test_nplayer_storm_and_wrong_agent_count_raise_at_the_factory():
    """Neither miswiring may fail deep inside the adapter or make_train."""
    # JaxMARL's N-player "storm" is a different (batched-dict) obs API entirely.
    with pytest.raises(NotImplementedError, match="storm_2p"):
        make_env("storm")
    # InTheGrid_2p accepts num_agents but always yields the red/blue pair, so
    # anything else used to surface as a bare IndexError inside _observations.
    with pytest.raises(NotImplementedError, match="2-player"):
        make_env("storm_2p", num_agents=3)


def test_config_overrides_applied():
    """``config_overrides`` mutate the default config; unknown fields raise."""
    _, spec = make_env("simu", config_overrides={"total_timesteps": 1234, "lr": 0.123})
    assert spec.default_config.total_timesteps == 1234
    assert spec.default_config.lr == pytest.approx(0.123)
    with pytest.raises(AttributeError):
        make_env("simu", config_overrides={"not_a_field": 1})


def test_flatobs_flattens_and_casts():
    """:class:`FlatObs` turns a multi-dim / integer obs into a flat float32 vector.

    Uses Overcooked (image obs) through the factory and checks the wrapped obs is 1-D
    float32 with the spec's obs_dim, and that masking is *not* spuriously exposed (the
    underlying Overcooked has no get_avail_actions).
    """
    env, spec = make_env("overcooked")
    assert isinstance(env, FlatObs)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    a0 = env.agents[0]
    assert obs[a0].ndim == 1 and obs[a0].shape[0] == spec.obs_dim
    assert obs[a0].dtype == np.float32
    # Overcooked is unmasked -> FlatObs must not advertise get_avail_actions.
    assert not hasattr(env, "get_avail_actions")
    assert not spec.masks_actions


def test_hanabi_exposes_masking():
    """Hanabi's ``get_legal_moves`` is surfaced as ``get_avail_actions`` (per-agent mask)."""
    env, spec = make_env("hanabi")
    assert isinstance(env, AvailActionsWrapper)
    assert hasattr(env, "get_avail_actions") and spec.masks_actions
    _, state = env.reset(jax.random.PRNGKey(0))
    avail = env.get_avail_actions(state)
    a0 = env.agents[0]
    assert avail[a0].shape == (spec.num_actions,)
