"""Reward-correctness and interface tests for the Simultaneous-Attack environment.

The reward formula is the crux of the environment (it is what makes a uniformly
random joint policy worth exactly zero), so these tests pin it down hard:

* an exact, library-independent reference reward computed with
  :class:`fractions.Fraction`;
* enumerated agreement between that reference and the *environment's* reward over
  every joint action;
* an exact ``E_uniform[r] == 0`` check via rational arithmetic, plus a confirmation
  that averaging the env's own rewards reproduces it;
* the all-attack (``+1``) and all-abstain (``0``) corner cases; and
* JaxMARL interface / jit / vmap sanity checks.
"""

import itertools
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conformal_marl.envs.simultaneous_attack import (
    ABSTAIN,
    ATTACK,
    SimultaneousAttack,
    default_r_death,
)


# --------------------------------------------------------------------------- helpers
def reference_r_death(num_agents: int) -> Fraction:
    """Exact rational death penalty: 1 / (N * (2**(N-1) - 1))."""
    return Fraction(1, num_agents * (2 ** (num_agents - 1) - 1))


def reference_reward(action_tuple, num_agents: int) -> Fraction:
    """Exact rational team reward for a joint action, straight from the spec."""
    k = sum(action_tuple)
    if k == num_agents:
        return Fraction(1)
    if k == 0:
        return Fraction(0)
    return -reference_r_death(num_agents) * k


def env_reward(env: SimultaneousAttack, action_tuple) -> float:
    """Shared reward the *environment* assigns to a joint action (via ``step``)."""
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    actions = {a: jnp.array(action_tuple[i]) for i, a in enumerate(env.agents)}
    _, _, rewards, _, _ = env.step(key, state, actions)
    # Reward is shared; every agent must see exactly the same value.
    vals = [float(rewards[a]) for a in env.agents]
    assert all(v == vals[0] for v in vals), f"reward not shared across agents: {vals}"
    return vals[0]


def all_joint_actions(num_agents: int):
    """All ``2**N`` joint actions as tuples of {ABSTAIN, ATTACK}."""
    return itertools.product((ABSTAIN, ATTACK), repeat=num_agents)


# --------------------------------------------------------------------- reward tests
@pytest.mark.parametrize("num_agents", [2, 3, 4])
def test_enumerated_reward_matches_formula(num_agents):
    """For every joint action, the env reward equals the exact spec reward."""
    env = SimultaneousAttack(num_agents=num_agents)
    # r_death itself must match the derived formula.
    assert env.r_death == pytest.approx(float(reference_r_death(num_agents)), rel=0, abs=1e-12)

    for action_tuple in all_joint_actions(num_agents):
        expected = float(reference_reward(action_tuple, num_agents))
        got = env_reward(env, action_tuple)
        assert got == pytest.approx(expected, rel=0, abs=1e-6), (
            f"N={num_agents} actions={action_tuple}: env={got} expected={expected}"
        )


@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_uniform_expectation_is_exactly_zero(num_agents):
    """E[r] under the uniform joint policy is exactly 0 (exact + env-based)."""
    # (a) Exact rational expectation over the spec reward.
    total = sum(
        reference_reward(a, num_agents) for a in all_joint_actions(num_agents)
    )
    expectation = total / Fraction(2**num_agents)
    assert expectation == Fraction(0), (
        f"N={num_agents}: exact E[r]={expectation} != 0"
    )

    # (b) The environment's own rewards must average to (floating-point) zero too.
    env = SimultaneousAttack(num_agents=num_agents)
    env_total = sum(env_reward(env, a) for a in all_joint_actions(num_agents))
    env_mean = env_total / (2**num_agents)
    assert env_mean == pytest.approx(0.0, abs=1e-7), (
        f"N={num_agents}: env mean reward={env_mean} != 0"
    )


@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_all_attack_is_one_and_all_abstain_is_zero(num_agents):
    """All-attack pays +1; all-abstain pays 0."""
    env = SimultaneousAttack(num_agents=num_agents)

    all_attack = tuple(ATTACK for _ in range(num_agents))
    all_abstain = tuple(ABSTAIN for _ in range(num_agents))

    assert env_reward(env, all_attack) == pytest.approx(1.0, abs=1e-6)
    assert env_reward(env, all_abstain) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("num_agents", [2, 3, 4])
def test_partial_attack_is_negative(num_agents):
    """Any partial attack (0 < K < N) yields a strictly negative reward."""
    env = SimultaneousAttack(num_agents=num_agents)
    for action_tuple in all_joint_actions(num_agents):
        k = sum(action_tuple)
        if 0 < k < num_agents:
            assert env_reward(env, action_tuple) < 0.0


def test_default_r_death_helper_matches_reference():
    """The exported ``default_r_death`` helper matches the rational reference."""
    for num_agents in [2, 3, 4, 5, 8]:
        assert default_r_death(num_agents) == pytest.approx(
            float(reference_r_death(num_agents)), rel=0, abs=1e-15
        )


def test_r_death_override_changes_reward_but_not_corners():
    """An explicit r_death override is honoured; corners stay at +1 / 0."""
    env = SimultaneousAttack(num_agents=3, r_death=0.25)
    assert env.r_death == 0.25
    # Single attacker (K=1) -> -r_death * 1.
    assert env_reward(env, (ATTACK, ABSTAIN, ABSTAIN)) == pytest.approx(-0.25, abs=1e-6)
    # Corners are independent of r_death.
    assert env_reward(env, (ATTACK, ATTACK, ATTACK)) == pytest.approx(1.0, abs=1e-6)
    assert env_reward(env, (ABSTAIN, ABSTAIN, ABSTAIN)) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ interface tests
@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_reset_returns_per_agent_obs(num_agents):
    """reset returns a per-agent obs dict covering exactly the env's agents."""
    env = SimultaneousAttack(num_agents=num_agents)
    obs, _ = env.reset(jax.random.PRNGKey(0))

    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(env.agents)
    assert len(env.agents) == num_agents
    for i, agent in enumerate(env.agents):
        assert obs[agent].shape == (num_agents,)
        # Documented obs design: a fixed one-hot of the agent index.
        np.testing.assert_allclose(np.asarray(obs[agent]), np.eye(num_agents)[i])


@pytest.mark.parametrize("num_agents", [2, 3, 4])
def test_step_sets_all_done_flags(num_agents):
    """Every step is terminal: all per-agent dones and ``__all__`` are True."""
    env = SimultaneousAttack(num_agents=num_agents)
    key = jax.random.PRNGKey(1)
    _, state = env.reset(key)
    actions = {a: jnp.array(ATTACK) for a in env.agents}

    _, _, _, dones, info = env.step(key, state, actions)

    assert "__all__" in dones
    assert bool(dones["__all__"]) is True
    for agent in env.agents:
        assert bool(dones[agent]) is True
    assert "num_attackers" in info  # info is well-formed


def test_action_and_observation_spaces():
    """Action space is Discrete(2); observation space matches the one-hot shape."""
    env = SimultaneousAttack(num_agents=3)
    for agent in env.agents:
        assert env.action_space(agent).n == 2
        assert env.observation_space(agent).shape == (3,)
    assert env.name == "SimultaneousAttack"


def test_step_is_jittable():
    """``jax.jit(env.step)`` runs and agrees with the eager reward."""
    env = SimultaneousAttack(num_agents=3)
    key = jax.random.PRNGKey(2)
    _, state = env.reset(key)
    actions = {a: jnp.array(ATTACK) for a in env.agents}

    jit_step = jax.jit(env.step)
    _, _, rewards, dones, _ = jit_step(key, state, actions)

    assert float(rewards["agent_0"]) == pytest.approx(1.0, abs=1e-6)
    assert bool(dones["__all__"]) is True


def test_reset_and_step_are_vmappable():
    """vmap of reset+step over a batch of keys gives batched outputs of right shape."""
    num_agents = 3
    batch = 8
    env = SimultaneousAttack(num_agents=num_agents)
    keys = jax.random.split(jax.random.PRNGKey(0), batch)

    # Batched reset.
    obs, states = jax.vmap(env.reset)(keys)
    for agent in env.agents:
        assert obs[agent].shape == (batch, num_agents)
    assert states.done.shape == (batch,)

    # Batched step with a per-batch joint action (here: everyone attacks).
    batched_actions = {
        a: jnp.full((batch,), ATTACK, dtype=jnp.int32) for a in env.agents
    }
    obs2, states2, rewards, dones, _ = jax.vmap(env.step)(keys, states, batched_actions)

    for agent in env.agents:
        assert obs2[agent].shape == (batch, num_agents)
        assert rewards[agent].shape == (batch,)
        assert dones[agent].shape == (batch,)
    assert dones["__all__"].shape == (batch,)
    # All-attack across the whole batch -> reward +1 everywhere.
    np.testing.assert_allclose(np.asarray(rewards["agent_0"]), np.ones(batch), atol=1e-6)
    assert bool(jnp.all(dones["__all__"])) is True


def test_vmap_reward_varies_with_batched_actions():
    """A batch mixing all-attack / all-abstain / partial yields the right rewards."""
    num_agents = 3
    env = SimultaneousAttack(num_agents=num_agents)
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    _, states = jax.vmap(env.reset)(keys)

    # Row 0: all attack (+1). Row 1: all abstain (0). Row 2: one attacker (-r_death).
    a0 = jnp.array([ATTACK, ABSTAIN, ATTACK])
    a1 = jnp.array([ATTACK, ABSTAIN, ABSTAIN])
    a2 = jnp.array([ATTACK, ABSTAIN, ABSTAIN])
    batched_actions = {"agent_0": a0, "agent_1": a1, "agent_2": a2}

    _, _, rewards, _, _ = jax.vmap(env.step)(keys, states, batched_actions)
    got = np.asarray(rewards["agent_0"])
    expected = np.array([1.0, 0.0, -float(reference_r_death(num_agents))])
    np.testing.assert_allclose(got, expected, atol=1e-6)
