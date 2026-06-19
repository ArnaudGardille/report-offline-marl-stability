"""Tests for the recurrence-ready Q-network interface.

These pin down the *contract* that M5 relies on: both the feed-forward
:class:`QNetwork` (M2 default) and the :class:`GRUQNetwork` drop-in honour the
same ``apply(params, carry, (obs, resets)) -> (carry, q)`` signature and
``initialize_carry(hidden_dim, *batch)`` shape, run under ``jax.jit`` on the
pinned stack, and (for the GRU) actually use the carry + reset flag.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from conformal_marl.networks import GRUQNetwork, QNetwork


@pytest.mark.parametrize("Net", [QNetwork, GRUQNetwork])
def test_recurrent_api_shapes_and_jit(Net):
    """Both nets share the (carry, (obs, resets)) -> (carry, q) API under jit."""
    action_dim, hidden_dim, time, batch, obs_dim = 3, 8, 4, 5, 6
    net = Net(action_dim=action_dim, hidden_dim=hidden_dim)
    carry = Net.initialize_carry(hidden_dim, batch)
    x = (jnp.zeros((time, batch, obs_dim)), jnp.zeros((time, batch)))

    params = net.init(jax.random.PRNGKey(0), carry, x)
    new_carry, q = jax.jit(net.apply)(params, carry, x)

    assert q.shape == (time, batch, action_dim)
    # Carry keeps its (batch, *) leading shape (zero-width for the FF cell).
    assert new_carry.shape[0] == batch


def test_ff_carry_is_zero_width_and_unchanged():
    """The feed-forward carry is a zero-width placeholder threaded unchanged."""
    net = QNetwork(action_dim=2, hidden_dim=8)
    carry = QNetwork.initialize_carry(8, 5)
    assert carry.shape == (5, 0)
    x = (jnp.zeros((1, 5, 4)), jnp.zeros((1, 5)))
    params = net.init(jax.random.PRNGKey(0), carry, x)
    new_carry, _ = net.apply(params, carry, x)
    assert new_carry.shape == (5, 0)


def test_gru_carry_and_reset_semantics():
    """The GRU has a real carry and the reset flag zeroes the hidden state."""
    net = GRUQNetwork(action_dim=2, hidden_dim=8)
    fresh = GRUQNetwork.initialize_carry(8, 4)
    assert fresh.shape == (4, 8)

    params = net.init(jax.random.PRNGKey(0), fresh, (jnp.zeros((1, 4, 5)), jnp.zeros((1, 4))))

    # With reset=1, a non-zero incoming carry must be wiped -> same output as a
    # fresh-carry pass on the same input.
    nonzero = jnp.ones((4, 8))
    inp = jnp.ones((1, 4, 5))
    _, q_reset = net.apply(params, nonzero, (inp, jnp.ones((1, 4))))
    _, q_fresh = net.apply(params, jnp.zeros((4, 8)), (inp, jnp.zeros((1, 4))))
    assert bool(jnp.allclose(q_reset, q_fresh))

    # Without reset, a non-zero carry changes the output (the carry matters).
    _, q_noreset = net.apply(params, nonzero, (inp, jnp.zeros((1, 4))))
    assert not bool(jnp.allclose(q_noreset, q_fresh))
