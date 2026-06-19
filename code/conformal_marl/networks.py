"""Recurrent-ready Q-network interface for the IQL backbone.

The whole point of this module is the **interface**, not the (currently trivial)
recurrence. Every network here speaks the JaxMARL/PureJaxRL recurrent contract::

    apply(params, carry, (obs, resets)) -> (carry, q_values)
    Network.initialize_carry(hidden_dim, *batch_dims) -> carry

so that the M2 training loop threads a hidden-state ``carry`` through its
``lax.scan`` from day one. For M2 the environments (Simultaneous-Attack,
MPE simple_spread) are effectively memoryless, so the default cell is a
**feed-forward MLP** whose carry is a zero-sized placeholder that is returned
unchanged. M5 (SMAX / Overcooked, partially observable) can then swap in
:class:`GRUQNetwork` -- which has the *same* signature and a real carry -- without
restructuring the scan loop.

Design choices
--------------
* ``(obs, resets)`` is the recurrent input tuple. ``resets`` is the per-step
  ``done`` flag; a real recurrent cell uses it to zero its hidden state at
  episode boundaries (see :class:`GRUQNetwork`). The feed-forward cell ignores
  it. Carrying it now means the loop never has to change shape later.
* Inputs are ``(time, batch, ...)`` -- time on axis 0 -- as in the reference
  JaxMARL ``ScannedRNN``. During env stepping we pass ``time == 1`` and let the
  training loop's own ``lax.scan`` provide the time recursion.
* ``initialize_carry`` is a ``@staticmethod`` taking ``hidden_dim`` then any
  number of batch dims, exactly like JaxMARL's ``ScannedRNN.initialize_carry``.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal


class QNetwork(nn.Module):
    """Feed-forward Q-network exposing the recurrent ``(carry, x) -> (carry, q)`` API.

    The carry is a zero-width placeholder (shape ``(*batch, 0)``) threaded
    unchanged through time, so this drops into a recurrent ``lax.scan`` loop while
    doing no actual recurrence. ``hidden_dim`` sizes the MLP trunk.

    Parameters
    ----------
    action_dim:
        Number of discrete actions (the Q-head width).
    hidden_dim:
        Width of the MLP hidden layers.
    init_scale:
        Orthogonal-init gain for the Dense layers.
    """

    action_dim: int
    hidden_dim: int = 64
    init_scale: float = 1.0

    @nn.compact
    def __call__(
        self, carry: chex.Array, x: tuple[chex.Array, chex.Array]
    ) -> tuple[chex.Array, chex.Array]:
        obs, _resets = x  # resets unused by the feed-forward cell (kept for the API)
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        h = nn.relu(h)
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(h)
        h = nn.relu(h)
        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(h)
        # Carry is returned unchanged: no memory in the feed-forward cell.
        return carry, q_vals

    @staticmethod
    def initialize_carry(hidden_dim: int, *batch_dims: int) -> chex.Array:
        """Zero-width placeholder carry of shape ``(*batch_dims, 0)``.

        ``hidden_dim`` is accepted (and ignored) so this matches the recurrent
        signature; the feed-forward cell has no hidden state to carry.
        """
        del hidden_dim
        return jnp.zeros((*batch_dims, 0))


class GRUQNetwork(nn.Module):
    """GRU-based Q-network with the identical recurrent API (for M5 drop-in).

    Not used in M2's default config, but committed now to *prove* the interface is
    genuinely recurrence-ready: same ``(carry, (obs, resets)) -> (carry, q)``
    signature, same ``initialize_carry(hidden_dim, *batch)`` shape contract. The
    ``resets`` flag zeroes the hidden state at episode boundaries, following the
    JaxMARL ``ScannedRNN`` pattern.

    Implementation note: time is unrolled with a **Python loop over the (static)
    time axis**, applying a single ``nn.GRUCell`` submodule each step, rather than
    Flax's ``nn.scan``. On the project's pinned stack (jax 0.4.30 / flax 0.10.4)
    ``nn.scan`` calls ``jax.api_util.debug_info`` / ``lu.wrap_init(..., debug_info=)``
    which do not exist in jax 0.4.30 -- so the reference JaxMARL ``ScannedRNN``
    (which uses ``nn.scan``) crashes here. A static Python unroll keeps the
    submodule call inside the module trace (no tracer leak) and is equivalent. M2's
    loop calls the net with ``time == 1`` (one GRU step per env step, recursion
    handled by the loop's own ``lax.scan``), so the unroll is length 1 in practice;
    longer sequences (BPTT) also work, just unrolled. Verified by the network tests.
    """

    action_dim: int
    hidden_dim: int = 64
    init_scale: float = 1.0

    @nn.compact
    def __call__(
        self, carry: chex.Array, x: tuple[chex.Array, chex.Array]
    ) -> tuple[chex.Array, chex.Array]:
        obs, resets = x  # obs: (time, batch, obs_dim); resets: (time, batch)

        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        cell = nn.GRUCell(self.hidden_dim)
        time_steps = embedding.shape[0]  # static (Python int) under jit
        outs = []
        for t in range(time_steps):
            # Zero the hidden state wherever an episode boundary occurred.
            carry = jnp.where(
                resets[t][..., np.newaxis],
                GRUQNetwork.initialize_carry(self.hidden_dim, *embedding.shape[1:-1]),
                carry,
            )
            carry, y = cell(carry, embedding[t])
            outs.append(y)
        hidden_seq = jnp.stack(outs, axis=0)  # (time, batch, hidden)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(hidden_seq)
        return carry, q_vals

    @staticmethod
    def initialize_carry(hidden_dim: int, *batch_dims: int) -> chex.Array:
        return nn.GRUCell(hidden_dim, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, hidden_dim)
        )
