"""Transition schema and flashbax replay buffer for the IQL backbone.

The :class:`Transition` struct is deliberately **over-provisioned** for the later
milestones so that M3 (conformal selection) and M5 (recurrent / SMAX) never need
to change the on-disk buffer format:

* ``obs`` / ``next_obs`` -- per-agent observations, each ``(N, obs_dim)`` (agents
  on the leading axis, canonical agent order).
* ``actions`` -- the **joint** action ``(N,)``: every agent's action is stored, so
  any agent ``a`` can recover the others' actions ``u_{-a}`` (this is exactly what
  M3's conformal score over other agents' policies needs).
* ``reward`` -- per-agent reward ``(N,)`` (cooperative envs broadcast the team
  reward; kept per-agent for generality).
* ``done`` -- per-agent terminal flag ``(N,)`` for the transition.
* ``avail_actions`` -- per-agent available-action mask at ``obs`` ``(N, A)`` (all
  ones when the env has no masking). Stored so the masked target/greedy choice is
  reproducible from the buffer alone.
* ``behavior_log_prob`` -- per-agent log-prob of the action actually taken under
  the epsilon-greedy *behaviour* policy at collection time ``(N,)``. Cheap to log
  and required by M3's importance-ratio / conformal machinery (you cannot recover
  the behaviour policy after the fact, so it must be stored).

All fields are plain arrays with a leading agent axis, which keeps the struct a
flat pytree that flashbax and ``jax.vmap`` handle without special-casing. M5 will
add a recurrent-hidden-state field; because flashbax derives its layout from the
example transition, adding a field is backward compatible (no format migration).

The buffer itself is a flashbax **item buffer** (one stored item == one
:class:`Transition`). We store ``next_obs`` explicitly rather than relying on a
flat-buffer's implicit s/s' stitching, because (a) it makes every sampled item a
self-contained TD tuple -- the natural unit for M3's per-transition scoring -- and
(b) it sidesteps episode-boundary stitching bugs.
"""

from __future__ import annotations

from typing import Any

import chex
import jax.numpy as jnp
from flax import struct

import flashbax as fbx


@struct.dataclass
class Transition:
    """A single environment transition with the full joint action.

    Every field carries a leading agent axis of size ``N`` (canonical agent
    order). See the module docstring for the per-field contract and the rationale
    for each one's presence (forward-provisioning M3/M5).
    """

    obs: chex.Array  # (N, obs_dim)
    actions: chex.Array  # (N,) joint action -- all agents, so u_{-a} is recoverable
    reward: chex.Array  # (N,)
    done: chex.Array  # (N,)
    next_obs: chex.Array  # (N, obs_dim)
    avail_actions: chex.Array  # (N, A) available-action mask at `obs`
    behavior_log_prob: chex.Array  # (N,) log prob of `actions` under the behaviour policy


def make_replay_buffer(
    max_size: int,
    min_size: int,
    sample_batch_size: int,
) -> Any:
    """Create a flashbax item buffer over :class:`Transition` items.

    Parameters
    ----------
    max_size:
        Buffer capacity in transitions.
    min_size:
        Minimum number of stored transitions before sampling is allowed
        (``buffer.can_sample`` is False until then).
    sample_batch_size:
        Default number of items drawn per ``buffer.sample`` call. For the
        LaBER-style "draw L, train on B<=L" seam this is the **large** batch ``L``
        (the sampler decides the trained subset ``B``).

    Returns
    -------
    A flashbax ``TrajectoryBuffer`` configured in item mode. Its
    ``init / add / sample / can_sample`` are pure functions over a buffer state.
    ``add`` expects a *batch* of transitions (leading add-batch axis); ``sample``
    returns ``sample.experience`` with a leading ``sample_batch_size`` axis.
    """
    return fbx.make_item_buffer(
        max_length=max_size,
        min_length=min_size,
        sample_batch_size=sample_batch_size,
        add_batches=True,  # we add a batch of transitions per env step
        add_sequences=False,
    )


def empty_transition(num_agents: int, obs_dim: int, num_actions: int) -> Transition:
    """A zero-filled example transition, used to initialise the buffer layout.

    flashbax derives the stored item structure (shapes/dtypes) from this example,
    so it must match exactly what :func:`collect`-time code adds.
    """
    return Transition(
        obs=jnp.zeros((num_agents, obs_dim), dtype=jnp.float32),
        actions=jnp.zeros((num_agents,), dtype=jnp.int32),
        reward=jnp.zeros((num_agents,), dtype=jnp.float32),
        done=jnp.zeros((num_agents,), dtype=jnp.bool_),
        next_obs=jnp.zeros((num_agents, obs_dim), dtype=jnp.float32),
        avail_actions=jnp.ones((num_agents, num_actions), dtype=jnp.float32),
        behavior_log_prob=jnp.zeros((num_agents,), dtype=jnp.float32),
    )
