"""Swappable replay-sampler seam -- the extension point M3 builds on.

A *sampler* turns a buffer state into a training batch plus per-sample weights::

    sampler(buffer, buffer_state, rng, ctx, batch_size) -> (batch, weights, info)

* ``batch`` is a :class:`~conformal_marl.replay.Transition` whose leading axis is
  the **trained** batch size ``B`` (``B <= L`` where ``L`` is what the buffer was
  configured to draw -- this is the LaBER large-batch-then-subsample seam).
* ``weights`` is a ``(B,)`` array of per-sample loss weights. The TD loss
  multiplies the per-sample squared error by these (``uniform`` -> all ones, so
  the loss is unchanged).
* ``info`` is an arbitrary dict for diagnostics (e.g. effective sample size).

Everything a future score needs is passed through :class:`SamplerContext`
(current online params, target params, the network ``apply`` fn, env metadata),
so M3 can register ``conformal_policy`` / ``exact_ratio`` samplers and have them
slot into the *unchanged* training loop simply by name. The contract is
intentionally functional: a sampler takes the flashbax buffer object plus its
state and returns plain arrays, so it composes under ``jit``/``vmap`` and ``scan``.

To add a sampler in M3::

    @register_sampler("conformal_policy")
    def conformal_policy_sampler(buffer, buffer_state, rng, ctx, batch_size):
        sample = buffer.sample(buffer_state, rng)        # draws L items
        scores = compute_conformal_scores(sample.experience, ctx)  # uses ctx.*
        idx = top_b_by_score(scores, batch_size)         # subsample to B
        batch = tree_index(sample.experience, idx)
        weights = ...                                    # selection/IS weights
        return batch, weights, {"scores": scores}

No edit to ``iql.py`` is required: the loop looks the sampler up by name.
"""

from __future__ import annotations

from typing import Any, Callable

import chex
import jax.numpy as jnp
from flax import struct

from conformal_marl.replay import Transition

# A sampler: (buffer, buffer_state, rng, ctx, batch_size) -> (batch, weights, info)
Sampler = Callable[
    [Any, Any, chex.PRNGKey, "SamplerContext", int],
    tuple[Transition, chex.Array, dict],
]

_SAMPLER_REGISTRY: dict[str, Sampler] = {}


@struct.dataclass
class SamplerContext:
    """Everything a sampler/score might need, threaded from the training loop.

    Carried as a flax struct so it survives ``jit``/``scan`` as a pytree. Arrays
    (params) are dynamic leaves; the network apply fn and integer env metadata are
    marked ``pytree_node=False`` so they are treated as static.
    """

    online_params: Any  # current online Q-network params
    target_params: Any  # target Q-network params
    apply_fn: Callable = struct.field(pytree_node=False)  # network apply (batched fwd)
    num_agents: int = struct.field(pytree_node=False)
    num_actions: int = struct.field(pytree_node=False)
    hidden_dim: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False, default=0.99)


def register_sampler(name: str) -> Callable[[Sampler], Sampler]:
    """Decorator: register ``fn`` under ``name`` in the global sampler registry."""

    def deco(fn: Sampler) -> Sampler:
        if name in _SAMPLER_REGISTRY:
            raise ValueError(f"sampler {name!r} already registered")
        _SAMPLER_REGISTRY[name] = fn
        return fn

    return deco


def get_sampler(name: str) -> Sampler:
    """Look up a registered sampler by name (raises if unknown)."""
    try:
        return _SAMPLER_REGISTRY[name]
    except KeyError as exc:
        known = sorted(_SAMPLER_REGISTRY)
        raise KeyError(f"unknown sampler {name!r}; registered: {known}") from exc


def list_samplers() -> list[str]:
    """Names of all registered samplers."""
    return sorted(_SAMPLER_REGISTRY)


@register_sampler("uniform")
def uniform_sampler(
    buffer: Any,
    buffer_state: Any,
    rng: chex.PRNGKey,
    ctx: SamplerContext,
    batch_size: int,
) -> tuple[Transition, chex.Array, dict]:
    """Plain uniform sampling: flashbax uniform draw, unit weights.

    Draws ``buffer.sample_batch_size`` items (here configured to equal the trained
    batch size, so no subsampling happens) and weights every sample by 1, leaving
    the TD loss unchanged. ``ctx`` is unused but part of the contract so M3's
    samplers are signature-compatible.
    """
    del ctx, batch_size
    sample = buffer.sample(buffer_state, rng)
    batch: Transition = sample.experience
    weights = jnp.ones((batch.reward.shape[0],), dtype=jnp.float32)
    return batch, weights, {}
