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
import jax
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
    (params, the current epsilon, the calibration batch) are dynamic leaves; the
    network apply fn and integer env metadata are marked ``pytree_node=False`` so
    they are treated as static.

    The two M3 additions (both optional so the M2 ``uniform`` path is unchanged):

    * ``epsilon`` -- the *current* (annealed) epsilon-greedy exploration rate. The
      conformal score's ``pi_cur`` is the epsilon-greedy policy under the online net
      with *this* epsilon, so it must be threaded from the loop (it is not otherwise
      recoverable inside a sampler). ``None`` for samplers that do not need it.
    * ``cal_batch`` -- a :class:`Transition` batch of *fresh* on-policy transitions
      held out from the buffer, used as the inductive-conformal calibration set ``C``.
      Its leading axis is the calibration size ``n``. ``None`` for ``uniform``.
    """

    online_params: Any  # current online Q-network params
    target_params: Any  # target Q-network params
    apply_fn: Callable = struct.field(pytree_node=False)  # network apply (batched fwd)
    num_agents: int = struct.field(pytree_node=False)
    num_actions: int = struct.field(pytree_node=False)
    hidden_dim: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False, default=0.99)
    epsilon: Any = None  # current annealed eps-greedy rate (scalar array) or None
    cal_batch: Any = None  # fresh held-out calibration Transition batch (n, ...) or None


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


# --------------------------------------------------------------------------- M3
# Conformal selection samplers. Both follow the LaBER recipe: the buffer is
# configured to draw a *large* batch L uniformly; we score those L items per agent,
# turn the scores into inductive-conformal p-values against a fresh calibration set
# (threaded via ``ctx.cal_batch``), then subsample down to the trained size B with a
# bias toward conforming (large-p) transitions via Gumbel-top-k. The trained batch
# carries **per-agent weights** ``(B, N)`` -- each agent weights its own action-column
# by its own p-value, which is the principled per-agent design (agent a should
# down-weight transitions whose *other-agent* actions u_{-a} are unlikely under the
# present policies, independently of what the other agents are doing). The IQL loss
# accepts either ``(B,)`` or ``(B, N)`` weights (see ``iql._loss_fn``).


def _conformal_select(
    buffer: Any,
    buffer_state: Any,
    rng: chex.PRNGKey,
    ctx: SamplerContext,
    batch_size: int,
    score_fn: Callable,
) -> tuple[Transition, chex.Array, dict]:
    """Shared LaBER-style conformal selection used by both M3 samplers.

    ``score_fn(batch, params, apply_fn, epsilon, hidden_dim) -> (L, N)`` is the
    non-conformity score (ratio-free or exact). Steps:

    1. Draw a large batch ``L`` uniformly from the buffer.
    2. Score it per agent (``L, N``) under the *online* params + current epsilon.
    3. Score the calibration set the same way; p-values ``p_i in (0, 1]`` per agent.
    4. Subsample to ``B`` without replacement via Gumbel-top-k on the per-transition
       conformity ``sum_a log p_i,a`` (selecting transitions the present could
       plausibly have produced), and weight each kept agent-column by its p-value.
    """
    # Local import to avoid a module-load cycle (conformal imports replay only).
    from conformal_marl.conformal import (
        conformal_p_values,
        gumbel_top_k,
        tree_index,
    )

    rng_draw, rng_sel = jax.random.split(rng)
    sample = buffer.sample(buffer_state, rng_draw)
    large: Transition = sample.experience  # (L, N, ...)

    params = ctx.online_params
    eps = ctx.epsilon
    scores_LN = score_fn(large, params, ctx.apply_fn, eps, ctx.hidden_dim)  # (L, N)
    cal_scores_nN = score_fn(
        ctx.cal_batch, params, ctx.apply_fn, eps, ctx.hidden_dim
    )  # (n, N)

    p_LN = conformal_p_values(scores_LN, cal_scores_nN)  # (L, N) in (0, 1]

    # Per-transition conformity for the subsample selection: sum of per-agent log-p
    # (== log of the product of agent p-values). Higher => more conforming overall.
    log_p_L = jnp.sum(jnp.log(p_LN), axis=-1)  # (L,)
    idx_B = gumbel_top_k(log_p_L, batch_size, rng_sel)  # (B,) distinct indices

    batch: Transition = tree_index(large, idx_B)
    weights_BN = p_LN[idx_B]  # (B, N) : each agent weights its own column by its p

    info = {
        "p_values": p_LN,  # (L, N) full p-values over the large batch (diagnostic)
        "selected_p": weights_BN,  # (B, N) p-values of the kept transitions
        "mean_p": jnp.mean(p_LN),
    }
    return batch, weights_BN, info


@register_sampler("conformal_policy")
def conformal_policy_sampler(
    buffer: Any,
    buffer_state: Any,
    rng: chex.PRNGKey,
    ctx: SamplerContext,
    batch_size: int,
) -> tuple[Transition, chex.Array, dict]:
    """Ratio-free conformal selection (``A^policy``): no ``pi_past``, masking-safe.

    The differentiator sampler (exercised hard at M5): uses only the current policy's
    other-agent log-probs, so it needs no behaviour density and never blows up on
    support gaps.
    """
    from conformal_marl.conformal import policy_scores

    return _conformal_select(
        buffer, buffer_state, rng, ctx, batch_size, policy_scores
    )


@register_sampler("exact_ratio")
def exact_ratio_sampler(
    buffer: Any,
    buffer_state: Any,
    rng: chex.PRNGKey,
    ctx: SamplerContext,
    batch_size: int,
) -> tuple[Transition, chex.Array, dict]:
    """Exact importance-ratio conformal selection (``A^exact``).

    Recovers Foerster's other-agent importance ratio ``rho_a`` as the score
    ``A = -log rho_a`` (uses the stored ``behavior_log_prob`` for ``pi_past``), then
    the same p-value + Gumbel-top-k subsample as :func:`conformal_policy_sampler`.
    """
    from conformal_marl.conformal import exact_scores

    return _conformal_select(
        buffer, buffer_state, rng, ctx, batch_size, exact_scores
    )
