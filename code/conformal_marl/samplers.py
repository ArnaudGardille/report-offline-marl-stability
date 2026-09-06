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
    # M6' support-floor: epsilon-mixing of the conformal *selection* distribution with a
    # uniform floor, used by the support-floored samplers below. ``support_floor in
    # [0, 1)``: the per-transition selection probability is mixed
    # ``(1 - f) * conformal + f * uniform``, so every in-buffer transition keeps
    # selection mass >= f / L. This is the constructive ``supp(d_w) >= supp(d^BR)``
    # guarantee of the support-starvation theorem (condition (b)). ``0.0`` -> the M3
    # samplers' pure top-p selection (no floor). Static (controls a traced computation
    # but is itself a Python float).
    support_floor: float = struct.field(pytree_node=False, default=0.0)


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

    Draws the buffer's ``sample_batch_size`` items ``L`` and, when ``L > batch_size``,
    keeps the first ``B`` items, weighting every kept sample by 1 so the TD loss is
    unchanged. Flashbax's item buffer draws each of the ``L`` positions independently
    and uniformly with replacement, so its first ``B`` positions are already an ordinary
    uniform replay batch; selecting them again would only add work.

    Returning exactly ``B`` items makes ``large_batch_size`` usable as an experimental knob.
    The selection samplers below follow the LaBER recipe -- draw ``L``, keep ``B`` --
    so if this one instead trained on all ``L`` items, raising ``L`` would hand the
    uniform control ``L/B`` times more gradient data than every selecting arm and
    quietly turn a replay-mechanism comparison into a batch-size comparison. Equal
    trained-batch size is what makes the arms comparable.

    When ``L == B`` the draw is returned untouched, on the original RNG stream, so
    existing single-draw results are bit-identical.
    """
    del ctx
    sample = buffer.sample(buffer_state, rng)
    batch: Transition = sample.experience
    drawn = batch.reward.shape[0]  # static under jit
    if drawn > batch_size:
        batch = jax.tree.map(lambda x: x[:batch_size], batch)
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


# ------------------------------------------------------------------------ M6'
# Support-floored conformal selection -- the constructive side of the
# support-starvation theorem. The M3 selectors above drive the all-attack mass to 0
# because those rare transitions get tiny p-values and the pure top-p Gumbel subsample
# never keeps them (support starvation, condition (b) of the theorem violated). These
# samplers add an epsilon-mixing **support floor**: the selection distribution is
# ``(1 - f) * softmax(conformal log-p) + f * uniform``, so every in-buffer transition --
# including all-attack -- retains selection probability >= f / L. That guarantees
# ``supp(d_w) >= supp(d_buffer) >= supp(d^BR)``.
#
# Condition (a) of the theorem -- weights must be the *exact* importance ratio rho -- is
# the second knob, the loss weighting:
#   * ``exact_ratio_floor``  : weights = rho_a = exp(-A^exact_a) (exact IS correction);
#   * ``policy_floor``       : weights = the ratio-free p-values (NOT rho).
# Together with the floor on/off this lets the demo pin the "iff": only
# exact-rho + floor recovers the best response; dropping either collapses.


def _floored_log_select_weights(
    log_p_L: chex.Array, floor: float
) -> chex.Array:
    """Mix a conformal log-selection score with a uniform floor, in log space.

    Given per-transition conformal log-scores ``log_p_L`` (here ``sum_a log p_i,a``),
    returns ``log( (1 - f) * softmax(log_p_L) + f * (1/L) )`` -- the log-weights whose
    Gumbel-top-k draws from the **epsilon-mixed** selection distribution. With ``f > 0``
    every transition has weight ``>= f / L`` so no transition can be starved out of the
    batch (the support-floor guarantee). ``f = 0`` returns ``log_p_L`` unchanged (a
    monotone shift of softmax has identical Gumbel-top-k law), i.e. the M3 behaviour.
    """
    if floor <= 0.0:
        return log_p_L
    length = log_p_L.shape[0]
    log_softmax_L = jax.nn.log_softmax(log_p_L)  # normalised conformal selection (log)
    log_uniform = -jnp.log(jnp.asarray(length, dtype=log_p_L.dtype))  # log(1/L)
    log_floor = jnp.log(jnp.asarray(floor, dtype=log_p_L.dtype)) + log_uniform
    log_keep = jnp.log1p(-jnp.asarray(floor, dtype=log_p_L.dtype)) + log_softmax_L
    # log( (1-f) softmax + f/L ) = logaddexp( log_keep , log_floor ).
    return jnp.logaddexp(log_keep, log_floor)


def _conformal_select_floor(
    buffer: Any,
    buffer_state: Any,
    rng: chex.PRNGKey,
    ctx: SamplerContext,
    batch_size: int,
    score_fn: Callable,
    rho_weights: bool,
) -> tuple[Transition, chex.Array, dict]:
    """Support-floored conformal selection with selectable loss weighting.

    Same LaBER skeleton as :func:`_conformal_select`, with two changes that realise the
    theorem's two conditions:

    * **Selection** uses :func:`_floored_log_select_weights` with ``ctx.support_floor``
      (condition (b): guaranteed support coverage).
    * **Weighting** is either the exact importance ratio ``rho_a = exp(-A^exact_a)``
      (``rho_weights=True``; condition (a)) or the ratio-free p-values
      (``rho_weights=False``; the "drop the exact ratio" ablation).
    """
    from conformal_marl.conformal import (
        conformal_p_values,
        exact_scores,
        gumbel_top_k,
        tree_index,
    )

    rng_draw, rng_sel = jax.random.split(rng)
    sample = buffer.sample(buffer_state, rng_draw)
    large: Transition = sample.experience  # (L, N, ...)

    params = ctx.online_params
    eps = ctx.epsilon
    scores_LN = score_fn(large, params, ctx.apply_fn, eps, ctx.hidden_dim)  # (L, N)
    cal_scores_nN = score_fn(ctx.cal_batch, params, ctx.apply_fn, eps, ctx.hidden_dim)
    p_LN = conformal_p_values(scores_LN, cal_scores_nN)  # (L, N) in (0, 1]

    log_p_L = jnp.sum(jnp.log(p_LN), axis=-1)  # (L,)
    log_sel_L = _floored_log_select_weights(log_p_L, ctx.support_floor)
    idx_B = gumbel_top_k(log_sel_L, batch_size, rng_sel)  # (B,) distinct indices

    batch: Transition = tree_index(large, idx_B)

    if rho_weights:
        # Exact importance ratio rho_a = exp(-A^exact_a) per agent, on the kept batch.
        # Reuse the exact score so this is exactly Foerster's other-agent ratio. Clip the
        # log to keep the weight finite (a stale transition can have rho ~ 0; a huge rho
        # is capped so a single sample cannot dominate the self-normalised loss).
        exact_LN = exact_scores(large, params, ctx.apply_fn, eps, ctx.hidden_dim)  # (L,N)
        rho_LN = jnp.exp(jnp.clip(-exact_LN, -20.0, 20.0))
        weights_BN = rho_LN[idx_B]  # (B, N)
    else:
        weights_BN = p_LN[idx_B]  # (B, N) ratio-free p-value weights

    info = {
        "p_values": p_LN,
        "selected_p": p_LN[idx_B],
        "mean_p": jnp.mean(p_LN),
    }
    return batch, weights_BN, info


@register_sampler("exact_ratio_floor")
def exact_ratio_floor_sampler(
    buffer: Any,
    buffer_state: Any,
    rng: chex.PRNGKey,
    ctx: SamplerContext,
    batch_size: int,
) -> tuple[Transition, chex.Array, dict]:
    """Exact-ratio score + support floor + exact-rho loss weights (both theorem conditions).

    The *recovering* configuration of the support-starvation theorem: conformal selection
    floored so the best-response support is never starved (condition (b)) **and** the loss
    reweighted by the exact importance ratio ``rho`` (condition (a)). The demo shows this
    recovers the true best response on Simultaneous-Attack (return -> 1).
    """
    from conformal_marl.conformal import exact_scores

    return _conformal_select_floor(
        buffer, buffer_state, rng, ctx, batch_size, exact_scores, rho_weights=True
    )


@register_sampler("policy_floor")
def policy_floor_sampler(
    buffer: Any,
    buffer_state: Any,
    rng: chex.PRNGKey,
    ctx: SamplerContext,
    batch_size: int,
) -> tuple[Transition, chex.Array, dict]:
    """Ratio-free score + support floor, but ratio-free (NOT rho) loss weights.

    The "drop the exact ratio" ablation: the support floor is present (condition (b)
    holds) but the weights are the ratio-free p-values, so condition (a) fails. Per the
    theorem this should still miss the true best response -- isolating that the support
    floor alone is not sufficient; the exact ratio is also required.
    """
    from conformal_marl.conformal import policy_scores

    return _conformal_select_floor(
        buffer, buffer_state, rng, ctx, batch_size, policy_scores, rho_weights=False
    )


# --------------------------------------------------------------- baseline integrity
#
# Automated-search candidate code is imported into *this* interpreter, so the module
# it edits and the registry the benchmark scores against share an address space. The
# built-in samplers are what every candidate is measured *against*: if a candidate can
# rebind one of them, it moves its own control arm and the reported delta is
# meaningless.
#
# Sealing a snapshot here -- at the end of this module's body, before any candidate
# module has had a chance to run -- gives the benchmark something trustworthy to
# compare against later. ``register_sampler`` refuses duplicate names, but that guards
# only the decorator; a direct ``_SAMPLER_REGISTRY[name] = ...`` assignment does not
# go through it.
#
# This is a *correctness* guard against accident and casual tampering, not a security
# boundary: code sharing an interpreter can defeat any in-process check, and a
# determined candidate still has the whole runtime. Real isolation needs an OS-level
# sandbox. What it buys is that the common failure -- a candidate that shadows a
# baseline name and silently scores itself against itself -- is caught loudly instead
# of producing a plausible wrong number.
_TRUSTED_BASELINES: dict[str, Sampler] = dict(_SAMPLER_REGISTRY)


def trusted_baselines() -> list[str]:
    """Names sealed before any candidate module could be imported."""
    return sorted(_TRUSTED_BASELINES)


def assert_baseline_integrity(*names: str) -> None:
    """Raise if a sealed baseline no longer resolves to the function sealed here.

    Call this immediately before *using* a baseline as a control arm, not only at
    import time: the registry is consulted at call time, so an import-time check alone
    is a snapshot that says nothing about what a later lookup will return.
    """
    checked = names or tuple(_TRUSTED_BASELINES)
    for name in checked:
        if name not in _TRUSTED_BASELINES:
            raise RuntimeError(f"{name!r} is not a sealed baseline sampler")
        if _SAMPLER_REGISTRY.get(name) is not _TRUSTED_BASELINES[name]:
            raise RuntimeError(
                f"the control sampler {name!r} is no longer the frozen implementation "
                "sealed at import; refusing to report a score against a modified "
                "baseline"
            )
