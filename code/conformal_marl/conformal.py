"""Conformal transition-selection machinery -- the scientific core of M3.

This module implements, as pure jittable functions over batched
:class:`~conformal_marl.replay.Transition` pytrees:

1. **Current epsilon-greedy log-probs** ``log pi_cur_j(u_j | s_j)`` under the
   *online* Q-network (:func:`current_log_probs`). For an available action ``u``::

       pi_cur_j(u) = eps / |avail_j|  +  (1 - eps) * 1[u == argmax_{avail} Q_j(s_j)]

   i.e. exactly the epsilon-greedy collection rule, evaluated with the present net
   and the present epsilon. (Masking is respected: unavailable actions get prob 0.)

2. **Non-conformity scores**, *per learning agent* ``a`` (Algorithm 1 is "per
   agent a"): a score grading how incompatible a stored transition is with the
   **other** agents' **current** policies. Large score == stale.

   * Exact (importance-ratio) score::

         A^exact_a(tau) = -log rho_a
                        = -sum_{j != a} log pi_cur_j(u_j) + sum_{j != a} log pi_past_j(u_j)

     uses the stored ``behavior_log_prob`` for ``pi_past`` and online params for
     ``pi_cur``. (Recovers Foerster et al.'s importance ratio.)

   * Ratio-free score::

         A^policy_a(tau) = -sum_{j != a} log pi_cur_j(u_j)

     no ``pi_past``, hence no support / blow-up problem -- the fallback that
     extends to masked / black-box settings.

   Both are returned as ``(B, N)`` arrays: entry ``[i, a]`` is agent ``a``'s score
   for transition ``i``, aggregated over the *other* agents ``j != a``.

3. **Inductive conformal p-values** (:func:`conformal_p_values`), per agent, via a
   vectorised rank against a held-out calibration set ``C`` of fresh scores::

       p_i = (1 + #{ j : A^cal_j >= A_i }) / (n + 1)   in (0, 1]

   Small p == more non-conforming == stale. Under H0 (the scored transition is
   exchangeable with calibration) ``p`` is (super-)uniform on ``(0, 1]``.

4. **Gumbel-top-k** (:func:`gumbel_top_k`): a jittable, fixed-size, without-
   replacement weighted subsample. Sampling ``k`` items with the Gumbel-top-k trick
   on ``log w_i`` draws (without replacement) proportionally to ``w_i``; we use it
   to subsample the large LaBER batch toward conformity (``w_i propto p_i``).

Everything is written against batched arrays with a leading transition axis ``B``
(or calibration axis ``n``) and an agent axis ``N``, so it composes under
``jax.jit`` / ``jax.vmap`` / ``lax.scan`` without special-casing. Scoring reuses the
network's feed-forward path exactly as the IQL loss does (one shared forward over
all ``(transition, agent)`` rows).
"""

from __future__ import annotations

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp

from conformal_marl.replay import Transition

# Large negative used to mask unavailable actions before an argmax (mirrors iql.py).
_MASKED_Q = -1e9
# Floor on probabilities before taking a log, so log-probs stay finite.
_PROB_EPS = 1e-8


def _forward_q_flat(
    apply_fn: Callable,
    params: Any,
    obs_MA: chex.Array,
    hidden_dim: int,
) -> chex.Array:
    """Feed-forward Q over a flat ``(M, obs_dim)`` batch -> ``(M, A)``.

    Mirrors the forward used by the IQL loss: a zero-width carry and zero resets,
    a unit time axis added then dropped. ``M`` is whatever flat row count the caller
    passes (typically ``B * N`` (transition, agent) rows).
    """
    from conformal_marl.networks import QNetwork

    m = obs_MA.shape[0]
    carry = QNetwork.initialize_carry(hidden_dim, m)
    obs_TMA = obs_MA[jnp.newaxis, :, :]  # (1, M, obs_dim)
    resets_TM = jnp.zeros((1, m))
    _, q_TMA = apply_fn(params, carry, (obs_TMA, resets_TM))
    return q_TMA[0]  # (M, A)


def current_log_probs(
    batch: Transition,
    params: Any,
    apply_fn: Callable,
    epsilon: chex.Array,
    hidden_dim: int,
) -> chex.Array:
    """Per-agent log-prob of each stored action under the *current* eps-greedy policy.

    For every transition ``i`` and agent ``j`` this evaluates ``log pi_cur_j(u_j | s_j)``
    where ``u_j = batch.actions[i, j]``, ``pi_cur`` is the epsilon-greedy policy under
    ``params`` (the online net) with the supplied ``epsilon`` and the stored
    available-action mask. Returns ``(B, N)``.

    The probability of the taken action is::

        p = (1 - eps) * 1[u == greedy] + eps * (1 / |avail|)

    restricted to available actions (an unavailable stored action would get prob 0,
    but the buffer never stores unavailable actions). Identical algebra to the
    behaviour log-prob computed at collection time, but with the *current* net.
    """
    b, n = batch.actions.shape
    obs_BN = batch.obs.reshape((b * n,) + batch.obs.shape[2:])  # (B*N, obs_dim)
    avail_BN = batch.avail_actions.reshape((b * n,) + batch.avail_actions.shape[2:])
    actions_BN = batch.actions.reshape((b * n,))

    q_BNA = _forward_q_flat(apply_fn, params, obs_BN, hidden_dim)

    # Masked greedy action per row.
    masked_q = jnp.where(avail_BN > 0, q_BNA, _MASKED_Q)
    greedy_BN = jnp.argmax(masked_q, axis=-1)

    n_avail_BN = jnp.sum(avail_BN > 0, axis=-1).astype(jnp.float32)
    is_greedy_BN = (actions_BN == greedy_BN).astype(jnp.float32)
    # Whether the stored action is itself available (defensive; normally always true).
    taken_avail_BN = jnp.take_along_axis(
        avail_BN, actions_BN[:, jnp.newaxis], axis=-1
    )[:, 0]
    p_BN = (1.0 - epsilon) * is_greedy_BN + epsilon * (1.0 / n_avail_BN)
    p_BN = jnp.where(taken_avail_BN > 0, p_BN, _PROB_EPS)
    log_p_BN = jnp.log(jnp.clip(p_BN, _PROB_EPS, 1.0))
    return log_p_BN.reshape((b, n))


def _others_sum(per_agent_BN: chex.Array) -> chex.Array:
    """Sum a per-agent quantity over the *other* agents: ``out[i, a] = sum_{j!=a} x[i, j]``.

    Computed as ``total_i - x[i, a]`` (the "leave-one-out" sum). Returns ``(B, N)``.
    """
    total_B1 = jnp.sum(per_agent_BN, axis=-1, keepdims=True)  # (B, 1)
    return total_B1 - per_agent_BN  # (B, N) : excludes own column a


def policy_scores(
    batch: Transition,
    params: Any,
    apply_fn: Callable,
    epsilon: chex.Array,
    hidden_dim: int,
) -> chex.Array:
    """Ratio-free non-conformity score ``A^policy_a`` per agent, shape ``(B, N)``.

    ``A^policy_a(tau) = -sum_{j != a} log pi_cur_j(u_j)``. No ``pi_past``: this is the
    fallback that needs no behaviour density and never blows up on support gaps.
    """
    log_cur_BN = current_log_probs(batch, params, apply_fn, epsilon, hidden_dim)
    return -_others_sum(log_cur_BN)


def exact_scores(
    batch: Transition,
    params: Any,
    apply_fn: Callable,
    epsilon: chex.Array,
    hidden_dim: int,
) -> chex.Array:
    """Exact importance-ratio non-conformity score ``A^exact_a`` per agent ``(B, N)``.

    ``A^exact_a(tau) = -log rho_a = -sum_{j!=a} log pi_cur_j(u_j) + sum_{j!=a} log pi_past_j(u_j)``,
    using the stored ``behavior_log_prob`` for ``pi_past`` (the eps-greedy density at
    collection time) and online params for ``pi_cur``. Recovers Foerster's ratio:
    ``rho_a = prod_{j!=a} pi_cur_j(u_j) / pi_past_j(u_j)``, so ``A = -log rho_a``.
    """
    log_cur_BN = current_log_probs(batch, params, apply_fn, epsilon, hidden_dim)
    log_past_BN = batch.behavior_log_prob  # (B, N)
    return _others_sum(log_past_BN) - _others_sum(log_cur_BN)


def conformal_p_values(
    scores_BN: chex.Array,
    cal_scores_nN: chex.Array,
) -> chex.Array:
    """Inductive conformal p-values, per agent, vectorised (no Python/loop).

    For each agent ``a`` and scored transition ``i`` with score ``A_i = scores_BN[i, a]``
    and calibration scores ``A^cal = cal_scores_nN[:, a]`` (held-out, fresh)::

        p_i = (1 + #{ j : A^cal_j >= A_i }) / (n + 1)   in (0, 1]

    Returns ``(B, N)`` with every entry in ``(0, 1]``. Larger score (more
    non-conforming / staler) -> more calibration scores below it -> smaller count of
    ``>=`` -> smaller p. Under H0 (scored ~ calibration, exchangeable) p is
    (super-)uniform.

    The ``>=`` (rather than ``>``) is the standard conservative convention that makes
    the bound hold with ties; ``+1`` in numerator and denominator counts the test
    point itself, guaranteeing ``p in (0, 1]`` and exact super-uniformity.
    """
    n = cal_scores_nN.shape[0]
    # Broadcast compare: (B, 1, N) vs (1, n, N) -> (B, n, N); count over the cal axis.
    ge_BnN = cal_scores_nN[jnp.newaxis, :, :] >= scores_BN[:, jnp.newaxis, :]
    count_BN = jnp.sum(ge_BnN, axis=1)  # (B, N) : #{ cal >= score } per agent
    return (1.0 + count_BN.astype(jnp.float32)) / (n + 1.0)


def gumbel_top_k(
    log_weights_B: chex.Array,
    k: int,
    rng: chex.PRNGKey,
) -> chex.Array:
    """Gumbel-top-k: indices of ``k`` items drawn *without replacement* ~ ``w_i``.

    Adds i.i.d. Gumbel noise to ``log_weights_B`` and takes the top ``k`` -- the
    classic Gumbel-top-k trick, which is *exactly* sequential sampling without
    replacement with probabilities proportional to ``w_i = exp(log_weights_B)``
    (Kool et al., 2019). Fixed output size ``k`` and fully jittable (no boolean
    masking / dynamic shapes).

    Parameters
    ----------
    log_weights_B:
        ``(B,)`` log-weights (unnormalised is fine). Use ``log p_i`` to bias toward
        conforming transitions.
    k:
        Number of indices to return (``k <= B``).
    rng:
        PRNG key for the Gumbel perturbation.

    Returns
    -------
    ``(k,)`` int array of distinct indices into ``[0, B)``.
    """
    b = log_weights_B.shape[0]
    gumbel = jax.random.gumbel(rng, (b,))
    perturbed = log_weights_B + gumbel
    # top-k indices (largest perturbed). jax.lax.top_k returns values, indices.
    _, idx = jax.lax.top_k(perturbed, k)
    return idx


def tree_index(tree: Any, idx: chex.Array) -> Any:
    """Index every leaf of a pytree along its leading axis by ``idx`` (a gather)."""
    return jax.tree_util.tree_map(lambda x: x[idx], tree)
