"""Uniform replay must honour the trained batch size when the buffer draws more.

The selection samplers follow the LaBER recipe: draw ``L``, keep ``B``. Uniform
previously ignored its ``batch_size`` argument and trained on all ``L`` items, so
raising ``large_batch_size`` handed the uniform *control* ``L/B`` times more gradient
data than every selecting arm -- turning a replay-mechanism comparison into a
batch-size comparison, in the control's favour. That made ``large_batch_size``
unusable, which in turn left conformal selection inert everywhere it was configured.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.iql import IQLConfig
from conformal_marl.replay import Transition, empty_transition, make_replay_buffer
from conformal_marl.samplers import uniform_sampler

_N, _A, _D = 3, 2, 4


def _drawn(draw: int) -> Transition:
    """A drawn batch of ``draw`` items, each tagged with its position in the draw."""
    return Transition(
        obs=jnp.zeros((draw, _N, _D)),
        actions=jnp.zeros((draw, _N), jnp.int32),
        reward=jnp.tile(jnp.arange(draw, dtype=jnp.float32)[:, None], (1, _N)),
        done=jnp.zeros((draw, _N)),
        next_obs=jnp.zeros((draw, _N, _D)),
        avail_actions=jnp.ones((draw, _N, _A)),
        next_avail_actions=jnp.ones((draw, _N, _A)),
        behavior_log_prob=jnp.zeros((draw, _N)),
        hidden=jnp.zeros((draw, _N, 0)),
        next_hidden=jnp.zeros((draw, _N, 0)),
    )


def _buffer(draw: int):
    experience = _drawn(draw)

    def sample(state, rng):
        del state, rng
        return SimpleNamespace(experience=experience)

    return SimpleNamespace(sample=sample)


def _ids(batch: Transition) -> np.ndarray:
    return np.asarray(batch.reward[:, 0]).astype(int)


def test_uniform_keeps_the_first_trained_batch_from_a_large_draw():
    batch_size, draw = 64, 256
    sample = jax.jit(
        lambda key: uniform_sampler(_buffer(draw), None, key, None, batch_size)
    )
    batch, weights, _ = sample(jax.random.PRNGKey(0))

    assert batch.reward.shape[0] == batch_size
    assert weights.shape[0] == batch_size
    assert jnp.all(weights == 1.0)
    assert _ids(batch).tolist() == list(range(batch_size))


def test_uniform_prefix_reaches_all_stored_buffer_items():
    """The returned B positions remain iid uniform over stored transitions.

    Flashbax samples every position of its L draw independently with replacement. The
    test therefore checks reachability and balance over buffer *items*, not over the L
    positions of one synthetic draw.
    """
    item_count, batch_size, draw, repeats = 32, 8, 64, 400
    base = empty_transition(_N, _D, _A)
    stored = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (item_count,) + x.shape), base
    ).replace(
        reward=jnp.tile(jnp.arange(item_count, dtype=jnp.float32)[:, None], (1, _N))
    )
    buffer = make_replay_buffer(
        max_size=draw, min_size=item_count, sample_batch_size=draw
    )
    state = buffer.add(buffer.init(base), stored)
    sample = jax.jit(
        lambda key: uniform_sampler(buffer, state, key, None, batch_size)
    )
    counts = np.zeros(item_count)
    for key in range(repeats):
        batch, _, _ = sample(jax.random.PRNGKey(key))
        np.add.at(counts, _ids(batch), 1)

    expected = counts.sum() / item_count
    assert counts.min() > 0.0
    assert abs(counts.mean() - expected) < 1e-6
    # Per-cell count variation is O(sqrt(expected)); this loose deterministic bound
    # catches a materially skewed or unreachable buffer item without becoming flaky.
    assert counts.std() < 6.0 * np.sqrt(expected)


def test_uniform_is_unchanged_when_the_draw_equals_the_batch():
    """No subsample, no extra key consumption: existing single-draw runs are exact."""
    batch_size = 64
    buffer = _buffer(batch_size)
    key = jax.random.PRNGKey(7)

    batch, _, _ = jax.jit(
        lambda k: uniform_sampler(buffer, None, k, None, batch_size)
    )(key)
    plain = buffer.sample(None, key).experience
    assert jnp.array_equal(batch.reward, plain.reward)
    # Order preserved too, so nothing downstream sees a gratuitous permutation.
    assert _ids(batch).tolist() == list(range(batch_size))


def test_large_batch_size_makes_selection_expressible():
    """The config knob the selection samplers need must actually differ from B."""
    cfg = IQLConfig(batch_size=64, large_batch_size=256)
    assert cfg.draw_batch_size() == 256
    assert IQLConfig(batch_size=64).draw_batch_size() == 64


def test_shipped_learning_scenarios_actually_exercise_selection():
    """The learning tier must be able to express the mechanism it is studying.

    With ``L == B`` the conformal samplers' Gumbel-top-k degenerates to a permutation
    and ``support_floor`` defaults to 0, so every "selection" sampler silently reduces
    to reweighted uniform replay -- and the ratio-free and exact-ratio variants become
    indistinguishable. Both stages must configure a real draw-then-keep.
    """
    from conformal_marl.autoresearch.frozen.learning_benchmark import (
        DEFAULT_LEARNING_SCENARIOS,
        HELDOUT_LEARNING_SCENARIOS,
    )

    for scenario in DEFAULT_LEARNING_SCENARIOS + HELDOUT_LEARNING_SCENARIOS:
        overrides = scenario.config_overrides
        assert overrides["large_batch_size"] > overrides["batch_size"], scenario.name
        assert overrides["support_floor"] > 0.0, scenario.name
