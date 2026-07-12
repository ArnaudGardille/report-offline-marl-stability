"""Editable JAX replay sampler; uniform replay is the baseline."""

from __future__ import annotations

from conformal_marl.samplers import SamplerContext, register_sampler, uniform_sampler


@register_sampler("autoresearch_candidate")
def sample_replay(buffer, buffer_state, rng, ctx: SamplerContext, batch_size: int):
    return uniform_sampler(buffer, buffer_state, rng, ctx, batch_size)
