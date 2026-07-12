"""Editable target-teammate policy helpers.

The baseline does not alter teammate policies. Candidate samplers may import functions
from this module, but frozen training code never trusts values supplied from outside the
candidate package.
"""

from __future__ import annotations

import jax.numpy as jnp


def normalize_policy(probabilities):
    values = jnp.maximum(jnp.asarray(probabilities, dtype=jnp.float32), 0.0)
    return values / jnp.maximum(jnp.sum(values, axis=-1, keepdims=True), 1e-8)
