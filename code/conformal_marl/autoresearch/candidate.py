"""Editable baseline candidate for the automated-research loop.

The orchestrator may replace this file, but the API and every module under ``frozen``
are part of the benchmark contract and remain read-only during a search run.
"""

from __future__ import annotations

import numpy as np

from conformal_marl.autoresearch.api import Estimate, OperatorBatch


def estimate_operator(batch: OperatorBatch, rng: np.random.Generator) -> Estimate:
    """Uniform-replay baseline: ignore policy drift and average observed targets."""

    del rng
    return Estimate(value=float(np.mean(batch.outcomes)))
