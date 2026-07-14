"""Editable operator estimator; uniform replay is the baseline."""

from __future__ import annotations

import numpy as np

from conformal_marl.autoresearch.api import Estimate, OperatorBatch


def estimate_operator(batch: OperatorBatch, rng: np.random.Generator) -> Estimate:
    del rng
    return Estimate(value=float(np.mean(batch.outcomes)))
