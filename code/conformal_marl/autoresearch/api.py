"""Narrow API exposed to automatically generated replay candidates.

The first research tier estimates one current-policy perceived Bellman backup from
replay collected under a different teammate policy. Candidates receive observations,
the behaviour and target teammate policies, and a private random generator. They never
receive the true conditional means used by the frozen benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class OperatorBatch:
    """Observed replay for a single ``(state, ego action)`` Bellman cell."""

    teammate_actions: np.ndarray
    outcomes: np.ndarray
    behavior_probs: np.ndarray
    target_probs: np.ndarray

    def validate(self) -> None:
        actions = np.asarray(self.teammate_actions)
        outcomes = np.asarray(self.outcomes)
        behavior = np.asarray(self.behavior_probs)
        target = np.asarray(self.target_probs)

        if actions.ndim != 1 or outcomes.shape != actions.shape:
            raise ValueError("teammate_actions and outcomes must be aligned 1-D arrays")
        if behavior.ndim != 1 or target.shape != behavior.shape:
            raise ValueError("behavior_probs and target_probs must be aligned 1-D arrays")
        if len(actions) == 0 or len(behavior) == 0:
            raise ValueError("operator batches and action spaces must be non-empty")
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("teammate_actions must contain integer action indices")
        if np.any(actions < 0) or np.any(actions >= len(behavior)):
            raise ValueError("teammate action outside the declared action space")
        if not np.all(np.isfinite(outcomes)):
            raise ValueError("outcomes must be finite")
        if np.any(behavior <= 0.0):
            raise ValueError("behavior policy must have full support")
        if np.any(target < 0.0):
            raise ValueError("target probabilities must be non-negative")
        if not np.isclose(behavior.sum(), 1.0) or not np.isclose(target.sum(), 1.0):
            raise ValueError("behavior and target policies must each sum to one")

    def readonly(self) -> "OperatorBatch":
        """Return a defensive copy whose arrays cannot be modified by a candidate."""

        arrays = []
        for value in (
            self.teammate_actions,
            self.outcomes,
            self.behavior_probs,
            self.target_probs,
        ):
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            arrays.append(copied)
        return OperatorBatch(*arrays)


@dataclass(frozen=True)
class Estimate:
    """One scalar Bellman-backup estimate plus JSON-compatible diagnostics."""

    value: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("candidate returned a non-finite estimate")
        if any(not np.isfinite(value) for value in self.diagnostics.values()):
            raise ValueError("candidate returned non-finite diagnostics")


class OperatorEstimator(Protocol):
    """Callable implemented by ``candidate.estimate_operator``."""

    def __call__(self, batch: OperatorBatch, rng: np.random.Generator) -> Estimate: ...
