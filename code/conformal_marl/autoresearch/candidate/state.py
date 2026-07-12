"""Candidate-owned immutable configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateConfig:
    support_floor: float = 0.05
    ratio_clip: float = 10.0
