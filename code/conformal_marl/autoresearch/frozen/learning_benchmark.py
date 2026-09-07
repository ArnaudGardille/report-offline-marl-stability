"""Frozen paired-seed learning screen for replay candidates.

The screen asks two separate scientific questions. Cooperative STORM measures whether a
candidate improves learning under policy-induced replay mismatch. Simultaneous-Attack is
a support-preservation gate: replay selection is known to starve its rare optimal joint
action, so a candidate may not trade that failure for a STORM gain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from conformal_marl.envs.registry import make_env
from conformal_marl.iql import make_train
from conformal_marl.samplers import assert_baseline_integrity, get_sampler

# Score assigned to a candidate that regresses a support gate. Sorting by
# ``primary_score`` alone must rank such a candidate below *every* gate-passing one,
# including a passing candidate that merely scores badly -- a finite cap cannot
# promise that, so rejection is -inf.
REJECTED_SCORE = float("-inf")


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read-only view, so a candidate module cannot retune a frozen scenario."""
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class LearningScenario:
    """One immutable environment/configuration in the learning screen.

    ``frozen=True`` only protects the dataclass fields; the mappings themselves are
    wrapped in :class:`~types.MappingProxyType` so that candidate code -- which is
    imported into this same process -- cannot mutate the pinned budgets in place.
    """

    name: str
    env_name: str
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    return_scale: float = 1.0
    support_gate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_kwargs", _freeze(self.env_kwargs))
        object.__setattr__(self, "config_overrides", _freeze(self.config_overrides))

    def validate(self) -> None:
        if not self.name or self.return_scale <= 0.0:
            raise ValueError("learning scenarios need a name and positive return scale")
        if self.config_overrides.get("eval_during_training") is not True:
            raise ValueError("learning scenarios must record an evaluation curve")

    def fingerprint(self) -> str:
        """Stable hash of everything that defines this scenario's measurement.

        Recorded in the result so a reported score can be tied back to the exact
        budget it was produced under, rather than trusted on the module's word.
        """
        payload = json.dumps(
            {
                "name": self.name,
                "env_name": self.env_name,
                "env_kwargs": dict(self.env_kwargs),
                "config_overrides": dict(self.config_overrides),
                "return_scale": self.return_scale,
                "support_gate": self.support_gate,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_LEARNING_SCENARIOS = (
    LearningScenario(
        name="simultaneous_attack_support",
        env_name="simu",
        env_kwargs={"num_agents": 3},
        config_overrides={
            "total_timesteps": 8_192,
            "num_envs": 8,
            "num_steps": 8,
            "num_grad_steps": 4,
            "learning_starts": 128,
            "hidden_dim": 32,
            "batch_size": 64,
            "large_batch_size": 256,
            "support_floor": 0.05,
            "buffer_size": 8_192,
            "cal_size": 32,
            "epsilon_anneal_time": 4_096,
            "eval_episodes": 32,
            "eval_steps": 2,
            "eval_during_training": True,
        },
        return_scale=4.0,
        support_gate=True,
    ),
    # Fixed-design sensitivity check: at 16_384 steps none of three deliberately
    # diversity-collapsed STORM probes separated from equal-law replay and their
    # realized ordering was inverted. At 65_536 steps the ordering is monotone and
    # the two strongest degradations separate; the mildest still does not. See
    # ``scripts/screen_power_check.py``. The return divisor follows the observed
    # 2.445 curve maximum rather than the previously assumed converged-play 16.0.
    LearningScenario(
        name="storm_cooperative",
        env_name="storm_2p",
        env_kwargs={"num_inner_steps": 48},
        config_overrides={
            "total_timesteps": 65_536,
            "num_envs": 8,
            "num_steps": 32,
            "num_grad_steps": 4,
            "learning_starts": 256,
            "hidden_dim": 64,
            "batch_size": 64,
            "large_batch_size": 256,
            "support_floor": 0.05,
            "buffer_size": 65_536,
            "cal_size": 64,
            "epsilon_anneal_time": 10_000,
            "eval_episodes": 32,
            "eval_steps": 48,
            "eval_during_training": True,
        },
        return_scale=2.0,
    ),
)

# Confirmatory scenarios for the frozen finalist. These change the *task*, not just
# the random seed: a fourth Simultaneous-Attack agent (which enlarges the joint action
# space and makes the rare all-attack optimum exponentially rarer) and a longer STORM
# horizon.
#
# ``return_scale`` is deliberately not a free constant here. Both stages compare a
# normalized delta against the same ``minimum_score``, so any scale mismatch between
# search and held-out silently changes how strict the confirmatory gate is relative to
# the screen it confirms:
#
# * Simultaneous-Attack keeps ``4.0``. Its team reward is ``+1`` for the all-attack
#   joint action regardless of ``N`` (see :mod:`conformal_marl.envs.simultaneous_attack`),
#   so the achievable return does not change with the agent count -- measured mean
#   uniform return 0.982 at N=3 and 0.975 at N=4. An earlier revision of this scenario
#   used ``5.0``, which made the confirmatory support gate 25% more permissive in raw
#   return than the screening gate.
# * STORM uses the recalibrated search budget and scales its return divisor with the
#   horizon, because return is a sum over inner steps: ``2.0 * 64 / 48``. This
#   equalizes the mechanical part of the task shift; it is not an empirical
#   calibration of the held-out task, whose result remains confirmatory.
HELDOUT_LEARNING_SCENARIOS = (
    LearningScenario(
        name="simultaneous_attack_support_4_agents",
        env_name="simu",
        env_kwargs={"num_agents": 4},
        config_overrides={
            "total_timesteps": 8_192,
            "num_envs": 8,
            "num_steps": 8,
            "num_grad_steps": 4,
            "learning_starts": 128,
            "hidden_dim": 32,
            "batch_size": 64,
            "large_batch_size": 256,
            "support_floor": 0.05,
            "buffer_size": 8_192,
            "cal_size": 32,
            "epsilon_anneal_time": 4_096,
            "eval_episodes": 32,
            "eval_steps": 2,
            "eval_during_training": True,
        },
        return_scale=4.0,
        support_gate=True,
    ),
    LearningScenario(
        name="storm_cooperative_long_horizon",
        # JaxMARL's bare ``storm`` is the N-player InTheGrid, whose observation API
        # StormAdapter does not handle; ``storm_2p`` is the adapted cooperative game.
        env_name="storm_2p",
        env_kwargs={"num_inner_steps": 64},
        config_overrides={
            "total_timesteps": 65_536,
            "num_envs": 8,
            "num_steps": 32,
            "num_grad_steps": 4,
            "learning_starts": 256,
            "hidden_dim": 64,
            "batch_size": 64,
            "large_batch_size": 256,
            "support_floor": 0.05,
            "buffer_size": 65_536,
            "cal_size": 64,
            "epsilon_anneal_time": 10_000,
            "eval_episodes": 32,
            "eval_steps": 64,
            "eval_during_training": True,
        },
        return_scale=2.0 * 64 / 48,
    ),
)


def _run_sampler(scenario: LearningScenario, sampler_name: str, seeds: np.ndarray) -> dict:
    overrides = dict(scenario.config_overrides)
    overrides["sampler_name"] = sampler_name
    env, spec = make_env(
        scenario.env_name,
        config_overrides=overrides,
        **scenario.env_kwargs,
    )
    # Compile one run and reuse it for every seed. A vmapped JIT makes the XLA graph
    # and CPU compile time grow with the screening sample size.
    train = jax.jit(make_train(spec.default_config, env))
    output_keys = ("returns", "end_greedy_return")
    outputs = []
    for seed in seeds:
        run = train(jax.random.PRNGKey(int(seed)))
        outputs.append({key: run[key] for key in output_keys})
    out = jax.tree.map(lambda *values: jnp.stack(values), *outputs)
    # Pull back only the two curves the score reads. ``make_train`` also returns
    # parameters and per-update traces; retaining and stacking those is pure overhead.
    return jax.device_get(out)


def _sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


def _critical_value(n: int, confidence: float) -> float:
    """Student-t critical value on ``n - 1`` degrees of freedom.

    The paired delta is a mean over ``n`` seeds divided by an *estimated* standard
    error, so it is t-distributed, not normal. Substituting the normal 1.96 (as this
    screen originally did) understates the interval by ~21% at the default 8 seeds
    (t = 2.365) and by 6.5x at the 2-seed minimum (t = 12.706). That error runs in
    the unsafe direction: it shrinks the interval, so the non-inferiority gate stops
    rejecting real support regressions.
    """
    return float(stats.t.ppf(confidence, df=n - 1))


def run_learning_benchmark(
    candidate_sampler: str,
    *,
    seed: int,
    num_seeds: int,
    scenarios: Iterable[LearningScenario] = DEFAULT_LEARNING_SCENARIOS,
    confidence: float = 0.975,
    support_tolerance: float = 0.05,
    reference_controls: tuple[str, ...] = (),
) -> dict:
    """Compare candidate and uniform replay on identical random seeds.

    The primary utility is ``0.6 * normalized mean training return + 0.4 * normalized
    final return``. Scenario scores are one-sided Student-t lower *screening* bounds on
    the paired candidate-minus-uniform utility (see :func:`_critical_value`).

    They are deliberately not called confidence bounds. The arithmetic is that of a
    one-sided 97.5% bound, but nothing here carries the coverage that name implies:

    * BFTS evaluates every node against the *same* fixed seed set, so the bounds across
      a campaign are neither independent nor a single pre-registered test.
    * The candidate whose bound is reported is the argmax over that campaign, so the
      reported value is subject to the winner's curse; conditional on being selected it
      is biased upward.
    * No multiplicity adjustment is applied across the ~12 nodes of a campaign.

    Read them as a *ranking and promotion statistic* that is conservative relative to
    the raw mean, not as an interval with 97.5% coverage of the true improvement. The
    confirmatory held-out stage is closer to a genuine pre-registered test -- one frozen
    candidate, fresh master seed, fresh scenarios -- but it inherits the selection that
    chose the candidate, so it bounds the finalist's performance, not the search's.

    A support-gate scenario is judged by a *non-inferiority* test: the candidate passes
    only if its lower bound stays above ``-support_tolerance``, so a candidate that is
    merely too noisy to clear the bar is rejected rather than waved through. Failing any
    gate sets ``primary_score`` to ``-inf`` -- a hard rejection, not a cap, so that a
    gate-failing candidate can never outrank a gate-passing one under plain
    ``max(primary_score)`` ranking. ``raw_score`` keeps the uncapped value for
    diagnosis.

    ``reference_controls`` names additional samplers to measure the candidate against
    on the *same* paired seeds. They are **reported and never gate**: ``primary_score``
    is always the bound against uniform replay. "Beat uniform" is a bar noise can
    clear, so a stronger reference is worth reporting -- but a gate that no known
    method passes prunes the entire search and yields an uninterpretable null, so
    promoting one of these to a gate should follow evidence that something clears it,
    not precede it.

    Every scenario sets ``large_batch_size > batch_size`` and a non-zero
    ``support_floor``, so the selection samplers are actually exercised: they draw
    ``L``, score it, and keep ``B``. That matters for what the controls *mean* --
    with ``L == B``, Gumbel-top-k is a permutation and ``support_floor`` defaults to
    ``0.0``, so a control named for the support-floored recovering configuration would
    silently degrade to exact-ratio loss weights over a uniform draw, and the tier
    could not express the paper's own mechanism at all.
    """

    if num_seeds < 2:
        raise ValueError("the paired learning screen requires at least two seeds")
    scenario_list = tuple(scenarios)
    if not scenario_list:
        raise ValueError("at least one learning scenario is required")
    for scenario in scenario_list:
        scenario.validate()

    # Resolve every control name up front. Deferring this means a typo surfaces as a
    # KeyError inside a jit trace *after* the candidate's full training run has been
    # paid for.
    controls = tuple(dict.fromkeys(reference_controls))
    for name in controls:
        get_sampler(name)
    assert_baseline_integrity(*(("uniform",) + controls))

    root = np.random.SeedSequence(seed)
    seed_values = root.generate_state(num_seeds, dtype=np.uint32)
    critical_value = _critical_value(num_seeds, confidence)
    results: dict[str, dict[str, Any]] = {}
    lower_bounds = []
    gate_failures = []

    for scenario in scenario_list:
        # Re-checked per scenario rather than once at import: ``make_train`` resolves
        # the sampler by name at call time, and the candidate's own arm has already
        # executed by the second iteration.
        assert_baseline_integrity(*(("uniform",) + controls))
        baseline = _run_sampler(scenario, "uniform", seed_values)
        candidate = _run_sampler(scenario, candidate_sampler, seed_values)
        candidate_curve = np.asarray(candidate["returns"], dtype=np.float64)
        candidate_final = np.asarray(candidate["end_greedy_return"], dtype=np.float64)
        candidate_auc = candidate_curve.mean(axis=1) / scenario.return_scale

        def _paired(control: dict) -> tuple[np.ndarray, float, float]:
            """Paired candidate-minus-control utility on the shared seeds."""
            control_curve = np.asarray(control["returns"], dtype=np.float64)
            control_final = np.asarray(control["end_greedy_return"], dtype=np.float64)
            control_auc = control_curve.mean(axis=1) / scenario.return_scale
            final_delta = (candidate_final - control_final) / scenario.return_scale
            delta = 0.6 * (candidate_auc - control_auc) + 0.4 * final_delta
            mean = float(delta.mean())
            return delta, mean, mean - critical_value * _sem(delta)

        baseline_final = np.asarray(baseline["end_greedy_return"], dtype=np.float64)
        baseline_auc = (
            np.asarray(baseline["returns"], dtype=np.float64).mean(axis=1)
            / scenario.return_scale
        )
        paired_delta, mean_delta, lower_bound = _paired(baseline)

        # Reported, never gating -- see the docstring. Reuses ``seed_values``, so the
        # reference arms are paired with the candidate exactly as uniform is.
        reference_bounds: dict[str, dict[str, float]] = {}
        for name in controls:
            reference = _run_sampler(scenario, name, seed_values)
            _, reference_mean, reference_bound = _paired(reference)
            reference_bounds[name] = {
                "mean_utility_delta": reference_mean,
                "lower_screening_bound": reference_bound,
            }

        if scenario.support_gate:
            # Simultaneous-Attack is a support-preservation gate, not a scored
            # objective. It may only *cap* the score on a support regression; a
            # candidate must never be able to lift primary_score by improving the
            # gate scenario (support-starvation theorem: that trade is null-to-
            # negative and the search must not be steered toward it).
            if lower_bound < -support_tolerance:
                gate_failures.append(scenario.name)
        else:
            lower_bounds.append(lower_bound)

        results[scenario.name] = {
            "support_gate": scenario.support_gate,
            "fingerprint": scenario.fingerprint(),
            "paired_utility_delta": paired_delta.tolist(),
            "mean_utility_delta": mean_delta,
            "utility_sem": _sem(paired_delta),
            # Named for its control: this is the bound the score and the gate use.
            "lower_screening_bound": lower_bound,
            "reference_controls": reference_bounds,
            "baseline_final_return": baseline_final.tolist(),
            "candidate_final_return": candidate_final.tolist(),
            "baseline_auc": baseline_auc.tolist(),
            "candidate_auc": candidate_auc.tolist(),
        }

    if not lower_bounds:
        raise ValueError("at least one non-gate (scored) scenario is required")
    # Worst scored scenario, not the mean of them. Promotion is the claim "improves on
    # every scored task", i.e. an intersection-union test, and ``mean(bounds) > t``
    # does not imply ``min(bounds) > t``: averaging lets a large win on one task pay
    # for a regression on another. Identical to the mean while exactly one scenario is
    # scored, which is why the distinction was invisible -- and why it has to be fixed
    # before a second scored scenario is added rather than after.
    raw_score = float(np.min(lower_bounds))
    primary_score = REJECTED_SCORE if gate_failures else raw_score
    return {
        "status": "ok" if not gate_failures else "support_regression",
        "metric_direction": "maximize",
        "primary_score": primary_score,
        "raw_score": raw_score,
        "candidate_sampler": candidate_sampler,
        "num_seeds": num_seeds,
        "confidence": confidence,
        "critical_value": critical_value,
        "support_tolerance": support_tolerance,
        "scoring_control": "uniform",
        "reference_controls": list(controls),
        "paired_seeds": seed_values.astype(np.uint64).tolist(),
        "support_gate_failures": gate_failures,
        "scenarios": results,
    }
