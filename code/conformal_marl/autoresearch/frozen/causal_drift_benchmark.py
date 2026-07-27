"""Frozen causal screen for replay gains specific to teammate-policy drift.

The benchmark isolates replay from online data-collection feedback. An ego agent starts
from the exact best-response values against a fixed two-action teammate. A frozen replay
stream then either keeps that teammate policy unchanged or applies an exogenous policy
switch. Candidate and controls see paired randomness and identical pre-switch data.

The primary statistic is a difference-in-differences: improvement over a control under
drift minus improvement over that same control without drift. Generic prioritisation gains
that also occur in the stationary regime therefore do not count as drift correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from conformal_marl.replay import Transition, empty_transition, make_replay_buffer
from conformal_marl.samplers import SamplerContext, get_sampler


@dataclass(frozen=True)
class CausalDriftConfig:
    pre_attack_prob: float = 0.9
    post_attack_prob: float = 0.1
    pre_size: int = 256
    post_size: int = 256
    calibration_size: int = 64
    draw_batch_size: int = 128
    train_batch_size: int = 32
    updates: int = 64
    learning_rate: float = 0.2
    epsilon: float = 0.2
    support_floor: float = 0.1

    def validate(self) -> None:
        for value in (self.pre_attack_prob, self.post_attack_prob):
            if not 0.0 < value < 1.0:
                raise ValueError("partner probabilities must have full support")
        if self.pre_attack_prob <= 0.25 or self.post_attack_prob >= 0.25:
            raise ValueError("the intervention must switch the analytic best response")
        if min(self.pre_size, self.post_size, self.calibration_size) < 8:
            raise ValueError("causal replay blocks are too small")
        if not 1 <= self.train_batch_size <= self.draw_batch_size:
            raise ValueError("invalid causal replay batch sizes")
        if self.draw_batch_size > self.pre_size + self.post_size:
            raise ValueError("draw batch exceeds the frozen replay stream")
        if self.updates < 2 or not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("invalid causal update budget")
        if not np.isclose(self.epsilon / 2.0, self.post_attack_prob):
            raise ValueError("post policy must match epsilon-greedy abstain")
        if not np.isclose(1.0 - self.epsilon / 2.0, self.pre_attack_prob):
            raise ValueError("pre policy must match epsilon-greedy attack")


SEARCH_CAUSAL_DRIFT_CONFIG = CausalDriftConfig()
HELDOUT_CAUSAL_DRIFT_CONFIG = CausalDriftConfig(
    pre_attack_prob=0.85,
    post_attack_prob=0.15,
    pre_size=320,
    post_size=320,
    calibration_size=80,
    draw_batch_size=160,
    train_batch_size=40,
    updates=80,
    epsilon=0.3,
)


def _expected_action_values(partner_attack_prob: float) -> np.ndarray:
    """Exact ego values for [abstain, attack] in two-agent Simultaneous-Attack."""
    q = partner_attack_prob
    return np.asarray([-0.5 * q, 1.5 * q - 0.5], dtype=np.float64)


def _reward(actions: np.ndarray) -> np.ndarray:
    attackers = actions.sum(axis=1)
    return np.where(attackers == 2, 1.0, np.where(attackers == 0, 0.0, -0.5))


def _transition_block(
    ego_uniforms: np.ndarray,
    partner_uniforms: np.ndarray,
    partner_attack_prob: float,
) -> Transition:
    size = len(ego_uniforms)
    ego = (ego_uniforms < 0.5).astype(np.int32)
    partner = (partner_uniforms < partner_attack_prob).astype(np.int32)
    actions = np.stack((ego, partner), axis=1)
    rewards = np.repeat(_reward(actions)[:, None], 2, axis=1)
    obs = np.broadcast_to(np.eye(2, dtype=np.float32), (size, 2, 2)).copy()
    base = empty_transition(2, 2, 2)
    partner_taken_prob = np.where(
        partner == 1, partner_attack_prob, 1.0 - partner_attack_prob
    )
    behavior_log_prob = np.stack(
        (np.full(size, np.log(0.5)), np.log(partner_taken_prob)),
        axis=1,
    )
    hidden = np.zeros((size, 2, 0), dtype=np.float32)
    return base.replace(
        obs=jnp.asarray(obs),
        actions=jnp.asarray(actions),
        reward=jnp.asarray(rewards, dtype=jnp.float32),
        done=jnp.ones((size, 2), dtype=jnp.bool_),
        next_obs=jnp.asarray(obs),
        avail_actions=jnp.ones((size, 2, 2), dtype=jnp.float32),
        next_avail_actions=jnp.ones((size, 2, 2), dtype=jnp.float32),
        behavior_log_prob=jnp.asarray(behavior_log_prob, dtype=jnp.float32),
        hidden=jnp.asarray(hidden),
        next_hidden=jnp.asarray(hidden),
    )


def paired_streams(
    seed: int, config: CausalDriftConfig
) -> tuple[Transition, Transition, Transition, Transition]:
    """Return drift/no-drift replay and calibration blocks with paired uniforms."""
    config.validate()
    rng = np.random.default_rng(seed)
    total = config.pre_size + config.post_size
    ego_uniforms = rng.random(total)
    partner_uniforms = rng.random(total)

    pre = _transition_block(
        ego_uniforms[:config.pre_size],
        partner_uniforms[:config.pre_size],
        config.pre_attack_prob,
    )
    stationary_post = _transition_block(
        ego_uniforms[config.pre_size:],
        partner_uniforms[config.pre_size:],
        config.pre_attack_prob,
    )
    drift_post = _transition_block(
        ego_uniforms[config.pre_size:],
        partner_uniforms[config.pre_size:],
        config.post_attack_prob,
    )
    stationary = jax.tree_util.tree_map(
        lambda left, right: jnp.concatenate((left, right), axis=0),
        pre,
        stationary_post,
    )
    drift = jax.tree_util.tree_map(
        lambda left, right: jnp.concatenate((left, right), axis=0),
        pre,
        drift_post,
    )

    cal_ego = rng.random(config.calibration_size)
    cal_partner = rng.random(config.calibration_size)
    stationary_cal = _transition_block(
        cal_ego, cal_partner, config.pre_attack_prob
    )
    drift_cal = _transition_block(cal_ego, cal_partner, config.post_attack_prob)
    return stationary, drift, stationary_cal, drift_cal


def _apply_q(params, carry, network_input):
    obs, _resets = network_input
    agent_index = jnp.argmax(obs, axis=-1)
    q_values = jnp.where(
        (agent_index == 0)[..., None],
        jnp.asarray(params["ego_q"]),
        jnp.asarray(params["partner_q"]),
    )
    return carry, q_values


def _sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _screening_bound(values: np.ndarray, screening_z: float) -> float:
    return float(values.mean() - screening_z * _sem(values))


def _run_regime(
    sampler_name: str,
    replay: Transition,
    calibration: Transition,
    partner_attack_prob: float,
    *,
    sample_seed: int,
    config: CausalDriftConfig,
) -> float:
    buffer = make_replay_buffer(
        max_size=config.pre_size + config.post_size + 1,
        min_size=8,
        sample_batch_size=config.draw_batch_size,
    )
    buffer_state = buffer.init(empty_transition(2, 2, 2))
    buffer_state = buffer.add(buffer_state, replay)
    sampler = get_sampler(sampler_name)

    ego_q = _expected_action_values(config.pre_attack_prob)
    partner_q = (
        np.asarray([0.0, 1.0])
        if partner_attack_prob > 0.5
        else np.asarray([1.0, 0.0])
    )
    utilities = []
    root_key = jax.random.PRNGKey(sample_seed)
    sample_keys = jax.random.split(root_key, config.updates)

    for key in sample_keys:
        params = {
            "ego_q": jnp.asarray(ego_q, dtype=jnp.float32),
            "partner_q": jnp.asarray(partner_q, dtype=jnp.float32),
        }
        context = SamplerContext(
            online_params=params,
            target_params=params,
            apply_fn=_apply_q,
            num_agents=2,
            num_actions=2,
            hidden_dim=0,
            epsilon=jnp.asarray(config.epsilon),
            cal_batch=calibration,
            support_floor=config.support_floor,
        )
        batch, weights, _info = sampler(
            buffer, buffer_state, key, context, config.train_batch_size
        )
        actions = np.asarray(batch.actions[:, 0], dtype=np.int32)
        rewards = np.asarray(batch.reward[:, 0], dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.ndim == 2:
            weight_values = weight_values[:, 0]
        if (
            weight_values.shape != rewards.shape
            or not np.all(np.isfinite(weight_values))
            or np.any(weight_values < 0.0)
        ):
            raise ValueError("candidate returned invalid causal replay weights")

        for action in (0, 1):
            selected = actions == action
            denominator = float(weight_values[selected].sum())
            if denominator > 1e-12:
                target = float(
                    np.dot(weight_values[selected], rewards[selected]) / denominator
                )
                ego_q[action] += config.learning_rate * (target - ego_q[action])

        true_values = _expected_action_values(partner_attack_prob)
        greedy = int(np.argmax(ego_q))
        best = float(np.max(true_values))
        worst = float(np.min(true_values))
        utilities.append(1.0 - (best - true_values[greedy]) / (best - worst))

    return float(np.mean(utilities))


def run_causal_drift_benchmark(
    candidate_sampler: str,
    *,
    seed: int,
    num_seeds: int,
    config: CausalDriftConfig = SEARCH_CAUSAL_DRIFT_CONFIG,
    control_samplers: tuple[str, ...] = ("uniform", "exact_ratio_floor"),
    screening_z: float = 1.645,
    no_drift_tolerance: float = 0.02,
) -> dict:
    """Run the paired difference-in-differences causal replay screen."""
    config.validate()
    if num_seeds < 2:
        raise ValueError("causal drift screen requires at least two seeds")
    if not control_samplers or "uniform" not in control_samplers:
        raise ValueError("causal drift controls must include uniform replay")

    seed_values = np.random.SeedSequence(seed).generate_state(
        num_seeds, dtype=np.uint32
    )
    methods = tuple(dict.fromkeys((candidate_sampler, *control_samplers)))
    utilities = {
        method: {"no_drift": [], "drift": []}
        for method in methods
    }
    for seed_value in seed_values:
        stationary, drift, stationary_cal, drift_cal = paired_streams(
            int(seed_value), config
        )
        for method in methods:
            utilities[method]["no_drift"].append(_run_regime(
                method,
                stationary,
                stationary_cal,
                config.pre_attack_prob,
                sample_seed=int(seed_value),
                config=config,
            ))
            utilities[method]["drift"].append(_run_regime(
                method,
                drift,
                drift_cal,
                config.post_attack_prob,
                sample_seed=int(seed_value),
                config=config,
            ))

    candidate_no_drift = np.asarray(
        utilities[candidate_sampler]["no_drift"], dtype=np.float64
    )
    candidate_drift = np.asarray(
        utilities[candidate_sampler]["drift"], dtype=np.float64
    )
    control_results = {}
    primary_bounds = []
    regressions = []
    for control in control_samplers:
        control_no_drift = np.asarray(utilities[control]["no_drift"], dtype=np.float64)
        control_drift = np.asarray(utilities[control]["drift"], dtype=np.float64)
        no_drift_delta = candidate_no_drift - control_no_drift
        drift_delta = candidate_drift - control_drift
        interaction = drift_delta - no_drift_delta
        no_drift_bound = _screening_bound(no_drift_delta, screening_z)
        drift_bound = _screening_bound(drift_delta, screening_z)
        interaction_bound = _screening_bound(interaction, screening_z)
        primary_bounds.extend((drift_bound, interaction_bound))
        if no_drift_bound < -no_drift_tolerance:
            regressions.append(control)
        control_results[control] = {
            "no_drift_delta": no_drift_delta.tolist(),
            "drift_delta": drift_delta.tolist(),
            "causal_interaction": interaction.tolist(),
            "no_drift_screening_bound": no_drift_bound,
            "drift_screening_bound": drift_bound,
            "interaction_screening_bound": interaction_bound,
        }

    pre_values = _expected_action_values(config.pre_attack_prob)
    post_values = _expected_action_values(config.post_attack_prob)
    immediate_regret = float(
        (post_values.max() - post_values[np.argmax(pre_values)])
        / (post_values.max() - post_values.min())
    )
    if immediate_regret < 0.25:
        raise ValueError("causal intervention lacks pre-registered teeth")

    raw_score = float(min(primary_bounds))
    primary_score = (
        min(raw_score, -no_drift_tolerance) if regressions else raw_score
    )
    return {
        "status": "ok" if not regressions else "no_drift_regression",
        "metric_direction": "maximize",
        "primary_score": primary_score,
        "raw_score": raw_score,
        "candidate_sampler": candidate_sampler,
        "control_samplers": list(control_samplers),
        "num_seeds": num_seeds,
        "paired_seeds": seed_values.astype(np.uint64).tolist(),
        "immediate_normalized_regret": immediate_regret,
        "no_drift_regressions": regressions,
        "candidate_utility": {
            "no_drift": candidate_no_drift.tolist(),
            "drift": candidate_drift.tolist(),
        },
        "controls": control_results,
    }
