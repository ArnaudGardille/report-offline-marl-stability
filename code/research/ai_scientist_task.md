# Constrained research task: target-policy replay in cooperative MARL

## Objective

Discover a replay or target-estimation method that improves sample efficiency in
off-policy cooperative multi-agent reinforcement learning when teammate policies change.
Under centralized training, joint actions and current teammate policies are available.
The conditional environment law

`P(s', r | s, u_a, u_-a)`

is stationary; only the teammate action distribution changes.

## Editable contract

You may replace files only inside `candidate/`. Preserve both public entry points exported
from `candidate/__init__.py`:

- `estimate_operator(batch, rng) -> Estimate`
- the sampler registered as `autoresearch_candidate`

Use `estimator.py`, `sampler.py`, `target_policy.py`, and `state.py` to separate concerns.

The implementation must remain generic, finite, JAX-compatible in the learning tier,
and must not inspect environment names, evaluation seeds, or evaluator internals.

`OperatorBatch` exposes exactly these NumPy arrays; do not invent optional fields:

- `teammate_actions`: shape `(N,)`, integer action index for each replay outcome;
- `outcomes`: shape `(N,)`, observed Bellman targets;
- `behavior_probs`: shape `(K,)`, behavior probability for each teammate action;
- `target_probs`: shape `(K,)`, target-policy probability for each teammate action.

For example, an exact per-sample density ratio is
`target_probs[teammate_actions] / behavior_probs[teammate_actions]`. The API validates
full behavior support. A stratified estimator may estimate one conditional mean per
observed action and recombine them using `target_probs`, but it must define principled
behavior when a finite batch does not cover every action.

## Evaluation ladder

1. The operator tier estimates the target-policy perceived Bellman backup from replay
   collected under a different behavior policy. It reports exact bias, variance and MSE.
   Candidates scoring below `-0.25` are not promoted.
2. The learning tier compares the candidate with uniform replay on identical seeds.
   Cooperative STORM measures early learning efficiency. Simultaneous-Attack is a hard
   support-preservation gate: a candidate that starves rare optimal joint actions is
   rejected even if it improves STORM.

## Research directions

Promising hypotheses include action-stratified conditional means, exact or self-normalized
importance sampling, clipped/ESS-controlled ratios, target teammate policies, and soft
support-preserving replay. Combine mechanisms only when the parent result motivates it.

Hard conformal filtering is known to collapse on Simultaneous-Attack through support
starvation. Do not try to hide or optimize around that negative result. Prefer minimal,
interpretable changes and use evaluator feedback to distinguish bias, variance, learning
gain, and support regression.

## Scientific constraints

- Never modify rewards, environments, seeds, budgets, metrics, frozen code, or output JSON.
- Never fabricate results or infer success from logs; evaluator JSON is authoritative.
- Do not add network access, subprocesses, package installation, or persistent side effects.
- Report negative or inconclusive outcomes honestly.
- AI Scientist usage must be disclosed in resulting manuscripts under its license.
