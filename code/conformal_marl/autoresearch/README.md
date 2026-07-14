# Constrained MARL autoresearch harness

This package separates the one file an automated researcher may edit from the benchmark
that decides whether the edit helped.

- `candidate.py` is editable.
- `api.py` defines the candidate contract.
- `frozen/` generates private data and computes objective metrics; it is read-only during
  an automated search.

The first tier estimates a perceived Bellman backup for one fixed state and ego action.
The environment's conditional target law is stationary given the teammate action, while
the teammate behaviour policy differs from the current target policy. The exact target is
known to the benchmark but never passed to the candidate.

Run the baseline from `code/`:

```bash
uv run python -m conformal_marl.autoresearch.evaluate \
  --tier operator \
  --trials 128 \
  --output outputs/autoresearch_operator.json
```

The scalar `primary_score` is the negative mean squared error across frozen mismatch
scenarios, so higher is better. Scenario-level bias, variance, and MSE remain available
for diagnosis. Later learning tiers must preserve the same separation: candidates may
change replay or target estimation, never environments, rewards, seeds, budgets, or score
calculation.
