# Constrained MARL autoresearch harness

This package separates the one file an automated researcher may edit from the benchmark
that decides whether the edit helped.

- `candidate/` is the only editable package.
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

The second tier performs paired-seed IQL training on cooperative STORM and
Simultaneous-Attack:

```bash
uv run python -m conformal_marl.autoresearch.evaluate \
  --tier learning \
  --num-seeds 8 \
  --output outputs/autoresearch_learning.json
```

Its score is a lower confidence bound on paired AUC/final-return improvement over
uniform replay. Simultaneous-Attack is a support-preservation gate, not a target to game:
a candidate with a material regression there is rejected even if it improves STORM.

## AI Scientist v2 orchestration

Keep the orchestrator in a sibling checkout; do not vendor it into this repository. After
checking out the constrained-BFTS adaptation, generate a machine-local config:

```bash
uv run python scripts/prepare_ai_scientist.py \
  --orchestrator /path/to/AI-Scientist-v2 \
  --base-ref research-ai-scientist-integration
```

The command validates the Git ref, candidate package, shared `uv` interpreter, and every
frozen path, then prints the exact hierarchical-campaign command. Each worktree imports
its own candidate through `PYTHONPATH` while reusing the prepared dependency environment,
which avoids reinstalling JAX for every node. Search outputs remain under ignored
`code/outputs/`; only reviewed candidate code and reproducible aggregate figures belong
in the paper repository. Run generated code in an OS-level sandbox, and disclose AI
Scientist use in any resulting manuscript.
