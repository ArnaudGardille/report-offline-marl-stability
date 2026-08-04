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

Its score is a one-sided Student-t lower *screening* bound on paired improvement over
uniform replay, weighting mean training return 0.6 and final greedy return 0.4. It is
not a confidence bound and should not be reported as one: BFTS reuses a single fixed
seed set across every node, applies no multiplicity adjustment, and reports the argmax
over the campaign, so the bound carries no nominal coverage. Treat it as a conservative
ranking statistic.
Simultaneous-Attack is a support-preservation gate, not a target to game: the candidate
must pass a non-inferiority test there, and failing it sets `primary_score` to `-inf` --
a hard rejection, so a gate-failing candidate can never outrank a gate-passing one under
`max(primary_score)` ranking. `raw_score` keeps the uncapped value for diagnosis, and
each scenario reports a `fingerprint` tying the score to the exact frozen budget that
produced it.

> **Known limitation.** At the pinned STORM budget the reward is very sparse -- a
> measured 2-seed baseline run produced non-zero greedy return at only 8 of 128 eval
> points, with a maximum of 0.5 against a `return_scale` of 16.0. The scored deltas
> therefore sit around 1e-3 while `support_tolerance` is 0.05, so the screen currently
> has little power to separate candidates. Recalibrating `return_scale` and raising the
> STORM budget (validated by a power check against a deliberately degraded sampler)
> is open work; the tier is safe to run but should not yet be read as a strong signal.

## AI Scientist v2 orchestration

Keep the orchestrator in a sibling checkout; do not vendor it into this repository. After
checking out the constrained-BFTS adaptation, generate a machine-local config:

```bash
uv run python scripts/prepare_ai_scientist.py \
  --orchestrator /path/to/AI-Scientist-v2 \
  --base-ref main
```

The command validates the Git ref, candidate package, shared `uv` interpreter, and every
frozen path, resolves the base ref to an immutable commit SHA, then prints the exact
hierarchical-campaign command. Each worktree imports
its own candidate through `PYTHONPATH` while reusing the prepared dependency environment,
which avoids reinstalling JAX for every node. Search outputs remain under ignored
`code/outputs/`; only reviewed candidate code and reproducible aggregate figures belong
in the paper repository. Run generated code in an OS-level sandbox, and disclose AI
Scientist use in any resulting manuscript.

The evaluator imports the frozen scorers before the generated candidate, which seals a
snapshot of the built-in samplers, and re-checks that snapshot before each control arm
runs. This catches a candidate that shadows a baseline name and would otherwise score
itself against itself. It is a correctness guard, not a security boundary: candidate
code shares this interpreter and can defeat any in-process check, so the OS-level
sandbox above is what actually contains it.

The search sees 8 paired seeds by default. After BFTS stops, the highest-ranked unique
candidate is frozen and rerun on a distinct master seed with a 128-trial mechanism sweep
and 32 paired learning seeds. The held-out result is confirmatory: it cannot select a
runner-up, is stored separately, and is never returned to the method designer. A failed
held-out evaluation therefore rejects the selected candidate instead of turning the final
benchmark into another tuning set.

The experimentalist may configure a search-only diagnostic sweep within validated bounds
(actions, sample sizes, drift strengths, noise, coverage, and scenario seed). This stage
is recorded but never contributes to cross-candidate ranking, because two candidates
tested on different designs do not have comparable scores. Every candidate is ranked on
the same frozen mechanism sweep. The diagnostic runs last, after every ranked stage:
the evaluator stops at the first failing stage, so a designed sweep that is rejected or
crashes earlier in the sequence would depress that candidate's ranking score and stage
count for reasons unrelated to its replay mechanism. The evaluator fixes repetition
counts and rejects unknown fields or sweeps above 128 cells. Held-out mechanism
validation ignores the designed diagnostic.

Learning promotion requires a positive paired screening bound on the scored STORM task
and no Simultaneous-Attack support regression. Merely completing training is not treated
as evidence of improved sample efficiency.
