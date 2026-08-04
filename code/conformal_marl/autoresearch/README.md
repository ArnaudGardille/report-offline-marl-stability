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

### Screen power

The budget this tier runs at is not a guess. `scripts/screen_power_check.py` measures
whether the screen can separate deliberately degraded replay from uniform replay, using
two probe samplers registered outside the frozen package: a distributionally null one
(uniform on an independently folded key), which fixes the seed-noise floor, and a family
that collapses each training batch onto `k` distinct transitions, which degrades replay
monotonically as `k` falls. A screen that cannot see the degraded probes cannot be
trusted when it reports an improvement.

The previously pinned budget (16,384 steps, `return_scale` 16.0) failed that check
outright: 4% of evaluation points were non-zero, and collapsing every batch onto a
*single* repeated transition moved the paired delta by 0.7x the noise floor, with the
probe ranking coming out anti-correlated with true degradation. At the current
65,536-step, `return_scale` 2.0, 32-eval-episode setting the probes rank monotonically
and the worst one moves the delta by ~3.6x the noise floor. `scripts/screen_budget_sweep.py`
records the budget/signal curve behind that choice.

Re-run the check after any change to the scored scenario:

```bash
uv run python scripts/screen_power_check.py --num-seeds 8
```

> **Remaining limitation.** Power is adequate, not generous: at 8 seeds the minimum
> detectable effect is roughly 0.04 in scaled utility, so only a fairly large real
> improvement is promotable, and the most degraded probe sits just at the edge of
> resolvability. The 32-seed held-out stage is correspondingly better powered than the
> search stage. Separately, `support_tolerance` (0.05) is still ~6x looser than the
> worst degradation the gate scenario actually exhibits (-0.008), so the support gate
> would not fire even on the most degraded probe; the gate scenario is deterministic
> across seeds (SEM exactly 0), so tightening it is possible but has not been done.

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
and 32 paired learning seeds. The held-out *learning* stage also shifts the task itself:
a fourth Simultaneous-Attack agent and a longer STORM horizon, neither seen during
search. The held-out *mechanism* stage re-runs the frozen sweep on a new seed only, so
it confirms against fresh randomness rather than a shifted task distribution -- do not
describe it as a distribution-shift test. The held-out result is confirmatory: it cannot select a
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
and no Simultaneous-Attack support regression. Promotion scores against uniform replay;
`--reference-control` adds a stronger comparison (exact-ratio weighting) on the same
paired seeds, reported on the confirmatory stage and never gating. It is reported
rather than enforced deliberately: a gate no known method clears prunes the whole
search and produces an uninterpretable null instead of a finding, so promoting this
comparison to a gate should follow evidence that something passes it. Note also that
with `large_batch_size` unset the buffer draws exactly `batch_size` items, so conformal
selection is inert and that control measures exact-ratio *weighting*, not the
support-floored recovering configuration its sampler is named for.

Merely completing training is not treated as evidence of improved sample efficiency.
