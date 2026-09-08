# AGENTS.md

Guidance for Codex working in this repository. Read this before editing.

## What this repo is

An AAMAS paper — **"Conformal Replay for Off-Policy Multi-Agent Reinforcement Learning"** —
and its JAX/JaxMARL implementation. The paper is the primary deliverable; the code produces
its figures. See [`README.md`](README.md) for the project overview and
[`code/README.md`](code/README.md) for the module-by-module guide.

- Paper: [`AAMAS_2024_sample.tex`](AAMAS_2024_sample.tex), bib in [`sample.bib`](sample.bib)
- Code: [`code/`](code/) — a `uv` project, package `conformal_marl`

## Conceptual landmines — do not "fix" these

These look like bugs and are not. They are the paper's actual findings.

1. **Conformal *selection* collapsing on Simultaneous-Attack is intentional.** Filtering
   replay toward currently-compatible transitions *starves exploration* and is
   null-to-negative; this is the **support-starvation theorem** (Theorem 4.2) and the
   paper's rigorous centerpiece, not a regression. Do **not** try to make `conformal_policy`
   / `exact_ratio` selection "win" on Simu. The headline is **detection + certified replay**;
   selection is the reframed negative.
2. **The detection benchmark rejecting action-masked envs (SMAX) is by design.** It raises
   `NotImplementedError` for masked envs because the fixed scripted-partner intervention and
   all-ones mask assume a state-independent action set. A mask-aware intervention is future
   work. Don't remove the guard.
3. **The AES baseline is an honest reconstruction, deliberately.** AES (arXiv:2601.19624) is
   an entropy *scheduler* with no change-point detector; we build a detector from its TD-error
   proxy so the comparison exists at all. Don't "correct" it toward a detector AES never had.

If you think one of these is wrong, raise it with the user rather than silently changing it.

## Building the paper

TeX binaries are under `/Library/TeX/texbin` (macOS). From the repo root:

```bash
latexmk -f -pdf -interaction=nonstopmode AAMAS_2024_sample.tex
```

- Use `-f` so it finishes through the benign `T1/zi4/m/it` font warning. `-halt-on-error=0`
  is **not** a valid flag — do not use it.
- The class is the vendored `aamas.cls` (acmart-based); it provides `theorem`/`proposition`/
  `lemma`/`proof` environments. Bib style is the vendored `ACM-Reference-Format.bst` via natbib.
- Custom macros are defined in the preamble (e.g. `\uma \picur \pipast \Qbr \KL \dTV \indic`).
  Reuse them; don't reinvent notation.
- After editing, recompile and confirm: exit 0, no undefined references/citations. The repo
  commits the built `AAMAS_2024_sample.pdf` and `.bbl`.

## Working in the code

- **Always use `uv`** from `code/`: `uv sync`, `uv run pytest -q`, `uv run python scripts/...`.
  Never `pip install` into the system Python.
- **Python is pinned to 3.12; JAX/jaxlib to 0.4.30; `chex < 0.1.91`.** JaxMARL 0.0.2 needs the
  `jax.tree_map` alias removed in JAX 0.6.0. **Do not bump JAX past ~0.5.x** — it breaks
  JaxMARL. `code/tests/test_smoke.py` guards this and will fail loudly if violated.
- macOS/Apple Silicon uses `jax[cpu]`. **Never install `jax-metal`** — it breaks JaxMARL.
- Run the full suite before claiming done: `uv run pytest -q`. Per-milestone tests
  are validity gates (martingale `E[M_t]=1`/Ville, p-value super-uniformity, IQL learns) —
  treat a failure there as a scientific regression, not a flaky test.
- Lint with `uv run ruff check` (line-length 100, target py312).

## Git / PR workflow

- GitHub repo is **private** (`ArnaudGardille/report-offline-marl-stability`); `gh` is authed.
- Convention: branch → PR → review (often with a sub-agent) → **squash-merge** (`gh pr merge
  --squash --delete-branch`). `main` is the integration branch.
- **A parallel implementation session may have uncommitted work in the main working tree.**
  For paper-only PRs, create an isolated worktree off `origin/main`
  (`git worktree add ... origin/main`) so you never touch the active tree; clean up with
  `git worktree remove --force` after merge. Verify you haven't clobbered in-flight code files.
- Only commit/push when the user asks. End commit messages with the
  `Co-Authored-By: Codex Opus 4.8 <noreply@anthropic.com>` trailer.

## What is and isn't committed

- **Committed:** the `.tex`/`.bib`/`.pdf`/`.bbl`, the camera-ready figures in
  `code/results/*.png`, the code, and `code/uv.lock`.
- **Gitignored (do not commit):** `sources/` (bulky, partly copyrighted reference PDFs —
  never push these), `code/outputs/` and `*.npz` (regenerable raw run data),
  `code/results/*_smoke.png` (throwaway CPU/CI verification artifacts), and LaTeX build
  artifacts (`.aux`, `.log`, `.fls`, …). See [`.gitignore`](.gitignore).

## Figure naming

The milestone figures are `code/results/m<milestone>_<name>[_<env>_<layout>].png` (env/layout
folded into the name so layouts don't clobber each other). Newer experiment figures append a
scale tag: `_smoke` = throwaway CPU/CI artifact (gitignored), `_full` = camera-ready GPU run,
and may drop the `m<milestone>_` prefix. The paper's `\graphicspath` includes `{code/results/}`.

## Memory

Persistent notes live in
`/Users/arnaud/.codex/memories/`
(indexed by `MEMORY.md`). Key entries: the paper direction, the JAX implementation plan, the
results summary, and the research-directions verdicts behind the detection/certified-replay
pivot. Memories are point-in-time — verify file:line claims against current code before relying
on them.
