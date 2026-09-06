# Running the heavy benchmarks on GPU (Colab)

The cheap experiments in this repo (Simultaneous-Attack, MPE `simple_spread`) train to a
meaningful policy on CPU in seconds-to-minutes and are run directly here. The heavy
**Overcooked** detection benchmark is produced at full scale here (8 seeds, longer
pre/post windows) — a GPU makes the multi-seed sweep fast. This document explains how to
produce the camera-ready detection figures on a GPU via Google Colab.

> **Why not SMAX?** SMAX is an action-masked env: its legal action set is *state-dependent*
> (units die / come into range). The detection benchmark's intervention is a **fixed**
> scripted partner action played for the whole post-swap window, and its scores run on an
> all-ones mask — both assume a state-independent action set. So `--experiment detection`
> rejects masked envs (`NotImplementedError`); a valid SMAX detection benchmark would need
> a mask-aware intervention (future work). SMAX's validated contribution is the double-DQN
> next-state target-mask fix in `conformal_marl/iql.py`, covered by the training tests.

> **Status:** the harness (`scripts/run_benchmark.py`) is verified end-to-end at `--smoke`
> scale on CPU (see the PR description for pasted output). The `--full` GPU path and the
> Colab CUDA install are **untested on the author's local machine** (no GPU here) — the
> notebook is written to be correct and self-contained, but treat the first GPU run as the
> real verification.

## TL;DR

1. Open `notebooks/colab_gpu.ipynb` in Colab. Set **Runtime → Change runtime type → GPU**
   (a T4 is plenty for `cramped_room` / `asymm_advantages`).
2. Run all cells. It clones the repo, installs **JAX 0.4.30 + CUDA 12**, runs a quick GPU
   sanity check, then runs `run_benchmark.py --full` for Overcooked (two layouts).
3. Figures are written to `code/results/m8_<env>_<config>_<experiment>_full.png` (the layout
   is folded into the name so layouts don't clobber) and can be downloaded from the Colab
   file browser (or the last cell zips them).

## The JAX CUDA install (the part that bites)

JaxMARL 0.0.2 calls `jax.tree_map`, which was **removed in JAX 0.6.0**, so this project is
**pinned to JAX 0.4.30** (the last 0.4.x with `tree_map` and cp310–cp312 wheels). Do **not**
bump it. The matching CUDA-12 install line for Colab (CUDA 12 runtime) is:

```bash
pip install --upgrade "jax[cuda12_pip]==0.4.30" jaxlib==0.4.30 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Notes / gotchas:

- The `[cuda12_pip]` extra pulls the bundled CUDA 12 / cuDNN wheels, so you do **not** need
  a system CUDA toolkit — this is what makes it work on a stock Colab GPU runtime.
- `-f .../jax_cuda_releases.html` is required so pip can find the `jaxlib` **+cuda12** wheel
  for 0.4.30 (the plain PyPI `jaxlib==0.4.30` is CPU-only).
- Colab ships its own (newer) JAX preinstalled; the cell above **downgrades** it. After the
  install you may need **Runtime → Restart session** once so the new `jaxlib` is the one
  imported. The notebook does the install in the first cell and prints
  `jax.devices()` afterwards; if it still shows CPU, restart and re-run from the install cell.
- Install the project's other pinned deps the same way:
  `pip install "jaxmarl==0.0.2" "chex<0.1.91" flax optax flashbax distrax matplotlib`.
  (`chex` is capped because `chex>=0.1.91` requires `jax>=0.7`, which drops `tree_map`.)
- We add `repo/code` to `sys.path` instead of `pip install -e .`, because `pyproject.toml`
  requires Python `>=3.12,<3.13` and Colab may run 3.10/3.11; the code itself runs fine on
  3.10+ — only the package-metadata pin is strict, and `sys.path` sidesteps it.

## Verifying the GPU is actually used

The harness prints its JAX backend on startup:

```
=== run_benchmark: env=overcooked experiment=detection scale=full | jax backend=gpu ===
```

If that says `cpu`, the CUDA `jaxlib` did not take — restart the runtime and re-run the
install cell. `jax.devices()` should list a `cuda`/`gpu` device.

## What `--full` runs

`scripts/run_benchmark.py` is a config-driven dispatcher; `--full` swaps in full-scale
budgets (the `SCALE` table in that file). The two camera-ready commands are:

```bash
# Overcooked detection (anytime-valid conformal martingale vs AES at matched false-alarm)
python scripts/run_benchmark.py --env overcooked --experiment detection --full --layout cramped_room
python scripts/run_benchmark.py --env overcooked --experiment detection --full --layout asymm_advantages
```

Detection requires an unmasked / fixed-action-set env (see the SMAX note above), so
Overcooked is the heavy detection env. Selection ablations (`--experiment selection`) work
the same way; on the easily-learned envs selection is expected to be neutral-to-slower (see
the `ablation.py` docstring), so detection is the headline.

Each run writes:

- `code/results/m8_<env>_<layout>_detection_full.png` — committed-style figure
  (matched-false-alarm calibration, detection-delay distributions, example traces).
- `code/outputs/m8_<env>_<layout>_detection_full.json` — raw numbers (gitignored).

## Approximate GPU cost

On a Colab T4, `--full` Overcooked detection (8 seeds × the env's tuned IQL budget + the
Simu false-alarm lab + the swap streams) is on the order of minutes per layout; the GPU's
win is running the multi-seed sweep in parallel rather than a longer single train. A smoke
equivalent (`--smoke`) runs in CI-time on CPU and is what is verified in this PR. Scale the
budgets in the `SCALE` table down (or up) for your session.

## Recovery benchmark on Overcooked (needs a GPU)

`scripts/recovery_benchmark.py` asks the decisive control question: after a co-learner
drift, does driving the replay buffer from the drift detector (flush-on-alarm + Channel-B
half-life) recover **return** faster than a static buffer? On **MPE `simple_spread` (CPU)**
the verdict is a clean **negative** — forgetting worsens post-drift return (an empirical
temporal analogue of support loss; see §5 of the paper). Overcooked is the
higher-coordination check, where partner
conventions matter more, but its sparse reward needs a real training budget: **at CPU-smoke
scale the pretrained policy never learns the task (returns stay ~0), so a meaningful
recovery result requires a GPU.** The code path is env-agnostic (verified end-to-end on
Overcooked at `--smoke`); to get a real result, run on Colab GPU:

```bash
# after the JAX 0.4.30 + CUDA install above:
python scripts/recovery_benchmark.py --full --drift scripted --env overcooked --seeds 8
# (also worth running --drift policy as the milder, realistic-drift comparison)
```

This writes `code/results/recovery_overcooked_<drift>_full.png` and the paired-verdict
JSON to `code/outputs/`. Expect it to either corroborate the MPE negative or, if
Overcooked's stronger coordination makes stale data genuinely harmful, show a positive —
either outcome is reportable. Budgets live in the `RecoveryConfig` `--full` branch of the
script; the `pretrain()` timestep count is the main GPU lever.
