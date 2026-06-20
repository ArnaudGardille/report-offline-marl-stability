# Running the heavy benchmarks on GPU (Colab)

The cheap experiments in this repo (Simultaneous-Attack, MPE `simple_spread`) train to a
meaningful policy on CPU in seconds-to-minutes and are run directly here. The **heavy**
JaxMARL benchmarks — **SMAX** (`3m`, `2s3z`, …) and **Overcooked** — need a GPU to train
to a non-trivial policy in reasonable wall-clock. This document explains how to produce the
camera-ready detection/selection figures on a GPU via Google Colab.

> **Status:** the harness (`scripts/run_benchmark.py`) is verified end-to-end at `--smoke`
> scale on CPU (see the PR description for pasted output). The `--full` GPU path and the
> Colab CUDA install are **untested on the author's local machine** (no GPU here) — the
> notebook is written to be correct and self-contained, but treat the first GPU run as the
> real verification.

## TL;DR

1. Open `notebooks/colab_gpu.ipynb` in Colab. Set **Runtime → Change runtime type → GPU**
   (a T4 is plenty for `3m` / `cramped_room`).
2. Run all cells. It clones the repo, installs **JAX 0.4.30 + CUDA 12**, runs a quick GPU
   sanity check, then runs `run_benchmark.py --full` for SMAX and Overcooked.
3. Figures are written to `code/results/m8_<env>_<experiment>_full.png` and can be
   downloaded from the Colab file browser (or the last cell zips them).

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
=== run_benchmark: env=smax experiment=detection scale=full | jax backend=gpu ===
```

If that says `cpu`, the CUDA `jaxlib` did not take — restart the runtime and re-run the
install cell. `jax.devices()` should list a `cuda`/`gpu` device.

## What `--full` runs

`scripts/run_benchmark.py` is a config-driven dispatcher; `--full` swaps in GPU-scale
budgets (the `SCALE` table in that file). The two camera-ready commands are:

```bash
# SMAX detection (anytime-valid conformal martingale vs AES at matched false-alarm)
python scripts/run_benchmark.py --env smax       --experiment detection --full --map-name 3m
python scripts/run_benchmark.py --env smax       --experiment detection --full --map-name 2s3z

# Overcooked detection
python scripts/run_benchmark.py --env overcooked --experiment detection --full --layout cramped_room
```

Selection ablations (`--experiment selection`) work the same way; on the easily-learned
envs selection is expected to be neutral-to-slower (see the `ablation.py` docstring), so the
detection experiment is the headline on the heavy envs.

Each run writes:

- `code/results/m8_<env>_detection_full.png` — committed-style figure (matched-false-alarm
  calibration, detection-delay distributions, example traces).
- `code/outputs/m8_<env>_detection_full.json` — raw numbers (gitignored).

## Approximate GPU cost

On a Colab T4, `--full` SMAX `3m` detection (8 seeds × ~2M-step IQL + the Simu false-alarm
lab + the swap streams) is on the order of tens of minutes; Overcooked `cramped_room` is
similar. A smoke equivalent (`--smoke`) runs in CI-time on CPU and is what is verified in
this PR. Scale the budgets in the `SCALE` table down if you only have a short session.
