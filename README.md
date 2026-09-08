# Conformal Replay for Off-Policy Multi-Agent Reinforcement Learning

This repository holds an AAMAS-format research paper and its JAX / [JaxMARL](https://github.com/FLAIROX/JaxMARL)
implementation. It studies the **perceived non-stationarity** that off-policy MARL
agents suffer when their co-learners' policies drift, and treats the *staleness* of a
stored transition as its *conformity* to the other agents' **current** policies —
measured with conformal prediction.

- **Paper (LaTeX):** [`AAMAS_2024_sample.tex`](AAMAS_2024_sample.tex) → [`AAMAS_2024_sample.pdf`](AAMAS_2024_sample.pdf)
- **Code (JAX/JaxMARL):** [`code/`](code/) — see [`code/README.md`](code/README.md) for the full module-by-module guide
- **GPU / Colab runbook:** [`code/RUN_ON_GPU.md`](code/RUN_ON_GPU.md)

## The idea in one paragraph

In off-policy MARL each agent perceives the other learners as part of its environment;
as their policies drift, the perceived dynamics become non-stationary and previously
collected transitions turn misleading. The usual remedy — shrinking the replay buffer —
discards the sample efficiency that motivates off-policy learning. We recast a stored
transition's staleness as its *conformity* to the other agents' current policies and
score it with a distribution-free, calibrated non-conformity score: an exact
`−log ρ` importance score when both past and present policies are available, a
ratio-free `−log π_cur` score when only the current policies can be queried, and a
learned model-based surprise score when the policies cannot be queried at all. From
that one score, three uses follow.

## Three contributions

1. **Theory — when selection starves.** The value off-policy MARL targets is a *perceived
   best-response* value. Reweighting replay toward currently-compatible transitions
   computes the best response of a **support-starved** perceived MDP, which coincides with
   the true best response only when the weights equal the exact importance ratio *and* the
   reweighted support covers the best-response support. Naively **filtering** for
   compatibility violates both and can move the fixed point away from the optimum by
   starving exploration (Theorem 4.2). We confirm this empirically — it is a deliberate
   **negative result**, not a bug.
2. **Certified replay.** Where the importance ratio is available, ρ-weighted conformal
   p-values restore calibration against the current law; where it is not, time-decay
   (NexCP) weights give a one-sided coverage guarantee that decays gracefully with the
   total-variation drift. Either way the replay batch is **certified** under quantified
   drift, not filtered.
3. **Detection (the headline).** A conformal **test martingale** flags when the perceived
   environment has drifted, with an *anytime* false-alarm guarantee via Ville's inequality,
   and drives a **two-channel controller** whose frozen alarm channel preserves the
   guarantee while an adaptive channel reads the drift rate. On a standard cooperative
   benchmark it detects co-learner drift faster than a TD-error heuristic (AES) at matched
   false-alarm, and the same martingale *certifies* the replay correction (Lemma 4.3).

## Environment suite

Implemented on an Independent Q-Learning (IQL) backbone via a central env factory
([`code/conformal_marl/envs/registry.py`](code/conformal_marl/envs/registry.py)):

| Env | Role |
| --- | --- |
| **Simultaneous-Attack** (custom) | diagnostic / martingale false-alarm lab; isolates co-learner non-stationarity in a single-state bandit |
| **MPE `simple_spread`** | primary drift-detection testbed (JaxMARL-native): the partner-swap detection headline and the return-recovery study both run here |
| **SMAX** | action-masked env (ratio loses support); used for training/selection — the current fixed-intervention detection benchmark guards masked envs out (see below) |
| **Overcooked** | partner-swap intervention corroborating the detection result at a smaller training budget |

## Repository layout

```
.
├── AAMAS_2024_sample.tex      # the paper (main deliverable)
├── AAMAS_2024_sample.pdf      # last compiled build (committed)
├── sample.bib                 # bibliography
├── aamas.cls, ACM-Reference-Format.bst   # AAMAS/acmart class + bst (vendored)
├── imgs/                      # report/explanatory figures used by the .tex
├── notebooks/colab_gpu.ipynb  # GPU runbook for the heavy benchmarks
├── code/                      # JAX/JaxMARL implementation (a uv project)
│   ├── conformal_marl/        # the package: envs, iql, conformal, martingale,
│   │                          #   controller, weighted_conformal, samplers, …
│   ├── scripts/               # runnable experiments & benchmarks
│   ├── tests/                 # toolchain and scientific validity tests
│   ├── results/               # committed camera-ready figures (*.png)
│   └── outputs/               # raw run JSON (gitignored; regenerable)
└── sources/                   # reference PDFs (gitignored — bulky/copyrighted)
```

## Building the paper

Requires a TeX distribution (`latexmk`; on macOS the TeX Live binaries live under
`/Library/TeX/texbin`). From the repo root:

```bash
latexmk -f -pdf -interaction=nonstopmode AAMAS_2024_sample.tex
```

This runs pdflatex/bibtex as needed and produces `AAMAS_2024_sample.pdf`. The `-f` (force)
flag lets it finish through benign font warnings. The class is the vendored
`aamas.cls` (acmart-based); no system install of the AAMAS class is needed.

## Running the code

The implementation is a self-contained [uv](https://docs.astral.sh/uv/) project under
[`code/`](code/), pinned to **Python 3.12** and **JAX 0.4.30** (see the constraint note
below). From `code/`:

```bash
uv sync                       # create .venv and install pinned deps
uv run pytest -q              # full toolchain + scientific validity suite

# headline experiments (CPU-smoke scale; GPU for full scale — see code/RUN_ON_GPU.md)
uv run python scripts/run_benchmark.py --env overcooked --experiment detection --smoke
uv run python scripts/support_starvation.py        # the support-starvation centerpiece
uv run python scripts/drift_demo.py                # martingale fires after a known swap
```

The cheap envs (Simultaneous-Attack, MPE `simple_spread`) train to a meaningful policy on
CPU in seconds-to-minutes; the heavy SMAX / Overcooked sweeps want a GPU — the Colab
runbook is [`code/RUN_ON_GPU.md`](code/RUN_ON_GPU.md) with
[`notebooks/colab_gpu.ipynb`](notebooks/colab_gpu.ipynb).

### Why JAX is pinned to 0.4.30 (do not bump)

JaxMARL 0.0.2 calls the `jax.tree_map` alias that was **removed in JAX 0.6.0**, so the
project holds JAX at the last 0.4.x release that still ships it (with `chex < 0.1.91`). A
regression guard in `code/tests/test_smoke.py` fails loudly if a dependency bump pulls JAX
forward again. Full rationale in [`code/README.md`](code/README.md#why-jax-is-pinned-to-0430).

## Results

Committed camera-ready figures live in [`code/results/`](code/results/) and feed the
paper's §5:

- `m6_support_starvation_collapse.png` — selection starves exploration on Simu (Theorem 4.2)
- `m8_detection_mpe.png`, `m5_detection_vs_aes.png` — detection beats the AES heuristic at matched false-alarm
- `m_calibration_pvalues*.png` — conformal p-values are conservatively super-uniform under H₀

## Status

The paper reads coherently end-to-end (abstract → §4 theory → §5 real JAX/JaxMARL results).
The full test suite passes. Known future work: full-scale convergence on the
masked / partially observable testbeds (SMAX, Overcooked), where the stored-state
GRU detector already runs but a mask-aware drift *intervention* is still needed for
the scripted benchmark (the current one guards masked envs out — see the note in
`code/RUN_ON_GPU.md`), and a conformal-occupancy "correct-the-target" probe.

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{conformalreplay_marl,
  title     = {Conformal Replay for Off-Policy Multi-Agent Reinforcement Learning},
  author    = {Gardille, Arnaud and Lahire, Thibault},
  booktitle = {Proc. of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year      = {2026},
  note      = {Under review}
}
```
