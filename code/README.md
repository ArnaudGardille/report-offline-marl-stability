# conformal-marl

JAX / [JaxMARL](https://github.com/FLAIROX/JaxMARL) implementation accompanying the
AAMAS paper **"Conformal Replay for Off-Policy Multi-Agent RL."**

The project implements two algorithms on top of an IQL backbone:

1. **Conformal selection** of replay transitions compatible with other agents'
   current policies, and
2. a **conformal test-martingale drift detector**.

> **Status — Milestone M4 (conformal drift-detection martingale).**
> In place: the uv project, the Simultaneous-Attack env (M1), the IQL backbone
> (M2), the conformal scores + p-values + selection samplers (M3), and now the
> conformal **test-martingale drift detector** (M4): a score-agnostic, jit/scan-based
> mixture power-martingale over a p-value stream with Ville's-inequality alarms and
> reset-on-alarm, validated against `E[M_t]=1` / Ville false-alarm control, with a
> controlled-swap demo on Simultaneous-Attack and Overcooked.

## Layout

```
code/
├── pyproject.toml          # uv project, requires-python >=3.12,<3.13
├── README.md               # this file
├── conformal_marl/
│   ├── __init__.py         # package marker + __version__ = "0.0.0"
│   ├── envs/
│   │   └── simultaneous_attack.py   # M1: 1-step cooperative coordination bandit
│   ├── networks.py         # M2: recurrence-ready Q-network (FF cell + GRU drop-in)
│   ├── replay.py           # M2: Transition schema + flashbax item buffer
│   ├── samplers.py         # M2/M3: swappable sampler seam (uniform + conformal)
│   ├── iql.py              # M2: pure-functional IQL (make_train -> train(rng))
│   ├── conformal.py        # M3: non-conformity scores + inductive conformal p-values
│   └── martingale.py       # M4: conformal drift-detection test martingale (Alg. 2)
├── scripts/
│   ├── train.py            # M2: CLI trainer (Simu / simple_spread, multi-seed)
│   ├── ablation.py         # M3: uniform vs conformal selection ablation
│   └── drift_demo.py       # M4: controlled-swap drift demo (Simu + Overcooked)
└── tests/
    ├── __init__.py
    ├── test_smoke.py            # toolchain smoke tests (jit/vmap + JaxMARL)
    ├── test_simultaneous_attack.py  # M1 reward-correctness + interface tests
    ├── test_iql_learns.py       # M2 fast learning test + vmap/sampler/schema checks
    ├── test_conformal.py        # M3 score algebra + p-value super-uniformity
    └── test_martingale.py       # M4 E[M_t]=1 / Ville / detection validity gate
```

## M2 — IQL backbone

`conformal_marl.iql.make_train(config, env)` returns a pure `train(rng) -> metrics`
closure: the whole run is a `jax.lax.scan`, `jax.jit`-compiled and `jax.vmap`-able
over a batch of seeds. Collection follows the PureJaxRL pattern (step `num_envs`
parallel envs for `num_steps`, then do `num_grad_steps` gradient updates, scanning
`num_updates` times) so compile cost scales with the chunk length, not the total
budget.

Key pieces and their extension points:

- **Network** (`networks.py`): every Q-net speaks the recurrent contract
  `apply(params, carry, (obs, resets)) -> (carry, q)` plus `initialize_carry`. The
  default `QNetwork` is a feed-forward MLP whose carry is a zero-width placeholder
  threaded unchanged; `GRUQNetwork` is a committed, same-signature recurrent
  drop-in for M5 (SMAX / Overcooked).
- **Transition schema** (`replay.py`): per-agent `obs`/`next_obs`, the **joint**
  `actions` (so other agents' actions `u_{-a}` are recoverable), per-agent
  `reward`/`done`, `avail_actions` mask, and the per-agent epsilon-greedy
  **`behavior_log_prob`** of the taken action — over-provisioned so M3/M5 never
  change the buffer format.
- **Sampler seam** (`samplers.py`): `sampler(buffer, buffer_state, rng, ctx,
  batch_size) -> (batch, weights, info)`, registered by name. `uniform` returns
  unit weights; the TD loss multiplies the per-sample squared error by `weights`.
  `ctx` (a `SamplerContext`) carries online/target params, the network apply fn,
  and env metadata, and the loop supports drawing `L` then training on `B<=L`
  (LaBER-style) — so M3 can add `conformal_policy` / `exact_ratio` samplers
  **without editing the training loop**.

Run real trainings (bigger than the committed fast test):

```bash
uv run python scripts/train.py --env simu   --seeds 8   # Simultaneous-Attack N=2
uv run python scripts/train.py --env spread --seeds 3   # MPE simple_spread
uv run python scripts/train.py --env both
```

## M4 — conformal drift-detection martingale

`conformal_marl.martingale` implements the paper's Algorithm 2: an *anytime-valid*
sequential test that consumes a stream of conformal p-values and alarms when the
perceived environment has **drifted**. It is **score-agnostic** (takes p-values, not
transitions), so the same detector works for any conditionally super-uniform stream.

- **Mixture power martingale.** For a fixed mixing parameter `ε`,
  `M_t^ε = Π_{j≤t} ε·p_j^{ε-1}` is a non-negative martingale with `E[M_t^ε]=1` under the
  null (each factor has mean `ε·∫₀¹ p^{ε-1}dp = 1`). To avoid choosing `ε`, the detector
  averages over a grid `ε ∈ {0.1,…,0.9}` (a uniform-prior approximation of the mixture
  `∫₀¹ … dε`; the single-factor closed form `∫₀¹ ε p^{ε-1}dε = (1/p+ln p−1)/(ln p)²` is
  kept as a reference in `mixture_closed_form_factor`). The grid average is itself a
  convex combination of unit-mean martingales, so `E[M_t]=1` exactly.
- **Log-space + `lax.scan`.** Per-grid running log-sums `L_t^ε = Σ_{j≤t}[ln ε+(ε−1)ln p_j]`
  are carried through a `jax.lax.scan`; the mixture is `logsumexp(L_t^ε)−ln(n_grid)`. This
  is numerically stable (`M_t` overflows float under strong drift; `log M_t` stays finite)
  and fully `jit`/`vmap`-able.
- **Ville's inequality + reset.** Under the null, `P(sup_t M_t ≥ 1/δ) ≤ δ`. The detector
  **alarms** when `M_t ≥ 1/δ`, then resets `M←1` (via `lax.cond` inside the scan) to
  re-arm for the next change point.

```bash
uv run python scripts/drift_demo.py                  # Simu + Overcooked controlled-swap demo
uv run python scripts/drift_demo.py --no-overcooked  # Simu only (fastest)
```

The demo trains a short IQL policy, **fixes** a pre-drift calibration set, then at a known
step swaps one agent's policy (Simu: flip the partner's action; Overcooked: a scripted
non-conforming partner) and feeds the incoming transitions' p-values to the martingale.
Figures land in `results/m4_drift_demo_{simu,overcooked}.png`; the detector is controlled
before the swap and crosses `1/δ` shortly after. (Note: Overcooked's task reward is sparse
and stays ~0 at the demo's tiny training budget — but the drift signal depends only on how
(un)likely the partner's actions are under the learner's policy, not on reward.)

## Requirements

- **Python 3.12** (pinned via `.python-version`). JAX has no wheels for the
  system Python 3.14, so 3.12 is required.
- [**uv**](https://docs.astral.sh/uv/) for environment and dependency management.
- **macOS / Apple Silicon (arm64):** we use **JAX CPU** (`jax[cpu]`).
  Do **not** install `jax-metal` — it breaks JaxMARL.
- **JAX is pinned to 0.4.30** (not the latest). JaxMARL 0.0.2 calls the
  `jax.tree_map` alias that was *removed* in JAX 0.6.0, so we hold JAX at the last
  0.4.x release that still provides it. See *Resolved versions* below.

## Setup

From this `code/` directory:

```bash
uv sync
```

This creates `.venv/` and installs all runtime and dev dependencies. The first
run pulls in many JaxMARL transitive deps (mujoco, wandb, matplotlib, ...) and
can take several minutes.

Runtime deps: `jaxmarl`, `jax[cpu]`, `flax`, `optax`, `flashbax`, `chex`, `numpy`.
Dev deps: `pytest`, `ruff`. Resolved versions are pinned in `uv.lock` (committed).

## Quick sanity check

```bash
uv run python -c "import jax, jaxmarl, flax, optax, flashbax, chex; \
print('jax', jax.__version__); print(jax.devices()); \
print(jax.vmap(jax.jit(lambda x: x*x))(jax.numpy.arange(5)).sum())"
```

You should see a CPU device (e.g. `[CpuDevice(id=0)]`) and `30`.

## Running tests

```bash
uv run pytest -q
```

`tests/test_smoke.py` exercises the toolchain end-to-end:

- a trivial `jax.jit` + `jax.vmap` numeric computation;
- importing JaxMARL, instantiating one MPE environment, resetting it with a
  `jax.random.PRNGKey`, and asserting per-agent observations are returned; and
- a **foundation regression guard** for the JAX pin: it asserts
  `jax.tree_map` exists, steps an MPE env through the *inherited* base
  `MultiAgentEnv.step` (the path that broke on JAX ≥ 0.6), and runs a short
  `jax.lax.scan` rollout under `jax.jit` (what M2's training loop relies on).
  These fail loudly if a dependency bump silently pulls JAX forward again.

## MPE simple_spread environment id

The registered JaxMARL id for **MPE simple_spread** (discovered by inspecting the
installed `jaxmarl.registration` registry — see *Resolved versions* below) is:

```
MPE_simple_spread_v3
```

Later milestones depend on this id; instantiate the env with
`jaxmarl.make("MPE_simple_spread_v3")`.

## Resolved versions

Pinned exactly in `uv.lock`. Key versions (Python 3.12.11):

- `jax` / `jaxlib`: **0.4.30** (pinned, with the `cpu` extra)
- `jaxmarl`: **0.0.2** (pinned)
- `flax`: **0.10.4**, `optax`: **0.2.5**, `flashbax`: **0.1.3**,
  `chex`: **0.1.90** (`<0.1.91`), `numpy`: **2.4.6**

### Why JAX is pinned to 0.4.30

M0 originally left uv free to resolve, which picked **JAX 0.10.2**. But
**JaxMARL 0.0.2** was written for an older JAX: its base
`MultiAgentEnv.step` (and the `baselines/QLearning` scripts forked at M2) call
the `jax.tree_map` alias, which JAX **removed in 0.6.0**. On 0.10.2 *any* env
stepped through the base class raised
`AttributeError: module 'jax' has no attribute 'tree_map'`.

The fix aligns JAX downward (JaxMARL stays at 0.0.2; no monkeypatching):

- **`jax`/`jaxlib` 0.4.30** — the last 0.4.x release that still ships
  `jax.tree_map` (deprecated there, removed at 0.6.0), with cp312 macOS-arm64
  wheels, and ≥ every downstream floor.
- **`chex < 0.1.91`** — chex 0.1.91+ requires `jax>=0.7.0` (no `tree_map`), so it
  is capped to keep a `tree_map`-bearing JAX. uv then co-resolves
  `flax 0.10.4` / `optax 0.2.5` / `flashbax 0.1.3` / `orbax-checkpoint 0.6.4`
  consistently. flashbax (jax≥0.4.25, no upper bound) does not constrain JAX
  upward.

The regression is guarded by `tests/test_smoke.py` (see below), which fails
loudly if a future bump pulls JAX past 0.5.x again.
