# conformal-marl

JAX / [JaxMARL](https://github.com/FLAIROX/JaxMARL) implementation accompanying the
AAMAS paper **"Conformal Replay for Off-Policy Multi-Agent RL."**

The project implements two algorithms on top of an IQL backbone:

1. **Conformal selection** of replay transitions compatible with other agents'
   current policies, and
2. a **conformal test-martingale drift detector**.

> **Status — Milestone M2 (IQL training backbone).**
> The uv project, the Simultaneous-Attack env (M1), and the IQL backbone (M2) are
> in place. M2 adds a pure-functional, jit+vmap-able IQL loop with a flashbax
> replay buffer, a recurrence-ready Q-network interface, and a swappable-sampler
> seam that later milestones (conformal selection / drift detection) plug into.

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
│   ├── samplers.py         # M2: swappable sampler seam (registry + `uniform`)
│   └── iql.py              # M2: pure-functional IQL (make_train -> train(rng))
├── scripts/
│   └── train.py            # M2: CLI trainer (Simu / simple_spread, multi-seed)
└── tests/
    ├── __init__.py
    ├── test_smoke.py            # toolchain smoke tests (jit/vmap + JaxMARL)
    ├── test_simultaneous_attack.py  # M1 reward-correctness + interface tests
    └── test_iql_learns.py       # M2 fast learning test + vmap/sampler/schema checks
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
