# conformal-marl

JAX / [JaxMARL](https://github.com/FLAIROX/JaxMARL) implementation accompanying the
AAMAS paper **"Conformal Replay for Off-Policy Multi-Agent RL."**

The project implements two algorithms on top of an IQL backbone:

1. **Conformal selection** of replay transitions compatible with other agents'
   current policies, and
2. a **conformal test-martingale drift detector**.

> **Status — Milestone M0 (project scaffold only).**
> This milestone sets up the uv project and a working Python / JAX / JaxMARL
> environment. No algorithm logic exists yet; feature modules land in later
> milestones.

## Layout

```
code/
├── pyproject.toml          # uv project, requires-python >=3.12,<3.13
├── README.md               # this file
├── conformal_marl/
│   └── __init__.py         # package marker + __version__ = "0.0.0"
└── tests/
    ├── __init__.py
    └── test_smoke.py       # toolchain smoke tests (jit/vmap + JaxMARL reset)
```

## Requirements

- **Python 3.12** (pinned via `.python-version`). JAX has no wheels for the
  system Python 3.14, so 3.12 is required.
- [**uv**](https://docs.astral.sh/uv/) for environment and dependency management.
- **macOS / Apple Silicon (arm64):** we use **JAX CPU** (`jax[cpu]`).
  Do **not** install `jax-metal` — it breaks JaxMARL.

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

- a trivial `jax.jit` + `jax.vmap` numeric computation, and
- importing JaxMARL, instantiating one MPE environment, resetting it with a
  `jax.random.PRNGKey`, and asserting per-agent observations are returned.

## MPE simple_spread environment id

The registered JaxMARL id for **MPE simple_spread** (discovered by inspecting the
installed `jaxmarl.registration` registry — see *Resolved versions* below) is:

```
MPE_simple_spread_v3
```

Later milestones depend on this id; instantiate the env with
`jaxmarl.make("MPE_simple_spread_v3")`.

## Resolved versions

Pinned exactly in `uv.lock`. Key versions resolved on M0 (Python 3.12.11):

- `jax` / `jaxlib`: **0.10.2**
- `jaxmarl`: **0.0.2**
- `flax`: **0.12.7**, `optax`: **0.2.8**, `flashbax`: **0.1.3**,
  `chex`: **0.1.92**, `numpy`: **2.4.6**

uv was left free to resolve versions; JaxMARL did not constrain `jax`/`jaxlib`
downward, so a current JAX (0.10.2) was selected. Imports and the MPE reset both
work on this combination (see the M0 PR verification output).
