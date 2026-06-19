"""End-to-end smoke tests for the M0 toolchain.

These prove that JAX (CPU), jit/vmap, and JaxMARL all import and run together.
No project algorithm logic is exercised here (none exists yet at M0).
"""

import jax
import jax.numpy as jnp


# The registered JaxMARL env id for MPE simple_spread, discovered by inspecting
# the installed jaxmarl.registration registry. Recorded in code/README.md too.
MPE_SIMPLE_SPREAD_ID = "MPE_simple_spread_v3"


def test_jit_vmap_numeric():
    """A trivial jit + vmap numeric computation returns the expected result."""
    square = jax.jit(lambda x: x * x)
    out = jax.vmap(square)(jnp.arange(5))
    # 0 + 1 + 4 + 9 + 16 == 30
    assert int(out.sum()) == 30
    # vmap preserves the leading axis.
    assert out.shape == (5,)


def test_jax_runs_on_cpu():
    """JAX exposes at least one device and it is a CPU device."""
    devices = jax.devices()
    assert len(devices) >= 1
    assert any(d.platform == "cpu" for d in devices)


def test_jaxmarl_mpe_simple_spread_reset():
    """JaxMARL imports; an MPE env instantiates and resets with per-agent obs."""
    import jaxmarl

    env = jaxmarl.make(MPE_SIMPLE_SPREAD_ID)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # obs is a dict keyed by agent name; assert we got one entry per agent.
    assert isinstance(obs, dict)
    assert len(env.agents) > 0
    for agent in env.agents:
        assert agent in obs, f"missing observation for agent {agent!r}"
        # Each per-agent observation is a non-empty JAX array.
        assert obs[agent].ndim >= 1
        assert obs[agent].shape[0] > 0
