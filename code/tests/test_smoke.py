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


# --------------------------------------------------------------------------------
# Foundation regression guard.
#
# JaxMARL 0.0.2 was written against an older JAX: its base ``MultiAgentEnv.step``
# (and the ``baselines/QLearning`` scripts forked at M2) call ``jax.tree_map``,
# which was *removed* in JAX 0.6.0. The project therefore pins JAX to 0.4.30, the
# last 0.4.x series that still carries ``jax.tree_map``. The tests below fail loudly
# if a future dependency bump silently pulls JAX forward again and reintroduces the
# ``AttributeError: module 'jax' has no attribute 'tree_map'`` that motivated this
# pin -- so they guard the *foundation*, not any project feature.
# --------------------------------------------------------------------------------
def test_jax_has_tree_map():
    """``jax.tree_map`` must exist (removed in JAX 0.6.0; required by JaxMARL 0.0.2)."""
    assert hasattr(jax, "tree_map"), (
        "jax.tree_map is missing -- JAX was bumped past 0.5.x and JaxMARL 0.0.2's "
        "base MultiAgentEnv.step will raise AttributeError. Re-pin JAX to a "
        "tree_map-bearing release (see code/README.md 'Resolved versions')."
    )


def test_jaxmarl_base_step_runs_unpatched():
    """Step an MPE env through the *inherited* base ``MultiAgentEnv.step``.

    This is the exact code path that broke on modern JAX: the base ``step`` calls
    ``jax.tree_map`` for its auto-reset. We invoke ``MultiAgentEnv.step`` explicitly
    (NOT any subclass override) so the test proves JaxMARL works *unpatched* on the
    pinned JAX -- it must run without ``AttributeError`` and return per-agent
    rewards/dones.
    """
    import jaxmarl
    from jaxmarl.environments.multi_agent_env import MultiAgentEnv

    env = jaxmarl.make(MPE_SIMPLE_SPREAD_ID)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Sample a legal action for each agent from its action space.
    key, key_act = jax.random.split(key)
    actions = {
        agent: env.action_space(agent).sample(jax.random.fold_in(key_act, i))
        for i, agent in enumerate(env.agents)
    }

    # Call the *base-class* step directly -- this is what M2's forked baselines rely
    # on, and what raised AttributeError on JAX >= 0.6.
    obs2, state2, rewards, dones, infos = MultiAgentEnv.step(env, key, state, actions)

    # Per-agent rewards and dones, plus the JaxMARL "__all__" aggregate done.
    assert isinstance(rewards, dict) and isinstance(dones, dict)
    for agent in env.agents:
        assert agent in rewards, f"missing reward for agent {agent!r}"
        assert agent in dones, f"missing done for agent {agent!r}"
        assert jnp.ndim(rewards[agent]) == 0  # scalar per-agent reward
    assert "__all__" in dones


def test_jaxmarl_scan_jit_rollout():
    """A short ``jax.lax.scan`` rollout of an MPE env under ``jax.jit``.

    M2's training loop drives JaxMARL envs inside a ``lax.scan`` carried under
    ``jax.jit``. This proves scan + jit over a JaxMARL env (auto-reset and all) works
    on the pinned JAX. We use the base ``env.step`` (auto-resetting) so episodes that
    terminate mid-rollout keep producing valid carries.
    """
    import jaxmarl

    env = jaxmarl.make(MPE_SIMPLE_SPREAD_ID)
    num_steps = 5

    def sample_actions(key):
        return {
            agent: env.action_space(agent).sample(jax.random.fold_in(key, i))
            for i, agent in enumerate(env.agents)
        }

    @jax.jit
    def rollout(key):
        reset_key, scan_key = jax.random.split(key)
        _, init_state = env.reset(reset_key)

        def body(carry, step_key):
            state = carry
            act_key, step_subkey = jax.random.split(step_key)
            actions = sample_actions(act_key)
            _, next_state, rewards, dones, _ = env.step(step_subkey, state, actions)
            # Stack per-agent rewards into a single vector for the scan output.
            reward_vec = jnp.stack([rewards[a] for a in env.agents])
            return next_state, (reward_vec, dones["__all__"])

        step_keys = jax.random.split(scan_key, num_steps)
        _, (rewards_seq, dones_seq) = jax.lax.scan(body, init_state, step_keys)
        return rewards_seq, dones_seq

    rewards_seq, dones_seq = rollout(jax.random.PRNGKey(0))

    # Shapes: (num_steps, num_agents) rewards and (num_steps,) aggregate dones.
    assert rewards_seq.shape == (num_steps, len(env.agents))
    assert dones_seq.shape == (num_steps,)
    # All finite -- the scan produced real numeric rewards at every step.
    assert bool(jnp.all(jnp.isfinite(rewards_seq)))
