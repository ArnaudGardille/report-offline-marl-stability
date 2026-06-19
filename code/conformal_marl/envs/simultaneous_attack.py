"""Simultaneous-Attack: a one-step cooperative coordination bandit for JaxMARL.

The environment is a single-state, single-step cooperative game with ``N`` agents.
Each agent picks one of two actions: ``0 = abstain`` or ``1 = attack``. Writing
``K = sum_i a_i`` for the number of attackers, every agent receives the *same*
(team) reward::

    r(a) = +1                if K == N        (all attack  -> success)
    r(a) = -r_death * K      if 0 < K < N     (partial attack -> "suicidal")
    r(a) =  0                if K == 0         (all abstain)

with the death penalty derived from the agent count::

    r_death = 1 / ( N * (2**(N - 1) - 1) ).

This particular ``r_death`` is the unique value (for fixed ``N >= 2``) that makes a
*uniformly random* joint policy have expected reward exactly ``0`` while the
all-attack joint action attains the optimum ``+1``. The point is to give learning a
non-trivial coordination problem: random play is worthless, partial coordination is
actively punished, and only full agreement pays off.

Proof of ``E_uniform[r] = 0``.
    Under the uniform joint policy each of the ``2**N`` joint actions is equally
    likely, and ``P(K = k) = C(N, k) * 2**(-N)``. Hence

        E[r] = 2**(-N) * ( 1 - r_death * sum_{k=1}^{N-1} k * C(N, k) ).

    Using ``sum_{k=0}^{N} k * C(N, k) = N * 2**(N-1)`` and dropping the ``k = 0`` and
    ``k = N`` terms (the ``k = N`` term being ``N`` itself),

        sum_{k=1}^{N-1} k * C(N, k) = N * 2**(N-1) - N = N * (2**(N-1) - 1),

    so the bracket vanishes for ``r_death = 1 / (N * (2**(N-1) - 1))`` and ``E[r] = 0``.

Design notes
------------
* **One step / always terminal.** Every call to :meth:`step` is terminal: each
  agent's ``done`` flag and the special ``"__all__"`` flag are set to ``True``. The
  base-class :meth:`MultiAgentEnv.step` therefore auto-resets the *returned* state and
  observation on every transition (the reward and done dicts still reflect the
  transition that just happened, computed before the reset).
* **Trivial observation.** Because the game has a single state, the observation
  carries no information. We hand each agent a fixed one-hot encoding of its own index
  (a length-``N`` float vector). This is constant across time and independent of the
  actions; it merely gives each agent a stable identity, which is convenient for
  downstream networks. The choice is arbitrary -- a zeros vector would work equally
  well -- and is documented here so tests can assert against it.
* **No action masking.** Both actions are always legal for every agent.
* **Pure / jittable.** :meth:`reset`, :meth:`step` and :meth:`step_env` contain no
  Python-side control flow that depends on traced values, so ``jax.jit`` and
  ``jax.vmap`` (e.g. over a batch of PRNG keys, or inside a ``lax.scan`` training
  loop) work directly.
"""

from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

# Action constants (documented for readers; the env logic only relies on the fact
# that attack == 1 and abstain == 0 so that ``sum_i a_i`` counts the attackers).
ABSTAIN = 0
ATTACK = 1


def default_r_death(num_agents: int) -> float:
    """Death penalty coefficient that makes uniform-policy reward zero.

    ``r_death = 1 / ( N * (2**(N - 1) - 1) )``. Defined for ``N >= 2`` (for ``N == 1``
    the denominator is ``0`` and the partial-attack regime ``0 < K < N`` is empty).
    """
    if num_agents < 2:
        raise ValueError(
            f"SimultaneousAttack requires num_agents >= 2 for the default r_death "
            f"(got {num_agents}); the partial-attack regime is empty otherwise."
        )
    return 1.0 / (num_agents * (2 ** (num_agents - 1) - 1))


@struct.dataclass
class State:
    """Environment state.

    The game is single-state, so there is nothing dynamic to carry. We keep ``done``
    and ``step`` to match the JaxMARL :class:`~jaxmarl.environments.multi_agent_env.State`
    convention and to interoperate with the auto-reset logic in the base ``step``.
    """

    done: chex.Array  # scalar bool, shared by all agents
    step: chex.Array  # scalar int step counter


class SimultaneousAttack(MultiAgentEnv):
    """One-step cooperative "simultaneous attack" coordination bandit.

    Parameters
    ----------
    num_agents:
        Number of agents ``N`` (default 2). Must be ``>= 2``.
    r_death:
        Optional override for the death-penalty coefficient. When ``None`` (the
        default) it is derived from ``num_agents`` via :func:`default_r_death`, which
        guarantees ``E_uniform[r] = 0``.
    """

    def __init__(self, num_agents: int = 2, r_death: Optional[float] = None) -> None:
        super().__init__(num_agents)
        if num_agents < 2:
            raise ValueError(
                f"SimultaneousAttack requires num_agents >= 2 (got {num_agents})."
            )

        self.agents = [f"agent_{i}" for i in range(num_agents)]

        # Death penalty: derived from N unless explicitly overridden.
        self.r_death = float(default_r_death(num_agents) if r_death is None else r_death)

        # Two discrete actions per agent: 0 = abstain, 1 = attack.
        self.action_spaces = {a: Discrete(2) for a in self.agents}
        # Trivial observation: a fixed one-hot of the agent index (length N, float32).
        self.observation_spaces = {
            a: Box(0.0, 1.0, (num_agents,), dtype=jnp.float32) for a in self.agents
        }

    # ------------------------------------------------------------------ reset
    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset to the (single) initial state and return per-agent observations."""
        del key  # the initial state is deterministic; no randomness needed
        state = State(done=jnp.array(False), step=jnp.array(0))
        return self.get_obs(state), state

    # ------------------------------------------------------------------- step
    #
    # We deliberately do NOT override ``step`` here: the inherited
    # :meth:`MultiAgentEnv.step` already implements the JaxMARL auto-reset convention
    # (compute the transition via :meth:`step_env`, then ``jax.lax.select`` the freshly
    # reset state/obs wherever ``dones["__all__"]`` is true). It works on the pinned
    # JAX (0.4.30) because that release still provides ``jax.tree_map``; the foundation
    # smoke guard in ``tests/test_smoke.py`` protects that assumption. Because every
    # step of this environment is terminal, the *returned* ``obs``/``state`` are always
    # the post-reset values, while ``rewards``/``dones`` describe the transition that
    # just occurred.

    # --------------------------------------------------------------- step_env
    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[
        Dict[str, chex.Array],
        State,
        Dict[str, chex.Array],
        Dict[str, chex.Array],
        Dict,
    ]:
        """Apply the joint action and return ``(obs, state, rewards, dones, info)``."""
        del key  # transition is deterministic given the actions

        # Stack the per-agent integer actions in canonical agent order.
        action_vec = jnp.array([actions[a] for a in self.agents]).reshape(-1)
        num_attackers = jnp.sum(action_vec == ATTACK)

        reward = self._reward(num_attackers)

        done = jnp.array(True)
        next_state = State(done=done, step=state.step + 1)

        obs = self.get_obs(next_state)
        rewards = {a: reward for a in self.agents}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        info = {"num_attackers": num_attackers}

        return obs, next_state, rewards, dones, info

    def _reward(self, num_attackers: chex.Array) -> chex.Array:
        """Shared team reward as a function of the number of attackers ``K``.

        ``+1`` if ``K == N``; ``-r_death * K`` if ``0 < K < N``; ``0`` if ``K == 0``.
        Written with :func:`jnp.where` so it is branch-free and jittable.
        """
        k = num_attackers.astype(jnp.float32)
        all_attack = num_attackers == self.num_agents
        none_attack = num_attackers == 0
        partial_reward = -self.r_death * k  # valid in the 0 < K < N regime
        reward = jnp.where(
            all_attack,
            1.0,
            jnp.where(none_attack, 0.0, partial_reward),
        )
        return reward.astype(jnp.float32)

    # ---------------------------------------------------------------- get_obs
    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Per-agent observation: a fixed one-hot of the agent's own index.

        The single-state game has no informative observation; the one-hot simply gives
        each agent a stable identity. It does not depend on ``state``.
        """
        del state
        one_hots = jnp.eye(self.num_agents, dtype=jnp.float32)
        return {a: one_hots[i] for i, a in enumerate(self.agents)}

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__
