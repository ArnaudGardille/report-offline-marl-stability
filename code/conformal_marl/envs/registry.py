"""Central JaxMARL environment factory -- the single seam for M7.

Every script in the project used to instantiate envs ad hoc: a bare
``jaxmarl.make("MPE_simple_spread_v3")`` here, a ``SimultaneousAttack(...)`` there,
and a hand-rolled ``FlatObs`` wrapper *copy-pasted* into three different drift demos
to make Overcooked's image observations digestible by the (Dense-net) IQL backbone.
This module replaces all of that with one entry point::

    env, spec = make_env(name, **overrides)

* ``env`` is a ready-to-train JaxMARL env (already wrapped: image/structured obs
  flattened to a vector, action-masking exposed under the name the IQL loop expects).
  It is *not* ``LogWrapper``-wrapped -- :func:`conformal_marl.iql.make_train` does that
  itself, exactly as before, so wrapping order is unchanged.
* ``spec`` is an :class:`EnvSpec`: static metadata (agents, obs/action dims, whether
  the env is partially observable, whether it masks actions) plus a **default
  :class:`~conformal_marl.iql.IQLConfig`** tuned for that env -- in particular the
  right ``net_type`` (feed-forward for the memoryless MPE / Simu, **GRU** for the
  partially observable SMAX / Overcooked / Hanabi / Switch-Riddle).

Discovered registry (jaxmarl 0.0.2, the pinned version)
-------------------------------------------------------
``jaxmarl.registration.registered_envs`` lists, among others, the discrete-action
envs this factory targets (continuous-action MABrax -- ``ant_4x2`` etc. -- is **out
of scope**: IQL is discrete; those need MADDPG, deferred):

* ``MPE_simple_spread_v3`` and the rest of the MPE suite -- vector obs, 5 actions,
  fully observed for our purposes -> feed-forward.
* ``SMAX`` / ``HeuristicEnemySMAX`` / ``LearnedPolicyEnemySMAX`` -- SMAC in JAX.
  Constructed with ``scenario=map_name_to_scenario(map_name)`` (NOT ``map_name=``).
  Discovered scenarios: ``3m, 2s3z, 25m, 3s5z, 8m, 5m_vs_6m, 10m_vs_11m, 27m_vs_30m,
  3s5z_vs_3s6z, 3s_vs_5z, 6h_vs_8z, smacv2_5_units, smacv2_10_units,
  smacv2_20_units``. Per-agent vector obs, ``get_avail_actions(state)`` masking,
  partially observed -> GRU.
* ``overcooked`` -- ``layout=overcooked_layouts[name]``. Discovered layouts:
  ``cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit``.
  Image obs ``(H, W, 26)`` -> flattened; partially observed -> GRU.
* ``hanabi`` -- standard 2-player (5 colors, 5 ranks, hand 5), obs 1680, 21 moves.
  Masks via ``get_legal_moves(state)`` (no ``get_avail_actions``), so we wrap it to
  expose ``get_avail_actions``. Partially observed -> GRU.
* ``switch_riddle`` -- the lightbulb/prisoners memory puzzle. Tiny (obs ``(2,)``,
  3 actions, horizon ``4N-6``), genuinely requires memory -> GRU. The cheapest
  partially observable JaxMARL env -- used for the fast GRU smoke-learn test.
* ``coin_game`` -- 2-player grid; obs ``(3, 3, 4)`` -> flattened; -> GRU.
* ``simu`` -- the project's custom :class:`SimultaneousAttack` (one-step coordination
  bandit), memoryless -> feed-forward.

Known-incompatible (instantiable, but **not** runnable by the per-agent IQL loop):

* ``storm`` / ``storm_2p`` -- the in-the-Matrix games return obs as a *single* dict
  ``{"inventory", "observations"}`` keyed by field, NOT per-agent (``obs[agent]``),
  and their reward is not a per-agent dict. The IQL loop's per-agent
  ``obs[a] / rewards[a]`` contract does not hold, so STORM needs a bespoke adapter
  that is out of M7's scope. :func:`make_env` raises a clear ``NotImplementedError``
  pointing here, rather than failing deep inside ``make_train``.

Two thin wrappers live here (moved out of the scripts):

* :class:`FlatObs` -- flattens each agent's observation to a 1-D float vector and
  exposes the per-agent ``observation_space(agent)`` signature the IQL ``EnvInfo``
  reader expects. (The bare Overcooked/Coin-Game ``observation_space()`` takes no
  agent and returns an image shape.)
* :class:`AvailActionsWrapper` -- aliases an env's ``get_legal_moves(state)`` to the
  ``get_avail_actions(state)`` name the IQL loop probes for, so masked envs (Hanabi)
  mask correctly without any change to ``iql.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box

from conformal_marl.envs.simultaneous_attack import SimultaneousAttack
from conformal_marl.iql import IQLConfig


# --------------------------------------------------------------------------- wrappers
class FlatObs(MultiAgentEnv):
    """Flatten each agent's observation to a 1-D float vector.

    Several JaxMARL envs hand each agent a multi-dimensional observation (Overcooked's
    ``(H, W, C)`` image, Coin-Game's ``(3, 3, 4)`` grid) and expose
    ``observation_space()`` with *no* agent argument -- neither of which the IQL MLP /
    GRU backbone (a Dense trunk over a flat obs vector, with the MPE-style
    ``observation_space(agent)`` signature) consumes directly. This wrapper flattens the
    obs and re-exposes the per-agent ``observation_space(agent)`` signature, so
    ``make_train`` trains on these envs unchanged.

    No learning-relevant transform: a flat trunk over the flattened grid is a valid (if
    basic) Q-net. It delegates everything else (``action_space``, ``get_avail_actions``
    if present, state) to the wrapped env. This is the single, shared copy of what used
    to be duplicated across ``scripts/{drift_demo,closed_loop_demo,detection_benchmark}``.
    """

    def __init__(self, env: MultiAgentEnv) -> None:
        self._env = env
        self.agents = env.agents
        self.num_agents = env.num_agents
        # The wrapped obs space takes no agent arg and returns the image/grid shape.
        flat_dim = int(np.prod(env.observation_space().shape))
        self._obs_space = Box(-1e9, 1e9, (flat_dim,), dtype=jnp.float32)

    def _flat(self, obs: dict) -> dict:
        return {a: obs[a].reshape(-1).astype(jnp.float32) for a in self.agents}

    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._flat(obs), state

    def step(self, key, state, actions):
        obs, state, rewards, dones, info = self._env.step(key, state, actions)
        return self._flat(obs), state, rewards, dones, info

    def step_env(self, key, state, actions):
        obs, state, rewards, dones, info = self._env.step_env(key, state, actions)
        return self._flat(obs), state, rewards, dones, info

    def observation_space(self, agent: str = ""):
        del agent
        return self._obs_space

    def action_space(self, agent: str = ""):
        return self._env.action_space(agent)

    def get_obs(self, state):
        return self._flat(self._env.get_obs(state))

    def get_avail_actions(self, state):
        # Forward masking if the wrapped env (or its own wrapper) exposes it.
        return self._env.get_avail_actions(state)

    def __getattr__(self, name):
        # Delegate any other attribute (e.g. max_steps) to the wrapped env. Guarded so
        # the wrapped env set in __init__ is found via normal lookup first.
        return getattr(self.__dict__["_env"], name)


class AvailActionsWrapper(MultiAgentEnv):
    """Expose ``get_avail_actions(state)`` for envs that name it ``get_legal_moves``.

    The IQL loop probes ``hasattr(env, "get_avail_actions")`` and, if present, calls
    ``env.get_avail_actions(state)`` to build the per-agent action mask. Hanabi (and
    only Hanabi, among the in-scope envs) implements the very same thing under the name
    ``get_legal_moves``. Rather than special-case Hanabi inside ``iql.py``, this thin
    wrapper aliases the method, so masked envs Just Work through the unchanged loop.
    """

    def __init__(self, env: MultiAgentEnv) -> None:
        self._env = env
        self.agents = env.agents
        self.num_agents = env.num_agents

    def reset(self, key):
        return self._env.reset(key)

    def step(self, key, state, actions):
        return self._env.step(key, state, actions)

    def step_env(self, key, state, actions):
        return self._env.step_env(key, state, actions)

    def observation_space(self, agent: str = ""):
        return self._env.observation_space(agent)

    def action_space(self, agent: str = ""):
        return self._env.action_space(agent)

    def get_obs(self, state):
        return self._env.get_obs(state)

    def get_avail_actions(self, state):
        return self._env.get_legal_moves(state)

    def __getattr__(self, name):
        return getattr(self.__dict__["_env"], name)


# --------------------------------------------------------------------------- spec
@dataclass
class EnvSpec:
    """Static metadata + a tuned default config for one factory env.

    Attributes
    ----------
    name:
        The factory key passed to :func:`make_env`.
    num_agents / num_actions / obs_dim:
        Shapes read off the (wrapped) env once, outside any traced code. ``obs_dim`` is
        computed from an actual ``reset`` (robust to envs like Hanabi whose
        ``observation_space()`` returns an empty shape in jaxmarl 0.0.2).
    partially_observable:
        Whether the env hides state the optimal policy needs -> the default ``net_type``
        is ``"gru"`` when True, ``"ff"`` when False.
    masks_actions:
        Whether the (wrapped) env exposes ``get_avail_actions`` (so the IQL loop masks).
    default_config:
        A ready :class:`IQLConfig` with a sensible ``net_type`` / ``hidden_dim`` /
        budget for this env. Callers can ``replace``/mutate it; :func:`make_env` already
        applies any ``**overrides`` to it.
    """

    name: str
    num_agents: int
    num_actions: int
    obs_dim: int
    partially_observable: bool
    masks_actions: bool
    default_config: IQLConfig = field(default=None)  # type: ignore[assignment]


# --------------------------------------------------------------------------- builders
# Each builder returns a *raw* (possibly wrapped) JaxMARL env. ``make_env`` reads the
# shapes off it and attaches a default config. Builders take ``**kwargs`` so env-level
# knobs (SMAX map_name, Overcooked layout, agent count) can be overridden by the caller.

# JaxMARL envs whose observations are multi-dimensional and must be flattened for the
# Dense/GRU trunk.
_NEEDS_FLAT = {"overcooked", "coin_game"}


def _build_simu(num_agents: int = 2, **_: Any) -> MultiAgentEnv:
    return SimultaneousAttack(num_agents=num_agents)


def _build_mpe(jaxmarl_id: str) -> Callable[..., MultiAgentEnv]:
    def build(**kwargs: Any) -> MultiAgentEnv:
        return jaxmarl.make(jaxmarl_id, **kwargs)

    return build


def _build_smax(smax_id: str) -> Callable[..., MultiAgentEnv]:
    def build(map_name: str = "3m", **kwargs: Any) -> MultiAgentEnv:
        # SMAX takes a *scenario array* via ``scenario=``, derived from a SMAC map name.
        from jaxmarl.environments.smax import map_name_to_scenario

        scenario = map_name_to_scenario(map_name)
        return jaxmarl.make(smax_id, scenario=scenario, **kwargs)

    return build


def _build_overcooked(layout: str = "cramped_room", **kwargs: Any) -> MultiAgentEnv:
    from jaxmarl.environments.overcooked import overcooked_layouts

    base = jaxmarl.make("overcooked", layout=overcooked_layouts[layout], **kwargs)
    return FlatObs(base)


def _build_coin_game(**kwargs: Any) -> MultiAgentEnv:
    return FlatObs(jaxmarl.make("coin_game", **kwargs))


def _build_hanabi(**kwargs: Any) -> MultiAgentEnv:
    # Hanabi masks via ``get_legal_moves``; wrap to expose ``get_avail_actions``.
    return AvailActionsWrapper(jaxmarl.make("hanabi", **kwargs))


def _build_switch_riddle(**kwargs: Any) -> MultiAgentEnv:
    return jaxmarl.make("switch_riddle", **kwargs)


def _build_storm(*_: Any, **__: Any) -> MultiAgentEnv:
    raise NotImplementedError(
        "STORM ('storm'/'storm_2p') returns obs as a single {'inventory','observations'} "
        "dict (not per-agent obs[agent]) and a non-per-agent reward, so the per-agent IQL "
        "loop cannot consume it without a bespoke adapter (out of M7 scope). See "
        "conformal_marl/envs/registry.py."
    )


# name -> (builder, partially_observable). ``masks_actions`` / ``obs_dim`` are derived
# from the *built* env (so they stay correct under overrides), not hardcoded here.
_BUILDERS: dict[str, tuple[Callable[..., MultiAgentEnv], bool]] = {
    "simu": (_build_simu, False),
    "MPE_simple_spread_v3": (_build_mpe("MPE_simple_spread_v3"), False),
    "MPE_simple_reference_v3": (_build_mpe("MPE_simple_reference_v3"), False),
    "MPE_simple_speaker_listener_v4": (_build_mpe("MPE_simple_speaker_listener_v4"), False),
    "SMAX": (_build_smax("SMAX"), True),
    "HeuristicEnemySMAX": (_build_smax("HeuristicEnemySMAX"), True),
    "LearnedPolicyEnemySMAX": (_build_smax("LearnedPolicyEnemySMAX"), True),
    "overcooked": (_build_overcooked, True),
    "coin_game": (_build_coin_game, True),
    "hanabi": (_build_hanabi, True),
    "switch_riddle": (_build_switch_riddle, True),
    "storm": (_build_storm, True),
    "storm_2p": (_build_storm, True),
}

# Convenience: the subset the IQL loop can actually *run* (everything but STORM).
IN_SCOPE_ENVS: tuple[str, ...] = tuple(
    n for n in _BUILDERS if n not in ("storm", "storm_2p")
)


def available_envs() -> list[str]:
    """All factory env names (including the known-incompatible STORM placeholders)."""
    return sorted(_BUILDERS)


# --------------------------------------------------------------------------- config
def _default_config(spec_name: str, partially_observable: bool, num_actions: int) -> IQLConfig:
    """A small, sensible default :class:`IQLConfig` per env.

    The *only* structurally important choice is ``net_type``: ``"gru"`` for the
    partially observable envs (so they can in principle learn the memory-dependent
    policy), ``"ff"`` otherwise. The budgets here are deliberately modest -- the
    factory's job is to make every env *instantiate and run*; full convergence is a
    GPU/Colab job, configured in the scripts. Conformal-sampler knobs are left at their
    ``IQLConfig`` defaults (the ``uniform`` sampler ignores them).
    """
    net_type = "gru" if partially_observable else "ff"

    if spec_name == "simu":
        return IQLConfig(
            total_timesteps=40_000, num_envs=8, num_steps=8, num_grad_steps=8,
            learning_starts=200, hidden_dim=64, lr=1e-3, gamma=0.99, batch_size=64,
            buffer_size=10_000, target_update_interval=100, epsilon_anneal_time=15_000,
            eval_episodes=64, eval_steps=2, net_type=net_type,
        )
    if spec_name.startswith("MPE"):
        return IQLConfig(
            total_timesteps=500_000, num_envs=16, num_steps=16, num_grad_steps=16,
            learning_starts=5_000, hidden_dim=128, lr=5e-4, gamma=0.95, batch_size=256,
            buffer_size=200_000, target_update_interval=200, epsilon_anneal_time=200_000,
            eval_episodes=32, eval_steps=26, net_type=net_type,
        )
    if spec_name == "switch_riddle":
        # Tiny memory puzzle: short horizon, small net, GRU.
        return IQLConfig(
            total_timesteps=60_000, num_envs=16, num_steps=16, num_grad_steps=8,
            learning_starts=512, hidden_dim=64, lr=5e-4, gamma=0.99, batch_size=128,
            buffer_size=50_000, target_update_interval=100, epsilon_anneal_time=30_000,
            eval_episodes=64, eval_steps=12, net_type=net_type,
        )
    if "SMAX" in spec_name:
        return IQLConfig(
            total_timesteps=2_000_000, num_envs=16, num_steps=64, num_grad_steps=8,
            learning_starts=2_000, hidden_dim=128, lr=5e-4, gamma=0.99, batch_size=128,
            buffer_size=200_000, target_update_interval=200, epsilon_anneal_time=500_000,
            eval_episodes=16, eval_steps=128, net_type=net_type,
        )
    # overcooked / coin_game / hanabi: image/structured obs, sparse-ish reward, GRU.
    return IQLConfig(
        total_timesteps=500_000, num_envs=16, num_steps=64, num_grad_steps=4,
        learning_starts=512, hidden_dim=128, lr=5e-4, gamma=0.99, batch_size=128,
        buffer_size=100_000, target_update_interval=100, epsilon_anneal_time=350_000,
        eval_episodes=8, eval_steps=400, net_type=net_type,
    )


def _probe_shapes(env: MultiAgentEnv) -> tuple[int, int, int]:
    """``(num_agents, num_actions, obs_dim)`` read off a (wrapped) env via one reset.

    ``obs_dim`` comes from an actual ``reset`` rather than ``observation_space`` because
    some jaxmarl 0.0.2 envs (Hanabi) return an empty-shaped observation space. The reset
    is cheap and run once, outside any traced training code.
    """
    agents = list(env.agents)
    a0 = agents[0]
    num_actions = int(env.action_space(a0).n)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    obs_dim = int(np.prod(np.asarray(obs[a0]).shape))
    return len(agents), num_actions, obs_dim


def make_env(name: str, config_overrides: dict | None = None, **env_kwargs: Any):
    """Build a ready-to-train env and its :class:`EnvSpec`.

    Parameters
    ----------
    name:
        A factory key (see :func:`available_envs`): the custom ``"simu"``, an MPE id
        (``"MPE_simple_spread_v3"``), ``"SMAX"`` / ``"HeuristicEnemySMAX"`` /
        ``"LearnedPolicyEnemySMAX"``, ``"overcooked"``, ``"coin_game"``, ``"hanabi"``,
        or ``"switch_riddle"``. (``"storm"`` / ``"storm_2p"`` raise -- see the module
        docstring.)
    config_overrides:
        Optional mapping of :class:`IQLConfig` field -> value, applied to the env's
        default config (e.g. ``{"total_timesteps": 1000, "num_envs": 2}`` to make a CI
        smoke run). ``net_type`` can be overridden here too.
    **env_kwargs:
        Forwarded to the underlying builder / ``jaxmarl.make`` -- e.g.
        ``map_name="2s3z"`` for SMAX, ``layout="coord_ring"`` for Overcooked,
        ``num_agents=3`` for ``simu`` / ``switch_riddle``.

    Returns
    -------
    ``(env, spec)``: the wrapped JaxMARL env (NOT LogWrapped -- ``make_train`` does that)
    and its :class:`EnvSpec` carrying the default (override-applied) :class:`IQLConfig`.
    """
    if name not in _BUILDERS:
        raise KeyError(
            f"unknown env {name!r}; available: {available_envs()}"
        )
    builder, partially_observable = _BUILDERS[name]
    env = builder(**env_kwargs)

    num_agents, num_actions, obs_dim = _probe_shapes(env)
    masks_actions = hasattr(env, "get_avail_actions")

    cfg = _default_config(name, partially_observable, num_actions)
    if config_overrides:
        for k, v in config_overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"IQLConfig has no field {k!r}")
            setattr(cfg, k, v)

    spec = EnvSpec(
        name=name,
        num_agents=num_agents,
        num_actions=num_actions,
        obs_dim=obs_dim,
        partially_observable=partially_observable,
        masks_actions=masks_actions,
        default_config=cfg,
    )
    return env, spec
