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

* ``storm_2p`` -- the 2-player in-the-Matrix game (JaxMARL ``InTheGrid_2p``), which
  exposes tuple-valued agent data and structured observations. :class:`StormAdapter`
  maps that API to the per-agent dictionaries consumed by IQL and flattens each
  agent's view and inventory. The factory uses a symmetric common-payoff matrix by
  default, making this entry a cooperative coordination benchmark rather than the
  library's general-sum Prisoner's Dilemma. Exactly 2 agents (the underlying env
  hardcodes a red/blue pair); other counts raise from the factory.

Known-incompatible (instantiable, but **not** runnable by the per-agent IQL loop):

* ``storm`` -- the *N-player* in-the-Matrix game (JaxMARL ``InTheGrid``) returns obs
  as a single ``{"observations", "inventory"}`` dict batched over agents, NOT the
  per-agent tuple ``storm_2p`` uses, so :class:`StormAdapter` does not apply. Use
  ``storm_2p``.

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
from conformal_marl.iql import IQLConfig, get_env_info


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

    def __getattr__(self, name):
        # Delegate any other attribute to the wrapped env -- crucially including
        # ``get_avail_actions``, but *only* if the wrapped env actually has it. This way
        # ``hasattr(FlatObs(env), "get_avail_actions")`` is True exactly when the
        # underlying env masks (so the IQL loop masks for Hanabi-via-FlatObs but not for
        # the unmasked Overcooked / Coin-Game). Guarded so the wrapped env set in
        # __init__ is found via normal lookup first.
        return getattr(self.__dict__["_env"], name)


class CastObs(MultiAgentEnv):
    """Cast each agent's observation to ``float32`` (shape unchanged).

    Some JaxMARL envs hand out integer observations (Switch-Riddle ``int32``,
    Coin-Game ``int8``). The IQL backbone stores obs in a ``float32`` flashbax buffer
    and feeds them to Dense layers, and ``LogWrapper`` asserts the reset/step obs dtypes
    match -- both of which require a consistent floating dtype. This wrapper casts obs to
    ``float32`` without flattening (use :class:`FlatObs` when a flatten is also needed;
    it already casts). Applied automatically by :func:`make_env` to envs whose reset obs
    is not floating point.
    """

    def __init__(self, env: MultiAgentEnv) -> None:
        self._env = env
        self.agents = env.agents
        self.num_agents = env.num_agents

    def _cast(self, obs: dict) -> dict:
        return {a: obs[a].astype(jnp.float32) for a in self.agents}

    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._cast(obs), state

    def step(self, key, state, actions):
        obs, state, rewards, dones, info = self._env.step(key, state, actions)
        return self._cast(obs), state, rewards, dones, info

    def step_env(self, key, state, actions):
        obs, state, rewards, dones, info = self._env.step_env(key, state, actions)
        return self._cast(obs), state, rewards, dones, info

    def observation_space(self, agent: str = ""):
        return self._env.observation_space(agent)

    def action_space(self, agent: str = ""):
        return self._env.action_space(agent)

    def get_obs(self, state):
        return self._cast(self._env.get_obs(state))

    def __getattr__(self, name):
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


class HanabiWrapper(AvailActionsWrapper):
    """Hanabi adapter: avail-actions aliasing **and** action-shape normalisation.

    Hanabi has three jaxmarl 0.0.2 quirks the per-agent IQL loop trips on; this wrapper
    (plus the loop's reset-based ``obs_dim`` probe) papers over all of them:

    * masks via ``get_legal_moves`` -> aliased to ``get_avail_actions`` (inherited from
      :class:`AvailActionsWrapper`);
    * an *empty* ``observation_space()`` shape -> handled by probing obs from a reset;
    * ``step`` expects each agent's action shaped ``(1,)``, not a scalar -> this wrapper
      reshapes scalar actions to ``(1,)`` on the way in. (Obs / reward / done come back in
      the standard per-agent form, so nothing else changes.)
    """

    def _reshape(self, actions: dict) -> dict:
        return {a: jnp.reshape(actions[a], (1,)) for a in self.agents}

    def step(self, key, state, actions):
        return self._env.step(key, state, self._reshape(actions))

    def step_env(self, key, state, actions):
        return self._env.step_env(key, state, self._reshape(actions))


class StormAdapter(MultiAgentEnv):
    """Adapt JaxMARL ``storm_2p``'s tuple API to the shared per-agent IQL contract.

    JaxMARL 0.0.2 represents each observation as ``(view, inventory)`` inside a
    tuple indexed by integer agent id. Actions and rewards are tuples too, while the
    done mapping contains only ``"__all__"``. The project's IQL loop expects flat
    float observations and per-agent dictionaries for all of these values. This
    adapter performs exactly those structural conversions and delegates dynamics to
    the original environment.

    Only the 2-player variant is adaptable: ``InTheGrid_2p`` hardcodes a red/blue
    pair, so its obs/reward tuples are always length 2 regardless of the
    ``num_agents`` it was constructed with. The N-player ``InTheGrid`` uses a
    different (batched-dict) API entirely -- see the module docstring.
    """

    def __init__(self, env: MultiAgentEnv) -> None:
        self._env = env
        # String keys can coexist with JaxMARL's ``"__all__"`` done key in a JAX
        # pytree. Raw STORM uses integer ids, whose mixture with a string key cannot
        # be sorted by tree flattening during the jitted auto-reset path.
        self.agents = [f"agent_{i}" for i in range(env.num_agents)]
        self.num_agents = env.num_agents
        raw_obs, _ = env.reset(jax.random.PRNGKey(0))
        flat_dim = int(
            np.prod(raw_obs[0]["observation"].shape)
            + np.prod(raw_obs[0]["inventory"].shape)
        )
        self._obs_space = Box(-1e9, 1e9, (flat_dim,), dtype=jnp.float32)

    def _observations(self, observations: tuple) -> dict:
        return {
            agent: jnp.concatenate(
                [
                    observations[i]["observation"].reshape(-1),
                    observations[i]["inventory"].reshape(-1),
                ]
            ).astype(jnp.float32)
            for i, agent in enumerate(self.agents)
        }

    def reset(self, key):
        observations, state = self._env.reset(key)
        return self._observations(observations), state

    def step_env(self, key, state, actions):
        action_tuple = tuple(actions[agent] for agent in self.agents)
        observations, state, rewards, dones, info = self._env.step_env(
            key, state, action_tuple
        )
        done = dones["__all__"]
        reward_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        done_dict = {agent: done for agent in self.agents}
        done_dict["__all__"] = done
        return self._observations(observations), state, reward_dict, done_dict, info

    def observation_space(self, agent: str | None = None):
        del agent
        return self._obs_space

    def action_space(self, agent: str | None = None):
        del agent
        return self._env.action_space()

    def get_obs(self, state):
        return self._observations(self._env.get_obs(state))

    def __getattr__(self, name):
        # ``__getattr__`` runs before ``__init__`` populates ``_env`` (pickle and
        # ``copy`` probe dunders on a bare instance), so a missing ``_env`` must
        # surface as AttributeError -- a KeyError here would escape ``hasattr``.
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)


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


def _build_coin_game(*_: Any, **__: Any) -> MultiAgentEnv:
    raise NotImplementedError(
        "Coin-Game returns ``dones`` as a bare bool and ``rewards`` as a tuple (not the "
        "per-agent dicts with '__all__' that the IQL loop's rewards[a] / dones['__all__'] "
        "contract needs), so it requires a bespoke adapter (out of M7 scope). See "
        "conformal_marl/envs/registry.py."
    )


def _build_hanabi(**kwargs: Any) -> MultiAgentEnv:
    # Hanabi masks via ``get_legal_moves`` and wants ``(1,)``-shaped actions; the adapter
    # exposes ``get_avail_actions`` and normalises action shapes.
    return HanabiWrapper(jaxmarl.make("hanabi", **kwargs))


def _build_switch_riddle(**kwargs: Any) -> MultiAgentEnv:
    return jaxmarl.make("switch_riddle", **kwargs)


def _build_storm_2p(**kwargs: Any) -> MultiAgentEnv:
    # Symmetric common-payoff *coordination* (Stag-Hunt-shaped) game, NOT the
    # library's general-sum Prisoner's Dilemma. With row = own mixture and column =
    # the other's: (C,C)=4 is the optimum, (D,D)=2 is a second, worse equilibrium,
    # and unilateral cooperation pays 0 -- worse than mutual defection, so
    # coordination is what has to be learned. JaxMARL wants one 2x2 matrix per
    # player; passing the same *symmetric* matrix twice makes the payoff common
    # (``r1 == r2`` for every joint action), which is what the cooperative
    # learning screen assumes.
    common_payoff = jnp.asarray([[4, 0], [0, 2]], dtype=jnp.int32)
    payoff_matrix = kwargs.pop("payoff_matrix", jnp.stack([common_payoff, common_payoff]))
    num_agents = kwargs.pop("num_agents", 2)
    if num_agents != 2:
        # InTheGrid_2p accepts num_agents but ignores it -- its obs/reward tuples are
        # always the red/blue pair -- so anything else would fail deep inside the
        # adapter with an IndexError instead of here.
        raise NotImplementedError(
            f"'storm_2p' is a strictly 2-player env (got num_agents={num_agents}); "
            "JaxMARL's N-player 'storm' uses a different, unadapted obs API. See "
            "conformal_marl/envs/registry.py."
        )
    base = jaxmarl.make("storm_2p", payoff_matrix=payoff_matrix, **kwargs)
    return StormAdapter(base)


def _build_storm(*_: Any, **__: Any) -> MultiAgentEnv:
    raise NotImplementedError(
        "'storm' is JaxMARL's N-player InTheGrid, which returns obs as a single "
        "{'observations','inventory'} dict batched over agents rather than the "
        "per-agent tuple StormAdapter adapts. Use 'storm_2p' for the adapted "
        "cooperative 2-player game. See conformal_marl/envs/registry.py."
    )


# name -> (builder, partially_observable). ``masks_actions`` / ``obs_dim`` are derived
# from the *built* env (so they stay correct under overrides), not hardcoded here.
#
# Two discrete envs from the registry are deliberately *omitted* because they break the
# parameter-shared per-agent IQL loop, not because they are continuous:
#  * ``MPE_simple_speaker_listener_v4`` -- heterogeneous agents (the speaker and listener
#    have *different* obs/action dims), so the loop's ``jnp.stack`` over agents and its
#    single shared net do not apply. (The homogeneous MPE envs below are fine.)
#  * ``LearnedPolicyEnemySMAX`` -- requires a *pretrained* enemy policy + params at
#    construction (``policy=``, ``params=``); SMAX + HeuristicEnemySMAX already cover the
#    discrete SMAX catalogue without that dependency.
_BUILDERS: dict[str, tuple[Callable[..., MultiAgentEnv], bool]] = {
    "simu": (_build_simu, False),
    "MPE_simple_spread_v3": (_build_mpe("MPE_simple_spread_v3"), False),
    "MPE_simple_reference_v3": (_build_mpe("MPE_simple_reference_v3"), False),
    "SMAX": (_build_smax("SMAX"), True),
    "HeuristicEnemySMAX": (_build_smax("HeuristicEnemySMAX"), True),
    "overcooked": (_build_overcooked, True),
    "hanabi": (_build_hanabi, True),
    "switch_riddle": (_build_switch_riddle, True),
    "storm_2p": (_build_storm_2p, True),
    # Known-incompatible with the per-agent IQL loop (instantiable only -> raise a clear
    # NotImplementedError pointing at the reason; never fail deep inside make_train).
    "coin_game": (_build_coin_game, True),
    "storm": (_build_storm, True),
}

# Names whose builders raise NotImplementedError (non-per-agent obs/reward APIs): the IQL
# loop cannot consume them without a bespoke adapter, deferred from M7.
INCOMPATIBLE_ENVS: tuple[str, ...] = ("coin_game", "storm")

# Convenience: the subset the IQL loop can actually instantiate *and run*.
IN_SCOPE_ENVS: tuple[str, ...] = tuple(
    n for n in _BUILDERS if n not in INCOMPATIBLE_ENVS
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
    if spec_name == "storm_2p":
        return IQLConfig(
            total_timesteps=250_000, num_envs=16, num_steps=32, num_grad_steps=8,
            learning_starts=512, hidden_dim=64, lr=5e-4, gamma=0.99, batch_size=128,
            buffer_size=100_000, target_update_interval=100, epsilon_anneal_time=150_000,
            eval_episodes=16, eval_steps=160, net_type=net_type,
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


def make_env(name: str, config_overrides: dict | None = None, **env_kwargs: Any):
    """Build a ready-to-train env and its :class:`EnvSpec`.

    Parameters
    ----------
    name:
        A factory key (see :func:`available_envs`): the custom ``"simu"``, an MPE id
        (``"MPE_simple_spread_v3"``), ``"SMAX"`` / ``"HeuristicEnemySMAX"`` /
        ``"LearnedPolicyEnemySMAX"``, ``"overcooked"``, ``"coin_game"``, ``"hanabi"``,
        ``"switch_riddle"``, or the adapted cooperative ``"storm_2p"``. (``"storm"``
        -- the N-player variant -- raises; see the module docstring.)
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

    # Cast integer observations (Switch-Riddle int32, etc.) to float32 so they match the
    # float32 buffer schema and the LogWrapper dtype assertion. FlatObs-wrapped envs are
    # already float, so this is a no-op for them.
    obs0, _ = env.reset(jax.random.PRNGKey(0))
    if not jnp.issubdtype(jnp.asarray(obs0[env.agents[0]]).dtype, jnp.floating):
        env = CastObs(env)

    # Read shapes via the IQL loop's own metadata reader, so the spec and the loop agree
    # exactly (obs_dim probed from a reset -> robust to Hanabi's empty obs space).
    info = get_env_info(env)
    num_agents, num_actions, obs_dim = info.num_agents, info.num_actions, info.obs_dim
    masks_actions = info.has_avail_actions

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
