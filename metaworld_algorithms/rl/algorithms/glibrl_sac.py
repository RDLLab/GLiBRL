from __future__ import annotations

import dataclasses
from collections import deque
from functools import partial
from typing import TYPE_CHECKING, Self, override, ClassVar, Deque

import distrax
import gymnasium as gym
import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState as FlaxTrainState
from jaxtyping import Array, Float, PRNGKeyArray

from metaworld_algorithms.config.envs import MetaLearningEnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    NeuralNetworkConfig,
    QValueFunctionConfig,
    VanillaNetworkConfig,
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import AlgorithmConfig, GLiBRLSACTrainingConfig
from metaworld_algorithms.config.utils import Activation, Optimizer
from metaworld_algorithms.monitoring.utils import log
from metaworld_algorithms.nn.base import VanillaNetwork
from metaworld_algorithms.nn.distributions import TanhMultivariateNormalDiag
from metaworld_algorithms.rl.algorithms.utils import TrainState
from metaworld_algorithms.rl.buffers import MultiTaskReplayBuffer, ReplayBuffer
from metaworld_algorithms.rl.networks import (
    ContinuousActionPolicy,
    Ensemble,
    QValueFunction,
)
from metaworld_algorithms.types import (
    Action,
    AuxPolicyOutputs,
    CheckpointMetadata,
    GymVectorEnv,
    LogDict,
    LogProb,
    MetaLearningAgent,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Reward,
    Task,
    TaskParam,
    TaskRBias,
    TaskRFeat,
    TaskRMean,
    TaskTBias,
    TaskTFeat,
    TaskTMean,
    TaskTuple,
    TaskWithObservation,
    Timestep,
)

from .base import BRLMetaLearningAlgorithm
from .glibrl import (
    GLiBRLConfig,
)
from .sac import CriticTrainState, Temperature


# ---------------------------------------------------------------------------
# Standalone JIT helpers (module level)
# ---------------------------------------------------------------------------

@jax.jit
def _sample_action(
    actor: TrainState, observation: TaskWithObservation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    actor: TrainState, observation: TaskWithObservation
) -> Float[Array, "... action_dim"]:
    return actor.apply_fn(actor.params, observation).mode()


@jax.jit
def _sample_action_dist(
    actor: TrainState,
    observation: TaskWithObservation,
    key: PRNGKeyArray,
) -> tuple[Action, LogProb, Action, Action, PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)
    if isinstance(dist, TanhMultivariateNormalDiag):
        mean = dist.pre_tanh_mean()
        std = dist.pre_tanh_std()
    else:
        mean = dist.mode()
        std = dist.stddev()
    return action, action_log_prob, mean, std, key


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GLiBRLSACConfig(AlgorithmConfig):
    """Configuration for GLiBRL-SAC: Bayesian task inference + SAC off-policy backbone."""

    # ----- SAC agent -----
    actor_config: ContinuousActionPolicyConfig = dataclasses.field(
        default_factory=ContinuousActionPolicyConfig
    )
    critic_config: QValueFunctionConfig = dataclasses.field(
        default_factory=QValueFunctionConfig
    )
    temperature_optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig(max_grad_norm=None)
    )
    initial_temperature: float = 1
    alpha_update_freq: int = 1
    policy_update_delay: int = 1
    num_critics: int = 2
    reward_scale: float = 1.0
    tau: float = 0.005

    # ----- GLiBRL model networks -----
    t_mix_config: NeuralNetworkConfig = dataclasses.field(
        default_factory=lambda: VanillaNetworkConfig(
            width=(32, 32),
            activation=Activation.ReLU,
            optimizer=OptimizerConfig(lr=1e-3, optimizer=Optimizer.Adam, max_grad_norm=1),
            use_layer_norm=True,
        )
    )
    t_bias_config: NeuralNetworkConfig = dataclasses.field(
        default_factory=lambda: VanillaNetworkConfig(
            width=(64, 64),
            activation=Activation.ReLU,
            optimizer=OptimizerConfig(lr=5e-4, optimizer=Optimizer.Adam),
            use_layer_norm=True,
        )
    )
    r_mix_config: NeuralNetworkConfig = dataclasses.field(
        default_factory=lambda: VanillaNetworkConfig(
            width=(64, 32),
            activation=Activation.ReLU,
            optimizer=OptimizerConfig(lr=1e-3, optimizer=Optimizer.Adam, max_grad_norm=2),
            use_layer_norm=True,
        )
    )
    r_bias_config: NeuralNetworkConfig = dataclasses.field(
        default_factory=lambda: VanillaNetworkConfig(
            width=(64, 32),
            activation=Activation.ReLU,
            optimizer=OptimizerConfig(lr=5e-4, optimizer=Optimizer.Adam),
            use_layer_norm=True,
        )
    )

    # ----- Optional SA encoder -----
    use_sa_encoder: bool = True
    hidden_state_dim: int = 32
    hidden_action_dim: int = 16
    s_feat_config: NeuralNetworkConfig = dataclasses.field(
        default_factory=lambda: VanillaNetworkConfig(
            width=(64, 32),
            activation=Activation.ReLU,
            optimizer=OptimizerConfig(lr=5e-5, optimizer=Optimizer.Adam, max_grad_norm=1),
        )
    )
    a_feat_config: NeuralNetworkConfig = dataclasses.field(
        default_factory=lambda: VanillaNetworkConfig(
            width=(32, 16),
            activation=Activation.ReLU,
            optimizer=OptimizerConfig(lr=5e-5, optimizer=Optimizer.Adam, max_grad_norm=1),
        )
    )

    # ----- Bayesian inference params -----
    transition_latent_dim: int = 8
    reward_latent_dim: int = 32
    full_bayesian: bool = True
    normalise_task: bool = True
    use_transition_in_policy: bool = True
    use_reward_in_policy: bool = True
    use_next_obs_in_r_mix: bool = True
    r_mix_quadratic: bool = False
    use_bias: bool = False
    predict_delta: bool = False
    t_sample_clip: float = 2e-4
    r_sample_clip: float = 1e-3
    update_rate: float = 1.0
    t_reg: float = 2e-4
    r_reg: float = 2e-4

    # ----- Misc -----
    meta_batch_size: int = 20
    num_env_tasks: int | None = None
    """Number of training envs stepped during rollout.
    If None, defaults to num_tasks (no decoupling).
    Set > num_tasks to step more envs than the gradient-step task batch size."""
    dtype: npt.DTypeLike = np.float32

    # state_bounds: ClassVar[jnp.ndarray] = GLiBRLConfig.state_bounds
    # reward_bounds: ClassVar[jnp.ndarray] = GLiBRLConfig.reward_bounds


# ---------------------------------------------------------------------------
# GLiBRLSAC algorithm
# ---------------------------------------------------------------------------

class GLiBRLSAC(BRLMetaLearningAlgorithm["GLiBRLSACConfig"]):
    """GLiBRL-SAC: Bayesian Generalized Linear task inference + SAC off-policy RL.

    Follows a PEARL-like framework with dual replay buffers:
    - Policy buffer: stores task-augmented observations for SAC updates.
    - Model buffer: stores raw observations for model (t_mix/r_mix) updates.

    Task inference is performed online via conjugate Gaussian-Linear posterior
    updates (identical to GLiBRL) without any gradient steps at test time.
    """

    # SAC components
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState

    # GLiBRL task inference components
    t_mix_model: TrainState
    r_mix_model: TrainState
    t_bias_model: TrainState
    r_bias_model: TrainState
    s_feat_model: TrainState | None
    a_feat_model: TrainState | None
    init_task: TaskTuple

    key: PRNGKeyArray
    # state_bounds: jax.Array
    # reward_bounds: jax.Array
    t_sample_clip: jax.Array
    gradient_step: jnp.ndarray

    # Non-pytree metadata
    tau: float = struct.field(pytree_node=False)
    alpha_update_freq: int = struct.field(pytree_node=False)
    policy_update_delay: int = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)
    reward_scale: float = struct.field(pytree_node=False)
    num_tasks: int = struct.field(pytree_node=False)
    num_env_tasks: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    transition_latent_dim: int = struct.field(pytree_node=False)
    reward_latent_dim: int = struct.field(pytree_node=False)
    task_dim: int = struct.field(pytree_node=False)
    t_reg: float = struct.field(pytree_node=False)
    r_reg: float = struct.field(pytree_node=False)
    r_sample_clip: float = struct.field(pytree_node=False)
    full_bayesian: bool = struct.field(pytree_node=False)
    use_bias: bool = struct.field(pytree_node=False)
    predict_delta: bool = struct.field(pytree_node=False)
    normalise_task: bool = struct.field(pytree_node=False)
    use_transition_in_policy: bool = struct.field(pytree_node=False)
    use_reward_in_policy: bool = struct.field(pytree_node=False)
    use_next_obs_in_r_mix: bool = struct.field(pytree_node=False)
    r_mix_quadratic: bool = struct.field(pytree_node=False)
    update_rate: float = struct.field(pytree_node=False)
    use_sa_encoder: bool = struct.field(pytree_node=False)
    hidden_state_dim: int = struct.field(pytree_node=False)
    hidden_action_dim: int = struct.field(pytree_node=False)
    dtype: npt.DTypeLike = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @override
    @staticmethod
    def initialize(
        config: "GLiBRLSACConfig",
        env_config: MetaLearningEnvConfig,
        seed: int = 1,
    ) -> "GLiBRLSAC":
        assert isinstance(env_config.action_space, gym.spaces.Box)
        assert isinstance(env_config.observation_space, gym.spaces.Box)

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_key, critic_key, alpha_key, model_key = jax.random.split(master_key, 5)

        s_dim = env_config.observation_space.shape[-1]
        a_dim = env_config.action_space.shape[-1]
        effective_s_dim = s_dim  # No CNN support in GLiBRLSAC

        # --- Build SA encoder dims ---
        if config.use_sa_encoder:
            t_mix_in_dim = config.hidden_state_dim + config.hidden_action_dim
        else:
            t_mix_in_dim = effective_s_dim + a_dim

        # --- t_mix network ---
        t_mix_net = VanillaNetwork(
            config=config.t_mix_config,
            head_dim=config.transition_latent_dim,
            activate_last=False,
        )

        # --- r_mix network ---
        if config.use_sa_encoder:
            sa_feat_dim = config.hidden_state_dim + config.hidden_action_dim
            r_mix_in_dim = (
                2 * config.hidden_state_dim + config.hidden_action_dim
                if config.use_next_obs_in_r_mix
                else sa_feat_dim
            )
        else:
            r_mix_in_dim = (
                2 * effective_s_dim + a_dim
                if config.use_next_obs_in_r_mix
                else effective_s_dim + a_dim
            )

        if config.r_mix_quadratic:
            effective_rld = r_mix_in_dim + r_mix_in_dim * (r_mix_in_dim + 1) // 2
            r_mix_net = VanillaNetwork(
                config=VanillaNetworkConfig(
                    width=(),
                    activation=Activation.ReLU,
                    optimizer=config.r_mix_config.optimizer,
                ),
                head_dim=1,
                activate_last=False,
            )
        else:
            effective_rld = config.reward_latent_dim
            r_mix_net = VanillaNetwork(
                config=config.r_mix_config,
                head_dim=effective_rld,
                activate_last=False,
            )

        # --- t_bias / r_bias networks ---
        t_bias_net = VanillaNetwork(
            config=config.t_bias_config,
            head_dim=effective_s_dim,
            activate_last=False,
        )
        r_bias_net = VanillaNetwork(
            config=config.r_bias_config,
            head_dim=1,
            activate_last=False,
        )

        # --- Create TrainStates for model networks ---
        t_mix_model = TrainState.create(
            params=t_mix_net.init(
                model_key,
                jnp.zeros((config.meta_batch_size, 1, t_mix_in_dim), dtype=config.dtype),
            ),
            tx=config.t_mix_config.optimizer.spawn(),
            apply_fn=t_mix_net.apply,
        )
        r_mix_model = TrainState.create(
            params=r_mix_net.init(
                model_key,
                jnp.zeros((config.meta_batch_size, 1, r_mix_in_dim), dtype=config.dtype),
            ),
            tx=config.r_mix_config.optimizer.spawn(),
            apply_fn=r_mix_net.apply,
        )
        t_bias_model = TrainState.create(
            params=t_bias_net.init(
                model_key,
                jnp.zeros((config.meta_batch_size, 1, t_mix_in_dim), dtype=config.dtype),
            ),
            tx=config.t_bias_config.optimizer.spawn(),
            apply_fn=t_bias_net.apply,
        )
        r_bias_model = TrainState.create(
            params=r_bias_net.init(
                model_key,
                jnp.zeros((config.meta_batch_size, 1, r_mix_in_dim), dtype=config.dtype),
            ),
            tx=config.r_bias_config.optimizer.spawn(),
            apply_fn=r_bias_net.apply,
        )

        # --- Bayesian prior dimensions ---
        mt_dim = config.transition_latent_dim, effective_s_dim
        mr_dim = effective_rld, 1
        xit_dim = config.transition_latent_dim, config.transition_latent_dim
        omegat_dim = effective_s_dim, effective_s_dim
        xir_dim = effective_rld, effective_rld
        omegar_dim = 1, 1

        init_mt = jnp.zeros((1, 1, *mt_dim), dtype=config.dtype) + 1e-5
        init_mr = jnp.zeros((1, 1, *mr_dim), dtype=config.dtype) + 1e-4
        init_nut = jnp.ones((1, 1, 1, 1), dtype=config.dtype) * effective_s_dim + 1
        init_nur = jnp.ones((1, 1, 1, 1), dtype=config.dtype) + 1
        init_xit = jnp.eye(config.transition_latent_dim, dtype=config.dtype).reshape(1, 1, *xit_dim)
        init_xit_inv = jnp.eye(config.transition_latent_dim, dtype=config.dtype).reshape(1, 1, *xit_dim)
        init_omegat = jnp.eye(effective_s_dim, dtype=config.dtype).reshape(1, 1, *omegat_dim)
        if not config.full_bayesian:
            init_omegat = init_omegat * init_nut
        init_xir = jnp.eye(effective_rld, dtype=config.dtype).reshape(1, 1, *xir_dim)
        init_xir_inv = jnp.eye(effective_rld, dtype=config.dtype).reshape(1, 1, *xir_dim)
        init_omegar = jnp.eye(1, dtype=config.dtype).reshape(1, 1, *omegar_dim)
        if not config.full_bayesian:
            init_omegar = init_omegar * init_nur

        init_task = (
            init_mt.repeat(config.num_tasks, 0),
            init_xit.repeat(config.num_tasks, 0),
            init_xit_inv.repeat(config.num_tasks, 0),
            init_omegat.repeat(config.num_tasks, 0),
            init_nut.repeat(config.num_tasks, 0),
            init_mr.repeat(config.num_tasks, 0),
            init_xir.repeat(config.num_tasks, 0),
            init_xir_inv.repeat(config.num_tasks, 0),
            init_omegar.repeat(config.num_tasks, 0),
            init_nur.repeat(config.num_tasks, 0),
        )

        # --- Compute task_dim (feature dimension concatenated with obs) ---
        transition_feat_dim = config.transition_latent_dim // 2 * (config.transition_latent_dim + 1)
        reward_feat_dim = effective_rld
        reward_part_dim = reward_feat_dim if config.use_reward_in_policy else 0
        task_dim = (
            (transition_feat_dim if config.use_transition_in_policy else 0) + reward_part_dim
        )

        # --- SAC actor (conditioned on [task_feats || obs]) ---
        actor_net = ContinuousActionPolicy(
            action_dim=int(np.prod(env_config.action_space.shape)),
            config=config.actor_config,
        )
        dummy_task_obs = jnp.zeros((config.meta_batch_size, task_dim + s_dim), dtype=jnp.float32)
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_key, dummy_task_obs),
            tx=config.actor_config.network_config.optimizer.spawn(),
        )

        # --- SAC critics ---
        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        dummy_action = jnp.zeros((config.meta_batch_size, a_dim), dtype=jnp.float32)
        critic_params = critic_net.init(critic_key, dummy_task_obs, dummy_action)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_params,
            target_params=critic_params,
            tx=config.critic_config.network_config.optimizer.spawn(),
        )

        # --- SAC temperature ---
        alpha_net = Temperature(config.initial_temperature)
        alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_key),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -float(np.prod(env_config.action_space.shape))

        # --- Optional SA encoders ---
        s_feat_model: TrainState | None = None
        a_feat_model: TrainState | None = None
        if config.use_sa_encoder:
            s_feat_net = VanillaNetwork(
                config=config.s_feat_config,
                head_dim=config.hidden_state_dim,
                activate_last=True,
            )
            a_feat_net = VanillaNetwork(
                config=config.a_feat_config,
                head_dim=config.hidden_action_dim,
                activate_last=True,
            )
            s_feat_model = TrainState.create(
                params=s_feat_net.init(
                    model_key,
                    jnp.zeros((config.meta_batch_size, 1, effective_s_dim), dtype=config.dtype),
                ),
                tx=config.s_feat_config.optimizer.spawn(),
                apply_fn=s_feat_net.apply,
            )
            a_feat_model = TrainState.create(
                params=a_feat_net.init(
                    model_key,
                    jnp.zeros((config.meta_batch_size, 1, a_dim), dtype=config.dtype),
                ),
                tx=config.a_feat_config.optimizer.spawn(),
                apply_fn=a_feat_net.apply,
            )

        return GLiBRLSAC(
            num_tasks=config.num_tasks,
            num_env_tasks=config.num_env_tasks if config.num_env_tasks is not None else config.num_tasks,
            actor=actor,
            critic=critic,
            alpha=alpha,
            t_mix_model=t_mix_model,
            r_mix_model=r_mix_model,
            t_bias_model=t_bias_model,
            r_bias_model=r_bias_model,
            s_feat_model=s_feat_model,
            a_feat_model=a_feat_model,
            init_task=init_task,
            key=algorithm_key,
            # state_bounds=config.state_bounds,
            # reward_bounds=config.reward_bounds,
            t_sample_clip=jnp.array(config.t_sample_clip, dtype=jnp.float32),
            tau=config.tau,
            target_entropy=target_entropy,
            num_critics=config.num_critics,
            reward_scale=config.reward_scale,
            state_dim=effective_s_dim,
            action_dim=a_dim,
            transition_latent_dim=config.transition_latent_dim,
            reward_latent_dim=effective_rld,
            task_dim=task_dim,
            t_reg=config.t_reg,
            r_reg=config.r_reg,
            r_sample_clip=config.r_sample_clip,
            full_bayesian=config.full_bayesian,
            use_bias=config.use_bias,
            predict_delta=config.predict_delta,
            normalise_task=config.normalise_task,
            use_transition_in_policy=config.use_transition_in_policy,
            use_reward_in_policy=config.use_reward_in_policy,
            use_next_obs_in_r_mix=config.use_next_obs_in_r_mix,
            r_mix_quadratic=config.r_mix_quadratic,
            update_rate=config.update_rate,
            use_sa_encoder=config.use_sa_encoder,
            hidden_state_dim=config.hidden_state_dim,
            hidden_action_dim=config.hidden_action_dim,
            dtype=config.dtype,
            gamma=config.gamma,
            gradient_step=jnp.zeros((), dtype=jnp.int32),
            alpha_update_freq=config.alpha_update_freq,
            policy_update_delay=config.policy_update_delay,
        )

    # -----------------------------------------------------------------------
    # Parameter count
    # -----------------------------------------------------------------------

    @override
    def get_num_params(self) -> dict[str, int]:
        actor_params = sum(x.size for x in jax.tree.leaves(self.actor.params))
        critic_params = sum(x.size for x in jax.tree.leaves(self.critic.params))
        t_mix_params = sum(x.size for x in jax.tree.leaves(self.t_mix_model.params))
        r_mix_params = sum(x.size for x in jax.tree.leaves(self.r_mix_model.params))
        t_bias_params = sum(x.size for x in jax.tree.leaves(self.t_bias_model.params)) if self.use_bias else 0
        r_bias_params = sum(x.size for x in jax.tree.leaves(self.r_bias_model.params)) if self.use_bias else 0
        sa_enc_params = 0
        if self.use_sa_encoder and self.s_feat_model is not None and self.a_feat_model is not None:
            sa_enc_params = (
                sum(x.size for x in jax.tree.leaves(self.s_feat_model.params))
                + sum(x.size for x in jax.tree.leaves(self.a_feat_model.params))
            )
        return {
            "actor_num_params": actor_params,
            "critic_num_params": critic_params,
            "model_num_params": t_mix_params + r_mix_params + t_bias_params + r_bias_params + sa_enc_params,
        }

    # -----------------------------------------------------------------------
    # Action sampling (SAC style)
    # -----------------------------------------------------------------------

    @override
    def sample_action(self, observation: TaskWithObservation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations: TaskWithObservation) -> Action:
        return jax.device_get(_eval_action(self.actor, observations))

    @override
    def sample_action_and_aux(
        self, observation: TaskWithObservation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        rets = _sample_action_dist(self.actor, observation, self.key)
        action, log_prob, mean, std = jax.device_get(rets[:-1])
        key = rets[-1]
        return (
            self.replace(key=key),
            action,
            {"log_prob": log_prob, "mean": mean, "std": std},
        )

    # -----------------------------------------------------------------------
    # Task inference helpers (verbatim from GLiBRL)
    # -----------------------------------------------------------------------

    @override
    def get_init_task(self) -> TaskTuple:
        return self.init_task

    def get_state_feat(self, s_feat_params, obs: jnp.ndarray) -> jnp.ndarray:
        if self.use_sa_encoder and s_feat_params is not None:
            return self.s_feat_model.apply_fn(s_feat_params, obs)
        return obs

    def get_action_feat(self, a_feat_params, action: jnp.ndarray) -> jnp.ndarray:
        if self.use_sa_encoder and a_feat_params is not None:
            return self.a_feat_model.apply_fn(a_feat_params, action)
        return action

    @jax.jit
    def get_taskt_feat_bias(
        self,
        t_mix_param: FrozenDict,
        t_bias_param: FrozenDict,
        sa: jnp.ndarray,
        obs: jnp.ndarray,
    ) -> tuple[TaskTFeat, TaskTBias]:
        ct = self.t_mix_model.apply_fn(t_mix_param, sa)
        if self.predict_delta:
            bt = obs
        elif self.use_bias:
            bt = self.t_bias_model.apply_fn(t_bias_param, sa)
        else:
            bt = jnp.zeros(shape=(self.state_dim,))
        return ct, bt

    @jax.jit
    def get_taskr_feat_bias(
        self,
        r_mix_param: FrozenDict,
        r_bias_param: FrozenDict,
        sas: jnp.ndarray,
    ) -> tuple[TaskRFeat, TaskRBias]:
        cr = self.r_mix_model.apply_fn(r_mix_param, sas)
        if self.use_bias:
            br = self.r_bias_model.apply_fn(r_bias_param, sas)
        else:
            br = jnp.zeros(shape=(1,))
        return cr, br

    @jax.jit
    def datafeat_to_taskfeat(
        self,
        obs: Observation,
        action: Action,
        obs_next: Observation,
        t_mix_params: FrozenDict,
        t_bias_params: FrozenDict,
        r_mix_params: FrozenDict,
        r_bias_params: FrozenDict,
        s_feat_params=None,
        a_feat_params=None,
    ) -> tuple[TaskTFeat, TaskTBias, TaskRFeat, TaskRBias]:
        s_feat = self.get_state_feat(s_feat_params, obs)
        a_feat = self.get_action_feat(a_feat_params, action)
        s_next_feat = self.get_state_feat(s_feat_params, obs_next)
        sa = jnp.concatenate([s_feat, a_feat], axis=-1)
        r_input = (
            jnp.concatenate([s_feat, a_feat, s_next_feat], axis=-1)
            if self.use_next_obs_in_r_mix
            else sa
        )
        ct, bt = self.get_taskt_feat_bias(t_mix_params, t_bias_params, sa, obs)
        cr, br = self.get_taskr_feat_bias(r_mix_params, r_bias_params, r_input)
        return ct, bt, cr, br

    @jax.jit
    def gather_feats(
        self,
        obs: Observation,
        action: Action,
        reward: Reward,
        obs_next: Observation,
        t_mix_params: FrozenDict,
        t_bias_params: FrozenDict,
        r_mix_params: FrozenDict,
        r_bias_params: FrozenDict,
        s_feat_params=None,
        a_feat_params=None,
    ) -> tuple[TaskTFeat, TaskRFeat, TaskTBias, TaskRBias, Observation, Reward]:
        if obs.ndim == 2:
            obs_ = jnp.expand_dims(obs, axis=[1, 2])
            action_ = jnp.expand_dims(action, axis=[1, 2])
            reward_ = jnp.expand_dims(reward, axis=[1, 2, -1])
            obs_next_ = jnp.expand_dims(obs_next, axis=[1, 2])
        else:
            obs_ = jnp.expand_dims(obs, axis=1).swapaxes(0, 2)
            action_ = jnp.expand_dims(action, axis=1).swapaxes(0, 2)
            reward_ = jnp.expand_dims(reward, axis=1).swapaxes(0, 2)
            obs_next_ = jnp.expand_dims(obs_next, axis=1).swapaxes(0, 2)

        ct, bt, cr, br = self.datafeat_to_taskfeat(
            obs_, action_, obs_next_,
            t_mix_params, t_bias_params,
            r_mix_params, r_bias_params,
            s_feat_params, a_feat_params,
        )

        s_minus_b = obs_next_ - bt
        r_minus_b = reward_ - br

        return ct, cr, bt, br, s_minus_b, r_minus_b

    @jax.jit
    def task_posterior_inner(
        self,
        task: TaskTuple,
        ct: TaskTFeat,
        cr: TaskRFeat,
        s_minus_b: Observation,
        r_minus_b: Reward,
    ) -> TaskTuple:
        mt, xit, xit_inv, omegat, nut, mr, xir, xir_inv, omegar, nur = task
        N = ct.shape[2]

        ct_transpose = jnp.swapaxes(ct, -1, -2)
        cr_transpose = jnp.swapaxes(cr, -1, -2)

        xit_new = xit + self.update_rate * ct_transpose @ ct
        if N == 1:
            tmp = xit_inv @ ct_transpose
            xit_new_inv = (
                xit_inv
                - self.update_rate * tmp @ tmp.swapaxes(-1, -2) / (1 + self.update_rate * ct @ tmp)
            )
        else:
            xit_new_inv = jnp.linalg.solve(xit_new, jnp.eye(xit_new.shape[-1], dtype=xit_new.dtype))
        xit_new_inv = (xit_new_inv + xit_new_inv.swapaxes(-1, -2)) / 2

        mt_new = xit_new_inv @ (self.update_rate * ct_transpose @ s_minus_b + xit @ mt)

        if self.full_bayesian:
            e_t = s_minus_b - ct @ mt
            nut_new = nut + N
            ct_T_et = ct_transpose @ e_t
            omega_update = self.update_rate * (
                e_t.swapaxes(-1, -2) @ e_t
                - ct_T_et.swapaxes(-1, -2) @ xit_new_inv @ ct_T_et
            )
            omegat_new = (omegat * nut + omega_update) / nut_new
            omegat_new = (omegat_new + omegat_new.swapaxes(-1, -2)) * 0.5
        else:
            omegat_new = omegat
            nut_new = nut

        xir_new = xir + self.update_rate * cr_transpose @ cr
        if N == 1:
            tmp = xir_inv @ cr_transpose
            xir_new_inv = (
                xir_inv
                - self.update_rate * tmp @ tmp.swapaxes(-1, -2) / (1 + self.update_rate * cr @ tmp)
            )
        else:
            xir_new_inv = jnp.linalg.solve(xir_new, jnp.eye(xir_new.shape[-1], dtype=xir_new.dtype))
        xir_new_inv = (xir_new_inv + xir_new_inv.swapaxes(-1, -2)) / 2

        mr_new = xir_new_inv @ (self.update_rate * cr_transpose @ r_minus_b + xir @ mr)

        if self.full_bayesian:
            e_r = r_minus_b - cr @ mr
            nur_new = nur + N
            cr_T_er = cr_transpose @ e_r
            omegar_update = self.update_rate * (
                e_r.swapaxes(-1, -2) @ e_r
                - cr_T_er.swapaxes(-1, -2) @ xir_new_inv @ cr_T_er
            )
            omegar_new = (omegar * nur + omegar_update) / nur_new
            omegar_new = (omegar_new + omegar_new.swapaxes(-1, -2)) * 0.5
        else:
            omegar_new = omegar
            nur_new = nur

        return (
            mt_new, xit_new, xit_new_inv,
            omegat_new, nut_new,
            mr_new, xir_new, xir_new_inv,
            jnp.abs(omegar_new), nur_new,
        )

    @jax.jit
    def task_posterior(
        self,
        task: TaskTuple,
        obs: Observation,
        action: Action,
        reward: Reward,
        obs_next: Observation,
        t_mix_params: FrozenDict,
        t_bias_params: FrozenDict,
        r_mix_params: FrozenDict,
        r_bias_params: FrozenDict,
        s_feat_params=None,
        a_feat_params=None,
    ) -> tuple[TaskTFeat, TaskRFeat, TaskTBias, TaskRBias, TaskTuple]:
        ct, cr, bt, br, s_minus_b, r_minus_b = self.gather_feats(
            obs, action, reward, obs_next,
            t_mix_params, t_bias_params,
            r_mix_params, r_bias_params,
            s_feat_params, a_feat_params,
        )
        new_task = self.task_posterior_inner(task, ct, cr, s_minus_b, r_minus_b)
        return ct, cr, bt, br, jax.device_get(new_task)

    @jax.jit
    def reshape_task(self, tasktuple: TaskTuple) -> Task:
        def get_lower_triangle(v):
            row, col = jnp.tril_indices(v.shape[-1])
            return v[..., row, col]

        mt, _, xit_inv, omegat, _, mr, _, xir_inv, omegar, nur = tasktuple

        if not self.use_transition_in_policy and not self.use_reward_in_policy:
            return jnp.zeros((mt.shape[0], mt.shape[1], 0), dtype=mt.dtype)

        parts = []

        if self.use_transition_in_policy:
            mt_feat = get_lower_triangle(mt @ mt.swapaxes(-1, -2))
            mt_reshaped = mt_feat.reshape(mt_feat.shape[0], mt_feat.shape[1], -1)
            if self.normalise_task:
                mt_norm = jnp.linalg.norm(mt_reshaped, axis=-1, keepdims=True) + 1e-8
                mt_reshaped = mt_reshaped / mt_norm
            parts.append(mt_reshaped)

        if self.use_reward_in_policy:
            mr_feat = mr
            mr_flat = mr_feat.reshape(mr_feat.shape[0], mr_feat.shape[1], -1)
            if self.normalise_task:
                mr_flat = mr_flat / (jnp.linalg.norm(mr_flat, axis=-1, keepdims=True) + 1e-8)
            parts.append(mr_flat)

        return jnp.concat(parts, axis=-1)

    @jax.jit
    def concat_task_with_observation(
        self,
        task: Task,
        observation: Observation,
    ) -> TaskWithObservation:
        if len(observation.shape) == 2:
            obs = jnp.expand_dims(observation, 1)
            return jnp.concatenate([obs, task], axis=-1).squeeze(1)
        elif len(observation.shape) == 3:
            obs = observation.swapaxes(0, 1)
        else:
            obs = observation.squeeze(1)
        return jnp.concatenate([obs, task], axis=-1)

    @jax.jit
    def t_likelihood_mean(self, ct: TaskTFeat, bt: TaskTBias, t_mu: TaskTMean) -> Observation:
        return ct @ t_mu + bt

    @jax.jit
    def r_likelihood_mean(self, cr: TaskRFeat, br: TaskRBias, r_mu: TaskRMean) -> Reward:
        return cr @ r_mu + br

    # -----------------------------------------------------------------------
    # SAC update
    # -----------------------------------------------------------------------

    @partial(jax.jit, static_argnames=())
    def _update_sac_inner(
        self,
        obs: jnp.ndarray,           # (N, B, obs_dim) — obs already contain task features
        next_obs: jnp.ndarray,      # (N, B, obs_dim)
        actions: jnp.ndarray,       # (N, B, act_dim)
        rewards: jnp.ndarray,       # (N, B, 1)
        dones: jnp.ndarray,         # (N, B, 1)
        sample_mask: jnp.ndarray | None = None,  # (N*B,) float, 1=valid 0=padding
    ) -> tuple["GLiBRLSAC", LogDict]:
        """SAC update step operating on (N, B, dim) task-batched data.

        obs/next_obs already contain task features concatenated, so no
        additional task conditioning is needed.

        Losses are averaged over the task dimension (first axis) first,
        then over the batch dimension — keeping per-task gradients balanced.

        If sample_mask is provided (e.g. when k < B valid samples per task),
        it is reshaped to (N, B) and used to normalise each task's batch
        independently before averaging across tasks.
        """
        key, actor_loss_key, critic_loss_key = jax.random.split(self.key, 3)

        N, B = obs.shape[:2]

        # Per-task per-sample weights: (N, B), rows sum to 1 within each task.
        if sample_mask is not None:
            mask = sample_mask.reshape(N, B)                            # (N, B)
            w = mask / (mask.sum(axis=1, keepdims=True) + 1e-8)        # (N, B)
        else:
            w = jnp.ones((N, B)) / B                                    # (N, B)

        def task_wmean(x):
            """Weighted mean over B dim per task, then mean over N.

            x: (N, B, ...) → scalar
            """
            # Broadcast w to match x's trailing dims
            w_b = w.reshape(N, B, *((1,) * (x.ndim - 2)))  # (N, B, 1, ...)
            return (x * w_b).sum(axis=1).mean() 

        # Double-Q SAC: Q bootstraps from target Q(s', a'~π) - α log π(a'|s')
        def update_critic(
                _critic: CriticTrainState,
            ) -> tuple[CriticTrainState, LogDict]:
                next_actions, next_log_probs = self.actor.apply_fn(
                    self.actor.params, next_obs
                ).sample_and_log_prob(seed=critic_loss_key)
                # next_log_probs = jnp.clip(next_log_probs, -20.0, 0.0)
                q_next = self.critic.apply_fn(
                    self.critic.target_params, next_obs, next_actions
                )
                min_q_next = jnp.min(q_next, axis=0)

                def critic_loss(params: FrozenDict):
                    next_q_value = jax.lax.stop_gradient(
                        rewards * self.reward_scale
                        + (1 - dones) * self.gamma * (
                            min_q_next - alpha_val * next_log_probs[..., None]
                        )
                    )
                    q_pred = self.critic.apply_fn(params, obs, actions)
                    td_sq = 0.5 * (q_pred - next_q_value[None]) ** 2
                    return task_wmean(td_sq.mean(axis=0)), task_wmean(q_pred.mean(axis=0))

                (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
                    critic_loss, has_aux=True
                )(_critic.params)
                flat_grads, _ = flatten_util.ravel_pytree(critic_grads)
                return _critic.apply_gradients(grads=critic_grads), {
                    "losses/qf_values": qf_values,
                    "losses/qf_loss": critic_loss_value,
                    "metrics/critic_grad_magnitude_preclip": jnp.linalg.norm(flat_grads),
                }

        def update_alpha(
            _alpha: TrainState, log_probs: jnp.ndarray  # (N, B)
        ) -> tuple[TrainState, jnp.ndarray, LogDict]:
            def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
                log_alpha = params["params"]["log_alpha"]
                return task_wmean(-log_alpha * (log_probs[..., None] + self.target_entropy))

            alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(_alpha.params)
            _alpha = _alpha.apply_gradients(grads=alpha_grads)
            alpha_vals = _alpha.apply_fn(_alpha.params)
            return (
                _alpha,
                alpha_vals,
                {
                    "losses/alpha_loss": alpha_loss_value,
                    "alpha": jnp.exp(_alpha.params["params"]["log_alpha"]).sum(),
                },
            )

        # Step 1: current alpha value (pre-update, stop-gradient)
        alpha_val = jax.lax.stop_gradient(self.alpha.apply_fn(self.alpha.params))

        # Step 2: update critic
        critic, critic_logs = update_critic(self.critic)

        # Step 3: update actor and alpha (every policy_update_delay critic steps)
        critic_params_fixed = jax.lax.stop_gradient(critic.params)

        def actor_loss(params: FrozenDict):
            action_samples, log_probs = self.actor.apply_fn(
                params, obs
            ).sample_and_log_prob(seed=actor_loss_key)
            q_values = critic.apply_fn(critic_params_fixed, obs, action_samples)
            min_qf_values = jnp.min(q_values, axis=0)
            loss = task_wmean(alpha_val * log_probs[..., None] - min_qf_values)
            return loss, log_probs

        def _do_actor_alpha_update():
            (loss_val, lp), grads = jax.value_and_grad(actor_loss, has_aux=True)(self.actor.params)
            _actor = self.actor.apply_gradients(grads=grads)
            flat, _ = flatten_util.ravel_pytree(grads)

            def _no_update_alpha(_alpha):
                return _alpha, _alpha.apply_fn(_alpha.params), {
                    "losses/alpha_loss": jnp.zeros(()),
                    "alpha": jnp.exp(_alpha.params["params"]["log_alpha"]).sum(),
                }
            _alpha, _, _alpha_logs = jax.lax.cond(
                (self.gradient_step % self.alpha_update_freq) == 0,
                lambda: update_alpha(self.alpha, lp),
                lambda: _no_update_alpha(self.alpha),
            )
            return _actor, _alpha, loss_val, flat, _alpha_logs

        def _skip_actor_alpha_update():
            # Keep actor fixed; sample log_probs for alpha bookkeeping only
            _, lp = self.actor.apply_fn(
                jax.lax.stop_gradient(self.actor.params), obs
            ).sample_and_log_prob(seed=actor_loss_key)
            flat, _ = flatten_util.ravel_pytree(
                jax.tree.map(jnp.zeros_like, self.actor.params)
            )
            _alpha_logs = {
                "losses/alpha_loss": jnp.zeros(()),
                "alpha": jnp.exp(self.alpha.params["params"]["log_alpha"]).sum(),
            }
            return self.actor, self.alpha, jnp.zeros(()), flat, _alpha_logs

        actor, alpha, actor_loss_value, flat_grads, alpha_logs = jax.lax.cond(
            (self.gradient_step % self.policy_update_delay) == 0,
            _do_actor_alpha_update,
            _skip_actor_alpha_update,
        )

        # Soft-update target Q network
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params, critic.target_params, self.tau,
            )
        )

        self = self.replace(key=key, actor=actor, critic=critic, alpha=alpha, gradient_step=self.gradient_step + 1)
        return self, {
            **critic_logs, **alpha_logs,
            "losses/actor_loss": actor_loss_value,
            "metrics/actor_grad_magnitude": jnp.linalg.norm(flat_grads),
        }

    @override
    def update(self, data: ReplayBufferSamples) -> tuple["GLiBRLSAC", LogDict]:
        # Add task dimension N=1 so _update_sac_inner always sees (N, B, dim)
        return self._update_sac_inner(
            data.observations[None],
            data.next_observations[None],
            data.actions[None],
            data.rewards[None],
            data.dones[None],
        )

    @jax.jit
    def _update_sac_inner_pearl(
        self,
        obs_pt: jnp.ndarray,      # (num_tasks, B_t, s_dim)
        act_pt: jnp.ndarray,      # (num_tasks, B_t, a_dim)
        rew_pt: jnp.ndarray,      # (num_tasks, B_t, 1)
        next_obs_pt: jnp.ndarray, # (num_tasks, B_t, s_dim)
        done_pt: jnp.ndarray,     # (num_tasks, B_t, 1)
        k: jnp.ndarray | None = None,  # scalar int — number of valid samples per task
    ) -> tuple["GLiBRLSAC", LogDict]:
        """SAC update with per-task Bayesian posterior computed from sampled batch.

        For each task, the posterior M_T and M_R are computed from the sampled
        per-task batch. Task features are then concatenated with observations
        to form task-conditioned inputs for the SAC actor and critic.

        Update order: critic first, then actor (standard SAC ordering — actor
        maximises E[Q(s, π(s)) - α log π(s)] and benefits from an accurate Q).
        """
        N = obs_pt.shape[0]   # number of sampled tasks (may be < self.num_tasks for Vel)
        B_t = obs_pt.shape[1]

        # Build flat sample mask: 1 for valid rows, 0 for padding.
        # Shape (N*B_t,): for each task block of B_t rows, first k are valid.
        if k is not None:
            per_task_valid = jnp.arange(B_t) < k          # (B_t,)
            sample_mask = jnp.tile(per_task_valid, N).astype(jnp.float32)  # (N*B_t,)
        else:
            sample_mask = None

        # Expand to 4D: (N, 1, B_t, dim) for datafeat_to_taskfeat
        obs_ = obs_pt[:, None, :, :]            # (N, 1, B_t, s_dim)
        act_ = act_pt[:, None, :, :]            # (N, 1, B_t, a_dim)
        rew_ = rew_pt[:, None, :, :]            # (N, 1, B_t, 1)
        next_obs_ = next_obs_pt[:, None, :, :]  # (N, 1, B_t, s_dim)

        # Stop-gradient on model params: SAC update does not train model networks
        t_mix_p = jax.lax.stop_gradient(self.t_mix_model.params)
        t_bias_p = jax.lax.stop_gradient(self.t_bias_model.params)
        r_mix_p = jax.lax.stop_gradient(self.r_mix_model.params)
        r_bias_p = jax.lax.stop_gradient(self.r_bias_model.params)
        s_feat_p = (
            jax.lax.stop_gradient(self.s_feat_model.params)
            if self.use_sa_encoder and self.s_feat_model is not None
            else None
        )
        a_feat_p = (
            jax.lax.stop_gradient(self.a_feat_model.params)
            if self.use_sa_encoder and self.a_feat_model is not None
            else None
        )

        # Compute per-task posteriors from the sampled batch
        ct, bt, cr, br = self.datafeat_to_taskfeat(
            obs_, act_, next_obs_,
            t_mix_p, t_bias_p, r_mix_p, r_bias_p, s_feat_p, a_feat_p,
        )
        s_minus_b = next_obs_ - bt  # (N, 1, B_t, s_dim)
        r_minus_b = rew_ - br       # (N, 1, B_t, 1)

        # Zero out padded rows so they contribute nothing to the Bayesian update
        if k is not None:
            row_mask = (jnp.arange(B_t) < k)[None, None, :, None]  # (1, 1, B_t, 1)
            ct = ct * row_mask
            cr = cr * row_mask
            s_minus_b = s_minus_b * row_mask
            r_minus_b = r_minus_b * row_mask

        # Broadcast the single shared prior to N tasks.
        # All rows of init_task are identical; repeat the first row for any N.
        init_task_n = jax.tree.map(lambda x: jnp.repeat(x[:1], N, axis=0), self.init_task)
        new_task = self.task_posterior_inner(init_task_n, ct, cr, s_minus_b, r_minus_b)

        # Task features: (N, 1, task_dim) → expand to (N, B_t, task_dim)
        task_feats = self.reshape_task(new_task)  # (N, 1, task_dim)
        task_dim = task_feats.shape[-1]
        task_feats_exp = jnp.broadcast_to(task_feats, (N, B_t, task_dim))

        # Flatten all to (B, dim) with B = N * B_t

        # Concatenate: [obs, task_feats] — consistent with concat_task_with_observation
        obs_with_task = jnp.concatenate((obs_pt, task_feats_exp), axis=-1)
        next_obs_with_task = jnp.concatenate((next_obs_pt, task_feats_exp), axis=-1)

        return self._update_sac_inner(
            obs_with_task, next_obs_with_task, act_pt, rew_pt, done_pt,
            sample_mask=sample_mask,
        )

    # -----------------------------------------------------------------------
    # Model update (Bayesian log-marginal likelihood loss)
    # -----------------------------------------------------------------------

    @jax.jit
    def update_models(self, data: ReplayBufferSamples) -> tuple["GLiBRLSAC", LogDict]:
        """Update t_mix/r_mix/t_bias/r_bias using Bayesian marginal likelihood.

        The flat batch (B, dim) is treated as a single meta-task to compute
        the Bayesian MLE loss. This learns model features useful across tasks.
        """
        obs      = data.observations
        next_obs = data.next_observations
        action   = data.actions
        reward   = data.rewards

        # Single-prior init_task (task index 0, shape (1, 1, ...))
        single_init_task = jax.tree.map(lambda x: x[:1], self.init_task)

        s_feat_params = (
            self.s_feat_model.params
            if self.use_sa_encoder and self.s_feat_model is not None
            else None
        )
        a_feat_params = (
            self.a_feat_model.params
            if self.use_sa_encoder and self.a_feat_model is not None
            else None
        )

        def model_loss(
            t_mix_params: FrozenDict,
            t_bias_params: FrozenDict,
            r_mix_params: FrozenDict,
            r_bias_params: FrozenDict,
            s_feat_p,
            a_feat_p,
        ):
            ct, cr, bt, br, s_minus_b, r_minus_b = self.gather_feats(
                obs, action, reward, next_obs,
                t_mix_params, t_bias_params,
                r_mix_params, r_bias_params,
                s_feat_p, a_feat_p,
            )

            new_task = self.task_posterior_inner(
                single_init_task, ct, cr, s_minus_b, r_minus_b
            )
            mt_new, xit_new, _, omegat_new, nut_new, mr_new, xir_new, _, omegar_new, nur_new = new_task

            s_dim = obs.shape[-1]
            N = obs.shape[0]
            t_loss = s_dim * jnp.linalg.slogdet(xit_new)[1]
            r_loss = jnp.linalg.slogdet(xir_new)[1]

            if self.full_bayesian:
                t_loss = t_loss + nut_new[..., 0, 0] * jnp.linalg.slogdet(0.5 * omegat_new)[1]
                r_loss = r_loss + nur_new[..., 0, 0] * jnp.linalg.slogdet(0.5 * omegar_new)[1]
            else:
                t_loss = t_loss - jnp.trace(
                    omegat_new @ mt_new.swapaxes(-1, -2) @ xit_new @ mt_new,
                    axis1=-2, axis2=-1,
                )
                r_loss = r_loss - jnp.trace(
                    omegar_new @ mr_new.swapaxes(-1, -2) @ xir_new @ mr_new,
                    axis1=-2, axis2=-1,
                )

            t_loss = t_loss.mean() * 0.5 / N
            r_loss = r_loss.mean() * 0.5 / N

            ct_norm = jnp.square(
                jnp.linalg.norm(ct, axis=-1) / self.transition_latent_dim ** 0.5
            ).mean()
            cr_norm = jnp.square(
                jnp.linalg.norm(cr, axis=-1) / self.reward_latent_dim ** 0.5
            ).mean()

            loss = t_loss + r_loss + self.t_reg * ct_norm + self.r_reg * cr_norm
            return loss, {
                "losses/t_loss": t_loss,
                "losses/r_loss": r_loss,
                "losses/ct_norm": ct_norm,
                "losses/cr_norm": cr_norm,
            }

        (_, model_logs), grads = jax.value_and_grad(
            model_loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True
        )(
            self.t_mix_model.params,
            self.t_bias_model.params,
            self.r_mix_model.params,
            self.r_bias_model.params,
            s_feat_params,
            a_feat_params,
        )

        t_mix_model = self.t_mix_model.apply_gradients(grads=grads[0])
        t_bias_model = self.t_bias_model.apply_gradients(grads=grads[1])
        r_mix_model = self.r_mix_model.apply_gradients(grads=grads[2])
        r_bias_model = self.r_bias_model.apply_gradients(grads=grads[3])
        s_feat_model = (
            self.s_feat_model.apply_gradients(grads=grads[4])
            if self.use_sa_encoder and self.s_feat_model is not None
            else self.s_feat_model
        )
        a_feat_model = (
            self.a_feat_model.apply_gradients(grads=grads[5])
            if self.use_sa_encoder and self.a_feat_model is not None
            else self.a_feat_model
        )

        return (
            self.replace(
                t_mix_model=t_mix_model,
                t_bias_model=t_bias_model,
                r_mix_model=r_mix_model,
                r_bias_model=r_bias_model,
                s_feat_model=s_feat_model,
                a_feat_model=a_feat_model,
            ),
            model_logs,
        )

    @jax.jit
    def update_models_pearl(
        self,
        obs_pt: jnp.ndarray,       # (num_tasks, B_t, s_dim)
        act_pt: jnp.ndarray,       # (num_tasks, B_t, a_dim)
        rew_pt: jnp.ndarray,       # (num_tasks, B_t, 1)
        next_obs_pt: jnp.ndarray,  # (num_tasks, B_t, s_dim)
    ) -> tuple["GLiBRLSAC", LogDict]:
        """Update t_mix/r_mix/t_bias/r_bias with per-task Bayesian posteriors.

        Each task's posterior is computed independently from its own sampled
        batch of (s, a, s', r) transitions. The loss is the Bayesian
        log-marginal likelihood summed across all tasks.
        """
        B_t = obs_pt.shape[1]

        obs_ = obs_pt[:, None, :, :]            # (N, 1, B_t, s_dim)
        act_ = act_pt[:, None, :, :]
        rew_ = rew_pt[:, None, :, :]
        next_obs_ = next_obs_pt[:, None, :, :]

        s_feat_params = (
            self.s_feat_model.params
            if self.use_sa_encoder and self.s_feat_model is not None
            else None
        )
        a_feat_params = (
            self.a_feat_model.params
            if self.use_sa_encoder and self.a_feat_model is not None
            else None
        )

        def model_loss(
            t_mix_params: FrozenDict,
            t_bias_params: FrozenDict,
            r_mix_params: FrozenDict,
            r_bias_params: FrozenDict,
            s_feat_p,
            a_feat_p,
        ):
            ct, bt, cr, br = self.datafeat_to_taskfeat(
                obs_, act_, next_obs_,
                t_mix_params, t_bias_params, r_mix_params, r_bias_params,
                s_feat_p, a_feat_p,
            )
            s_minus_b = next_obs_ - bt
            r_minus_b = rew_ - br

            # Broadcast the single shared prior to N tasks (N may differ from self.num_tasks).
            N_obs = obs_pt.shape[0]
            init_task_n = jax.tree.map(lambda x: jnp.repeat(x[:1], N_obs, axis=0), self.init_task)
            new_task = self.task_posterior_inner(init_task_n, ct, cr, s_minus_b, r_minus_b)
            mt_new, xit_new, _, omegat_new, nut_new, mr_new, xir_new, _, omegar_new, nur_new = new_task

            s_dim_val = obs_pt.shape[-1]
            N_val = B_t  # number of transitions per task

            t_loss = s_dim_val * jnp.linalg.slogdet(xit_new)[1]
            r_loss = jnp.linalg.slogdet(xir_new)[1]

            if self.full_bayesian:
                t_loss = t_loss + nut_new[..., 0, 0] * jnp.linalg.slogdet(0.5 * omegat_new)[1]
                r_loss = r_loss + nur_new[..., 0, 0] * jnp.linalg.slogdet(0.5 * omegar_new)[1]
            else:
                t_loss = t_loss - jnp.trace(
                    omegat_new @ mt_new.swapaxes(-1, -2) @ xit_new @ mt_new,
                    axis1=-2, axis2=-1,
                )
                r_loss = r_loss - jnp.trace(
                    omegar_new @ mr_new.swapaxes(-1, -2) @ xir_new @ mr_new,
                    axis1=-2, axis2=-1,
                )

            t_loss = t_loss.mean() * 0.5 / N_val
            r_loss = r_loss.mean() * 0.5 / N_val

            ct_norm = jnp.square(
                jnp.linalg.norm(ct, axis=-1) / self.transition_latent_dim ** 0.5
            ).mean()
            cr_norm = jnp.square(
                jnp.linalg.norm(cr, axis=-1) / self.reward_latent_dim ** 0.5
            ).mean()

            loss = t_loss + r_loss + self.t_reg * ct_norm + self.r_reg * cr_norm
            return loss, {
                "losses/t_loss": t_loss,
                "losses/r_loss": r_loss,
                "losses/ct_norm": ct_norm,
                "losses/cr_norm": cr_norm,
            }

        (_, model_logs), grads = jax.value_and_grad(
            model_loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True
        )(
            self.t_mix_model.params,
            self.t_bias_model.params,
            self.r_mix_model.params,
            self.r_bias_model.params,
            s_feat_params,
            a_feat_params,
        )

        t_mix_model = self.t_mix_model.apply_gradients(grads=grads[0]) \
                      if self.use_transition_in_policy else self.t_mix_model
        t_bias_model = self.t_bias_model.apply_gradients(grads=grads[1])
        r_mix_model = self.r_mix_model.apply_gradients(grads=grads[2]) \
                      if self.use_reward_in_policy else self.r_mix_model
        r_bias_model = self.r_bias_model.apply_gradients(grads=grads[3])
        s_feat_model = (
            self.s_feat_model.apply_gradients(grads=grads[4])
            if self.use_sa_encoder and self.s_feat_model is not None
            else self.s_feat_model
        )
        a_feat_model = (
            self.a_feat_model.apply_gradients(grads=grads[5])
            if self.use_sa_encoder and self.a_feat_model is not None
            else self.a_feat_model
        )

        return (
            self.replace(
                t_mix_model=t_mix_model,
                t_bias_model=t_bias_model,
                r_mix_model=r_mix_model,
                r_bias_model=r_bias_model,
                s_feat_model=s_feat_model,
                a_feat_model=a_feat_model,
            ),
            model_logs,
        )

    # -----------------------------------------------------------------------
    # Off-policy PEARL-like training loop
    # -----------------------------------------------------------------------

    @override
    def train(
        self,
        config: "GLiBRLSACTrainingConfig",
        envs: "GymVectorEnv",
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager=None,
        checkpoint_metadata: "CheckpointMetadata | None" = None,
        buffer_checkpoint: "ReplayBufferCheckpoint | None" = None,
    ) -> "GLiBRLSAC":
        from metaworld_algorithms.envs.metaworld import MetaworldMetaLearningConfig

        assert isinstance(env_config.observation_space, gym.spaces.Box)
        assert isinstance(env_config.action_space, gym.spaces.Box)

        s_dim = env_config.observation_space.shape[-1]
        a_dim = env_config.action_space.shape[-1]
        n_env_tasks = self.num_env_tasks  # number of envs stepped in rollout
        global_episodic_return: Deque[float] = deque([], maxlen=20 * n_env_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * n_env_tasks)
        _episode_return_accum = np.zeros(envs.num_envs, dtype=np.float64)
        _episode_length_accum = np.zeros(envs.num_envs, dtype=np.int64)

        start_step, episodes_ended = 0, 0
        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        # ---- Single MultiTaskReplayBuffer: raw (s, a, s', r) only ----
        buffer = MultiTaskReplayBuffer(
            total_capacity=config.buffer_size * n_env_tasks,
            num_tasks=n_env_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

        if buffer_checkpoint is not None:
            buffer.load_checkpoint(buffer_checkpoint)

        # Sample tasks and reset envs
        envs.call("sample_tasks")
        obs, _ = envs.reset()

        # init_task resized for rollout (n_env_tasks may differ from num_tasks)
        init_task_env = jax.tree.map(
            lambda x: jnp.resize(x, (n_env_tasks,) + x.shape[1:]), self.init_task
        )
        tasktuple = init_task_env

        start_time = __import__("time").time()

        sac_logs: LogDict = {}
        model_logs: LogDict = {}

        # PEARL-style: collect rollout_steps per iteration, then update
        rollout_steps = config.rollout_steps
        steps_per_iter = rollout_steps * envs.num_envs
        n_iters = config.total_steps // steps_per_iter

        # Recover iteration index from step checkpoint
        start_iter = start_step // rollout_steps

        # Per-task batch size for the update phase
        B_t = config.batch_size

        # ----------------------------------------------------------------
        # Local helper: run evaluation at any step
        # ----------------------------------------------------------------
        def _run_eval(at_step: int) -> None:
            print("- Evaluating on the test set...")
            _eval_envs = env_config.spawn_test(seed)
            _wrapped = self.wrap()
            try:
                (
                    mean_success_rate_dict,
                    mean_returns_dict,
                    mean_success_per_task_dict,
                    mean_t_loss,
                    mean_r_loss,
                    mean_t_loss_per_task,
                    mean_r_loss_per_task,
                ) = env_config.evaluate_metalearning(_eval_envs, _wrapped)

                for step in range(env_config.evaluation_adaptation_steps + 1):
                    print(
                        f"Mean eval success {mean_success_rate_dict[step]:.4f}, "
                        f"return: {mean_returns_dict[step]:.4f}, "
                        f"adapt step: {step}"
                    )

                if track:
                    eval_metrics: LogDict = {
                        f"charts/mean_success_rate_adapt_{step}": mean_success_rate_dict[step]
                        for step in range(env_config.evaluation_adaptation_steps + 1)
                    } | {
                        f"charts/mean_evaluation_return_adapt_{step}": mean_returns_dict[step]
                        for step in range(env_config.evaluation_adaptation_steps + 1)
                    } | {
                        f"charts/mean_t_loss_{step}": mean_t_loss[step]
                        for step in range(env_config.evaluation_adaptation_steps + 1)
                    } | {
                        f"charts/mean_r_loss_{step}": mean_r_loss[step]
                        for step in range(env_config.evaluation_adaptation_steps + 1)
                    }
                    log(eval_metrics, step=at_step)
            finally:
                _eval_envs.close()
                del _eval_envs, _wrapped

        # Visualise prior (before any model training) when starting fresh
        if start_iter == 0:
            _run_eval(at_step=0)

        for iter_idx in range(start_iter, n_iters):

            # ================================================================
            # Optionally resample tasks when the buffer is full
            # ================================================================
            if config.resample_tasks_when_full and buffer.full:
                envs.call("sample_tasks")
                obs, _ = envs.reset()
                tasktuple = init_task_env
                buffer.reset(save_rewards=False)

            # ================================================================
            # Collect rollout_steps environment transitions
            # ================================================================
            for rollout_step in range(rollout_steps):
                global_step = iter_idx * rollout_steps + rollout_step
                total_steps = global_step * envs.num_envs

                # ---- Task-conditioned observation for action selection ----
                task_feat = self.reshape_task(tasktuple)  # (N, 1, task_dim)
                obs_with_task = np.array(
                    self.concat_task_with_observation(task_feat, obs)
                )  # (N, s_dim + task_dim)

                # ---- Sample action ----
                if total_steps < config.warmstart_steps:
                    actions = envs.action_space.sample()
                else:
                    self, actions = self.sample_action(obs_with_task)
                actions = np.array(actions)

                # ---- Step environment ----
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                dones = np.logical_or(terminations, truncations)

                # ---- Online Bayesian task posterior update ----
                s_feat_p = (
                    self.s_feat_model.params
                    if self.use_sa_encoder and self.s_feat_model is not None
                    else None
                )
                a_feat_p = (
                    self.a_feat_model.params
                    if self.use_sa_encoder and self.a_feat_model is not None
                    else None
                )
                ret = self.task_posterior(
                    tasktuple, obs, actions, rewards, next_obs,
                    self.t_mix_model.params,
                    self.t_bias_model.params,
                    self.r_mix_model.params,
                    self.r_bias_model.params,
                    s_feat_p,
                    a_feat_p,
                )
                tasktuple_new = ret[-1]

                # Do not update belief on auto-reset transitions
                done_mask = dones.reshape((-1,) + (1,) * (tasktuple[0].ndim - 1))
                tasktuple = tuple(
                    jnp.where(done_mask, old, new)
                    for old, new in zip(tasktuple, tasktuple_new)
                )

                # ---- Store raw (s, a, s', r) — no task features in buffer ----
                buffer_next_obs = next_obs.copy()
                if "final_obs" in infos:
                    # gymnasium object array: done envs have the true final obs,
                    # non-done envs have None — use _final_obs mask when available.
                    mask = infos.get("_final_obs", dones.astype(bool))
                    for i in np.where(mask)[0]:
                        buffer_next_obs[i] = infos["final_obs"][i]

                buffer.add(obs, buffer_next_obs, actions, rewards, terminations)

                # Reset task posterior to prior at episode end
                tasktuple = tuple(
                    jnp.where(done_mask, init, cur)
                    for init, cur in zip(init_task_env, tasktuple)
                )
                # tasktuple = tasktuple_new

                obs = next_obs

                # ---- Logging ----
                _episode_return_accum += np.array(rewards, dtype=np.float64)
                _episode_length_accum += 1
                for i, done in enumerate(dones):
                    if done:
                        logged = False
                        try:
                            global_episodic_return.append(infos["final_info"]["episode"]["r"][i])
                            global_episodic_length.append(infos["final_info"]["episode"]["l"][i])
                            logged = True
                        except (KeyError, TypeError):
                            pass
                        if not logged:
                            global_episodic_return.append(_episode_return_accum[i])
                            global_episodic_length.append(_episode_length_accum[i])
                        _episode_return_accum[i] = 0.0
                        _episode_length_accum[i] = 0
                        episodes_ended += 1

                if global_step % 500 == 0 and global_episodic_return:
                    mean_return = np.mean(list(global_episodic_return))
                    print(
                        f"global_step={total_steps}, mean_episodic_return={mean_return:.3f}"
                    )
                    if track:
                        log(
                            {
                                "charts/mean_episodic_return": mean_return,
                                "charts/mean_episodic_length": np.mean(
                                    list(global_episodic_length)
                                ),
                            },
                            step=total_steps,
                        )

            # ================================================================
            # End of rollout: gradient updates
            # ================================================================
            total_steps = (iter_idx + 1) * steps_per_iter

            if total_steps > config.warmstart_steps:
                # ---- SAC updates ----
                # For each task, sample B_t transitions and compute per-task
                # posterior M_T, M_R from that batch; concatenate task features
                # with observations to form task-conditioned SAC inputs.
                _policy_hi = config.policy_batch_size_range[1] if config.policy_batch_size_range is not None else B_t

                def _sample_tasks(n_samples: int) -> tuple:
                    """Sample n_samples transitions from num_tasks randomly chosen tasks."""
                    N = self.num_env_tasks
                    observations, actions, next_observations, dones, rewards = buffer.sample_separate(n_samples, n_tasks=self.num_env_tasks)
                    obs  = observations.reshape(n_samples, N, -1).swapaxes(0, 1)
                    act  = actions.reshape(n_samples, N, -1).swapaxes(0, 1)
                    rew  = rewards.reshape(n_samples, N, -1).swapaxes(0, 1)
                    nobs = next_observations.reshape(n_samples, N, -1).swapaxes(0, 1)
                    done = dones.reshape(n_samples, N, -1).swapaxes(0, 1)
                    return obs, act, rew, nobs, done
     
                for _ in range(config.num_gradient_steps):
                    obs_pt, act_pt, rew_pt, nobs_pt, done_pt = _sample_tasks(_policy_hi)
                    if config.policy_batch_size_range is not None:
                        lo, hi = config.policy_batch_size_range
                        k = jnp.array(int(np.random.randint(lo, hi + 1)))
                    else:
                        k = None
                    self, sac_logs = self._update_sac_inner_pearl(
                        obs_pt, act_pt, rew_pt, nobs_pt, done_pt, k=k
                    )

                # ---- Model updates ----
                model_active = (
                    config.model_train_until_steps is None
                    or total_steps <= config.model_train_until_steps
                )
                if model_active:
                    for _ in range(config.num_model_steps):
                        obs_pt, act_pt, rew_pt, nobs_pt, _ = _sample_tasks(B_t)
                        self, model_logs = self.update_models_pearl(obs_pt, act_pt, rew_pt, nobs_pt)

                if track:
                    sps = int(total_steps / (__import__("time").time() - start_time))
                    log(
                        {
                            **{k: float(v) for k, v in sac_logs.items()},
                            **{k: float(v) for k, v in model_logs.items()},
                            "charts/SPS": sps,
                        },
                        step=total_steps,
                    )

            # ================================================================
            # Evaluation
            # ================================================================
            if total_steps % config.evaluation_frequency == 0 and total_steps > 0:
                _run_eval(at_step=total_steps)

        return self

    # -----------------------------------------------------------------------
    # Wrapped agent for evaluation (PEARL-like online inference)
    # -----------------------------------------------------------------------

    class GLiBRLSACWrapped(MetaLearningAgent):
        """Wraps GLiBRLSAC for evaluation with online Bayesian task inference."""

        _current_task: TaskTuple

        def __init__(self, agent: "GLiBRLSAC"):
            self._agent = agent
            self._init = True

        def init(self, num_envs: int | None = None) -> None:
            init_task = self._agent.get_init_task()
            if num_envs is not None and init_task[0].shape[0] != num_envs:
                init_task = jax.tree.map(
                    lambda x: jnp.resize(x, (num_envs,) + x.shape[1:]), init_task
                )
            self._current_task = init_task
            self._t_mix_model = self._agent.t_mix_model
            self._t_bias_model = self._agent.t_bias_model
            self._r_mix_model = self._agent.r_mix_model
            self._r_bias_model = self._agent.r_bias_model
            self._s_feat_model = self._agent.s_feat_model
            self._a_feat_model = self._agent.a_feat_model

            self.n_tasks = self._current_task[0].shape[0]
            self._posterior_fixed_idx = np.arange(self.n_tasks)
            self._obs = None
            self._action = None
            self._reward = None
            self._counter = np.zeros((self.n_tasks,), dtype=np.int32)
            self.adapt()

        def init_task(self) -> TaskTuple:
            return self._agent.init_task

        def _update_task(self, new_observation: Observation) -> TaskWithObservation:
            if self._obs is not None and len(self._posterior_fixed_idx) < self.n_tasks:
                ret = self._agent.task_posterior(
                    self._current_task,
                    self._obs,
                    self._action,
                    self._reward,
                    new_observation,
                    self._t_mix_model.params,
                    self._t_bias_model.params,
                    self._r_mix_model.params,
                    self._r_bias_model.params,
                    self._s_feat_model.params if self._s_feat_model is not None else None,
                    self._a_feat_model.params if self._a_feat_model is not None else None,
                )
                new_task = jax.device_get(ret[-1])
                self._counter += 1

                if len(self._posterior_fixed_idx) > 0:
                    new_task_list = []
                    for i in range(len(self._current_task)):
                        arr = np.array(new_task[i])
                        arr[self._posterior_fixed_idx] = np.array(
                            self._current_task[i]
                        )[self._posterior_fixed_idx]
                        new_task_list.append(arr)
                    self._current_task = tuple(new_task_list)
                    self._counter[self._posterior_fixed_idx] -= 1
                else:
                    self._current_task = new_task

            task_reshaped = self._agent.reshape_task(self._current_task)
            task_with_obs = self._agent.concat_task_with_observation(
                task_reshaped, new_observation
            )
            self._posterior_fixed_idx = np.empty(0)
            return task_with_obs

        def adapt_action(
            self, observations: npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], dict]:
            task_with_obs = self._update_task(observations)
            self._agent, action, aux_policy_outs = self._agent.sample_action_and_aux(
                task_with_obs
            )
            return action, aux_policy_outs

        def step(self, timestep: Timestep) -> None:
            self._obs = timestep.observation
            self._action = timestep.action
            self._reward = timestep.reward

        def adapt(self) -> None:
            self._adapted_task = self._current_task
            self._adapted_t_mix_model = self._t_mix_model
            self._adapted_t_bias_model = self._t_bias_model
            self._adapted_r_mix_model = self._r_mix_model
            self._adapted_r_bias_model = self._r_bias_model
            self._adapted_s_feat_model = self._s_feat_model
            self._adapted_a_feat_model = self._a_feat_model

        def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
            self._posterior_fixed_idx = np.argwhere(env_mask)
            self._counter[self._posterior_fixed_idx] = 0

            if len(self._posterior_fixed_idx) > 0:
                current_task_list = []
                for i in range(len(self._current_task)):
                    arr = np.array(self._current_task[i])
                    arr[self._posterior_fixed_idx] = np.array(
                        self._adapted_task[i]
                    )[self._posterior_fixed_idx]
                    current_task_list.append(arr)
                self._current_task = tuple(current_task_list)

            self._init = False

        def predictive_losses(
            self, obs_next_gt: Observation, r_gt: Reward
        ) -> tuple[np.ndarray, np.ndarray]:
            mt = self._current_task[0]
            mr = self._current_task[5]

            obs_ = self._obs[:, None, None]
            action_ = self._action[:, None, None]
            s_feat_p = self._s_feat_model.params if self._s_feat_model is not None else None
            a_feat_p = self._a_feat_model.params if self._a_feat_model is not None else None
            s_feat_ = self._agent.get_state_feat(s_feat_p, obs_)
            a_feat_ = self._agent.get_action_feat(a_feat_p, action_)
            sa = jnp.concatenate([s_feat_, a_feat_], axis=-1)
            ct, bt = self._agent.get_taskt_feat_bias(
                self._agent.t_mix_model.params, self._agent.t_bias_model.params, sa, obs_
            )
            obs_next_predicted = self._agent.t_likelihood_mean(ct, bt, mt)

            obs_next_ = obs_next_gt.reshape(obs_next_predicted.shape[0], 1, 1, -1)
            s_next_feat_ = self._agent.get_state_feat(s_feat_p, obs_next_)
            sas = (
                jnp.concatenate([sa, s_next_feat_], axis=-1)
                if self._agent.use_next_obs_in_r_mix
                else sa
            )
            cr, br = self._agent.get_taskr_feat_bias(
                self._agent.r_mix_model.params, self._agent.r_bias_model.params, sas
            )
            r_predicted = self._agent.r_likelihood_mean(cr, br, mr)

            obs_next_predicted = obs_next_predicted.reshape(-1, obs_next_gt.shape[-1])
            r_predicted = r_predicted.reshape(-1, 1)

            t_loss = np.linalg.norm(obs_next_predicted - obs_next_gt, axis=-1)
            r_loss = np.linalg.norm(r_gt[:, None] - r_predicted, axis=-1)
            return t_loss, r_loss

        def eval_action(
            self, observations: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            task_with_obs = self._update_task(observations)
            action = self._agent.eval_action(task_with_obs)
            return action

    @override
    def wrap(self) -> MetaLearningAgent:
        return GLiBRLSAC.GLiBRLSACWrapped(self)
