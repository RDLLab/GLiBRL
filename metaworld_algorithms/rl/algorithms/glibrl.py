import dataclasses
from typing import Self, override

from collections import defaultdict
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from flax.core import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray

from metaworld_algorithms.config.envs import MetaLearningEnvConfig
from metaworld_algorithms.config.networks import (ContinuousActionPolicyConfig, 
                                                  NeuralNetworkConfig,
                                                  VanillaNetworkConfig,)
from metaworld_algorithms.nn.base import VanillaNetwork
from metaworld_algorithms.config.utils import Activation, Optimizer
from metaworld_algorithms.config.rl import AlgorithmConfig
from metaworld_algorithms.nn.distributions import TanhMultivariateNormalDiag
from metaworld_algorithms.rl.algorithms.utils import TrainState

from metaworld_algorithms.config.optim import OptimizerConfig

from metaworld_algorithms.rl.networks import (
    ContinuousActionPolicy,
)

from metaworld_algorithms.types import (
    Action,
    Reward,
    AuxPolicyOutputs,
    LogDict,
    LogProb,
    MetaLearningAgent,
    Observation,
    ObsFeat,
    ActFeat,
    ObsActFeat,
    ObsActObsFeat,
    Task,
    TaskTuple,
    TaskTuple,
    TaskWithObservation,
    TaskTFeat,
    TaskTBias,
    TaskRFeat,
    TaskRBias,
    TaskTMean,
    TaskRMean,
    RolloutWithTask,
    Timestep
)

from .base import BRLMetaLearningAlgorithm
from .utils import (
    LinearFeatureBaseline,
    compute_gae,
    to_deterministic_minibatch_iterator_with_task,
    normalize_advantages,
)


@jax.jit
def _sample_action(
    policy: TrainState, observation: TaskWithObservation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    policy: TrainState, observation: TaskWithObservation
) -> Float[Array, "... action_dim"]:
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    return dist.mode()


@jax.jit
def _sample_action_dist(
    policy: TrainState,
    observation: TaskWithObservation,
    key: PRNGKeyArray,
) -> tuple[
    Action,
    LogProb,
    Action,
    Action,
    PRNGKeyArray,
]:
    key, action_key = jax.random.split(key)
    dist = policy.apply_fn(policy.params, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)

    if isinstance(dist, TanhMultivariateNormalDiag):
        # HACK: use pre-tanh distributions for kl divergence
        mean = dist.pre_tanh_mean()
        std = dist.pre_tanh_std()
    else:
        mean = dist.mode()
        std = dist.stddev()

    return action, action_log_prob, mean, std, key  # pyright: ignore[reportReturnType]


@dataclasses.dataclass(frozen=True)
class GLiBRLConfig(AlgorithmConfig):
    policy_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    
    s_feat_config: NeuralNetworkConfig = VanillaNetworkConfig(
        width=(64, 32), activation=Activation.ReLU,
        optimizer=OptimizerConfig(
            lr=2e-4,
            optimizer=Optimizer.Adam,
        ),
    )
    a_feat_config: NeuralNetworkConfig = VanillaNetworkConfig(
        width=(32, 16), activation=Activation.ReLU,
        optimizer=OptimizerConfig(
            lr=2e-4,
            optimizer=Optimizer.Adam,
        ),
    )

    hidden_state_dim: int = 32
    hidden_action_dim: int = 16
    
    t_mix_config: NeuralNetworkConfig = VanillaNetworkConfig(
        width=(64, 32), activation=Activation.ReLU,
        optimizer=OptimizerConfig(
            lr=2e-4, optimizer=Optimizer.Adam,
        ),
        use_layer_norm=True
    )
    
    t_bias_config: NeuralNetworkConfig = VanillaNetworkConfig(
        width=(64, 64), activation=Activation.ReLU,
        optimizer=OptimizerConfig(
            lr=2e-4,
            optimizer=Optimizer.Adam,
        ),
        use_layer_norm=True
    )
    r_mix_config: NeuralNetworkConfig = VanillaNetworkConfig(
        width=(128, 64), activation=Activation.ReLU,
        optimizer=OptimizerConfig(
            lr=2e-4, optimizer=Optimizer.Adam,
        ),
        use_layer_norm=True
    )
    r_bias_config: NeuralNetworkConfig = VanillaNetworkConfig(
        width=(128, 128), activation=Activation.ReLU,
        optimizer=OptimizerConfig(
            lr=2e-4, optimizer=Optimizer.Adam,
        ),
        use_layer_norm=True
    )

    full_bayesian: bool = True
    noise_mode: str = "full"  # "full" | "diag" | "scalar"

    horizon: int = 1000
    transition_latent_dim: int = 16
    reward_latent_dim: int = 256

    update_rate: float = 1

    t_reg: float = 2e-4
    r_reg: float = 2e-4

    meta_batch_size: int = 20
    clip_eps: float = 0.2

    entropy_coefficient: float = 5e-3
    normalize_advantages: bool = False
    gae_lambda: float = 0.95
    num_epochs: int = 10
    num_gradient_steps: int = 10

    use_sa_encoder: bool = True
    use_bias: bool = False
    normalise_task: bool = False
    quadratic_mt: bool = True

    use_task_encoder: bool = False
    task_encoder_dim: int = 64
    task_encoder_config: NeuralNetworkConfig = VanillaNetworkConfig(
        width=(256, 256), activation=Activation.ReLU,
        optimizer=OptimizerConfig(
            lr=5e-4, max_grad_norm=1, optimizer=Optimizer.Adam,
        ),
        use_layer_norm=True
    )

    dtype: npt.DTypeLike = np.float32



class GLiBRL(BRLMetaLearningAlgorithm[GLiBRLConfig]):
    policy: TrainState
    task_encoder_model: TrainState | None
    key: PRNGKeyArray

    horizon: int = struct.field(pytree_node=False)
    update_rate: float = struct.field(pytree_node=False)
    num_gradient_steps: int = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    policy_squash_tanh: bool = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    normalize_advantages: bool = struct.field(pytree_node=False)
    gae_lambda: float = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    hidden_state_dim: int = struct.field(pytree_node=False)
    hidden_action_dim: int = struct.field(pytree_node=False)
    t_reg: float = struct.field(pytree_node=False)
    r_reg: float = struct.field(pytree_node=False)
    full_bayesian: bool = struct.field(pytree_node=False)
    noise_mode: str = struct.field(pytree_node=False)
    use_bias: bool = struct.field(pytree_node=False)

    normalise_task: bool = struct.field(pytree_node=False)
    quadratic_mt: bool = struct.field(pytree_node=False)
    use_task_encoder: bool = struct.field(pytree_node=False)
    dtype: npt.DTypeLike = struct.field(pytree_node=False)
    

    @override
    @staticmethod
    def initialize(
        config: GLiBRLConfig,
        env_config: MetaLearningEnvConfig,
        seed: int = 1,
    ) -> "GLiBRL":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert config.noise_mode in ("full", "diag", "scalar"), (
            f"Unknown noise_mode: {config.noise_mode}"
        )
        assert config.full_bayesian or config.noise_mode == "full", (
            "noise_mode 'diag'/'scalar' requires full_bayesian=True"
        )

        master_key = jax.random.PRNGKey(seed)

        algorithm_key, policy_key, model_key = jax.random.split(master_key, 3)

        s_dim = env_config.observation_space.shape[-1]
        a_dim = env_config.action_space.shape[-1]

        policy_net = ContinuousActionPolicy(
            action_dim=int(np.prod(env_config.action_space.shape)),
            config=config.policy_config,
        )

        s_feat_net = VanillaNetwork(
            config=config.s_feat_config,
            head_dim=config.hidden_state_dim,
            activate_last=True
        )
        a_feat_net = VanillaNetwork(
            config=config.a_feat_config,
            head_dim=config.hidden_action_dim,
            activate_last=True
        )

        t_mix_net = VanillaNetwork(
            config=config.t_mix_config,
            head_dim=config.transition_latent_dim,
            activate_last=False
        )
        r_mix_net = VanillaNetwork(
            config=config.r_mix_config,
            head_dim=config.reward_latent_dim,
            activate_last=False
        )


        t_bias_net = VanillaNetwork(
            config=config.t_bias_config,
            head_dim=s_dim,
            activate_last=False
        )
        r_bias_net = VanillaNetwork(
            config=config.r_bias_config,
            head_dim=1,
            activate_last=False
        )


        s_feat_model = TrainState.create(
            params=s_feat_net.init(model_key, 
                jnp.zeros((config.meta_batch_size, 1, s_dim), dtype=config.dtype)
            ),
            tx=config.s_feat_config.optimizer.spawn(),
            apply_fn=s_feat_net.apply
        )

        a_feat_model = TrainState.create(
            params=a_feat_net.init(model_key, 
                jnp.zeros((config.meta_batch_size, 1, a_dim), dtype=config.dtype)
            ),
            tx=config.a_feat_config.optimizer.spawn(),
            apply_fn=a_feat_net.apply
        )

        t_mix_model = TrainState.create(
            params=t_mix_net.init(model_key, 
                jnp.zeros((config.meta_batch_size, 1, config.hidden_state_dim + config.hidden_action_dim), dtype=config.dtype)
            ),
            tx=config.t_mix_config.optimizer.spawn(),
            apply_fn=t_mix_net.apply
        )

        r_mix_model = TrainState.create(
            params=r_mix_net.init(model_key, 
                jnp.zeros((config.meta_batch_size, 1, 2 * config.hidden_state_dim + config.hidden_action_dim), dtype=config.dtype)
            ),
            tx=config.r_mix_config.optimizer.spawn(),
            apply_fn=r_mix_net.apply
        )

        t_bias_model = TrainState.create(
            params=t_bias_net.init(model_key, 
                jnp.zeros((config.meta_batch_size, 1,
                        config.hidden_state_dim + config.hidden_action_dim), dtype=config.dtype)
            ),
            tx=config.t_bias_config.optimizer.spawn(),
            apply_fn=t_bias_net.apply
        )

        r_bias_model = TrainState.create(
            params=r_bias_net.init(model_key, 
                jnp.zeros((config.meta_batch_size, 1,
                        2 * config.hidden_state_dim + config.hidden_action_dim), dtype=config.dtype)
            ),
            tx=config.r_bias_config.optimizer.spawn(),
            apply_fn=r_bias_net.apply
        )

                
        mt_dim = config.transition_latent_dim, s_dim
        mr_dim = config.reward_latent_dim, 1
        nut_dim = 1, 1
        nur_dim = 1, 1
        omegar_dim = 1, 1

        xit_dim = config.transition_latent_dim, config.transition_latent_dim
        # "full": omegat is an inverse-Wishart scale matrix, nut its dof.
        # "diag"/"scalar": omegat stores 2*rate and nut 2*shape of inverse-gamma posteriors.
        if config.noise_mode == "full":
            omegat_dim = s_dim, s_dim
        elif config.noise_mode == "diag":
            omegat_dim = 1, s_dim
        else:
            omegat_dim = 1, 1
        xir_dim = config.reward_latent_dim, config.reward_latent_dim


        init_mt = jnp.zeros((1, 1, *mt_dim), dtype=config.dtype).repeat(config.num_tasks, 0)
        init_mr = jnp.zeros((1, 1, *mr_dim), dtype=config.dtype).repeat(config.num_tasks, 0)
        if config.noise_mode == "full":
            init_nut = jnp.ones((1, 1, *nut_dim), dtype=config.dtype).repeat(config.num_tasks, 0) * s_dim + 1
        else:
            init_nut = jnp.ones((1, 1, *nut_dim), dtype=config.dtype).repeat(config.num_tasks, 0) * 2
        init_nur = jnp.ones((1, 1, *nur_dim), dtype=config.dtype).repeat(config.num_tasks, 0) + 1

        init_xit = jnp.eye(config.transition_latent_dim, dtype=config.dtype).reshape(1, 1, *xit_dim).repeat(config.num_tasks, 0)
        init_xit_inv = jnp.eye(config.transition_latent_dim, dtype=config.dtype).reshape(1, 1, *xit_dim).repeat(config.num_tasks, 0)
        if config.noise_mode == "full":
            init_omegat = jnp.eye(s_dim, dtype=config.dtype).reshape(1, 1, *omegat_dim).repeat(config.num_tasks, 0)
        else:
            init_omegat = jnp.ones((1, 1, *omegat_dim), dtype=config.dtype).repeat(config.num_tasks, 0)

        if not config.full_bayesian:
            init_omegat = init_omegat * init_nut

        init_xir = jnp.eye(config.reward_latent_dim, dtype=config.dtype).reshape(1, 1, *xir_dim).repeat(config.num_tasks, 0) 
        init_xir_inv = jnp.eye(config.reward_latent_dim, dtype=config.dtype).reshape(1, 1, *xir_dim).repeat(config.num_tasks, 0) 
        init_omegar = jnp.eye(1, dtype=config.dtype).reshape(1, 1, *omegar_dim).repeat(config.num_tasks, 0)

        if not config.full_bayesian:
            init_omegar = init_omegar * init_nur

        if config.use_task_encoder:
            task_dim = config.transition_latent_dim * s_dim + config.reward_latent_dim
            policy_task_dim = config.task_encoder_dim
        else:
            if config.quadratic_mt:
                mt_feat_dim = config.transition_latent_dim // 2 * (config.transition_latent_dim + 1)
            else:
                mt_feat_dim = config.transition_latent_dim * s_dim
            task_dim = mt_feat_dim + config.reward_latent_dim
            policy_task_dim = task_dim

        if config.use_task_encoder:
            task_encoder_net = VanillaNetwork(
                config=config.task_encoder_config,
                head_dim=config.task_encoder_dim,
                activate_last=False
            )
            task_encoder_model = TrainState.create(
                params=task_encoder_net.init(model_key,
                    jnp.zeros((config.meta_batch_size, 1, task_dim), dtype=config.dtype)
                ),
                tx=config.task_encoder_config.optimizer.spawn(),
                apply_fn=task_encoder_net.apply
            )
        else:
            task_encoder_model = None


        init_task = (
            init_mt,
            init_xit,
            init_xit_inv,  
            init_omegat,
            init_nut,
            init_mr,
            init_xir,
            init_xir_inv,  
            init_omegar,
            init_nur
        )

        dummy_task = jnp.zeros(
            (config.meta_batch_size, policy_task_dim), dtype=config.dtype
        )

        dummy_obs = jnp.zeros(
            (config.meta_batch_size, s_dim), dtype=config.dtype
        )


        dummy_input = jnp.concat([dummy_task, dummy_obs], axis=-1)


        policy = TrainState.create(
            params=policy_net.init(policy_key, dummy_input),
            tx=config.policy_config.network_config.optimizer.spawn(), 
            apply_fn=policy_net.apply
        )


        return GLiBRL(
            num_tasks=config.num_tasks,
            num_gradient_steps=config.num_gradient_steps,
            num_epochs=config.num_epochs,
            policy=policy,
            task_encoder_model=task_encoder_model,
            s_feat_model=s_feat_model,
            a_feat_model=a_feat_model,
            t_mix_model=t_mix_model,
            r_mix_model=r_mix_model,
            t_bias_model=t_bias_model,
            r_bias_model=r_bias_model,
            gamma=config.gamma,
            clip_eps=config.clip_eps,
            entropy_coefficient=config.entropy_coefficient,
            normalize_advantages=config.normalize_advantages,
            policy_squash_tanh=config.policy_config.squash_tanh,
            key=algorithm_key,
            gae_lambda=config.gae_lambda,
            state_dim=s_dim,
            hidden_state_dim=config.hidden_state_dim,
            hidden_action_dim=config.hidden_action_dim,
            transition_latent_dim=config.transition_latent_dim,
            reward_latent_dim=config.reward_latent_dim,
            task_dim=task_dim,
            init_task=init_task,
            t_reg=config.t_reg,
            r_reg=config.r_reg,
            use_sa_encoder=config.use_sa_encoder,
            use_bias=config.use_bias,
            horizon=config.horizon,
            update_rate=config.update_rate,
            normalise_task=config.normalise_task,
            quadratic_mt=config.quadratic_mt,
            use_task_encoder=config.use_task_encoder,
            full_bayesian=config.full_bayesian,
            noise_mode=config.noise_mode,
            dtype=config.dtype,
        )
    
    @override
    def get_num_params(self):
        s_feat_num_params = sum(
            x.size for x in jax.tree.leaves(self.s_feat_model.params)
        )
        a_feat_num_params = sum(
            x.size for x in jax.tree.leaves(self.a_feat_model.params)
        )
        t_mix_num_params = sum(
            x.size for x in jax.tree.leaves(self.t_mix_model.params)
        )
        r_mix_num_params = sum(
            x.size for x in jax.tree.leaves(self.r_mix_model.params)
        ) 
        if self.use_bias:
            t_bias_num_params = sum(
                x.size for x in jax.tree.leaves(self.t_bias_model.params)
            )
            r_bias_num_params = sum(
                x.size for x in jax.tree.leaves(self.r_bias_model.params)
            )
        else:
            t_bias_num_params = 0
            r_bias_num_params = 0
        if self.use_task_encoder and self.task_encoder_model is not None:
            task_encoder_num_params = sum(
                x.size for x in jax.tree.leaves(self.task_encoder_model.params)
            )
        else:
            task_encoder_num_params = 0
        return {
            "policy_num_params": sum(
                x.size for x in jax.tree.leaves(self.policy.params)
            ) + task_encoder_num_params,
            "model_num_params": s_feat_num_params
                                + a_feat_num_params
                                + t_mix_num_params
                                + r_mix_num_params
                                + t_bias_num_params
                                + r_bias_num_params,
        }
    
    def get_init_task(self) -> TaskTuple:
        return self.init_task
    
    def set_init_task(self, mt, xit, omegat,
        mr, xir, omegar) -> TaskTuple:
        xit_inv = jnp.linalg.solve(xit, jnp.eye(xit.shape[-1]))
        xir_inv = jnp.linalg.solve(xir, jnp.eye(xir.shape[-1]))
        return (mt, xit, xit_inv, omegat, mr, xir, xir_inv, omegar)
    
    def sample_action_and_aux(
        self, observation: TaskWithObservation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        rets = _sample_action_dist(self.policy, observation, self.key)
        action, log_prob, mean, std = jax.device_get(rets[:-1])
        key = rets[-1]
        return (
            self.replace(key=key),
            action,
            {"log_prob": log_prob, "mean": mean, "std": std},
        )

    def sample_action(
        self, observation: TaskWithObservation
    ) -> tuple[Self, Action]:
        action, key = _sample_action(self.policy, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    def eval_action(
        self, observations: TaskWithObservation) -> Action:
        action = _eval_action(self.policy, observations)
        return jax.device_get(action)

    def compute_advantages(self, rollouts: RolloutWithTask) -> RolloutWithTask:
        new_dones = np.zeros_like(rollouts.dones)
        new_dones[0] = 1.0
        rollouts = rollouts._replace(dones=new_dones)


        values, returns = LinearFeatureBaseline.get_baseline_values_and_returns(
            rollouts, self.gamma
        )
        rollouts = rollouts._replace(values=values, returns=returns)

        # NOTE: assume the final states are terminal
        dones = np.ones(rollouts.rewards.shape[1:], dtype=rollouts.rewards.dtype)
        rollouts = compute_gae(
            rollouts, self.gamma, self.gae_lambda, last_values=None, dones=dones
        )
        if self.normalize_advantages:
            rollouts = normalize_advantages(rollouts)
        return rollouts
    
    class GLiBRLWrapped(MetaLearningAgent):
        _current_task: TaskTuple

        def __init__(self, agent: "GLiBRL"):
            self._agent = agent

        def init(self) -> None:
            self._current_task = self._agent.get_init_task()
            self._s_feat_model = self._agent.s_feat_model
            self._a_feat_model = self._agent.a_feat_model
            self._t_mix_model = self._agent.t_mix_model
            self._t_bias_model = self._agent.t_bias_model
            self._r_mix_model = self._agent.r_mix_model
            self._r_bias_model = self._agent.r_bias_model
            self._task_encoder_model = self._agent.task_encoder_model


            self.n_tasks = self._current_task[0].shape[0]
            self._posterior_fixed_idx = np.arange(self.n_tasks)
            self._obs = None
            self._action = None
            self._reward = None
            self._counter = np.zeros((self.n_tasks,), dtype=np.int32)
            self.adapt()


        def _update_task(self, new_observation: Observation) -> TaskWithObservation:
            if len(self._posterior_fixed_idx) < self.n_tasks:
                ret = self._agent.task_posterior(
                    self._current_task, self._obs, self._action,
                    self._reward, new_observation,
                    self._t_mix_model.params,
                    self._t_bias_model.params,
                    self._r_mix_model.params,
                    self._r_bias_model.params,
                    self._s_feat_model.params,
                    self._a_feat_model.params,
                )
                new_task = ret[-1]
                self._counter += 1 

                if len(self._posterior_fixed_idx) > 0:
                    new_task_list = []
                    for i in range(len(self._current_task)):
                        new_task_list.append(
                            new_task[i].at[self._posterior_fixed_idx].set(
                                self._current_task[i][self._posterior_fixed_idx]
                            )
                        )
                    self._current_task = tuple(new_task_list)
                    self._counter[self._posterior_fixed_idx] -= 1
                else:
                    self._current_task = new_task

            task_reshaped = self._agent.reshape_task(self._current_task)
            task_with_obs = self._agent.concat_task_with_observation(
                task_reshaped,
                new_observation,
                self._task_encoder_model.params
                if self._task_encoder_model is not None
                else None,
            )
            self._posterior_fixed_idx = np.empty(0)
            return task_with_obs

        def adapt_action(
            self, observations: npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
            

            task_with_obs = self._update_task(observations)
            self._agent, action, aux_policy_outs = (
                self._agent.sample_action_and_aux(task_with_obs)
            )
            return action, aux_policy_outs

        def step(self, timestep: Timestep) -> None:
            self._obs = timestep.observation
            self._action = timestep.action
            self._reward = timestep.reward

        def adapt(self) -> None:
            self._adapted_task = self._current_task
            self._adapted_s_feat_model = self._s_feat_model
            self._adapted_a_feat_model = self._a_feat_model
            self._adapted_t_mix_model = self._t_mix_model
            self._adapted_t_bias_model = self._t_bias_model
            self._adapted_r_mix_model = self._r_mix_model
            self._adapted_r_bias_model = self._r_bias_model
            self._adapted_task_encoder_model = self._task_encoder_model

        def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
            self._posterior_fixed_idx = np.argwhere(env_mask)
            self._counter[self._posterior_fixed_idx] = 0

            if len(self._posterior_fixed_idx) > 0:
                current_task_list = []
                for i in range(len(self._current_task)):
                    current_task_list.append(
                        self._current_task[i].at[self._posterior_fixed_idx].set(
                            self._adapted_task[i][self._posterior_fixed_idx]
                        )
                    )
                self._current_task = tuple(current_task_list)

        def predictive_losses(self, obs_next_gt: Observation, r_gt: Reward):
            mt = self._current_task[0]
            mr = self._current_task[5]

            s_feat = self._agent.get_state_feat(self._agent.s_feat_model.params, self._obs)[:, None, None]
            a_feat = self._agent.get_action_feat(self._agent.a_feat_model.params, self._action)[:, None, None]

            sa_feat = self._agent.sa_concat(s_feat, a_feat)
            ct, bt = self._agent.get_taskt_feat_bias(self._agent.t_mix_model.params, self._agent.t_bias_model.params, sa_feat)
            obs_next_predicted = self._agent.t_likelihood_mean(ct, bt, mt)

            s_next_feat = self._agent.get_state_feat(self._agent.s_feat_model.params, obs_next_gt.reshape(obs_next_predicted.shape))
            sas_feat = self._agent.sa_s_concat(sa_feat, s_next_feat)
            cr, br = self._agent.get_taskr_feat_bias(self._agent.r_mix_model.params, self._agent.r_bias_model.params, sas_feat)
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
    def wrap(self):
        return GLiBRL.GLiBRLWrapped(self)

    @jax.jit
    def t_likelihood_mean(
        self,
        ct: TaskTFeat,
        bt: TaskTBias,
        t_mu: TaskTMean
    ) -> Observation:
        return ct @ t_mu + bt
    
    @jax.jit
    def r_likelihood_mean(
        self,
        cr: TaskRFeat,
        br: TaskRBias,
        r_mu: TaskRMean
    ) -> Reward:
        return cr @ r_mu + br

    @jax.jit
    def datafeat_to_taskfeat(
        self,
        s_feat: ObsFeat,
        a_feat: ActFeat,
        s_next_feat: ObsFeat,
        t_mix_params: FrozenDict,
        t_bias_params: FrozenDict,
        r_mix_params: FrozenDict,
        r_bias_params: FrozenDict
    ) -> tuple[TaskTFeat, TaskTBias, TaskRFeat, TaskRBias]:
        sa_feat = self.sa_concat(s_feat, a_feat)
        sas_feat = self.sa_s_concat(sa_feat, s_next_feat)
        ct, bt = self.get_taskt_feat_bias(t_mix_params,
                                          t_bias_params,
                                          sa_feat)
        cr, br = self.get_taskr_feat_bias(r_mix_params,
                                          r_bias_params,
                                          sas_feat)
        return ct, bt, cr, br


    @jax.jit
    def reshape_task(self, tasktuple: TaskTuple):
        def get_lower_triangle(v):
            row, col = jnp.tril_indices(v.shape[-1])
            return v[..., row, col]
        
        mt, _, xit_inv, omegat, _, \
            mr, _, xir_inv, omegar, nur = tasktuple

        if self.use_task_encoder:
            # Raw posterior means, flattened. The learned encoder consumes these
            # directly, so `quadratic_mt` / `normalise_task` are deliberately skipped.
            return jnp.concat(
                (
                    mt.reshape(mt.shape[0], mt.shape[1], -1),
                    mr.reshape(mr.shape[0], mr.shape[1], -1),
                ),
                axis=-1,
            )

        if self.quadratic_mt:
            mt_feat = get_lower_triangle(mt @ mt.swapaxes(-1, -2))
        else:
            mt_feat = mt

        mt_reshaped = mt_feat.reshape(mt_feat.shape[0], mt_feat.shape[1], -1)
        mr_reshaped = mr.reshape(mr.shape[0], mr.shape[1], -1)
        
        if self.normalise_task:
            mr_norm = jnp.linalg.norm(mr_reshaped, axis=-1, keepdims=True) + 1e-8
            mr_reshaped = mr_reshaped / mr_norm
            mt_norm = jnp.linalg.norm(mt_reshaped, axis=-1, keepdims=True) + 1e-8
            mt_reshaped = mt_reshaped / mt_norm

        reshaped_tasks = (
            mt_reshaped,
            mr_reshaped
        )

        return jnp.concat((reshaped_tasks), axis=-1)

    @jax.jit
    def encode_task(
        self, task: Task, task_encoder_params: FrozenDict | None = None
    ) -> Task:
        """Map the raw task vector through the learned representation network.

        A no-op unless `use_task_encoder` is set. `task_encoder_params` is taken from
        the current train state when not given; the policy loss passes them explicitly
        so that the encoder is differentiated through the RL objective.
        """
        if not self.use_task_encoder or self.task_encoder_model is None:
            return task
        if task_encoder_params is None:
            task_encoder_params = self.task_encoder_model.params
        return self.task_encoder_model.apply_fn(task_encoder_params, task)

    @jax.jit
    def concat_task_with_observation(
        self, task: Task,
        observation: Observation,
        task_encoder_params: FrozenDict | None = None,
        # compress_params: FrozenDict
    ) -> TaskWithObservation:

        task = self.encode_task(task, task_encoder_params)

        if len(observation.shape) == 2:
            obs = jnp.expand_dims(observation, 1)
        elif len(observation.shape) == 3:
            obs = observation.swapaxes(0, 1)
        else:
            obs = observation.squeeze(1)

        return jnp.concatenate([obs, task], axis=-1).squeeze()
    
    @jax.jit
    def get_state_feat(
        self, s_feat_param: FrozenDict, state: Observation
    ) -> ObsFeat:
        s_feat = self.s_feat_model.apply_fn(s_feat_param, state)
        return s_feat 
    
    @jax.jit
    def get_action_feat(
        self, a_feat_param: FrozenDict, action: Action
    ) -> ActFeat:
        a_feat = self.a_feat_model.apply_fn(a_feat_param, action)
        return a_feat 


    @jax.jit
    def sa_concat(
        self, s_feat: ObsFeat, a_feat: ActFeat  
    ) -> ObsActFeat:
        return jnp.concat([s_feat, a_feat], axis=-1)
    
    @jax.jit
    def sa_s_concat(
        self, sa_feat: ObsActFeat, s_next_feat: ObsFeat,  
    ) -> ObsActObsFeat:
        return jnp.concat([sa_feat, s_next_feat], axis=-1)
    
    @jax.jit
    def get_taskt_feat_bias(
        self, t_mix_param: FrozenDict, 
        t_bias_param: FrozenDict, 
        sa_feat: ObsActFeat
    ) -> tuple[TaskTFeat, TaskTBias]:
   
        ct = self.t_mix_model.apply_fn(t_mix_param, sa_feat)
        if self.use_bias:
            bt = self.t_bias_model.apply_fn(t_bias_param, sa_feat)
        else:
            bt = jnp.zeros(shape=(self.state_dim, ))
        
        return ct, bt
           
    @jax.jit
    def get_taskr_feat_bias(
        self, r_mix_param: FrozenDict, 
        r_bias_param: FrozenDict, 
        sas_feat: ObsActObsFeat
    ) -> tuple[TaskRFeat, TaskRBias]:
        
        cr = self.r_mix_model.apply_fn(r_mix_param, sas_feat)
        if self.use_bias:
            br = self.r_bias_model.apply_fn(r_bias_param, sas_feat)
        else:
            br = jnp.zeros(shape=(1, ))
        return cr, br
    
    @jax.jit
    def gather_feats(self, obs: Observation, action: Action,
        reward: Reward, obs_next: Observation,
        t_mix_params: FrozenDict,
        t_bias_params: FrozenDict,
        r_mix_params: FrozenDict,
        r_bias_params: FrozenDict,
        s_feat_params: FrozenDict,
        a_feat_params: FrozenDict,
    ) -> tuple[ObsFeat, ActFeat, TaskTFeat, TaskRFeat, TaskTBias, TaskRBias, Observation, Reward]:
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
        
        s_feat = self.get_state_feat(s_feat_params, obs_)
        a_feat = self.get_action_feat(a_feat_params, action_)
        s_next_feat = self.get_state_feat(s_feat_params, obs_next_)

        ct, bt, cr, br = self.datafeat_to_taskfeat(
            s_feat, a_feat, s_next_feat,
            t_mix_params, t_bias_params,
            r_mix_params, r_bias_params
        )

        s_minus_b = obs_next_ - bt
        r_minus_b = reward_ - br

        return (s_feat, a_feat, s_next_feat, ct, cr, bt, br, s_minus_b, r_minus_b)
    
    @jax.jit
    def task_posterior_inner(
        self, task: TaskTuple, ct: TaskTFeat, cr: TaskRFeat,
          s_minus_b: Observation, r_minus_b: Reward
    ) -> TaskTuple:
        
        mt, xit, xit_inv, omegat, nut, \
            mr, xir, xir_inv, omegar, nur = task
        N = ct.shape[2]

        ct_transpose = jnp.swapaxes(ct, -1, -2)
        cr_transpose = jnp.swapaxes(cr, -1, -2)

        s_minus_b_tranpose = jnp.swapaxes(s_minus_b, -1, -2)
        r_minus_b_tranpose = jnp.swapaxes(r_minus_b, -1, -2)

        mt_transpose = jnp.swapaxes(mt, -1, -2)
        mr_transpose = jnp.swapaxes(mr, -1, -2)

        xit_new = xit + self.update_rate * ct_transpose @ ct

        if N == 1:
            # matrx inversion lemma
            tmp = xit_inv @ ct_transpose
            tmp_transpose = jnp.swapaxes(tmp, -1, -2)
            xit_new_inv = xit_inv - self.update_rate * tmp @ tmp_transpose \
                                    / (1 + self.update_rate * ct @ tmp) 
        else:
            xit_new_inv = jnp.linalg.solve(
                xit_new, jnp.eye(xit_new.shape[-1], 
                                dtype=xit_new.dtype)
            )

        xitmt = xit @ mt
        tmp = self.update_rate * ct_transpose @ s_minus_b + xitmt
        tmp_transpose = jnp.swapaxes(tmp, -1, -2)
        mt_new = xit_new_inv @ tmp
        
        if self.full_bayesian:
            if self.noise_mode == "full":
                omegat_new = omegat + self.update_rate * s_minus_b_tranpose @ s_minus_b + \
                    mt_transpose @ xitmt - \
                    tmp_transpose @ mt_new
            else:
                # independent per-output BLR rate updates, vmapped over output dims;
                # equals the diagonal of the full-Wishart scale update
                def per_dim_rate(r_i, mt_i, xitmt_i, tmp_i, mt_new_i):
                    return (
                        self.update_rate * jnp.sum(r_i * r_i, axis=-1)
                        + jnp.sum(mt_i * xitmt_i, axis=-1)
                        - jnp.sum(tmp_i * mt_new_i, axis=-1)
                    )

                rate_update = jax.vmap(per_dim_rate, in_axes=-1, out_axes=-1)(
                    s_minus_b, mt, xitmt, tmp, mt_new
                )[..., None, :]
                if self.noise_mode == "scalar":
                    rate_update = jnp.sum(rate_update, axis=-1, keepdims=True)
                omegat_new = omegat + rate_update
        else:
            omegat_new = omegat

        
        xir_new = xir + self.update_rate * cr_transpose @ cr

        if N == 1:
            tmp = xir_inv @ cr_transpose
            tmp_transpose = jnp.swapaxes(tmp, -1, -2)
            xir_new_inv = xir_inv - self.update_rate * tmp @ tmp_transpose \
                                / (1 + self.update_rate * cr @ tmp)
        else:
            xir_new_inv = jnp.linalg.solve(
                xir_new, jnp.eye(xir_new.shape[-1], 
                                dtype=xir_new.dtype)
            )

        xirmr = xir @ mr
        tmp = self.update_rate * cr_transpose @ r_minus_b + xirmr
        tmp_transpose = jnp.swapaxes(tmp, -1, -2)
        mr_new = xir_new_inv @ tmp 
        
        if self.full_bayesian:
            omegar_new = omegar + self.update_rate * r_minus_b_tranpose @ r_minus_b + \
                mr_transpose @ xirmr - \
                tmp_transpose @ mr_new
        else:
            omegar_new = omegar
        
        nut_N = N * self.state_dim if self.noise_mode == "scalar" else N
        nut_new = nut + nut_N if self.full_bayesian else nut
        nur_new = nur + N if self.full_bayesian else nur

        new_task = (mt_new, xit_new, xit_new_inv, omegat_new, nut_new,
                mr_new, xir_new, xir_new_inv, omegar_new, nur_new)
    
            
        return new_task

    
    @jax.jit
    def task_posterior(self, task: TaskTuple,
                        obs: Observation, action: Action,
                        reward: Reward, obs_next: Observation,
                        t_mix_params: FrozenDict,
                        t_bias_params: FrozenDict,
                        r_mix_params: FrozenDict,
                        r_bias_params: FrozenDict,
                        s_feat_params: FrozenDict | None = None,
                        a_feat_params: FrozenDict | None = None,
    ) -> tuple[ObsFeat, ActFeat, ObsFeat, TaskTFeat, TaskRFeat, TaskTBias, TaskRBias, TaskTuple]:

        s_feat, a_feat, s_next_feat, ct, cr, bt, br, s_minus_b, r_minus_b = self.gather_feats(
            obs,
            action,
            reward,
            obs_next,
            t_mix_params,
            t_bias_params,
            r_mix_params,
            r_bias_params,
            s_feat_params,
            a_feat_params,
        )

        new_task = self.task_posterior_inner(task, ct, cr, s_minus_b, r_minus_b)
        return s_feat, a_feat, s_next_feat, ct, cr, bt, br, jax.device_get(new_task)

    @jax.jit
    def _update_inner_model(self, rollout: RolloutWithTask, task: TaskTuple) -> tuple[Self, TaskTuple, LogDict]:

        def model_loss(s_feat_params: FrozenDict,
                       a_feat_params: FrozenDict,
                       t_mix_params: FrozenDict,
                       t_bias_params: FrozenDict,
                       r_mix_params: FrozenDict,
                       r_bias_params: FrozenDict) -> tuple[Float[Array, ""], tuple[TaskTuple, LogDict]]:
            
            obs = rollout.observations
            action = rollout.actions
            reward = rollout.rewards
            next_obs = rollout.next_observations

            s_dim = obs.shape[-1]
            N = obs.shape[0]

            s_feat, a_feat, s_next_feat, ct, cr, bt, br, task_new = self.task_posterior(task, obs,
                                           action, reward,
                                           next_obs,
                                           t_mix_params,
                                           t_bias_params,
                                           r_mix_params,
                                           r_bias_params,
                                           s_feat_params,
                                           a_feat_params)
            
            mt_new, xit_new, _, omegat_new, nut_new, \
                mr_new, xir_new, _, omegar_new, nur_new = task_new
            
            t_loss = s_dim * jnp.linalg.slogdet(xit_new)[1] 
            r_loss = jnp.linalg.slogdet(xir_new)[1] 

            if self.full_bayesian:
                if self.noise_mode == "full":
                    t_noise_term = jnp.linalg.slogdet(0.5 * omegat_new)[1]
                else:
                    t_noise_term = jnp.log(0.5 * omegat_new).sum(axis=(-1, -2))
                t_loss = t_loss + nut_new[..., 0, 0] * t_noise_term
                r_loss = r_loss + nur_new[..., 0, 0] * jnp.linalg.slogdet(0.5 * omegar_new)[1]

            else:
                t_loss = t_loss - jnp.trace(
                    omegat_new @ mt_new.swapaxes(-1, -2) @ xit_new @ mt_new,
                axis1=-2, axis2=-1)

                r_loss = r_loss - jnp.trace(
                    omegar_new @ mr_new.swapaxes(-1, -2) @ xir_new @ mr_new,
                axis1=-2, axis2=-1)

          
            t_loss = t_loss.mean() * 0.5 / N
            r_loss = r_loss.mean() * 0.5 / N

            ct_norm = jnp.square(jnp.linalg.norm(ct, axis=-1) / self.transition_latent_dim ** 0.5).mean()
            bt_norm = jnp.square(jnp.linalg.norm(bt, axis=-1) / self.state_dim ** 0.5).mean()

            cr_norm = jnp.square(jnp.linalg.norm(cr, axis=-1) / self.reward_latent_dim ** 0.5).mean()
            br_norm = jnp.square(jnp.linalg.norm(br, axis=-1)).mean()

            
            loss = t_loss + r_loss + \
                   self.t_reg * (ct_norm + bt_norm) + \
                   self.r_reg * (cr_norm + br_norm)

            return loss, (task_new, {"losses/t_loss": t_loss, "losses/r_loss": r_loss, 
                          "losses/ct_norm": ct_norm, "losses/cr_norm": cr_norm,
                          "losses/bt_norm": bt_norm, "losses/br_norm": br_norm,
                          })

        (_, (task_new, model_logs)), model_grads = \
        jax.value_and_grad(model_loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True)(
            self.s_feat_model.params,
            self.a_feat_model.params,
            self.t_mix_model.params,
            self.t_bias_model.params,
            self.r_mix_model.params,
            self.r_bias_model.params,
        )

        s_feat_model = self.s_feat_model.apply_gradients(grads=model_grads[0])
        a_feat_model = self.a_feat_model.apply_gradients(grads=model_grads[1])
        t_mix_model = self.t_mix_model.apply_gradients(grads=model_grads[2])
        t_bias_model = self.t_bias_model.apply_gradients(grads=model_grads[3])
        r_mix_model = self.r_mix_model.apply_gradients(grads=model_grads[4])
        r_bias_model = self.r_bias_model.apply_gradients(grads=model_grads[5])


        return (
            task_new,
            self.replace(
                s_feat_model=s_feat_model,
                a_feat_model=a_feat_model,
                t_mix_model=t_mix_model,
                r_mix_model=r_mix_model,
                t_bias_model=t_bias_model,
                r_bias_model=r_bias_model,
            ),
            model_logs 
        )
    
    @jax.jit
    def  _update_inner_policy(self, data: RolloutWithTask):
        def policy_loss(policy_params: FrozenDict, task_encoder_params: FrozenDict | None):
            action_dist: distrax.Distribution
            new_log_probs: Float[Array, " *batch"]
            assert data.advantages is not None
            assert data.log_probs is not None

            if self.use_task_encoder:
                obs_task = jnp.concatenate(
                    [data.observations, self.encode_task(data.task, task_encoder_params)],
                    axis=-1,
                )
            else:
                obs_task = data.obs_task

            action_dist = self.policy.apply_fn(policy_params, obs_task)

            new_log_probs = action_dist.log_prob(data.actions)  # pyright: ignore[reportAssignmentType]
            log_ratio = new_log_probs.reshape(data.log_probs.shape) - data.log_probs
            ratio = jnp.exp(log_ratio)
            # For logs
            approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
            clip_fracs = jax.lax.stop_gradient(
                (jnp.abs(ratio - 1.0) > self.clip_eps).mean()
            )

            pg_loss1 = -data.advantages * ratio  # pyright: ignore[reportOptionalOperand]
            pg_loss2 = -data.advantages * jnp.clip(  # pyright: ignore[reportOptionalOperand]
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            entropy_loss = action_dist.entropy().mean()

            return pg_loss - self.entropy_coefficient * entropy_loss, {
                "losses/entropy_loss": entropy_loss,
                "losses/policy_loss": pg_loss,
                "losses/approx_kl": approx_kl,
                "losses/clip_fracs": clip_fracs,
            }
       
        (_, policy_logs), grads = jax.value_and_grad(
            policy_loss, argnums=(0, 1), has_aux=True
        )(
            self.policy.params,
            self.task_encoder_model.params if self.use_task_encoder and self.task_encoder_model is not None else None,
        )

        policy = self.policy.apply_gradients(grads=grads[0])
        if self.use_task_encoder and self.task_encoder_model is not None:
            task_encoder_model = self.task_encoder_model.apply_gradients(grads=grads[1])
        else:
            task_encoder_model = self.task_encoder_model
        # Compute features
        return (self.replace(
                    policy=policy,
                    task_encoder_model=task_encoder_model,
                ), policy_logs
                )



    @override
    def update(self, data: RolloutWithTask) -> tuple[Self, LogDict]:
        if self.use_task_encoder:
            # The task representation has to be recomputed inside the policy loss for
            # the encoder to get gradients, so only re-lay the raw task features from
            # (task, timestep, dim) into the (timestep, task, dim) layout the
            # minibatch iterator slices along.
            data = data._replace(task=np.asarray(data.task).swapaxes(0, 1))
        else:
            data = data._replace(
                obs_task=np.array(
                    self.concat_task_with_observation(
                        data.task,
                        data.observations,
                    ).swapaxes(0, 1)
                )
            )
        # print(data.obs_task[0, -1, 39:])
        data = self.compute_advantages(data)

        update_logs = defaultdict(list)

        minibatch_iterator = to_deterministic_minibatch_iterator_with_task(
            data, self.num_gradient_steps
        )
        

        for epoch in range(self.num_epochs):
            kl_losses = []
            policy_losses = []

            for step in range(self.num_gradient_steps):
                minibatch_rollout = next(minibatch_iterator)
                self, policy_logs = self._update_inner_policy(minibatch_rollout)
                kl_losses.append(policy_logs["losses/approx_kl"])
                policy_losses.append(policy_logs["losses/policy_loss"])
                        
            if epoch == 0:  # Initial KL and Loss
                update_logs["metrics/kl_before"] = [np.mean(kl_losses)]
                update_logs["metrics/policy_loss_before"] = [np.mean(policy_losses)]

            if epoch == self.num_epochs - 1:
                update_logs["metrics/kl_after"] = [np.mean(kl_losses)]
                update_logs["metrics/policy_loss_after"] = [np.mean(policy_losses)]

        ct_norms = []
        cr_norms = []

        for _ in range(1):
            task = self.get_init_task()
            for step in range(self.num_gradient_steps):
                minibatch_rollout = next(minibatch_iterator)
                task, self, model_logs = self._update_inner_model(minibatch_rollout, task)
                ct_norms.append(model_logs["losses/ct_norm"])
                cr_norms.append(model_logs["losses/cr_norm"])

        model_logs["metrics/ct_norm"] = np.mean(ct_norms)
        model_logs["metrics/cr_norm"] = np.mean(cr_norms)

        logs = model_logs | policy_logs
        for k, v in logs.items():
            update_logs[k].append(v)

        return self, update_logs
