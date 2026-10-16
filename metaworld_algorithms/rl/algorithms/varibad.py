import dataclasses
from typing import Self
try:
    from typing import override
except ImportError:
    def override(method):
        return method

from collections import defaultdict
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import flax.linen as nn

from flax import struct
from flax.core import FrozenDict
from optax import identity
from jaxtyping import Array, Float, PRNGKeyArray

from metaworld_algorithms.config.envs import MetaLearningEnvConfig
from metaworld_algorithms.config.networks import (ContinuousActionPolicyConfig, 
                                                  VanillaNetworkConfig,
                                                  RecurrentNeuralNetworkConfig)

from metaworld_algorithms.config.utils import Activation, Initializer
from metaworld_algorithms.config.rl import AlgorithmConfig
from metaworld_algorithms.rl.algorithms.base import VAEBasedMetaLearningAlgorithm  
from metaworld_algorithms.nn.distributions import TanhMultivariateNormalDiag
from metaworld_algorithms.rl.algorithms.utils import TrainState

from metaworld_algorithms.config.optim import OptimizerConfig

from metaworld_algorithms.rl.networks import (
    ContinuousActionPolicy,
    RNNEncoder,
    VAETransitionDecoder,
    VAERewardDecoder
)

from metaworld_algorithms.types import (
    Observation,
    Action,
    Reward,
    AuxPolicyOutputs,
    LogDict,
    LogProb,
    MetaLearningAgent,
    Task,
    TaskWithObservation,
    RolloutWithTask,
)

from .utils import (
    LinearFeatureBaseline,
    compute_gae,
    to_deterministic_minibatch_iterator_with_task,
    normalize_advantages,
)

from functools import partial



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
class VariBadConfig(AlgorithmConfig):
    policy_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    s_rnn_fe_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=()
    )
    a_rnn_fe_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=()
    )
    r_rnn_fe_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=()
    )

    s_t_decoder_fe_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=()
    )
    a_t_decoder_fe_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=()
    )
    t_out_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=(128, 64, 32)
    )

    
    s_r_decoder_fe_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=()
    )
    a_r_decoder_fe_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=(128, 64, 32)
    )
    r_out_config: VanillaNetworkConfig = VanillaNetworkConfig(
        width=(128, 64, 32)
    )
    
    rnn_config: RecurrentNeuralNetworkConfig = RecurrentNeuralNetworkConfig(
        width=128,
        activation=Activation.Tanh,
        recurrent_kernel_init=Initializer.ORTHOGONAL,
        kernel_init=Initializer.XAVIER_UNIFORM,
        bias_init=Initializer.ZEROS,
        optimizer=OptimizerConfig(lr=5e-4, max_grad_norm=1.0),
    )

    s_hidden_dim: int = 32
    a_hidden_dim: int = 16
    r_hidden_dim: int = 16

    task_dim: int = 5
    subsample_elbo: int = 50
    subsample_decode: int = 50

    meta_batch_size: int = 20
    clip_eps: float = 0.1
    entropy_coefficient: float = 5e-3
    normalize_advantages: bool = False
    gae_lambda: float = 0.95
    num_epochs: int = 2
    num_gradient_steps: int = 8
    n_vae_updates: int = 3

    dtype: npt.DTypeLike = np.float32

class VariBad(VAEBasedMetaLearningAlgorithm[VariBadConfig]):
    policy: TrainState

    vae_rnn_encoder: TrainState
    vae_s_decoder: TrainState
    vae_r_decoder: TrainState

    vae_f_mu: TrainState
    vae_f_logvar: TrainState


    key: PRNGKeyArray
    meta_batch_size: int = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    normalize_advantages: bool = struct.field(pytree_node=False)
    gae_lambda: float = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    num_gradient_steps: int = struct.field(pytree_node=False)
    subsample_elbo: int = struct.field(pytree_node=False)
    subsample_decode: int = struct.field(pytree_node=False)
    dtype: npt.DTypeLike = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    s_hidden_dim: int = struct.field(pytree_node=False)
    a_hidden_dim: int = struct.field(pytree_node=False)
    r_hidden_dim: int = struct.field(pytree_node=False)
    rnn_hidden_dim: int = struct.field(pytree_node=False)
    task_dim: int = struct.field(pytree_node=False)

    n_vae_updates: int = 3

    running_task_mean: Task | None = None
    running_task_var: Task | None = None
    n_updates: int = 0

    @override
    @staticmethod
    def initialize(
        config: VariBadConfig,
        env_config: MetaLearningEnvConfig,
        seed = 1
    ) -> "VariBad":

        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, policy_key, rnn_key, \
            tkey, rkey, mu_key, logvar_key  = jax.random.split(master_key, 7)


        s_dim = int(np.prod(env_config.observation_space.shape))
        a_dim = int(np.prod(env_config.action_space.shape))

        policy_net = ContinuousActionPolicy(
            config=config.policy_config,
            action_dim=a_dim,
        )

        f_mu_net = nn.Dense(
            config.task_dim,
        )

        f_logvar_net = nn.Dense(
            config.task_dim
        )

        rnn_encoder_net = RNNEncoder(
            s_rnn_fe_config=config.s_rnn_fe_config,
            a_rnn_fe_config=config.a_rnn_fe_config,
            r_rnn_fe_config=config.r_rnn_fe_config,
            rnn_config=config.rnn_config,
            s_hidden_dim=config.s_hidden_dim,
            a_hidden_dim=config.a_hidden_dim,
            r_hidden_dim=config.r_hidden_dim,
            f_mu=f_mu_net,
            f_logvar=f_logvar_net
        )

        

        t_decoder_net = VAETransitionDecoder(
            s_t_decoder_fe_config=config.s_t_decoder_fe_config,
            a_t_decoder_fe_config=config.a_t_decoder_fe_config,
            t_out_config=config.t_out_config,
            s_dim=s_dim,
            s_hidden_dim=config.s_hidden_dim,
            a_hidden_dim=config.a_hidden_dim
        )
        r_decoder_net = VAERewardDecoder(
            s_r_decoder_fe_config=config.s_r_decoder_fe_config,
            a_r_decoder_fe_config=config.a_r_decoder_fe_config,
            r_out_config=config.r_out_config,
            s_hidden_dim=config.s_hidden_dim,
            a_hidden_dim=config.a_hidden_dim
        )

        dummy_obs = jnp.array(
            [
                env_config.observation_space.sample()
                for _ in range(config.meta_batch_size)
            ], dtype=config.dtype
        )

        # Assuming diagonal covariance
        dummy_belief = jnp.zeros(
            (config.meta_batch_size, 2 * config.task_dim), dtype=config.dtype
        )

        dummy_task = jnp.zeros(
            (config.meta_batch_size, config.task_dim), dtype=config.dtype
        )

        dummy_input = jnp.concatenate(
            (dummy_obs, dummy_belief), axis=-1
        )

        dummy_action = jnp.zeros(
            (config.meta_batch_size, a_dim), dtype=config.dtype
        )

        dummy_reward = jnp.zeros(
            (config.meta_batch_size, 1), dtype=config.dtype
        )


        init_carry = jnp.zeros(
            (1, config.num_tasks, config.rnn_config.width)
        )   
        

        policy = TrainState.create(
            params=policy_net.init(policy_key, dummy_input),
            tx=config.policy_config.network_config.optimizer.spawn(),
            apply_fn=policy_net.apply,
        )


        rnn_encoder = TrainState.create(
            params=rnn_encoder_net.init(
                rnn_key,
                dummy_obs, dummy_action, dummy_reward,
                init_carry
            ),
            tx=config.rnn_config.optimizer.spawn(),
            apply_fn=rnn_encoder_net.apply,
        )

        t_decoder = TrainState.create(
            params=t_decoder_net.init(tkey, dummy_task, 
                                      dummy_obs, dummy_action),
            tx=config.s_t_decoder_fe_config.optimizer.spawn(),
            apply_fn=t_decoder_net.apply,
        )

        r_decoder = TrainState.create(
            params=r_decoder_net.init(rkey, dummy_task, 
                                      dummy_obs, dummy_action,
                                      dummy_obs),
            tx=config.s_r_decoder_fe_config.optimizer.spawn(),
            apply_fn=r_decoder_net.apply,
        )

        vae_f_mu = TrainState.create(
            params=f_mu_net.init(mu_key, init_carry),
            tx=identity(),
            apply_fn=f_mu_net.apply
        )

        vae_f_logvar = TrainState.create(
            params=f_logvar_net.init(logvar_key, init_carry),
            tx=identity(),
            apply_fn=f_logvar_net.apply
        )


        return VariBad(
            policy=policy,
            vae_rnn_encoder=rnn_encoder,
            vae_s_decoder=t_decoder,
            vae_r_decoder=r_decoder,
            vae_f_mu=vae_f_mu,
            vae_f_logvar=vae_f_logvar,
            action_dim=a_dim,
            s_hidden_dim=config.s_hidden_dim,
            a_hidden_dim=config.a_hidden_dim,
            r_hidden_dim=config.r_hidden_dim,
            task_dim=config.task_dim,
            meta_batch_size=config.meta_batch_size,
            clip_eps=config.clip_eps,
            entropy_coefficient=config.entropy_coefficient,
            normalize_advantages=config.normalize_advantages,
            gae_lambda=config.gae_lambda,
            num_epochs=config.num_epochs,
            num_gradient_steps=config.num_gradient_steps,
            dtype=config.dtype,
            subsample_elbo=config.subsample_elbo,
            subsample_decode=config.subsample_decode,
            n_vae_updates=config.n_vae_updates,
            key=algorithm_key,
            num_tasks=config.num_tasks,
            gamma=config.gamma,
            rnn_hidden_dim=config.rnn_config.width
        )
    

    @override
    def get_num_params(self):
        s_decoder_params = sum(
            x.size for x in jax.tree.leaves(self.vae_s_decoder.params)
        )
        r_decoder_params = sum(
            x.size for x in jax.tree.leaves(self.vae_r_decoder.params)
        )
        rnn_params = sum(
            x.size for x in jax.tree.leaves(self.vae_rnn_encoder.params)
        )
        policy_params = sum(
            x.size for x in jax.tree.leaves(self.policy.params)
        )

        return {
            "policy_num_params": policy_params,
            "vae_params": s_decoder_params + r_decoder_params + rnn_params
        }
    
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
        return jax.device_get(_eval_action(self.policy, observations))
    
    @jax.jit
    def concat_mean_logvar(
        self, mean: jax.Array, logvar: jax.Array
    ) -> Task:
        return jnp.concat([mean, logvar], axis=-1).swapaxes(0, 1)

    @jax.jit
    def concat_task_with_observation(
        self, task: Task,
        observation: Observation
    ) -> TaskWithObservation:
        
        if len(observation.shape) == 2:
            obs = jnp.expand_dims(observation, 1)
        elif len(observation.shape) == 3:
            obs = observation.swapaxes(0, 1)
        else:
            obs = observation.squeeze(1)

        if self.running_task_mean is not None:
            norm_task = (task - self.running_task_mean) \
                / jnp.sqrt(self.running_task_var + 1e-6)
        else:
            norm_task = task

        # print(norm_task[0])
        return jnp.concatenate([obs, norm_task], axis=-1).squeeze()

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
    

    class VariBadWrapped(MetaLearningAgent):
        carry: jax.Array
        mean: jax.Array
        logvar: jax.Array

        obs: Observation | None
        action: Action | None
        reward: Reward | None

        def __init__(self, agent: "VariBad"):
            self._agent = agent

        def _update_task(self, new_obs: Observation) -> TaskWithObservation:
            self.obs = new_obs

            if len(self._posterior_fixed_idx) < self._agent.meta_batch_size:

                new_mean, new_logvar, new_carry = self._agent.encode_step(
                    self._agent.vae_rnn_encoder.params, new_obs, self.action, self.reward, self.carry
                )

                if len(self._posterior_fixed_idx) != 0:
                    new_mean = new_mean.at[:, self._posterior_fixed_idx].set(
                            self.mean[:, self._posterior_fixed_idx].copy()
                    )
                    new_logvar = new_logvar.at[:, self._posterior_fixed_idx].set(
                            self.logvar[:, self._posterior_fixed_idx].copy()
                    )
                    new_carry = new_carry.at[:, self._posterior_fixed_idx].set(
                            self.carry[:, self._posterior_fixed_idx].copy()
                    )

                self.mean = new_mean
                self.logvar = new_logvar
                self.carry = new_carry
            
            self._posterior_fixed_idx = np.empty(0)

            task = self._agent.concat_mean_logvar(self.mean, self.logvar)
            return self._agent.concat_task_with_observation(task, self.obs)

        def adapt_action(self, observations):
            obs_task = self._update_task(observations)
            self._agent, action, aux_policy_outs = (
                self._agent.sample_action_and_aux(obs_task)
            )
            return action, aux_policy_outs


        def init(self):
            self.obs = None
            self.action = None
            self.reward = None

            init_mean, init_logvar, init_carry = \
                self._agent.get_prior_mean_logvar_meta()
            
            self.carry = init_carry

            self.mean = init_mean
            self.logvar = init_logvar


            self._posterior_fixed_idx = np.arange(self._agent.meta_batch_size)

            self.adapt()


        def predictive_losses(self, obs_next_gt: Observation, r_gt: Reward):
            return np.zeros((obs_next_gt.shape[0], )), np.zeros((r_gt.shape[0], ))


        def step(self, timestep):
            self.action = timestep.action
            self.reward = timestep.reward

        def adapt(self):
            self.adapt_carry = self.carry.copy()
            self.adapt_mean = self.mean.copy()
            self.adapt_logvar = self.logvar.copy()

        def reset(self, env_mask):
            self._posterior_fixed_idx = np.argwhere(env_mask)
            if len(self._posterior_fixed_idx) > 0:
                self.mean = self.mean.at[:, self._posterior_fixed_idx].set(
                    self.adapt_mean[:, self._posterior_fixed_idx]
                )
                self.logvar = self.logvar.at[:, self._posterior_fixed_idx].set(
                    self.adapt_logvar[:, self._posterior_fixed_idx]
                )
                self.carry = self.carry.at[:, self._posterior_fixed_idx].set(
                    self.adapt_carry[:, self._posterior_fixed_idx]
                )
        
        def eval_action(self, observations):
            obs_task = self._update_task(observations)
            action = self._agent.eval_action(obs_task)
            return action
        
    @override
    def wrap(self):
        return VariBad.VariBadWrapped(self)
    
    @jax.jit
    def get_prior_mean_logvar_meta(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        zeros = jnp.zeros(
            (1, self.meta_batch_size, self.rnn_hidden_dim)
        )   
        prior_mean = self.vae_f_mu.apply_fn(self.vae_f_mu.params, zeros)
        prior_logvar = self.vae_f_logvar.apply_fn(self.vae_f_logvar.params, zeros)
        return prior_mean, prior_logvar, zeros
    
    @partial(jax.jit, static_argnames=('batch_size', ))
    def get_prior_mean_logvar_vae(self, batch_size: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        zeros = jnp.zeros(
            (1, batch_size, self.rnn_hidden_dim)
        )   
        prior_mean = self.vae_f_mu.apply_fn(self.vae_f_mu.params, zeros)
        prior_logvar = self.vae_f_logvar.apply_fn(self.vae_f_logvar.params, zeros)
        return prior_mean, prior_logvar, zeros
    
    @jax.jit
    def encode_step(self, rnn_params: FrozenDict, obs_next: Observation,
                actions: Action, rewards: Reward, carry: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        mean, logvar, carry_out = self.vae_rnn_encoder.apply_fn(
            rnn_params, 
            obs_next,
            actions,
            rewards,
            carry
        )
        return mean, logvar, carry_out

    @jax.jit
    def encode(self, rnn_params: FrozenDict, obs_next: Observation,
                actions: Action, rewards: Reward) -> tuple[jax.Array, jax.Array, jax.Array]:
        prior_mean, prior_logvar, zeros = self.get_prior_mean_logvar_vae(obs_next.shape[-2])
        mean, logvar, carry = self.encode_step(rnn_params, obs_next, actions, rewards, zeros)
        return jnp.concat((prior_mean, mean)), jnp.concat((prior_logvar, logvar)), carry
    
    @jax.jit
    def encode_init(self, rnn_params: FrozenDict, obs_next: Observation,
                actions: Action, rewards: Reward) -> tuple[jax.Array, jax.Array, jax.Array]:
        zeros = jnp.zeros((1, self.num_tasks, self.rnn_hidden_dim))   
        mean, logvar, carry = self.encode_step(rnn_params, obs_next, actions, rewards, zeros)
        return mean[-1:], logvar[-1:], carry


    @jax.jit
    def update_vae(self, obs: Observation,
                   actions: Action,
                   obs_next: Observation,
                   rewards: Reward ) -> tuple[Self, LogDict]:
        
        key, sample_key, elbo_key, decoder_key = jax.random.split(self.key, 4)

        def vae_loss(
            rnn_params: FrozenDict,
            t_decoder_params: FrozenDict,
            r_decoder_params: FrozenDict,
        ) -> tuple[Float[Array, "1"], LogDict]:
            
            mean, logvar, _ = self.encode(rnn_params, obs_next, actions, rewards)
            std = jnp.exp(0.5 * logvar)
            samples = jax.random.normal(
                sample_key, std.shape, self.dtype
            ) * std + mean

            batch_size = samples.shape[1]

            elbo_indices = jax.random.choice(
                elbo_key,
                jnp.arange(self.subsample_elbo),
                shape=(self.subsample_elbo * batch_size, )
            )
            task_indices = jnp.arange(batch_size) \
                                .repeat(self.subsample_elbo)

            samples = samples[elbo_indices, task_indices, :]
            samples = samples.reshape(self.subsample_elbo, batch_size, -1)

            dec_obs = obs[None, :].repeat(self.subsample_elbo, axis=0)
            dec_obs_next = obs_next[None, :].repeat(self.subsample_elbo, axis=0)
            dec_actions = actions[None, :].repeat(self.subsample_elbo, axis=0)
            dec_rewards = rewards[None, :].repeat(self.subsample_elbo, axis=0)

            indices0 = jnp.arange(self.subsample_elbo). \
                           repeat(self.subsample_decode * batch_size)
            indices1 = jax.random.choice(
                decoder_key,
                jnp.arange(self.subsample_decode),
                shape=(self.subsample_elbo * self.subsample_decode 
                       * batch_size, )
            )

            indices2 = jnp.arange(batch_size).repeat(
                self.subsample_elbo * self.subsample_decode
            )   

            dec_obs = dec_obs[indices0, indices1, indices2, :].reshape(
                self.subsample_elbo, self.subsample_decode,
                batch_size, -1
            )
            dec_actions = dec_actions[indices0, indices1, indices2, :].reshape(
                self.subsample_elbo, self.subsample_decode,
                batch_size, -1
            )
            dec_obs_next = dec_obs_next[indices0, indices1, indices2, :].reshape(
                self.subsample_elbo, self.subsample_decode,
                batch_size, -1
            )
            dec_rewards = dec_rewards[indices0, indices1, indices2, :].reshape(
                self.subsample_elbo, self.subsample_decode,
                batch_size, -1
            )

            dec = samples[None, :].repeat(
                self.subsample_decode, axis=0
            ).swapaxes(0, 1)

            reward_pred = self.vae_r_decoder.apply_fn(
                r_decoder_params, dec, dec_obs, dec_actions, dec_obs_next
            ) 

            reward_loss = (reward_pred - dec_rewards) ** 2
            reward_loss = jnp.mean(reward_loss, axis=-1).sum(axis=0)

            next_state_pred = self.vae_s_decoder.apply_fn(
                t_decoder_params, dec, dec_obs, dec_actions
            )

            state_loss = (next_state_pred - dec_obs_next) ** 2
            state_loss = jnp.mean(state_loss, axis=-1).sum(axis=0)

            all_means = jnp.concatenate(
                [jnp.zeros((1, *mean.shape[1:])),
                 mean]
            )
            all_logvars = jnp.concatenate(
                [jnp.zeros((1, *logvar.shape[1:])),
                 logvar]
            )

            mu = all_means[1:]
            m = all_means[:-1]

            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_div = 0.5 * (
                        jnp.sum(logS, axis=-1) - jnp.sum(logE, axis=-1) 
                        - self.task_dim
                        + jnp.sum(1 / jnp.exp(logS) * jnp.exp(logE), axis=-1)
                        + ((m - mu) / jnp.exp(logS) * (m - mu)).sum(axis=-1)
                    )
            
            kl_div = kl_div[elbo_indices, task_indices].reshape(
                self.subsample_elbo, samples.shape[1]
            ).sum(axis=[0, 1])

            loss = (state_loss + reward_loss + kl_div).mean()

            return loss, {
                "vae_loss": loss,
                "reward_loss": reward_loss.mean(),
                "state_loss": state_loss.mean(),
                "kl_div": kl_div.mean(),
            }
        
        (_, log), grads = jax.value_and_grad(vae_loss, argnums=(0, 1, 2), has_aux=True)(
            self.vae_rnn_encoder.params,
            self.vae_s_decoder.params,
            self.vae_r_decoder.params
        )

        vae_rnn_encoder = self.vae_rnn_encoder.apply_gradients(
            grads=grads[0]
        )
        vae_s_decoder = self.vae_s_decoder.apply_gradients(
            grads=grads[1]
        )
        vae_r_decoder = self.vae_r_decoder.apply_gradients(
            grads=grads[2]
        )

        return self.replace(
            key=key,
            vae_rnn_encoder=vae_rnn_encoder,
            vae_s_decoder=vae_s_decoder,
            vae_r_decoder=vae_r_decoder
        ), log

    @jax.jit
    def update_policy(self, rollouts: RolloutWithTask) -> tuple[Self, LogDict]:
        assert rollouts.advantages is not None
        assert rollouts.obs_task is not None

        def policy_loss(policy_params: FrozenDict):
            action_dist = self.policy.apply_fn(policy_params, rollouts.obs_task)
            new_log_probs = action_dist.log_prob(rollouts.actions)  # pyright: ignore[reportAssignmentType]
            log_ratio = new_log_probs.reshape(rollouts.log_probs.shape) - rollouts.log_probs
            ratio = jnp.exp(log_ratio)

            # For logs
            approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
            clip_fracs = jax.lax.stop_gradient(
                (jnp.abs(ratio - 1.0) > self.clip_eps).mean()
            )

            pg_loss1 = -rollouts.advantages * ratio  # pyright: ignore[reportOptionalOperand]
            pg_loss2 = -rollouts.advantages * jnp.clip(  # pyright: ignore[reportOptionalOperand]
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
       
        (_, policy_logs), policy_grads = jax.value_and_grad(policy_loss, argnums=0, has_aux=True)(
            self.policy.params
        )
        
        policy = self.policy.apply_gradients(grads=policy_grads)
        # Compute features
        return (self.replace(
                    policy=policy
                ), policy_logs
                )

    @override
    def update(self, 
               data: RolloutWithTask,
               vae_states: Observation,
               vae_actions: Action,
               vae_next_states: Observation,
               vae_rewards: Reward
            ) -> tuple[Self, LogDict]:
        data = data._replace(
            obs_task=self.concat_task_with_observation(
                data.task, data.observations
            ).swapaxes(0, 1)
        )
        data = self.compute_advantages(data)

        update_logs = defaultdict(list)

        new_n_updates = self.n_updates + data.observations.shape[0] * data.observations.shape[1]

        update_logs["metrics/task_mean"] = jnp.mean(data.task, axis=[0, 1])
        update_logs["metrics/task_var"] = jnp.var(data.task, axis=[0, 1])
        

        if self.running_task_mean is None:
            self = self.replace(
                running_task_mean=jnp.mean(data.task, axis=[0, 1]),
                running_task_var=jnp.var(data.task, axis=[0, 1]),
                n_updates=new_n_updates
            )

        else:
            running_task_mean_new = (self.running_task_mean * self.n_updates 
                                        + jnp.sum(data.task, axis=[0, 1]))  / new_n_updates
            running_task_var_new = (self.running_task_var * self.n_updates
                                    + jnp.var(data.task, axis=[0, 1]) * new_n_updates
                                    + (running_task_mean_new - self.running_task_mean) ** 2 
                                    * self.n_updates * data.observations.shape[0] * data.observations.shape[1] / new_n_updates) / new_n_updates
            self = self.replace(
                running_task_mean=running_task_mean_new,
                running_task_var=running_task_var_new,
                n_updates=new_n_updates
            )

        minibatch_iterator = to_deterministic_minibatch_iterator_with_task(data, self.num_gradient_steps)
        
        if self.running_task_mean is not None:
            norm_task = (data.task - self.running_task_mean) / jnp.sqrt(self.running_task_var + 1e-6)
            update_logs["metrics/normed_task"] = norm_task.max(), norm_task.min(), norm_task.mean()

        for epoch in range(self.num_epochs):
            for step in range(self.num_gradient_steps):
                minibatch_rollout = next(minibatch_iterator)
                self, logs = self.update_policy(minibatch_rollout)
                if epoch == 0 and step == 0:  # Initial KL and Loss
                    update_logs["metrics/kl_before"] = [logs["losses/approx_kl"]]
                    update_logs["metrics/policy_loss_before"] = [
                        logs["losses/policy_loss"]
                    ]

        for _ in range(self.n_vae_updates):
            self, vae_log = self.update_vae(vae_states, vae_actions, vae_next_states, vae_rewards)

        logs = logs | vae_log
        for k, v in logs.items():
            update_logs[k].append(v)

        return self, update_logs
