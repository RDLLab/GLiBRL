"""This module defines the base abstract classes for reinforcement learning algorithms.

It provides a hierarchy of classes for different types of algorithms, including:
- `Algorithm`: A generic base class for any RL algorithm.
- `OnPolicyAlgorithm`: For on-policy algorithms like PPO.
- `OffPolicyAlgorithm`: For off-policy algorithms like SAC.
- `MetaLearningAlgorithm`: A base class for meta-learning algorithms.
- `GradientBasedMetaLearningAlgorithm`: For gradient-based meta-RL (e.g., MAML).
- `RNNBasedMetaLearningAlgorithm`: For recurrent-based meta-RL (e.g., RL^2).
- `VAEBasedMetaLearningAlgorithm`: For VAE-based meta-RL (e.g., PEARL).
- `BRLMetaLearningAlgorithm`: For Bayesian meta-RL.
"""
import abc
import time
from collections import deque
from typing import Deque, Generic, Self, TypeVar
try:
    from typing import override
except ImportError:
    def override(method):
        return method

import numpy as np
import numpy.typing as npt
import orbax.checkpoint as ocp
from flax import struct
from flax.core import FrozenDict
from jaxtyping import Float, Array

import jax

import jax.numpy as jnp


from metaworld_algorithms.checkpoint import get_checkpoint_save_args
from metaworld_algorithms.config.envs import EnvConfig, MetaLearningEnvConfig
from metaworld_algorithms.config.rl import (
    AlgorithmConfig,
    GradientBasedMetaLearningTrainingConfig,
    MetaLearningTrainingConfig,
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
    RNNBasedMetaLearningTrainingConfig,
    BRLMetaLearningTrainingConfig,
    VAEBasedMetaLearningTrainingConfig,
    TrainingConfig,
)

from metaworld_algorithms.monitoring.utils import log
from metaworld_algorithms.rl.buffers import (
    AbstractReplayBuffer,
    MultiTaskRolloutBuffer,
    ModelLearningBuffer,
    MultiTaskRolloutBufferWithTask
)
from metaworld_algorithms.types import (
    Action,
    Task,
    TaskTuple,
    Reward,
    TaskWithObservation,
    AuxPolicyOutputs,
    CheckpointMetadata,
    GymVectorEnv,
    LogDict,
    MetaLearningAgent,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    RNNState,
    Rollout,
    RolloutWithTask
)

from metaworld_algorithms.rl.algorithms.utils import TrainState


AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
EnvConfigType = TypeVar("EnvConfigType", bound=EnvConfig)

MetaLearningTrainingConfigType = TypeVar(
    "MetaLearningTrainingConfigType", bound=MetaLearningTrainingConfig
)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout, list[Rollout])


class Algorithm(
    abc.ABC,
    Generic[AlgorithmConfigType, TrainingConfigType, EnvConfigType, DataType],
    struct.PyTreeNode,
):
    """The base class for all reinforcement learning algorithms.

    This is an abstract class that defines the basic interface for an RL algorithm.
    It is generic over the algorithm, training, and environment configurations, as
    well as the data type used for training.

    Attributes:
        num_tasks: The number of tasks.
        gamma: The discount factor.
    """
    """Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    num_tasks: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)

    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: EnvConfigType, seed: int = 1
    ) -> "Algorithm":
        """Initialize the algorithm.

        Args:
            config: The algorithm configuration.
            env_config: The environment configuration.
            seed: The random seed.

        Returns:
            An initialized instance of the algorithm.
        """
        ...

    @abc.abstractmethod
    def get_num_params(self) -> dict[str, int]:
        """Get the number of parameters in the algorithm's models.

        Returns:
            A dictionary mapping model names to the number of parameters.
        """
        ...

    @abc.abstractmethod
    def train(
        self,
        config: TrainingConfigType,
        envs: GymVectorEnv,
        env_config: EnvConfigType,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        """Train the algorithm.

        Args:
            config: The training configuration.
            envs: The vectorized environment.
            env_config: The environment configuration.
            run_timestamp: The timestamp of the run.
            seed: The random seed.
            track: Whether to track the training with a logger.
            checkpoint_manager: The checkpoint manager.
            checkpoint_metadata: The checkpoint metadata.
            buffer_checkpoint: The replay buffer checkpoint.

        Returns:
            The trained algorithm.
        """
        ...


class MetaLearningAlgorithm(
    Algorithm[
        AlgorithmConfigType,
        MetaLearningTrainingConfigType,
        MetaLearningEnvConfig,
        DataType,
    ],
    Generic[AlgorithmConfigType, MetaLearningTrainingConfigType, DataType],
):
    """The base class for all meta-learning algorithms.

    This class extends the `Algorithm` class with methods specific to meta-learning.
    """
    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: MetaLearningEnvConfig, seed: int = 1
    ) -> "MetaLearningAlgorithm":
        """Initialize the meta-learning algorithm.

        Args:
            config: The algorithm configuration.
            env_config: The meta-learning environment configuration.
            seed: The random seed.

        Returns:
            An initialized instance of the meta-learning algorithm.
        """
        ...

    @abc.abstractmethod
    def update(self, data: DataType) -> tuple[Self, LogDict]:
        """Perform a single update step of the algorithm.

        Args:
            data: The data to use for the update.

        Returns:
            A tuple containing the updated algorithm and a dictionary of logs.
        """
        ...

    @abc.abstractmethod
    def wrap(self) -> MetaLearningAgent:
        """Wrap the algorithm in a `MetaLearningAgent` interface for evaluation.

        Returns:
            A `MetaLearningAgent` instance.
        """
        ...

    @abc.abstractmethod
    def train(
        self,
        config: MetaLearningTrainingConfigType,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        """Train the meta-learning algorithm.

        Args:
            config: The training configuration.
            envs: The vectorized environment.
            env_config: The meta-learning environment configuration.
            run_timestamp: The timestamp of the run.
            seed: The random seed.
            track: Whether to track the training with a logger.
            checkpoint_manager: The checkpoint manager.
            checkpoint_metadata: The checkpoint metadata.
            buffer_checkpoint: The replay buffer checkpoint.

        Returns:
            The trained meta-learning algorithm.
        """
        ...


class BRLMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, BRLMetaLearningTrainingConfig, RolloutWithTask
    ],
    Generic[AlgorithmConfigType],
):
    """Base class for Bayesian Meta-Reinforcement Learning algorithms.

    Attributes:
        transition_latent_dim: The dimension of the transition latent space.
        reward_latent_dim: The dimension of the reward latent space.
        task_dim: The dimension of the task embedding.
        init_task: The initial task embedding.
        use_sa_encoder: Whether to use the state-action feature encoder.
        s_feat_model: The state feature model (None when `use_sa_encoder` is False).
        a_feat_model: The action feature model (None when `use_sa_encoder` is False).
        t_mix_model: The transition mixing model.
        r_mix_model: The reward mixing model.
        t_bias_model: The transition bias model.
        r_bias_model: The reward bias model.
    """

    transition_latent_dim: int = struct.field(pytree_node=False)
    reward_latent_dim: int = struct.field(pytree_node=False)
    task_dim: int = struct.field(pytree_node=False)
    use_sa_encoder: bool = struct.field(pytree_node=False)
    init_task: TaskTuple
    s_feat_model: TrainState | None
    a_feat_model: TrainState | None
    t_mix_model: TrainState
    r_mix_model: TrainState
    t_bias_model: TrainState
    r_bias_model: TrainState

    @abc.abstractmethod
    def wrap(self) -> MetaLearningAgent:
        """Wrap the algorithm in a `MetaLearningAgent` interface for evaluation.

        Returns:
            A `MetaLearningAgent` instance.
        """
        ...

    @abc.abstractmethod
    def get_init_task(self) -> TaskTuple:
        """Get the initial task embedding.

        Returns:
            The initial task embedding.
        """
        ...

    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: TaskWithObservation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        """Sample an action and auxiliary policy outputs.

        Args:
            observation: The observation, including the task embedding.

        Returns:
            A tuple containing the updated algorithm, the sampled action, and auxiliary outputs.
        """
        ...

    @abc.abstractmethod
    def reshape_task(
        self, tasktuple: TaskTuple
    ) -> Task:
        """Reshape the task tuple into a task vector.

        Args:
            tasktuple: The task tuple.

        Returns:
            The task vector.
        """
        ...

    @abc.abstractmethod
    def concat_task_with_observation(
        self, task: Task, observation: Observation
    ) -> TaskWithObservation:
        """Concatenate the task embedding with the observation.

        Args:
            task: The task embedding.
            observation: The observation.

        Returns:
            The observation with the task embedding concatenated.
        """
        ...

    @abc.abstractmethod
    def task_posterior(
        self, task: Task, obs: Observation, action: Action,
        reward: Reward, obs_next: Observation,
        t_mix_params: FrozenDict,
        t_bias_params: FrozenDict,
        r_mix_params: FrozenDict,
        r_bias_params: FrozenDict,
        s_feat_params=None,
        a_feat_params=None,
    ) -> tuple: ...

    def get_mr_raw_dim(self) -> int | None:
        return None

    def spawn_rollout_buffer(
        self,
        task_dim: int,
        env_config: EnvConfig,
        training_config: BRLMetaLearningTrainingConfig,
        seed: int | None = None,
        dtype: npt.DTypeLike = np.float64
    ) -> MultiTaskRolloutBufferWithTask:

        return MultiTaskRolloutBufferWithTask(
            task_dim=task_dim,
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
            dtype=dtype,
            mr_raw_dim=self.get_mr_raw_dim(),
        )

    @override
    def train(
        self,
        config: BRLMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:

        # Initialize tracking deques for returns and lengths
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0
        _episode_return_accum: np.ndarray | None = None
        _episode_length_accum: np.ndarray | None = None
        eval_envs = env_config.spawn_test(seed)
        # Load from checkpoint if available
        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        # Initialize rollout buffer and evaluation environment
        rollout_buffer = self.spawn_rollout_buffer(self.task_dim,
                                                    env_config, config, seed, dtype=np.float64)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
        )

        # Main training loop
        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            envs.call("sample_tasks")

            # --- Sampling phase ---
            obs, _ = envs.reset()
            rollout_buffer.reset()
            episode_started = np.ones((envs.num_envs,))
            _episode_return_accum = np.zeros(envs.num_envs, dtype=np.float64)
            _episode_length_accum = np.zeros(envs.num_envs, dtype=np.int64)

            tasktuple = self.get_init_task()

            while not rollout_buffer.ready:
                task_reshaped = self.reshape_task(tasktuple)
                task_with_obs = self.concat_task_with_observation(task_reshaped, obs)
                self, actions, aux_policy_outs = self.sample_action_and_aux(task_with_obs)
                actions = np.array(actions)

                # Step environment
                next_obs, rewards, terminations, truncations, infos = envs.step(
                    actions,
                )

                rollout_buffer.add(
                    obs,
                    next_obs,
                    task_reshaped,
                    actions,
                    rewards,
                    episode_started,
                    action_idx=None,
                    value=aux_policy_outs.get("value"),
                    log_prob=aux_policy_outs.get("log_prob"),
                    mean=aux_policy_outs.get("mean"),
                    std=aux_policy_outs.get("std"),
                    # mr_raw=mr_raw_step,
                )

                # Update task posterior
                ret = self.task_posterior(tasktuple, obs, actions, rewards, next_obs,
                    self.t_mix_model.params,
                    self.t_bias_model.params,
                    self.r_mix_model.params,
                    self.r_bias_model.params,
                    self.s_feat_model.params if self.use_sa_encoder and self.s_feat_model is not None else None,
                    self.a_feat_model.params if self.use_sa_encoder and self.a_feat_model is not None else None,
                )

                episode_started = np.logical_or(terminations, truncations)

                tasktuple_new = ret[-1]
                #done_mask = episode_started.reshape((-1,) + (1,) * (tasktuple[0].ndim - 1))
                #tasktuple = tuple(
                #    jnp.where(done_mask, old, new)
                #    for old, new in zip(tasktuple, tasktuple_new)
                #)
                tasktuple = tasktuple_new

                _episode_return_accum += np.array(rewards, dtype=np.float64)
                _episode_length_accum += 1
                obs = next_obs

                # Log episodic returns
                for i, env_ended in enumerate(episode_started):
                    if env_ended:
                        logged = False
                        try:
                            global_episodic_return.append(
                                infos["final_info"]["episode"]["r"][i]
                            )
                            global_episodic_length.append(
                                infos["final_info"]["episode"]["l"][i]
                            )
                            logged = True
                        except (KeyError, TypeError):
                            pass
                        if not logged:
                            global_episodic_return.append(_episode_return_accum[i])
                            global_episodic_length.append(_episode_length_accum[i])
                        _episode_return_accum[i] = 0.0
                        _episode_length_accum[i] = 0

            rollouts = rollout_buffer.get()

            if global_episodic_return:
                mean_episodic_return = np.mean(list(global_episodic_return))
                print("- Mean episodic return: ", mean_episodic_return)
                if track:
                    log(
                        {"charts/mean_episodic_returns": mean_episodic_return},
                        step=global_step,
                    )

            # Update models
            print("- Computing update")
            self, logs = self.update(rollouts)

            if track:
                log(
                    {"charts/mean_episodic_returns": mean_episodic_return},
                    step=global_step,
                )
                if "metrics/ct_norm" in logs and "metrics/cr_norm" in logs:
                    log(
                        {"charts/ct_norm": float(logs["metrics/ct_norm"][0]),
                            "charts/cr_norm": float(logs["metrics/cr_norm"][0])},
                        step=global_step
                    )


            # --- Evaluation phase ---

            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                wrapped_agent = self.wrap()
                
                mean_success_rate_dict, mean_returns_dict, mean_success_per_task_dict, \
                        mean_t_loss, mean_r_loss, mean_t_loss_per_task, mean_r_loss_per_task  = (
                        env_config.evaluate_metalearning(eval_envs, wrapped_agent)
                )


                for step in range(env_config.evaluation_adaptation_steps + 1):
                    print(
                        f"Mean evaluation success rate learnt {mean_success_rate_dict[step]:.4f}, return: {mean_returns_dict[step]:.4f}, adaptation step: {step}"
                    )
                    print(
                        "Mean evaluation success rate per task learnt", mean_success_per_task_dict
                    )
                    print(
                        f"Mean t loss learnt {mean_t_loss[step]:.4f}", f"Mean r loss learnt {mean_r_loss[step]:.4f}"
                    )
                    print(
                        f"Mean t loss per task learnt {mean_t_loss_per_task}", f"Mean r loss per task learnt {mean_r_loss_per_task}"
                    )
            
                eval_metrics = {
                    f"charts/mean_success_rate_adapt_{step}": mean_success_rate_dict[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_evaluation_return_adapt_{step}": mean_returns_dict[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                } | {
                    f"charts/{task_name}_success_rate_adapt_{step}": float(success_rate) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, success_rate in mean_success_per_task_dict[step].items()
                } | {
                    f"charts/mean_t_loss_{step}": mean_t_loss[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_r_loss_{step}": mean_r_loss[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/{task_name}_t_loss_{step}": float(loss) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, loss in mean_t_loss_per_task[step].items()
                } | {
                    f"charts/{task_name}_r_loss_{step}": float(loss) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, loss in mean_r_loss_per_task[step].items()
                }

                metrics = eval_metrics

                if track:
                    log(metrics, step=global_step)
                
                # --- Checkpointing ---
                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): float(v)
                            for k, v in metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            print(logs)
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)
        eval_envs.close()
        del eval_envs
        return self


class GradientBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, GradientBasedMetaLearningTrainingConfig, list[Rollout]
    ],
    Generic[AlgorithmConfigType],
):
    """Base class for Gradient-Based Meta-Reinforcement Learning algorithms (e.g., MAML).

    This class implements a training loop that consists of an outer loop for meta-updates
    and an inner loop for task-specific adaptation.
    """
    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        """Sample an action and auxiliary policy outputs.

        Args:
            observation: The observation from the environment.

        Returns:
            A tuple containing the updated algorithm, the sampled action, and auxiliary outputs.
        """
        ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: GradientBasedMetaLearningTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        """Spawn a rollout buffer for multi-task learning.

        Args:
            env_config: The environment configuration.
            training_config: The training configuration.
            seed: The random seed.

        Returns:
            A multi-task rollout buffer.
        """
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @abc.abstractmethod
    def adapt(self, rollouts: Rollout) -> Self:
        """Adapt the policy to a specific task using the provided rollouts.

        Args:
            rollouts: The rollouts collected for the task.

        Returns:
            The adapted algorithm.
        """
        ...

    @abc.abstractmethod
    def init_ensemble_networks(self) -> Self:
        """Initialize the ensemble of networks for meta-learning.

        Returns:
            The algorithm with initialized networks.
        """
        ...

    @override
    def train(
        self,
        config: GradientBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        # Initialize tracking deques and counters
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
            * (config.num_inner_gradient_steps + 1)
        )

        # Main meta-training loop
        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            envs.call("sample_tasks")
            self = self.init_ensemble_networks()
            all_rollouts: list[Rollout] = []

            # --- Inner loop: Sampling and Adaptation ---
            # Collect num_inner_gradient_steps datasets for adaptation + 1 dataset for meta-update
            for _step in range(config.num_inner_gradient_steps + 1):
                print(f"- Collecting inner step {_step}")
                obs, _ = envs.reset()
                rollout_buffer.reset()
                episode_started = np.ones((envs.num_envs,))

                # Collect rollouts for the current inner step
                while not rollout_buffer.ready:
                    self, actions, aux_policy_outs = self.sample_action_and_aux(obs)
                    actions = np.array(actions)


                    next_obs, rewards, terminations, truncations, infos = envs.step(
                        actions
                    )

                    rollout_buffer.add(
                        obs,
                        actions,
                        rewards,
                        episode_started,
                        value=aux_policy_outs.get("value"),
                        log_prob=aux_policy_outs.get("log_prob"),
                        mean=aux_policy_outs.get("mean"),
                        std=aux_policy_outs.get("std"),
                    )

                    episode_started = np.logical_or(terminations, truncations)
                    obs = next_obs

                    for i, env_ended in enumerate(episode_started):
                        if env_ended:
                            global_episodic_return.append(
                                infos["final_info"]["episode"]["r"][i]
                            )
                            global_episodic_length.append(
                                infos["final_info"]["episode"]["l"][i]
                            )
                
                # Store rollouts and adapt the policy
                rollouts = rollout_buffer.get()
                all_rollouts.append(rollouts)

                # Adapt policy for the next inner step (except for the last one)
                if _step < config.num_inner_gradient_steps:
                    print(f"- Adaptation step {_step}")
                    self = self.adapt(rollouts)

            if global_episodic_return:
                mean_episodic_return = np.mean(list(global_episodic_return))
                print("- Mean episodic return: ", mean_episodic_return)
                if track:
                    log(
                        {"charts/mean_episodic_returns": mean_episodic_return},
                        step=global_step,
                    )

            # --- Outer loop: Meta-Update ---
            print("- Computing outer step")
            self, logs = self.update(all_rollouts)
            
            # --- Evaluation ---
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")

                # print(
                #     jnp.abs(self.policy.params["params"]["ContinuousActionPolicyTorso_0"]["VanillaNetwork_0"]["MLP_0"]["layer_0"]["kernel"]).mean(axis=-1)
                # )

                mean_success_rate_dict, mean_returns_dict, mean_success_per_task_dict, \
                    mean_t_loss, mean_r_loss, mean_t_loss_per_task, mean_r_loss_per_task  = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                for step in range(env_config.evaluation_adaptation_steps + 1):
                    print(
                        f"Mean evaluation success rate {mean_success_rate_dict[step]:.4f}, return: {mean_returns_dict[step]:.4f}, adaptation step: {step}"
                    )
                    print(
                        "Mean evaluation success rate per task", mean_success_per_task_dict
                    )
                    print(
                        f"Mean t loss {mean_t_loss[step]:.4f}", f"Mean r loss {mean_r_loss[step]:.4f}"
                    )
                    print(
                        f"Mean t loss per task {mean_t_loss_per_task}", f"Mean r loss per task {mean_r_loss_per_task}"
                    )
                
                # Log evaluation metrics
                eval_metrics = {
                    f"charts/mean_success_rate_adapt_{step}": mean_success_rate_dict[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_evaluation_return_adapt_{step}": mean_returns_dict[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                } | {
                    f"charts/{task_name}_success_rate_adapt_{step}": float(success_rate) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, success_rate in mean_success_per_task_dict[step].items()
                } | {
                    f"charts/mean_t_loss_{step}": mean_t_loss[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_r_loss_{step}": mean_r_loss[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/{task_name}_t_loss_{step}": float(loss) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, loss in mean_t_loss_per_task[step].items()
                } | {
                    f"charts/{task_name}_r_loss_{step}": float(loss) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, loss in mean_r_loss_per_task[step].items()
                }

                if track:
                    log(eval_metrics, step=global_step)

                # --- Checkpointing ---
                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): float(v)
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # --- Logging ---
            print(logs)
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self
    
class VAEBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, VAEBasedMetaLearningTrainingConfig, RolloutWithTask
    ],
    Generic[AlgorithmConfigType]
):
    """Base class for VAE-based Meta-Reinforcement Learning algorithms (e.g., PEARL).

    This class uses a VAE to infer a latent task representation, which is then
    used as input to the policy.

    Attributes:
        task_dim: The dimension of the latent task space.
        vae_rnn_encoder: The VAE RNN encoder model.
    """
    task_dim: int
    vae_rnn_encoder: TrainState

    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: TaskWithObservation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        """Sample an action and auxiliary policy outputs.

        Args:
            observation: The observation, including the task embedding.

        Returns:
            A tuple containing the updated algorithm, the sampled action, and auxiliary outputs.
        """
        ...

    @abc.abstractmethod
    def concat_mean_logvar(
        self, mean: jax.Array, logvar: jax.Array
    ) -> Task:
        """Concatenate the mean and log-variance of the latent task variable.

        Args:
            mean: The mean of the latent task variable.
            logvar: The log-variance of the latent task variable.

        Returns:
            The concatenated task representation.
        """
        ...

    @abc.abstractmethod
    def concat_task_with_observation(
        self, task: Task, observation: Observation
    ) -> TaskWithObservation:
        """Concatenate the task representation with the observation.

        Args:
            task: The task representation.
            observation: The observation.

        Returns:
            The observation with the task representation concatenated.
        """
        ...

    @abc.abstractmethod
    def encode_step(
        self, rnn_params: FrozenDict, obs_next: Observation,
            actions: Action, rewards: Reward, carry: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Perform a single step of the VAE RNN encoder.

        Args:
            rnn_params: The parameters of the VAE RNN encoder.
            obs_next: The next observation.
            actions: The actions taken.
            rewards: The rewards received.
            carry: The recurrent state of the RNN.

        Returns:
            A tuple containing the mean, log-variance, and next recurrent state.
        """
        ...

    @abc.abstractmethod
    def encode_init(
        self, rnn_params: FrozenDict, obs_next: Observation,
            actions: Action, rewards: Reward
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Initialize the VAE RNN encoder.

        Args:
            rnn_params: The parameters of the VAE RNN encoder.
            obs_next: The next observation.
            actions: The actions taken.
            rewards: The rewards received.

        Returns:
            A tuple containing the mean, log-variance, and initial recurrent state.
        """
        ...

    def spawn_rollout_buffer(
        self, 
        task_dim: int,
        env_config: EnvConfig,
        training_config: VAEBasedMetaLearningTrainingConfig,
        seed: int | None = None,
        dtype: npt.DTypeLike = np.float64
    ) -> MultiTaskRolloutBufferWithTask:
        """Spawn a rollout buffer for VAE-based meta-learning.

        Args:
            task_dim: The dimension of the task embedding.
            env_config: The environment configuration.
            training_config: The training configuration.
            seed: The random seed.
            dtype: The data type for the buffer.

        Returns:
            A multi-task rollout buffer.
        """
        
        return MultiTaskRolloutBufferWithTask(
            task_dim=task_dim,
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
            dtype=dtype
        )
    
    def spawn_model_learning_buffer(
        self, 
        env_config: EnvConfig,
        training_config: VAEBasedMetaLearningTrainingConfig,
        seed: int | None = None,
        dtype: npt.DTypeLike = np.float64
    ) -> ModelLearningBuffer:
        """Spawn a model learning buffer for the VAE.

        Args:
            env_config: The environment configuration.
            training_config: The training configuration.
            seed: The random seed.
            dtype: The data type for the buffer.

        Returns:
            A model learning buffer.
        """
        
        return ModelLearningBuffer(
            max_buffer_size=training_config.max_buffer_size,
            sample_size=training_config.sample_size,
            n_rollout_steps=training_config.rollouts_per_task * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
            dtype=dtype
        )
        

    @override
    def train(
        self,
        config: VAEBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed = 1,
        track = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None
    ) -> Self:
        # Initialize tracking deques and counters
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        # Initialize buffers and eval environment
        rollout_buffer = self.spawn_rollout_buffer(
            2 * self.task_dim, env_config, config, seed
        )

        model_buffer = self.spawn_model_learning_buffer(
            env_config, config, seed
        )

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
        )

        # Main training loop
        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")
            envs.call("sample_tasks")

            # --- VAE Encoding ---
            # Get previous rollouts to initialize encoder
            prev_rollouts = rollout_buffer.get()

            mean, logvar, carry = self.encode_init(
                self.vae_rnn_encoder.params, 
                prev_rollouts.next_observations, 
                prev_rollouts.actions,
                prev_rollouts.rewards
            )

            # --- Sampling phase ---
            obs, _ = envs.reset()
            rollout_buffer.reset()
            episode_started = np.ones((envs.num_envs, ))

            # Collect rollouts
            while not rollout_buffer.ready:
                # Use latent task representation to sample action
                task = self.concat_mean_logvar(mean, logvar)
                obs_w_task = self.concat_task_with_observation(task, obs)
                self, actions, aux_policy_outs = (
                    self.sample_action_and_aux(obs_w_task)
                )
                actions = np.array(actions)
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                
                # Add experience to buffer
                rollout_buffer.add(
                    obs,
                    next_obs,
                    task,
                    actions,
                    rewards,
                    episode_started,
                    value=aux_policy_outs.get("value"),
                    log_prob=aux_policy_outs.get("log_prob"),
                    mean=aux_policy_outs.get("mean"),
                    std=aux_policy_outs.get("std"),
                )

                # Update latent task representation with new experience
                mean, logvar, carry = self.encode_step(
                    self.vae_rnn_encoder.params, next_obs, actions, rewards, carry
                )

                episode_started = np.logical_or(terminations, truncations)
                obs = next_obs

                # Log episodic returns
                for i, env_ended in enumerate(episode_started):
                    if env_ended:
                        global_episodic_return.append(
                            infos["final_info"]["episode"]["r"][i]
                        )
                        global_episodic_length.append(
                            infos["final_info"]["episode"]["l"][i]
                        )

            # --- Update phase ---
            rollouts = rollout_buffer.get()

            # Add recent rollouts to the VAE model buffer and sample a batch
            model_buffer.add(
                rollouts.observations,
                rollouts.actions,
                rollouts.next_observations, rollouts.rewards
            )

            vae_obs, vae_action, vae_obs_next, vae_reward = model_buffer.sample()

            if global_episodic_return:
                mean_episodic_return = np.mean(list(global_episodic_return))
                print("- Mean episodic return: ", mean_episodic_return)
                if track:
                    log(
                        {"charts/mean_episodic_returns": mean_episodic_return},
                        step=global_step,
                    )

            # Update models
            print("- Computing update")
            self, logs = self.update(rollouts, vae_obs, vae_action, vae_obs_next, vae_reward)

            # --- Evaluation and Logging ---
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")

                # mean_success_rate now becomes a dict where keys are number of adaptations steps,
                # values are corresponding mean success rate. 
                mean_success_rate_dict, mean_returns_dict, mean_success_per_task_dict, \
                    mean_t_loss, mean_r_loss, mean_t_loss_per_task, mean_r_loss_per_task  = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                for step in range(env_config.evaluation_adaptation_steps + 1):
                    print(
                        f"Mean evaluation success rate {mean_success_rate_dict[step]:.4f}, return: {mean_returns_dict[step]:.4f}, adaptation step: {step}"
                    )
                    print(
                        "Mean evaluation success rate per task", mean_success_per_task_dict
                    )
                    print(
                        f"Mean t loss {mean_t_loss[step]:.4f}", f"Mean r loss {mean_r_loss[step]:.4f}"
                    )
                    print(
                        f"Mean t loss per task {mean_t_loss_per_task}", f"Mean r loss per task {mean_r_loss_per_task}"
                    )
                
                # Log evaluation metrics
                eval_metrics = {
                    f"charts/mean_success_rate_adapt_{step}": mean_success_rate_dict[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_evaluation_return_adapt_{step}": mean_returns_dict[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                } | {
                    f"charts/{task_name}_success_rate_adapt_{step}": float(success_rate) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, success_rate in mean_success_per_task_dict[step].items()
                } | {
                    f"charts/mean_t_loss_{step}": mean_t_loss[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_r_loss_{step}": mean_r_loss[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/{task_name}_t_loss_{step}": float(loss) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, loss in mean_t_loss_per_task[step].items()
                } | {
                    f"charts/{task_name}_r_loss_{step}": float(loss) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, loss in mean_r_loss_per_task[step].items()
                }


                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning_on_train(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        eval_metrics[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                if track:
                    log(eval_metrics, step=global_step)
                
                # --- Checkpointing ---
                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): float(v)
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            print(
                {
                    k: v
                    for k, v in logs.items()
                    if not (k.startswith("nn") or k.startswith("data"))
                }
            )
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self


class RNNBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, RNNBasedMetaLearningTrainingConfig, Rollout
    ],
    Generic[AlgorithmConfigType],
):
    """Base class for RNN-based Meta-Reinforcement Learning algorithms (e.g., RL^2).

    This class uses a recurrent neural network to maintain a state that encodes
    task-specific information.
    """
    @abc.abstractmethod
    def sample_action_and_aux(
        self, state: RNNState, observation: Observation
    ) -> tuple[Self, RNNState, Action, AuxPolicyOutputs]:
        """Sample an action and auxiliary policy outputs.

        Args:
            state: The current RNN state.
            observation: The observation from the environment.

        Returns:
            A tuple containing the updated algorithm, the next RNN state, the sampled
            action, and auxiliary outputs.
        """
        ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: RNNBasedMetaLearningTrainingConfig,
        example_state: RNNState,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        """Spawn a rollout buffer for RNN-based meta-learning.

        Args:
            env_config: The environment configuration.
            training_config: The training configuration.
            example_state: An example RNN state to determine the state dimension.
            seed: The random seed.

        Returns:
            A multi-task rollout buffer.
        """
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            rnn_state_dim=example_state.shape[-1],
            seed=seed,
        )

    @abc.abstractmethod
    def init_recurrent_state(self, batch_size: int) -> tuple[Self, RNNState]:
        """Initialize the recurrent state of the RNN.

        Args:
            batch_size: The batch size (number of environments).

        Returns:
            A tuple containing the algorithm and the initial RNN state.
        """
        ...

    @abc.abstractmethod
    def reset_recurrent_state(
        self, current_state: RNNState, reset_mask: npt.NDArray[np.bool_]
    ) -> tuple[Self, RNNState]:
        """Reset the recurrent state for environments that are done.

        Args:
            current_state: The current RNN state.
            reset_mask: A boolean mask indicating which environments to reset.

        Returns:
            A tuple containing the algorithm and the reset RNN state.
        """
        ...

    @override
    def train(
        self,
        config: RNNBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        # Initialize tracking deques and counters
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        # Initialize RNN state and rollout buffer
        _, example_state = self.init_recurrent_state(config.meta_batch_size)
        rollout_buffer = self.spawn_rollout_buffer(
            env_config, config, example_state, seed
        )

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
        )

        # Main training loop
        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            # --- Sampling phase ---
            envs.call("sample_tasks")
            self, states = self.init_recurrent_state(config.meta_batch_size)
            obs, _ = envs.reset()
            rollout_buffer.reset()
            episode_started = np.ones((envs.num_envs,))

            # Collect rollouts
            sampling_step = 0
            while not rollout_buffer.ready:
                sampling_step += 1
                if sampling_step % 10 == 0:
                    print(f"[rl/algorithms/base.py] Sampling step {sampling_step}", end="\r")
                # Sample action using current RNN state
                self, next_states, actions, aux_policy_outs = (
                    self.sample_action_and_aux(states, obs)
                )
                # print(type(envs.unwrapped.envs[0].unwrapped))
                # exit()
                # print(envs.unwrapped.envs[0].unwrapped.target_velocity)
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                
                # Add experience to buffer, including RNN state
                rollout_buffer.add(
                    obs,
                    actions,
                    rewards,
                    episode_started,
                    value=aux_policy_outs.get("value"),
                    log_prob=aux_policy_outs.get("log_prob"),
                    mean=aux_policy_outs.get("mean"),
                    std=aux_policy_outs.get("std"),
                    rnn_state=states,
                )

                episode_started = np.logical_or(terminations, truncations)
                obs = next_obs
                states = next_states

                # Log episodic returns
                for i, env_ended in enumerate(episode_started):
                    if env_ended:
                        global_episodic_return.append(
                            infos["final_info"]["episode"]["r"][i]
                        )
                        global_episodic_length.append(
                            infos["final_info"]["episode"]["l"][i]
                        )
            # print(time.time() - start)

            # --- Update phase ---
            rollouts = rollout_buffer.get()

            if global_episodic_return:
                mean_episodic_return = np.mean(list(global_episodic_return))
                print("- Mean episodic return: ", mean_episodic_return)
                if track:
                    log(
                        {"charts/mean_episodic_returns": mean_episodic_return},
                        step=global_step,
                    )

            # Update models
            print("- Computing update")
            self, logs = self.update(rollouts)

            # --- Evaluation and Logging ---
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                mean_success_rate_dict, mean_returns_dict, mean_success_per_task_dict, \
                     _, _, _, _ = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                # for step in range(env_config.evaluation_adaptation_steps + 1):
                #     print(
                #         f"Mean evaluation success rate {mean_success_rate_dict[step]:.4f}, return: {mean_returns_dict[step]:.4f}, adaptation step: {step}"
                #     )
                #     print(
                #         "Mean evaluation success rate per task", mean_success_per_task_dict
                #     )
                #     print(
                #         f"Mean t loss {mean_t_loss[step]:.4f}", f"Mean r loss {mean_r_loss[step]:.4f}"
                #     )
                #     print(
                #         f"Mean t loss per task {mean_t_loss_per_task}", f"Mean r loss per task {mean_r_loss_per_task}"
                #     )
                # mean_success_rate now becomes a dict where keys are number of adaptations steps,
                # values are corresponding mean success rate. 

                eval_metrics = {
                    f"charts/mean_success_rate_adapt_{step}": mean_success_rate_dict[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_evaluation_return_adapt_{step}": mean_returns_dict[step] 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                } | {
                    f"charts/{task_name}_success_rate_adapt_{step}": float(success_rate) 
                    for step in range(env_config.evaluation_adaptation_steps + 1)
                    for task_name, success_rate in mean_success_per_task_dict[step].items()
                }

                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning_on_train(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        eval_metrics[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                for step in range(env_config.evaluation_adaptation_steps + 1):
                    print(
                        f"Mean evaluation success rate {mean_success_rate_dict[step]:.4f}, return: {mean_returns_dict[step]:.4f}, adaptation step: {step}"
                    )

                if track:
                    log(eval_metrics, step=global_step)
                
                # --- Checkpointing ---
                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): float(v)
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self


class OffPolicyAlgorithm(
    Algorithm[
        AlgorithmConfigType, OffPolicyTrainingConfig, EnvConfig, ReplayBufferSamples
    ],
    Generic[AlgorithmConfigType],
):
    """Base class for off-policy reinforcement learning algorithms.

    This class implements a training loop for off-policy algorithms, which involves
    collecting experience into a replay buffer and sampling from it to perform updates.
    """
    @abc.abstractmethod
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> AbstractReplayBuffer:
        """Spawn a replay buffer.

        Args:
            env_config: The environment configuration.
            config: The training configuration.
            seed: The random seed.

        Returns:
            An instance of a replay buffer.
        """
        ...

    @abc.abstractmethod
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        """Perform a single update step of the algorithm using data from the replay buffer.

        Args:
            data: A batch of replay buffer samples.

        Returns:
            A tuple containing the updated algorithm and a dictionary of logs.
        """
        ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        """Sample an action from the policy.

        Args:
            observation: The observation from the environment.

        Returns:
            A tuple containing the algorithm and the sampled action.
        """
        ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action:
        """Get the deterministic action for evaluation.

        Args:
            observations: The observations from the environment.

        Returns:
            The deterministic action.
        """
        ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        """Reset the agent. (No-op for most off-policy agents)."""
        del env_mask
        pass  # For evaluation interface compatibility

    @override
    def train(
        self,
        config: OffPolicyTrainingConfig,
        envs: GymVectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        # Initialize tracking deques and counters
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        done = np.full((envs.num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        # Initialize replay buffer
        replay_buffer = self.spawn_replay_buffer(env_config, config, seed)
        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

        # Main training loop
        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            # --- Sampling phase ---
            # Warm-start phase: take random actions to fill the replay buffer
            if global_step < config.warmstart_steps:
                actions = envs.action_space.sample()
            else:
                self, actions = self.sample_action(obs)
            actions = np.array(actions)

            # Step environment
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            done = np.logical_or(terminations, truncations)

            # Add experience to replay buffer
            buffer_obs = next_obs
            if "final_obs" in infos:
                buffer_obs = np.where(
                    done[:, None], np.stack(infos["final_obs"]), next_obs
                )
            replay_buffer.add(obs, buffer_obs, actions, rewards, done)

            obs = next_obs

            # Log episodic returns
            for i, env_ended in enumerate(done):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            # --- Update phase ---
            if global_step > config.warmstart_steps:
                # Sample data from replay buffer and update models
                data = replay_buffer.sample(config.batch_size)
                self, logs = self.update(data)

                # Logging
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * envs.num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)

                    if track:
                        log({"charts/SPS": sps} | logs, step=total_steps)

                # --- Evaluation and Checkpointing ---
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and done.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        log(eval_metrics, step=total_steps)

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not done.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                                buffer=replay_buffer,
                            ),
                            metrics={
                                k.removeprefix("charts/"): float(v)
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()

        return self


class OnPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OnPolicyTrainingConfig, EnvConfig, Rollout],
    Generic[AlgorithmConfigType],
):
    """Base class for on-policy reinforcement learning algorithms.

    This class implements a training loop for on-policy algorithms, which involves
    collecting a batch of experience in a rollout buffer and then updating the
    policy with that batch.
    """
    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        """Sample an action and auxiliary policy outputs (e.g., value, log_prob).

        Args:
            observation: The observation from the environment.

        Returns:
            A tuple containing the algorithm, the sampled action, and auxiliary outputs.
        """
        ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        """Sample an action from the policy.

        Args:
            observation: The observation from the environment.

        Returns:
            A tuple containing the algorithm and the sampled action.
        """
        ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action:
        """Get the deterministic action for evaluation.

        Args:
            observations: The observations from the environment.

        Returns:
            The deterministic action.
        """
        ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        """Reset the agent. (No-op for most on-policy agents)."""
        del env_mask
        pass  # For evaluation interface compatibility

    @abc.abstractmethod
    def update(
        self,
        data: Rollout,
        dones: Float[npt.NDArray, "task 1"],
        next_obs: Float[Observation, " task"] | None = None,
    ) -> tuple[Self, LogDict]:
        """Perform a single update step of the algorithm using a rollout.

        Args:
            data: A rollout of experience.
            dones: A mask indicating which episodes terminated.
            next_obs: The next observations after the rollout.

        Returns:
            A tuple containing the updated algorithm and a dictionary of logs.
        """
        ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: OnPolicyTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        """Spawn a rollout buffer for on-policy training.

        Args:
            env_config: The environment configuration.
            training_config: The training configuration.
            seed: The random seed.

        Returns:
            A multi-task rollout buffer.
        """
        return MultiTaskRolloutBuffer(
            training_config.rollout_steps,
            self.num_tasks,
            env_config.observation_space,
            env_config.action_space,
            seed,
        )

    @override
    def train(
        self,
        config: OnPolicyTrainingConfig,
        envs: GymVectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        # Initialize tracking deques and counters
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        episode_started = np.ones((envs.num_envs,))
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        start_time = time.time()

        # Main training loop
        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            # --- Sampling phase ---
            self, actions, aux_policy_outs = self.sample_action_and_aux(obs)
            actions = np.array(actions)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # Add experience to rollout buffer
            rollout_buffer.add(
                obs,
                actions,
                rewards,
                episode_started,
                value=aux_policy_outs.get("value"),
                log_prob=aux_policy_outs.get("log_prob"),
                mean=aux_policy_outs.get("mean"),
                std=aux_policy_outs.get("std"),
            )

            episode_started = np.logical_or(terminations, truncations)
            obs = next_obs

            # Log episodic returns
            for i, env_ended in enumerate(episode_started):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            # Logging
            if global_step % 1_000 == 0:
                sps_steps = (global_step - start_step) * envs.num_envs
                sps = int(sps_steps / (time.time() - start_time))
                print("SPS:", sps)

                if track:
                    log({"charts/SPS": sps}, step=total_steps)

            if rollout_buffer.ready:
                # Get completed rollout and update models
                rollouts = rollout_buffer.get()
                self, logs = self.update(
                    rollouts,
                    dones=terminations,
                    next_obs=np.where(
                        episode_started[:, None], np.stack(infos["final_obs"]), next_obs
                    ),
                )
                rollout_buffer.reset()

                if track:
                    log(logs, step=total_steps)

                # --- Evaluation and Checkpointing ---
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and episode_started.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        log(eval_metrics, step=total_steps)

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not episode_started.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                            ),
                            metrics={
                                k.removeprefix("charts/"): float(v)
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()
                    episode_started = np.ones((envs.num_envs,))

        return self
