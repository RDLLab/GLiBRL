from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy.typing as npt
from numpy import float64

if TYPE_CHECKING:
    from metaworld_algorithms.rl.algorithms.base import Algorithm

    from .envs import EnvConfig


@dataclass(frozen=True)
class AlgorithmConfig:
    """Base configuration for a reinforcement learning algorithm."""

    num_tasks: int
    """The number of tasks in the environment. Relevant for multi-task and meta-learning."""
    gamma: float = 0.99
    """The discount factor for future rewards."""

    def spawn(self, env: "EnvConfig", seed: int) -> "Algorithm":
        """Spawns an algorithm instance from its configuration.

        This factory method uses a helper to find the appropriate algorithm class
        for this configuration object's type and then initializes it.

        Args:
            env: The environment configuration.
            seed: The random seed.

        Returns:
            An initialized algorithm instance.
        """
        from metaworld_algorithms.rl.algorithms import get_algorithm_for_config

        return get_algorithm_for_config(self).initialize(self, env, seed)


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Base configuration for a training loop."""

    total_steps: int
    """The total number of environment steps to run during training."""
    evaluation_frequency: int = 200_000 // 500
    """The frequency (in total environment episodes) at which to run evaluation."""


@dataclass(frozen=True)
class OffPolicyTrainingConfig(TrainingConfig):
    """Configuration for an off-policy training loop (e.g., for SAC)."""

    warmstart_steps: int = int(4e3)
    """Number of initial steps to run with a random policy to populate the replay buffer."""
    buffer_size: int = int(1e6)
    """The maximum size of the replay buffer."""
    batch_size: int = 1280
    """The number of transitions to sample from the buffer for each training step."""


@dataclass(frozen=True)
class OnPolicyTrainingConfig(TrainingConfig):
    """Configuration for an on-policy training loop (e.g., for PPO)."""

    rollout_steps: int = 10_000
    """The number of environment steps to collect for each policy update."""


@dataclass(frozen=True)
class MetaLearningTrainingConfig(TrainingConfig):
    """Base configuration for a meta-learning training loop."""

    meta_batch_size: int = 20
    """The number of tasks to sample for each meta-update."""
    rollouts_per_task: int = 10
    """The number of episodes (rollouts) to collect for each task in the meta-batch."""
    evaluate_on_train: bool = False
    """Whether to run an additional evaluation on the training tasks."""

    evaluation_frequency: int = 1_000_000
    """The frequency (in total environment timesteps) at which to run meta-evaluation."""
    max_test_steps: int = 200
    """The maximum number of steps per episode during meta-testing."""


@dataclass(frozen=True)
class GradientBasedMetaLearningTrainingConfig(MetaLearningTrainingConfig):
    """Configuration for gradient-based meta-learning (e.g., MAML)."""

    num_inner_gradient_steps: int = 1
    """The number of gradient steps for inner-loop adaptation to a new task."""


@dataclass(frozen=True)
class RNNBasedMetaLearningTrainingConfig(MetaLearningTrainingConfig):
    """Configuration for RNN-based meta-learning (e.g., RL^2).

    These algorithms use a recurrent network to maintain a hidden state that
    encodes task-specific information, allowing for adaptation within an episode.
    """

    ...


@dataclass(frozen=True)
class BRLMetaLearningTrainingConfig(MetaLearningTrainingConfig):
    """Configuration for Bayesian RL-based meta-learning algorithms.

    These methods typically use a context composed of recent transitions
    to infer a belief over the task.
    """

    max_buffer_size: int = 500
    """The maximum size of the context buffer for each task."""
    sample_size: int = 10
    """The number of transitions to sample to form the context."""

@dataclass(frozen=True)
class VAEBasedMetaLearningTrainingConfig(MetaLearningTrainingConfig):
    """Configuration for VAE-based meta-learning (e.g., VariBAD).

    These algorithms use a Variational Autoencoder to encode the context
    (recent transitions) into a latent task variable.
    """

    max_buffer_size: int = 500
    """The maximum size of the context buffer for the VAE."""
    sample_size: int = 10
    """The number of transitions to sample to encode the context."""


@dataclass(frozen=True, kw_only=True)
class GLiBRLSACTrainingConfig(TrainingConfig):
    """Configuration for GLiBRL-SAC: off-policy meta-learning with Bayesian task inference.

    Combines the Bayesian task inference of GLiBRL with an off-policy SAC backbone,
    following a PEARL-like dual replay buffer framework.
    """

    warmstart_steps: int = int(4e3)
    """Number of initial random-action steps to populate the replay buffers."""
    buffer_size: int = int(1e6)
    """Maximum capacity of each replay buffer (policy buffer and model buffer)."""
    batch_size: int = 256
    """Number of transitions sampled per model update step."""
    policy_batch_size_range: tuple[int, int] | None = None
    """If set, policy update batch size is sampled uniformly from this range each step. If None, uses batch_size."""
    meta_batch_size: int = 20
    """Number of parallel environments (tasks) running simultaneously."""
    evaluate_on_train: bool = False
    """Whether to run an additional evaluation on the training tasks."""
    evaluation_frequency: int = 200_000
    """Frequency (in total environment timesteps) at which to run meta-evaluation."""
    max_test_steps: int = 200
    """Maximum steps per episode during meta-testing."""
    rollout_steps: int = 500
    """Number of environment steps to collect per iteration before updating."""
    num_gradient_steps: int = 1
    """Number of SAC gradient updates per rollout iteration."""
    num_model_steps: int = 5
    """Number of model (t_mix/r_mix) gradient updates per rollout iteration."""
    model_train_until_steps: int | None = None
    """Stop model (T/R) updates after this many total env steps. None = always update."""
    resample_tasks_when_full: bool = False
    """If True, resample tasks and reset envs at the start of each iteration once
    the buffer is full. Appropriate for multi-task benchmarks (ML10/ML45,
    HalfCheetahVel) where task variety matters. Leave False for fixed-task
    settings (HalfCheetahDir) where the same tasks should fill the buffer."""