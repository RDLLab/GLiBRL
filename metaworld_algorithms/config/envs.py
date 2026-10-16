import abc
from dataclasses import dataclass
from functools import cached_property

import gymnasium as gym

from metaworld_algorithms.types import Agent, GymVectorEnv, MetaLearningAgent


@dataclass(frozen=True)
class EnvConfig(abc.ABC):
    """Abstract base class for environment configurations.

    This class defines the basic structure and properties that all environment
    configurations must have. It uses a dataclass for easy creation of
    configuration objects and is frozen to make them immutable.

    Args:
        env_id: The ID of the gym environment.
        use_one_hot: Whether to use one-hot encoding for the observations.
        max_episode_steps: The maximum number of steps per episode.
        evaluation_num_episodes: The number of episodes to run for evaluation.
        terminate_on_success: Whether to terminate the episode upon success.
    """

    env_id: str
    use_one_hot: bool = True
    max_episode_steps: int = 500
    evaluation_steps: int = 500
    evaluation_num_episodes: int = 50
    terminate_on_success: bool = False

    @cached_property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        """The action space of the environment."""
        ...

    @cached_property
    @abc.abstractmethod
    def observation_space(self) -> gym.Space:
        """The observation space of the environment."""
        ...

    @abc.abstractmethod
    def spawn(self, seed: int = 1) -> GymVectorEnv:
        """Spawns a vectorized environment.

        Args:
            seed: The random seed for the environment.

        Returns:
            A vectorized gym environment.
        """
        ...

    @abc.abstractmethod
    def evaluate(
        self, envs: GymVectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        """Evaluates an agent in the environment.

        Args:
            envs: The vectorized environments to evaluate on.
            agent: The agent to evaluate.

        Returns:
            A tuple containing the mean return, mean success rate, and a
            dictionary of other evaluation metrics.
        """
        ...


@dataclass(frozen=True)
class MetaLearningEnvConfig(EnvConfig):
    """Abstract base class for meta-learning environment configurations.

    This class extends `EnvConfig` with additional properties and methods
    required for meta-learning settings.

    Args:
        recurrent_info_in_obs: Whether to include recurrent information
            (e.g., previous action, reward) in the observation.
    """

    recurrent_info_in_obs: bool = False

    @abc.abstractmethod
    def spawn_test(self, seed: int = 1) -> GymVectorEnv:
        """Spawns a vectorized test environment for meta-learning.

        Args:
            seed: The random seed for the environment.

        Returns:
            A vectorized gym environment for testing.
        """
        ...

    @abc.abstractmethod
    def evaluate_metalearning(
        self, envs: GymVectorEnv, agent: MetaLearningAgent
    ) -> tuple[dict[int, float], dict[int, float], dict[int, dict[str, float]]]:
        """Evaluates a meta-learning agent.

        Args:
            envs: The vectorized environments to evaluate on.
            agent: The meta-learning agent to evaluate.

        Returns:
            A tuple containing dictionaries for returns, success rates, and
            other metrics, keyed by task index.
        """
        ...

    @abc.abstractmethod
    def evaluate_metalearning_on_train(
        self, envs: GymVectorEnv, agent: MetaLearningAgent
    ) -> tuple[float, float, dict[str, float]]:
        """Evaluates a meta-learning agent on the training tasks.

        Args:
            envs: The vectorized environments to evaluate on.
            agent: The meta-learning agent to evaluate.

        Returns:
            A tuple containing the mean return, mean success rate, and a
            dictionary of other evaluation metrics.
        """
        ...
