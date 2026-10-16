"""This module contains functions for evaluating reinforcement learning agents in the Meta-World environment.

It provides two main evaluation functions:
- `evaluation`: For standard evaluation of a trained agent on a set of tasks.
- `metalearning_evaluation`: For evaluating a meta-learning agent's ability to adapt to new tasks.
"""
from __future__ import annotations

from typing import NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import jax

from metaworld.env_dict import ALL_V3_ENVIRONMENTS

global_counter = 0

class Agent(Protocol):
    """A protocol for a reinforcement learning agent."""
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Get the action for the given observation in evaluation mode.

        Args:
            observations: The observations from the environment.

        Returns:
            The actions to take.
        """
        ... 

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        """Reset the agent's state for the given environments.

        Args:
            env_mask: A boolean mask indicating which environments to reset.
        """
        ...


class MetaLearningAgent(Agent, Protocol):
    """A protocol for a meta-learning agent."""
    def init(self) -> None:
        """Initialize the agent for a new meta-episode."""
        ...

    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
        """Get the action for the given observation during adaptation.

        Args:
            observations: The observations from the environment.

        Returns:
            A tuple of the actions to take and auxiliary policy outputs.
        """
        ...

    def step(self, timestep: Timestep) -> None:
        """Process a single step of interaction with the environment.

        Args:
            timestep: The timestep data.
        """
        ...

    def adapt(self) -> None:
        """Adapt the agent to the current task."""
        ...


def _get_task_names(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
) -> list[str]:
    """Get the task names from the environment.

    Args:
        envs: The vectorized environment.

    Returns:
        A list of task names.
    """
    metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
    return [
        metaworld_cls_to_task_name[task_name]
        for task_name in envs.get_attr("task_name")
    ]


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_episodes: int = 50,
) -> tuple[float, float, dict[str, float], dict[str, list[float]],
            dict[str, int], dict[str, float], dict[str, float]]:
    global global_counter
    terminate_on_success = np.all(eval_envs.get_attr("terminate_on_success")).item()
    eval_envs.call("toggle_terminate_on_success", True)

    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    agent.reset(np.ones(eval_envs.num_envs, dtype=np.bool_))

    # hand_init = np.array([eval_envs.envs[i].unwrapped.hand_init_pos for i in range(len(eval_envs.envs))])
    # goal = np.array([eval_envs.envs[i].unwrapped._target_pos for i in range(len(eval_envs.envs))])

    task_names = _get_task_names(eval_envs)
    successes = {task_name: 0 for task_name in set(task_names)}
    predictive_t_error = {task_name: [] for task_name in set(task_names)}
    predictive_r_error = {task_name: [] for task_name in set(task_names)}

    episodic_returns: dict[str, list[float]] = {
        task_name: [] for task_name in set(task_names)
    }

    def eval_done(returns):
        return all(len(r) >= num_episodes for _, r in returns.items())

    counter = 0

    while not eval_done(episodic_returns):
        actions = jax.device_get(agent.eval_action(obs))
        counter += 1
        
        next_obs, reward, terminations, truncations, infos = eval_envs.step(actions)
        agent.step(Timestep(obs, actions, reward, None, None, None))

        t_loss, r_loss = agent.predictive_losses(next_obs, reward)
        obs = next_obs
        dones = np.logical_or(terminations, truncations)
        agent.reset(dones)

        for i, env_ended in enumerate(dones):
            if not env_ended:
                predictive_t_error[task_names[i]].append(t_loss[i])
                predictive_r_error[task_names[i]].append(r_loss[i])

            else:
                episodic_returns[task_names[i]].append(
                    float(infos["final_info"]["episode"]["r"][i])
                )
                if len(episodic_returns[task_names[i]]) <= num_episodes:
                    successes[task_names[i]] += int(infos["final_info"]["success"][i])


    episodic_returns = {
        task_name: returns[:num_episodes]
        for task_name, returns in episodic_returns.items()
    }

    success_rate_per_task = {
        task_name: task_successes / num_episodes
        for task_name, task_successes in successes.items()
    }
    mean_success_rate = np.mean(list(success_rate_per_task.values()))
    mean_returns = np.mean(list(episodic_returns.values()))

    error_count_per_task = {task_name: len(errors) for task_name, errors in predictive_t_error.items()}
    total_error_t_per_task = {task_name: sum(errors) for task_name, errors in predictive_t_error.items()}
    total_error_r_per_task = {task_name: sum(errors) for task_name, errors in predictive_r_error.items()}

    eval_envs.call("toggle_terminate_on_success", terminate_on_success)

    return (
        float(mean_success_rate),
        float(mean_returns),
        success_rate_per_task,
        episodic_returns,
        error_count_per_task,
        total_error_t_per_task,
        total_error_r_per_task,
    )


def metalearning_evaluation(
    agent: MetaLearningAgent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_evals: int = 10,  # Assuming 40 goals per test task and meta batch size of 20
    adaptation_steps: int = 1,
    adaptation_episodes: int = 10,
    evaluation_episodes: int = 3,
) -> tuple[dict[int, float], dict[int, float], dict[int, dict[str, float]],
           dict[int, float], dict[int, float], dict[int, dict[str, float]],
           dict[int, float]]:
    """Evaluate a meta-learning agent.

    This function evaluates a meta-learning agent by testing its ability to adapt to new tasks.
    It runs a number of evaluations, and for each evaluation, it allows the agent to adapt for a
    certain number of steps. The agent's performance is evaluated before and after each adaptation step.

    Args:
        agent: The meta-learning agent to evaluate.
        eval_envs: The vectorized environment to evaluate the agent in.
        num_evals: The number of meta-evaluation loops to run.
        adaptation_steps: The number of adaptation steps to perform.
        adaptation_episodes: The number of episodes to run for each adaptation step.
        evaluation_episodes: The number of episodes to run for each evaluation.

    Returns:
        A tuple containing:
            - mean_success_rate_dict: A dictionary mapping adaptation steps to the mean success rate.
            - mean_return_dict: A dictionary mapping adaptation steps to the mean return.
            - success_rate_per_task_dict: A dictionary mapping adaptation steps to a dictionary of
              task names and their success rates.
            - mean_t_loss_dict: A dictionary mapping adaptation steps to the mean transition loss.
            - mean_r_loss_dict: A dictionary mapping adaptation steps to the mean reward loss.
            - mean_t_per_task_dict: A dictionary mapping adaptation steps to a dictionary of
              task names and their mean transition loss.
            - mean_r_per_task_dict: A dictionary mapping adaptation steps to a dictionary of
              task names and their mean reward loss.
    """
    # Configure environments for meta-evaluation
    eval_envs.call("toggle_sample_tasks_on_reset", False)
    eval_envs.call("toggle_terminate_on_success", False)
    task_names = _get_task_names(eval_envs)

    # Initialize dictionaries to store metrics for each adaptation step
    mean_success_rate_dict = {i:0.0 for i in range(adaptation_steps + 1)}
    mean_return_dict = {i:0.0 for i in range(adaptation_steps + 1)}

    mean_t_loss_dict = {i:0.0 for i in range(adaptation_steps + 1)}
    mean_r_loss_dict = {i:0.0 for i in range(adaptation_steps + 1)}
    mean_t_per_task_dict = {}
    mean_r_per_task_dict = {}

    total_t_loss_dict = {i:[0.0, 0] for i in range(adaptation_steps + 1)}
    total_r_loss_dict = {i:[0.0, 0] for i in range(adaptation_steps + 1)}

    t_total_loss_per_task = {i:np.zeros((len(set(task_names)), 2)) for i in range(adaptation_steps + 1)}
    r_total_loss_per_task = {i:np.zeros((len(set(task_names)), 2)) for i in range(adaptation_steps + 1)}

    success_rate_per_task = {i:np.zeros((num_evals, len(set(task_names)))) for i in range(adaptation_steps + 1)}
    success_rate_per_task_dict = {}

    # Main meta-evaluation loop
    for i in range(num_evals):
        obs: npt.NDArray[np.float64]
        # Sample new tasks for this evaluation run
        eval_envs.call("sample_tasks")
        agent.init()
        
        # Loop through adaptation steps (including zero-shot at step 0)
        for step in range(adaptation_steps + 1):
            obs, _ = eval_envs.reset()
            episodes_elapsed = np.zeros((eval_envs.num_envs,), dtype=np.uint16)
            # Determine number of episodes for adaptation
            episode_to_adapt = adaptation_episodes if step > 0 else -1

            # Adaptation phase
            while not (episodes_elapsed >= episode_to_adapt).all():
                actions, aux_policy_outs = agent.adapt_action(obs)
                next_obs, rewards, terminations, truncations, _ = eval_envs.step(
                    actions
                )
                agent.step(
                    Timestep(
                        obs,
                        actions,
                        rewards,
                        terminations,
                        truncations,
                        aux_policy_outs,
                    )
                )
                episodes_elapsed += np.logical_or(terminations, truncations)
                obs = next_obs
 
            # Adapt the agent after collecting experience
            agent.adapt()
            print(f"Finished Adapt Step {step}, {episode_to_adapt} Episodes Adapted.")

            # Evaluation phase
            # Run standard evaluation
            mean_success_rate, mean_return, _success_rate_per_task, _, \
                error_count_per_task, total_t_loss_per_task, total_r_loss_per_task  = evaluation(
                agent, eval_envs, evaluation_episodes
            )
            # Aggregate metrics
            mean_success_rate_dict[step] += mean_success_rate
            mean_return_dict[step] += mean_return
            success_rate_per_task[step][i] = np.array(list(_success_rate_per_task.values()))

            total_t = 0
            total_r = 0
            counter = 0
            
            # Aggregate predictive losses per task
            for j, task_name in enumerate(set(task_names)):
                total_t += total_t_loss_per_task[task_name]
                total_r += total_r_loss_per_task[task_name]
                counter += error_count_per_task[task_name]

                t_total_loss_per_task[step][j] += np.array([total_t_loss_per_task[task_name], 
                                                              error_count_per_task[task_name]])
                r_total_loss_per_task[step][j] += np.array([total_r_loss_per_task[task_name], 
                                                              error_count_per_task[task_name]])
            # Aggregate total predictive losses
            total_t_loss_dict[step][0] += total_t
            total_t_loss_dict[step][1] += counter

            total_r_loss_dict[step][0] += total_r
            total_r_loss_dict[step][1] += counter
            
    # Calculate mean losses per task
    t_losses_per_task = {
        step: t_total_loss_per_task[step][:, 0] / t_total_loss_per_task[step][:, 1] for step in range(adaptation_steps + 1)
    }
    r_losses_per_task = {
        step: r_total_loss_per_task[step][:, 0] / r_total_loss_per_task[step][:, 1] for step in range(adaptation_steps + 1)
    }

    # Finalize and average metrics across all evaluations
    for step in range(adaptation_steps + 1):
        mean_success_rate_dict[step] = mean_success_rate_dict[step] / num_evals
        mean_return_dict[step] = mean_return_dict[step] / num_evals

        success_rates = success_rate_per_task[step].mean(axis=0)
        task_success_rates = {
            task_name: success_rates[i] for i, task_name in enumerate(set(task_names))
        }
        success_rate_per_task_dict[step] = task_success_rates

        mean_t_loss_dict[step] = total_t_loss_dict[step][0] / total_t_loss_dict[step][1]
        mean_r_loss_dict[step] = total_r_loss_dict[step][0] / total_r_loss_dict[step][1]

        mean_t_per_task_dict[step] = {task_name: t_losses_per_task[step][i] for i, task_name in enumerate(set(task_names))}
        mean_r_per_task_dict[step] = {task_name: r_losses_per_task[step][i] for i, task_name in enumerate(set(task_names))}


    return (
        mean_success_rate_dict,
        mean_return_dict,
        success_rate_per_task_dict,
        mean_t_loss_dict,
        mean_r_loss_dict,
        mean_t_per_task_dict,
        mean_r_per_task_dict
    )

class Timestep(NamedTuple):
    """Represents a single step in the environment.

    Attributes:
        observation: The observation from the environment.
        action: The action taken by the agent.
        reward: The reward received from the environment.
        terminated: Whether the episode has terminated.
        truncated: Whether the episode has been truncated.
        aux_policy_outputs: Auxiliary outputs from the policy.
    """
    observation: npt.NDArray
    action: npt.NDArray
    reward: npt.NDArray
    terminated: npt.NDArray
    truncated: npt.NDArray
    aux_policy_outputs: dict[str, npt.NDArray]
