from typing import Any, List, NamedTuple, Protocol, TypedDict, Union, TypeAlias
try:
    from typing import override
except ImportError:
    def override(method):
        return method

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float, Int
from metaworld_algorithms.monitoring.utils import Histogram

Observation = Float[np.ndarray, "... obs_dim"]
Action = Float[np.ndarray, "... action_dim"]
Reward = Float[np.ndarray, "... 1"]

PlannerState = Float[Array, "... n_sim state_dim"]
PlannerAction = Int[Array, "... n_sim action_dim"]
PlannerObservation = Int[Array, "... n_sim obs_dim"]
PlannerReward = Float[Array, "... n_sim 1"]

StateIdx = Int[Array, "... n_sim 1"] 
ObsIdx = Int[Array, "... n_sim 1"] 
ObsDepth = Int[Array, "... n_sim 1"]
ActionIdx = Int[Array, "... n_sim 1"] 
ActionDepth = Int[Array, "... n_sim 1"]

NodeIdx = Int[Array, "... 2"] # parent idx, action/obs_idx
NodeVc = Int[Array, "... 1"] # visit_count
NodeVal = Float[Array, "... 1"] 

NodeHash = Float[Array, "..."]
NodeCount = Int[Array, "..."]

ObsFeat = Float[np.ndarray, "... obs_feat"]
ActFeat = Float[np.ndarray, "... act_feat"]
RewardFeat = Float[np.ndarray, "... reward_dim"]
ObsActFeat = Float[np.ndarray, "... obs_action_dim"]
ObsActObsFeat = Float[np.ndarray, "... obs_action_obs_dim"]

Task = Float[np.ndarray, "... task_dim"]
TaskWithObservation = Float[np.ndarray, "... task_obs_dim"]
TaskParam = Float[np.ndarray, "... t1 t2"]
TaskTuple = tuple[TaskParam, TaskParam, TaskParam, TaskParam, TaskParam,
                  TaskParam, TaskParam, TaskParam, TaskParam, TaskParam,
                  TaskParam]
TaskTMean = Float[np.ndarray, "... dt ds"]
TaskTCov = Float[np.ndarray, "... ds ds"] 
TaskRMean = Float[np.ndarray, "... dr 1"]
TaskRCov = Float[np.ndarray, "... 1 1"] 

TaskTFeat = Float[np.ndarray, "... dt"]
TaskTBias = Float[np.ndarray, "... obs_dim"]
TaskRFeat = Float[np.ndarray, "... dr"]
TaskRBias = Float[np.ndarray, "... 1"]

RNNState = Float[np.ndarray, "... rnn_state_dim"]
LayerActivations = Float[Array, "batch_size layer_dim"]

Value = Float[np.ndarray, "... 1"]
LogProb = Float[np.ndarray, "... 1"]


LogDict: TypeAlias = dict[str, float | Float[Array, ""] | Histogram]
AuxPolicyOutputs: TypeAlias = dict[str, npt.NDArray]
LayerActivationsDict: TypeAlias = dict[str, Float[Array, "batch_size layer_dim"]]
Intermediates: TypeAlias = dict[str, Union[tuple[LayerActivations, ...], "Intermediates"]]


class ReplayBufferSamples(NamedTuple):
    observations: Float[Observation, " batch"]
    actions: Float[Action, " batch"]
    next_observations: Float[Observation, " batch"]
    dones: Float[np.ndarray, "batch 1"]
    rewards: Float[np.ndarray, "batch 1"]


class Rollout(NamedTuple):
    # Standard timestep data
    observations: Float[Observation, "timestep task"]
    actions: Float[Action, "timestep task"]
    rewards: Float[np.ndarray, "timestep task 1"]
    dones: Float[np.ndarray, "timestep task 1"]

    # Auxiliary policy outputs
    log_probs: Float[LogProb, "timestep task"] | None = None
    means: Float[Action, "timestep task"] | None = None
    stds: Float[Action, "timestep task"] | None = None
    values: Float[np.ndarray, "timestep task 1"] | None = None
    rnn_states: Float[RNNState, "timestep task"] | None = None

    # Computed statistics about observed rewards
    returns: Float[np.ndarray, "timestep task 1"] | None = None
    advantages: Float[np.ndarray, "timestep task 1"] | None = None
    valids: Float[np.ndarray, "episode timestep 1"] | None = None

    @classmethod
    def from_list(cls, items: list) -> "Rollout":
        if Timestep.is_timestep(items[0]):
            items = list(map(lambda x: Timestep.to_rollout(x), items))
        return cls(*map(lambda *xs: np.stack(xs), *items))

class RolloutWithTask(NamedTuple):
    observations: Float[Observation, "timestep task"]
    next_observations: Float[Observation, "timestep task"]
    actions: Float[Action, "timestep task"]
    rewards: Float[np.ndarray, "timestep task 1"]
    dones: Float[np.ndarray, "timestep task 1"]
    task: Float[Task, "task timestep taskdim"] | None

    obs_task: Float[TaskWithObservation, "timestep task taskdim+obs_dim"] | None = None
    action_idx: Float[np.ndarray, "timestep task 1"] = None
    # Auxiliary policy outputs
    log_probs: Float[LogProb, "timestep task"] | None = None
    means: Float[Action, "timestep task"] | None = None
    stds: Float[Action, "timestep task"] | None = None
    values: Float[np.ndarray, "timestep task 1"] | None = None
    rnn_states: Float[RNNState, "timestep task"] | None = None
    mr_raw: Float[np.ndarray, "timestep task reward_latent_dim"] | None = None

    # Computed statistics about observed rewards
    returns: Float[np.ndarray, "timestep task 1"] | None = None
    advantages: Float[np.ndarray, "timestep task 1"] | None = None
    valids: Float[np.ndarray, "episode timestep 1"] | None = None



    @override
    @classmethod
    def from_list(cls, items: list) -> "RolloutWithTask":
        if TimestepWithTask.is_timestep(items[0]):
            items = list(map(lambda x: TimestepWithTask.to_rollout(x), items))
        return cls(*map(lambda *xs: np.stack(xs) if xs[0] is not None else None, *items))


class Timestep(NamedTuple):
    observation: npt.NDArray
    action: npt.NDArray
    reward: npt.NDArray
    terminated: npt.NDArray
    truncated: npt.NDArray
    aux_policy_outputs: dict[str, npt.NDArray]

    @classmethod
    def is_timestep(cls, item: Any) -> bool:
        return hasattr(item, "_fields") and item._fields == cls._fields

    @staticmethod
    def to_rollout(item: "Timestep") -> Rollout:
        log_probs = item.aux_policy_outputs.get("log_prob")
        if log_probs is not None:
            log_probs = log_probs[..., None]

        return Rollout(
            observations=item.observation,
            actions=item.action,
            rewards=item.reward[..., None],
            dones=item.truncated[..., None],
            log_probs=log_probs,
            means=item.aux_policy_outputs.get("mean"),
            stds=item.aux_policy_outputs.get("std"),
            values=item.aux_policy_outputs.get("value"),
        )
    
class TimestepWithTask(Timestep):
    task: Task
    next_observations: npt.NDArray

    @override
    @staticmethod
    def to_rollout(item: "TimestepWithTask") -> RolloutWithTask:
        log_probs = item.aux_policy_outputs.get("log_prob")
        if log_probs is not None:
            log_probs = log_probs[..., None]

        return RolloutWithTask(
            observations=item.observation,
            next_observations=item.next_observations,
            task=item.task,
            actions=item.action,
            rewards=item.reward[..., None],
            dones=item.truncated[..., None],
            log_probs=log_probs,
            means=item.aux_policy_outputs.get("mean"),
            stds=item.aux_policy_outputs.get("std"),
            values=item.aux_policy_outputs.get("value"),
        )


class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None: ...


class MetaLearningAgent(Agent, Protocol):
    def init(self) -> None: ...

    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]: ...

    def step(self, timestep: Timestep) -> None: ...

    def adapt(self) -> None: ...


class CheckpointMetadata(TypedDict):
    timestamp: str
    step: int
    episodes_ended: int


class RNGCheckpoint(TypedDict):
    python_rng_state: tuple[Any, ...]
    global_numpy_rng_state: dict[str, Any]


class ReplayBufferCheckpoint(TypedDict):
    data: dict[str, npt.NDArray[np.float32] | int | bool]
    rng_state: Any


GymVectorEnv: TypeAlias = gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv
EnvCheckpoint: TypeAlias = list[tuple[str, dict[str, Any]]]
