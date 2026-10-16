from dataclasses import dataclass

from metaworld_algorithms.config.utils import Initializer, StdType

from .nn import NeuralNetworkConfig, RecurrentNeuralNetworkConfig, VanillaNetworkConfig


@dataclass(frozen=True)
class ContinuousActionPolicyConfig:
    """Configuration for a continuous action policy network.

    This dataclass defines the settings for a policy network that outputs
    actions in a continuous space. It includes parameters for the network
    architecture, action distribution, and initialization.
    """

    network_config: NeuralNetworkConfig = VanillaNetworkConfig(width=(400, 400, 400))
    """The config for the neural network to use for function approximation."""

    squash_tanh: bool = True
    """Whether to squash the outputs with a tanh function. This is commonly
    used to bound the actions to a specific range, typically [-1, 1]."""

    log_std_min: float | None = -20.0
    """The minimum possible log standard deviation for each action distribution.
    This helps to prevent numerical instability by clipping the log standard
    deviation to a reasonable range."""

    log_std_max: float | None = 2.0
    """The maximum possible log standard deviation for each action distribution.
    This helps to prevent numerical instability by clipping the log standard
    deviation to a reasonable range."""

    std_type: StdType = StdType.MLP_HEAD
    """How to learn the standard deviation of the distribution.
    - `MLP_HEAD`: The standard deviation is state-dependent and is an output
      from the last layer of the MLP torso.
    - `PARAM`: The standard deviation is a learned parameter for each action
      dimension and is state-independent."""

    head_kernel_init: Initializer | None = None
    """Override the initializer to use for the MLP head weights."""

    head_bias_init: Initializer | None = None
    """Override the initializer to use for the MLP head biases."""


@dataclass(frozen=True)
class RecurrentContinuousActionPolicyConfig:
    """Configuration for a recurrent continuous action policy network.

    This dataclass defines the settings for a policy network that uses a
    recurrent neural network (RNN) to handle sequential data. It is suitable
    for partially observable environments where memory is important.
    """

    network_config: RecurrentNeuralNetworkConfig = RecurrentNeuralNetworkConfig()
    """The config for the neural network to use for function approximation."""

    encoder_config: NeuralNetworkConfig | None = VanillaNetworkConfig(
        width=(400, 400),
    )
    """The config for the neural network to use for encoding the observations.
    The optimizer config for this network is ignored."""

    squash_tanh: bool = True
    """Whether to squash the outputs with a tanh function."""

    log_std_min: float | None = -20.0
    """The minimum possible log standard deviation for each action distribution."""

    log_std_max: float | None = 2.0
    """The maximum possible log standard deviation for each action distribution."""

    std_type: StdType = StdType.MLP_HEAD
    """How to learn the standard deviation of the distribution.
    - `MLP_HEAD`: The standard deviation is an output from the last layer of
      the MLP torso and is therefore state-dependent.
    - `PARAM`: The standard deviation is a learned parameter per action
      dimension and is state-independent."""

    head_kernel_init: Initializer | None = None
    """Override the initializer to use for the MLP head weights."""

    head_bias_init: Initializer | None = None
    """Override the initializer to use for the MLP head biases."""

    activate_head: bool = False
    """Whether to activate the MLP head after the RNN layer."""


@dataclass(frozen=True)
class QValueFunctionConfig:
    """Configuration for a Q-value function network.

    This dataclass defines the settings for a network that estimates the
    Q-value (action-value) for a given state-action pair.
    """

    network_config: NeuralNetworkConfig = VanillaNetworkConfig(width=(400, 400, 400))
    """The config for the neural network to use for function approximation."""

    use_classification: bool = False
    """Whether to use classification instead of regression for the Q-value
    estimation. If True, the network will output a distribution over a
    discrete set of Q-values."""


@dataclass(frozen=True)
class ValueFunctionConfig(QValueFunctionConfig):
    """Configuration for a state-value function network.

    This class inherits from `QValueFunctionConfig` and is used to define a
    network that estimates the state-value (V-value) for a given state.
    It is commonly used in actor-critic architectures.
    """

    ...
