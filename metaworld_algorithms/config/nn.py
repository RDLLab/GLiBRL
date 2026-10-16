from dataclasses import dataclass

from .optim import OptimizerConfig
from .utils import Activation, CellType, Initializer


@dataclass(frozen=True, kw_only=True)
class NeuralNetworkConfig:
    """Base configuration for a neural network.

    This dataclass defines the fundamental properties of a neural network,
    including layer widths, initializers, activation functions, and the optimizer.
    """

    width: tuple = (400, 400, 400)
    """The number of neurons in the hidden layers."""

    kernel_init: Initializer = Initializer.HE_UNIFORM
    """The initializer to use for hidden layer weights."""
    # TODO: How to pass arguments to kernel_init

    bias_init: Initializer = Initializer.ZEROS
    """The initializer to use for hidden layer biases."""
    # TODO: How to pass arguments to bias_init

    use_bias: bool = True
    """Whether or not to use bias terms across the network."""

    activation: Activation = Activation.ReLU
    """The activation function to use."""

    optimizer: OptimizerConfig = OptimizerConfig()
    """The optimizer to use for the network."""


@dataclass(frozen=True, kw_only=True)
class RecurrentNeuralNetworkConfig:
    """Configuration for a recurrent neural network (RNN).

    This dataclass defines the properties of an RNN, including the cell type,
    hidden state dimension, initializers, and activation function.
    """

    width: int = 256
    """The dimension of the recurrent layer and the hidden state."""

    cell_type: CellType = CellType.GRU
    """The type of recurrent cell to use (e.g., GRU, LSTM)."""

    recurrent_kernel_init: Initializer = Initializer.HE_UNIFORM
    """The initializer to use for the recurrent layer weights."""

    kernel_init: Initializer = Initializer.HE_UNIFORM
    """The initializer to use for the recurrent layer weights."""

    bias_init: Initializer = Initializer.ZEROS
    """The initializer to use for the recurrent layer biases."""

    use_bias: bool = True
    """Whether or not to use bias terms across the network."""

    carry_init: Initializer = Initializer.ZEROS
    """The initializer to use for the recurrent carry (initial hidden state)."""

    activation: Activation = Activation.ReLU
    """The activation function to use after the recurrent layer."""

    optimizer: OptimizerConfig = OptimizerConfig()
    """The optimizer to use for the whole network."""


@dataclass(frozen=True, kw_only=True)
class VanillaNetworkConfig(NeuralNetworkConfig):
    """Configuration for a standard feed-forward (vanilla) network.

    This extends the base `NeuralNetworkConfig` with options for skip
    connections and layer normalization.
    """

    use_skip_connections: bool = False
    """Whether to use skip connections (residual connections) between layers."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""


@dataclass(frozen=True, kw_only=True)
class MultiHeadConfig(NeuralNetworkConfig):
    """Configuration for a multi-head network.

    This is typically used in multi-task learning, where each task has its
    own output head.
    """

    num_tasks: int
    """The number of tasks, which determines the number of heads."""


@dataclass(frozen=True, kw_only=True)
class SoftModulesConfig(NeuralNetworkConfig):
    """Configuration for a Soft-Modules network.

    This architecture is designed for multi-task reinforcement learning,
    where a neural network learns to compose a set of shared modules
    in a task-specific manner.
    """

    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    width: int = 256
    """The number of neurons in the Dense layers around the network."""

    module_width: int = 256
    """The number of neurons in each module in the Base Policy Network.
    Corresponds to `d` in the original paper."""

    depth: int = 2
    """The number of Base Policy Network modules layers.
    (e.g., 2 for MT10, 4 for MT50 in the paper)."""

    num_modules: int = 2
    """The number of modules to use in each Base Policy Network layer.
    (e.g., 2 for MT10, 4 for MT50 in the paper)."""

    embedding_dim: int = 400
    """The dimension of the observation / task index embedding.
    Corresponds to `D` in the original paper."""


@dataclass(frozen=True, kw_only=True)
class PaCoConfig(NeuralNetworkConfig):
    """Configuration for a Parameter Composition (PaCo) network.

    PaCo learns a set of base parameter vectors and combines them
    linearly to generate task-specific network parameters.
    """

    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    num_parameter_sets: int = 5
    """The number of parameter sets. Corresponds to `K` in the paper."""


@dataclass(frozen=True, kw_only=True)
class CAREConfig(NeuralNetworkConfig):
    """Configuration for a CARE (Context-Aware Representation learning) network.

    CARE uses a context encoder to generate a task embedding, which is then
    used to modulate the policy network through a Mixture-of-Experts (MoE) layer.
    """

    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    encoder_width: int = 50
    """The width of the Context Encoder network."""

    encoder_depth: int = 2
    """The depth of the Context Encoder network."""

    encoder_temperature: float = 1.0
    """The temperature of the encoder attention."""

    embedding_dim: int = 50
    """The dimensionality of the context embedding."""

    num_experts: int = 6
    """The number of experts in CARE's task-dependent MoE observation encoder.
    (e.g., 6 for MT10, 10 for MT50 in the paper)."""


@dataclass(frozen=True, kw_only=True)
class FiLMConfig(NeuralNetworkConfig):
    """Configuration for a FiLM (Feature-wise Linear Modulation) network.

    FiLM uses a context encoder to generate parameters (gamma and beta)
    that modulate the activations of the main network on a per-feature basis.
    """

    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    encoder_width: int = 50
    """The width of the Context Encoder network."""

    encoder_depth: int = 2
    """The depth of the Context Encoder network."""

    embedding_dim: int = 50
    """The dimensionality of the context embedding."""


@dataclass(frozen=True, kw_only=True)
class MOOREConfig(NeuralNetworkConfig):
    """Configuration for a MOORE (Mixture of Orthogonal Experts) network.

    MOORE uses a set of experts with orthogonality constraints on their
    weights to encourage diversity and specialization.
    """

    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    num_experts: int = 4
    """The number of orthogonal experts.
    (e.g., 4 for MT10, 6 for MT50 in the original paper)."""
