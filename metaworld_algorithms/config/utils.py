import enum

import flax.linen
import jax
import optax


def _uniform_init(bound: float) -> jax.nn.initializers.Initializer:
    import metaworld_algorithms.nn.initializers

    return metaworld_algorithms.nn.initializers.uniform(bound)


class Initializer(enum.Enum):
    """An enumeration of JAX neural network weight initializers.

    This class provides a configurable way to select different weight
    initialization strategies for neural networks.
    """

    ZEROS = enum.member(lambda: jax.nn.initializers.zeros)  # noqa: E731
    HE_NORMAL = enum.member(jax.nn.initializers.he_normal)
    HE_UNIFORM = enum.member(jax.nn.initializers.he_uniform)
    XAVIER_NORMAL = enum.member(jax.nn.initializers.xavier_normal)
    XAVIER_UNIFORM = enum.member(jax.nn.initializers.xavier_uniform)
    CONSTANT = enum.member(jax.nn.initializers.constant)
    UNIFORM = enum.member(_uniform_init)
    ORTHOGONAL = enum.member(jax.nn.initializers.orthogonal)

    def __call__(self, *args, **kwargs):
        """Instantiates the selected initializer.

        Allows the enum member to be called like a function, returning the
        underlying JAX initializer.
        """
        return self.value(*args, **kwargs)


class Activation(enum.Enum):
    """An enumeration of JAX/Flax activation functions.

    This class provides a configurable way to select different activation
    functions for neural networks.
    """

    ReLU = enum.member(jax.nn.relu)
    Tanh = enum.member(jax.nn.tanh)
    Sin = enum.member(jax.numpy.sin)
    LeakyReLU = enum.member(jax.nn.leaky_relu)
    PReLU = enum.member(lambda x: flax.linen.PReLU()(x))  # noqa: E731
    ReLU6 = enum.member(jax.nn.relu6)
    SiLU = enum.member(jax.nn.silu)
    GELU = enum.member(jax.nn.gelu)
    GLU = enum.member(jax.nn.glu)
    ELU = enum.member(jax.nn.elu)
    SELU = enum.member(jax.nn.selu)
    Identity = enum.member(lambda x: x)

    def __call__(self, *args, **kwargs):
        """Applies the selected activation function.

        Allows the enum member to be called like a function, applying the
        underlying JAX/Flax activation.
        """
        return self.value(*args, **kwargs)


class Optimizer(enum.Enum):
    """An enumeration of Optax optimizers.

    This class provides a configurable way to select different optimizers
    for training neural networks.
    """

    Adam = enum.member(optax.adam)
    AdamW = enum.member(optax.adamw)
    RMSProp = enum.member(optax.rmsprop)
    SGD = enum.member(optax.sgd)

    def __call__(self, learning_rate: float, **kwargs):
        """Instantiates the selected optimizer.

        Allows the enum member to be called like a function, returning an
        initialized Optax optimizer.
        """
        return self.value(learning_rate, **kwargs)


class StdType(enum.Enum):
    """Defines the method for parametrizing the standard deviation in a stochastic policy."""

    MLP_HEAD = enum.auto()
    """The standard deviation is state-dependent, produced by a separate head of the main network."""
    PARAM = enum.auto()
    """The standard deviation is a state-independent, learnable parameter for each action dimension."""


class CellType(enum.Enum):
    """An enumeration of Flax recurrent neural network (RNN) cells.

    This class provides a configurable way to select different types of
    recurrent cells (e.g., GRU, LSTM).
    """

    RNN = enum.member(flax.linen.RNNCellBase)
    LSTM = enum.member(flax.linen.OptimizedLSTMCell)
    GRU = enum.member(flax.linen.GRUCell)

    def __call__(self, *args, **kwargs):
        """Instantiates the selected RNN cell.

        Allows the enum member to be called like a function, returning an
        initialized Flax RNN cell.
        """
        return self.value(*args, **kwargs)
