from dataclasses import dataclass

import jax
import optax

from metaworld_algorithms.optim.dummy import dummy_multitask_optimizer
from metaworld_algorithms.optim.gradnorm import gradnorm
from metaworld_algorithms.optim.pcgrad import pcgrad

from optax import Schedule

from .utils import Optimizer


@dataclass(frozen=True, kw_only=True)
class OptimizerConfig:
    """Base configuration for an optimizer.

    This dataclass defines the settings for an optax optimizer, including
    the learning rate, optimizer type, and gradient clipping.
    """

    lr: float | Schedule = 3e-4
    """The learning rate or a learning rate schedule."""

    optimizer: Optimizer = Optimizer.Adam
    """The optimizer to use (e.g., Adam, AdamW)."""

    max_grad_norm: float | None = None
    """The maximum norm for gradient clipping. If None, no clipping is applied."""

    @property
    def requires_split_task_losses(self) -> bool:
        """Indicates whether the optimizer requires per-task losses.
        For standard optimizers, this is False. For multi-task optimizers
        like PCGrad or GradNorm, this will be True.
        """
        return False

    def spawn(self) -> optax.GradientTransformation:
        """Creates and returns an optax optimizer instance based on the config.

        Returns:
            An optax.GradientTransformation object.
        """
        # From https://github.com/araffin/sbx/blob/master/sbx/ppo/policies.py#L120
        optim_kwargs = {}
        if self.optimizer == Optimizer.Adam:
            optim_kwargs["eps"] = 1e-5
            optim = self.optimizer(learning_rate=self.lr, **optim_kwargs)

        elif self.optimizer == Optimizer.AdamW:
            # optim_kwargs["eps"] = 1e-5
            optim_kwargs["weight_decay"] = 1e-4
            optim = self.optimizer(learning_rate=self.lr, **optim_kwargs)

        else:
            optim = self.optimizer(learning_rate=self.lr, **optim_kwargs)

        if self.max_grad_norm is not None:
            optim = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optim,
            )
        return optim


@dataclass(frozen=True, kw_only=True)
class DummyMultiTaskConfig(OptimizerConfig):
    """A dummy multi-task optimizer that simply sums the gradients from all tasks.

    This is equivalent to standard training on the summed loss.
    """

    @property
    def requires_split_task_losses(self) -> bool:
        """This optimizer requires per-task losses to sum them."""
        return True

    def spawn(self) -> optax.GradientTransformation:
        """Creates the optimizer chain, including the dummy multi-task step."""
        return optax.chain(
            dummy_multitask_optimizer(),
            super().spawn(),
        )


@dataclass(frozen=True, kw_only=True)
class PCGradConfig(OptimizerConfig):
    """Configuration for the PCGrad (Projecting Conflicting Gradients) optimizer.

    PCGrad mitigates negative interference between task gradients by projecting
    any conflicting gradients onto the normal plane of other task gradients.
    """

    num_tasks: int
    """The number of tasks for PCGrad to handle."""
    cosine_sim_logs: bool = False
    """Whether to log the cosine similarity between task gradients."""

    @property
    def requires_split_task_losses(self) -> bool:
        """PCGrad needs per-task losses to compute per-task gradients."""
        return True

    def spawn(self) -> optax.GradientTransformationExtraArgs:
        """Creates the optimizer chain, including the PCGrad transformation."""
        return optax.chain(
            pcgrad(num_tasks=self.num_tasks, cosine_sim_logs=self.cosine_sim_logs),
            super().spawn(),
        )


@dataclass(frozen=True, kw_only=True)
class GradNormConfig(OptimizerConfig):
    """Configuration for the GradNorm optimizer.

    GradNorm dynamically balances training in multi-task learning by normalizing
    the gradient magnitudes of different tasks. It learns a weight for each
    task's loss function automatically.
    """

    num_tasks: int
    """The number of tasks for GradNorm to handle."""
    gradnorm_optimizer: OptimizerConfig
    """The optimizer for learning the GradNorm task weights."""
    initial_weights: jax.Array | None = None
    """The initial weights for each task's loss. If None, they are initialized
    to 1.0 for each task."""
    asymmetry: float = 0.12
    """A hyperparameter from the GradNorm paper that controls the relative
    learning rates of tasks."""

    @property
    def requires_split_task_losses(self) -> bool:
        """GradNorm needs per-task losses to compute per-task gradients and
        dynamically adjust task weights."""
        return True

    def spawn(self) -> optax.GradientTransformation:
        """Creates the optimizer chain, including the GradNorm transformation."""
        return optax.chain(
            gradnorm(
                optim=self.gradnorm_optimizer,
                num_tasks=self.num_tasks,
                asymmetry=self.asymmetry,
                initial_weights=self.initial_weights,
            ),
            super().spawn(),
        )
