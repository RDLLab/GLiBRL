"""
RL^2 on Metaworld ML10 Benchmark.

This script trains an RL^2 (Reinforcement Learning with Recurrent Neural Networks) agent
on the Metaworld ML10 benchmark environment. ML10 consists of 10 meta-training tasks
and 5 meta-testing tasks.

The RL^2 algorithm treats the meta-learning problem as a standard RL problem by
using a recurrent policy (GRU-based) that receives the previous action, reward,
and termination signal as part of its observation. This allows the policy to "learn to learn"
within the recurrent dynamics over multiple episodes (trials).

Key Features:
- **Algorithm**: RL^2 (PPO-based optimization of a recurrent policy).
- **Environment**: Metaworld ML10 (v2).
- **Network**: A GRU-based recurrent policy with a tanh activation.
- **Training**: Meta-training loop with periodic evaluation on test tasks.

Training Setup:
- **Total Steps**: 20,000,000 environment steps.
- **Meta-Batch Size**: 20 tasks sampled per iteration.
- **Rollouts per Task**: 10 episodes per task in the meta-batch.
- **Episode Length**: 500 steps.
- **Steps per Iteration**: 100,000 steps (20 tasks * 10 episodes * 500 steps).
- **Optimization**: PPO with 10 epochs per batch, using BPTT with chunk length 250.

Evaluation Setup:
- **Frequency**: Every 1,000,000 steps.
- **Test Tasks**: Evaluated on the 5 held-out test tasks of ML10.
- **Episodes**: 3 evaluation episodes per task.

Usage:
    python examples/meta_learning/rl2_ml10.py [OPTIONS]

Options (via `tyro`):
    --seed INT                  Random seed (default: 1).
    --track                     Enable WandB tracking (default: False).
    --wandb-project STR         WandB project name.
    --wandb-entity STR          WandB entity/username.
    --data-dir PATH             Directory to save results (default: ./run_results).
    --resume                    Resume from the latest checkpoint (default: False).
    --evaluation-frequency INT  Steps between evaluations (default: 1,000,000).

Example:
    # Run with default settings
    python examples/meta_learning/rl2_ml10.py

    # Run with tracking enabled
    python examples/meta_learning/rl2_ml10.py --track --wandb-project my-project --wandb-entity my-user
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from metaworld_algorithms.config.networks import (
    RecurrentContinuousActionPolicyConfig,
)
from metaworld_algorithms.config.nn import (
    RecurrentNeuralNetworkConfig,
    VanillaNetworkConfig,
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import (
    RNNBasedMetaLearningTrainingConfig,
)
from metaworld_algorithms.config.utils import Activation, CellType, Initializer, StdType
from metaworld_algorithms.envs import MetaworldMetaLearningConfig
from metaworld_algorithms.rl.algorithms import RL2Config
from metaworld_algorithms.run import Run


@dataclass
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False
    evaluation_frequency: int = 1_000_000


def main() -> None:
    args = tyro.cli(Args)

    meta_batch_size = 20
    num_tasks = 10

    run = Run(
        run_name="ml10_rl2",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldMetaLearningConfig(
            env_id="ML10",
            meta_batch_size=meta_batch_size,
            recurrent_info_in_obs=True,
            evaluation_adaptation_steps=0,
            evaluation_adaptation_episodes=1,
            evaluation_num_episodes=4
        ),
        algorithm=RL2Config(
            num_tasks=meta_batch_size,
            meta_batch_size=meta_batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            policy_config=RecurrentContinuousActionPolicyConfig(
                encoder_config=VanillaNetworkConfig(
                    width=(256, 256),
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                    activation=Activation.Tanh,
                ),
                network_config=RecurrentNeuralNetworkConfig(
                    width=256,
                    cell_type=CellType.GRU,
                    activation=Activation.Tanh,
                    recurrent_kernel_init=Initializer.ORTHOGONAL,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                    optimizer=OptimizerConfig(lr=5e-4, max_grad_norm=1.0),
                ),
                log_std_min=np.log(1e-6),
                log_std_max=None,
                std_type=StdType.MLP_HEAD,
                squash_tanh=False,
                head_kernel_init=Initializer.XAVIER_UNIFORM,
                head_bias_init=Initializer.ZEROS,
                activate_head=True,
            ),
            num_epochs=10,
            chunk_len=250,
            normalize_advantages=False,
        ),
        training_config=RNNBasedMetaLearningTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            total_steps=int(2_000_000 * num_tasks),
            evaluation_frequency=args.evaluation_frequency,
            rollouts_per_task=10,
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        run.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=run,
            resume="allow",
        )

    run.start()


if __name__ == "__main__":
    main()
