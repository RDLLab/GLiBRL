from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
)
from metaworld_algorithms.config.nn import (
    VanillaNetworkConfig,
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import (
    VAEBasedMetaLearningTrainingConfig
)
from metaworld_algorithms.config.utils import Activation, CellType, Initializer, StdType
from metaworld_algorithms.envs import MetaworldMetaLearningConfig
from metaworld_algorithms.rl.algorithms import VariBadConfig
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
        run_name="ml10_varibad",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldMetaLearningConfig(
            env_id="ML10",
            meta_batch_size=meta_batch_size,
            evaluation_adaptation_steps=0,
            evaluation_adaptation_episodes=1,
            evaluation_num_episodes=3
        ),
        algorithm=VariBadConfig(
            num_tasks=meta_batch_size,
            meta_batch_size=meta_batch_size,
            policy_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    width=(128, 128),
                    activation=Activation.Tanh,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                ),
                log_std_min=np.log(1e-6),
                log_std_max=np.log(1),
                std_type=StdType.MLP_HEAD,
            ),
            clip_eps=0.1,
            n_vae_updates=3,
            num_epochs=2,
            num_gradient_steps=8
        ),
        training_config=VAEBasedMetaLearningTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            total_steps=int(2_000_000 * num_tasks),
            evaluation_frequency=args.evaluation_frequency,
            rollouts_per_task=2,
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
