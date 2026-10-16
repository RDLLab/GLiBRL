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
    BRLMetaLearningTrainingConfig,
)
from metaworld_algorithms.config.utils import Activation, Initializer, Optimizer, StdType
from metaworld_algorithms.envs import MetaworldMetaLearningConfig
from metaworld_algorithms.rl.algorithms import GLiBRLConfig
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
        run_name="ml10_glibrl",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldMetaLearningConfig(
            env_id="ML10",
            meta_batch_size=meta_batch_size,
            evaluation_adaptation_steps=0,
            evaluation_adaptation_episodes=1,
            evaluation_num_episodes=3
        ),
        algorithm=GLiBRLConfig(
            transition_latent_dim=16,
            reward_latent_dim=256,
            num_tasks=meta_batch_size,
            meta_batch_size=meta_batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            use_bias=False,
            entropy_coefficient=5e-3,
            policy_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    width=(256, 256),
                    activation=Activation.Tanh,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                    optimizer=OptimizerConfig(lr=5e-4, max_grad_norm=1, optimizer=Optimizer.Adam),
                ),
                log_std_min=np.log(1e-6),
                log_std_max=np.log(2),
                std_type=StdType.MLP_HEAD,
                squash_tanh=False,
                head_kernel_init=Initializer.XAVIER_UNIFORM,
                head_bias_init=Initializer.ZEROS,
            ),
            num_epochs=10,
            horizon=500,
            update_rate=1,
            normalize_advantages=False, 
            num_gradient_steps=20,
            dtype=np.float32,
            t_reg=5e-3,
            r_reg=1e-3,
            full_bayesian=True,
            noise_mode="diag",  
            normalise_task=True,
            quadratic_mt=True,
            use_task_encoder=False,
            task_encoder_dim=64,
        ),
        training_config=BRLMetaLearningTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            rollouts_per_task=10,
            total_steps=int(2_000_000 * num_tasks),
            evaluation_frequency=args.evaluation_frequency
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
