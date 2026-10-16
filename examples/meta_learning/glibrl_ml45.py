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
from metaworld_algorithms.config.utils import Activation, Initializer, StdType, Optimizer
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
    enable_planner: bool = False
    evaluation_frequency: int = 4_500_000


def main() -> None:
    args = tyro.cli(Args)

    meta_batch_size = 45
    num_tasks = 45
    
    p_config = ContinuousActionPolicyConfig(
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
    )

    run = Run(
        run_name="ml45_glibrl",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldMetaLearningConfig(
            env_id="ML45",
            meta_batch_size=meta_batch_size,
            total_goals_per_task_test=45,
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
            policy_config=p_config,
            num_epochs=10,
            horizon=500,
            update_rate=1,
            normalize_advantages=False, 
            num_gradient_steps=20,
            dtype=np.float32,
            t_reg=5e-3,
            r_reg=1e-3,
            full_bayesian=True,
            noise_mode="full",  
            normalise_task=True
        ),
        training_config=BRLMetaLearningTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            rollouts_per_task=10,
            total_steps=int(1_200_000 * num_tasks),
            evaluation_frequency=args.evaluation_frequency,
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
