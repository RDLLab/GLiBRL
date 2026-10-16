from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optax
import tyro

from metaworld_algorithms.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig, ValueFunctionConfig
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import GLiBRLSACTrainingConfig
from metaworld_algorithms.config.utils import Activation, Initializer, Optimizer, StdType
from metaworld_algorithms.envs.mujoco_meta import HalfCheetahDirConfig
from metaworld_algorithms.rl.algorithms import GLiBRLSACConfig
from metaworld_algorithms.run import Run


@dataclass
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    meta_batch_size = 2
    rollout_steps = 200
    num_gradient_steps = 500

    actor_lr  = 3e-4
    critic_lr = 4e-4

    run = Run(
        run_name="halfcheetahdir_glibrlsac",
        seed=args.seed,
        data_dir=args.data_dir,
        env=HalfCheetahDirConfig(
            env_id="halfcheetah-dir",
            meta_batch_size=meta_batch_size,
            max_episode_steps=200,
            evaluation_adaptation_steps=0,
            evaluation_adaptation_episodes=10,
            evaluation_num_episodes=1,
        ),
        algorithm=GLiBRLSACConfig(
            initial_temperature=1,
            num_tasks=meta_batch_size,
            transition_latent_dim=2,
            reward_latent_dim=8,
            meta_batch_size=meta_batch_size,
            actor_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    width=(256, 256),
                    activation=Activation.ReLU,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                    optimizer=OptimizerConfig(
                        lr=actor_lr,
                        optimizer=Optimizer.Adam,
                        max_grad_norm=10
                    ),
                ),
                log_std_min=np.log(1e-6),
                log_std_max=np.log(2),
                std_type=StdType.MLP_HEAD,
                squash_tanh=True,
                head_kernel_init=Initializer.XAVIER_UNIFORM,
                head_bias_init=Initializer.ZEROS,
            ),
            critic_config=QValueFunctionConfig(
                network_config=VanillaNetworkConfig(
                    width=(256, 256),
                    activation=Activation.ReLU,
                    optimizer=OptimizerConfig(
                        lr=critic_lr,
                        optimizer=Optimizer.Adam,
                        max_grad_norm=10
                    ),
                ),
            ),
            reward_scale=1.0,
            t_reg=5e-2,
            r_reg=1e-2,
            full_bayesian=True,
            normalise_task=True,
            use_transition_in_policy=False,
            use_reward_in_policy=True,
            use_next_obs_in_r_mix=False,
            dtype=np.float32,
        ),
        training_config=GLiBRLSACTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            total_steps=int(1_000_000),
            buffer_size=int(2e5),
            evaluation_frequency=200_000,
            warmstart_steps=1000,
            rollout_steps=rollout_steps,
            batch_size=128,
            policy_batch_size_range=(1, 200),
            num_model_steps=1,
            num_gradient_steps=num_gradient_steps,
            resample_tasks_when_full=False
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
