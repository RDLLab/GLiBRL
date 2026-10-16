from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from metaworld_algorithms.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import GLiBRLSACTrainingConfig
from metaworld_algorithms.config.utils import Activation, Initializer, Optimizer, StdType
from metaworld_algorithms.envs.mujoco_meta import HalfCheetahVelConfig
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

    num_env_tasks = 20   # all training tasks stepped every rollout
    meta_batch_size = 10 # tasks sampled per gradient step

    run = Run(
        run_name="halfcheetahvel_glibrlsac",
        seed=args.seed,
        data_dir=args.data_dir,
        env=HalfCheetahVelConfig(
            env_id="halfcheetah-vel",
            meta_batch_size=num_env_tasks,  # env config drives how many envs are spawned
            num_train_tasks=num_env_tasks,
            num_eval_tasks=30,
            vel_range_train=(0.0, 3.0),
            vel_range_test=(0.0, 3.0),
            max_episode_steps=200,
            evaluation_adaptation_steps=0,
            evaluation_adaptation_episodes=10,
            evaluation_num_episodes=1,
        ),
        algorithm=GLiBRLSACConfig(
            initial_temperature=0.2,
            num_tasks=meta_batch_size,       # gradient-step task batch size
            num_env_tasks=num_env_tasks,     # total training envs in buffer/rollout
            transition_latent_dim=2,
            reward_latent_dim=8,
            meta_batch_size=meta_batch_size,
            actor_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    width=(256, 256),
                    activation=Activation.ReLU,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                    optimizer=OptimizerConfig(lr=5e-4, optimizer=Optimizer.Adam, max_grad_norm=1.0),
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
                    optimizer=OptimizerConfig(lr=5e-4, optimizer=Optimizer.Adam, max_grad_norm=1.0),
                ),
            ),
            t_reg=5e-2,
            r_reg=1e-2,
            full_bayesian=True,
            normalise_task=True,
            use_transition_in_policy=False,
            use_reward_in_policy=True,
            use_next_obs_in_r_mix=False,
            dtype=np.float32,
            reward_scale=1,
            tau=5e-3
        ),
        training_config=GLiBRLSACTrainingConfig(
            meta_batch_size=num_env_tasks,  # controls env spawn count via base.py
            warmstart_steps=500,
            rollout_steps=200,
            buffer_size=int(1e5),
            evaluate_on_train=False,
            total_steps=int(1_000_000),
            evaluation_frequency=100_000,
            num_gradient_steps=2000,
            batch_size=128,
            policy_batch_size_range=(1, 200),
            num_model_steps=50,
            resample_tasks_when_full=False,  # fixed tasks, never resample
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

