"""Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/infra/experiment.py"""

import gc
import pathlib
import random
import time
from dataclasses import dataclass

import jax
import numpy as np
import orbax.checkpoint as ocp
import wandb

from metaworld_algorithms.checkpoint import (
    Checkpoint,
    get_checkpoint_restore_args,
    get_last_agent_checkpoint_save_args,
    get_metadata_only_restore_args,
    load_env_checkpoints,
    get_checkpoint_save_args
)
from metaworld_algorithms.config.envs import EnvConfig, MetaLearningEnvConfig
from metaworld_algorithms.config.rl import (
    AlgorithmConfig,
    OffPolicyTrainingConfig,
    TrainingConfig,
)
from metaworld_algorithms.rl.algorithms import (
    Algorithm,
    OffPolicyAlgorithm,
    get_algorithm_for_config,
)
from metaworld_algorithms.rl.algorithms.base import (
    MetaLearningAlgorithm,
    BRLMetaLearningAlgorithm,
    OnPolicyAlgorithm,
)
from metaworld_algorithms.types import CheckpointMetadata


@dataclass
class Run:
    """
    Orchestrates a meta-reinforcement learning experiment.

    This class is responsible for the entire lifecycle of a run, including:
    - Setting up the environment and the learning algorithm from configuration objects.
    - Managing checkpointing (saving and resuming).
    - Launching the training process.
    - Handling final evaluation and logging of results.

    Attributes:
        run_name (str): The name of the run, used for logging and directory naming.
        seed (int): The random seed for reproducibility.
        data_dir (pathlib.Path): The base directory where experiment data (checkpoints, logs) will be stored.
        env (EnvConfig): A configuration object that defines the environment for the experiment.
        algorithm (AlgorithmConfig): A configuration object that defines the learning algorithm and its hyperparameters.
        training_config (TrainingConfig): A configuration object that defines training parameters like total steps, batch sizes, etc.
        checkpoint (bool): If True, enables checkpointing of the agent and training state. Defaults to True.
        max_checkpoints_to_keep (int): The maximum number of checkpoints to keep. Defaults to 10.
        best_checkpoint_metric (str): The metric used to determine the "best" checkpoint.
            Defaults to "mean_success_rate_adapt_0".
        resume (bool): If True, attempts to resume training from the latest checkpoint. Defaults to False.
        resume_idx (int): The index of the checkpoint to resume from in the list of all checkpoints. Defaults to -1 (the latest).
    """
    run_name: str
    seed: int
    data_dir: pathlib.Path

    env: EnvConfig
    algorithm: AlgorithmConfig
    training_config: TrainingConfig

    checkpoint: bool = True
    max_checkpoints_to_keep: int = 20
    best_checkpoint_metric: str = "mean_success_rate_adapt_0"
    resume: bool = False
    resume_idx: int = -1

    def __post_init__(self) -> None:
        """
        This method is called after the dataclass is initialized. 
        It sets up internal state, such as disabling wandb by default and generating a timestamp for the run.
        """
        self._wandb_enabled = False
        self._wandb_run_id: str | None = None
        self._timestamp = str(int(time.time()))

    def _get_data_dir(self) -> pathlib.Path:
        return self.data_dir / f"{self.run_name}_{self.seed}"

    def _get_latest_checkpoint_metadata(self) -> CheckpointMetadata | None:
        checkpoint_manager = ocp.CheckpointManager(
            pathlib.Path(self._get_data_dir() / "checkpoints").absolute(),
            item_names=("metadata",),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=self.max_checkpoints_to_keep,
                create=True,
                best_fn=lambda x: x[self.best_checkpoint_metric],
                enable_async_checkpointing=False
            ),
        )
        if checkpoint_manager.latest_step() is not None:
            ckpt: Checkpoint = checkpoint_manager.restore(  # pyright: ignore [reportAssignmentType]
                checkpoint_manager.latest_step(),
                args=get_metadata_only_restore_args(),
            )
            return ckpt["metadata"]
        else:
            return None

    def enable_wandb(self, **wandb_kwargs) -> None:
        """
        Enables logging to Weights & Biases. It constructs a unique `run_id` for wandb based on the run name, seed, and timestamp. 
        If resume is True and a checkpoint is found, it reuses the `run_id` from the previous run to ensure continuity in logging.
        
        :param wandb_kwargs: Keyword arguments passed directly to the wandb.init() function.
        """
        self._wandb_enabled = True

        latest_ckpt_metadata = self._get_latest_checkpoint_metadata()
        if latest_ckpt_metadata is not None and self.resume:
            existing_run_timestamp = latest_ckpt_metadata.get("timestamp")
            if not existing_run_timestamp:
                print(
                    "WARNING: Resume is on, a checkpoint was found, but there's no timestamp in the checkpoint."
                )
                run_id = f"{self.run_name}_{self.seed}"
            else:
                run_id = f"{existing_run_timestamp}_{self.run_name}_{self.seed}"
        else:
            run_id = f"{self._timestamp}_{self.run_name}_{self.seed}"

        self._wandb_run_id = run_id
        print(wandb_kwargs)
        wandb.init(
            dir=str(self._get_data_dir()), id=run_id, name=self.run_name, **wandb_kwargs
        )

    def start(self) -> None:
        """
        This is the main entry point of the Run class. It executes the entire experiment workflow:
        1. **Device Check**: It first checks for the presence of a GPU or TPU and raises a `RuntimeEError` if no accelerator is found.
        2. **Environment Setup**: It calls the spawn() method of the env configuration object to create the training environment.
        3. **Algorithm Initialization**: It uses the `get_algorithm_for_config` factory function to get the appropriate algorithm class based on the `algorithm` configuration object. 
        It then calls the `initialize()` static method of the algorithm class to create an instance of the agent.
        4. **Checkpoint Handling**:
            - If `checkpoint` is `True`, it sets up an `orbax.checkpoint.CheckpointManager` to handle saving and restoring checkpoints. 
            The items to be checkpointed include the agent's state, environment states, RNG states, and metadata. For off-policy algorithms, the replay buffer is also included.
            - If `resume` is `True`, it restores the latest checkpoint, including the agent, environment states, RNG states, and replay buffer.
        5. **Training**:  It calls the `train()` method of the algorithm instance, passing all the necessary configurations, environments, and checkpoint-related objects. 
        The train() method is a generator that yields metrics at each logging step.
        6. **Final Evaluation & Cleanup**: 
            - After training is complete, the environments are closed.
            - If the experiment is a meta-learning setup, it spawns a separate set of test environments.
            - It then runs a final evaluation on the trained agent. For meta-learning, this involves evaluating the agent's performance after a certain number of adaptation steps.
            - The final evaluation metrics are logged.
        7. **Checkpoint Saving**:
            - A final checkpoint of the trained agent is saved.
            - If `wandb` is enabled, the final and best model checkpoints are logged as artifacts to `wandb`.
        8. **Wandb Finish**: If `wandb` is enabled, `wandb.finish()` is called to signify the end of the run.
        """
        # Abort if no hardware accelerator is found.
        if jax.device_count("gpu") < 1 and jax.device_count("tpu") < 1:
            raise RuntimeError(
                "No accelerator found, aborting. Devices: %s" % jax.devices()
            )

        # Create the vectorized environments for training.
        envs = self.env.spawn(seed=self.seed)

        # Initialize the reinforcement learning algorithm.
        algorithm_cls = get_algorithm_for_config(self.algorithm) 
        algorithm: Algorithm
        algorithm = algorithm_cls.initialize(self.algorithm, self.env, seed=self.seed)
        is_off_policy = isinstance(algorithm, OffPolicyAlgorithm)

        # Prepare for checkpointing.
        buffer_checkpoint = None
        checkpoint_manager = None
        checkpoint_metadata = None
        envs_checkpoint = None

        # Seed random number generators for reproducibility.
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Set up the CheckpointManager if checkpointing is enabled.
        if self.checkpoint:
            checkpoint_items = (
                "agent",
                "env_states",
                "rngs",
                "metadata",
            )
            # Off-policy algorithms require saving the replay buffer as well.
            if is_off_policy:
                checkpoint_items += ("buffer",)

            checkpoint_manager = ocp.CheckpointManager(
                pathlib.Path(self._get_data_dir() / "checkpoints").absolute(),
                item_names=checkpoint_items,
                options=ocp.CheckpointManagerOptions(
                    enable_async_checkpointing=False,
                    max_to_keep=self.max_checkpoints_to_keep,
                    create=True,
                    best_fn=lambda x: x[self.best_checkpoint_metric],
                ),
            )

            # If resuming, restore the state from the specified checkpoint.
            if self.resume and checkpoint_manager.all_steps()[self.resume_idx] is not None:
                if is_off_policy:
                    assert isinstance(self.training_config, OffPolicyTrainingConfig)
                    rb = algorithm.spawn_replay_buffer(
                        self.env,
                        self.training_config,
                    )
                else:
                    rb = None
                # Restore the checkpoint from the CheckpointManager.
                ckpt: Checkpoint = checkpoint_manager.restore(  # pyright: ignore [reportAssignmentType]
                    checkpoint_manager.all_steps()[self.resume_idx],
                    args=get_checkpoint_restore_args(algorithm, rb),
                )
                algorithm = ckpt["agent"]

                if is_off_policy:
                    buffer_checkpoint = ckpt["buffer"]  # pyright: ignore [reportTypedDictNotRequiredAccess]

                # Load the environment and RNG states.
                envs_checkpoint = ckpt["env_states"]
                load_env_checkpoints(envs, envs_checkpoint)

                random.setstate(ckpt["rngs"]["python_rng_state"])
                np.random.set_state(ckpt["rngs"]["global_numpy_rng_state"])

                checkpoint_metadata: CheckpointMetadata | None = ckpt["metadata"]
                assert checkpoint_metadata is not None

                self._timestamp = checkpoint_metadata.get("timestamp", self._timestamp)

                print(f"Loaded checkpoint at step {checkpoint_metadata['step']}")

        # Track the number of parameters in the model with wandb.
        if self._wandb_enabled:
            wandb.config.update(algorithm.get_num_params())

        # Start the training process.
        agent = algorithm.train(
            config=self.training_config,
            envs=envs,
            env_config=self.env,
            run_timestamp=self._timestamp,
            seed=self.seed,
            track=self._wandb_enabled,
            checkpoint_manager=checkpoint_manager,
            checkpoint_metadata=checkpoint_metadata,
            buffer_checkpoint=buffer_checkpoint,
        )

        # Cleanup and final evaluation after training.
        if self.checkpoint:
            if isinstance(
                agent, (OnPolicyAlgorithm, OffPolicyAlgorithm)
            ) and not isinstance(self.env, MetaLearningEnvConfig):
                mean_success_rate, mean_returns, mean_success_per_task = (
                    self.env.evaluate(envs, agent)
                )



            # For meta-learning, run a specific evaluation protocol.
            if isinstance(agent, MetaLearningAlgorithm) and isinstance(
                self.env, MetaLearningEnvConfig
            ):
                gc.collect()
               
                eval_envs = self.env.spawn_test(self.seed)

                mean_success_rate_dict, mean_returns_dict, mean_success_per_task_dict, \
                   mean_t_loss, mean_r_loss, mean_t_loss_per_task, mean_r_loss_per_task  = (
                   self.env.evaluate_metalearning(eval_envs, agent.wrap())
                )

                eval_envs.close()
                del eval_envs

            else:
                raise ValueError("Invalid agent / env combination.")
            
            # Collate final metrics from the evaluation.
            # final_metrics_0 = {
            #     f"charts/mean_success_rate_adapt_{step}_0.0": mean_success_rate_dict_0[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/mean_evaluation_return_adapt_{step}_0.0": mean_returns_dict_0[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            # } | {
            #     f"charts/{task_name}_success_rate_adapt_{step}_0.0": float(success_rate) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, success_rate in mean_success_per_task_dict_0[step].items()
            # } | {
            #     f"charts/mean_t_loss_{step}_0.0": mean_t_loss_0[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/mean_r_loss_{step}_0.0": mean_r_loss_0[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/{task_name}_t_loss_{step}_0.0": float(loss) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, loss in mean_t_loss_per_task_0[step].items()
            # } | {
            #     f"charts/{task_name}_r_loss_{step}_0.0": float(loss) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, loss in mean_r_loss_per_task_0[step].items()
            # }


            # final_metrics_1 = {
            #     f"charts/mean_success_rate_adapt_{step}_0.1": mean_success_rate_dict_1[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/mean_evaluation_return_adapt_{step}_0.1": mean_returns_dict_1[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            # } | {
            #     f"charts/{task_name}_success_rate_adapt_{step}_0.1": float(success_rate) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, success_rate in mean_success_per_task_dict_1[step].items()
            # } | {
            #     f"charts/mean_t_loss_{step}_0.1": mean_t_loss_1[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/mean_r_loss_{step}_0.1": mean_r_loss_1[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/{task_name}_t_loss_{step}_0.1": float(loss) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, loss in mean_t_loss_per_task_1[step].items()
            # } | {
            #     f"charts/{task_name}_r_loss_{step}_0.1": float(loss) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, loss in mean_r_loss_per_task_1[step].items()
            # }



            # final_metrics_5 = {
            #     f"charts/mean_success_rate_adapt_{step}_0.5": mean_success_rate_dict_5[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/mean_evaluation_return_adapt_{step}_0.5": mean_returns_dict_5[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            # } | {
            #     f"charts/{task_name}_success_rate_adapt_{step}_0.5": float(success_rate) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, success_rate in mean_success_per_task_dict_5[step].items()
            # } | {
            #     f"charts/mean_t_loss_{step}_0.5": mean_t_loss_5[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/mean_r_loss_{step}_0.5": mean_r_loss_5[step] 
            #     for step in range(self.env.evaluation_adaptation_steps + 1) 
            # } | {
            #     f"charts/{task_name}_t_loss_{step}_0.5": float(loss) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, loss in mean_t_loss_per_task_5[step].items()
            # } | {
            #     f"charts/{task_name}_r_loss_{step}_0.5": float(loss) 
            #     for step in range(self.env.evaluation_adaptation_steps + 1)
            #     for task_name, loss in mean_r_loss_per_task_5[step].items()
            # }


            final_metrics = {
                    f"charts/mean_success_rate_adapt_{step}": mean_success_rate_dict[step] 
                    for step in range(self.env.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_evaluation_return_adapt_{step}": mean_returns_dict[step] 
                    for step in range(self.env.evaluation_adaptation_steps + 1)
                } | {
                    f"charts/{task_name}_success_rate_adapt_{step}": float(success_rate) 
                    for step in range(self.env.evaluation_adaptation_steps + 1)
                    for task_name, success_rate in mean_success_per_task_dict[step].items()
                } | {
                    f"charts/mean_t_loss_{step}": mean_t_loss[step] 
                    for step in range(self.env.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/mean_r_loss_{step}": mean_r_loss[step] 
                    for step in range(self.env.evaluation_adaptation_steps + 1) 
                } | {
                    f"charts/{task_name}_t_loss_{step}": float(loss) 
                    for step in range(self.env.evaluation_adaptation_steps + 1)
                    for task_name, loss in mean_t_loss_per_task[step].items()
                } | {
                    f"charts/{task_name}_r_loss_{step}": float(loss) 
                    for step in range(self.env.evaluation_adaptation_steps + 1)
                    for task_name, loss in mean_r_loss_per_task[step].items()
                }
 

            # final = final_metrics_0 | final_metrics_1 | final_metrics_5 | final_metrics
            # final = final_metrics_0 | final_metrics_1 | final_metrics
            # final = final_metrics_0 | final_metrics_5 | final_metrics

            final = final_metrics
            if not self.resume:
                assert checkpoint_manager is not None
                checkpoint_manager.wait_until_finished()

                # checkpoint_manager.save(
                #     self.training_config.total_steps,
                #     args=get_checkpoint_save_args(agent, final),
                #     metrics={
                #         k.removeprefix("charts/"): v for k, v in final.items()
                #     },
                # )
                # checkpoint_manager.wait_until_finished()

                # Save the final agent state and metrics.
                if checkpoint_manager._options.max_to_keep is not None:
                    checkpoint_manager._options.max_to_keep += 1
                    
                checkpoint_manager.save(
                    self.training_config.total_steps + 1,
                    # args=get_last_agent_checkpoint_save_args(agent, final),
                    args=get_checkpoint_save_args(
                        agent,
                        envs,
                        self.training_config.total_steps + 1,
                        0,
                        self._timestamp
                    ),
                    metrics={
                        k.removeprefix("charts/"): float(v) for k, v in final.items()
                    },
                )
                checkpoint_manager.wait_until_finished()

                # Log the final model checkpoint as a wandb artifact.
                if self._wandb_enabled:
                    assert wandb.run is not None
                    wandb.log(final, step=self.training_config.total_steps + 1)
                    wandb.log(final_metrics, step=self.training_config.total_steps + 1)
                    final_ckpt_artifact = wandb.Artifact(
                        f"{wandb.run.id}_final_agent_checkpoint", type="model"
                    )
                    final_ckpt_dir = checkpoint_manager._get_save_directory(
                        self.training_config.total_steps + 1, checkpoint_manager.directory
                    )
                    final_ckpt_artifact.add_dir(str(final_ckpt_dir))
                    wandb.log_artifact(final_ckpt_artifact)

                    # Log the best model checkpoint as a wandb artifact.
                    best_step = checkpoint_manager.best_step()
                    assert best_step is not None
                    best_ckpt_artifact = wandb.Artifact(
                        f"{wandb.run.id}_best_agent_checkpoint", type="model"
                    )
                    best_ckpt_dir = checkpoint_manager._get_save_directory(
                        best_step, checkpoint_manager.directory
                    )
                    best_ckpt_artifact.add_dir(str(best_ckpt_dir))
                    wandb.log_artifact(best_ckpt_artifact)

                checkpoint_manager.close()

            envs.close()
            del envs
        # Finalize the wandb run.
        if self._wandb_enabled:
            wandb.finish()

        return agent
