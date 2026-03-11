"""Extended AgentTrainer with prediction auxiliary loss support.

This provides a drop-in replacement for rllm.trainer.agent_trainer.AgentTrainer
that uses PredictiveAgentWorkflowTrainer internally.
"""

from typing import Any

import ray

from rllm.data import Dataset
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
from rllm.trainer.verl.train_agent_ppo import TaskRunner


class PredictiveAgentTrainer:
    """
    A wrapper class that extends AgentTrainer with prediction auxiliary loss.

    This is a drop-in replacement for AgentTrainer that:
    1. Uses PredictiveAgentWorkflowTrainer internally when workflow is enabled
    2. Maintains the same API as AgentTrainer
    3. Adds prediction loss support via config

    Usage:
        ```python
        from rllm_ext.training import PredictiveAgentTrainer

        trainer = PredictiveAgentTrainer(
            workflow_class=PredictiveToolWorkflow,
            workflow_args={...},
            config=config,
            train_dataset=train_data,
            val_dataset=val_data,
        )
        trainer.train()
        ```
    """

    def __init__(
        self,
        workflow_class: type | None = None,
        workflow_args: dict[str, Any] | None = None,
        agent_class: type | None = None,
        env_class: type | None = None,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        config: dict[str, Any] | list[str] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
    ):
        """
        Initialize the PredictiveAgentTrainer.

        Args:
            Same as AgentTrainer, but uses PredictiveAgentWorkflowTrainer internally.
        """
        if workflow_class is not None and config.rllm.workflow.use_workflow:
            if agent_class is not None:
                raise ValueError("agent_class is not supported when using workflow")
            if agent_args is not None:
                raise ValueError("agent_args is not supported when using workflow")
            if env_class is not None:
                raise ValueError("env_class is not supported when using workflow")
            if env_args is not None:
                raise ValueError("env_args is not supported when using workflow")

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}

        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

        self.config = config

        if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if val_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.val_files = val_dataset.get_verl_data_path()

    def train(self):
        """Run training with PredictiveAgentWorkflowTrainer."""
        # Check if Ray is not initialized
        if not ray.is_initialized():
            # read off all the `ray_init` settings from the config
            if self.config is not None and hasattr(self.config, "ray_init"):
                ray_init_settings = {k: v for k, v in self.config.ray_init.items() if v is not None}
            else:
                ray_init_settings = {}
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

        # Use our custom TaskRunner that uses PredictiveAgentWorkflowTrainer
        runner = PredictiveTaskRunner.remote()

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                agent_class=self.agent_class,
                env_class=self.env_class,
                agent_args=self.agent_args,
                env_args=self.env_args,
            )
        )


@ray.remote(num_cpus=1)
class PredictiveTaskRunner:
    """
    Extended TaskRunner that uses PredictiveAgentWorkflowTrainer.

    This is a copy of rllm.trainer.verl.train_agent_ppo.TaskRunner
    with the only modification being the use of PredictiveAgentWorkflowTrainer
    instead of AgentWorkflowPPOTrainer.

    Note: We cannot inherit from TaskRunner because it's already a @ray.remote actor.
    Instead, we copy the run() method here.
    """

    def run(self, config, workflow_class=None, workflow_args=None, agent_class=None, env_class=None, agent_args=None, env_args=None):
        """Execute training with PredictiveAgentWorkflowTrainer."""
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"PredictiveTaskRunner hostname: {__import__('socket').gethostname()}, PID: {__import__('os').getpid()}")
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        # Download the checkpoint from HDFS to the local machine.
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
            from rllm_ext.training.predictive_fsdp_workers import (
                AsyncPredictiveActorRolloutRefWorker,
                PredictiveActorRolloutRefWorker,
            )

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            prediction_loss_enabled = config.actor_rollout_ref.actor.get("prediction_loss_weight", 0) > 0
            if prediction_loss_enabled:
                actor_rollout_cls = (
                    AsyncPredictiveActorRolloutRefWorker
                    if config.actor_rollout_ref.rollout.mode == "async"
                    else PredictiveActorRolloutRefWorker
                )
            else:
                actor_rollout_cls = (
                    AsyncActorRolloutRefWorker
                    if config.actor_rollout_ref.rollout.mode == "async"
                    else ActorRolloutRefWorker
                )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        from verl.trainer.ppo.reward import load_reward_manager

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # === KEY CHANGE: Use PredictiveAgentWorkflowTrainer instead of AgentWorkflowPPOTrainer ===
        if config.rllm.workflow.use_workflow:
            from rllm.trainer.env_agent_mappings import WORKFLOW_CLASS_MAPPING
            from rllm_ext.training import PredictiveAgentWorkflowTrainer  # Import our custom trainer

            if workflow_class is None:
                workflow_class = WORKFLOW_CLASS_MAPPING[config.rllm.workflow.name]
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                for key, value in config.rllm.workflow.get("workflow_args").items():
                    if value is not None:
                        if key in workflow_args and isinstance(workflow_args[key], dict):
                            workflow_args[key].update(value)
                        else:
                            workflow_args[key] = value

            # Use our custom trainer instead of AgentWorkflowPPOTrainer
            trainer = PredictiveAgentWorkflowTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                workflow_class=workflow_class,
                workflow_args=workflow_args,
            )

        else:
            # Non-workflow path uses original AgentPPOTrainer
            from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
            from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer

            if env_class is None:
                env_class = ENV_CLASS_MAPPING[config.rllm.env.name]
            if agent_class is None:
                agent_class = AGENT_CLASS_MAPPING[config.rllm.agent.name]

            env_args = env_args or {}
            agent_args = agent_args or {}
            if config.rllm.env.get("env_args") is not None:
                env_args.update(config.rllm.env.get("env_args"))
            if config.rllm.agent.get("agent_args") is not None:
                agent_args.update(config.rllm.agent.get("agent_args"))

            trainer = AgentPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                env_class=env_class,
                agent_class=agent_class,
                env_args=env_args,
                agent_args=agent_args,
            )

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()
