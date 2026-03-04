"""Extended AgentWorkflowPPOTrainer with prediction auxiliary loss support."""

from __future__ import annotations

from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm_ext.training.predictive_agent_workflow_engine import PredictiveAgentWorkflowEngine


class PredictiveAgentWorkflowTrainer(AgentWorkflowPPOTrainer):
    """
    Extended PPO trainer that supports prediction auxiliary loss.

    This trainer:
    1. Uses PredictiveAgentWorkflowEngine to collect prediction data
    2. Overrides actor initialization to use PredictiveActor
    3. Maintains all existing training functionality from AgentWorkflowPPOTrainer

    Usage:
        ```python
        from rllm_ext.training import PredictiveAgentWorkflowTrainer

        trainer = PredictiveAgentWorkflowTrainer(
            workflow_class=PredictiveToolWorkflow,
            workflow_args=workflow_args,
            config=config,
            ...
        )
        trainer.train()
        ```
    """

    def init_workers(self):
        """
        Override init_workers to use extended workflow engine and actor.

        This method:
        1. Calls parent's init_workers first (sets up actor_rollout_wg, critic, etc.)
        2. Replaces AgentWorkflowEngine with PredictiveAgentWorkflowEngine
        3. Replaces DataParallelPPOActor with PredictiveActor
        """
        # Step 1: Configure prediction loss BEFORE parent init
        # (so actor can read the config during creation)
        # self._configure_prediction_loss()

        # Step 2: Call parent's init_workers
        # This creates:
        # - self.actor_rollout_wg with DataParallelPPOActor
        # - self.critic_wg
        # - self.ref_policy_wg (if needed)
        # - self.agent_execution_engine (AgentWorkflowEngine - will be replaced)
        super().init_workers()

        # Step 3: Replace workflow engine with our extended version
        self._replace_workflow_engine()

        # Step 4: Replace actor with our extended version
        if hasattr(self.config, "actor_rollout_ref") and hasattr(self.config.actor_rollout_ref, "actor"):
            if self.config.actor_rollout_ref.actor.get("prediction_loss_weight", 0) > 0:
                self._inject_predictive_actor()

    # def _configure_prediction_loss(self):
    #     """
    #     Add prediction loss configuration to actor config.

    #     Reads from config.rllm.prediction_loss if available, otherwise uses defaults.
    #     """
    #     # Get prediction loss config from hydra config
    #     pred_config = self.config.rllm.get("prediction_loss", {})

    #     # Set defaults
    #     defaults = {
    #         "enabled": False,  # Disabled by default
    #         "weight": 0.1,
    #         "loss_type": "cross_entropy",
    #         "temperature": 1.0,
    #     }

    #     # Merge with provided config
    #     for key, value in defaults.items():
    #         if key not in pred_config:
    #             pred_config[key] = value

    #     # Add to actor config (actor is under actor_rollout_ref)
    #     # This will be read when DataParallelPPOActor is created
    #     if hasattr(self.config, "actor_rollout_ref") and hasattr(self.config.actor_rollout_ref, "actor"):
    #         self.config.actor_rollout_ref.actor.prediction_loss_weight = pred_config["weight"] if pred_config["enabled"] else 0
    #         self.config.actor_rollout_ref.actor.prediction_loss_type = pred_config["loss_type"]
    #         self.config.actor_rollout_ref.actor.prediction_temperature = pred_config["temperature"]

    def _replace_workflow_engine(self):
        """
        Replace the default AgentWorkflowEngine with PredictiveAgentWorkflowEngine.

        This is called after parent's init_workers to swap in our custom engine.
        """
        import asyncio

        from rllm.engine.rollout.verl_engine import VerlEngine

        # Create new rollout engine (same as parent)
        rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
        )

        # Create PredictiveAgentWorkflowEngine with same config
        predictive_engine = PredictiveAgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            retry_limit=self.config.rllm.workflow.retry_limit,
        )

        # Initialize the new engine's worker pool
        asyncio.run_coroutine_threadsafe(predictive_engine.initialize_pool(), self._loop).result()

        # Replace the engine
        # Note: shutdown the old engine's pool first to avoid resource leaks
        if hasattr(self.agent_execution_engine, "executor") and self.agent_execution_engine.executor is not None:
            self.agent_execution_engine.shutdown()

        self.agent_execution_engine = predictive_engine
        print("PredictiveAgentWorkflowEngine injected successfully")

    def _inject_predictive_actor(self):
        """
        Replace the default actor with PredictiveActor.

        This is called after parent's init_workers to swap in our custom actor.
        """
        # Only inject if prediction loss is enabled
        actor_config = self.config.actor_rollout_ref.actor
        if actor_config.get("prediction_loss_weight", 0) > 0:
            from rllm_ext.training.predictive_actor import PredictiveActor

            # Get the current actor module and optimizer
            actor_module = self.actor_rollout_wg.actor.actor_module
            actor_optimizer = self.actor_rollout_wg.actor.actor_optimizer

            # Create PredictiveActor with same module and optimizer
            predictive_actor = PredictiveActor(
                config=actor_config,
                actor_module=actor_module,
                actor_optimizer=actor_optimizer,
            )

            # Replace the actor in the worker group
            self.actor_rollout_wg.actor = predictive_actor
            print("PredictiveActor injected successfully")
