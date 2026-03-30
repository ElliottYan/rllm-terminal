"""Training extensions for prediction auxiliary loss.

This module provides extended training components that add prediction loss
to the standard PPO training pipeline.

Key components:
- PredictiveAgentTrainer: Drop-in replacement for AgentTrainer with prediction loss support
- PredictiveAgentWorkflowEngine: Extends workflow engine to collect prediction data
- PredictiveActor: Extends actor to compute prediction auxiliary loss
- PredictiveAgentWorkflowTrainer: Low-level trainer with prediction loss support

Usage (High-level API):
    ```python
    from rllm_ext.training import PredictiveAgentTrainer
    from rllm_ext.workflows import PredictiveToolWorkflow

    # Just replace AgentTrainer with PredictiveAgentTrainer
    trainer = PredictiveAgentTrainer(
        workflow_class=PredictiveToolWorkflow,
        workflow_args={...},
        config=config,
        train_dataset=train_data,
        val_dataset=val_data,
    )
    trainer.train()
    ```

Usage (Low-level API):
    ```python
    from rllm_ext.training import PredictiveAgentWorkflowTrainer

    trainer = PredictiveAgentWorkflowTrainer(
        workflow_class=PredictiveToolWorkflow,
        workflow_args={...},
        config=config,
        ...
    )
    trainer.train()
    ```
"""

from rllm_ext.training.predictive_agent_workflow_engine import PredictiveAgentWorkflowEngine
from rllm_ext.training.predictive_actor import PredictiveActor, create_predictive_actor

__all__ = [
    "PredictiveAgentWorkflowEngine",
    "PredictiveActor",
    "create_predictive_actor",
]

try:
    from rllm_ext.training.predictive_agent_trainer import PredictiveAgentTrainer

    __all__.append("PredictiveAgentTrainer")  # High-level API (recommended)
except ImportError:
    PredictiveAgentTrainer = None

try:
    from rllm_ext.training.predictive_agent_workflow_trainer import (
        PredictiveAgentWorkflowTrainer,
    )

    __all__.append("PredictiveAgentWorkflowTrainer")  # Low-level API
except ImportError:
    PredictiveAgentWorkflowTrainer = None
