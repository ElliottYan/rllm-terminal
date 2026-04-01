"""
Extension package for custom experiments built on top of rllm.

Design goal:
- Keep experimental / project-specific logic isolated from the core `rllm` package.
- Avoid modifying existing `rllm` code paths; only compose/extend via inheritance.

Main exports:
- PredictiveToolWorkflow: Workflow with prediction sub-step
- PredictiveToolAgent: Agent with prediction storage
- PredictiveToolEnvironment: Environment with prediction similarity reward
- training module: Training extensions with auxiliary loss support
"""

from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent, PredictionRecord
from rllm_ext.environments.predictive_tool_env import PredictiveToolEnvironment
from rllm_ext.workflows.predictive_tool_workflow import (
    GenerativeSupportConfig,
    PredictiveToolWorkflow,
    PredictionConfig,
    TrajectoryLoggingConfig,
)

__all__ = [
    "PredictiveToolWorkflow",
    "PredictionConfig",
    "GenerativeSupportConfig",
    "TrajectoryLoggingConfig",
    "PredictiveToolAgent",
    "PredictionRecord",
    "PredictiveToolEnvironment",
]

# Optional: training module imports
try:
    from rllm_ext.training import (
        PredictiveAgentTrainer,
        PredictiveAgentWorkflowEngine,
        PredictiveAgentWorkflowTrainer,
        PredictiveActor,
    )

    __all__.extend(["PredictiveAgentTrainer", "PredictiveAgentWorkflowEngine", "PredictiveAgentWorkflowTrainer", "PredictiveActor"])
except ImportError:
    pass  # Training extensions are optional
