from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rllm.agents.tool_agent import ToolAgent


@dataclass
class PredictionRecord:
    """A minimal, structured container for storing predictions for a single step."""

    prompt: str
    prediction: str
    metadata: dict[str, Any] | None = None


class PredictiveToolAgent(ToolAgent):
    """
    A ToolAgent extension that stores a per-step prediction record.

    Notes:
    - We intentionally keep this lightweight; the workflow is responsible for *asking*
      the model to produce predictions and for attaching them to the current step.
    - This class only provides a stable place (Step.info keys) to store the prediction
      data without modifying `rllm.agents.tool_agent.ToolAgent`.
    """

    # Namespaced keys to avoid collisions with existing code
    INFO_KEY_PREDICTION = "rllm_ext.prediction"

    def set_step_prediction(self, *, prediction: PredictionRecord) -> None:
        """
        Attach the prediction record to the current trajectory step (latest step).
        """
        step = self.get_current_state()
        if step is None:
            return
        step.info[self.INFO_KEY_PREDICTION] = {
            "prompt": prediction.prompt,
            "prediction": prediction.prediction,
            "metadata": prediction.metadata or {},
        }

