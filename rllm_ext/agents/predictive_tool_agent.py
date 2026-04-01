from __future__ import annotations

from dataclasses import dataclass, field
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
    - This class only provides a stable place (Step.info keys) to store prediction
      data without modifying `rllm.agents.tool_agent.ToolAgent`.
    """

    # Namespaced keys to avoid collisions with existing code
    INFO_KEY_PREDICTION = "rllm_ext.prediction"
    INFO_KEY_IMAGINE = "rllm_ext.imagine"
    INFO_KEY_GENERATIVE_SUPPORT = "rllm_ext.generative_support"
    INFO_KEY_CANDIDATE_ACTION = "rllm_ext.candidate_action"
    INFO_KEY_FINAL_ACTION = "rllm_ext.final_action"
    INFO_KEY_ACTION_REVISED = "rllm_ext.action_revised"
    INFO_KEY_ACTUAL_OUTPUT = "rllm_ext.actual_output"
    INFO_KEY_ACTUAL_TOOL_OUTPUTS = "rllm_ext.actual_tool_outputs"

    def set_step_prediction(self, *, prediction: PredictionRecord) -> None:
        """
        Attach the prediction record to the current trajectory step (latest step).
        """
        step = self.get_current_state()
        if step is None:
            return
        # Ensure step.info is a dict before accessing it
        if step.info is None:
            step.info = {}
        step.info[self.INFO_KEY_PREDICTION] = {
            "prompt": prediction.prompt,
            "prediction": prediction.prediction,
            "metadata": prediction.metadata or {},
        }

    def set_step_imagine(self, *, prediction: PredictionRecord) -> None:
        """
        Attach the imagine record to the current trajectory step (latest step).
        """
        step = self.get_current_state()
        if step is None:
            return
        if step.info is None:
            step.info = {}
        step.info[self.INFO_KEY_IMAGINE] = {
            "prompt": prediction.prompt,
            "prediction": prediction.prediction,
            "metadata": prediction.metadata or {},
        }

    def set_step_generative_support(
        self,
        *,
        mode: str,
        prompt: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach one structured generative-support record to the current step."""
        step = self.get_current_state()
        if step is None:
            return
        if step.info is None:
            step.info = {}
        step.info[self.INFO_KEY_GENERATIVE_SUPPORT] = {
            "mode": mode,
            "prompt": prompt,
            "text": text,
            "metadata": metadata or {},
        }
