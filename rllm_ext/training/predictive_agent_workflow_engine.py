"""Extended AgentWorkflowEngine that collects prediction data for auxiliary loss."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent

if TYPE_CHECKING:
    from verl import DataProto


class PredictiveAgentWorkflowEngine(AgentWorkflowEngine):
    """
    Extended AgentWorkflowEngine that collects prediction data for auxiliary loss computation.

    This class extends the base AgentWorkflowEngine to:
    1. Extract prediction metadata from episode steps
    2. Store prediction targets in DataProto non_tensor_batch
    3. Keep all changes isolated in rllm_ext
    """

    @staticmethod
    def _extract_actual_output(step) -> str:
        """
        Extract post-action actual output text for a step.

        Preferred source is step.info populated by PredictiveToolWorkflow after env.step().
        Falls back to step.observation for backward compatibility with older rollouts.
        """
        step_info = step.info if isinstance(step.info, dict) else {}

        actual_output = step_info.get(PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT)
        if isinstance(actual_output, str) and actual_output:
            return actual_output

        actual_tool_outputs = step_info.get(PredictiveToolAgent.INFO_KEY_ACTUAL_TOOL_OUTPUTS)
        if isinstance(actual_tool_outputs, dict) and actual_tool_outputs:
            normalized = {str(k): str(v) for k, v in actual_tool_outputs.items()}
            return " ".join(normalized[k] for k in sorted(normalized.keys()))

        # Backward-compatible fallback: older trajectories may only have step.observation.
        observation = getattr(step, "observation", None)
        if isinstance(observation, dict):
            tool_outputs = observation.get("tool_outputs", {})
            if isinstance(tool_outputs, dict) and tool_outputs:
                normalized = {str(k): str(v) for k, v in tool_outputs.items()}
                return " ".join(normalized[k] for k in sorted(normalized.keys()))

        return ""

    @staticmethod
    def _build_prediction_loss_example(step) -> dict | None:
        """
        Build one supervised prediction example from a rollout step.

        The prediction loss is trained on the same context as the prediction sub-step:
        prompt/history + assistant action + user prediction prompt -> assistant actual output.
        """
        step_info = step.info if isinstance(step.info, dict) else {}
        prediction_record = step_info.get(PredictiveToolAgent.INFO_KEY_PREDICTION)
        if not isinstance(prediction_record, dict):
            return None

        prediction_prompt = prediction_record.get("prompt")
        if not isinstance(prediction_prompt, str) or not prediction_prompt.strip():
            return None

        actual_output = PredictiveAgentWorkflowEngine._extract_actual_output(step).strip()
        if not actual_output:
            return None

        prompt_messages = copy.deepcopy(step.chat_completions) if isinstance(step.chat_completions, list) else []

        if prompt_messages and isinstance(prompt_messages[-1], dict) and prompt_messages[-1].get("role") == "assistant":
            prompt_messages = prompt_messages[:-1]

        if (
            not prompt_messages
            or not isinstance(prompt_messages[-1], dict)
            or prompt_messages[-1].get("role") != "user"
            or prompt_messages[-1].get("content") != prediction_prompt
        ):
            prompt_messages.append({"role": "user", "content": prediction_prompt})

        return {
            "prompt_messages": prompt_messages,
            "target_text": f"<prediction>{actual_output}</prediction>",
            "actual": actual_output,
            "prediction_prompt": prediction_prompt,
        }

    def transform_results_for_verl(self, episodes, task_ids: np.ndarray) -> "DataProto":
        """
        Transform episode results into Verl-compatible DataProto with prediction data.

        Extends the base implementation to add prediction_targets field.
        """
        # First, collect prediction targets in sync with parent batch row construction.
        # When stepwise advantage is disabled, parent creates one row per trajectory;
        # otherwise it creates one row per step.
        prediction_targets = []
        stepwise_enabled = bool(self.config.rllm.stepwise_advantage.enable)

        for i, episode in enumerate(episodes):
            if episode is None:
                # Episode was dropped, but we need to track this
                continue

            if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
                # Empty trajectory, dropped
                continue

            for trajectory in episode.trajectories:
                if len(trajectory.steps) == 0:
                    continue

                if stepwise_enabled:
                    for step in trajectory.steps:
                        example = self._build_prediction_loss_example(step)
                        prediction_targets.append(
                            {
                                "examples": [example] if example is not None else [],
                                "has_prediction_target": example is not None,
                            }
                        )
                else:
                    trajectory_examples = []
                    for step in trajectory.steps:
                        example = self._build_prediction_loss_example(step)
                        if example is not None:
                            trajectory_examples.append(example)

                    prediction_targets.append(
                        {
                            "examples": trajectory_examples,
                            "has_prediction_target": bool(trajectory_examples),
                        }
                    )

        # Call parent implementation to get the base batch
        batch = super().transform_results_for_verl(episodes, task_ids)

        expected_size = len(batch.non_tensor_batch["step_ids"])

        if len(prediction_targets) != expected_size:
            raise RuntimeError(
                f"prediction_targets misaligned with verl batch: got {len(prediction_targets)} entries, expected {expected_size}"
            )

        # Add prediction_targets directly to non_tensor_batch in-place
        # This avoids creating a new DataProto and holding TensorDict references
        batch.non_tensor_batch["prediction_targets"] = np.array(prediction_targets, dtype=object)

        return batch
