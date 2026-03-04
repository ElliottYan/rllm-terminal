"""Extended AgentWorkflowEngine that collects prediction data for auxiliary loss."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine

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

    def transform_results_for_verl(self, episodes, task_ids: np.ndarray) -> "DataProto":
        """
        Transform episode results into Verl-compatible DataProto with prediction data.

        Extends the base implementation to add prediction_targets field.
        """
        # First, collect prediction targets in sync with step generation
        # We need to mirror the logic in parent class to maintain correct ordering
        prediction_targets = []

        for i, episode in enumerate(episodes):
            if episode is None:
                # Episode was dropped, but we need to track this
                continue

            if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
                # Empty trajectory, dropped
                continue

            for trajectory in episode.trajectories:
                for step_idx, step in enumerate(trajectory.steps):
                    # Extract prediction metadata from step.info
                    # Prediction is stored in step.info["rllm_ext.prediction"] by PredictiveToolAgent
                    prediction_record = step.info.get("rllm_ext.prediction")

                    # Get prediction text and metadata
                    if prediction_record:
                        pred_text = prediction_record.get("prediction")
                        pred_raw = prediction_record.get("metadata", {}).get("raw_text")
                        pred_reasoning = prediction_record.get("metadata", {}).get("reasoning", "")
                        has_prediction = pred_text is not None
                    else:
                        pred_text = None
                        pred_raw = None
                        pred_reasoning = ""
                        has_prediction = False

                    # Get actual tool output from observation
                    actual_output = ""
                    if hasattr(step, "observation") and step.observation:
                        tool_outputs = step.observation.get("tool_outputs", {})
                        if tool_outputs:
                            # Concatenate all tool outputs
                            actual_output = " ".join(tool_outputs.values())

                    prediction_targets.append(
                        {
                            "prediction": pred_text or "",
                            "actual": actual_output,
                            "prediction_raw_text": pred_raw or "",
                            "prediction_reasoning": pred_reasoning,
                            "has_prediction": has_prediction,
                        }
                    )

        # Call parent implementation to get the base batch
        batch = super().transform_results_for_verl(episodes, task_ids)

        # Ensure prediction_targets matches batch size
        # The parent class creates one entry per step across all trajectories
        expected_size = len(batch.non_tensor_batch["step_ids"])

        while len(prediction_targets) < expected_size:
            prediction_targets.append(
                {
                    "prediction": "",
                    "actual": "",
                    "prediction_raw_text": "",
                    "prediction_reasoning": "",
                    "has_prediction": False,
                }
            )

        prediction_targets = prediction_targets[:expected_size]

        # Add prediction_targets directly to non_tensor_batch in-place
        # This avoids creating a new DataProto and holding TensorDict references
        batch.non_tensor_batch["prediction_targets"] = np.array(prediction_targets, dtype=object)

        return batch
