"""Extended AgentWorkflowEngine that computes prediction_mask for auxiliary loss."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from verl.utils.torch_functional import pad_sequence_to_length

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine

if TYPE_CHECKING:
    from verl import DataProto


class PredictiveAgentWorkflowEngine(AgentWorkflowEngine):
    """
    Extended AgentWorkflowEngine that computes prediction_mask for auxiliary loss.

    Instead of building separate prediction_targets for a second forward pass,
    this engine computes a prediction_mask that identifies which response tokens
    are prediction assistant turns (marked with ``rllm_prediction: True``).

    The prediction_mask is added to ``batch.batch`` and the PPO response_mask is
    adjusted to exclude prediction tokens: ``response_mask *= (1 - prediction_mask)``.
    """

    @staticmethod
    def _compute_prediction_mask_cumulative(messages, chat_parser) -> torch.Tensor:
        """
        Compute prediction mask for cumulative multi-turn messages.

        Parallels ``ChatTemplateParser.tokenize_and_mask_cumulative`` but tracks
        which tokens come from assistant messages marked with ``rllm_prediction: True``.

        Returns:
            Tensor of the same length as the response from
            ``tokenize_and_mask_cumulative``, with 1 for prediction assistant
            tokens and 0 for everything else.
        """
        try:
            first_assistant_idx = next(
                i for i, msg in enumerate(messages) if msg["role"] == "assistant"
            )
        except StopIteration:
            raise ValueError("No assistant message found in chat_completions") from None

        prediction_mask = []

        for i in range(first_assistant_idx, len(messages)):
            is_asst = messages[i]["role"] == "assistant"
            is_prediction = is_asst and messages[i].get("rllm_prediction", False)

            if is_asst:
                response = chat_parser.parse(
                    [messages[i]],
                    is_first_msg=False,
                    add_generation_prompt=False,
                    accumulate_reasoning=True,
                )
                response = response[len(chat_parser.generation_prompt) :]
                ids = chat_parser.tokenizer.encode(response, add_special_tokens=False)
                if is_prediction:
                    prediction_mask.extend([1] * len(ids))
                else:
                    prediction_mask.extend([0] * len(ids))
            else:
                response = chat_parser.parse(
                    [messages[i]],
                    is_first_msg=False,
                    add_generation_prompt=True,
                    accumulate_reasoning=False,
                )
                ids = chat_parser.tokenizer.encode(response, add_special_tokens=False)
                prediction_mask.extend([0] * len(ids))

        return torch.tensor(prediction_mask, dtype=torch.long)

    def transform_results_for_verl(self, episodes, task_ids: np.ndarray) -> "DataProto":
        """
        Transform episode results into Verl-compatible DataProto with prediction_mask.

        Computes a prediction_mask per row in sync with the parent's row construction,
        then adjusts response_mask to exclude prediction tokens from PPO loss.
        """
        # Compute prediction_masks in sync with parent's row construction.
        prediction_masks = []
        stepwise_enabled = bool(self.config.rllm.stepwise_advantage.enable)
        chat_parser = self.rollout_engine.chat_parser

        for episode in episodes:
            if episode is None:
                continue

            if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
                continue

            for trajectory in episode.trajectories:
                if len(trajectory.steps) == 0:
                    continue

                if stepwise_enabled:
                    for _step in trajectory.steps:
                        # Stepwise mode uses tokenize_and_mask (single-turn).
                        # Prediction mask not applicable in single-turn mode.
                        prediction_masks.append(torch.zeros(0, dtype=torch.long))
                else:
                    if len(trajectory.steps) > 1:
                        # Cumulative: compute prediction mask from last step's chat_completions
                        chat_completions = trajectory.steps[-1].chat_completions
                        pred_mask = self._compute_prediction_mask_cumulative(
                            chat_completions, chat_parser
                        )
                        prediction_masks.append(pred_mask)
                    else:
                        # Single step: no prediction mask applicable
                        prediction_masks.append(torch.zeros(0, dtype=torch.long))

        # Call parent implementation to get the base batch
        batch = super().transform_results_for_verl(episodes, task_ids)

        expected_size = len(batch.non_tensor_batch["step_ids"])

        if len(prediction_masks) != expected_size:
            raise RuntimeError(
                f"prediction_masks misaligned with verl batch: got {len(prediction_masks)} entries, expected {expected_size}"
            )

        # Pad prediction_masks to max_response_length
        max_response_length = self.config.data.max_response_length
        if prediction_masks:
            pred_masks_padded = torch.nn.utils.rnn.pad_sequence(
                prediction_masks, batch_first=True, padding_value=0
            )
            pred_masks_padded = pad_sequence_to_length(
                pred_masks_padded, max_response_length, 0, left_pad=False
            )
            pred_masks_padded = pred_masks_padded[:, :max_response_length]
        else:
            pred_masks_padded = torch.zeros(
                expected_size, max_response_length, dtype=torch.long
            )

        # Add prediction_mask to batch tensors and adjust response_mask
        batch.batch["prediction_mask"] = pred_masks_padded
        batch.batch["response_mask"] = batch.batch["response_mask"] * (
            1 - pred_masks_padded.long()
        )

        return batch
