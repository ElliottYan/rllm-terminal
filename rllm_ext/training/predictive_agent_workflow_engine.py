"""Extended AgentWorkflowEngine that prepares prediction supervision for training."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent

try:
    from verl.utils.torch_functional import pad_sequence_to_length
except ImportError:
    def pad_sequence_to_length(tensor, max_length, padding_value, left_pad=False):
        """Fallback used in tests when optional verl deps are unavailable."""
        if tensor.size(1) >= max_length:
            return tensor

        pad_shape = (tensor.size(0), max_length - tensor.size(1), *tensor.shape[2:])
        padding = torch.full(
            pad_shape,
            padding_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if left_pad:
            return torch.cat([padding, tensor], dim=1)
        return torch.cat([tensor, padding], dim=1)

if TYPE_CHECKING:
    from verl import DataProto


class PredictiveAgentWorkflowEngine(AgentWorkflowEngine):
    """
    Extended AgentWorkflowEngine that prepares prediction supervision.

    When prediction turns are kept in the transcript, this engine computes a
    ``prediction_mask`` that identifies which response tokens belong to those
    assistant messages so PPO can exclude them while the auxiliary CE loss still
    trains on the same forward pass.

    When prediction turns are intentionally omitted from the transcript,
    prediction supervision is reconstructed from ``Step.info`` and attached as
    ``prediction_targets`` for a separate auxiliary forward pass.
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

        actual_tool_outputs = step_info.get(
            PredictiveToolAgent.INFO_KEY_ACTUAL_TOOL_OUTPUTS
        )
        if isinstance(actual_tool_outputs, dict) and actual_tool_outputs:
            normalized = {str(k): str(v) for k, v in actual_tool_outputs.items()}
            return " ".join(normalized[k] for k in sorted(normalized.keys()))

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

        raw_messages = (
            copy.deepcopy(step.chat_completions)
            if isinstance(step.chat_completions, list)
            else []
        )
        prompt_messages = PredictiveAgentWorkflowEngine._strip_auxiliary_transcript(
            raw_messages
        )

        generative_record = step_info.get(
            PredictiveToolAgent.INFO_KEY_GENERATIVE_SUPPORT, {}
        )
        generative_mode = (
            generative_record.get("mode", "none")
            if isinstance(generative_record, dict)
            else "none"
        )
        generative_prompt = (
            generative_record.get("prompt", "")
            if isinstance(generative_record, dict)
            else ""
        )
        generative_text = (
            generative_record.get("text", "") if isinstance(generative_record, dict) else ""
        )

        if (
            not prompt_messages
            or not isinstance(prompt_messages[-1], dict)
            or prompt_messages[-1].get("role") != "user"
            or prompt_messages[-1].get("content") != prediction_prompt
        ):
            if (
                generative_mode == "post_action_simulator"
                and isinstance(generative_prompt, str)
                and generative_prompt.strip()
                and isinstance(generative_text, str)
                and generative_text.strip()
            ):
                prompt_messages.append({"role": "user", "content": generative_prompt})
                prompt_messages.append({"role": "assistant", "content": generative_text})
            prompt_messages.append({"role": "user", "content": prediction_prompt})

        return {
            "prompt_messages": prompt_messages,
            "target_text": f"<prediction>{actual_output}</prediction>",
            "actual": actual_output,
            "prediction_prompt": prediction_prompt,
        }

    @staticmethod
    def _build_imagine_loss_example(step) -> dict | None:
        """
        Build one supervised imagine example from a rollout step.

        Scheme C uses the actual env output as supervision for the imagine stage.
        """
        step_info = step.info if isinstance(step.info, dict) else {}
        imagine_record = step_info.get(PredictiveToolAgent.INFO_KEY_IMAGINE)
        if not isinstance(imagine_record, dict):
            return None

        imagine_prompt = imagine_record.get("prompt")
        if not isinstance(imagine_prompt, str) or not imagine_prompt.strip():
            return None

        actual_output = PredictiveAgentWorkflowEngine._extract_actual_output(step).strip()
        if not actual_output:
            return None

        raw_messages = (
            copy.deepcopy(step.chat_completions)
            if isinstance(step.chat_completions, list)
            else []
        )
        prompt_messages = PredictiveAgentWorkflowEngine._strip_auxiliary_transcript(
            raw_messages
        )
        prompt_messages.append({"role": "user", "content": imagine_prompt})

        return {
            "prompt_messages": prompt_messages,
            "target_text": f"<imagine>{actual_output}</imagine>",
            "actual": actual_output,
            "prediction_prompt": imagine_prompt,
            "kind": "imagine",
        }

    @staticmethod
    def _messages_contain_prediction_turns(messages) -> bool:
        return any(
            isinstance(message, dict)
            and (
                message.get("rllm_prediction", False)
                or message.get("rllm_imagine", False)
            )
            for message in messages or []
        )

    @staticmethod
    def _compute_prediction_mask_last_turn(messages, chat_parser) -> torch.Tensor:
        """
        Compute prediction mask for stepwise rows tokenized with tokenize_and_mask().

        In stepwise mode, each row only trains on the last assistant turn.
        """
        try:
            last_assistant_idx = max(
                i for i, msg in enumerate(messages) if msg["role"] == "assistant"
            )
        except ValueError:
            raise ValueError("No assistant message found in chat_completions") from None

        last_message = messages[last_assistant_idx]
        response = chat_parser.parse(
            [last_message],
            is_first_msg=False,
            add_generation_prompt=False,
            accumulate_reasoning=True,
        )
        response = response[len(chat_parser.generation_prompt) :].rstrip("\n")
        ids = chat_parser.tokenizer.encode(response, add_special_tokens=False)

        is_prediction = bool(
            last_message.get("rllm_prediction", False)
            or last_message.get("rllm_imagine", False)
        )
        fill_value = 1 if is_prediction else 0
        return torch.full((len(ids),), fill_value, dtype=torch.long)

    @staticmethod
    def _strip_auxiliary_tags(text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        stripped = text
        for tag in ("simulation", "prediction", "imagine"):
            while True:
                begin = f"<{tag}>"
                end = f"</{tag}>"
                start = stripped.find(begin)
                if start == -1:
                    break
                stop = stripped.find(end, start + len(begin))
                if stop == -1:
                    break
                stripped = stripped[:start] + stripped[stop + len(end) :]
        return stripped.strip()

    @classmethod
    def _strip_auxiliary_transcript(cls, messages) -> list[dict]:
        cleaned_messages = []
        messages = list(messages or [])
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            next_message = messages[idx + 1] if idx + 1 < len(messages) else None
            if (
                message.get("role") == "user"
                and isinstance(next_message, dict)
                and (
                    next_message.get("rllm_prediction")
                    or next_message.get("rllm_simulation")
                    or next_message.get("rllm_imagine")
                )
            ):
                continue
            if (
                message.get("rllm_prediction")
                or message.get("rllm_simulation")
                or message.get("rllm_imagine")
            ):
                continue
            normalized = copy.deepcopy(message)
            if normalized.get("role") == "assistant":
                normalized["content"] = cls._strip_auxiliary_tags(
                    normalized.get("content", "")
                )
            cleaned_messages.append(normalized)
        return cleaned_messages

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
            is_prediction = is_asst and (
                messages[i].get("rllm_prediction", False)
                or messages[i].get("rllm_imagine", False)
            )

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
        Transform episode results into Verl-compatible DataProto with prediction supervision.

        Computes either a prediction_mask (when prediction turns are present in the
        transcript) or prediction_targets (when prediction turns were kept out of
        the transcript to preserve rollout context).
        """
        prediction_masks = []
        prediction_targets = []
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
                    for step in trajectory.steps:
                        if self._messages_contain_prediction_turns(
                            step.chat_completions
                        ):
                            pred_mask = self._compute_prediction_mask_last_turn(
                                step.chat_completions, chat_parser
                            )
                            prediction_target = {
                                "examples": [],
                                "has_prediction_target": False,
                            }
                        else:
                            pred_mask = torch.zeros(0, dtype=torch.long)
                            examples = []
                            imagine_example = self._build_imagine_loss_example(step)
                            if imagine_example is not None:
                                examples.append(imagine_example)
                            prediction_example = self._build_prediction_loss_example(
                                step
                            )
                            if prediction_example is not None:
                                examples.append(prediction_example)
                            prediction_target = {
                                "examples": examples,
                                "has_prediction_target": bool(examples),
                            }
                        prediction_masks.append(pred_mask)
                        prediction_targets.append(prediction_target)
                else:
                    trajectory_is_cumulative = (
                        len(trajectory.steps) <= 1 or trajectory.is_cumulative()
                    )
                    if len(trajectory.steps) > 1:
                        chat_completions = trajectory.steps[-1].chat_completions
                        if trajectory_is_cumulative and self._messages_contain_prediction_turns(chat_completions):
                            pred_mask = self._compute_prediction_mask_cumulative(
                                chat_completions, chat_parser
                            )
                            prediction_target = {
                                "examples": [],
                                "has_prediction_target": False,
                            }
                        else:
                            pred_mask = torch.zeros(0, dtype=torch.long)
                            trajectory_examples = []
                            for step in trajectory.steps:
                                imagine_example = self._build_imagine_loss_example(
                                    step
                                )
                                if imagine_example is not None:
                                    trajectory_examples.append(imagine_example)
                                prediction_example = (
                                    self._build_prediction_loss_example(step)
                                )
                                if prediction_example is not None:
                                    trajectory_examples.append(prediction_example)
                            prediction_target = {
                                "examples": trajectory_examples,
                                "has_prediction_target": bool(trajectory_examples),
                            }
                        prediction_masks.append(pred_mask)
                        prediction_targets.append(prediction_target)
                    else:
                        chat_completions = trajectory.steps[0].chat_completions
                        if self._messages_contain_prediction_turns(chat_completions):
                            prediction_masks.append(
                                self._compute_prediction_mask_cumulative(
                                    chat_completions, chat_parser
                                )
                            )
                            prediction_targets.append(
                                {
                                    "examples": [],
                                    "has_prediction_target": False,
                                }
                            )
                        else:
                            prediction_masks.append(torch.zeros(0, dtype=torch.long))
                            examples = []
                            imagine_example = self._build_imagine_loss_example(
                                trajectory.steps[0]
                            )
                            if imagine_example is not None:
                                examples.append(imagine_example)
                            prediction_example = self._build_prediction_loss_example(
                                trajectory.steps[0]
                            )
                            if prediction_example is not None:
                                examples.append(prediction_example)
                            prediction_targets.append(
                                {
                                    "examples": examples,
                                    "has_prediction_target": bool(examples),
                                }
                            )

        # Call parent implementation to get the base batch
        batch = super().transform_results_for_verl(episodes, task_ids)

        expected_size = len(batch.non_tensor_batch["step_ids"])

        if len(prediction_masks) != expected_size:
            raise RuntimeError(
                f"prediction_masks misaligned with verl batch: got {len(prediction_masks)} entries, expected {expected_size}"
            )
        if len(prediction_targets) != expected_size:
            raise RuntimeError(
                f"prediction_targets misaligned with verl batch: got {len(prediction_targets)} entries, expected {expected_size}"
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
        batch.non_tensor_batch["prediction_targets"] = np.array(
            prediction_targets, dtype=object
        )

        return batch
