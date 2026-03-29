from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
from rllm.workflows.workflow import TerminationReason, Workflow

from rllm_ext.agents.predictive_tool_agent import PredictionRecord, PredictiveToolAgent

logger = logging.getLogger(__name__)


@dataclass
class PredictionConfig:
    """
    Configuration for prediction sub-step.
    """

    enabled: bool = True
    enable_prediction_loss: bool = False
    enable_prediction_step: bool = False
    max_tokens: int = 256
    # If True, prediction prompt/answer are kept in live message history for future turns.
    # The per-step training snapshot always includes prediction so RL reward is attached
    # to the prediction assistant turn.
    add_prediction_to_messages: bool = True
    simple_tir: bool = (
        False  # if True, filter out steps without tool calls from training data
    )


@dataclass
class TrajectoryLoggingConfig:
    """
    Configuration for saving trajectory and step logs to disk.
    """

    enabled: bool = False
    log_dir: str = "logs/predictive_trajectories"
    include_step_chat_completions: bool = True
    include_final_messages: bool = True
    pretty_json: bool = True


class PredictiveToolWorkflow(Workflow):
    """
    Workflow that adds an explicit *prediction* sub-step before executing tool action.

    Per turn:
    1) model produces action (tool calls)
    2) model predicts the outcome of executing that action (with explicit tags)
    3) environment executes the action and returns tool outputs / observations

    Design goals:
    - Use existing rllm v0.2 workflow training stack (AgentWorkflowEngine + AgentWorkflowPPOTrainer)
    - Keep all changes isolated from core `rllm` by implementing this workflow in `rllm_ext`
    - Store prediction artifacts in `Step.info` for future loss design
    """

    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        max_steps: int = 5,
        prediction_cfg: dict[str, Any] | PredictionConfig | None = None,
        trajectory_logging: dict[str, Any] | TrajectoryLoggingConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Resolve mappings if strings are provided (mirrors existing workflows).
        agent_cls = (
            AGENT_CLASS_MAPPING[agent_cls] if isinstance(agent_cls, str) else agent_cls
        )
        env_cls = ENV_CLASS_MAPPING[env_cls] if isinstance(env_cls, str) else env_cls

        agent_args = dict(agent_args) if agent_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}

        # Instantiate
        self.agent = agent_cls(**agent_args)
        self.env = env_cls(**env_args)
        self.max_steps = max_steps

        if prediction_cfg is None:
            self.prediction_cfg = PredictionConfig()
        elif isinstance(prediction_cfg, PredictionConfig):
            self.prediction_cfg = prediction_cfg
        else:
            self.prediction_cfg = PredictionConfig(**prediction_cfg)

        if trajectory_logging is None:
            self.trajectory_logging = TrajectoryLoggingConfig()
        elif isinstance(trajectory_logging, TrajectoryLoggingConfig):
            self.trajectory_logging = trajectory_logging
        else:
            self.trajectory_logging = TrajectoryLoggingConfig(**trajectory_logging)

        if self.prediction_cfg.enabled and not isinstance(
            self.agent, PredictiveToolAgent
        ):
            # We don't hard-require PredictiveToolAgent, but it provides a clean storage API.
            # Keeping this as a runtime check makes failure modes obvious.
            raise TypeError(
                f"PredictiveToolWorkflow requires agent_cls to be PredictiveToolAgent when prediction is enabled, got {type(self.agent)}"
            )

    def _build_prediction_prompt(self, action_obj: Any) -> Optional[str]:
        """
        Build a prediction prompt that encourages the model to reason before predicting.

        The prompt asks the model to output in the same format as tool calls:
        - <prediction> tag acts as the separator (like  in tool calls)
        - Content before <prediction> is reasoning
        - Content inside <prediction> tags is the final prediction

        This structure allows us to reuse the existing reasoning parsing logic.
        """
        # ToolAgent emits OpenAI-style tool calls as list[dict]; we keep it generic.
        try:
            action_json = json.dumps(action_obj, ensure_ascii=False)
        except Exception:
            action_json = str(action_obj)
        # Skip prediction when action format is unexpected or contains no executable tool call.
        if not isinstance(action_obj, list) or not action_obj:
            return None
        has_real_tool_call = False
        for tool_call in action_obj:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            tool_name = function.get("name")
            if tool_name and tool_name != "finish":
                has_real_tool_call = True
                break
        if not has_real_tool_call:
            return None

        return (
            "You are now in **PREDICTION MODE**.\n"
            "- Do NOT call any tools.\n"
            "- Do NOT output any ``` ... ``` blocks.\n\n"
            "Your task: Predict what the tool will output.\n\n"
            "Format (use <prediction> tag as separator):\n"
            "1. First, think step-by-step about what the tool will do (this is your reasoning)\n"
            "2. Then, provide your final prediction inside <prediction> tags\n\n"
            "Example:\n"
            "The tool will execute python code 'print(1+1)'. This calculates 1+1, so the result will be 2.\n"
            "<prediction>2</prediction>\n\n"
            f"ACTION_JSON:\n{action_json}\n\n"
            "Now provide your reasoning and prediction:\n"
        )

    @staticmethod
    def _extract_tagged_block(text: str, tag: str) -> str | None:
        """
        Extract the inner content of a simple XML-like block: <tag> ... </tag>.
        Returns None if not found.
        """
        if not text:
            return None
        begin = f"<{tag}>"
        end = f"</{tag}>"
        start = text.find(begin)
        if start == -1:
            return None
        start += len(begin)
        stop = text.find(end, start)
        if stop == -1:
            return None
        return text[start:stop].strip()

    @staticmethod
    def _extract_actual_tool_outputs(observation: Any) -> tuple[dict[str, str], str]:
        """
        Extract post-action tool outputs from environment observation.

        Returns:
            - tool_outputs map
            - concatenated output text
        """
        if not isinstance(observation, dict):
            return {}, ""

        tool_outputs = observation.get("tool_outputs")
        if not isinstance(tool_outputs, dict) or not tool_outputs:
            return {}, ""

        normalized_outputs = {str(k): str(v) for k, v in tool_outputs.items()}
        ordered_output = " ".join(
            normalized_outputs[k] for k in sorted(normalized_outputs.keys())
        )
        return normalized_outputs, ordered_output

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """
        Convert values into JSON-serializable data recursively.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(k): PredictiveToolWorkflow._to_jsonable(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PredictiveToolWorkflow._to_jsonable(v) for v in value]
        if hasattr(value, "value"):
            return getattr(value, "value")
        return str(value)

    @staticmethod
    def _sanitize_uid(uid: str) -> str:
        return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in uid)

    def _step_log_record(self, step_idx: int, step) -> dict[str, Any]:
        step_info = step.info if isinstance(step.info, dict) else {}
        prediction_record = (
            step_info.get(PredictiveToolAgent.INFO_KEY_PREDICTION, {}) or {}
        )
        prediction_metadata = (
            prediction_record.get("metadata", {})
            if isinstance(prediction_record, dict)
            else {}
        )

        actual_output = step_info.get(PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT, "")
        if not isinstance(actual_output, str):
            actual_output = str(actual_output)

        return {
            "step_idx": step_idx,
            "reward": float(step.reward),
            "done": bool(step.done),
            "model_response": step.model_response,
            "action": self._to_jsonable(step.action),
            "observation_pre_action": self._to_jsonable(step.observation),
            "tool_output": {
                "actual_output_text": actual_output,
                "tool_outputs": self._to_jsonable(
                    step_info.get(PredictiveToolAgent.INFO_KEY_ACTUAL_TOOL_OUTPUTS, {})
                ),
            },
            "predict_target": {
                "prediction": prediction_record.get("prediction", "")
                if isinstance(prediction_record, dict)
                else "",
                "actual": actual_output,
                "has_prediction": bool(prediction_record.get("prediction"))
                if isinstance(prediction_record, dict)
                else False,
                "prompt": prediction_record.get("prompt", "")
                if isinstance(prediction_record, dict)
                else "",
                "prediction_reasoning": prediction_metadata.get("reasoning", "")
                if isinstance(prediction_metadata, dict)
                else "",
                "prediction_raw_text": prediction_metadata.get("raw_text", "")
                if isinstance(prediction_metadata, dict)
                else "",
            },
            "step_info": self._to_jsonable(step_info),
            "whole_messages_step_snapshot": self._to_jsonable(step.chat_completions)
            if self.trajectory_logging.include_step_chat_completions
            else None,
        }

    def _save_episode_log(
        self, episode: Episode, uid: str, termination_reason: TerminationReason
    ) -> None:
        if not self.trajectory_logging.enabled:
            return

        try:
            log_dir = Path(self.trajectory_logging.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            if episode.trajectories:
                trajectory = episode.trajectories[0]
                step_logs = [
                    self._step_log_record(step_idx, step)
                    for step_idx, step in enumerate(trajectory.steps)
                ]
                trajectory_task = trajectory.task
            else:
                step_logs = []
                trajectory_task = ""

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            safe_uid = self._sanitize_uid(uid)
            file_path = log_dir / f"{timestamp}_{safe_uid}.json"

            payload = {
                "episode_id": uid,
                "timestamp_utc": timestamp,
                "termination_reason": termination_reason.value
                if isinstance(termination_reason, TerminationReason)
                else str(termination_reason),
                "task": self._to_jsonable(episode.task),
                "trajectory_task": self._to_jsonable(trajectory_task),
                "metrics": self._to_jsonable(episode.metrics),
                "num_steps": len(step_logs),
                "steps": step_logs,
                "whole_messages_final": self._to_jsonable(self.agent.messages)
                if self.trajectory_logging.include_final_messages
                else None,
            }

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(
                    payload,
                    f,
                    ensure_ascii=False,
                    indent=2 if self.trajectory_logging.pretty_json else None,
                )
                if not self.trajectory_logging.pretty_json:
                    f.write("\n")
        except Exception as exc:
            logger.warning("Failed to save trajectory log for uid=%s: %s", uid, exc)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute a multi-step tool workflow, with an extra prediction call per step.
        """
        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0.0, False, info)

        for step_idx in range(self.max_steps):
            # 1) Action
            output: ModelOutput = await self.rollout_engine.get_model_response(
                self.agent.chat_completions,
                application_id=f"{uid}:act:{step_idx}",
                **kwargs,
            )
            response = output.text
            action = self.agent.update_from_model(
                response, return_action_dict=True
            )  # Get action but don't append to messages yet
            raw_action = action.action
            action_reasoning = output.reasoning  # Extract reasoning from action output

            # 2) Prediction sub-step
            prediction_prompt = None
            prediction_payload = None
            prediction_step_enabled = bool(
                getattr(self.prediction_cfg, "enable_prediction_step", False)
            )
            prediction_loss_enabled = bool(
                getattr(self.prediction_cfg, "enable_prediction_loss", False)
            )
            # When prediction_step is disabled but loss is enabled, embed prediction
            # messages into chat_completions to reuse the main forward pass.
            embed_prediction_in_chat = (
                prediction_loss_enabled and not prediction_step_enabled
            )

            if prediction_step_enabled or prediction_loss_enabled:
                prediction_prompt = self._build_prediction_prompt(raw_action)
                if prediction_prompt is not None:
                    cur_step = self.agent.get_current_state()
                    # Attach action reasoning to the assistant message.
                    if action_reasoning:
                        if (
                            self.agent.messages
                            and self.agent.messages[-1].get("role") == "assistant"
                        ):
                            self.agent.messages[-1]["reasoning"] = action_reasoning
                        if (
                            cur_step is not None
                            and cur_step.chat_completions
                            and cur_step.chat_completions[-1].get("role") == "assistant"
                        ):
                            cur_step.chat_completions[-1]["reasoning"] = (
                                action_reasoning
                            )

                    if prediction_step_enabled:
                        # OLD behaviour: model generates a prediction via a separate
                        # rollout call.  Prediction messages are appended to both
                        # agent.messages and cur_step.chat_completions.
                        self.agent.set_step_prediction(
                            prediction=PredictionRecord(
                                prompt=prediction_prompt,
                                prediction="",
                                metadata={
                                    "step_idx": step_idx,
                                    "raw_text": "",
                                    "reasoning": "",
                                    "prediction_rollout_skipped": False,
                                },
                            )
                        )

                        pred_messages = self.agent.chat_completions.copy()
                        pred_messages.append(
                            {"role": "user", "content": prediction_prompt}
                        )

                        pred_output: ModelOutput = (
                            await self.rollout_engine.get_model_response(
                                pred_messages,
                                application_id=f"{uid}:pred:{step_idx}",
                                max_tokens=self.prediction_cfg.max_tokens,
                                **kwargs,
                            )
                        )

                        prediction_raw_text = pred_output.text or ""
                        prediction_reasoning = None
                        prediction_text = None
                        if "<prediction>" in prediction_raw_text:
                            reasoning_part, _, _ = prediction_raw_text.partition(
                                "<prediction>"
                            )
                            prediction_reasoning = reasoning_part.strip()
                            prediction_text = self._extract_tagged_block(
                                prediction_raw_text, "prediction"
                            )

                        self.agent.set_step_prediction(
                            prediction=PredictionRecord(
                                prompt=prediction_prompt,
                                prediction=prediction_text or "",
                                metadata={
                                    "step_idx": step_idx,
                                    "raw_text": prediction_raw_text,
                                    "reasoning": prediction_reasoning,
                                    "prediction_rollout_skipped": False,
                                },
                            )
                        )
                        prediction_payload = {
                            "text": prediction_text or "",
                            "raw_text": prediction_raw_text or "",
                            "prompt": prediction_prompt,
                        }

                        prediction_prompt_message = {
                            "role": "user",
                            "content": prediction_prompt,
                        }
                        pred_message = {
                            "role": "assistant",
                            "content": prediction_text or "",
                        }
                        if prediction_reasoning:
                            pred_message["reasoning"] = prediction_reasoning

                        if self.prediction_cfg.add_prediction_to_messages:
                            self.agent.messages.append(prediction_prompt_message)
                            self.agent.messages.append(pred_message)

                        if cur_step is not None:
                            cur_step.chat_completions.append(
                                copy.deepcopy(prediction_prompt_message)
                            )
                            cur_step.chat_completions.append(
                                copy.deepcopy(pred_message)
                            )

                    elif embed_prediction_in_chat:
                        # NEW: store minimal prediction record for logging only.
                        # The actual prediction loss data is embedded directly into
                        # chat_completions after env.step (see step 4 below).
                        self.agent.set_step_prediction(
                            prediction=PredictionRecord(
                                prompt=prediction_prompt,
                                prediction="",
                                metadata={
                                    "step_idx": step_idx,
                                    "prediction_embedded_in_chat": True,
                                },
                            )
                        )

            # 3) Execute in env.
            env_action = raw_action
            if prediction_payload is not None and hasattr(
                self.env, "similarity_config"
            ):
                env_action = {
                    "tool_calls": raw_action,
                    "prediction": prediction_payload,
                }
            next_obs, reward, done, step_info = await self.run_in_executor(
                self.env.step, env_action
            )

            # 4) Embed prediction messages into chat_completions (before update_from_env).
            # When embed_prediction_in_chat is active, insert prediction user + assistant
            # messages containing the actual tool output BEFORE the env observation messages.
            if embed_prediction_in_chat and prediction_prompt is not None:
                _, actual_output = self._extract_actual_tool_outputs(next_obs)
                if actual_output:
                    cur_step = self.agent.get_current_state()
                    if cur_step is not None:
                        pred_msg_user = {"role": "user", "content": prediction_prompt}
                        pred_msg_asst = {
                            "role": "assistant",
                            "content": f"<prediction>{actual_output}</prediction>",
                            "rllm_prediction": True,
                        }
                        cur_step.chat_completions.append(copy.deepcopy(pred_msg_user))
                        cur_step.chat_completions.append(copy.deepcopy(pred_msg_asst))

                        if self.prediction_cfg.add_prediction_to_messages:
                            self.agent.messages.append(pred_msg_user)
                            self.agent.messages.append(pred_msg_asst)

            self.agent.update_from_env(next_obs, reward, done, step_info)

            # Update the current Step fields for training
            cur_step = self.agent.get_current_state()
            if cur_step is not None:
                cur_step.reward = float(reward)
                cur_step.done = bool(done)
                cur_step.info.update(step_info or {})
                actual_tool_outputs, actual_output = self._extract_actual_tool_outputs(
                    next_obs
                )
                cur_step.info[PredictiveToolAgent.INFO_KEY_ACTUAL_TOOL_OUTPUTS] = (
                    actual_tool_outputs
                )
                cur_step.info[PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT] = (
                    actual_output
                )

            # Check for early termination conditions
            if output.finish_reason == "length":
                # Model response exceeded max length
                episode = self._build_episode(
                    task, uid, TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
                )
                return episode

            if done:
                # Episode completed successfully
                episode = self._build_episode(task, uid, TerminationReason.ENV_DONE)
                return episode

        # Max steps exceeded
        episode = self._build_episode(task, uid, TerminationReason.MAX_TURNS_EXCEEDED)
        return episode

    def _has_real_tool_call(self, step) -> bool:
        """
        Check if a step contains a real tool call (not just finish or empty).

        Args:
            step: Step object to check

        Returns:
            True if step has a real tool call, False otherwise
        """
        action = step.action
        if not action or not isinstance(action, list):
            return False

        # Check if any tool call is not "finish"
        for tool_call in action:
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                if tool_name and tool_name != "finish":
                    return True

        return False

    def _build_episode(
        self, task: dict, uid: str, termination_reason: TerminationReason
    ) -> Episode:
        """
        Build an Episode from the agent's trajectory.

        Args:
            task: Original task dictionary
            uid: Episode unique identifier
            termination_reason: Reason for episode termination

        Returns:
            Episode object with trajectory
        """
        # Get the trajectory from the agent (ToolAgent maintains its own trajectory)
        agent_trajectory = copy.deepcopy(self.agent.trajectory)

        # Filter out steps without tool calls if simple_tir is enabled
        if self.prediction_cfg.simple_tir:
            original_steps = agent_trajectory.steps
            filtered_steps = [
                step for step in original_steps if self._has_real_tool_call(step)
            ]
            agent_trajectory.steps = filtered_steps

        # Set the task on the trajectory
        agent_trajectory.task = task.get("question", task.get("task", ""))

        # Create episode
        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = termination_reason
        episode.trajectories = [agent_trajectory]

        # Compute episode-level correctness based on total reward
        total_reward = sum(step.reward for step in agent_trajectory.steps)
        agent_trajectory.reward = float(total_reward)
        episode.is_correct = total_reward > 0

        # Add basic metrics
        episode.metrics = {
            "num_steps": len(agent_trajectory.steps),
            "total_reward": total_reward,
            "prediction_enabled": self.prediction_cfg.enabled,
            "simple_tir": self.prediction_cfg.simple_tir,
        }

        self._save_episode_log(episode, uid, termination_reason)

        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        super().reset(task, uid)
        # Keep env reset compatible with ToolEnvironment (returns (task, info))
        return self.env.reset(task)
