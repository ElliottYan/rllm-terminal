from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Optional

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
from rllm.workflows.workflow import TerminationReason, Workflow

from rllm_ext.agents.predictive_tool_agent import PredictionRecord, PredictiveToolAgent


@dataclass
class PredictionConfig:
    """
    Configuration for prediction sub-step.
    """

    enabled: bool = True
    max_tokens: int = 256
    add_prediction_to_messages: bool = True  # if True, prediction becomes part of training text
    simple_tir: bool = False  # if True, filter out steps without tool calls from training data


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
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Resolve mappings if strings are provided (mirrors existing workflows).
        agent_cls = AGENT_CLASS_MAPPING[agent_cls] if isinstance(agent_cls, str) else agent_cls
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

        if self.prediction_cfg.enabled and not isinstance(self.agent, PredictiveToolAgent):
            # We don't hard-require PredictiveToolAgent, but it provides a clean storage API.
            # Keeping this as a runtime check makes failure modes obvious.
            raise TypeError(f"PredictiveToolWorkflow requires agent_cls to be PredictiveToolAgent when prediction is enabled, got {type(self.agent)}")

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
        # if action_json['function']
        try:
            assert len(action_obj) == 1
            assert action_obj[0]['type'] == 'function'
        except:
            breakpoint()
        if action_obj[0]['function']['name'] == 'finish':
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

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute a multi-step tool workflow, with an extra prediction call per step.
        """
        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0.0, False, info)

        for step_idx in range(self.max_steps):
            # 1) Action
            output: ModelOutput = await self.rollout_engine.get_model_response(self.agent.chat_completions, application_id=f"{uid}:act:{step_idx}", **kwargs)
            response = output.text
            action = self.agent.update_from_model(response, return_action_dict=True)  # Get action but don't append to messages yet
            raw_action = action.action
            action_reasoning = output.reasoning  # Extract reasoning from action output

            # 2) Prediction sub-step
            prediction_text = None
            prediction_reasoning = None
            prediction_raw_text = None
            prediction_prompt = None
            if self.prediction_cfg.enabled:
                prediction_prompt = self._build_prediction_prompt(raw_action)
                # the tool call is not a finish tool.
                if prediction_prompt is not None:
                    pred_messages = self.agent.chat_completions.copy()
                    pred_messages.append({"role": "user", "content": prediction_prompt})

                    pred_output: ModelOutput = await self.rollout_engine.get_model_response(
                        pred_messages,
                        application_id=f"{uid}:pred:{step_idx}",
                        max_tokens=self.prediction_cfg.max_tokens,
                        **kwargs,
                    )

                    # Parse prediction: extract reasoning (before <prediction>) and content (inside <prediction>)
                    # This mirrors the parse_completion logic for tool calls which uses  as separator
                    prediction_raw_text = pred_output.text or ""

                    if "<prediction>" in prediction_raw_text:
                        # Split on first <prediction> tag (mirrors partition(")") logic)
                        reasoning_part, _, prediction_part = prediction_raw_text.partition("<prediction>")
                        prediction_reasoning = reasoning_part.strip()

                        # Extract content between <prediction> and </prediction>
                        prediction_text = self._extract_tagged_block(prediction_raw_text, "prediction")
                    else:
                        # No <prediction> tag found - treat as error/missing prediction
                        prediction_text = None

                    # Store for future loss design
                    self.agent.set_step_prediction(
                        prediction=PredictionRecord(
                            prompt=prediction_prompt,
                            prediction=prediction_text,
                            metadata={
                                "step_idx": step_idx,
                                "raw_text": prediction_raw_text,
                                "reasoning": prediction_reasoning,
                            },
                        )
                    )

                    # Optionally make prediction part of actual trajectory text (so it can be learned via RL later)
                    if self.prediction_cfg.add_prediction_to_messages:
                        # NOTE: this directly mutates ToolAgent's message history (kept intentionally isolated to rllm_ext).
                        # Insert prediction messages BEFORE the action response
                        # Order: action -> prediction -> tool_output
                        self.agent.messages.append({"role": "user", "content": prediction_prompt})

                        # Store prediction with reasoning field (same format as action messages)
                        # This ensures tokenize_and_mask will handle it correctly during training
                        pred_message = {"role": "assistant", "content": prediction_text or ""}
                        if prediction_reasoning:
                            pred_message["reasoning"] = prediction_reasoning
                        self.agent.messages.append(pred_message)

                        # Now append the action response (tool calls) with reasoning field
                        action_message = {"role": "assistant", "content": output.content, "tool_calls": raw_action}
                        if action_reasoning:
                            action_message["reasoning"] = action_reasoning
                        self.agent.messages.append(action_message)

            # 3) Execute in env (ToolEnvironment expects raw tool_calls list/dict/str, not Action dataclass)
            next_obs, reward, done, step_info = await self.run_in_executor(self.env.step, raw_action)
            self.agent.update_from_env(next_obs, reward, done, step_info)

            # Update the current Step fields for training
            cur_step = self.agent.get_current_state()
            if cur_step is not None:
                cur_step.reward = float(reward)
                cur_step.done = bool(done)
                cur_step.info.update(step_info or {})
                if prediction_text is not None:
                    cur_step.info["rllm_ext.prediction_text"] = prediction_text
                    cur_step.info["rllm_ext.prediction_raw_text"] = prediction_raw_text
                if prediction_prompt is not None:
                    cur_step.info["rllm_ext.prediction_prompt"] = prediction_prompt

            # Check for early termination conditions
            if output.finish_reason == "length":
                # Model response exceeded max length
                episode = self._build_episode(task, uid, TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
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

    def _build_episode(self, task: dict, uid: str, termination_reason: TerminationReason) -> Episode:
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
            filtered_steps = [step for step in original_steps if self._has_real_tool_call(step)]
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
        episode.is_correct = total_reward > 0

        # Add basic metrics
        episode.metrics = {
            "num_steps": len(agent_trajectory.steps),
            "total_reward": total_reward,
            "prediction_enabled": self.prediction_cfg.enabled,
            "simple_tir": self.prediction_cfg.simple_tir,
        }

        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        super().reset(task, uid)
        # Keep env reset compatible with ToolEnvironment (returns (task, info))
        return self.env.reset(task)
