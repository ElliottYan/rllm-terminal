from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

from rllm_ext.agents.predictive_tool_agent import PredictionRecord, PredictiveToolAgent


@dataclass
class PredictionConfig:
    """
    Configuration for prediction sub-step.
    """

    enabled: bool = True
    max_tokens: int = 256
    add_prediction_to_messages: bool = True  # if True, prediction becomes part of training text


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

    def _build_prediction_prompt(self, action_obj: Any) -> str:
        """
        Build a stable, minimal instruction that asks the model to predict what will happen.
        Uses explicit <prediction>...</prediction> tags for extraction.
        """
        # ToolAgent emits OpenAI-style tool calls as list[dict]; we keep it generic.
        try:
            action_json = json.dumps(action_obj, ensure_ascii=False)
        except Exception:
            action_json = str(action_obj)

        return (
            "You are now in **PREDICTION MODE**.\n"
            "- Do NOT call any tools.\n"
            "- Do NOT output any ``` ... ``` blocks.\n"
            "- Output MUST follow this exact format:\n\n"
            "<prediction>\n"
            "... final predicted tool result text only ...\n"
            "</prediction>\n\n"
            f"ACTION_JSON:\n{action_json}\n"
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

    async def run(self, task: dict, uid: str, **kwargs):
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

            # 2) Prediction sub-step
            prediction_text = None
            prediction_raw = None
            if self.prediction_cfg.enabled:
                prediction_prompt = self._build_prediction_prompt(raw_action)
                pred_messages = self.agent.chat_completions.copy()
                pred_messages.append({"role": "user", "content": prediction_prompt})

                pred_output: ModelOutput = await self.rollout_engine.get_model_response(
                    pred_messages,
                    application_id=f"{uid}:pred:{step_idx}",
                    max_tokens=self.prediction_cfg.max_tokens,
                    **kwargs,
                )

                # Extract prediction from <prediction>...</prediction> tags
                prediction_raw_text = pred_output.text or ""
                prediction_text = self._extract_tagged_block(prediction_raw_text, "prediction")

                # Store for future loss design
                self.agent.set_step_prediction(
                    prediction=PredictionRecord(
                        prompt=prediction_prompt,
                        prediction=prediction_text,
                        metadata={
                            "step_idx": step_idx,
                            "raw_text": prediction_raw_text,
                        },
                    )
                )

                # Optionally make prediction part of actual trajectory text (so it can be learned via RL later)
                if self.prediction_cfg.add_prediction_to_messages:
                    # NOTE: this directly mutates ToolAgent's message history (kept intentionally isolated to rllm_ext).
                    # Insert prediction messages BEFORE the action response
                    # Order: action -> prediction -> tool_output
                    self.agent.messages.append({"role": "user", "content": prediction_prompt})
                    self.agent.messages.append({"role": "assistant", "content": prediction_raw_text})
                    # Now append the action response (tool calls)
                    self.agent.messages.append({"role": "assistant", "content": response, "tool_calls": raw_action})

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

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def reset(self, task: dict | None = None, uid: str | None = None):
        super().reset(task, uid)
        # Keep env reset compatible with ToolEnvironment (returns (task, info))
        return self.env.reset(task)
