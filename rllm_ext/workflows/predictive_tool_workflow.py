from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
from rllm.workflows.workflow import TerminationReason, Workflow

from rllm_ext.agents.predictive_tool_agent import PredictionRecord, PredictiveToolAgent

logger = logging.getLogger(__name__)


GenerativeSupportMode = Literal[
    "none",
    "pre_action_world_model",
    "post_action_simulator",
]


@dataclass
class PredictionConfig:
    """
    Legacy prediction config kept for backward compatibility.
    """

    enabled: bool = True
    enforce_max_prompt_length: bool = True
    enable_prediction_loss: bool = False
    enable_prediction_step: bool = False
    max_tokens: int = 256
    add_prediction_to_messages: bool = True
    simple_tir: bool = False


@dataclass
class GenerativeSupportConfig:
    """
    Unified config for baseline prediction, scheme A, and scheme B.

    Notes:
    - ``mode="none"`` means the baseline ``action -> prediction -> env`` flow.
    - ``simple_tir`` is kept only for backward compatibility with existing
      trajectory filtering behavior.
    """

    mode: GenerativeSupportMode = "none"
    max_tokens: int = 256
    add_to_live_messages: bool = False
    add_to_step_chat_completions: bool = True
    enable_prediction: bool = True
    train_generative_with_ppo: bool = True
    enforce_max_prompt_length: bool = True
    simple_tir: bool = False


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
    Tool workflow with optional generative support around tool execution.

    Supported per-step flows:
    - baseline: ``action -> prediction -> env`` (``mode="none"``)
    - scheme A: ``candidate -> finalize(action+prediction) -> env``
    - scheme B: ``action -> simulator -> prediction -> env``
    """

    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        max_steps: int = 5,
        prediction_cfg: dict[str, Any] | GenerativeSupportConfig | None = None,
        generative_support_cfg: dict[str, Any] | GenerativeSupportConfig | None = None,
        trajectory_logging: dict[str, Any] | TrajectoryLoggingConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        agent_cls = (
            AGENT_CLASS_MAPPING[agent_cls] if isinstance(agent_cls, str) else agent_cls
        )
        env_cls = ENV_CLASS_MAPPING[env_cls] if isinstance(env_cls, str) else env_cls

        agent_args = dict(agent_args) if agent_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}

        self.agent = agent_cls(**agent_args)
        self.env = env_cls(**env_args)
        self.max_steps = max_steps

        if prediction_cfg is not None and generative_support_cfg is not None:
            raise ValueError(
                "Use only one of prediction_cfg or generative_support_cfg."
            )

        cfg_source = (
            generative_support_cfg
            if generative_support_cfg is not None
            else prediction_cfg
        )
        self._uses_generative_support_cfg = generative_support_cfg is not None or (
            isinstance(cfg_source, dict) and self._contains_new_support_keys(cfg_source)
        ) or isinstance(cfg_source, GenerativeSupportConfig)

        if self._uses_generative_support_cfg:
            self.generative_support_cfg = self._resolve_support_cfg(cfg_source)
            self.prediction_cfg = self.generative_support_cfg
            self.legacy_prediction_cfg = None
        else:
            self.legacy_prediction_cfg = self._resolve_legacy_prediction_cfg(cfg_source)
            self.prediction_cfg = self.legacy_prediction_cfg
            self.generative_support_cfg = GenerativeSupportConfig(
                mode="none",
                max_tokens=self.legacy_prediction_cfg.max_tokens,
                add_to_live_messages=self.legacy_prediction_cfg.add_prediction_to_messages,
                add_to_step_chat_completions=self.legacy_prediction_cfg.add_prediction_to_messages,
                enable_prediction=bool(
                    self.legacy_prediction_cfg.enabled
                    and self.legacy_prediction_cfg.enable_prediction_step
                ),
                train_generative_with_ppo=False,
                enforce_max_prompt_length=self.legacy_prediction_cfg.enforce_max_prompt_length,
                simple_tir=self.legacy_prediction_cfg.simple_tir,
            )

        if trajectory_logging is None:
            self.trajectory_logging = TrajectoryLoggingConfig()
        elif isinstance(trajectory_logging, TrajectoryLoggingConfig):
            self.trajectory_logging = trajectory_logging
        else:
            self.trajectory_logging = TrajectoryLoggingConfig(**trajectory_logging)

        support_enabled = self._active_prediction_enabled() or self._active_mode() != "none"
        if support_enabled and not isinstance(self.agent, PredictiveToolAgent):
            raise TypeError(
                "PredictiveToolWorkflow requires PredictiveToolAgent when "
                "prediction or generative support is enabled."
            )

    @staticmethod
    def _contains_new_support_keys(cfg: dict[str, Any]) -> bool:
        return any(
            key in cfg
            for key in (
                "mode",
                "add_to_live_messages",
                "add_to_step_chat_completions",
                "enable_prediction",
                "train_generative_with_ppo",
            )
        )

    @staticmethod
    def _resolve_legacy_prediction_cfg(
        cfg: dict[str, Any] | PredictionConfig | None,
    ) -> PredictionConfig:
        if cfg is None:
            return PredictionConfig()
        if isinstance(cfg, PredictionConfig):
            return cfg
        return PredictionConfig(**dict(cfg))

    @staticmethod
    def _resolve_support_cfg(
        cfg: dict[str, Any] | GenerativeSupportConfig | None,
    ) -> GenerativeSupportConfig:
        if cfg is None:
            return GenerativeSupportConfig()
        if isinstance(cfg, GenerativeSupportConfig):
            return cfg

        cfg_dict = dict(cfg)

        # Legacy field compatibility.
        if "add_prediction_to_messages" in cfg_dict:
            cfg_dict.setdefault(
                "add_to_live_messages",
                bool(cfg_dict.pop("add_prediction_to_messages")),
            )
        if "enabled" in cfg_dict:
            enabled = bool(cfg_dict.pop("enabled"))
            if not enabled:
                cfg_dict.setdefault("enable_prediction", False)
                cfg_dict.setdefault("mode", "none")
        if "enable_prediction_step" in cfg_dict or "enable_prediction_loss" in cfg_dict:
            cfg_dict.setdefault(
                "enable_prediction",
                bool(cfg_dict.pop("enable_prediction_step", False))
                or bool(cfg_dict.pop("enable_prediction_loss", False)),
            )

        return GenerativeSupportConfig(**cfg_dict)

    def _active_mode(self) -> str:
        if self._uses_generative_support_cfg:
            return self.generative_support_cfg.mode
        return "legacy"

    def _active_prediction_enabled(self) -> bool:
        if self._uses_generative_support_cfg:
            return self.generative_support_cfg.enable_prediction
        return bool(self.legacy_prediction_cfg and self.legacy_prediction_cfg.enabled)

    def _active_simple_tir(self) -> bool:
        if self._uses_generative_support_cfg:
            return self.generative_support_cfg.simple_tir
        return bool(self.legacy_prediction_cfg and self.legacy_prediction_cfg.simple_tir)

    @staticmethod
    def _is_finish_tool_name(tool_name: Any) -> bool:
        return isinstance(tool_name, str) and tool_name == "finish"

    @classmethod
    def _has_real_tool_call_payload(cls, action_obj: Any) -> bool:
        if not isinstance(action_obj, list) or not action_obj:
            return False

        for tool_call in action_obj:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            tool_name = function.get("name")
            if isinstance(tool_name, str) and tool_name and not cls._is_finish_tool_name(
                tool_name
            ):
                return True

        return False

    @staticmethod
    def _normalize_tool_call_function(function_payload: Any) -> dict[str, Any] | None:
        if hasattr(function_payload, "to_dict"):
            function_payload = function_payload.to_dict()

        if not isinstance(function_payload, dict):
            return None

        if "function" in function_payload and isinstance(
            function_payload["function"], dict
        ):
            function_payload = function_payload["function"]

        if "name" not in function_payload:
            return None

        normalized = copy.deepcopy(function_payload)
        arguments = normalized.get("arguments")
        if isinstance(arguments, dict):
            normalized["arguments"] = json.dumps(arguments, ensure_ascii=False)
        return normalized

    def _parse_tool_calls_from_response(self, response: str) -> list[dict[str, Any]]:
        if not isinstance(response, str) or not response.strip():
            return []

        tool_parser = getattr(self.agent, "tool_parser", None)
        if tool_parser is None:
            return []

        try:
            parsed_tool_calls = tool_parser.parse(response)
        except Exception as exc:
            logger.debug("Failed to parse tool calls from response: %s", exc)
            return []

        normalized_calls = []
        for tool_call in parsed_tool_calls:
            function_payload = self._normalize_tool_call_function(tool_call)
            if function_payload is None:
                continue
            normalized_calls.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": function_payload,
                }
            )

        return normalized_calls

    @staticmethod
    def _action_signature(action_obj: Any) -> list[dict[str, Any]]:
        if not isinstance(action_obj, list):
            return []

        signature = []
        for tool_call in action_obj:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    pass
            signature.append(
                {
                    "name": function.get("name"),
                    "arguments": PredictiveToolWorkflow._to_jsonable(arguments),
                }
            )
        return signature

    @classmethod
    def _actions_match(cls, lhs: Any, rhs: Any) -> bool:
        return cls._action_signature(lhs) == cls._action_signature(rhs)

    @staticmethod
    def _build_json_prompt_block(label: str, payload: Any) -> str:
        try:
            json_payload = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            json_payload = str(payload)
        return f"{label}:\n{json_payload}\n"

    def _build_prediction_prompt(self, action_obj: Any) -> Optional[str]:
        if not self._has_real_tool_call_payload(action_obj):
            return None

        return (
            "You are now in PREDICTION MODE.\n"
            "- Do NOT call any tools.\n"
            "- Do NOT output code fences.\n"
            "- If the tool may fail, say so briefly before the prediction.\n\n"
            + self._build_json_prompt_block("ACTION_JSON", action_obj)
            + "Respond in this format:\n"
            "Reasoning if needed.\n"
            "<prediction>short tool output prediction</prediction>\n"
        )

    def _build_finalize_prompt(self, candidate_action: Any) -> str:
        prediction_line = (
            "Then output a short <prediction>...</prediction> block that matches "
            "the likely real tool output as closely as possible.\n"
            if self.generative_support_cfg.enable_prediction
            else ""
        )
        return (
            "You are revising a candidate tool action before execution.\n"
            "First imagine the key consequences of executing the candidate action.\n"
            "Then decide whether to keep it or revise it.\n"
            + prediction_line
            + "Finally output the tool call that should actually be executed.\n\n"
            + self._build_json_prompt_block(
                "CANDIDATE_ACTION_JSON", candidate_action
            )
            + "Required structure:\n"
            "Optional reasoning.\n"
            "<simulation>natural language imagined outcome</simulation>\n"
            + (
                "<prediction>short tool output prediction</prediction>\n"
                if self.generative_support_cfg.enable_prediction
                else ""
            )
            + "Final tool call.\n"
        )

    def _build_simulator_prompt(self, action_obj: Any) -> Optional[str]:
        if not self._has_real_tool_call_payload(action_obj):
            return None
        return (
            "You are now in SIMULATION MODE.\n"
            "The action is already fixed. Do not revise it.\n"
            "Expand what this exact tool call will likely return, including likely "
            "failure modes and uncertainty.\n\n"
            + self._build_json_prompt_block("ACTION_JSON", action_obj)
            + "Respond in this format:\n"
            "Optional reasoning.\n"
            "<simulation>likely output / likely failure / uncertainty</simulation>\n"
        )

    @staticmethod
    def _extract_tagged_block(text: str, tag: str) -> str | None:
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
    def _extract_reasoning_prefix(text: str, tag: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        begin = f"<{tag}>"
        start = text.find(begin)
        if start == -1:
            return text.strip()
        return text[:start].strip()

    @staticmethod
    def _strip_auxiliary_tags(text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        stripped = text
        for tag in ("simulation", "prediction"):
            stripped = re.sub(
                rf"<{tag}>.*?</{tag}>",
                "",
                stripped,
                flags=re.DOTALL,
            )
        return stripped.strip()

    @staticmethod
    def _extract_actual_tool_outputs(observation: Any) -> tuple[dict[str, str], str]:
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

    def _set_latest_assistant_reasoning(self, reasoning: str) -> None:
        if not reasoning:
            return

        if self.agent.messages and self.agent.messages[-1].get("role") == "assistant":
            self.agent.messages[-1]["reasoning"] = reasoning

        cur_step = self.agent.get_current_state()
        if (
            cur_step is not None
            and cur_step.chat_completions
            and cur_step.chat_completions[-1].get("role") == "assistant"
        ):
            cur_step.chat_completions[-1]["reasoning"] = reasoning

    def _patch_latest_assistant_message(
        self,
        *,
        tool_calls: Any,
        live_content: str,
        step_content: str | None = None,
    ) -> None:
        if step_content is None:
            step_content = live_content

        if self.agent.messages and self.agent.messages[-1].get("role") == "assistant":
            self.agent.messages[-1]["content"] = live_content
            if tool_calls:
                self.agent.messages[-1]["tool_calls"] = copy.deepcopy(tool_calls)
            else:
                self.agent.messages[-1].pop("tool_calls", None)

        cur_step = self.agent.get_current_state()
        if (
            cur_step is not None
            and cur_step.chat_completions
            and cur_step.chat_completions[-1].get("role") == "assistant"
        ):
            cur_step.chat_completions[-1]["content"] = step_content
            if tool_calls:
                cur_step.chat_completions[-1]["tool_calls"] = copy.deepcopy(tool_calls)
            else:
                cur_step.chat_completions[-1].pop("tool_calls", None)

    def _append_auxiliary_turn(
        self,
        *,
        user_prompt: str,
        assistant_text: str,
        assistant_reasoning: str = "",
        assistant_flags: dict[str, Any] | None = None,
        add_to_live_messages: bool,
        add_to_step_chat_completions: bool,
    ) -> None:
        if not assistant_text:
            return

        prompt_message = {"role": "user", "content": user_prompt}
        assistant_message = {"role": "assistant", "content": assistant_text}
        if assistant_reasoning:
            assistant_message["reasoning"] = assistant_reasoning
        if assistant_flags:
            assistant_message.update(assistant_flags)

        cur_step = self.agent.get_current_state()
        if add_to_live_messages:
            self.agent.messages.append(copy.deepcopy(prompt_message))
            self.agent.messages.append(copy.deepcopy(assistant_message))
        if add_to_step_chat_completions and cur_step is not None:
            cur_step.chat_completions.append(copy.deepcopy(prompt_message))
            cur_step.chat_completions.append(copy.deepcopy(assistant_message))

    def _build_prediction_payload(
        self,
        *,
        prompt: str | None,
        prediction_text: str,
        raw_text: str,
    ) -> dict[str, Any] | None:
        if not prompt:
            return None
        if not prediction_text and not raw_text:
            return None
        return {
            "text": prediction_text,
            "raw_text": raw_text,
            "prompt": prompt,
        }

    def _build_action_first_step(
        self,
        *,
        response: str,
        output: ModelOutput,
    ) -> dict[str, Any]:
        action = self.agent.update_from_model(response, return_action_dict=True)
        self._set_latest_assistant_reasoning(output.reasoning or "")
        return {
            "action": action.action,
            "step": self.agent.get_current_state(),
            "committed_output": output,
        }

    async def _run_prediction_call(
        self,
        *,
        uid: str,
        step_idx: int,
        action_obj: Any,
        application_suffix: str,
        base_messages: list[dict[str, Any]] | None = None,
        store_in_live_messages: bool,
        store_in_step_chat_completions: bool,
    ) -> tuple[str | None, dict[str, Any] | None]:
        prediction_prompt = (
            self._build_prediction_prompt(action_obj)
            if self.generative_support_cfg.enable_prediction
            else None
        )
        if prediction_prompt is None:
            return None, None

        pred_messages = (
            copy.deepcopy(base_messages)
            if base_messages is not None
            else copy.deepcopy(self.agent.chat_completions)
        )
        pred_messages.append({"role": "user", "content": prediction_prompt})

        pred_output: ModelOutput = await self.rollout_engine.get_model_response(
            pred_messages,
            application_id=f"{uid}:{application_suffix}:{step_idx}",
            enforce_max_prompt_length=self.generative_support_cfg.enforce_max_prompt_length,
            max_tokens=self.generative_support_cfg.max_tokens,
        )

        prediction_raw_text = pred_output.text or ""
        prediction_text = (
            self._extract_tagged_block(prediction_raw_text, "prediction") or ""
        )
        prediction_reasoning = self._extract_reasoning_prefix(
            prediction_raw_text, "prediction"
        )

        self.agent.set_step_prediction(
            prediction=PredictionRecord(
                prompt=prediction_prompt,
                prediction=prediction_text,
                metadata={
                    "step_idx": step_idx,
                    "raw_text": prediction_raw_text,
                    "reasoning": prediction_reasoning,
                    "mode": self.generative_support_cfg.mode,
                },
            )
        )

        self._append_auxiliary_turn(
            user_prompt=prediction_prompt,
            assistant_text=prediction_raw_text,
            assistant_reasoning=prediction_reasoning,
            assistant_flags={"rllm_prediction": True},
            add_to_live_messages=store_in_live_messages,
            add_to_step_chat_completions=store_in_step_chat_completions,
        )

        prediction_payload = self._build_prediction_payload(
            prompt=prediction_prompt,
            prediction_text=prediction_text,
            raw_text=prediction_raw_text,
        )
        return prediction_prompt, prediction_payload

    async def _run_pre_action_world_model_step(
        self,
        *,
        uid: str,
        step_idx: int,
        model_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_output: ModelOutput = await self.rollout_engine.get_model_response(
            self.agent.chat_completions,
            application_id=f"{uid}:cand:{step_idx}",
            enforce_max_prompt_length=self.generative_support_cfg.enforce_max_prompt_length,
            **model_kwargs,
        )
        candidate_response = candidate_output.text or ""
        candidate_action = self._parse_tool_calls_from_response(candidate_response)
        candidate_has_real_tool = self._has_real_tool_call_payload(candidate_action)

        if not candidate_has_real_tool:
            committed = self._build_action_first_step(
                response=candidate_response,
                output=candidate_output,
            )
            return {
                **committed,
                "env_action": committed["action"],
            }

        finalize_prompt = self._build_finalize_prompt(candidate_action)
        finalize_messages = copy.deepcopy(self.agent.chat_completions)
        finalize_messages.append({"role": "user", "content": finalize_prompt})

        finalize_output: ModelOutput = await self.rollout_engine.get_model_response(
            finalize_messages,
            application_id=f"{uid}:final:{step_idx}",
            enforce_max_prompt_length=self.generative_support_cfg.enforce_max_prompt_length,
            max_tokens=self.generative_support_cfg.max_tokens,
            **model_kwargs,
        )
        finalize_response = finalize_output.text or ""
        parsed_final_action = self._parse_tool_calls_from_response(finalize_response)
        final_has_real_tool = self._has_real_tool_call_payload(parsed_final_action)

        executed_action = (
            parsed_final_action if final_has_real_tool else copy.deepcopy(candidate_action)
        )
        action_revised = final_has_real_tool and not self._actions_match(
            candidate_action, parsed_final_action
        )
        final_prediction = (
            self._extract_tagged_block(finalize_response, "prediction") or ""
        )
        if not final_has_real_tool:
            final_prediction = ""

        prediction_prompt = (
            self._build_prediction_prompt(executed_action)
            if self.generative_support_cfg.enable_prediction
            else None
        )

        committed = self._build_action_first_step(
            response=finalize_response,
            output=finalize_output,
        )
        cur_step = committed["step"]
        if cur_step is not None:
            cur_step.action = copy.deepcopy(executed_action)
            cur_step.info[PredictiveToolAgent.INFO_KEY_CANDIDATE_ACTION] = copy.deepcopy(
                candidate_action
            )
            cur_step.info[PredictiveToolAgent.INFO_KEY_FINAL_ACTION] = copy.deepcopy(
                executed_action
            )
            cur_step.info[PredictiveToolAgent.INFO_KEY_ACTION_REVISED] = bool(
                action_revised
            )

        simulation_text = (
            self._extract_tagged_block(finalize_response, "simulation") or ""
        )
        self.agent.set_step_generative_support(
            mode="pre_action_world_model",
            prompt=finalize_prompt,
            text=finalize_response,
            metadata={
                "step_idx": step_idx,
                "candidate_has_real_tool": True,
                "final_tool_call_present": final_has_real_tool,
                "candidate_final_action_match": self._actions_match(
                    candidate_action, executed_action
                ),
                "simulation": simulation_text,
            },
        )

        if prediction_prompt is not None:
            self.agent.set_step_prediction(
                prediction=PredictionRecord(
                    prompt=prediction_prompt,
                    prediction=final_prediction,
                    metadata={
                        "step_idx": step_idx,
                        "raw_text": finalize_response,
                        "reasoning": self._extract_reasoning_prefix(
                            finalize_response, "prediction"
                        ),
                        "mode": "pre_action_world_model",
                    },
                )
            )

        live_content = (
            finalize_response
            if self.generative_support_cfg.add_to_live_messages
            else self._strip_auxiliary_tags(finalize_response)
        )
        keep_full_step_text = (
            self.generative_support_cfg.add_to_step_chat_completions
            and self.generative_support_cfg.train_generative_with_ppo
        )
        step_content = (
            finalize_response
            if keep_full_step_text
            else self._strip_auxiliary_tags(finalize_response)
        )
        self._patch_latest_assistant_message(
            tool_calls=executed_action,
            live_content=live_content,
            step_content=step_content,
        )

        prediction_payload = self._build_prediction_payload(
            prompt=prediction_prompt,
            prediction_text=final_prediction,
            raw_text=finalize_response,
        )
        return {
            **committed,
            "action": executed_action,
            "env_action": executed_action,
            "prediction_payload": prediction_payload,
        }

    async def _run_post_action_simulator_step(
        self,
        *,
        uid: str,
        step_idx: int,
        action_obj: Any,
    ) -> dict[str, Any]:
        simulator_prompt = self._build_simulator_prompt(action_obj)
        if simulator_prompt is None:
            return {"prediction_payload": None}

        sim_messages = copy.deepcopy(self.agent.chat_completions)
        sim_messages.append({"role": "user", "content": simulator_prompt})
        sim_output: ModelOutput = await self.rollout_engine.get_model_response(
            sim_messages,
            application_id=f"{uid}:sim:{step_idx}",
            enforce_max_prompt_length=self.generative_support_cfg.enforce_max_prompt_length,
            max_tokens=self.generative_support_cfg.max_tokens,
        )
        simulator_text = sim_output.text or ""
        simulator_reasoning = self._extract_reasoning_prefix(
            simulator_text, "simulation"
        )
        simulation_block = self._extract_tagged_block(simulator_text, "simulation") or ""

        self.agent.set_step_generative_support(
            mode="post_action_simulator",
            prompt=simulator_prompt,
            text=simulator_text,
            metadata={
                "step_idx": step_idx,
                "simulation": simulation_block,
                "prediction_enabled": self.generative_support_cfg.enable_prediction,
            },
        )

        self._append_auxiliary_turn(
            user_prompt=simulator_prompt,
            assistant_text=simulator_text,
            assistant_reasoning=simulator_reasoning,
            assistant_flags={"rllm_simulation": True},
            add_to_live_messages=self.generative_support_cfg.add_to_live_messages,
            add_to_step_chat_completions=(
                self.generative_support_cfg.add_to_step_chat_completions
                and self.generative_support_cfg.train_generative_with_ppo
            ),
        )

        prediction_base_messages = copy.deepcopy(self.agent.chat_completions)
        prediction_base_messages.append({"role": "user", "content": simulator_prompt})
        if simulator_text:
            prediction_base_messages.append(
                {
                    "role": "assistant",
                    "content": simulator_text,
                    "rllm_simulation": True,
                }
            )

        _, prediction_payload = await self._run_prediction_call(
            uid=uid,
            step_idx=step_idx,
            action_obj=action_obj,
            application_suffix="pred",
            base_messages=prediction_base_messages,
            store_in_live_messages=self.generative_support_cfg.add_to_live_messages,
            store_in_step_chat_completions=self.generative_support_cfg.add_to_step_chat_completions,
        )
        return {"prediction_payload": prediction_payload}

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

    async def _run_legacy_episode(self, task: dict, uid: str, **kwargs) -> Episode:
        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0.0, False, info)

        legacy_cfg = self.legacy_prediction_cfg or PredictionConfig()

        for step_idx in range(self.max_steps):
            output: ModelOutput = await self.rollout_engine.get_model_response(
                self.agent.chat_completions,
                application_id=f"{uid}:act:{step_idx}",
                enforce_max_prompt_length=legacy_cfg.enforce_max_prompt_length,
                **kwargs,
            )
            response = output.text or ""
            action = self.agent.update_from_model(response, return_action_dict=True)
            raw_action = action.action
            action_reasoning = output.reasoning
            cur_step = self.agent.get_current_state()

            prediction_prompt = None
            prediction_payload = None
            prediction_step_enabled = bool(legacy_cfg.enable_prediction_step)
            prediction_loss_enabled = bool(legacy_cfg.enable_prediction_loss)
            embed_prediction_in_chat = (
                prediction_loss_enabled and not prediction_step_enabled
            )

            if prediction_step_enabled or prediction_loss_enabled:
                prediction_prompt = self._build_prediction_prompt(raw_action)
                if prediction_prompt is not None:
                    if action_reasoning:
                        self._set_latest_assistant_reasoning(action_reasoning)

                    if prediction_step_enabled:
                        self.agent.set_step_prediction(
                            prediction=PredictionRecord(
                                prompt=prediction_prompt,
                                prediction="",
                                metadata={
                                    "step_idx": step_idx,
                                    "raw_text": "",
                                    "reasoning": "",
                                    "prediction_rollout_skipped": False,
                                    "mode": "legacy_prediction_step",
                                },
                            )
                        )

                        pred_messages = copy.deepcopy(self.agent.chat_completions)
                        pred_messages.append(
                            {"role": "user", "content": prediction_prompt}
                        )

                        pred_output: ModelOutput = (
                            await self.rollout_engine.get_model_response(
                                pred_messages,
                                application_id=f"{uid}:pred:{step_idx}",
                                enforce_max_prompt_length=legacy_cfg.enforce_max_prompt_length,
                                max_tokens=legacy_cfg.max_tokens,
                                **kwargs,
                            )
                        )

                        prediction_raw_text = pred_output.text or ""
                        prediction_text = (
                            self._extract_tagged_block(
                                prediction_raw_text, "prediction"
                            )
                            or ""
                        )
                        prediction_reasoning = self._extract_reasoning_prefix(
                            prediction_raw_text, "prediction"
                        )

                        self.agent.set_step_prediction(
                            prediction=PredictionRecord(
                                prompt=prediction_prompt,
                                prediction=prediction_text,
                                metadata={
                                    "step_idx": step_idx,
                                    "raw_text": prediction_raw_text,
                                    "reasoning": prediction_reasoning,
                                    "prediction_rollout_skipped": False,
                                    "mode": "legacy_prediction_step",
                                },
                            )
                        )
                        prediction_payload = {
                            "text": prediction_text,
                            "raw_text": prediction_raw_text,
                            "prompt": prediction_prompt,
                        }

                        if legacy_cfg.add_prediction_to_messages:
                            prediction_prompt_message = {
                                "role": "user",
                                "content": prediction_prompt,
                            }
                            pred_message = {
                                "role": "assistant",
                                "content": prediction_text,
                            }
                            if prediction_reasoning:
                                pred_message["reasoning"] = prediction_reasoning
                            if cur_step is not None:
                                cur_step.chat_completions.append(
                                    copy.deepcopy(prediction_prompt_message)
                                )
                                cur_step.chat_completions.append(
                                    copy.deepcopy(pred_message)
                                )
                            self.agent.messages.append(
                                copy.deepcopy(prediction_prompt_message)
                            )
                            self.agent.messages.append(copy.deepcopy(pred_message))

                    elif embed_prediction_in_chat:
                        self.agent.set_step_prediction(
                            prediction=PredictionRecord(
                                prompt=prediction_prompt,
                                prediction="",
                                metadata={
                                    "step_idx": step_idx,
                                    "prediction_embedded_in_chat": True,
                                    "mode": "legacy_prediction_loss_only",
                                },
                            )
                        )

            env_action = raw_action
            if prediction_payload is not None and hasattr(self.env, "similarity_config"):
                env_action = {
                    "tool_calls": raw_action,
                    "prediction": prediction_payload,
                }
            next_obs, reward, done, step_info = await self.run_in_executor(
                self.env.step, env_action
            )

            if embed_prediction_in_chat and prediction_prompt is not None:
                _, actual_output = self._extract_actual_tool_outputs(next_obs)
                if actual_output and legacy_cfg.add_prediction_to_messages:
                    cur_step = self.agent.get_current_state()
                    pred_msg_user = {"role": "user", "content": prediction_prompt}
                    pred_msg_asst = {
                        "role": "assistant",
                        "content": f"<prediction>{actual_output}</prediction>",
                        "rllm_prediction": True,
                    }
                    if cur_step is not None:
                        cur_step.chat_completions.append(copy.deepcopy(pred_msg_user))
                        cur_step.chat_completions.append(copy.deepcopy(pred_msg_asst))
                    self.agent.messages.append(copy.deepcopy(pred_msg_user))
                    self.agent.messages.append(copy.deepcopy(pred_msg_asst))

            self.agent.update_from_env(next_obs, reward, done, step_info)

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

            if output.finish_reason == "length":
                return self._build_episode(
                    task, uid, TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
                )

            if done:
                return self._build_episode(task, uid, TerminationReason.ENV_DONE)

        return self._build_episode(task, uid, TerminationReason.MAX_TURNS_EXCEEDED)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        if not self._uses_generative_support_cfg:
            return await self._run_legacy_episode(task, uid, **kwargs)

        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0.0, False, info)

        for step_idx in range(self.max_steps):
            committed_result: dict[str, Any]

            if self.generative_support_cfg.mode == "pre_action_world_model":
                committed_result = await self._run_pre_action_world_model_step(
                    uid=uid,
                    step_idx=step_idx,
                    model_kwargs=kwargs,
                )
            else:
                action_output: ModelOutput = await self.rollout_engine.get_model_response(
                    self.agent.chat_completions,
                    application_id=f"{uid}:act:{step_idx}",
                    enforce_max_prompt_length=self.generative_support_cfg.enforce_max_prompt_length,
                    **kwargs,
                )
                action_response = action_output.text or ""
                committed_result = self._build_action_first_step(
                    response=action_response,
                    output=action_output,
                )

                raw_action = committed_result["action"]
                prediction_payload = None

                if self._has_real_tool_call_payload(raw_action):
                    if self.generative_support_cfg.mode == "post_action_simulator":
                        aux_result = await self._run_post_action_simulator_step(
                            uid=uid,
                            step_idx=step_idx,
                            action_obj=raw_action,
                        )
                        prediction_payload = aux_result.get("prediction_payload")
                    elif self.generative_support_cfg.enable_prediction:
                        _, prediction_payload = await self._run_prediction_call(
                            uid=uid,
                            step_idx=step_idx,
                            action_obj=raw_action,
                            application_suffix="pred",
                            store_in_live_messages=self.generative_support_cfg.add_to_live_messages,
                            store_in_step_chat_completions=self.generative_support_cfg.add_to_step_chat_completions,
                        )

                committed_result["env_action"] = raw_action
                committed_result["prediction_payload"] = prediction_payload

            env_action = committed_result.get("env_action")
            prediction_payload = committed_result.get("prediction_payload")
            if prediction_payload is not None and hasattr(self.env, "similarity_config"):
                env_action = {
                    "tool_calls": env_action,
                    "prediction": prediction_payload,
                }

            next_obs, reward, done, step_info = await self.run_in_executor(
                self.env.step, env_action
            )

            self.agent.update_from_env(next_obs, reward, done, step_info)

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

            committed_output = committed_result["committed_output"]
            if committed_output.finish_reason == "length":
                return self._build_episode(
                    task, uid, TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
                )

            if done:
                return self._build_episode(task, uid, TerminationReason.ENV_DONE)

        return self._build_episode(task, uid, TerminationReason.MAX_TURNS_EXCEEDED)

    def _has_real_tool_call(self, step) -> bool:
        return self._has_real_tool_call_payload(step.action)

    def _classify_step_action(self, step) -> str:
        action = step.action
        if not action or not isinstance(action, list):
            return "other"

        has_finish = False
        for tool_call in action:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            tool_name = function.get("name", "")
            if tool_name and tool_name != "finish":
                return "real_tool"
            if tool_name == "finish":
                has_finish = True

        return "finish_only" if has_finish else "other"

    def _build_episode(
        self, task: dict, uid: str, termination_reason: TerminationReason
    ) -> Episode:
        agent_trajectory = copy.deepcopy(self.agent.trajectory)
        original_steps = list(agent_trajectory.steps)

        real_tool_steps = 0
        finish_only_steps = 0
        other_action_steps = 0
        action_revised_steps = 0
        candidate_real_tool_steps = 0
        prediction_present_steps = 0
        simulation_present_steps = 0

        for step in original_steps:
            action_kind = self._classify_step_action(step)
            if action_kind == "real_tool":
                real_tool_steps += 1
            elif action_kind == "finish_only":
                finish_only_steps += 1
            else:
                other_action_steps += 1

            step_info = step.info if isinstance(step.info, dict) else {}
            if step_info.get(PredictiveToolAgent.INFO_KEY_ACTION_REVISED):
                action_revised_steps += 1
            if self._has_real_tool_call_payload(
                step_info.get(PredictiveToolAgent.INFO_KEY_CANDIDATE_ACTION)
            ):
                candidate_real_tool_steps += 1

            prediction_record = step_info.get(PredictiveToolAgent.INFO_KEY_PREDICTION)
            if isinstance(prediction_record, dict) and prediction_record.get(
                "prediction"
            ):
                prediction_present_steps += 1

            generative_record = step_info.get(
                PredictiveToolAgent.INFO_KEY_GENERATIVE_SUPPORT
            )
            if isinstance(generative_record, dict) and (
                self._extract_tagged_block(generative_record.get("text", ""), "simulation")
                or generative_record.get("metadata", {}).get("simulation")
            ):
                simulation_present_steps += 1

        original_num_steps = len(original_steps)

        if self._active_simple_tir():
            filtered_steps = [
                step for step in original_steps if self._has_real_tool_call(step)
            ]
            agent_trajectory.steps = filtered_steps

        agent_trajectory.task = task.get("question", task.get("task", ""))

        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = termination_reason
        episode.trajectories = [agent_trajectory]
        episode.info["generative_support_mode"] = self._active_mode()

        total_reward = sum(step.reward for step in agent_trajectory.steps)
        agent_trajectory.reward = float(total_reward)
        episode.is_correct = total_reward > 0

        active_mode = self._active_mode()
        prediction_present_rate = (
            prediction_present_steps / real_tool_steps if real_tool_steps > 0 else 0.0
        )
        candidate_real_tool_rate = (
            candidate_real_tool_steps / original_num_steps
            if original_num_steps > 0
            else 0.0
        )

        episode.metrics = {
            "num_steps": len(agent_trajectory.steps),
            "num_steps_before_filter": original_num_steps,
            "total_reward": total_reward,
            "prediction_enabled": self._active_prediction_enabled(),
            "simple_tir": self._active_simple_tir(),
            "real_tool_steps": real_tool_steps,
            "finish_only_steps": finish_only_steps,
            "other_action_steps": other_action_steps,
            "action_revised_steps": action_revised_steps,
            "candidate_real_tool_steps": candidate_real_tool_steps,
            "candidate_real_tool_rate": candidate_real_tool_rate,
            "prediction_present_steps": prediction_present_steps,
            "prediction_present_rate": prediction_present_rate,
            "simulation_present_steps": simulation_present_steps,
            "real_tool_step_ratio": (
                real_tool_steps / original_num_steps if original_num_steps > 0 else 0.0
            ),
            "finish_only_step_ratio": (
                finish_only_steps / original_num_steps
                if original_num_steps > 0
                else 0.0
            ),
            "other_action_step_ratio": (
                other_action_steps / original_num_steps
                if original_num_steps > 0
                else 0.0
            ),
            "revision_rate": (
                action_revised_steps / candidate_real_tool_steps
                if candidate_real_tool_steps > 0
                else 0.0
            ),
            "simulation_present_rate": (
                simulation_present_steps / real_tool_steps if real_tool_steps > 0 else 0.0
            ),
            "mode_is_legacy": float(active_mode == "legacy"),
            "mode_is_none": float(active_mode == "none"),
            "mode_is_pre_action_world_model": float(
                active_mode == "pre_action_world_model"
            ),
            "mode_is_post_action_simulator": float(
                active_mode == "post_action_simulator"
            ),
            "has_any_real_tool_step": float(real_tool_steps > 0),
            "has_any_finish_only_step": float(finish_only_steps > 0),
        }

        self._save_episode_log(episode, uid, termination_reason)
        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        super().reset(task, uid)
        return self.env.reset(task)
