"""PredictiveToolEnvironment with prediction similarity reward."""

from __future__ import annotations

from typing import Any

from rllm.environments.tools.tool_env import ToolEnvironment
from rllm_ext.rewards.prediction_similarity import (
    SimilarityConfig,
    compute_prediction_similarity_reward,
)


class PredictiveToolEnvironment(ToolEnvironment):
    """
    A ToolEnvironment extension that adds prediction similarity reward.

    This environment:
    - Accepts enhanced action format with prediction metadata
    - Computes n-gram similarity between prediction and actual tool output
    - Adds similarity reward as auxiliary reward (summed with main reward)
    - Stores similarity metrics in info for logging/analysis

    Enhanced action format:
        {
            "tool_calls": [...],  # Standard tool calls (list, dict, or str)
            "prediction": {       # Optional prediction metadata
                "text": "predicted output",
                "raw_text": "full model output",
                "prompt": "prediction prompt",
            }
        }

    Backward compatibility: Standard action formats (list/dict/str) work
    without modification - similarity is only computed when prediction is provided.
    """

    def __init__(
        self,
        task: dict | None = None,
        tools: list[str] | None = None,
        tool_map: dict[str, type] | None = None,
        reward_fn: Any | None = None,
        max_steps: int = 10,
        similarity_config: SimilarityConfig | dict | None = None,
    ):
        """
        Initialize PredictiveToolEnvironment.

        Args:
            task: Task dictionary.
            tools: List of tool names to use.
            tool_map: Mapping of tool names to tool classes.
            reward_fn: Reward function for evaluating final answers.
            max_steps: Maximum number of steps.
            similarity_config: Configuration for prediction similarity reward.
                Can be SimilarityConfig object or dict (will be converted).
        """
        # Initialize parent ToolEnvironment
        super().__init__(task, tools, tool_map, reward_fn, max_steps)

        # Initialize similarity config
        if similarity_config is None:
            self.similarity_config = SimilarityConfig(enabled=False)
        elif isinstance(similarity_config, dict):
            self.similarity_config = SimilarityConfig(**similarity_config)
        else:
            self.similarity_config = similarity_config

    def step(self, action: list[dict] | str | dict):
        """
        Execute one step in the environment.

        Handles both standard action format and enhanced action format with prediction.

        Args:
            action: Can be:
                - Standard: list[dict], str, or dict (original ToolEnvironment API)
                - Enhanced: dict with "tool_calls" and optional "prediction" keys

        Returns:
            tuple: (next_obs, reward, done, info)
                - next_obs: Observation dictionary with tool_outputs
                - reward: Total reward (main reward + similarity reward if applicable)
                - done: Whether episode is done
                - info: Metadata dict, includes prediction_similarity if computed
        """
        # Handle enhanced action format
        prediction = None
        tool_calls = action
        if isinstance(action, dict) and "tool_calls" in action:
            prediction = action.get("prediction")
            tool_calls = action["tool_calls"]

        # Call parent step() logic
        next_obs, reward, done, info = super().step(tool_calls)

        # Compute similarity reward if prediction available and tools were executed
        if (
            self.similarity_config.enabled
            and prediction is not None
            and "tool_outputs" in next_obs
        ):
            tool_outputs = next_obs.get("tool_outputs", {})
            actual_outputs = self._concatenate_tool_outputs(
                tool_outputs=tool_outputs,
                tool_calls=tool_calls,
            )
            prediction_text = self._resolve_prediction_text_for_reward(prediction)
            actual_length = len(actual_outputs.split())

            # Keep behavior aligned with reward function:
            # skip similarity computation entirely for very short actual outputs.
            if (
                actual_outputs
                and prediction_text
                and actual_length >= self.similarity_config.min_length
            ):
                similarity_reward = compute_prediction_similarity_reward(
                    prediction_text,
                    actual_outputs,
                    self.similarity_config,
                )

                # Add to main reward
                reward += similarity_reward

                # Store in info for logging/analysis
                info["prediction_similarity"] = {
                    "score": (
                        similarity_reward / self.similarity_config.weight
                        if self.similarity_config.weight > 0
                        else 0.0
                    ),  # Normalized score [0, 1]
                    "raw_reward": similarity_reward,
                    "prediction_length": len(prediction_text.split()),
                    "actual_length": actual_length,
                    "prediction_for_reward": prediction_text,
                    "actual_output_for_reward": actual_outputs,
                }

        return next_obs, reward, done, info

    @staticmethod
    def _extract_tagged_block(text: str, tag: str) -> str | None:
        """
        Extract inner content from a simple XML-like tag: <tag>...</tag>.
        Returns None when the block is missing or malformed.
        """
        if not isinstance(text, str) or not text:
            return None

        begin = f"<{tag}>"
        end = f"</{tag}>"
        start = text.find(begin)
        if start < 0:
            return None
        start += len(begin)
        stop = text.find(end, start)
        if stop < 0:
            return None

        return text[start:stop].strip()

    @classmethod
    def _resolve_prediction_text_for_reward(cls, prediction: Any) -> str:
        """
        Resolve the prediction text used for reward.

        Priority:
        1) content inside <prediction>...</prediction> from raw_text
        2) content inside <prediction>...</prediction> from text
        3) plain text field as backward-compatible fallback
        """
        if not isinstance(prediction, dict):
            return ""

        raw_text = prediction.get("raw_text", "")
        tagged_from_raw = cls._extract_tagged_block(raw_text, "prediction")
        if tagged_from_raw:
            return tagged_from_raw

        text = prediction.get("text", "")
        tagged_from_text = cls._extract_tagged_block(text, "prediction")
        if tagged_from_text:
            return tagged_from_text

        return text.strip() if isinstance(text, str) else ""

    @staticmethod
    def _concatenate_tool_outputs(tool_outputs: Any, tool_calls: Any) -> str:
        """
        Concatenate current-step tool outputs in deterministic order.

        Order policy:
        1) IDs in the current tool call list order
        2) remaining outputs ordered by key
        """
        if not isinstance(tool_outputs, dict) or not tool_outputs:
            return ""

        output_map = {str(k): str(v) for k, v in tool_outputs.items()}
        ordered_values: list[str] = []
        used_ids: set[str] = set()

        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                if tool_call_id is None:
                    continue
                tool_call_id = str(tool_call_id)
                if tool_call_id in output_map and tool_call_id not in used_ids:
                    ordered_values.append(output_map[tool_call_id])
                    used_ids.add(tool_call_id)

        for tool_call_id in sorted(output_map.keys()):
            if tool_call_id in used_ids:
                continue
            ordered_values.append(output_map[tool_call_id])

        return " ".join(ordered_values).strip()

    @staticmethod
    def from_dict(env_args: dict) -> "PredictiveToolEnvironment":
        """
        Create PredictiveToolEnvironment from dictionary arguments.

        Args:
            env_args: Dictionary with environment configuration.
                May include 'similarity_config' key.

        Returns:
            PredictiveToolEnvironment instance.
        """
        # Extract standard args
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)

        # Extract similarity config
        similarity_config = env_args.pop("similarity_config", None)

        # Remaining env_args will be stored in `task`
        return PredictiveToolEnvironment(
            task=env_args,
            tools=tools,
            tool_map=tool_map,
            max_steps=max_steps,
            reward_fn=reward_fn,
            similarity_config=similarity_config,
        )

    def reset(self, task: dict | None = None) -> tuple[Any, dict]:
        """
        Reset the environment.

        Args:
            task: Optional new task to set.

        Returns:
            tuple: (task, info)
        """
        # Update task if provided
        if task is not None:
            self.task = task
        # Call parent's reset()
        return super().reset()
