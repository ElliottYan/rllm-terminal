from __future__ import annotations

from typing import Any

from rllm.environments.tools.tool_env import ToolEnvironment


class PredictiveToolEnvironment(ToolEnvironment):
    """
    A ToolEnvironment extension reserved for prediction-aware training.

    For now this is a thin wrapper to keep the training stack isolated from
    the core `rllm.environments.tools.ToolEnvironment`. Later, you can:
    - add prediction-specific reward shaping
    - compute auxiliary targets/labels from tool outputs
    - emit richer metadata in `info`
    """

    @staticmethod
    def from_dict(env_args: dict) -> "PredictiveToolEnvironment":
        # Delegate to ToolEnvironment semantics but return the subclass type.
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)

        # Remaining env_args will be stored in `task` (matching ToolEnvironment behavior)
        return PredictiveToolEnvironment(task=env_args, tools=tools, tool_map=tool_map, max_steps=max_steps, reward_fn=reward_fn)

    def reset(self, task: dict | None = None) -> tuple[Any, dict]:
        # Keep signature compatible with existing workflows that call env.reset(task)
        self.task = task if task is not None else self.task
        return super().reset()

