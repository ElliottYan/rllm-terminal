from types import SimpleNamespace

import numpy as np
import torch

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm_ext.training.predictive_agent_workflow_engine import (
    PredictiveAgentWorkflowEngine,
)


class _DummyTokenizer:
    name_or_path = "dummy"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 31 + 1 for ch in text]


class _DummyChatParser:
    generation_prompt = "<assistant>"

    def __init__(self):
        self.tokenizer = _DummyTokenizer()

    def parse(
        self,
        messages,
        is_first_msg=False,
        add_generation_prompt=False,
        accumulate_reasoning=False,
    ):
        parts = []
        for msg in messages:
            parts.append(f"<{msg['role']}>{msg.get('content', '')}</{msg['role']}>")
        result = "".join(parts)
        if add_generation_prompt:
            result += self.generation_prompt
        return result


def test_compute_prediction_mask_cumulative_identifies_prediction_tokens():
    messages = [
        {"role": "user", "content": "solve"},
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "predict"},
        {"role": "assistant", "content": "result", "rllm_prediction": True},
    ]
    chat_parser = _DummyChatParser()

    mask = PredictiveAgentWorkflowEngine._compute_prediction_mask_cumulative(
        messages, chat_parser
    )

    assert mask.dtype == torch.long

    # Manually compute expected lengths
    # message[1] (assistant "action"): strip generation_prompt from parsed output
    asst_parsed = chat_parser.parse(
        [messages[1]],
        is_first_msg=False,
        add_generation_prompt=False,
        accumulate_reasoning=True,
    )
    asst_text = asst_parsed[len(chat_parser.generation_prompt) :]
    n_action_tokens = len(
        chat_parser.tokenizer.encode(asst_text, add_special_tokens=False)
    )

    # message[2] (user "predict"): non-assistant, so mask=0
    user_parsed = chat_parser.parse(
        [messages[2]],
        is_first_msg=False,
        add_generation_prompt=True,
        accumulate_reasoning=False,
    )
    n_user_tokens = len(
        chat_parser.tokenizer.encode(user_parsed, add_special_tokens=False)
    )

    # message[3] (assistant "result", rllm_prediction=True): mask=1
    pred_parsed = chat_parser.parse(
        [messages[3]],
        is_first_msg=False,
        add_generation_prompt=False,
        accumulate_reasoning=True,
    )
    pred_text = pred_parsed[len(chat_parser.generation_prompt) :]
    n_pred_tokens = len(
        chat_parser.tokenizer.encode(pred_text, add_special_tokens=False)
    )

    total_len = n_action_tokens + n_user_tokens + n_pred_tokens
    assert len(mask) == total_len

    expected_mask = [0] * n_action_tokens + [0] * n_user_tokens + [1] * n_pred_tokens
    assert mask.tolist() == expected_mask


def test_compute_prediction_mask_cumulative_all_zeros_without_prediction():
    messages = [
        {"role": "user", "content": "solve"},
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "tool output"},
    ]
    chat_parser = _DummyChatParser()

    mask = PredictiveAgentWorkflowEngine._compute_prediction_mask_cumulative(
        messages, chat_parser
    )

    assert (mask == 0).all()


def test_transform_results_for_verl_adds_prediction_mask_and_adjusts_response_mask(
    monkeypatch,
):
    engine = PredictiveAgentWorkflowEngine.__new__(PredictiveAgentWorkflowEngine)
    engine.config = SimpleNamespace(
        rllm=SimpleNamespace(stepwise_advantage=SimpleNamespace(enable=False)),
        data=SimpleNamespace(max_response_length=128),
    )
    engine.rollout_engine = SimpleNamespace(chat_parser=_DummyChatParser())

    class _DummyBatch:
        def __init__(self):
            self.batch = {
                "response_mask": torch.ones((1, 128), dtype=torch.long),
            }
            self.non_tensor_batch = {
                "step_ids": np.array(["task0_agent"], dtype=object)
            }

    monkeypatch.setattr(
        AgentWorkflowEngine,
        "transform_results_for_verl",
        lambda self, episodes, task_ids: _DummyBatch(),
    )

    messages = [
        {"role": "user", "content": "solve"},
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "predict"},
        {"role": "assistant", "content": "result", "rllm_prediction": True},
    ]
    steps = [Step(chat_completions=messages), Step(chat_completions=messages)]
    episode = Episode(trajectories=[Trajectory(steps=steps)])

    batch = engine.transform_results_for_verl(
        [episode], np.array(["task0"], dtype=object)
    )

    assert "prediction_mask" in batch.batch
    pred_mask = batch.batch["prediction_mask"]
    assert pred_mask.shape == (1, 128)
    assert pred_mask.dtype == torch.long

    # Some tokens should be marked as prediction
    assert pred_mask.sum() > 0

    # response_mask should be adjusted: no overlap with prediction_mask
    response_mask = batch.batch["response_mask"]
    assert (response_mask * pred_mask).sum() == 0


def test_transform_results_for_verl_stepwise_mode_uses_zero_masks(monkeypatch):
    engine = PredictiveAgentWorkflowEngine.__new__(PredictiveAgentWorkflowEngine)
    engine.config = SimpleNamespace(
        rllm=SimpleNamespace(stepwise_advantage=SimpleNamespace(enable=True)),
        data=SimpleNamespace(max_response_length=64),
    )
    engine.rollout_engine = SimpleNamespace(chat_parser=_DummyChatParser())

    class _DummyBatch:
        def __init__(self):
            self.batch = {
                "response_mask": torch.ones((2, 64), dtype=torch.long),
            }
            self.non_tensor_batch = {
                "step_ids": np.array(
                    ["task0_agent_step0", "task0_agent_step1"], dtype=object
                ),
            }

    monkeypatch.setattr(
        AgentWorkflowEngine,
        "transform_results_for_verl",
        lambda self, episodes, task_ids: _DummyBatch(),
    )

    steps = [
        Step(
            chat_completions=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        ),
        Step(
            chat_completions=[
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        ),
    ]
    episode = Episode(trajectories=[Trajectory(steps=steps)])

    batch = engine.transform_results_for_verl(
        [episode], np.array(["task0"], dtype=object)
    )

    # Stepwise mode: prediction_mask should be all zeros
    pred_mask = batch.batch["prediction_mask"]
    assert pred_mask.shape == (2, 64)
    assert (pred_mask == 0).all()
