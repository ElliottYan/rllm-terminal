from importlib.machinery import ModuleSpec
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


def _make_stub_module(name: str, *, is_package: bool = False) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = ModuleSpec(name=name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []
        module.__spec__.submodule_search_locations = []
    return module


def _install_optional_dependency_stubs() -> None:
    if "pylatexenc" not in sys.modules:
        latex2text = _make_stub_module("pylatexenc.latex2text")

        class _LatexNodes2Text:
            def latex_to_text(self, expr):
                return expr

        latex2text.LatexNodes2Text = _LatexNodes2Text
        pylatexenc = _make_stub_module("pylatexenc", is_package=True)
        pylatexenc.latex2text = latex2text
        sys.modules["pylatexenc"] = pylatexenc
        sys.modules["pylatexenc.latex2text"] = latex2text

    if "httpx" not in sys.modules:
        httpx = _make_stub_module("httpx")

        class _DummyResponse:
            is_success = True
            status_code = 200
            text = ""

            def json(self):
                return {}

        class _DummyClient:
            def get(self, *args, **kwargs):
                return _DummyResponse()

            def post(self, *args, **kwargs):
                return _DummyResponse()

            def close(self):
                return None

        httpx.Client = _DummyClient
        httpx.AsyncClient = _DummyClient
        sys.modules["httpx"] = httpx

    if "firecrawl" not in sys.modules:
        firecrawl = _make_stub_module("firecrawl")
        firecrawl.FirecrawlApp = SimpleNamespace
        sys.modules["firecrawl"] = firecrawl

    if "transformers" not in sys.modules:
        transformers = _make_stub_module("transformers")

        class _PreTrainedTokenizerBase:
            pass

        transformers.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
        sys.modules["transformers"] = transformers

    if "PIL" not in sys.modules:
        pil = _make_stub_module("PIL", is_package=True)
        image = _make_stub_module("PIL.Image")
        pil.Image = image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image


_install_optional_dependency_stubs()

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent
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
    assert mask.sum() > 0


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


def test_build_prediction_loss_example_keeps_action_and_adds_simulator_transcript():
    step = Step(
        chat_completions=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "TOOL:print(2 + 2)"},
        ],
        info={
            PredictiveToolAgent.INFO_KEY_GENERATIVE_SUPPORT: {
                "mode": "post_action_simulator",
                "prompt": "SIMULATOR PROMPT",
                "text": "<simulation>Likely returns 4</simulation>",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_PREDICTION: {
                "prompt": "PREDICTION PROMPT",
                "prediction": "",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "4",
        },
    )

    example = PredictiveAgentWorkflowEngine._build_prediction_loss_example(step)

    assert example["target_text"] == "<prediction>4</prediction>"
    assert example["prompt_messages"][2]["role"] == "assistant"
    assert example["prompt_messages"][2]["content"] == "TOOL:print(2 + 2)"
    assert example["prompt_messages"][-3]["content"] == "SIMULATOR PROMPT"
    assert example["prompt_messages"][-2]["content"] == "<simulation>Likely returns 4</simulation>"
    assert example["prompt_messages"][-1]["content"] == "PREDICTION PROMPT"


def test_build_prediction_loss_example_strips_auxiliary_tags_from_pre_action_finalize():
    step = Step(
        chat_completions=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "solve"},
            {
                "role": "assistant",
                "content": (
                    "Reasoning.\n"
                    "<simulation>Likely returns 4</simulation>\n"
                    "<prediction>4</prediction>\n"
                    "TOOL:print(2 + 2)"
                ),
            },
        ],
        info={
            PredictiveToolAgent.INFO_KEY_GENERATIVE_SUPPORT: {
                "mode": "pre_action_world_model",
                "prompt": "FINALIZE PROMPT",
                "text": "raw finalize",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_PREDICTION: {
                "prompt": "PREDICTION PROMPT",
                "prediction": "",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "4",
        },
    )

    example = PredictiveAgentWorkflowEngine._build_prediction_loss_example(step)

    assert "<simulation>" not in example["prompt_messages"][-2]["content"]
    assert "<prediction>" not in example["prompt_messages"][-2]["content"]
    assert example["prompt_messages"][-1]["content"] == "PREDICTION PROMPT"


def test_build_imagine_loss_example_uses_actual_output_as_target():
    step = Step(
        chat_completions=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "TOOL:print(2 + 2)"},
        ],
        info={
            PredictiveToolAgent.INFO_KEY_IMAGINE: {
                "prompt": "IMAGINE PROMPT",
                "prediction": "4",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "6",
        },
    )

    example = PredictiveAgentWorkflowEngine._build_imagine_loss_example(step)

    assert example["target_text"] == "<imagine>6</imagine>"
    assert example["prompt_messages"][-1]["content"] == "IMAGINE PROMPT"
    assert example["kind"] == "imagine"


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
    step = Step(chat_completions=messages)
    episode = Episode(trajectories=[Trajectory(steps=[step])])

    batch = engine.transform_results_for_verl(
        [episode], np.array(["task0"], dtype=object)
    )

    pred_mask = batch.batch["prediction_mask"]
    assert pred_mask.shape == (1, 128)
    assert pred_mask.sum() > 0
    assert (batch.batch["response_mask"] * pred_mask).sum() == 0


def test_transform_results_for_verl_stepwise_mode_computes_mask_per_step(monkeypatch):
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
                {"role": "user", "content": "predict"},
                {"role": "assistant", "content": "4", "rllm_prediction": True},
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

    pred_mask = batch.batch["prediction_mask"]
    assert pred_mask.shape == (2, 64)
    assert pred_mask[0].sum() > 0
    assert pred_mask[1].sum() == 0


def test_transform_results_for_verl_non_cumulative_trajectory_falls_back_to_prediction_targets(
    monkeypatch,
):
    engine = PredictiveAgentWorkflowEngine.__new__(PredictiveAgentWorkflowEngine)
    engine.config = SimpleNamespace(
        rllm=SimpleNamespace(stepwise_advantage=SimpleNamespace(enable=False)),
        data=SimpleNamespace(max_response_length=64),
    )
    engine.rollout_engine = SimpleNamespace(chat_parser=_DummyChatParser())

    class _DummyBatch:
        def __init__(self):
            self.batch = {
                "response_mask": torch.ones((1, 64), dtype=torch.long),
            }
            self.non_tensor_batch = {
                "step_ids": np.array(["task0_agent"], dtype=object)
            }

    monkeypatch.setattr(
        AgentWorkflowEngine,
        "transform_results_for_verl",
        lambda self, episodes, task_ids: _DummyBatch(),
    )

    first_step = Step(
        chat_completions=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "action"},
            {"role": "user", "content": "predict"},
            {"role": "assistant", "content": "4", "rllm_prediction": True},
        ],
        info={
            PredictiveToolAgent.INFO_KEY_PREDICTION: {
                "prompt": "PREDICTION PROMPT",
                "prediction": "",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "4",
        },
    )
    second_step = Step(
        chat_completions=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "action"},
            {"role": "tool", "content": "4", "tool_call_id": "tool-0"},
            {"role": "assistant", "content": "FINAL 4"},
        ]
    )
    episode = Episode(trajectories=[Trajectory(steps=[first_step, second_step])])

    batch = engine.transform_results_for_verl(
        [episode], np.array(["task0"], dtype=object)
    )

    pred_mask = batch.batch["prediction_mask"]
    assert pred_mask.shape == (1, 64)
    assert pred_mask.sum() == 0

    prediction_target = batch.non_tensor_batch["prediction_targets"][0]
    assert prediction_target["has_prediction_target"] is True
    assert len(prediction_target["examples"]) == 1
    assert prediction_target["examples"][0]["target_text"] == "<prediction>4</prediction>"


def test_transform_results_for_verl_collects_imagine_and_prediction_examples(
    monkeypatch,
):
    engine = PredictiveAgentWorkflowEngine.__new__(PredictiveAgentWorkflowEngine)
    engine.config = SimpleNamespace(
        rllm=SimpleNamespace(stepwise_advantage=SimpleNamespace(enable=False)),
        data=SimpleNamespace(max_response_length=64),
    )
    engine.rollout_engine = SimpleNamespace(chat_parser=_DummyChatParser())

    class _DummyBatch:
        def __init__(self):
            self.batch = {
                "response_mask": torch.ones((1, 64), dtype=torch.long),
            }
            self.non_tensor_batch = {
                "step_ids": np.array(["task0_agent"], dtype=object)
            }

    monkeypatch.setattr(
        AgentWorkflowEngine,
        "transform_results_for_verl",
        lambda self, episodes, task_ids: _DummyBatch(),
    )

    step = Step(
        chat_completions=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "TOOL_ALT:print(3 + 3)"},
        ],
        info={
            PredictiveToolAgent.INFO_KEY_IMAGINE: {
                "prompt": "IMAGINE PROMPT",
                "prediction": "4",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_PREDICTION: {
                "prompt": "PREDICTION PROMPT",
                "prediction": "",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "6",
        },
    )
    episode = Episode(trajectories=[Trajectory(steps=[step])])

    batch = engine.transform_results_for_verl(
        [episode], np.array(["task0"], dtype=object)
    )

    prediction_target = batch.non_tensor_batch["prediction_targets"][0]
    assert prediction_target["has_prediction_target"] is True
    assert [example["target_text"] for example in prediction_target["examples"]] == [
        "<imagine>6</imagine>",
        "<prediction>6</prediction>",
    ]
