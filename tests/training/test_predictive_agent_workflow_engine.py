from types import SimpleNamespace

import numpy as np

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent
from rllm_ext.training.predictive_agent_workflow_engine import PredictiveAgentWorkflowEngine


def test_extract_actual_output_prefers_post_action_info():
    step = Step(
        observation={"tool_outputs": {"pre_step_call": "stale observation output"}},
        info={
            PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "post action output",
            PredictiveToolAgent.INFO_KEY_ACTUAL_TOOL_OUTPUTS: {"call_1": "post action output"},
        },
    )

    actual = PredictiveAgentWorkflowEngine._extract_actual_output(step)

    assert actual == "post action output"


def test_extract_actual_output_uses_tool_output_map_from_info():
    step = Step(
        observation={"tool_outputs": {"old_call": "old output"}},
        info={
            PredictiveToolAgent.INFO_KEY_ACTUAL_TOOL_OUTPUTS: {
                "call_1": "tool result A",
                "call_2": "tool result B",
            }
        },
    )

    actual = PredictiveAgentWorkflowEngine._extract_actual_output(step)

    assert actual == "tool result A tool result B"


def test_extract_actual_output_falls_back_to_observation_for_backward_compatibility():
    step = Step(observation={"tool_outputs": {"call_1": "legacy output"}}, info={})

    actual = PredictiveAgentWorkflowEngine._extract_actual_output(step)

    assert actual == "legacy output"


def test_build_prediction_loss_example_uses_prediction_prompt_context():
    step = Step(
        chat_completions=[
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "<tool_call>calculator</tool_call>"},
            {"role": "user", "content": "predict the tool output"},
            {"role": "assistant", "content": "<prediction>wrong guess</prediction>"},
        ],
        info={
            PredictiveToolAgent.INFO_KEY_PREDICTION: {
                "prompt": "predict the tool output",
                "prediction": "wrong guess",
                "metadata": {},
            },
            PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "2",
        },
    )

    example = PredictiveAgentWorkflowEngine._build_prediction_loss_example(step)

    assert example is not None
    assert example["prompt_messages"] == step.chat_completions[:-1]
    assert example["target_text"] == "<prediction>2</prediction>"


def test_transform_results_for_verl_aligns_prediction_targets_with_trajectory_rows(monkeypatch):
    engine = PredictiveAgentWorkflowEngine.__new__(PredictiveAgentWorkflowEngine)
    engine.config = SimpleNamespace(
        rllm=SimpleNamespace(stepwise_advantage=SimpleNamespace(enable=False)),
    )

    class _DummyBatch:
        def __init__(self):
            self.non_tensor_batch = {"step_ids": np.array(["task0_agent"], dtype=object)}

    monkeypatch.setattr(AgentWorkflowEngine, "transform_results_for_verl", lambda self, episodes, task_ids: _DummyBatch())

    steps = [
        Step(
            chat_completions=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "action1"},
                {"role": "user", "content": "predict"},
                {"role": "assistant", "content": "guess1"},
            ],
            info={
                PredictiveToolAgent.INFO_KEY_PREDICTION: {"prompt": "predict", "prediction": "guess1", "metadata": {}},
                PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "out1",
            },
        ),
        Step(
            chat_completions=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "action1"},
                {"role": "user", "content": "predict"},
                {"role": "assistant", "content": "guess1"},
                {"role": "assistant", "content": "action2"},
                {"role": "user", "content": "predict again"},
                {"role": "assistant", "content": "guess2"},
            ],
            info={
                PredictiveToolAgent.INFO_KEY_PREDICTION: {"prompt": "predict again", "prediction": "guess2", "metadata": {}},
                PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT: "out2",
            },
        ),
    ]
    episode = Episode(trajectories=[Trajectory(steps=steps)])

    batch = engine.transform_results_for_verl([episode], np.array(["task0"], dtype=object))

    prediction_targets = batch.non_tensor_batch["prediction_targets"]
    assert len(prediction_targets) == 1
    assert len(prediction_targets[0]["examples"]) == 2
