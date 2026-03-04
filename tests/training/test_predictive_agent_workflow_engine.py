from rllm.agents.agent import Step
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
