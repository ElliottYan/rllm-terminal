import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import ModuleSpec
import json
import sys
from types import ModuleType, SimpleNamespace


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

from rllm.agents.agent import Action, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent
from rllm_ext.workflows.predictive_tool_workflow import PredictiveToolWorkflow


class _DummyToolParser:
    def parse(self, response: str):
        if "TOOL_ALT:" in response:
            return [
                {
                    "name": "python",
                    "arguments": '{"code": "print(3 + 3)"}',
                }
            ]
        if "TOOL:" in response:
            return [
                {
                    "name": "python",
                    "arguments": '{"code": "print(2 + 2)"}',
                }
            ]
        return []


class _DummyPredictiveAgent(PredictiveToolAgent):
    def __init__(self, **kwargs):
        self.system_prompt = "sys"
        self.tool_parser = _DummyToolParser()
        self._trajectory = Trajectory()
        self.messages = []
        self.current_observation = None
        self.reset()

    def _format_observation_as_messages(self, obs):
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_output_str,
                            "tool_call_id": tool_call_id,
                        }
                    )
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})
        return messages

    def update_from_env(self, observation, reward, done, info, **kwargs):
        self.messages.extend(self._format_observation_as_messages(observation))
        self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        step_idx = len(self._trajectory.steps)
        parsed_tool_calls = self.tool_parser.parse(response)

        if parsed_tool_calls:
            tool_calls_dict = [
                {
                    "id": f"tool-{step_idx}-{idx}",
                    "type": "function",
                    "function": copy.deepcopy(tool_call),
                }
                for idx, tool_call in enumerate(parsed_tool_calls)
            ]
        else:
            tool_calls_dict = [
                {
                    "id": f"finish-{step_idx}",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {"response": response},
                    },
                }
            ]

        assistant_message = {"role": "assistant", "content": response}
        if tool_calls_dict[0]["function"]["name"] != "finish":
            assistant_message["tool_calls"] = copy.deepcopy(tool_calls_dict)
        self.messages.append(assistant_message)

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=copy.deepcopy(tool_calls_dict),
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)
        return Action(action=tool_calls_dict)

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None

    @property
    def chat_completions(self):
        return self.messages

    @property
    def trajectory(self):
        return self._trajectory

    def get_current_state(self):
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]

    def get_current_state(self):
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]


class _DummyPlainAgent:
    def __init__(self, **kwargs):
        self.system_prompt = "sys"
        self.tool_parser = _DummyToolParser()
        self._trajectory = Trajectory()
        self.messages = []
        self.current_observation = None
        self.reset()

    def _format_observation_as_messages(self, obs):
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_output_str,
                            "tool_call_id": tool_call_id,
                        }
                    )
        return messages

    def update_from_env(self, observation, reward, done, info, **kwargs):
        self.messages.extend(self._format_observation_as_messages(observation))
        self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        step_idx = len(self._trajectory.steps)
        parsed_tool_calls = self.tool_parser.parse(response)
        if parsed_tool_calls:
            tool_calls_dict = [
                {
                    "id": f"tool-{step_idx}-{idx}",
                    "type": "function",
                    "function": copy.deepcopy(tool_call),
                }
                for idx, tool_call in enumerate(parsed_tool_calls)
            ]
        else:
            tool_calls_dict = [
                {
                    "id": f"finish-{step_idx}",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {"response": response},
                    },
                }
            ]

        assistant_message = {"role": "assistant", "content": response}
        if tool_calls_dict[0]["function"]["name"] != "finish":
            assistant_message["tool_calls"] = copy.deepcopy(tool_calls_dict)
        self.messages.append(assistant_message)

        self._trajectory.steps.append(
            Step(
                chat_completions=copy.deepcopy(self.chat_completions),
                action=copy.deepcopy(tool_calls_dict),
                model_response=response,
                observation=self.current_observation,
            )
        )
        return Action(action=tool_calls_dict)

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None

    @property
    def chat_completions(self):
        return self.messages

    @property
    def trajectory(self):
        return self._trajectory

    def get_current_state(self):
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]


class _DummyEnv:
    def __init__(self, **kwargs):
        self.task = None
        self.step_count = 0

    def reset(self, task=None):
        self.task = task or {}
        self.step_count = 0
        return self.task, {}

    def step(self, action):
        self.step_count += 1
        tool_calls = action["tool_calls"] if isinstance(action, dict) else action
        tool_name = tool_calls[0]["function"]["name"]

        if tool_name == "finish":
            return {}, 1.0, True, {"response": action, "metadata": {}}

        arguments = tool_calls[0]["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        code = arguments["code"]
        tool_output = "6" if "3 + 3" in code else "4"

        return (
            {"tool_outputs": {tool_calls[0]["id"]: tool_output}},
            0.0,
            False,
            {"response": action, "metadata": {}},
        )


class _DummyRolloutEngine:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get_model_response(self, messages, **kwargs):
        self.calls.append(
            {
                "application_id": kwargs.get("application_id"),
                "enforce_max_prompt_length": kwargs.get(
                    "enforce_max_prompt_length"
                ),
                "messages": copy.deepcopy(messages),
            }
        )
        text = self.responses.pop(0)
        return ModelOutput(
            text=text,
            content=text,
            reasoning="",
            tool_calls=[],
            prompt_ids=[],
            completion_ids=[1],
            prompt_length=1,
            completion_length=1,
            finish_reason="stop",
        )


def _run_workflow(*, responses, prediction_cfg=None, generative_support_cfg=None):
    rollout_engine = _DummyRolloutEngine(responses)
    task = {"question": "What is 2 + 2?"}

    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = PredictiveToolWorkflow(
            agent_cls=_DummyPredictiveAgent,
            env_cls=_DummyEnv,
            max_steps=4,
            prediction_cfg=prediction_cfg,
            generative_support_cfg=generative_support_cfg,
            rollout_engine=rollout_engine,
            executor=executor,
        )
        episode = asyncio.run(workflow.run(task=task, uid="task-0"))
    return episode, workflow, rollout_engine


def test_legacy_prediction_loss_keeps_rollout_context_clean_and_stores_metadata():
    episode, workflow, rollout_engine = _run_workflow(
        responses=["TOOL:print(2 + 2)", "FINAL 4"],
        prediction_cfg={
            "enabled": True,
            "enable_prediction_loss": True,
            "enable_prediction_step": False,
            "add_prediction_to_messages": False,
        },
    )

    trajectory = episode.trajectories[0]
    assert len(trajectory.steps) == 2

    second_action_call = next(
        call for call in rollout_engine.calls if call["application_id"] == "task-0:act:1"
    )
    assert not any(
        "PREDICTION MODE" in message.get("content", "")
        for message in second_action_call["messages"]
    )
    assert not any(
        message.get("rllm_prediction", False)
        for message in second_action_call["messages"]
    )

    final_training_messages = trajectory.steps[-1].chat_completions
    assert not any(
        "PREDICTION MODE" in message.get("content", "")
        for message in final_training_messages
    )
    assert not any(
        message.get("rllm_prediction", False) for message in final_training_messages
    )

    first_step_info = trajectory.steps[0].info
    prediction_record = first_step_info[PredictiveToolAgent.INFO_KEY_PREDICTION]
    assert "PREDICTION MODE" in prediction_record["prompt"]
    assert first_step_info[PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT] == "4"
    assert workflow.legacy_prediction_cfg is not None
    assert workflow._uses_generative_support_cfg is False
    assert episode.metrics["mode_is_legacy"] == 1.0
    assert episode.info["generative_support_mode"] == "legacy"


def test_legacy_prediction_messages_still_reach_live_context_when_explicitly_enabled():
    episode, workflow, rollout_engine = _run_workflow(
        responses=["TOOL:print(2 + 2)", "FINAL 4"],
        prediction_cfg={
            "enabled": True,
            "enable_prediction_loss": True,
            "enable_prediction_step": False,
            "add_prediction_to_messages": True,
        },
    )

    trajectory = episode.trajectories[0]
    second_action_call = next(
        call for call in rollout_engine.calls if call["application_id"] == "task-0:act:1"
    )
    assert any(
        "PREDICTION MODE" in message.get("content", "")
        for message in second_action_call["messages"]
        if message.get("role") == "user"
    )
    assert any(
        message.get("rllm_prediction", False)
        for message in trajectory.steps[-1].chat_completions
    )
    assert any(
        "PREDICTION MODE" in message.get("content", "")
        for message in workflow.agent.messages
        if message.get("role") == "user"
    )


def test_legacy_prediction_step_inherits_prompt_length_enforcement_setting():
    _episode, _workflow, rollout_engine = _run_workflow(
        responses=["TOOL:print(2 + 2)", "<prediction>4</prediction>", "FINAL 4"],
        prediction_cfg={
            "enabled": True,
            "enable_prediction_loss": False,
            "enable_prediction_step": True,
            "enforce_max_prompt_length": False,
            "add_prediction_to_messages": False,
        },
    )

    pred_call = next(
        call for call in rollout_engine.calls if call["application_id"] == "task-0:pred:0"
    )
    assert pred_call["enforce_max_prompt_length"] is False


def test_legacy_prediction_disabled_allows_non_predictive_agent():
    rollout_engine = _DummyRolloutEngine(["TOOL:print(2 + 2)", "FINAL 4"])
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = PredictiveToolWorkflow(
            agent_cls=_DummyPlainAgent,
            env_cls=_DummyEnv,
            max_steps=4,
            prediction_cfg={"enabled": False},
            rollout_engine=rollout_engine,
            executor=executor,
        )
        episode = asyncio.run(workflow.run(task={"question": "What is 2 + 2?"}, uid="task-plain"))

    assert len(episode.trajectories[0].steps) == 2


def test_pre_action_world_model_candidate_does_not_create_step_and_can_revise_action():
    finalize_text = (
        "Reasoning.\n"
        "<simulation>Running the original code would be wrong; use 3 + 3.</simulation>\n"
        "<prediction>6</prediction>\n"
        "TOOL_ALT:print(3 + 3)"
    )
    episode, workflow, rollout_engine = _run_workflow(
        responses=["TOOL:print(2 + 2)", finalize_text, "FINAL 6"],
        generative_support_cfg={
            "mode": "pre_action_world_model",
            "enable_prediction": True,
            "add_to_live_messages": False,
            "add_to_step_chat_completions": True,
        },
    )

    trajectory = episode.trajectories[0]
    assert len(trajectory.steps) == 2
    assert [call["application_id"] for call in rollout_engine.calls] == [
        "task-0:cand:0",
        "task-0:final:0",
        "task-0:cand:1",
    ]

    first_step = trajectory.steps[0]
    assert first_step.model_response == finalize_text
    assert first_step.info[PredictiveToolAgent.INFO_KEY_ACTION_REVISED] is True
    assert (
        first_step.info[PredictiveToolAgent.INFO_KEY_CANDIDATE_ACTION][0]["function"]["arguments"]
        == '{"code": "print(2 + 2)"}'
    )
    assert (
        first_step.info[PredictiveToolAgent.INFO_KEY_FINAL_ACTION][0]["function"]["arguments"]
        == '{"code": "print(3 + 3)"}'
    )
    assert first_step.info[PredictiveToolAgent.INFO_KEY_PREDICTION]["prediction"] == "6"
    assert first_step.info[PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT] == "6"

    second_candidate_call = next(
        call for call in rollout_engine.calls if call["application_id"] == "task-0:cand:1"
    )
    assert not any(
        "<simulation>" in message.get("content", "")
        for message in second_candidate_call["messages"]
    )
    assert not any(
        "<prediction>" in message.get("content", "")
        for message in second_candidate_call["messages"]
    )
    assert workflow._uses_generative_support_cfg is True
    assert episode.metrics["mode_is_pre_action_world_model"] == 1.0
    assert episode.info["generative_support_mode"] == "pre_action_world_model"


def test_pre_action_world_model_falls_back_to_candidate_action_when_finalize_parse_fails():
    episode, _workflow, _rollout_engine = _run_workflow(
        responses=[
            "TOOL:print(2 + 2)",
            "Reasoning only.\n<simulation>Looks like 4.</simulation>\n<prediction>4</prediction>",
            "FINAL 4",
        ],
        generative_support_cfg={
            "mode": "pre_action_world_model",
            "enable_prediction": True,
        },
    )

    first_step = episode.trajectories[0].steps[0]
    assert (
        first_step.action[0]["function"]["arguments"] == '{"code": "print(2 + 2)"}'
    )
    assert first_step.info[PredictiveToolAgent.INFO_KEY_ACTION_REVISED] is False
    assert first_step.info[PredictiveToolAgent.INFO_KEY_PREDICTION]["prediction"] == ""
    assert first_step.info[PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT] == "4"


def test_post_action_simulator_keeps_auxiliary_turns_out_of_live_context_by_default():
    simulator_text = (
        "Reasoning.\n"
        "<simulation>Likely returns 4 from the python tool.</simulation>"
    )
    prediction_text = "Short reasoning.\n<prediction>4</prediction>"
    episode, workflow, rollout_engine = _run_workflow(
        responses=["TOOL:print(2 + 2)", simulator_text, prediction_text, "FINAL 4"],
        generative_support_cfg={
            "mode": "post_action_simulator",
            "enable_prediction": True,
            "add_to_live_messages": False,
            "add_to_step_chat_completions": True,
        },
    )

    trajectory = episode.trajectories[0]
    first_step_messages = trajectory.steps[0].chat_completions
    assert any(message.get("rllm_simulation") for message in first_step_messages)
    assert any(message.get("rllm_prediction") for message in first_step_messages)
    assert not any(message.get("rllm_simulation") for message in workflow.agent.messages)
    assert not any(message.get("rllm_prediction") for message in workflow.agent.messages)

    second_action_call = next(
        call for call in rollout_engine.calls if call["application_id"] == "task-0:act:1"
    )
    assert not any(
        message.get("rllm_simulation") or message.get("rllm_prediction")
        for message in second_action_call["messages"]
    )
    assert first_step_messages[-1]["rllm_prediction"] is True
    assert first_step_messages[-3]["rllm_simulation"] is True
    assert episode.metrics["mode_is_post_action_simulator"] == 1.0
    assert episode.metrics["simulation_present_rate"] == 1.0


def test_post_action_simulator_does_not_duplicate_context_when_live_messages_enabled():
    simulator_text = (
        "Reasoning.\n"
        "<simulation>Likely returns 4 from the python tool.</simulation>"
    )
    prediction_text = "Short reasoning.\n<prediction>4</prediction>"
    _episode, workflow, rollout_engine = _run_workflow(
        responses=["TOOL:print(2 + 2)", simulator_text, prediction_text, "FINAL 4"],
        generative_support_cfg={
            "mode": "post_action_simulator",
            "enable_prediction": True,
            "add_to_live_messages": True,
            "add_to_step_chat_completions": True,
        },
    )

    pred_call = next(
        call for call in rollout_engine.calls if call["application_id"] == "task-0:pred:0"
    )
    simulator_prompts = [
        message
        for message in pred_call["messages"]
        if message.get("role") == "user" and "SIMULATION MODE" in message.get("content", "")
    ]
    assert len(simulator_prompts) == 1
    assert sum(
        1
        for message in pred_call["messages"]
        if message.get("role") == "assistant"
        and message.get("content") == simulator_text
    ) == 1
    assert any(message.get("rllm_simulation") for message in workflow.agent.messages)


def test_auxiliary_turns_do_not_duplicate_reasoning_field_when_raw_text_is_kept():
    simulator_text = (
        "Reasoning.\n"
        "<simulation>Likely returns 4 from the python tool.</simulation>"
    )
    prediction_text = "Short reasoning.\n<prediction>4</prediction>"
    episode, _workflow, _rollout_engine = _run_workflow(
        responses=["TOOL:print(2 + 2)", simulator_text, prediction_text, "FINAL 4"],
        generative_support_cfg={
            "mode": "post_action_simulator",
            "enable_prediction": True,
            "add_to_live_messages": False,
            "add_to_step_chat_completions": True,
        },
    )

    first_step_messages = episode.trajectories[0].steps[0].chat_completions
    simulation_message = next(
        message for message in first_step_messages if message.get("rllm_simulation")
    )
    prediction_message = next(
        message for message in first_step_messages if message.get("rllm_prediction")
    )
    assert "reasoning" not in simulation_message
    assert "reasoning" not in prediction_message


def test_pre_action_imagine_then_revise_stores_imagine_record_and_keeps_it_out_of_live_context():
    imagine_text = "Reasoning.\n<imagine>4</imagine>"
    finalize_text = "Reasoning.\n<prediction>6</prediction>\nTOOL_ALT:print(3 + 3)"
    episode, workflow, rollout_engine = _run_workflow(
        responses=["TOOL:print(2 + 2)", imagine_text, finalize_text, "FINAL 6"],
        generative_support_cfg={
            "mode": "pre_action_imagine_then_revise",
            "enable_prediction": True,
            "add_to_live_messages": False,
            "add_to_step_chat_completions": True,
        },
    )

    assert [call["application_id"] for call in rollout_engine.calls] == [
        "task-0:cand:0",
        "task-0:imagine:0",
        "task-0:final:0",
        "task-0:cand:1",
    ]

    first_step = episode.trajectories[0].steps[0]
    imagine_record = first_step.info[PredictiveToolAgent.INFO_KEY_IMAGINE]
    assert imagine_record["prediction"] == "4"
    assert "IMAGINE MODE" in imagine_record["prompt"]
    assert first_step.info[PredictiveToolAgent.INFO_KEY_ACTION_REVISED] is True
    assert first_step.info[PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT] == "6"
    assert episode.metrics["mode_is_pre_action_imagine_then_revise"] == 1.0
    assert episode.metrics["imagine_present_rate"] == 1.0
    assert episode.info["generative_support_mode"] == "pre_action_imagine_then_revise"

    second_candidate_call = next(
        call for call in rollout_engine.calls if call["application_id"] == "task-0:cand:1"
    )
    assert not any(
        "IMAGINE MODE" in message.get("content", "")
        for message in second_candidate_call["messages"]
    )
    assert not any(
        "IMAGINED_CANDIDATE_OUTPUT" in message.get("content", "")
        for message in second_candidate_call["messages"]
    )
    assert not any("IMAGINE MODE" in message.get("content", "") for message in workflow.agent.messages)


def test_finish_only_action_does_not_trigger_new_generative_support():
    episode, _workflow, rollout_engine = _run_workflow(
        responses=["FINAL 4"],
        generative_support_cfg={
            "mode": "post_action_simulator",
            "enable_prediction": True,
        },
    )

    assert len(episode.trajectories[0].steps) == 1
    assert [call["application_id"] for call in rollout_engine.calls] == ["task-0:act:0"]
