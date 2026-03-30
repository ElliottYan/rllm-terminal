import asyncio
import copy
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace


def _install_optional_dependency_stubs() -> None:
    if "pylatexenc" not in sys.modules:
        latex2text = ModuleType("pylatexenc.latex2text")

        class _LatexNodes2Text:
            def latex_to_text(self, expr):
                return expr

        latex2text.LatexNodes2Text = _LatexNodes2Text
        pylatexenc = ModuleType("pylatexenc")
        pylatexenc.latex2text = latex2text
        sys.modules["pylatexenc"] = pylatexenc
        sys.modules["pylatexenc.latex2text"] = latex2text

    if "httpx" not in sys.modules:
        httpx = ModuleType("httpx")

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
        firecrawl = ModuleType("firecrawl")
        firecrawl.FirecrawlApp = SimpleNamespace
        sys.modules["firecrawl"] = firecrawl

    if "transformers" not in sys.modules:
        transformers = ModuleType("transformers")

        class _PreTrainedTokenizerBase:
            pass

        transformers.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
        sys.modules["transformers"] = transformers

    if "PIL" not in sys.modules:
        pil = ModuleType("PIL")
        image = ModuleType("PIL.Image")
        pil.Image = image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image


_install_optional_dependency_stubs()

from rllm.agents.agent import Action, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent
from rllm_ext.workflows.predictive_tool_workflow import PredictiveToolWorkflow


class _DummyPredictiveAgent(PredictiveToolAgent):
    def __init__(self, **kwargs):
        self.system_prompt = "sys"
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
        if response.startswith("TOOL:"):
            tool_calls_dict = [
                {
                    "id": f"tool-{step_idx}",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": '{"code": "print(2 + 2)"}',
                    },
                }
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

        return (
            {"tool_outputs": {tool_calls[0]["id"]: "4"}},
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


def _run_workflow(
    add_prediction_to_messages: bool,
    *,
    enable_prediction_loss: bool = True,
    enable_prediction_step: bool = False,
    enforce_max_prompt_length: bool = True,
):
    responses = ["TOOL:print(2 + 2)"]
    if enable_prediction_step:
        responses.append("<prediction>4</prediction>")
    responses.append("FINAL 4")
    rollout_engine = _DummyRolloutEngine(responses)
    task = {"question": "What is 2 + 2?"}

    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = PredictiveToolWorkflow(
            agent_cls=_DummyPredictiveAgent,
            env_cls=_DummyEnv,
            max_steps=4,
            prediction_cfg={
                "enabled": True,
                "enable_prediction_loss": enable_prediction_loss,
                "enable_prediction_step": enable_prediction_step,
                "enforce_max_prompt_length": enforce_max_prompt_length,
                "add_prediction_to_messages": add_prediction_to_messages,
            },
            rollout_engine=rollout_engine,
            executor=executor,
        )

        episode = asyncio.run(workflow.run(task=task, uid="task-0"))
    return episode, workflow, rollout_engine


def test_prediction_loss_keeps_rollout_context_clean_and_stores_metadata():
    episode, workflow, rollout_engine = _run_workflow(
        add_prediction_to_messages=False
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

    assert not any(
        "PREDICTION MODE" in message.get("content", "")
        for message in workflow.agent.messages
    )
    assert not any(
        message.get("rllm_prediction", False) for message in workflow.agent.messages
    )

    first_step_info = trajectory.steps[0].info
    prediction_record = first_step_info[PredictiveToolAgent.INFO_KEY_PREDICTION]
    assert "PREDICTION MODE" in prediction_record["prompt"]
    assert (
        first_step_info[PredictiveToolAgent.INFO_KEY_ACTUAL_OUTPUT] == "4"
    )


def test_prediction_messages_still_reach_live_context_when_explicitly_enabled():
    episode, workflow, rollout_engine = _run_workflow(
        add_prediction_to_messages=True
    )

    trajectory = episode.trajectories[0]
    assert len(trajectory.steps) == 2

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


def test_prediction_step_inherits_prompt_length_enforcement_setting():
    _episode, _workflow, rollout_engine = _run_workflow(
        add_prediction_to_messages=False,
        enable_prediction_loss=False,
        enable_prediction_step=True,
        enforce_max_prompt_length=False,
    )

    pred_call = next(
        call for call in rollout_engine.calls if call["application_id"] == "task-0:pred:0"
    )
    assert pred_call["enforce_max_prompt_length"] is False
