import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent
from rllm_ext.environments.predictive_tool_env import PredictiveToolEnvironment
from rllm_ext.workflows.predictive_tool_workflow import PredictiveToolWorkflow


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    """
    PPO training example (workflow mode) that adds an explicit prediction sub-step:
    action -> predict outcome -> env.step(action).

    This keeps core `rllm` untouched by implementing everything in `rllm_ext`.
    """
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    # Enable workflow training path
    config.rllm.workflow.use_workflow = True

    agent_args = {
        "tools": ["python"],
        "parser_name": "qwen",
        "system_prompt": (
            "You are a math assistant that can write python to solve math problems.\n"
            "You will use tools when helpful.\n"
            "Before executing a tool action, you will also be asked to predict what the tool will output."
        ),
    }
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    workflow_args = {
        "agent_cls": PredictiveToolAgent,
        "env_cls": PredictiveToolEnvironment,
        "agent_args": agent_args,
        "env_args": env_args,
        "max_steps": 10,
        "prediction_cfg": {
            "enabled": True,
            "max_tokens": 256,
            "add_prediction_to_messages": True,
        },
    }

    trainer = AgentTrainer(
        workflow_class=PredictiveToolWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

