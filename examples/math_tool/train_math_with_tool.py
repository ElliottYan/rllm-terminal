import hydra

from dataset_loader import load_dataset_with_path_override
from rllm.agents import ToolAgent
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = load_dataset_with_path_override(config, "deepscaler_math", "train")
    test_dataset = load_dataset_with_path_override(config, "aime2024", "test")

    agent_args = {"tools": ["python"], "parser_name": "qwen", "system_prompt": "You are a math assistant that can write python to solve math problems."}
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
