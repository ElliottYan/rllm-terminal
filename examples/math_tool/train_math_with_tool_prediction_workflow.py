import hydra
import os

from hydra.utils import get_original_cwd
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
    # Set TENSORBOARD_DIR environment variable before training starts
    # Priority: 1) Existing TENSORBOARD_DIR env var
    #            2) config.trainer.tensorboard_dir if it exists and is not None
    #            3) Fallback to <original_cwd>/tensorboard/predict_test
    if "TENSORBOARD_DIR" not in os.environ or not os.environ.get("TENSORBOARD_DIR"):
        if hasattr(config, "trainer") and hasattr(config.trainer, "tensorboard_dir"):
            tb_dir_from_cfg = config.trainer.tensorboard_dir
            if tb_dir_from_cfg is not None:
                os.environ["TENSORBOARD_DIR"] = tb_dir_from_cfg
                print(f"TENSORBOARD_DIR set from config.trainer.tensorboard_dir: {tb_dir_from_cfg}")
        else:
            # Fallback to default path
            original_cwd = get_original_cwd()
            tensorboard_dir = os.path.join(original_cwd, "tensorboard", "predict_test")
            os.environ["TENSORBOARD_DIR"] = tensorboard_dir
            print(f"TENSORBOARD_DIR set to default path: {tensorboard_dir}")
    else:
        print(f"TENSORBOARD_DIR already set: {os.environ['TENSORBOARD_DIR']}")

    # Ensure directory exists
    os.makedirs(os.environ["TENSORBOARD_DIR"], exist_ok=True)

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
        "similarity_config": {
            "enabled": True,
            "weight": 0.1,        # Max 0.1 reward per step for accurate predictions
            "n": 4,               # BLEU-4 style n-gram similarity
            "min_length": 4,      # Skip outputs shorter than 4 words
            "smoothing": True,    # Add +1 smoothing
        },
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

