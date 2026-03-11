git aimport hydra
import os

from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

from rllm_ext.agents.predictive_tool_agent import PredictiveToolAgent
from rllm_ext.environments.predictive_tool_env import PredictiveToolEnvironment
from rllm_ext.workflows.predictive_tool_workflow import PredictiveToolWorkflow
from rllm_ext import PredictiveAgentTrainer

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    """
    PPO training example (workflow mode) that adds an explicit prediction sub-step:
    action -> predict outcome -> env.step(action).

    This keeps core `rllm` untouched by implementing everything in `rllm_ext`.
    """
    # Print current hydra config for reproducibility/debugging.
    try:
        print("Current config (resolved):")
        print(OmegaConf.to_yaml(config, resolve=True))
    except Exception:
        print("Current config (unresolved):")
        print(OmegaConf.to_yaml(config, resolve=False))

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

    # Read feature flags from config (with defaults)
    enable_prediction = config.get("enable_prediction", False)
    enable_similarity_reward = config.get("enable_similarity_reward", False)
    # max_steps can be set either at:
    # 1) top-level `max_steps` (recommended for this example script)
    # 2) `rllm.workflow.workflow_args.max_steps`
    max_steps = config.get("max_steps", None)
    if max_steps is None and config.get("rllm") is not None and config.rllm.get("workflow") is not None:
        workflow_args_cfg = config.rllm.workflow.get("workflow_args")
        if workflow_args_cfg is not None:
            max_steps = workflow_args_cfg.get("max_steps")
    if max_steps is None:
        max_steps = 10

    # prediction_cfg can be set either at:
    # 1) top-level `prediction_cfg` (recommended for this example script)
    # 2) `rllm.workflow.workflow_args.prediction_cfg`
    prediction_cfg_from_config = {}
    if config.get("prediction_cfg") is not None:
        prediction_cfg_from_config = OmegaConf.to_container(config.get("prediction_cfg"), resolve=True)
    elif (
        config.get("rllm") is not None
        and config.rllm.get("workflow") is not None
        and config.rllm.workflow.get("workflow_args") is not None
        and config.rllm.workflow.workflow_args.get("prediction_cfg") is not None
    ):
        prediction_cfg_from_config = OmegaConf.to_container(
            config.rllm.workflow.workflow_args.get("prediction_cfg"),
            resolve=True,
        )

    default_prediction_cfg = {
        "enabled": enable_prediction,
        "collect_loss_targets": config.actor_rollout_ref.actor.get("prediction_loss_weight", 0) > 0,
        "max_tokens": 256,
        "add_prediction_to_messages": True,
    }
    prediction_cfg = {**default_prediction_cfg, **prediction_cfg_from_config}

    # trajectory_logging can be set either at:
    # 1) top-level `trajectory_logging` (recommended for this example script)
    # 2) `rllm.workflow.workflow_args.trajectory_logging`
    trajectory_logging_from_config = {}
    if config.get("trajectory_logging") is not None:
        trajectory_logging_from_config = OmegaConf.to_container(
            config.get("trajectory_logging"),
            resolve=True,
        )
    elif (
        config.get("rllm") is not None
        and config.rllm.get("workflow") is not None
        and config.rllm.workflow.get("workflow_args") is not None
        and config.rllm.workflow.workflow_args.get("trajectory_logging") is not None
    ):
        trajectory_logging_from_config = OmegaConf.to_container(
            config.rllm.workflow.workflow_args.get("trajectory_logging"),
            resolve=True,
        )

    agent_args = {
        "tools": ["python"],
        "parser_name": "qwen",
        "system_prompt": (
            "You are a math assistant that can write python to solve math problems.\n"
            "You will use tools when helpful.\n"
            + ("Before executing a tool action, you will also be asked to predict what the tool will output." if enable_prediction else "")
        ),
    }
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
        "similarity_config": {
            "enabled": enable_similarity_reward,  # Controlled by config.enable_similarity_reward
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
        "max_steps": int(max_steps),
        "prediction_cfg": prediction_cfg,
    }
    if trajectory_logging_from_config:
        workflow_args["trajectory_logging"] = trajectory_logging_from_config

    use_predictive_trainer = True
    if use_predictive_trainer is True:
        AgentTrainer = PredictiveAgentTrainer
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
