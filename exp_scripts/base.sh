#!/usr/bin/env bash
#set -xeuo pipefail
set -x
ray stop

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/auto_env.sh"

EXP="${EXP:-pred_max2}"
PROJ_DIR="${PROJ_DIR:-${LOCAL_PWD}/exp/${EXP}}"
OFFLOAD="${OFFLOAD:-False}"
NNODE="${NNODE:-1}"
NGPUS="${NGPUS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_MINI_SIZE="${TRAIN_MINI_SIZE:-32}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
MAX_STEPS="${MAX_STEPS:-2}"

ENABLE_PREDICTION="${ENABLE_PREDICTION:-false}"
ENABLE_SIMILARITY_REWARD="${ENABLE_SIMILARITY_REWARD:-false}"
PREDICTION_MAX_TOKENS="${PREDICTION_MAX_TOKENS:-256}"
ADD_PREDICTION_TO_MESSAGES="${ADD_PREDICTION_TO_MESSAGES:-true}"
SIMPLE_TIR="${SIMPLE_TIR:-false}"

PREDICTION_LOSS_WEIGHT="${PREDICTION_LOSS_WEIGHT:-0.0}"
PREDICTION_LOSS_TYPE="${PREDICTION_LOSS_TYPE:-cross_entropy}"
PREDICTION_TEMPERATURE="${PREDICTION_TEMPERATURE:-1.0}"

SAVE_FREQ="${SAVE_FREQ:-1}"

PROJECT_NAME="${PROJECT_NAME:-rllm-agent}"
# RAY_LAUNCH_SCRIPT="${RAY_LAUNCH_SCRIPT:-/$LOCAL_DIR/rllm-terminal/examples/math_tool/ray_launch.py}"
RAY_LAUNCH_SCRIPT=$LOCAL_DIR/rllm-terminal/examples/math_tool/ray_launch.py
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${LOCAL_PWD}/examples/math_tool/train_math_with_tool_prediction_workflow.py}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

EXTRA_HYDRA_ARGS=("$@")

mkdir -p "${PROJ_DIR}"

cd "${LOCAL_PWD}"
if [[ "${SKIP_INSTALL}" != "1" ]]; then
    uv pip install --system -e . -i http://pypi.sankuai.com/simple/ --trusted-host pypi.sankuai.com
fi

launch="$(python3 "${RAY_LAUNCH_SCRIPT}")"
eval "${launch}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${LOCAL_PWD}:${PYTHONPATH:-}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOCAL_PWD}/tensorboard/${EXP}}"
mkdir -p "${TENSORBOARD_DIR}"

export RAY_DEBUG=legacy

export WANDB_MODE=offline

export WANDB_DIR=$LOCAL_PWD/wandb-negative
export WANDB_CACHE_DIR=$WANDB_DIR/.cache/wandb
export WANDB_CONFIG_DIR=$WANDB_DIR/.config/wandb
export WANDB_DATA_DIR=$WANDB_DIR/.config/wandb-data
export WANDB_ARTIFACT_DIR=$WANDB_DIR/artifacts
mkdir -p $WANDB_DIR
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR
mkdir -p $WANDB_DATA_DIR
mkdir -p $WANDB_ARTIFACT_DIR


CMD=(
    python3 "${TRAIN_SCRIPT}"
    "algorithm.adv_estimator=grpo"
    "actor_rollout_ref.actor.policy_loss.loss_mode=gpg"
    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=500"
    "data.max_prompt_length=2048"
    "data.max_response_length=8192"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.hybrid_engine=True"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_MINI_SIZE}"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.clip_ratio_high=0.28"
    "actor_rollout_ref.actor.kl_loss_coef=0.001"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.temperature=0.6"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.8"
    "actor_rollout_ref.rollout.n=8"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.6"
    "actor_rollout_ref.rollout.val_kwargs.top_p=0.95"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "algorithm.kl_ctrl.kl_coef=0.001"
    "rllm.mask_truncated_samples=False"
    "trainer.critic_warmup=0"
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXP}"
    "trainer.val_before_train=False"
    "trainer.n_gpus_per_node=${NGPUS}"
    "trainer.nnodes=${NNODE}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=20"
    "trainer.default_hdfs_dir=null"
    "trainer.default_local_dir=${PROJ_DIR}"
    "rllm.agent.max_steps=${MAX_STEPS}"
    "+max_steps=${MAX_STEPS}"
    "rllm.stepwise_advantage.enable=False"
    "+enable_prediction=${ENABLE_PREDICTION}"
    "+enable_similarity_reward=${ENABLE_SIMILARITY_REWARD}"
    "+prediction_cfg.max_tokens=${PREDICTION_MAX_TOKENS}"
    "+prediction_cfg.add_prediction_to_messages=${ADD_PREDICTION_TO_MESSAGES}"
    "+prediction_cfg.simple_tir=${SIMPLE_TIR}"
    "+actor_rollout_ref.actor.prediction_loss_weight=${PREDICTION_LOSS_WEIGHT}"
    "+actor_rollout_ref.actor.prediction_loss_type=${PREDICTION_LOSS_TYPE}"
    "+actor_rollout_ref.actor.prediction_temperature=${PREDICTION_TEMPERATURE}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
)

if [[ "${#EXTRA_HYDRA_ARGS[@]}" -gt 0 ]]; then
    CMD+=("${EXTRA_HYDRA_ARGS[@]}")
fi

"${CMD[@]}"
