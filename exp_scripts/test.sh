set -x
source auto_env.sh

EXP=test
PROJ_DIR=${LOCAL_PWD}/exp/${EXP}
mkdir -p $PROJ_DIR

offload=False

cd $LOCAL_PWD
# WORK_DIR=/workdir/rllm-terminal
# cd $WORK_DIR
# cp $LOCAL_PWD/rllm/registry/dataset_registry.json $WORK_DIR/rllm/registry/dataset_registry.json
uv pip install --system -e . -i http://pypi.sankuai.com/simple/ --trusted-host pypi.sankuai.com

launch=$(python3 /workdir/rllm-terminal/examples/math_tool/ray_launch.py)
eval "${launch}"

export HYDRA_FULL_ERROR=1

export PYTHONPATH=./rllm-terminal:$PYTHONPATHie
export TENSORBOARD_DIR=$LOCAL_PWD/tensorboard/$EXP
mkdir -p $TENSORBOARD_DIR


NNODE=1
NGPUS=2

# if [ $rank -eq 0 ]; then
# python3 /workdir/rllm-terminal/examples/math_tool/train_math_with_tool.py \
python3 $LOCAL_PWD/examples/math_tool/train_math_with_tool_prediction_workflow.py \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='4b-math-tool' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=$NNODE \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    rllm.agent.max_steps=2 \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=100
# fi
