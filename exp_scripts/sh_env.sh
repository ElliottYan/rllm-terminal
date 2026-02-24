set -x

pip3 install --upgrade pip -i http://pypi.sankuai.com/simple/ --trusted-host pypi.sankuai.com

pip3 install --upgrade setuptools>=64.0.0 -i http://pypi.sankuai.com/simple/ --trusted-host pypi.sankuai.com

#LOCAL_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/yanjianhao03/
LOCAL_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/yanjianhao/
source $LOCAL_DIR/envs.sh

# pip3 install hope==3.6.8 -i http://pypi.sankuai.com/simple/ --trusted-host pypi.sankuai.com --upgrade
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

pip3 install hope==3.6.8 -i http://pypi.sankuai.com/simple/ --trusted-host pypi.sankuai.com --upgrade

# MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/yinyongjing/hfmodels/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen3-8B

# LOCAL_PWD=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/yinyongjing
LOCAL_PWD=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/yanjianhao/rllm-terminal

EXP=mathtool_local_tensorboard_gspo
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

export PYTHONPATH=./rllm-terminal:$PYTHONPATH
export TENSORBOARD_DIR=$LOCAL_PWD/tensorboard/$EXP
mkdir -p $TENSORBOARD_DIR
