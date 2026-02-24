#!/bin/bash
set -x

# 自动检测当前路径并source对应的环境配置脚本
CURRENT_DIR=$(pwd)

echo "Current directory: $CURRENT_DIR"

# 检查当前路径前缀并source对应的环境脚本
if [[ "$CURRENT_DIR" == /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/* ]]; then
    echo "Detected ssd_pool path, sourcing sh_env.sh..."
    source $(dirname "$0")/sh_env.sh
elif [[ "$CURRENT_DIR" == /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/* ]]; then
    echo "Detected hdd_pool path, sourcing zw_env.sh..."
    source $(dirname "$0")/zw_env.sh
else
    echo "Warning: Unknown path prefix, defaulting to sh_env.sh..."
    source $(dirname "$0")/sh_env.sh
fi
