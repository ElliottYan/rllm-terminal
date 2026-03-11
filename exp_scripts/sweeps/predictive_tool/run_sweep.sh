#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/exp_scripts/base.sh"
BASE_ENV_FILE="${SCRIPT_DIR}/base.env"
MATRIX_FILE="${SCRIPT_DIR}/matrix.tsv"

source "${BASE_ENV_FILE}"

ONLY_RUN_ID="${ONLY_RUN_ID:-}"
START_FROM_INDEX="${START_FROM_INDEX:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
    echo "base script not found: ${BASE_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${MATRIX_FILE}" ]]; then
    echo "matrix file not found: ${MATRIX_FILE}" >&2
    exit 1
fi

run_idx=0
selected_idx=0

while IFS=$'\t' read -r run_id enable_prediction enable_similarity_reward add_prediction_to_messages simple_tir prediction_max_tokens prediction_loss_weight total_epochs max_steps extra_overrides; do
    if [[ -z "${run_id}" || "${run_id}" == \#* || "${run_id}" == "run_id" ]]; then
        continue
    fi

    run_idx=$((run_idx + 1))
    if (( run_idx < START_FROM_INDEX )); then
        continue
    fi

    if [[ -n "${ONLY_RUN_ID}" && "${run_id}" != "${ONLY_RUN_ID}" ]]; then
        continue
    fi

    selected_idx=$((selected_idx + 1))
    exp_name="${SWEEP_NAME}__${run_id}"
    proj_dir="${REPO_ROOT}/exp/${SWEEP_NAME}/${run_id}"
    tb_dir="${REPO_ROOT}/tensorboard/${SWEEP_NAME}/${run_id}"

    extra_args=()
    if [[ -n "${extra_overrides:-}" && "${extra_overrides}" != "-" ]]; then
        IFS=';' read -r -a extra_args <<< "${extra_overrides}"
    fi

    cmd=(
        env
        "EXP=${exp_name}"
        "PROJ_DIR=${proj_dir}"
        "TENSORBOARD_DIR=${tb_dir}"
        "PROJECT_NAME=${PROJECT_NAME}"
        "NNODE=${NNODE}"
        "NGPUS=${NGPUS}"
        "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
        "TRAIN_MINI_SIZE=${TRAIN_MINI_SIZE}"
        "SKIP_INSTALL=${SKIP_INSTALL}"
        "ENABLE_PREDICTION=${enable_prediction}"
        "ENABLE_SIMILARITY_REWARD=${enable_similarity_reward}"
        "ADD_PREDICTION_TO_MESSAGES=${add_prediction_to_messages}"
        "SIMPLE_TIR=${simple_tir}"
        "PREDICTION_MAX_TOKENS=${prediction_max_tokens}"
        "PREDICTION_LOSS_WEIGHT=${prediction_loss_weight}"
        "PREDICTION_LOSS_TYPE=${PREDICTION_LOSS_TYPE}"
        "PREDICTION_TEMPERATURE=${PREDICTION_TEMPERATURE}"
        "TOTAL_EPOCHS=${total_epochs}"
        "MAX_STEPS=${max_steps}"
        bash "${BASE_SCRIPT}"
    )

    if [[ "${#extra_args[@]}" -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "========== [${selected_idx}] ${run_id} =========="
    echo "EXP=${exp_name}"
    echo "PROJ_DIR=${proj_dir}"
    echo "TENSORBOARD_DIR=${tb_dir}"
    echo "FLAGS: pred=${enable_prediction}, sim=${enable_similarity_reward}, add_msg=${add_prediction_to_messages}, simple_tir=${simple_tir}, max_tok=${prediction_max_tokens}, pred_loss=${prediction_loss_weight}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN CMD: '
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
done < "${MATRIX_FILE}"

if (( selected_idx == 0 )); then
    echo "No runs selected. Check ONLY_RUN_ID=${ONLY_RUN_ID} and START_FROM_INDEX=${START_FROM_INDEX}."
else
    echo "Sweep finished. Total selected runs: ${selected_idx}"
fi
