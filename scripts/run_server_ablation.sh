#!/usr/bin/env bash
set -euo pipefail

# Run from RPCANet-main after copying the integration files described in README.
# Override these environment variables to match the server.
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
NUDT_DATA_ROOT="${NUDT_DATA_ROOT:-./RiRUFold_ADMM/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./result_rirufold_rcp}"
EPOCHS="${EPOCHS:-400}"
SEEDS="${SEEDS:-42 3407 2026}"
EVALUATE_AFTER_TRAIN="${EVALUATE_AFTER_TRAIN:-1}"

variants=(
  rirufold_admm
  rirufold_rcp_product
  rirufold_rcp
  rirufold_rcp_nofeedback
  rirufold_rcp_global
)
if [[ "${INCLUDE_BLOB:-0}" == "1" ]]; then
  variants+=(rirufold_rcp_blob)
fi
datasets=(nudt irstd1k sirstaug)

for seed in ${SEEDS}; do
  for dataset in "${datasets[@]}"; do
    root="${DATA_ROOT}"
    if [[ "${dataset}" == "nudt" ]]; then
      root="${NUDT_DATA_ROOT}"
    fi
    for variant in "${variants[@]}"; do
      run_name="seed_${seed}/${dataset}/${variant}"
      if [[ -e "${OUTPUT_ROOT}/${run_name}" ]]; then
        echo "Refusing to overwrite existing run: ${OUTPUT_ROOT}/${run_name}" >&2
        echo "Choose a new OUTPUT_ROOT or remove/archive that exact run explicitly." >&2
        exit 2
      fi
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
        --net-name "${variant}" \
        --dataset "${dataset}" \
        --data-root "${root}" \
        --epochs "${EPOCHS}" \
        --lr 1e-4 \
        --batch-size 4 \
        --gpu "${GPU_ID}" \
        --seed "${seed}" \
        --admm-stage-num 5 \
        --admm-hidden-channels 24 \
        --dc-weight 0.05 \
        --bg-weight 0.02 \
        --prox-weight 0.005 \
        --save-iter-step 100 \
        --log-per-iter 10 \
        --base-dir "${OUTPUT_ROOT}" \
        --run-name "${run_name}"

      if [[ "${EVALUATE_AFTER_TRAIN}" == "1" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python RiRUFold_ADMM/scripts/evaluate_rirufold_rcp.py \
          --net-name "${variant}" \
          --checkpoint "${OUTPUT_ROOT}/${run_name}/latest.pkl" \
          --dataset "${dataset}" \
          --seed "${seed}" \
          --data-root "${root}" \
          --base-size 256 \
          --batch-size 4 \
          --stage-num 5 \
          --hidden-channels 24 \
          --threshold 0.5 \
          --gpu "${GPU_ID}" \
          --output "${OUTPUT_ROOT}/metrics/seed_${seed}/${dataset}/${variant}.json"
      fi
    done
  done
done

if [[ "${EVALUATE_AFTER_TRAIN}" == "1" ]]; then
  python RiRUFold_ADMM/scripts/summarize_server_results.py \
    --input-root "${OUTPUT_ROOT}/metrics" \
    --output "${OUTPUT_ROOT}/summary.csv"
fi
