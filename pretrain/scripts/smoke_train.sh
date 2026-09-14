#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODUJO_BASE_MODEL:-Alexhu1999/Modujo-9B-A1B}
DATA_DIR=${MODUJO_DATA_DIR:-/workspace/datasets/modujo}
OUTPUT_ROOT=${MODUJO_OUTPUT_DIR:-/workspace/output}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} swift pt \
  --model "$MODEL_DIR" \
  --model_type modujo_qwen4_exp \
  --external_plugins "$ROOT_DIR/common/modujo_swift_plugin.py" \
  --dataset "$DATA_DIR/smoke_bilingual.jsonl" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --bf16 true \
  --tf32 true \
  --max_length 512 \
  --packing false \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing true \
  --learning_rate 3e-4 \
  --optim adamw_bnb_8bit \
  --max_steps 2 \
  --save_strategy no \
  --logging_steps 1 \
  --report_to none \
  --output_dir "$OUTPUT_ROOT/modujo-smoke"
