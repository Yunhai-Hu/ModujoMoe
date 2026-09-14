#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODUJO_DENSE_MODEL:-/workspace/models/modujo-dense-700m}
DATA_DIR=${MODUJO_DATA_DIR:-/workspace/datasets/modujo}
OUTPUT_ROOT=${MODUJO_OUTPUT_DIR:-/workspace/output}
export USE_HF=1
unset LMDEPLOY_USE_MODELSCOPE VLLM_USE_MODELSCOPE MODELSCOPE_CACHE
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_PROJECT=${WANDB_PROJECT:-modujo-pretrain}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Dense model not found at $MODEL_DIR" >&2
  echo "Run: python pretrain/scripts/prepare_dense_model.py --output $MODEL_DIR" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} swift pt \
  --model "$MODEL_DIR" \
  --model_type qwen3 \
  --dataset "$DATA_DIR/smoke_bilingual.jsonl" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --bf16 true \
  --tf32 true \
  --max_length 512 \
  --packing false \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing false \
  --learning_rate 3e-4 \
  --optim adamw_torch \
  --max_steps 2 \
  --save_strategy no \
  --logging_steps 1 \
  --report_to wandb \
  --output_dir "$OUTPUT_ROOT/modujo-dense-700m-smoke" \
  "$@"
