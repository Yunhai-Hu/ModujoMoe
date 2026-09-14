#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODUJO_SMALL_MODEL:-/workspace/models/modujo-723m}
DATA_DIR=${MODUJO_DATA_DIR:-/workspace/datasets/modujo}
OUTPUT_ROOT=${MODUJO_OUTPUT_DIR:-/workspace/output}
export USE_HF=1
unset LMDEPLOY_USE_MODELSCOPE VLLM_USE_MODELSCOPE MODELSCOPE_CACHE
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_PROJECT=${WANDB_PROJECT:-modujo-pretrain}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Small model not found at $MODEL_DIR" >&2
  echo "Run: python pretrain/scripts/prepare_small_model.py --output $MODEL_DIR" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} swift pt \
  --model "$MODEL_DIR" \
  --model_type modujo_qwen4_exp \
  --external_plugins "$ROOT_DIR/common/modujo_swift_plugin.py" \
  --experts_impl grouped_mm \
  --dataset "$DATA_DIR/train_bilingual_packed_2048.jsonl" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --bf16 true \
  --tf32 true \
  --max_length 2048 \
  --packing false \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing false \
  --num_train_epochs 3 \
  --learning_rate 3e-4 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --warmup_ratio 0.02 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --optim paged_adamw_8bit \
  --max_grad_norm 1.0 \
  --router_aux_loss_coef 0.001 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --report_to wandb \
  --dataset_num_proc 8 \
  --output_dir "$OUTPUT_ROOT/modujo-723m-pretrain" \
  "$@"
