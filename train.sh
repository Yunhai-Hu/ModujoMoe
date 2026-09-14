#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=/workspace/output/modujo-pretrain/v0-20260914-114152/checkpoint-25
OUT_DIR=/workspace/output/modujo-pretrain-2048-ga64

CUDA_VISIBLE_DEVICES=0 swift pt \
  --model "$MODEL_DIR" \
  --model_type modujo_qwen4_exp \
  --external_plugins /workspace/modujo_pretrain/modujo_swift_plugin.py \
  --dataset /workspace/datasets/modujo/train_bilingual_packed_2048.jsonl \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --bf16 true \
  --tf32 true \
  --max_length 2048 \
  --packing false \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --gradient_checkpointing false \
  --learning_rate 3e-4 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --warmup_ratio 0.02 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --optim adamw_bnb_8bit \
  --max_grad_norm 1.0 \
  --router_aux_loss_coef 0.001 \
  --max_steps 100 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --report_to none \
  --dataset_num_proc 8 \
  --output_dir "$OUT_DIR" \
  "$@"
