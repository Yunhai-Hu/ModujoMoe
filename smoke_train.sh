#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=/root/.cache/huggingface/hub/models--Alexhu1999--Modujo-9B-A1B/snapshots/a18b15449fbce9d4e182138704d8d015f2dadfbf
OUT_DIR=/workspace/output/modujo-smoke

CUDA_VISIBLE_DEVICES=0 swift pt \
  --model "$MODEL_DIR" \
  --model_type modujo_qwen4_exp \
  --external_plugins /workspace/modujo_pretrain/modujo_swift_plugin.py \
  --dataset /workspace/datasets/modujo/smoke_bilingual.jsonl \
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
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --weight_decay 0.1 \
  --optim adamw_bnb_8bit \
  --max_grad_norm 1.0 \
  --max_steps 2 \
  --save_strategy no \
  --logging_steps 1 \
  --report_to none \
  --dataset_num_proc 4 \
  --output_dir "$OUT_DIR"
