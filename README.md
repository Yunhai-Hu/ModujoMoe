# ModujoMoe

An end-to-end training workspace for the Modujo 9B-A1B MoE language model,
built around ModelScope ms-swift.

## Training stages

| Stage | Directory | Status | Input | Output |
| --- | --- | --- | --- | --- |
| 1 | [`pretrain/`](pretrain/) | Active | Raw bilingual text | Base model |
| 2 | [`sft/`](sft/) | Planned | Instruction conversations | Assistant model |
| 3 | [`rl/`](rl/) | Planned | Prompts, rewards and rollouts | Aligned policy |
| 4 | [`opd/`](opd/) | Planned | To be defined | Final model |

Shared ms-swift model registration lives in [`common/`](common/). Each stage
owns its data preparation, configuration, scripts, documentation, evaluation,
and output conventions. Generated datasets, logs, caches, and checkpoints are
excluded from Git.

## Current stage: pretraining

The source checkpoint is randomly initialized and must be pretrained with full
parameters. The current recipe uses FineWeb-Edu English and FineWeb 2 Chinese,
offline-packed to 2048 tokens. Dataset revisions are recorded in
[`sources.json`](sources.json).

```bash
export MODUJO_BASE_MODEL=Alexhu1999/Modujo-9B-A1B
python pretrain/data/prepare_smoke_data.py
bash pretrain/scripts/smoke_train.sh

python pretrain/data/prepare_train_data.py
python pretrain/data/pack_documents.py
bash pretrain/scripts/train.sh
```

On one 96 GB RTX PRO 6000, the tuned recipe uses micro-batch 2, gradient
accumulation 64, BF16, and no gradient checkpointing. Micro-batch 2 peaks at
93.41 GiB with fused `grouped_mm` experts; micro-batch 3 runs out of memory.
Grouped experts cut the measured micro-step time from about 27 to 12.76 seconds;
continuous SM utilization is still limited by the single-GPU kernel path.
