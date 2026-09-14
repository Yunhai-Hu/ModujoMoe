# Pretraining

This stage turns the randomly initialized Modujo checkpoint into a base model.

1. Run `python data/prepare_smoke_data.py` and `bash scripts/smoke_train.sh`.
2. Run `python data/prepare_train_data.py`.
3. Run `python data/pack_documents.py`.
4. Run `bash scripts/train.sh`.

For a cheaper architecture-level test, build and train the 723M-parameter
variant. It retains the Qwen4-Exp hybrid-attention/MoE layout but uses 16
layers, width 768, and 8 top-2 experts:

```bash
python pretrain/scripts/prepare_small_model.py
bash pretrain/scripts/small_smoke_train.sh
bash pretrain/scripts/small_train.sh
```

The generated checkpoint defaults to `/workspace/models/modujo-723m` and is
kept outside Git. The commands accept overrides; run the Python command with
`--help`, or pass normal `swift pt` arguments after the shell command.
The formal small-model recipe trains on the packed 2048-token dataset with a
single-GPU micro-batch of 8 and gradient accumulation of 8 (64 sequences or
131,072 tokens per optimizer step) for three epochs. The smoke launcher remains
a two-step, 512-token pipeline check.

For a similarly sized dense baseline, build and train the 702M-parameter Qwen3
variant. It uses 16 layers, width 1024, and a 3072-wide dense MLP while reusing
the Modujo tokenizer:

```bash
python pretrain/scripts/prepare_dense_model.py
bash pretrain/scripts/dense_smoke_train.sh
bash pretrain/scripts/dense_train.sh
```

The dense checkpoint defaults to `/workspace/models/modujo-dense-700m` and is
also kept outside Git.

The current data tranche is a 50/50 document mixture of FineWeb-Edu English and
FineWeb 2 Chinese. Dataset revisions and source files are pinned in
`../sources.json`. Generated data and checkpoints remain outside the repository.

The RTX PRO 6000 profile uses fused `grouped_mm` experts. It cuts the measured
2x2048-token micro-step from roughly 27 seconds to 12.76 seconds. During GA64,
SM utilization remains around 24-27% with brief optimizer bursts near 98%, so
further improvement requires a Megatron/kernel backend change. Peak allocated
memory is 93.41 GiB.
The launcher also enables expandable CUDA allocator segments to reduce
fragmentation near this memory limit.
