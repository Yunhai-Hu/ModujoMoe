# Pretraining

This stage turns the randomly initialized Modujo checkpoint into a base model.

1. Run `python data/prepare_smoke_data.py` and `bash scripts/smoke_train.sh`.
2. Run `python data/prepare_train_data.py`.
3. Run `python data/pack_documents.py`.
4. Run `bash scripts/train.sh`.

The current data tranche is a 50/50 document mixture of FineWeb-Edu English and
FineWeb 2 Chinese. Dataset revisions and source files are pinned in
`../sources.json`. Generated data and checkpoints remain outside the repository.

The RTX PRO 6000 profile uses fused `grouped_mm` experts. This raises measured
GPU utilization from about 27% to 98% and cuts a 2x2048-token step from roughly
27 seconds to 12.76 seconds. Peak allocated memory is 93.41 GiB.
The launcher also enables expandable CUDA allocator segments to reduce
fragmentation near this memory limit.
