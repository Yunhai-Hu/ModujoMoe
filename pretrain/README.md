# Pretraining

This stage turns the randomly initialized Modujo checkpoint into a base model.

1. Run `python data/prepare_smoke_data.py` and `bash scripts/smoke_train.sh`.
2. Run `python data/prepare_train_data.py`.
3. Run `python data/pack_documents.py`.
4. Run `bash scripts/train.sh`.

The current data tranche is a 50/50 document mixture of FineWeb-Edu English and
FineWeb 2 Chinese. Dataset revisions and source files are pinned in
`../sources.json`. Generated data and checkpoints remain outside the repository.
