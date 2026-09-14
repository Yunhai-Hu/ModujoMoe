# ModujoMoe

Pretraining recipes for the randomly initialized, text-only Modujo
`Qwen4ExpForCausalLM` checkpoint using ModelScope ms-swift.

## Files

- `modujo_swift_plugin.py`: registers the text-only Qwen4-Exp architecture.
- `prepare_smoke_data.py` / `smoke_train.sh`: two-step compatibility check.
- `prepare_train_data.py`: streams the reproducible bilingual source tranche.
- `pack_documents.py`: creates near-full 2048-token samples offline.
- `train.sh`: full-parameter MoE pretraining command.
- `sources.json`: pinned dataset revisions and source shard metadata.

Generated datasets, logs, caches, and checkpoints stay outside this repository.

The initial bilingual mixture uses FineWeb-Edu (`sample-10BT`, English,
ODC-By-1.0) and FineWeb 2 (`cmn_Hani`, Chinese, ODC-By-1.0). The checkpoint is
randomly initialized, so training must use `swift pt --tuner_type full`.

## Usage

```bash
python prepare_smoke_data.py
bash smoke_train.sh

python prepare_train_data.py
python pack_documents.py
bash train.sh
```

The configured effective batch is 64 sequences (about 131k tokens per optimizer
update at 2048 tokens) and the run saves every 10 optimizer steps.
