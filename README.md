# ModujoMoe

This directory adapts the text-only `Qwen4ExpForCausalLM` checkpoint to
ModelScope ms-swift and records reproducible pretraining inputs.

The initial bilingual mixture uses FineWeb-Edu (`sample-10BT`, English,
ODC-By-1.0) and FineWeb 2 (`cmn_Hani`, Chinese, ODC-By-1.0). The checkpoint is
randomly initialized, so training must use `swift pt --tuner_type full`.

Run `python prepare_smoke_data.py`, then `bash smoke_train.sh` for the two-step
forward/backward validation. A long run should only be launched after recording
the observed tokens/second and peak VRAM from this validation.

The first real tranche is produced with `python prepare_train_data.py`. Run it
through `python pack_documents.py` to create near-full 2048-token samples, then
use `bash train.sh`. The configured effective batch is 64 sequences (about 131k
tokens per optimizer update) and the run saves every 10 optimizer steps.
