#!/usr/bin/env python3
"""Build a reproducible dense Qwen3 checkpoint comparable to Modujo-723M."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3Config, set_seed


DEFAULT_SOURCE = "Alexhu1999/Modujo-9B-A1B"
DEFAULT_OUTPUT = "/workspace/models/modujo-dense-700m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source checkpoint for tokenizer and token IDs")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_config(source: str, seed: int) -> Qwen3Config:
    source_config = AutoConfig.from_pretrained(source)
    return Qwen3Config(
        vocab_size=source_config.vocab_size,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000},
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=source_config.pad_token_id,
        bos_token_id=source_config.bos_token_id,
        eos_token_id=source_config.eos_token_id,
        dtype="bfloat16",
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output}")

    set_seed(args.seed)
    config = make_config(args.source, args.seed)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 600_000_000 <= parameter_count < 800_000_000:
        raise RuntimeError(f"Dense model has unexpected parameter count: {parameter_count:,}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    tokenizer.save_pretrained(args.output)
    summary = {
        "architecture": "Qwen3ForCausalLM (dense)",
        "source_tokenizer": args.source,
        "seed": args.seed,
        "total_parameters": parameter_count,
        "total_parameters_billions": parameter_count / 1_000_000_000,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "dtype": "bfloat16",
    }
    (args.output / "parameter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {parameter_count:,}-parameter dense model to {args.output}")


if __name__ == "__main__":
    main()
