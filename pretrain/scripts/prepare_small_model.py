#!/usr/bin/env python3
"""Build a reproducible sub-1B Modujo checkpoint for pipeline tests."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed


DEFAULT_SOURCE = "Alexhu1999/Modujo-9B-A1B"
DEFAULT_OUTPUT = "/workspace/models/modujo-723m"
LAYER_PATTERN = ("linear_attention",) * 3 + ("qwen_sparse_attention",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_config(source: str, seed: int):
    config = AutoConfig.from_pretrained(source)
    config.hidden_size = 768
    config.num_hidden_layers = 16
    config.num_attention_heads = 12
    config.num_key_value_heads = 2
    config.head_dim = 64
    config.linear_key_head_dim = 64
    config.linear_value_head_dim = 64
    config.linear_num_key_heads = 6
    config.linear_num_value_heads = 12
    config.num_experts = 8
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 512
    config.shared_expert_intermediate_size = 512
    config.ple_embed_dim = 768
    config.hc_lowrank = 96
    config.heads_per_ngram = 6
    config.layer_types = list((LAYER_PATTERN * 4)[: config.num_hidden_layers])
    config.seed = seed
    config.dtype = "bfloat16"
    return config


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output}")

    set_seed(args.seed)
    config = make_config(args.source, args.seed)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= 1_000_000_000:
        raise RuntimeError(f"Small model has {parameter_count:,} parameters (expected <1B)")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    tokenizer.save_pretrained(args.output)
    summary = {
        "source_model": args.source,
        "seed": args.seed,
        "total_parameters": parameter_count,
        "total_parameters_billions": parameter_count / 1_000_000_000,
        "dtype": "bfloat16",
    }
    (args.output / "parameter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {parameter_count:,}-parameter model to {args.output}")


if __name__ == "__main__":
    main()
