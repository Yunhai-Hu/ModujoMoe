"""Offline-pack raw documents into near-full 2048-token pretraining samples."""

import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL = "/root/.cache/huggingface/hub/models--Alexhu1999--Modujo-9B-A1B/snapshots/a18b15449fbce9d4e182138704d8d015f2dadfbf"
SOURCE = Path("/workspace/datasets/modujo/train_bilingual_40k.jsonl")
OUTPUT = Path("/workspace/datasets/modujo/train_bilingual_packed_2048.jsonl")
TARGET_TOKENS = 1984  # leaves room for Swift-added boundary tokens


def flush(handle, tokenizer, token_buffer):
    text = tokenizer.decode(token_buffer, skip_special_tokens=False)
    record = {"messages": [{"role": "assistant", "content": text}]}
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    buffer = []
    count = 0
    with SOURCE.open(encoding="utf-8") as source, OUTPUT.open("w", encoding="utf-8") as output:
        for line in source:
            text = json.loads(line)["messages"][0]["content"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            while tokens:
                room = TARGET_TOKENS - len(buffer)
                take = tokens[:room]
                buffer.extend(take)
                tokens = tokens[room:]
                if len(buffer) == TARGET_TOKENS:
                    flush(output, tokenizer, buffer)
                    count += 1
                    buffer = []
                elif tokens or len(buffer) + len(separator) <= TARGET_TOKENS:
                    buffer.extend(separator)
        if buffer:
            flush(output, tokenizer, buffer)
            count += 1
    print(f"Wrote {count} packed samples to {OUTPUT}")


if __name__ == "__main__":
    main()
