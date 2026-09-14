"""Prepare the first reproducible 50/50 English-Chinese training tranche."""

from prepare_smoke_data import DATA_DIR, write_dataset


SOURCES = (
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 20_000),
    ("HuggingFaceFW/fineweb-2", "cmn_Hani", 20_000),
)


if __name__ == "__main__":
    write_dataset(DATA_DIR / "train_bilingual_40k.jsonl", SOURCES)
