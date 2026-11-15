from collections import Counter
from pathlib import Path
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

ENTITY_TYPES = ["ART", "CON", "LOC", "MAT", "PER", "SPE"]
LABEL_NAMES = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ("B", "I")]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}

DATASET_FEATURES = Features({
    "id": Value("string"),
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=LABEL_NAMES)),
})


def load_iob_file(filepath: Path | str) -> list[dict]:
    """Parse a 3-column IOB text file (word, pos, tag) delimited by blank lines."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    sentences = []
    tokens, tags = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({
                        "id": str(len(sentences)),
                        "tokens": tokens,
                        "ner_tags": [LABEL_TO_ID[t] for t in tags],
                    })
                    tokens, tags = [], []
                continue

            parts = line.split()
            if len(parts) >= 3:
                word, _, tag = parts[0], parts[1], parts[2]
                if tag in LABEL_TO_ID:
                    tokens.append(word)
                    tags.append(tag)

    if tokens:
        sentences.append({
            "id": str(len(sentences)),
            "tokens": tokens,
            "ner_tags": [LABEL_TO_ID[t] for t in tags],
        })

    return sentences


def load_all_splits(data_dir):
    data_path = Path(data_dir)
    raw_splits = {
        "train": load_iob_file(data_path / "train.txt"),
        "validation": load_iob_file(data_path / "val.txt"),
        "test": load_iob_file(data_path / "test.txt"),
    }

    hf_dataset = DatasetDict({
        split: Dataset.from_list(records, features=DATASET_FEATURES)
        for split, records in raw_splits.items()
    })

    return raw_splits, hf_dataset


def print_dataset_statistics(raw_splits):
    print("=" * 60)
    print("DATASET STATISTICS (Fold 1)")
    print("=" * 60)
    for split, records in raw_splits.items():
        n_sentences = len(records)
        n_words = sum(len(r["tokens"]) for r in records)
        tag_counts = Counter(tag for r in records for tag in r["ner_tags"])
        entity_counts = {e: tag_counts[LABEL_TO_ID[f"B-{e}"]] for e in ENTITY_TYPES}

        print(f"\n[{split.upper()}] {n_sentences:,} sentences | {n_words:,} words")
        for e in ENTITY_TYPES:
            print(f"  {e:4s}: {entity_counts[e]:5d} entities")
    print("=" * 60 + "\n")
