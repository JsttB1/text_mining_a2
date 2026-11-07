"""Read the original three-column archaeological IOB files without changing labels."""

from pathlib import Path

ENTITY_TYPES = ("ART", "CON", "LOC", "MAT", "PER", "SPE")
LABEL_NAMES = ("O",) + tuple(
    f"{prefix}-{entity}" for entity in ENTITY_TYPES for prefix in ("B", "I")
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_NAMES)}


def read_iob(path: str | Path) -> list[dict]:
    """Return sentence records with word tokens and integer word-level NER tags.

    Blank lines delimit sentences; POS tags are ignored. Subword alignment is a
    separate step. Invalid rows and unknown labels fail with a source line number.
    """
    path = Path(path)
    records = []
    tokens, tags = [], []

    def finish_sentence():
        if tokens:
            records.append({"id": str(len(records)), "tokens": tokens.copy(),
                            "ner_tags": tags.copy()})
            tokens.clear()
            tags.clear()

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            finish_sentence()
            continue
        columns = line.split()
        if len(columns) != 3:
            raise ValueError(f"{path}:{line_number}: expected word, POS tag, NER label")
        word, _, label = columns
        if label not in LABEL_TO_ID:
            raise ValueError(f"{path}:{line_number}: unknown NER label {label!r}")
        tokens.append(word)
        tags.append(LABEL_TO_ID[label])
    finish_sentence()
    if not records:
        raise ValueError(f"{path}: no sentences found")
    return records


def split_statistics(records: list[dict]) -> dict:
    """Count labels and entity starts; flag non-IOB2 transitions without fixing them."""
    label_counts = dict.fromkeys(LABEL_NAMES, 0)
    invalid_inside_tags = 0
    for record in records:
        previous = "O"
        for tag in record["ner_tags"]:
            label = LABEL_NAMES[tag]
            label_counts[label] += 1
            if label.startswith("I-") and previous not in (
                "B-" + label[2:], "I-" + label[2:]
            ):
                invalid_inside_tags += 1
            previous = label
    lengths = [len(record["tokens"]) for record in records]
    return {
        "sequences": len(records), "word_tokens": sum(lengths),
        "min_words": min(lengths), "max_words": max(lengths),
        "mean_words": round(sum(lengths) / len(lengths), 2),
        "label_counts": label_counts,
        "entity_starts": {entity: label_counts[f"B-{entity}"] for entity in ENTITY_TYPES},
        "invalid_iob2_inside_tags": invalid_inside_tags,
    }
