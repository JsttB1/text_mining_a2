"""Build, verify, and save the complete fold1 Hugging Face conversion."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from archaeo_ner.data import LABEL_NAMES, read_iob
from archaeo_ner.preprocessing import (
    CHECKPOINT, REVISION, MAX_LENGTH, IGNORE_INDEX, build_datasets, tokenize_datasets,
    model_inputs, restore_word_sequences,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-files-only", action="store_true",
                        help="Use the cached tokenizer without network access")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    records = {}
    for split, name in (("train", "train.txt"), ("validation", "val.txt"), ("test", "test.txt")):
        path = ROOT / "data/raw/fold1" / name
        content = path.read_bytes()
        expected = manifest["files"][name]
        if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        records[split] = read_iob(path)
    raw = build_datasets(records)
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT, revision=REVISION, use_fast=True, local_files_only=args.local_files_only,
    )
    tokenized = tokenize_datasets(raw, tokenizer)
    stats = {}
    for split in raw:
        lengths, word_count, ignored = [], 0, 0
        word_labels = []
        for aligned in tokenized[split]:
            retained = [label for label in aligned["labels"] if label != IGNORE_INDEX]
            word_labels.append(retained)
            if len(aligned["input_ids"]) != len(aligned["labels"]):
                raise ValueError(f"{split}/{aligned['source_id']}: input/label lengths differ")
            lengths.append(len(aligned["input_ids"]))
            word_count += len(retained)
            ignored += aligned["labels"].count(IGNORE_INDEX)
        restored_labels = restore_word_sequences(raw[split], tokenized[split], word_labels)
        if restored_labels != list(raw[split]["ner_tags"]):
            raise ValueError(f"{split}: word-label round trip failed")
        stats[split] = {
            "source_sequences": len(raw[split]), "model_chunks": len(lengths),
            "words_preserved": word_count,
            "input_tokens_including_special": sum(lengths),
            "max_input_tokens": max(lengths), "ignored_positions": ignored,
            "continuation_subwords": ignored - 2 * len(lengths),
            "truncated_sequences": 0, "dropped_words": 0,
        }

    # A real dynamically padded batch must mask padding in the labels as well.
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=IGNORE_INDEX)
    training_inputs = model_inputs(tokenized["train"])
    batch = collator([training_inputs[i] for i in range(4)])
    if batch["labels"].shape != batch["input_ids"].shape:
        raise ValueError("Collated input/label shapes differ")
    if not (batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all():
        raise ValueError("Padding was not masked")

    destination = ROOT / "data/processed/fold1"
    raw.save_to_disk(str(destination / "hf_words"))
    tokenized.save_to_disk(str(destination / "hf_tokenized"))
    tokenizer.save_pretrained(destination / "tokenizer")
    # Check the actual saved artifacts, including label schema and every row.
    for folder, original in (("hf_words", raw), ("hf_tokenized", tokenized)):
        restored = load_from_disk(str(destination / folder))
        for split in original:
            if restored[split].features != original[split].features or restored[split].to_dict() != original[split].to_dict():
                raise ValueError(f"Saved {folder}/{split} failed round-trip verification")

    config = {
        "checkpoint": CHECKPOINT, "revision": REVISION,
        "max_length": MAX_LENGTH, "truncation": False,
        "overflow_policy": "nonoverlapping_whole_word_chunks_with_source_offsets",
        "label_strategy": "first_subword_only", "ignore_index": IGNORE_INDEX,
        "padding": "dynamic_per_batch", "label_names": list(LABEL_NAMES),
        "source_manifest": manifest,
        "versions": {package: version(package) for package in ("transformers", "datasets", "tokenizers", "torch")},
        "split_statistics": stats,
        "saved_dataset_round_trip_verified": True,
        "dynamic_padding_verified": True,
    }
    (ROOT / "reports/preprocessing_config.json").write_text(json.dumps(config, indent=2) + "\n")
    lines = [
        "# Tokenization and alignment audit", "",
        f"Checkpoint: `{CHECKPOINT}`. Pinned revision: `{REVISION}`.", "",
        "Every original word has exactly one supervised first subword. Continuation",
        "subwords and special tokens have label `-100`. Dynamic batch padding also",
        "uses `-100`. Gold B-/I- labels are preserved at word level.", "",
        "No sequences were truncated and no words were dropped. Inputs include",
        "[CLS]/[SEP] and fit the 512-token limit. One validation sentence originally",
        "has 544 tokens; it is split into two nonoverlapping whole-word chunks.",
        "Source IDs and word offsets allow reconstruction before entity scoring.",
        "An entity spanning a chunk boundary retains its original B-/I- labels.", "",
        "| Split | Source sentences | Model chunks | Words preserved | Input tokens | Max length | Continuation subwords |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in stats.items():
        values = [row[key] for key in ("source_sequences", "model_chunks", "words_preserved", "input_tokens_including_special", "max_input_tokens", "continuation_subwords")]
        lines.append(f"| {split} | " + " | ".join(map(str, values)) + " |")
    lines += ["", "## Alignment examples", "",
              "Examples below come from training data. `IGNORE` denotes label `-100`."]
    selected = []
    for row in raw["train"]:
        enc = tokenizer(row["tokens"], is_split_into_words=True)
        ids = enc.word_ids()
        has_subwords = len([w for w in ids if w is not None]) > len(row["tokens"])
        has_inside = any(LABEL_NAMES[label].startswith("I-") for label in row["ner_tags"])
        if has_subwords and has_inside and len(enc["input_ids"]) <= 45:
            selected.append((row, enc))
        if len(selected) == 3:
            break
    for row, enc in selected:
        aligned = next(chunk["labels"] for chunk in tokenized["train"] if chunk["source_id"] == row["id"])
        lines += ["", f"### Training sentence {row['id']}", "", "```text",
                  "Subword | Word index | Source word | Label"]
        for piece, word_id, label in zip(enc.tokens(), enc.word_ids(), aligned, strict=True):
            word = "-" if word_id is None else row["tokens"][word_id]
            name = "IGNORE" if label == IGNORE_INDEX else LABEL_NAMES[label]
            lines.append(f"{piece} | {word_id} | {word} | {name}")
        lines.append("```")
    lines += ["", "## Verification and next step", "",
              "All word labels were recovered in their original order on every split.",
              "Both saved DatasetDicts were reloaded and compared in full, including",
              "their features. A real dynamically padded batch passed the masking check.", "",
              "The word dataset has Sequence(ClassLabel) NER features; the tokenized",
              "dataset also records source_id, word_start, and word_end for each chunk.",
              "Remove these metadata columns with model_inputs() before collation.",
              "Use the saved tokenizer with DataCollatorForTokenClassification during",
              "training. For evaluation, retain predictions only where labels != -100",
              "and call restore_word_sequences() to reconstruct original sentences",
              "before computing entity scores. Chunking reduces context for the one",
              "long validation sentence; it does not change or remove gold labels.", "",
              "Next: load the matching pretrained model and implement baseline training.",
              "No model weights have been downloaded or trained in this step.", "",
              "## References", "",
              "- [Assignment-linked course: checkpoint and dataset structure](https://huggingface.co/learn/llm-course/chapter7/2)",
              "- [Token classification guide: first-subword labels and collator](https://huggingface.co/docs/transformers/tasks/token_classification)", ""]
    (ROOT / "reports/tokenization_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("Saved and verified Hugging Face datasets, tokenizer, configuration, and alignment audit.")


if __name__ == "__main__":
    main()
