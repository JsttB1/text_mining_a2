"""Word-label and exact-span IOB2 metrics, including all six entity types."""

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from seqeval.scheme import Entities, IOB2

from archaeo_ner.data import ENTITY_TYPES, LABEL_NAMES
from archaeo_ner.preprocessing import restore_word_sequences


def scores(tp, fp, fn):
    """Use zero for undefined precision/recall/F1 and retain their denominators."""
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "support": tp + fn, "predicted": tp + fp,
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_sequences(gold, predicted):
    if len(gold) != len(predicted) or any(len(g) != len(p) for g, p in zip(gold, predicted)):
        raise ValueError("Gold/prediction sentence and word counts must match")
    if any(tag not in LABEL_NAMES for seq in gold + predicted for tag in seq):
        raise ValueError("Unknown NER label")
    gold_count, pred_count, correct = Counter(), Counter(), Counter()
    for gold_seq, pred_seq in zip(gold, predicted):
        gold_count.update(gold_seq)
        pred_count.update(pred_seq)
        correct.update(g for g, p in zip(gold_seq, pred_seq) if g == p)
    token_scores = {
        label: scores(correct[label], pred_count[label] - correct[label],
                      gold_count[label] - correct[label])
        for label in LABEL_NAMES if label != "O"
    }
    # seqeval's strict IOB2 extractor does not repair orphan I-tags into B-tags.
    gold_entities, pred_entities = Entities(gold, IOB2), Entities(predicted, IOB2)
    entity_scores = {}
    for entity in ENTITY_TYPES:
        expected, actual = gold_entities.filter(entity), pred_entities.filter(entity)
        entity_scores[entity] = scores(len(expected & actual), len(actual - expected),
                                       len(expected - actual))
    micro = scores(*(sum(row[key] for row in entity_scores.values()) for key in ("tp", "fp", "fn")))
    macro_f1 = sum(row["f1"] for row in entity_scores.values()) / len(ENTITY_TYPES)
    words = sum(gold_count.values())
    return {
        "token_labels": token_scores, "entities": entity_scores,
        "entity_micro": micro, "entity_macro_f1": macro_f1,
        "word_accuracy": sum(correct.values()) / words if words else 0.0,
        "sentences": len(gold), "words": words,
        "policy": "strict IOB2 exact spans and type; orphan I-tags ignored as entities; zero_division=0; macro over all six types; O excluded from entity averages",
    }


def decode_predictions(prediction, raw, tokenized):
    logits, label_ids = prediction.predictions, prediction.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_ids = np.argmax(logits, axis=-1)
    chunk_predictions, chunk_gold = [], []
    for predicted, gold in zip(predicted_ids, label_ids, strict=True):
        mask = gold != -100
        chunk_predictions.append([LABEL_NAMES[int(i)] for i in predicted[mask]])
        chunk_gold.append([LABEL_NAMES[int(i)] for i in gold[mask]])
    gold = restore_word_sequences(raw, tokenized, chunk_gold)
    predicted = restore_word_sequences(raw, tokenized, chunk_predictions)
    expected = [[LABEL_NAMES[i] for i in row["ner_tags"]] for row in raw]
    if gold != expected:
        raise ValueError("Evaluation gold labels differ from the original word dataset")
    return gold, predicted


def trainer_metrics(raw, tokenized):
    def compute(prediction):
        gold, predicted = decode_predictions(prediction, raw, tokenized)
        result = evaluate_sequences(gold, predicted)
        return {
            "entity_precision": result["entity_micro"]["precision"],
            "entity_recall": result["entity_micro"]["recall"],
            "entity_micro_f1": result["entity_micro"]["f1"],
            "entity_macro_f1": result["entity_macro_f1"],
            "word_accuracy": result["word_accuracy"],
        }
    return compute


def export_results(split, raw, gold, predicted, output_dir, report_dir):
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_sequences(gold, predicted)
    (report_dir / f"{split}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, expected, actual in zip(raw, gold, predicted, strict=True):
            stream.write(json.dumps({"id": row["id"], "tokens": row["tokens"],
                                     "gold": expected, "predicted": actual}, ensure_ascii=False) + "\n")
    for kind in ("token_labels", "entities"):
        with (report_dir / f"{split}_{kind}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["label", *next(iter(result[kind].values()))],
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows({"label": label, **row} for label, row in result[kind].items())
    lines = [f"# {split.title()} results", "", result["policy"], "",
             "Scores below are percentages. Support counts are gold words/entities."]
    for kind, heading in (("token_labels", "Word-level B-/I-labels"), ("entities", "Exact full entities")):
        lines += ["", f"## {heading}", "", "| Label | Precision | Recall | F1 | Support | Predicted |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for label, row in result[kind].items():
            lines.append(f"| {label} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} | {row['support']} | {row['predicted']} |")
    lines += ["", f"Entity micro F1: **{result['entity_micro']['f1']:.2%}**.",
              f"Entity macro F1 (six types): **{result['entity_macro_f1']:.2%}**.",
              f"Word accuracy (including O): {result['word_accuracy']:.2%}.", "",
              "Macro F1 is the unweighted mean of the six per-type entity F1 scores.",
              "Micro F1 = 2TP / (2TP + FP + FN), pooling counts over entity types.",
              "Undefined precision/recall/F1 are recorded as zero. In particular,",
              "a label with zero gold support has undefined recall, represented as zero.", "",
              "References: [averaging definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html),",
              "[strict IOB2 scoring](https://github.com/chakki-works/seqeval).", ""]
    (report_dir / f"{split}_results.md").write_text("\n".join(lines))
    return result
