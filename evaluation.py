from collections import Counter
import numpy as np
from seqeval.scheme import Entities, IOB2
from data_loader import ENTITY_TYPES, LABEL_NAMES

def calc_prf(tp, fp, fn):
    """Calculate Precision, Recall, and F1 from TP, FP, FN counts."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
    }


def evaluate_predictions(gold_sequences, pred_sequences):
    """Compute token-level, entity-level, and micro/macro metrics."""
    # Token-level counts (B- and I- labels)
    correct_tokens, total_gold_tokens = Counter(), Counter()
    total_pred_tokens = Counter()

    for gold_seq, pred_seq in zip(gold_sequences, pred_sequences):
        for g, p in zip(gold_seq, pred_seq):
            total_gold_tokens[g] += 1
            total_pred_tokens[p] += 1
            if g == p:
                correct_tokens[g] += 1

    token_scores = {}
    for label in LABEL_NAMES:
        if label == "O":
            continue
        tp = correct_tokens[label]
        fp = total_pred_tokens[label] - tp
        fn = total_gold_tokens[label] - tp
        token_scores[label] = calc_prf(tp, fp, fn)

    total_words = sum(total_gold_tokens.values())
    word_accuracy = sum(correct_tokens.values()) / total_words if total_words > 0 else 0.0

    # Entity-level scores using strict IOB2 matching (seqeval)
    gold_entities = Entities(gold_sequences, IOB2)
    pred_entities = Entities(pred_sequences, IOB2)

    entity_scores = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for entity in ENTITY_TYPES:
        exp = gold_entities.filter(entity)
        act = pred_entities.filter(entity)
        tp = len(exp & act)
        fp = len(act - exp)
        fn = len(exp - act)

        entity_scores[entity] = calc_prf(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Macro and Micro scores over all entities
    micro_scores = calc_prf(total_tp, total_fp, total_fn)
    macro_p = sum(entity_scores[e]["precision"] for e in ENTITY_TYPES) / len(ENTITY_TYPES)
    macro_r = sum(entity_scores[e]["recall"] for e in ENTITY_TYPES) / len(ENTITY_TYPES)
    macro_f1 = sum(entity_scores[e]["f1"] for e in ENTITY_TYPES) / len(ENTITY_TYPES)

    return {
        "token_labels": token_scores,
        "entities": entity_scores,
        "entity_micro": micro_scores,
        "entity_macro": {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1,
        },
        "entity_macro_f1": macro_f1,
        "word_accuracy": word_accuracy,
    }


def postprocess_predictions(predictions, labels):
    """Filter out -100 subword tokens and convert IDs to label names."""
    if hasattr(predictions, "ndim") and predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        preds = []
        labs = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                preds.append(LABEL_NAMES[p])
                labs.append(LABEL_NAMES[l])
        true_predictions.append(preds)
        true_labels.append(labs)

    return true_labels, true_predictions


def build_compute_metrics_fn():
    """Create compute_metrics callback for Hugging Face Trainer."""
    def compute_metrics(p):
        predictions, labels = p
        true_labs, true_preds = postprocess_predictions(predictions, labels)
        metrics = evaluate_predictions(true_labs, true_preds)
        return {
            "entity_micro_f1": metrics["entity_micro"]["f1"],
            "entity_macro_f1": metrics["entity_macro_f1"],
            "word_accuracy": metrics["word_accuracy"],
        }

    return compute_metrics


def format_results_markdown(metrics, title="Test Set Results"):
    lines = [
        f"# {title}",
        "",
        "## 1. Word-level B- and I- Tokens",
        "",
        "| Label | Precision | Recall | F1 Score | Support | Predicted |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in metrics["token_labels"].items():
        lines.append(
            f"| {label} | {row['precision']:.2%} | {row['recall']:.2%} | "
            f"{row['f1']:.2%} | {row['support']} | {row['predicted']} |"
        )

    lines += [
        "",
        "## 2. Full Exact Entities (per Entity Type)",
        "",
        "| Entity Type | Precision | Recall | F1 Score | Support | Predicted |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entity, row in metrics["entities"].items():
        lines.append(
            f"| {entity} | {row['precision']:.2%} | {row['recall']:.2%} | "
            f"{row['f1']:.2%} | {row['support']} | {row['predicted']} |"
        )

    # Summary rows for Entity Table
    micro = metrics["entity_micro"]
    macro = metrics.get("entity_macro", {"precision": 0.0, "recall": 0.0, "f1": metrics["entity_macro_f1"]})
    lines.append(
        f"| **Micro Average** | **{micro['precision']:.2%}** | **{micro['recall']:.2%}** | "
        f"**{micro['f1']:.2%}** | {micro['support']} | {micro['predicted']} |"
    )
    lines.append(
        f"| **Macro Average** | **{macro['precision']:.2%}** | **{macro['recall']:.2%}** | "
        f"**{macro['f1']:.2%}** | {micro['support']} | {micro['predicted']} |"
    )

    lines += [
        "",
        "## 3. Overall Entity Summary",
        "",
        f"- **Entity Micro F1**: {micro['f1']:.2%} (P: {micro['precision']:.2%}, R: {micro['recall']:.2%})",
        f"- **Entity Macro F1**: {macro['f1']:.2%} (P: {macro['precision']:.2%}, R: {macro['recall']:.2%})",
        f"- **Overall Word Accuracy**: {metrics['word_accuracy']:.2%}",
        "",
    ]
    return "\n".join(lines)
