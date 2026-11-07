"""Metric checks for exact entities, zero support, and chunk reconstruction."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from archaeo_ner.data import ENTITY_TYPES, LABEL_NAMES, LABEL_TO_ID
from archaeo_ner.evaluation import decode_predictions, evaluate_sequences


class EvaluationTests(unittest.TestCase):
    def test_exact_partial_wrong_type_missing_extra_and_invalid(self):
        gold = [["B-LOC", "I-LOC", "O"]]
        cases = [
            (["B-LOC", "I-LOC", "O"], (1, 0, 0)),
            (["B-LOC", "O", "O"], (0, 1, 1)),
            (["B-ART", "I-ART", "O"], (0, 1, 1)),
            (["O", "O", "O"], (0, 0, 1)),
            (["B-LOC", "I-LOC", "B-ART"], (1, 1, 0)),
            (["I-LOC", "I-LOC", "O"], (0, 0, 1)),
        ]
        for predicted, counts in cases:
            with self.subTest(predicted=predicted):
                result = evaluate_sequences(gold, [predicted])
                micro = result["entity_micro"]
                self.assertEqual(tuple(micro[k] for k in ("tp", "fp", "fn")), counts)
                self.assertAlmostEqual(micro["f1"], f1_score(gold, [predicted], mode="strict", scheme=IOB2, zero_division=0))

    def test_macro_micro_and_word_scores_against_libraries(self):
        gold = [[f"B-{kind}", f"I-{kind}", "O"] for kind in ENTITY_TYPES]
        gold += [["B-ART", "O", "O"]] * 4
        predicted = [row.copy() for row in gold]
        predicted[1] = ["O", "O", "O"]
        predicted[3] = ["B-ART", "I-ART", "O"]
        result = evaluate_sequences(gold, predicted)
        for average in ("micro", "macro"):
            actual = result["entity_micro"]["f1"] if average == "micro" else result["entity_macro_f1"]
            self.assertAlmostEqual(actual, f1_score(gold, predicted, average=average, mode="strict", scheme=IOB2, zero_division=0))
        expected = precision_recall_fscore_support(
            sum(gold, []), sum(predicted, []), labels=list(LABEL_NAMES[1:]), zero_division=0,
        )
        for i, label in enumerate(LABEL_NAMES[1:]):
            for key, array in zip(("precision", "recall", "f1", "support"), expected):
                self.assertAlmostEqual(result["token_labels"][label][key], array[i])

    def test_zero_support_and_six_type_macro(self):
        result = evaluate_sequences([["B-LOC", "O"]], [["B-LOC", "O"]])
        self.assertEqual(result["entity_micro"]["f1"], 1)
        self.assertAlmostEqual(result["entity_macro_f1"], 1 / 6)
        self.assertEqual(result["token_labels"]["I-SPE"]["support"], 0)
        self.assertEqual(result["token_labels"]["I-SPE"]["recall"], 0)
        empty_entities = evaluate_sequences([["O"]], [["O"]])
        self.assertEqual(empty_entities["entity_micro"]["f1"], 0)

    def test_sentence_boundaries_are_part_of_entity_identity(self):
        result = evaluate_sequences([["B-LOC"], ["O"]], [["O"], ["B-LOC"]])
        self.assertEqual(result["entity_micro"]["tp"], 0)

    def test_chunk_decoding_masks_padding_and_reconstructs_entity(self):
        gold_ids = [LABEL_TO_ID["B-LOC"], LABEL_TO_ID["I-LOC"]]
        labels = np.array([[-100, gold_ids[0], -100, -100], [-100, gold_ids[1], -100, -100]])
        logits = np.zeros((2, 4, len(LABEL_NAMES)))
        logits[:, :, LABEL_TO_ID["B-ART"]] = 1  # Wrong ignored-token predictions must not count.
        for i, label in enumerate(gold_ids):
            logits[i, 1, label] = 10
        raw = [{"id": "0", "tokens": ["Blue", "Lane"], "ner_tags": gold_ids}]
        chunks = [{"source_id": "0", "word_start": 0, "word_end": 1},
                  {"source_id": "0", "word_start": 1, "word_end": 2}]
        gold, predicted = decode_predictions(SimpleNamespace(predictions=logits, label_ids=labels), raw, chunks)
        self.assertEqual(gold, [["B-LOC", "I-LOC"]])
        self.assertEqual(evaluate_sequences(gold, predicted)["entity_micro"]["f1"], 1)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            evaluate_sequences([["O"]], [["O", "O"]])


if __name__ == "__main__":
    unittest.main()
