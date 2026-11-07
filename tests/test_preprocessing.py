"""Offline alignment checks using a tiny real WordPiece tokenizer."""

from pathlib import Path
import sys
import tempfile
import unittest

from transformers import BertTokenizerFast, DataCollatorForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from archaeo_ner.data import LABEL_NAMES, LABEL_TO_ID
from archaeo_ner.preprocessing import (
    build_datasets, model_inputs, restore_word_sequences, tokenize_and_align,
)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        vocab = Path(cls.directory.name) / "vocab.txt"
        vocab.write_text("\n".join([
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "Blue", "Boar", "Lane", "pot", "##tery", ".",
        ]) + "\n")
        cls.tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False,
                                         model_max_length=512)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def batch(self, words, labels):
        return {"id": ["sample"], "tokens": [words], "ner_tags": [labels]}

    def test_first_subword_and_entity_boundaries(self):
        labels = [LABEL_TO_ID[name] for name in ["B-LOC", "I-LOC", "B-ART", "O"]]
        result = tokenize_and_align(self.batch(["Blue", "Lane", "pottery", "."], labels), self.tokenizer)
        self.assertEqual(result["labels"][0], [-100, labels[0], labels[1], labels[2], -100, 0, -100])
        self.assertEqual(result["word_start"], [0])
        self.assertEqual(result["word_end"], [4])

    def test_whole_word_chunks_and_entity_reconstruction(self):
        # A multiword entity crosses the chunk boundary; pottery must stay whole.
        labels = [LABEL_TO_ID[name] for name in ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC"]]
        words = ["Blue", "pottery", "Lane", "Boar", "Lane"]
        batch = self.batch(words, labels)
        result = tokenize_and_align(batch, self.tokenizer, max_length=5)
        self.assertEqual(result["word_start"], [0, 2])
        self.assertEqual(result["word_end"], [2, 5])
        self.assertTrue(all(len(ids) <= 5 for ids in result["input_ids"]))
        rows = [dict(zip(result, values)) for values in zip(*result.values())]
        raw = [{"id": "sample", "tokens": words, "ner_tags": labels}]
        values = [[tag for tag in row["labels"] if tag != -100] for row in rows]
        self.assertEqual(restore_word_sequences(raw, rows, values), [labels])
        rows[1]["word_start"] = 1
        with self.assertRaisesRegex(ValueError, "Chunk offsets"):
            restore_word_sequences(raw, rows, values)

    def test_schema_and_dynamic_padding(self):
        records = [{"id": "0", "tokens": ["Blue"], "ner_tags": [0]},
                   {"id": "1", "tokens": ["pottery", "Lane"], "ner_tags": [1, 0]}]
        raw = build_datasets({"train": records})
        self.assertEqual(raw["train"].features["ner_tags"].feature.names, list(LABEL_NAMES))
        aligned = raw["train"].map(
            tokenize_and_align, batched=True, fn_kwargs={"tokenizer": self.tokenizer},
            remove_columns=raw["train"].column_names,
        )
        model_rows = model_inputs(aligned)
        batch = DataCollatorForTokenClassification(self.tokenizer)(list(model_rows))
        self.assertEqual(tuple(batch["labels"].shape), (2, 5))
        self.assertEqual(batch["labels"][0].tolist(), [-100, 0, -100, -100, -100])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 0, 0])

    def test_missing_word_fails(self):
        with self.assertRaisesRegex(ValueError, "dropped words"):
            tokenize_and_align(self.batch(["Blue", ""], [0, 0]), self.tokenizer)

    def test_word_larger_than_budget_fails(self):
        with self.assertRaisesRegex(ValueError, "single word exceeds"):
            tokenize_and_align(self.batch(["pottery"], [0]), self.tokenizer, max_length=3)

    def test_mismatched_labels_fail(self):
        with self.assertRaisesRegex(ValueError, "equal nonzero lengths"):
            tokenize_and_align(self.batch(["Blue"], []), self.tokenizer)


if __name__ == "__main__":
    unittest.main()
