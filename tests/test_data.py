"""Small checks for input boundary and label-handling errors."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from archaeo_ner.data import LABEL_TO_ID, read_iob, split_statistics


class DataTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(content, encoding="utf-8")
            return read_iob(path)

    def test_boundaries_and_final_sentence(self):
        records = self.parse("\nRoman JJ B-PER\nroad NN O\n\n\nYork NNP B-LOC")
        self.assertEqual([row["tokens"] for row in records], [["Roman", "road"], ["York"]])
        self.assertEqual(records[0]["ner_tags"], [LABEL_TO_ID["B-PER"], LABEL_TO_ID["O"]])
        self.assertEqual([row["id"] for row in records], ["0", "1"])

    def test_malformed_row(self):
        with self.assertRaisesRegex(ValueError, r":1: expected"):
            self.parse("word O")

    def test_unknown_label(self):
        with self.assertRaisesRegex(ValueError, "unknown NER label"):
            self.parse("word NN B-UNKNOWN")

    def test_empty_input(self):
        with self.assertRaisesRegex(ValueError, "no sentences"):
            self.parse("\n\n")

    def test_entity_counts_and_invalid_transitions(self):
        records = self.parse("Blue NN B-LOC\nBoar NN I-LOC\nLane NN I-LOC\n\nclay NN I-MAT\n")
        stats = split_statistics(records)
        self.assertEqual(stats["entity_starts"]["LOC"], 1)
        self.assertEqual(stats["label_counts"]["I-LOC"], 2)
        self.assertEqual(stats["invalid_iob2_inside_tags"], 1)
        self.assertEqual(records[1]["ner_tags"], [LABEL_TO_ID["I-MAT"]])


if __name__ == "__main__":
    unittest.main()
