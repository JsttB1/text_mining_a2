# Text Mining - Assignment 2: Sequence Labelling (Archaeo-NER)

This project fine-tunes a BERT Transformer (`bert-base-cased`) for Named Entity Recognition (NER) on archaeological texts using the **fold 1** split of [Archaeo-NER-data-English](https://github.com/alexbrandsen/Archaeo-NER-data-English).

The model predicts six archaeological entity types using strict IOB2 tagging:
- `ART`: Artefact
- `CON`: Context / Feature
- `LOC`: Location
- `MAT`: Material
- `PER`: Person
- `SPE`: Species

---

## 1. Project Structure

```text
.
├── data/
│   └── raw/fold1/              # Raw IOB files (train.txt, val.txt, test.txt)
├── results/                    # Saved evaluation tables & metrics for the report
│   ├── test_results_summary.md # Results tables (tokens, entities, macro/micro F1)
│   ├── data_audit.md           # Dataset statistics and token/entity distributions
│   └── test_metrics.json       # Raw numerical metrics
├── download_data.py            # Simple script to download fold 1 files
├── data_loader.py              # IOB file parser and dataset loader (Task 1 & 2)
├── tokenization.py             # Subword tokenization and label alignment (Task 2)
├── evaluation.py               # Token, entity, and micro/macro metrics (Tasks 4, 5, 6)
├── trainer.py                  # Hugging Face Trainer setup and execution (Task 3)
├── run_ner_pipeline.py         # Main entrypoint: coordinates the pipeline
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 2. Pipeline Modules Overview

The codebase is split into modular components that handle one responsibility each:

1. **[`data_loader.py`](file:///mnt/c/Users/nikol/OneDrive%20-%20Universiteit%20Leiden/semester_3/Text_Mining/assignments/a2/text_mining_a2/data_loader.py)**:
   - Reads 3-column IOB text rows (`word`, `pos`, `tag`), groups sentences by blank lines, and maps tags to integer IDs.
   - Converts the splits into a Hugging Face `DatasetDict`.
2. **[`tokenization.py`](file:///mnt/c/Users/nikol/OneDrive%20-%20Universiteit%20Leiden/semester_3/Text_Mining/assignments/a2/text_mining_a2/tokenization.py)**:
   - Implements the Hugging Face Chapter 7 subword alignment: the first subword of each word gets the word's NER label, while subsequent subwords and special tokens receive `-100`.
3. **[`evaluation.py`](file:///mnt/c/Users/nikol/OneDrive%20-%20Universiteit%20Leiden/semester_3/Text_Mining/assignments/a2/text_mining_a2/evaluation.py)**:
   - Computes Precision, Recall, and F1 for all `B-` and `I-` token labels.
   - Computes exact entity matches per entity type using strict IOB2 matching with `seqeval`.
   - Computes overall Entity Micro F1 and Macro F1 scores.
4. **[`trainer.py`](file:///mnt/c/Users/nikol/OneDrive%20-%20Universiteit%20Leiden/semester_3/Text_Mining/assignments/a2/text_mining_a2/trainer.py)**:
   - Configures `AutoModelForTokenClassification` and `TrainingArguments` (AdamW, 3 epochs, lr=2e-5, weight decay=0.01).
   - Trains using `Trainer`, saves the model checkpoint, runs test set predictions, and formats results.
5. **[`run_ner_pipeline.py`](file:///mnt/c/Users/nikol/OneDrive%20-%20Universiteit%20Leiden/semester_3/Text_Mining/assignments/a2/text_mining_a2/run_ner_pipeline.py)**:
   - Clean top-level coordinator. It contains no low-level logic—it simply parses CLI arguments and calls each module in sequence.

---

## 3. Installation & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data (if not already downloaded):**
   ```bash
   python download_data.py
   ```

3. **Inspect dataset statistics:**
   ```bash
   python run_ner_pipeline.py --data_stats
   ```

4. **Run the full training & evaluation pipeline:**
   ```bash
   python run_ner_pipeline.py --epochs 3 --batch_size 8 --learning_rate 2e-5
   ```

5. **Evaluate an existing model checkpoint without re-training:**
   ```bash
   python run_ner_pipeline.py --eval_only --model_path outputs/baseline/final_model
   ```

---

## 4. Key Experimental Results (Test Set)

- **Overall Entity Micro F1:** **66.06%**
- **Overall Entity Macro F1:** **52.16%**
- **Token Accuracy:** **96.44%**

| Entity Type | Precision | Recall | F1 Score | Support |
| :--- | ---: | ---: | ---: | ---: |
| **PER** (Person) | 77.58% | 90.46% | **83.52%** | 283 |
| **LOC** (Location) | 68.82% | 81.25% | **74.52%** | 144 |
| **ART** (Artefact) | 51.40% | 65.48% | **57.59%** | 168 |
| **CON** (Context) | 41.58% | 57.14% | **48.14%** | 147 |
| **MAT** (Material) | 40.00% | 20.34% | **26.97%** | 59 |
| **SPE** (Species) | 14.29% | 50.00% | **22.22%** | 2 |

Detailed per-token and per-entity tables are saved in [`results/test_results.md`](file:///mnt/c/Users/nikol/OneDrive%20-%20Universiteit%20Leiden/semester_3/Text_Mining/assignments/a2/text_mining_a2/results/test_results.md).
