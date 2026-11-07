# Text Mining Assignment 2: archaeological NER

Fine-tune a Hugging Face token classifier on the six archaeological entity types
in Archaeo-NER-data-English, using only the supplied **fold1** split.

See [CHECKLIST.md](CHECKLIST.md) for shared progress, task owners, assignment
requirements, and submission checks. The completed data preparation step is
summarized in [reports/data_audit.md](reports/data_audit.md).
Subword alignment and length checks are documented in
[reports/tokenization_audit.md](reports/tokenization_audit.md).
The completed baseline achieved **66.06% test entity micro F1** and **52.16%
macro F1**. See the [run summary](reports/baseline/summary.md) and
[full test tables](reports/baseline/test_results.md).

## Setup

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python scripts/download_data.py
```

`requirements.txt` lists direct dependencies. `requirements-lock.txt` records
the verified Python 3.12 macOS arm64 environment; on a matching platform use
`python -m pip install -r requirements-lock.txt` for the exact versions.
PyTorch selects an available accelerator (Apple MPS on this run) or falls back to
CPU. The selected device and complete training configuration are recorded per run.

## Layout

- `src/archaeo_ner/`: IOB parsing, label alignment, sentence reconstruction, and evaluation.
- `scripts/download_data.py`: reproducible download of the original fold1 files.
- `scripts/prepare_data.py`: checksum verification, word-level JSONL conversion, and data audit.
- `scripts/tokenize_data.py`: typed Hugging Face datasets, tokenization, alignment, and verification.
- `scripts/train.py`: baseline fine-tuning, validation, final test predictions, and metrics export.
- `data/raw/fold1/`: local `train.txt`, `val.txt`, `test.txt` (ignored by Git).
- `data/manifest.json`: source revision, URLs, byte sizes, and SHA-256 checksums.
- `reports/`: data/alignment audits and baseline result tables; the final report is pending.
- `outputs/`: local model checkpoints, prediction files, and training state (ignored by Git).

The dataset is from [Archaeo-NER-data-English](https://github.com/alexbrandsen/Archaeo-NER-data-English).
The download script pins its source revision; it does not reshuffle, clean, or
re-split the data. Consult the upstream repository for dataset attribution and terms.

Each nonempty row has **three columns: word, POS tag, NER label**; blank lines
separate sequences. The NER converter should read the first and last columns.

| Split | Sequences | Word tokens |
| --- | ---: | ---: |
| Train | 1,992 | 53,178 |
| Validation | 850 | 17,546 |
| Test | 864 | 17,537 |

Entity codes are `ART`, `CON`, `LOC`, `MAT`, `PER`, and `SPE`.
The training label set has 13 labels (B/I for each type plus O). Test has no
`I-SPE` examples, so evaluation must explicitly handle labels with zero support.

## Implementation plan

The data preparation, training, and evaluation code has been implemented and run.
The report and discussion remain. Reproduce data preparation with:

```bash
python scripts/prepare_data.py
python scripts/tokenize_data.py
python -m unittest discover -s tests -v
```

The tokenizer is downloaded on first use and cached under `.cache/huggingface/`.
Use `python scripts/tokenize_data.py --local-files-only` for subsequent offline runs.
Generated files are under `data/processed/fold1/` (ignored by Git):

- `hf_words/`: the original sentence records with typed NER labels.
- `hf_tokenized/`: model inputs, aligned labels, and chunk reconstruction metadata.
- `tokenizer/`: the saved pinned BERT tokenizer.

Load the datasets with `datasets.load_from_disk()`. The chosen checkpoint is
`google-bert/bert-base-cased`, pinned to revision
`cd5ef92a9fb2f889e972770a36d4ed042daf221e`. The configuration, versions, and
source checksums are in `reports/preprocessing_config.json`.

Only the first subword of each word has a supervised label; other subwords and
special/padding tokens use `-100`. One validation sentence exceeds 512 subword
tokens and is split at a word boundary. All words and original labels are retained.
Call `model_inputs()` from `archaeo_ner.preprocessing` to remove metadata before
using `DataCollatorForTokenClassification`. For metrics, select predictions where
labels are not `-100`, then call `restore_word_sequences()` with the original
word dataset, tokenized dataset, and per-chunk word predictions. This restores
sentence boundaries before entity scoring. Add `src/` to `PYTHONPATH` when
importing the package outside the provided scripts.

1. Parse IOB sentences into word lists and label lists; inspect split statistics.
2. Build Hugging Face datasets and align word labels with subword tokens.
3. Fine-tune a baseline using the assignment's default hyperparameter requirement.
   Record the exact chosen defaults and seed. Use validation for development;
   reserve test for final reporting.
4. Evaluate B/I labels at word level and exact entity spans per entity type.
5. Compute entity-level macro/micro F1, export tables, and discuss errors.

Reference: [Hugging Face token classification tutorial](https://huggingface.co/learn/llm-course/chapter7/2).

## Baseline training and evaluation

After data preparation, run:

```bash
python scripts/train.py --smoke-test
python scripts/train.py
```

The first command performs two real optimizer steps on 16 training examples to
check model/device compatibility. The second starts again from the pretrained
checkpoint and uses all 1,992 training sentences. Once the model weights have
been downloaded, add `--local-files-only` to run without network access.

We interpret the assignment's default-settings requirement as the baseline in
its linked tutorial: learning rate `2e-5`, weight decay `0.01`, and three epochs.
These differ from bare `TrainingArguments` defaults (`5e-5` and `0.0`). Batch size
is 8, gradient accumulation is 1, the scheduler is linear with no warmup, and the
seed is 42. Standard PyTorch AdamW is selected explicitly for MPS compatibility;
other optimizer settings retain their installed-library defaults. Run metadata
records all arguments and versions. A fixed seed helps reproducibility, but
results may vary across hardware and software versions.

Validation is evaluated after each epoch, with long inputs reconstructed before
entity scoring. The final epoch is used for test evaluation; there is no early
stopping, hyperparameter search, or test-based checkpoint selection. The model
and data stay local; no Hub upload or external experiment tracker is enabled.

Output files:

- `outputs/baseline/final_model/`: trained model and tokenizer.
- `outputs/baseline/checkpoint-*/`: latest resumable training checkpoint.
- `outputs/baseline/{validation,test}_predictions.jsonl`: original words, gold labels, and predictions.
- `reports/baseline/`: run configuration, training history, per-label/per-entity CSV tables, JSON metrics, and readable result summaries.

An existing baseline directory is protected against an accidental fresh run.
Resume interrupted training with `--resume outputs/baseline/checkpoint-N`,
replacing N with the saved checkpoint step. Recompute results from the saved final
model with `python scripts/train.py --evaluate-only --local-files-only`.

Entity metrics require exact span and type matches under strict IOB2. Orphan
I-tags are not repaired into entity starts. Macro F1 averages all six entity-type
F1 scores; micro F1 pools their TP/FP/FN counts. O is excluded from entity averages.
Undefined scores are represented as zero, with support and prediction counts
retained so that absent labels such as test I-SPE are visible. Metrics follow
[seqeval's strict mode](https://github.com/chakki-works/seqeval) and the
[standard averaging definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

## Submission

Due November 16, 2026. Submit the PDF report and Python code separately.
The report is limited to three pages; reflection may be on a fourth page.
Write the report in Overleaf using the university account and document work
division and AI assistance. The assignment PDF remains local and untracked.
