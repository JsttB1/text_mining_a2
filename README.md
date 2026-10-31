# Text Mining Assignment 2: archaeological NER

Fine-tune a Hugging Face token classifier on the six archaeological entity types
in Archaeo-NER-data-English, using only the supplied **fold1** split.

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
PyTorch can use CPU, Apple MPS, or CUDA where supported. Training configuration
and accelerator choice will be implemented next.

## Layout

- `src/archaeo_ner/`: future conversion, alignment, training, and evaluation code.
- `scripts/download_data.py`: reproducible download of the original fold1 files.
- `data/raw/fold1/`: local `train.txt`, `val.txt`, `test.txt` (ignored by Git).
- `data/manifest.json`: source revision, URLs, byte sizes, and SHA-256 checksums.
- `reports/`: report source and tables when ready.
- `outputs/`: future generated metrics and model checkpoints (ignored by Git).

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

1. Parse IOB sentences into word lists and label lists; inspect split statistics.
2. Build Hugging Face datasets and align word labels with subword tokens.
3. Fine-tune a baseline using the assignment's default hyperparameter requirement.
   Record the exact chosen defaults and seed. Use validation for development;
   reserve test for final reporting.
4. Evaluate B/I labels at word level and exact entity spans per entity type.
5. Compute entity-level macro/micro F1, export tables, and discuss errors.

Reference: [Hugging Face token classification tutorial](https://huggingface.co/learn/llm-course/chapter7/2).

## Submission

Due November 16, 2026. Submit the PDF report and Python code separately.
The report is limited to three pages; reflection may be on a fourth page.
Write the report in Overleaf using the university account and document work
division and AI assistance. The assignment PDF remains local and untracked.
