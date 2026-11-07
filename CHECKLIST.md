# Assignment 2: shared checklist

Last updated: 7 November 2026.

**Current status:** Assignment coding tasks 1-6 are complete and verified:
data conversion, baseline training, token/entity evaluation, and macro/micro F1.
Test entity micro F1 is **66.06%**, and macro F1 is **52.16%**. The trained model,
predictions, and result tables are saved. Next: review errors and prepare the
three-page Overleaf report, reflection, and submission.

Source: `Assignment_2_sequence_labelling (1).pdf`, pages 1-3 (provided locally).
Checked boxes indicate work verified in this repository, not assumed completion.
The assignment PDF and raw/processed data are ignored by Git; this checklist can
be shared on its own, and the scripts reproduce the data outputs.

## Assignment rules and team setup

- [x] Read the assignment instructions and record the deliverables here.
- [ ] Confirm the two-person group and Brightspace enrolment. Group changes/self-enrolment remain open until **9 November 2026**, according to the PDF.
- [ ] Agree on owners in the table below and review each other's work.
- [ ] Both members complete/review the Hugging Face token-classification tutorial and week 6/8 prerequisites.
- [ ] Review lecture 1's AI-use rules (the lecture slides have not been checked here).
- [ ] Create the report in **Overleaf using a university account**.

Suggested division only; replace TBD with names before treating this as agreed:

| Work | Suggested lead | Reviewer | Status |
| --- | --- | --- | --- |
| Data conversion and tokenizer alignment | Member A (TBD) | Member B (TBD) | Implemented and verified; student review pending |
| Training and reproducibility | Member A (TBD) | Member B (TBD) | Three epochs complete; model reload verified |
| Evaluation and results tables | Member B (TBD) | Member A (TBD) | Exported and independently verified |
| Data description and discussion | Member B (TBD) | Member A (TBD) | Data audit available |
| Final report, reflection, submission | Both | Both | Not started |

## 0. Environment and project setup

- [x] Create the project structure and Python environment.
- [x] Record dependencies in `requirements.txt` and `requirements-lock.txt`.
- [x] Check installed dependency consistency with `.venv/bin/python -m pip check`.
- [x] Verify the pinned fast tokenizer loads and saves successfully.
- [x] Verify model loading and two real optimizer steps on Apple MPS with finite loss.
- [ ] Have the other member reproduce the setup using `README.md`.

## 1. Download and verify fold1 (assignment task 1) - complete

- [x] Provide `scripts/download_data.py` with a pinned source revision.
- [x] Download `train.txt`, `val.txt`, and `test.txt` from **fold1 only**.
- [x] Record source URLs, byte sizes, and SHA-256 checksums in `data/manifest.json`.
- [x] Verify all local files against the saved manifest.
- [x] Preserve the supplied splits without reshuffling or resplitting.

| Split | Sequences | Word tokens |
| --- | ---: | ---: |
| Train | 1,992 | 53,178 |
| Validation | 850 | 17,546 |
| Test | 864 | 17,537 |

Evidence: `scripts/prepare_data.py`, `reports/data_audit.md`, and
`reports/data_statistics.json`. Checksums confirm agreement with the previously
saved manifest; no fresh upstream comparison was needed for this step.

## 2. Convert data and align labels (assignment task 2) - complete

**Completion confirmed on 7 November 2026:** no implementation items remain for
this step. Rechecked the saved word datasets against the original IOB files and
reconstructed all 88,261 word labels from the saved tokenized chunks. Every model
input fits the 512-token limit. Student review remains separate from this
technical completion status.

- [x] Write our own parser in `src/archaeo_ner/data.py`.
- [x] Read word/POS/NER rows, retain words and NER labels, and preserve blank-line sentence boundaries.
- [x] Reject malformed rows and unknown labels with useful file/line errors.
- [x] Define a consistent mapping for O plus B-/I- labels for ART, CON, LOC, MAT, PER, and SPE.
- [x] Run conversion to `id`, `tokens`, and integer `ner_tags` JSONL records for all three splits.
- [x] Export label mappings and reproducible split/entity/label statistics.
- [x] Check IOB2 transitions: zero invalid I-tags in all three splits.
- [x] Verify parser behavior on sentence boundaries, missing trailing blank lines, malformed input, and invalid I-tag transitions.
- [x] Build a Hugging Face `DatasetDict` with `Sequence(ClassLabel)` NER features from these records.
- [x] Select `google-bert/bert-base-cased` and its matching fast tokenizer; pin the revision in the configuration.
- [x] Align word labels with subword tokens using `word_ids()`.
- [x] Label only each word's first subword; mask continuation subwords and special/padding tokens with `-100`.
- [x] Check input lengths, preserve all words, and implement/test sentence reconstruction for later evaluation.
- [x] Inspect three aligned examples with multiword entities and words split into subwords in `reports/tokenization_audit.md`.
- [x] Run the complete dataset/tokenization conversion and save reproducible configuration.
- [x] Reload and compare every saved row and dataset feature; verify dynamic padding and pass all 11 tests.

All 88,261 original words are preserved. One validation sentence produces 544
subword tokens before splitting, so it is stored as two whole-word chunks within
BERT's 512-token limit. Validation has 850 source sentences and 851 model chunks.
Chunk source IDs and word offsets are saved for reassembling predictions before
entity scoring. Training/test sentence counts are unchanged. Chunking reduces
context for that validation sentence; it does not drop or modify gold labels.

Evidence: [conversion script](scripts/tokenize_data.py),
[alignment implementation](src/archaeo_ner/preprocessing.py),
[alignment examples and audit](reports/tokenization_audit.md), and
[saved configuration](reports/preprocessing_config.json).

**Next actionable task:** review error examples and use `reports/baseline/` to
prepare the report. [Run summary](reports/baseline/summary.md) and
[test tables](reports/baseline/test_results.md) are ready to share.

## 3. Fine-tune and run final test evaluation (assignment task 3) - complete

- [x] Establish which defaults are being used: the linked tutorial baseline (learning rate 2e-5, weight decay 0.01, three epochs), rather than bare library defaults. The PDF does not specify numeric values.
- [x] Record the checkpoint, all hyperparameters, package versions, seed 42, and Apple MPS device in `outputs/baseline/run_config.json`.
- [x] Fine-tune on the supplied training split for three epochs / 747 optimizer steps using those defaults.
- [x] Evaluate validation each epoch; use the final epoch for test without test-driven tuning or checkpoint selection.
- [x] Save the model, training logs, and configuration. Reload the model and reproduce predictions for eight validation sentences.
- [x] Run final test predictions; retain all 17,537 word-level gold labels and predictions across 864 sentences.

## 4. Evaluate token labels and complete entities (assignment tasks 4-5) - complete

- [x] Export precision, recall, and F1 for each of the 12 B-/I- labels across all six types.
- [x] Exclude special tokens, padding, and duplicate subword predictions from word-level scoring.
- [x] Use `restore_word_sequences()` to reassemble chunks before entity scoring, including validation metrics.
- [x] Handle zero-support labels explicitly: test has **zero I-SPE tokens**. Undefined scores are recorded as zero with support/prediction counts shown.
- [x] Implement strict IOB2 entity matching: the full span and entity type must match; orphan I-tags are not repaired.
- [x] Export entity precision, recall, and F1 separately for all six entity types.
- [x] Verify correct, partial, wrong-type, missing, extra, and invalid-IOB entities; compare with seqeval and scikit-learn. All 17 tests pass.
- [x] Save test tables and counts in `reports/baseline/`; save full predictions in `outputs/baseline/`. Verify all exported validation/test metrics independently against seqeval and scikit-learn.

## 5. Compute macro and micro F1 (assignment task 6) - complete

- [x] Look up and cite macro/micro definitions; references are in `README.md` and the result export code.
- [x] Compute entity macro F1 as the unweighted mean of the six entity-type F1 scores; document zero-support handling.
- [x] Compute entity micro F1 from pooled entity true positives, false positives, and false negatives.
- [x] Exclude the O class from entity averages and distinguish entity metrics from token metrics.
- [x] Verify the calculations independently against the per-type counts and include them in the result summaries.

## 6. Write the report

- [ ] **Introduction:** briefly explain archaeological NER and the task.
- [ ] **Data:** describe fold1, split sizes, and the distribution across entity types using `reports/data_audit.md`.
- [ ] **Data:** explain two challenges. Evidence-backed candidates are class imbalance/rare labels and multi-token entity boundaries; connect them to examples.
- [ ] Confirm full entity-code meanings from the dataset documentation before expanding abbreviations in the report.
- [ ] **Results:** include test precision/recall/F1 for B- and I-labels per entity type.
- [ ] **Results:** include test precision/recall/F1 for full entities per entity type.
- [ ] **Results:** include entity macro and micro F1.
- [ ] **Discussion:** explain differences between entity types.
- [ ] **Discussion:** explain differences between B-label, I-label, and full-entity scores.
- [ ] **Discussion:** explain why macro and micro F1 differ.
- [ ] **Reflection:** describe the actual division of work between both members.
- [ ] **Reflection:** disclose the AI assistant, tasks supported, and whether it helped or made errors (maximum one paragraph for the AI-use reflection).
- [ ] Keep all main content, tables, references, and any appendices within **three pages**. Reflection may be on a fourth page.
- [ ] Write original prose; keep reused code clean and concise and credit sources.

## 7. Final review and submission

- [ ] Both members review the implementation, results, discussion, and reflection.
- [ ] Re-run the documented workflow and confirm that the reported tables match saved results.
- [ ] Export and visually review the final Overleaf PDF for readability and page limits.
- [ ] Submit the **PDF report and Python code separately** through Brightspace; do not put the PDF inside a ZIP.
- [ ] Submit **on or before Monday, 16 November 2026**. The PDF does not specify a time; check Brightspace for the exact cutoff.
- [ ] Confirm the uploaded files are readable and retain the submission confirmation.

The five grading categories are worth up to two points each: overall writing and
length; data/statistics/two challenges; token and full-entity results; per-type
and macro/micro results; discussion.

## Work completed on 7 November 2026

Reviewed the assignment PDF and existing project. Verified task 1 against the
manifest, implemented and ran all of task 2, and produced data/alignment audits.
Saved and reloaded the typed datasets and tokenizer, checked complete word-label
reconstruction, and passed 11 parser/alignment tests. Then implemented training
and evaluation, passed all 17 tests, and completed the three-epoch baseline on
Apple MPS. Exported and independently verified validation/test metrics. Reloaded
the saved final model and reproduced eight validation predictions. No final
Overleaf report has been written or submitted.

Reproduce the workflow on a fresh checkout from the project root:

```bash
source .venv/bin/activate
python scripts/prepare_data.py
python scripts/tokenize_data.py
python -m unittest discover -s tests -v
python scripts/train.py --smoke-test
python scripts/train.py
```

On a new checkout, follow the setup instructions in `README.md` and run
`python scripts/download_data.py` first.
After the tokenizer has been cached, use `python scripts/tokenize_data.py
--local-files-only` to reproduce the conversion offline.
The existing baseline is protected from accidental overwriting. Use
`python scripts/train.py --evaluate-only --local-files-only` to recompute its
metrics from the saved final model. Model files and predictions are local and
ignored by Git; the scripts and `reports/baseline/` tables can be shared.

AI assistance log for the eventual reflection: Codex assisted with reading the
assignment, checking repository status, writing this checklist, implementing and
running the parser/data audit, building the Hugging Face datasets, aligning labels,
handling long inputs without word loss, implementing/training the baseline,
extending evaluation, exporting results, and checking the saved model and metrics.
Both students should review the code and describe their own experience in the reflection.
