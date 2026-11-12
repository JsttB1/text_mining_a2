# Baseline method and metric conventions

## Training

The baseline starts from `google-bert/bert-base-cased`, revision
`cd5ef92a9fb2f889e972770a36d4ed042daf221e`, with a newly initialized 13-label
token-classification head. All model parameters are fine-tuned on fold1 train.
The two-step smoke check uses a separate model instance; the full run starts
again from the original pretrained checkpoint with seed 42.

The assignment requests default hyperparameter settings without listing values.
This implementation uses the [assignment-linked tutorial's baseline](https://huggingface.co/learn/llm-course/chapter7/2):
learning rate 2e-5, weight decay 0.01, and three epochs. These are explicitly the
tutorial settings, rather than the bare `TrainingArguments` learning rate of
5e-5 and weight decay of 0.0. No hyperparameter search was performed.

| Setting | Value |
| --- | --- |
| Train/evaluation batch size | 8 |
| Epochs | 3 |
| Optimizer updates | 747 (249 per epoch) |
| Optimizer | PyTorch AdamW |
| Adam beta1 / beta2 / epsilon | 0.9 / 0.999 / 1e-8 |
| Scheduler | Linear decay, no warmup |
| Gradient accumulation | 1 |
| Gradient clipping norm | 1.0 |
| Seed | 42 |
| Precision | Float32; fp16/bf16 disabled |
| Device | Apple MPS |
| Checkpoint used for test | Final epoch |

Standard PyTorch AdamW is selected explicitly for MPS compatibility rather than
depending on the installed library's fused-optimizer default. Logging frequency,
local checkpoint retention, and external reporting settings are operational
choices; all effective arguments and package versions are recorded in
`run_config.json`. Reproducibility across different hardware/library versions
does not imply bit-for-bit identical results.

## Labels and sequences

Original splits and sentence labels are preserved. Each word contributes one
supervised first subword. Special tokens, continuation subwords, and padding use
label -100 and are excluded from scoring. One long validation sentence is split
at a word boundary; its predictions are reassembled into the original sentence
before metrics are computed. No words are dropped. Training and test sentences
fit the model's 512-token limit without splitting.

## Evaluation

Validation is evaluated after each epoch. The final epoch is used for the test
set, without early stopping, checkpoint selection, or tuning on test results.
Every original test word is retained in `outputs/baseline/test_predictions.jsonl`.

- Token-level precision, recall, and F1 are computed separately for all 12 B-/I-
  labels. They are classification metrics over original words, not subwords.
- Full entities require matching sentence, start/end word positions, and type.
  Extraction follows [seqeval's strict IOB2 scheme](https://github.com/chakki-works/seqeval).
  Orphan I-tags do not become entity starts automatically. Partial entity spans
  receive no correct-entity credit.
- Entity macro F1 is the unweighted mean of the six per-type F1 scores.
- Entity micro F1 pools TP, FP, and FN across the six types, then computes
  `2 * TP / (2 * TP + FP + FN)`.
- O is excluded from entity averages. Word accuracy includes O and is supplemental.
- Undefined precision/recall/F1 are represented as zero. Support and prediction
  counts remain visible. Test I-SPE has zero gold support, so its zero recall is
  an undefined-value convention, not evidence of observed I-SPE mistakes.

Macro/micro conventions follow the [scikit-learn definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).
All six types are included in the macro denominator, including a type with zero
gold/predicted entities. The implementation is tested against scikit-learn token
metrics and seqeval strict entity F1, as well as explicit partial-span,
wrong-type, absent-label, and chunk-boundary cases.

## Sharing and report preparation

The CSV tables, JSON metrics, method notes, and Markdown summaries in this folder
can be shared with the teammate. Full predictions and the trained model are under
`outputs/` and are ignored by Git. Keep the final assignment report within its
page limit and write it in Overleaf. These notes are supporting material, not the
completed submission report.
