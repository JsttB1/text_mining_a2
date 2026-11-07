# Baseline run summary

Completed on 7 November 2026. The final three-epoch BERT model achieved
**66.06% entity micro F1** and **52.16% entity macro F1** on the supplied fold1 test set.

| Split | Sentences | Words scored | Entity micro F1 | Entity macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 850 | 17,546 | 60.48% | 53.91% |
| Test | 864 | 17,537 | 66.06% | 52.16% |

See [test tables](test_results.md), [validation tables](validation_results.md),
and [method notes](method.md). The CSV files contain precision, recall, F1,
support, predicted count, TP, FP, and FN for all 12 word labels and all six
entity types. JSON metrics retain full numerical precision.

## Training history

| Epoch | Validation loss | Entity micro F1 | Entity macro F1 |
| --- | ---: | ---: | ---: |
| 1 | 0.1868 | 56.77% | 46.84% |
| 2 | 0.1889 | 60.86% | 54.67% |
| 3 | 0.1998 | 60.48% | 53.91% |

Training completed 747 optimizer steps in about 165 seconds on Apple MPS,
excluding model download and final result export. The final epoch was used as
specified before training; epoch 2's slightly better validation F1 was not used
to change the checkpoint-selection policy. This is one seeded baseline run,
not a comparison of tuned models.

## Findings to examine in the report

- Entity F1 varies from 83.52% for PER and 74.52% for LOC to 26.97% for MAT and
  22.22% for SPE. Macro F1 gives each type equal weight, so weak types lower it
  more than they lower pooled micro F1.
- Test SPE has only two gold entities. Its 50% recall means one correct entity;
  conclusions about this type should be limited by that small count.
- Boundary/continuation errors matter: CON has B-label F1 of 54.44%, I-label F1
  of 13.33%, and exact entity F1 of 48.14%. Inspect prediction examples before
  explaining the cause of these differences.
- Test I-SPE has no gold examples. Its displayed zero recall is the documented
  undefined-score convention. I-MAT has six gold examples and no predicted
  I-MAT words, which is an observed failure to recover those continuation labels.
- Validation F1 peaks at epoch 2 while validation loss rises across epochs.
  These observations do not establish a robust optimal epoch from a single run.

## Verification

All 17 unit tests passed. Every exported validation/test word and gold label was
compared with the original dataset. All reported token metrics match independent
scikit-learn calculations; all entity metrics match seqeval strict IOB2 scoring.
The final model reloads with the correct 13-label mapping and reproduces the
saved predictions on eight validation sentences. Evidence and the saved model's
SHA-256 checksum are in [verification.json](verification.json).

The final model is at `outputs/baseline/final_model/` (about 412 MiB).
Full predictions and the latest resumable checkpoint are under `outputs/baseline/`.
Those files are ignored by Git. The final report, student work-division/AI-use
reflection, and Brightspace submission remain unfinished.
