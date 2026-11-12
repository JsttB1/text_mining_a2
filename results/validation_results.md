# Validation results

strict IOB2 exact spans and type; orphan I-tags ignored as entities; zero_division=0; macro over all six types; O excluded from entity averages

Scores below are percentages. Support counts are gold words/entities.

## Word-level B-/I-labels

| Label | Precision | Recall | F1 | Support | Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-ART | 55.10% | 68.65% | 61.13% | 252 | 314 |
| I-ART | 52.00% | 26.90% | 35.45% | 145 | 75 |
| B-CON | 38.55% | 61.96% | 47.53% | 163 | 262 |
| I-CON | 50.00% | 14.00% | 21.88% | 50 | 14 |
| B-LOC | 73.91% | 79.87% | 76.77% | 149 | 161 |
| I-LOC | 71.58% | 73.91% | 72.73% | 92 | 95 |
| B-MAT | 47.06% | 40.00% | 43.24% | 40 | 34 |
| I-MAT | 0.00% | 0.00% | 0.00% | 7 | 0 |
| B-PER | 78.23% | 91.69% | 84.43% | 337 | 395 |
| I-PER | 91.32% | 70.92% | 79.84% | 282 | 219 |
| B-SPE | 45.10% | 74.19% | 56.10% | 31 | 51 |
| I-SPE | 100.00% | 28.57% | 44.44% | 14 | 4 |

## Exact full entities

| Label | Precision | Recall | F1 | Support | Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| ART | 48.41% | 60.32% | 53.71% | 252 | 314 |
| CON | 34.73% | 55.83% | 42.82% | 163 | 262 |
| LOC | 65.84% | 71.14% | 68.39% | 149 | 161 |
| MAT | 41.18% | 35.00% | 37.84% | 40 | 34 |
| PER | 71.14% | 83.38% | 76.78% | 337 | 395 |
| SPE | 35.29% | 58.06% | 43.90% | 31 | 51 |

Entity micro F1: **60.48%**.
Entity macro F1 (six types): **53.91%**.
Word accuracy (including O): 94.86%.

Macro F1 is the unweighted mean of the six per-type entity F1 scores.
Micro F1 = 2TP / (2TP + FP + FN), pooling counts over entity types.
Undefined precision/recall/F1 are recorded as zero. In particular,
a label with zero gold support has undefined recall, represented as zero.

References: [averaging definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html),
[strict IOB2 scoring](https://github.com/chakki-works/seqeval).
