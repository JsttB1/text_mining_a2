# Test results

strict IOB2 exact spans and type; orphan I-tags ignored as entities; zero_division=0; macro over all six types; O excluded from entity averages

Scores below are percentages. Support counts are gold words/entities.

## Word-level B-/I-labels

| Label | Precision | Recall | F1 | Support | Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-ART | 57.01% | 72.62% | 63.87% | 168 | 214 |
| I-ART | 52.83% | 37.33% | 43.75% | 75 | 53 |
| B-CON | 47.03% | 64.63% | 54.44% | 147 | 202 |
| I-CON | 40.00% | 8.00% | 13.33% | 75 | 15 |
| B-LOC | 76.47% | 90.28% | 82.80% | 144 | 170 |
| I-LOC | 78.95% | 81.08% | 80.00% | 148 | 152 |
| B-MAT | 46.67% | 23.73% | 31.46% | 59 | 30 |
| I-MAT | 0.00% | 0.00% | 0.00% | 6 | 0 |
| B-PER | 80.91% | 94.35% | 87.11% | 283 | 330 |
| I-PER | 95.73% | 80.00% | 87.16% | 140 | 117 |
| B-SPE | 14.29% | 50.00% | 22.22% | 2 | 7 |
| I-SPE | 0.00% | 0.00% | 0.00% | 0 | 0 |

## Exact full entities

| Label | Precision | Recall | F1 | Support | Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| ART | 51.40% | 65.48% | 57.59% | 168 | 214 |
| CON | 41.58% | 57.14% | 48.14% | 147 | 202 |
| LOC | 68.82% | 81.25% | 74.52% | 144 | 170 |
| MAT | 40.00% | 20.34% | 26.97% | 59 | 30 |
| PER | 77.58% | 90.46% | 83.52% | 283 | 330 |
| SPE | 14.29% | 50.00% | 22.22% | 2 | 7 |

Entity micro F1: **66.06%**.
Entity macro F1 (six types): **52.16%**.
Word accuracy (including O): 96.44%.

Macro F1 is the unweighted mean of the six per-type entity F1 scores.
Micro F1 = 2TP / (2TP + FP + FN), pooling counts over entity types.
Undefined precision/recall/F1 are recorded as zero. In particular,
a label with zero gold support has undefined recall, represented as zero.

References: [averaging definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html),
[strict IOB2 scoring](https://github.com/chakki-works/seqeval).
