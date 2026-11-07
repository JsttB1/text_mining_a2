# Tokenization and alignment audit

Checkpoint: `google-bert/bert-base-cased`. Pinned revision: `cd5ef92a9fb2f889e972770a36d4ed042daf221e`.

Every original word has exactly one supervised first subword. Continuation
subwords and special tokens have label `-100`. Dynamic batch padding also
uses `-100`. Gold B-/I- labels are preserved at word level.

No sequences were truncated and no words were dropped. Inputs include
[CLS]/[SEP] and fit the 512-token limit. One validation sentence originally
has 544 tokens; it is split into two nonoverlapping whole-word chunks.
Source IDs and word offsets allow reconstruction before entity scoring.
An entity spanning a chunk boundary retains its original B-/I- labels.

| Split | Source sentences | Model chunks | Words preserved | Input tokens | Max length | Continuation subwords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1992 | 1992 | 53178 | 69218 | 336 | 12056 |
| validation | 850 | 851 | 17546 | 24665 | 459 | 5417 |
| test | 864 | 864 | 17537 | 23576 | 137 | 4311 |

## Alignment examples

Examples below come from training data. `IGNORE` denotes label `-100`.

### Training sentence 4

```text
Subword | Word index | Source word | Label
[CLS] | None | - | IGNORE
Blue | 0 | Blue | B-LOC
Bo | 1 | Boar | I-LOC
##ar | 1 | Boar | IGNORE
Lane | 2 | Lane | I-LOC
and | 3 | and | O
associated | 4 | associated | O
deposits | 5 | deposits | O
. | 6 | . | O
[SEP] | None | - | IGNORE
```

### Training sentence 10

```text
Subword | Word index | Source word | Label
[CLS] | None | - | IGNORE
1923 | 0 | 1923 | O
Ordnance | 1 | Ordnance | O
Survey | 2 | Survey | O
map | 3 | map | O
, | 4 | , | O
with | 5 | with | O
Blue | 6 | Blue | B-LOC
Bo | 7 | Boar | I-LOC
##ar | 7 | Boar | IGNORE
Inn | 8 | Inn | I-LOC
and | 9 | and | O
Admiral | 10 | Admiral | B-LOC
Rodney | 11 | Rodney | I-LOC
P | 12 | PH | I-LOC
##H | 12 | PH | IGNORE
. | 13 | . | O
[SEP] | None | - | IGNORE
```

### Training sentence 14

```text
Subword | Word index | Source word | Label
[CLS] | None | - | IGNORE
The | 0 | The | B-LOC
Blue | 1 | Blue | I-LOC
Bo | 2 | Boar | I-LOC
##ar | 2 | Boar | IGNORE
Inn | 3 | Inn | I-LOC
( | 4 | ( | O
John | 5 | John | O
Flower | 6 | Flower | O
) | 7 | ) | O
. | 8 | . | O
[SEP] | None | - | IGNORE
```

## Verification and next step

All word labels were recovered in their original order on every split.
Both saved DatasetDicts were reloaded and compared in full, including
their features. A real dynamically padded batch passed the masking check.

The word dataset has Sequence(ClassLabel) NER features; the tokenized
dataset also records source_id, word_start, and word_end for each chunk.
Remove these metadata columns with model_inputs() before collation.
Use the saved tokenizer with DataCollatorForTokenClassification during
training. For evaluation, retain predictions only where labels != -100
and call restore_word_sequences() to reconstruct original sentences
before computing entity scores. Chunking reduces context for the one
long validation sentence; it does not change or remove gold labels.

Next: load the matching pretrained model and implement baseline training.
No model weights have been downloaded or trained in this step.

## References

- [Assignment-linked course: checkpoint and dataset structure](https://huggingface.co/learn/llm-course/chapter7/2)
- [Token classification guide: first-subword labels and collator](https://huggingface.co/docs/transformers/tasks/token_classification)
