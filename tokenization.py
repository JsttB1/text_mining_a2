"""Subword tokenization and label alignment for token classification."""

from datasets import DatasetDict
from transformers import AutoTokenizer

DEFAULT_MODEL_CHECKPOINT = "google-bert/bert-base-cased"


def get_tokenizer(model_checkpoint: str = DEFAULT_MODEL_CHECKPOINT):
    """Load pretrained fast tokenizer."""
    return AutoTokenizer.from_pretrained(model_checkpoint)


def align_labels_with_tokens(examples: dict, tokenizer, max_length: int = 512) -> dict:
    """Align word-level NER tags with subword tokens (Hugging Face NLP Course Chapter 7).

    - The first subword token of each word is assigned that word's gold label.
    - Subsequent subwords and special tokens are set to -100 to ignore them in loss.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], padding)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word
                label_ids.append(labels[word_idx])
            else:
                # Continuation subwords
                label_ids.append(-100)
            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def tokenize_dataset(hf_dataset: DatasetDict, tokenizer, max_length: int = 512) -> DatasetDict:
    """Apply subword tokenization and label alignment across all dataset splits."""
    return hf_dataset.map(
        lambda examples: align_labels_with_tokens(examples, tokenizer, max_length=max_length),
        batched=True,
    )
