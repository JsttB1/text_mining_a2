"""Model training and evaluation orchestration using Hugging Face Trainer."""

from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from data_loader import ID_TO_LABEL, LABEL_NAMES, LABEL_TO_ID
from evaluation import build_compute_metrics_fn, evaluate_predictions, format_results_markdown


def train_and_evaluate(
    tokenized_dataset,
    tokenizer,
    model_checkpoint: str = "google-bert/bert-base-cased",
    output_dir: str | Path = "outputs/baseline",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    eval_only: bool = False,
    model_path: str | None = None,
) -> tuple[dict, str]:
    """Train a BERT token classification model and evaluate on the test split."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Model source selection
    model_source = model_path if (eval_only and model_path) else model_checkpoint
    print(f"Loading model: {model_source}")

    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(LABEL_NAMES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # 2. Setup training arguments (compatible with both newer and older transformers)
    import inspect
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=str(output_path),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        save_total_limit=1,
        seed=seed,
        logging_steps=50,
        report_to="none",
        **{eval_key: "epoch"},
    )

    # 3. Initialize Trainer
    trainer_params = inspect.signature(Trainer.__init__).parameters
    tokenizer_kwarg = {"processing_class": tokenizer} if "processing_class" in trainer_params else {"tokenizer": tokenizer}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics_fn(),
        **tokenizer_kwarg,
    )

    # 4. Train model (if not evaluation-only)
    if not eval_only:
        print("\nStarting model fine-tuning...")
        trainer.train()
        final_model_path = output_path / "final_model"
        print(f"Saving fine-tuned model to {final_model_path}")
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))

    # 5. Predict on test set
    print("\nRunning final evaluation on test set...")
    test_predictions, test_labels, _ = trainer.predict(tokenized_dataset["test"])
    test_preds = np.argmax(test_predictions, axis=2)

    # Filter out -100 subword markers to reconstruct word-level tags
    true_predictions = [
        [LABEL_NAMES[p_id] for (p_id, l_id) in zip(pred, lab) if l_id != -100]
        for pred, lab in zip(test_preds, test_labels)
    ]
    true_labels = [
        [LABEL_NAMES[l_id] for (p_id, l_id) in zip(pred, lab) if l_id != -100]
        for pred, lab in zip(test_preds, test_labels)
    ]

    metrics = evaluate_predictions(true_labels, true_predictions)
    markdown_output = format_results_markdown(metrics, title="Test Set Evaluation (Fold 1)")

    return metrics, markdown_output
