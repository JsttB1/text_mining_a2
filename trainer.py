from pathlib import Path
from data_loader import ID_TO_LABEL, LABEL_NAMES, LABEL_TO_ID
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from evaluation import (
    build_compute_metrics_fn,
    evaluate_predictions,
    format_results_markdown,
    postprocess_predictions,
)

def train_and_evaluate(
    tokenized_dataset,
    tokenizer,
    model_checkpoint="google-bert/bert-base-cased",
    output_dir="outputs/baseline",
    epochs= 3,
    batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    seed=42,
    eval_only=False,
    model_path=None,
):
    """Train a BERT token classification model and evaluate on the test split."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_source = model_path if (eval_only and model_path) else model_checkpoint
    print(f"Loading model: {model_source}")

    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(LABEL_NAMES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

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
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics_fn(),
        processing_class=tokenizer,
    )

    # Train model (if not evaluation-only)
    if not eval_only:
        print("\nStarting model fine-tuning...")
        trainer.train()
        final_model_path = output_path / "final_model"
        print(f"Saving fine-tuned model to {final_model_path}")
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))

    # Predict on test set
    print("\nRunning final evaluation on test set...")
    test_predictions, test_labels, _ = trainer.predict(tokenized_dataset["test"])
    true_labels, true_predictions = postprocess_predictions(test_predictions, test_labels)

    metrics = evaluate_predictions(true_labels, true_predictions)
    markdown_output = format_results_markdown(metrics, title="Test Set Evaluation (Fold 1)")

    return metrics, markdown_output
