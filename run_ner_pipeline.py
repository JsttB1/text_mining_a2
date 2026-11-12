"""Main entrypoint for the Archaeo-NER sequence labelling pipeline.

Coordinates the end-to-end workflow by calling:
  1. data_loader: Reads and parses fold 1 IOB files.
  2. tokenization: Aligns labels to BERT subwords.
  3. trainer: Trains the model and evaluates on the test set.
  4. evaluation: Formats and outputs test metrics.
"""

import argparse
from pathlib import Path
from transformers import set_seed

from data_loader import load_all_splits, print_dataset_statistics
from tokenization import DEFAULT_MODEL_CHECKPOINT, get_tokenizer, tokenize_dataset
from trainer import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Archaeo-NER Sequence Labelling Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/raw/fold1", help="Path to raw fold1 directory")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline", help="Directory for model checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for output tables and metrics")
    parser.add_argument("--model_checkpoint", type=str, default=DEFAULT_MODEL_CHECKPOINT, help="Base transformer model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and eval")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (AdamW)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval_only", action="store_true", help="Evaluate existing checkpoint without training")
    parser.add_argument("--model_path", type=str, default=None, help="Saved model directory for --eval_only")
    parser.add_argument("--data_stats", action="store_true", help="Print dataset statistics and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Load data
    raw_splits, hf_dataset = load_all_splits(args.data_dir)

    if args.data_stats:
        print_dataset_statistics(raw_splits)
        return

    # 2. Tokenize and align labels
    tokenizer = get_tokenizer(args.model_checkpoint)
    tokenized_dataset = tokenize_dataset(hf_dataset, tokenizer)

    # 3. Train and evaluate
    metrics, markdown_report = train_and_evaluate(
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        eval_only=args.eval_only,
        model_path=args.model_path,
    )

    # 4. Display and save results
    print(markdown_report)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "test_results_summary.md"
    output_path.write_text(markdown_report, encoding="utf-8")
    print(f"\nSaved final results table to: {output_path}")


if __name__ == "__main__":
    main()
