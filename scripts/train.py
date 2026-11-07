"""Fine-tune the pinned BERT baseline and export reconstructed word-level results."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT / "src"))

import torch
from datasets import load_from_disk
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

from archaeo_ner.data import LABEL_NAMES, LABEL_TO_ID
from archaeo_ner.evaluation import decode_predictions, export_results, trainer_metrics
from archaeo_ner.preprocessing import CHECKPOINT, REVISION, model_inputs


class CheckLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "eval_loss"):
            if key in (logs or {}) and not math.isfinite(logs[key]):
                raise ValueError(f"Nonfinite {key} at step {state.global_step}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Two optimizer steps on 16 train examples; no test access")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", help="Path to a checkpoint from this run")
    parser.add_argument("--evaluate-only", action="store_true", help="Re-evaluate the saved final model")
    args = parser.parse_args()
    if args.smoke_test and (args.resume or args.evaluate_only):
        parser.error("Smoke mode cannot resume or evaluate the final model")
    output = ROOT / "outputs" / ("smoke" if args.smoke_test else "baseline")
    if not args.smoke_test and output.exists() and any(output.iterdir()) and not (args.resume or args.evaluate_only):
        raise ValueError("Baseline output exists; use --resume or --evaluate-only to preserve it")
    output.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "reports/baseline"
    data_dir = ROOT / "data/processed/fold1"
    config = json.loads((ROOT / "reports/preprocessing_config.json").read_text())
    if config["checkpoint"] != CHECKPOINT or config["revision"] != REVISION or config["label_names"] != list(LABEL_NAMES):
        raise ValueError("Preprocessing configuration does not match the training model/labels")
    raw = load_from_disk(str(data_dir / "hf_words"))
    tokenized = load_from_disk(str(data_dir / "hf_tokenized"))
    for split in raw:
        if raw[split].features["ner_tags"].feature.names != list(LABEL_NAMES):
            raise ValueError("Saved dataset label schema mismatch")
    tokenizer = AutoTokenizer.from_pretrained(data_dir / "tokenizer", local_files_only=True)
    set_seed(42)
    torch.set_num_threads(4)
    model_source = output / "final_model" if args.evaluate_only else CHECKPOINT
    model = AutoModelForTokenClassification.from_pretrained(
        model_source, **({} if args.evaluate_only else {"revision": REVISION}),
        id2label=dict(enumerate(LABEL_NAMES)), label2id=LABEL_TO_ID,
        local_files_only=args.local_files_only or args.evaluate_only,
    )
    training_args = TrainingArguments(
        output_dir=str(output), learning_rate=2e-5, weight_decay=0.01,
        num_train_epochs=3, per_device_train_batch_size=8, per_device_eval_batch_size=8,
        optim="adamw_torch", seed=42,
        eval_strategy="no" if args.smoke_test else "epoch",
        save_strategy="no" if args.smoke_test else "epoch", save_total_limit=1,
        logging_steps=1 if args.smoke_test else 10, logging_first_step=True,
        logging_nan_inf_filter=False, report_to="none", push_to_hub=False,
        dataloader_pin_memory=False, disable_tqdm=True,
        max_steps=2 if args.smoke_test else -1,
    )
    print(f"Device: {training_args.device}; model: {CHECKPOINT}; smoke: {args.smoke_test}", flush=True)
    train_data = model_inputs(tokenized["train"])
    if args.smoke_test:
        train_data = train_data.select(range(16))
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data,
        eval_dataset=model_inputs(tokenized["validation"]),
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=trainer_metrics(raw["validation"], tokenized["validation"]),
        callbacks=[CheckLoss()],
    )
    if not args.evaluate_only:
        run_config = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": CHECKPOINT, "revision": REVISION,
            "baseline_policy": "Assignment-linked tutorial settings (2e-5, 0.01, 3 epochs), not bare TrainingArguments defaults; no hyperparameter search",
            "optimizer_note": "Standard torch AdamW selected explicitly for MPS compatibility rather than version-dependent fused default",
            "selection_policy": "Final epoch; no early stopping or test-driven selection",
            "device": str(training_args.device), "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "versions": {p: version(p) for p in ("torch", "transformers", "datasets", "accelerate", "seqeval")},
            "training_arguments": training_args.to_dict(), "preprocessing": config,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        training = trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_metrics("train", training.metrics)
        trainer.save_state()
        if args.smoke_test:
            print("Smoke test passed: two real optimizer steps with finite loss; no test predictions.", flush=True)
            return
        trainer.save_model(str(output / "final_model"))
        tokenizer.save_pretrained(output / "final_model")
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        (report_dir / "training_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")
    for split in ("validation", "test"):
        trainer.compute_metrics = trainer_metrics(raw[split], tokenized[split])
        prediction = trainer.predict(model_inputs(tokenized[split]), metric_key_prefix=split)
        gold, predicted = decode_predictions(prediction, raw[split], tokenized[split])
        result = export_results(split, raw[split], gold, predicted, output, report_dir)
        trainer.save_metrics(split, prediction.metrics)
        print(f"{split}: micro F1={result['entity_micro']['f1']:.4f}, macro F1={result['entity_macro_f1']:.4f}", flush=True)
    print("Baseline complete; final model, predictions, logs, and report tables saved.", flush=True)


if __name__ == "__main__":
    main()
