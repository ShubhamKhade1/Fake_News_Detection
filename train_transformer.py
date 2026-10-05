"""Stretch goal: Compare calibrated Logistic Regression baseline against a Multilingual Transformer.

Models supported:
- 'google/muril-base-cased' (Pretrained specifically for 17 Indian languages + English)
- 'xlm-roberta-base' (Multilingual RoBERTa across 100 languages)

Usage:
    python train_transformer.py --model google/muril-base-cased --data data/news.csv
    # For fast CPU run:
    python train_transformer.py --model google/muril-base-cased --epochs 1 --max_samples 1200
"""
import sys, os, inspect, argparse, pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_transformer_comparison(data_path="data/news.csv", model_name="google/muril-base-cased", epochs=3, batch_size=16, max_samples=None):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError:
        print("\n[!] 'transformers', 'torch', or 'datasets' is not installed.")
        print("    Install them with: pip install torch transformers datasets accelerate")
        print(f"    Then rerun: python train_transformer.py --model {model_name}\n")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from preprocess import detect_language

    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    print(f"Compute Device detected: {device_name}")
    if not torch.cuda.is_available():
        print("Notice: Training on CPU. To accelerate, you can pass '--epochs 1' or '--max_samples 1200'.")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path).dropna(subset=["text", "label"])
    if "lang" not in df:
        df["lang"] = df["text"].apply(detect_language)

    if max_samples and max_samples < len(df):
        print(f"Subsampling to {max_samples} balanced rows for faster execution...")
        per_group = max(1, max_samples // (len(df["lang"].unique()) * 2))
        sub_dfs = [group.sample(n=min(len(group), per_group), random_state=42) for _, group in df.groupby(["lang", "label"])]
        df = pd.concat(sub_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Train / test split per language
    train_dfs, test_dfs = [], []
    for lang in df["lang"].unique():
        lang_df = df[df["lang"] == lang]
        if len(lang_df) > 1:
            stratify = lang_df["label"] if len(lang_df["label"].unique()) > 1 else None
            tr, te = train_test_split(lang_df, test_size=0.2, random_state=42, stratify=stratify)
            train_dfs.append(tr)
            test_dfs.append(te)
        else:
            train_dfs.append(lang_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

    print(f"Training samples: {len(train_df)} | Evaluation samples: {len(test_df)}")

    print(f"Initializing tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_func(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(tokenize_func, batched=True)
    test_ds = test_ds.map(tokenize_func, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Handle evaluation strategy parameter name compatibility across transformers versions
    training_args_kwargs = {
        "output_dir": "./models/transformer_checkpoints",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "logging_steps": 25,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
    }
    
    # Check eval strategy argument name in current transformers version
    arg_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in arg_params:
        training_args_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in arg_params:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    # Build Trainer kwargs with compatibility for transformers 4.x vs 5.x
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": test_ds,
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print(f"\n--- Fine-tuning {model_name} ({epochs} epochs) ---")
    trainer.train()

    print("\n--- Evaluating Transformer on Test Set ---")
    eval_results = trainer.evaluate()
    print(f"Overall Results for {model_name}:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Per-language evaluation
    print(f"\n--- Per-Language Evaluation for {model_name} ---")
    for lang in test_df["lang"].unique():
        lang_sub = test_df[test_df["lang"] == lang].reset_index(drop=True)
        if len(lang_sub) > 0:
            sub_ds = Dataset.from_pandas(lang_sub[["text", "label"]])
            sub_ds = sub_ds.map(tokenize_func, batched=True)
            preds_raw = trainer.predict(sub_ds)
            lang_acc = preds_raw.metrics.get("test_accuracy", 0.0)
            lang_f1 = preds_raw.metrics.get("test_f1", 0.0)
            print(f"  [{lang}] Samples: {len(lang_sub)} | Accuracy: {lang_acc:.4f} | F1: {lang_f1:.4f}")

    print("\n[+] Transformer evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Transformer Benchmark")
    parser.add_argument("--data", default="data/news.csv", help="Path to news dataset CSV")
    parser.add_argument("--model", default="google/muril-base-cased", choices=["google/muril-base-cased", "xlm-roberta-base"], help="Model name")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default 1 for CPU)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max samples limit for fast CPU benchmark (e.g. 1200)")
    args = parser.parse_args()

    run_transformer_comparison(
        data_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
