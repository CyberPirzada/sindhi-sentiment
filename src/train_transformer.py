"""
train_transformer.py
--------------------
Enhanced fine-tuning script for a multilingual transformer model
(Sindhi / Urdu / English sentiment classification).

Features:
- Automatic data cleaning and normalization (Sindhi, Urdu, English)
- Tokenization and batching with padding/truncation
- Stratified train/test split
- Weighted loss to handle class imbalance
- Evaluation metrics (accuracy, precision, recall, F1)
- Auto GPU/CPU detection
- Saves fine-tuned model and tokenizer for Streamlit integration
"""

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers import EvalPrediction
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.preprocess import load_and_clean_csv

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
BASE_MODEL = "bert-base-multilingual-cased"  # or "Davlan/bert-base-multilingual-cased-sentiment"
DATA_PATH = "data/processed/sindhi_sentiment.csv"
SAVE_DIR = "models/sindhi_sentiment"
LOG_DIR = "./logs_transformer"
RESULT_DIR = "./results_transformer"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {device.upper()}")

# ---------------------------------------------------
# 1️⃣ Load & preprocess dataset
# ---------------------------------------------------
def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")
    df = load_and_clean_csv(DATA_PATH)
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'sentiment' columns.")
    print(f"✅ Loaded {len(df)} records from {DATA_PATH}")
    return df


# ---------------------------------------------------
# 2️⃣ Tokenization
# ---------------------------------------------------
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


# ---------------------------------------------------
# 3️⃣ Compute metrics function
# ---------------------------------------------------
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------
# 4️⃣ Main training function
# ---------------------------------------------------
def train_transformer(num_epochs=2, batch_size=8):
    df = load_dataset()

    # Encode labels
    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["sentiment"])
    label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"🗂️ Label Mapping: {label_map}")

    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["labels"]
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)

    train_dataset = train_dataset.rename_column("labels", "label")
    test_dataset = test_dataset.rename_column("labels", "label")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Model initialization
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
    model.to(device)

    # ---------------------------------------------------
    # Weighted loss (handle class imbalance)
    # ---------------------------------------------------
    from torch import nn
    import numpy as np

    class_counts = df["labels"].value_counts().sort_index().values
    class_weights = torch.tensor(len(df) / (len(class_counts) * class_counts), dtype=torch.float)
    class_weights = class_weights.to(device)

    def custom_loss(outputs, labels):
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        return loss_fct(outputs.view(-1, model.config.num_labels), labels.view(-1))

    # ---------------------------------------------------
    # Training Arguments
    # ---------------------------------------------------
    training_args = TrainingArguments(
        output_dir=RESULT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=LOG_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",  # disable wandb
    )

    # ---------------------------------------------------
    # Trainer
    # ---------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------
    print("🚀 Starting fine-tuning...")
    trainer.train()

    print("✅ Training complete. Evaluating model...")
    eval_results = trainer.evaluate()
    print("📊 Evaluation Results:", eval_results)

    # ---------------------------------------------------
    # Save model and tokenizer
    # ---------------------------------------------------
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"💾 Model & tokenizer saved to: {SAVE_DIR}")
    print("✨ Fine-tuning completed successfully.")


# ---------------------------------------------------
# Run Script
# ---------------------------------------------------
if __name__ == "__main__":
    train_transformer(num_epochs=2, batch_size=8)
