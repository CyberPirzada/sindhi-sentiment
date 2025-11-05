"""
utils.py
---------
General helper functions for saving, loading, and managing models, tokenizers, and metadata.
Supports both scikit-learn models (via joblib) and Hugging Face Transformers.
"""

import os
import json
import joblib
from datetime import datetime
from typing import Any, Union

# -------------------------------------
# 📦 Joblib-based model utilities
# -------------------------------------

def save_model(obj: Any, path: str):
    """
    Save a model, vectorizer, or transformer-compatible object.
    Automatically creates directories if missing.
    Also creates a metadata file for traceability.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

    meta = {
        "path": path,
        "type": str(type(obj)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Model saved successfully at: {path}")


def load_model(path: str):
    """
    Load a model or vectorizer from disk.
    Also checks for existence of a metadata file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found at: {path}")

    obj = joblib.load(path)
    print(f"✅ Loaded model from: {path}")

    meta_path = path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"ℹ️ Model metadata: {meta.get('timestamp')} | {meta.get('type')}")

    return obj


# -------------------------------------
# 🤗 Hugging Face model utilities
# -------------------------------------

def save_transformer(model, tokenizer, save_dir: str):
    """
    Save a Hugging Face transformer model and tokenizer.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    meta = {
        "save_dir": save_dir,
        "model_class": str(type(model)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Transformer model and tokenizer saved at: {save_dir}")


def load_transformer(save_dir: str):
    """
    Load a transformer model and tokenizer from the given directory.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"❌ Transformer directory not found at: {save_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    meta_path = os.path.join(save_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"ℹ️ Transformer metadata: {meta.get('timestamp')} | {meta.get('model_class')}")

    print(f"✅ Transformer model and tokenizer loaded from: {save_dir}")
    return model, tokenizer


# -------------------------------------
# 🧹 Utility: check paths and cleanup
# -------------------------------------

def ensure_dir(path: str):
    """Ensure that the directory exists."""
    os.makedirs(path, exist_ok=True)


def list_saved_models(directory: str):
    """List all saved joblib or transformer model directories."""
    if not os.path.exists(directory):
        print("⚠️ No models found.")
        return []

    models = [f for f in os.listdir(directory) if f.endswith(".joblib") or os.path.isdir(os.path.join(directory, f))]
    print(f"📁 Found models: {models}")
    return models
