# src/preprocess.py
"""
Preprocessing utilities for Sindhi Sentiment project.

✅ Fixes recursion issue between preprocess_text() and normalize_sindhi()
✅ Adds modular and safe structure for normalization, detection, and translation
✅ Optimized for reuse in Streamlit app and training scripts

Features:
- clean_text: remove URLs, emojis, unwanted chars
- normalize_sindhi: normalize Arabic/Sindhi letter variants
- handle_negation: replace common Sindhi negations with 'not'
- detect_language: detect language using langdetect
- translate_text: optional translation fallback (googletrans)
- preprocess_text: full pipeline (detect → translate → clean → normalize → negation)
- load_and_clean_csv: load raw CSV, preprocess, save processed file
"""

import re
from pathlib import Path
import pandas as pd

# Optional dependencies
try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from googletrans import Translator
    _translator = Translator()
except Exception:
    _translator = None


# ---------------------------------------------------------------------
# 🧹 1. Clean text
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove URLs, mentions, digits, emojis, and unwanted symbols."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+|@\w+|#\w+|\d+", " ", text)
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------
# 🔡 2. Normalize Sindhi/Arabic characters
# ---------------------------------------------------------------------
def normalize_sindhi(text: str) -> str:
    """Normalize Sindhi/Arabic character variants for consistent representation."""
    if not isinstance(text, str):
        return ""

    replacements = {
        "ى": "ي",
        "ئ": "ي",
        "ك": "ڪ",  # Arabic Kaaf → Sindhi Kaaf
        "ھ": "ه",
        "ۀ": "ه",
        "ؤ": "و",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    text = re.sub(r"[^\u0600-\u06FFa-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------
# 🚫 3. Handle negations
# ---------------------------------------------------------------------
def handle_negation(text: str) -> str:
    """Replace common Sindhi negations with English 'not'."""
    if not isinstance(text, str):
        return ""

    negations = [
        "نه", "ناهي", "نٿو", "نه ٿو", "نه هئي", "نه آهي",
        "نه ٿيندو", "نا", "نٿي", "نٿيون"
    ]
    for n in negations:
        text = re.sub(rf"\b{re.escape(n)}\b", " not ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------
# 🌐 4. Detect language
# ---------------------------------------------------------------------
def detect_language(text: str) -> str:
    """Detect language using langdetect; returns 'unknown' if unavailable."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    if detect is None:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------
# 🌍 5. Translate text (fallback)
# ---------------------------------------------------------------------
def translate_text(text: str, dest_language: str = "ur") -> str:
    """
    Translate text using googletrans; returns original text on failure.
    Default destination: Urdu ('ur'), as it shares script similarity.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    if _translator is None:
        return text
    try:
        translated = _translator.translate(text, dest=dest_language)
        return translated.text if translated and translated.text else text
    except Exception:
        return text


# ---------------------------------------------------------------------
# ⚙️ 6. Main preprocessing pipeline
# ---------------------------------------------------------------------
def preprocess_text(text: str, translate_fallback: bool = True) -> str:
    """
    Full preprocessing pipeline:
    1. Detect language
    2. Translate non-Sindhi text if requested
    3. Clean unwanted characters
    4. Normalize Sindhi/Arabic letters
    5. Replace negation words with 'not'
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Detect language
    lang = detect_language(text)

    # Step 2: Optional translation
    if translate_fallback and lang not in ("sd", "ur", "ar"):
        text = translate_text(text, dest_language="ur")

    # Step 3–5: Clean → Normalize → Handle Negation
    text = clean_text(text)
    text = normalize_sindhi(text)
    text = handle_negation(text)
    return text


# ---------------------------------------------------------------------
# 📁 7. CSV loader + cleaner
# ---------------------------------------------------------------------
def load_and_clean_csv(input_path, output_path="data/processed/sindhi_sentiment.csv", translate_fallback=True):
    """
    Load CSV with 'text' column, preprocess, and save cleaned version.
    """
    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column (case-insensitive).")

    df["text"] = df["text"].astype(str).apply(
        lambda t: preprocess_text(t, translate_fallback=translate_fallback)
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Cleaned data saved to: {output_path}")
    return df


# ---------------------------------------------------------------------
# 🧭 8. Compatibility alias for app.py
# ---------------------------------------------------------------------
def normalize_sindhi_alias(text: str) -> str:
    """
    Alias kept for backward compatibility.
    It directly calls preprocess_text() but does NOT overwrite the main normalize_sindhi().
    """
    return preprocess_text(text, translate_fallback=True)

# Keep both available without overwriting the real normalize_sindhi function.
# app.py will continue to work as before.
