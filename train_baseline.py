"""
train_baseline.py
-----------------
Enhanced TF-IDF + Logistic Regression training pipeline for Sindhi Sentiment Classification.

Key Features:
- TF-IDF text vectorization (unigram + bigram)
- Logistic Regression with GridSearchCV hyperparameter tuning
- Automatic language normalization & translation fallback
- Cross-validation scoring (accuracy, F1)
- Detailed evaluation metrics and sample predictions
- Model + vectorizer saving for Streamlit app integration
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.preprocess import load_and_clean_csv
from src.utils import save_model

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
RAW_DATA_PATH = "data/raw/sindhi_sentiment.csv"
PROCESSED_DATA_PATH = "data/processed/sindhi_sentiment.csv"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
MODEL_PATH = "models/logreg_model.joblib"


# ---------------------------------------------------
# Training Function
# ---------------------------------------------------
def train_baseline(force_clean: bool = False, translate_fallback: bool = True):
    """
    Train enhanced baseline TF-IDF + Logistic Regression model.

    Args:
        force_clean (bool): Reprocess raw data if True.
        translate_fallback (bool): Use translation fallback for non-Sindhi text.
    """

    # ---------------------------------------------------
    # 1️⃣ Load / preprocess data
    # ---------------------------------------------------
    if not os.path.exists(PROCESSED_DATA_PATH) or force_clean:
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(
                f"❌ No processed or raw dataset found at {RAW_DATA_PATH}"
            )
        print("🧹 Cleaning & preprocessing dataset...")
        df = load_and_clean_csv(
            RAW_DATA_PATH,
            output_path=PROCESSED_DATA_PATH,
            translate_fallback=translate_fallback,
        )
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH)

    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("❌ Dataset must include 'text' and 'sentiment' columns.")

    print(f"✅ Loaded {len(df)} samples.")

    # ---------------------------------------------------
    # 2️⃣ Split data
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"] if df["sentiment"].nunique() > 1 else None,
    )

    # ---------------------------------------------------
    # 3️⃣ TF-IDF Vectorization
    # ---------------------------------------------------
    print("🔤 Converting text to TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        lowercase=False,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ---------------------------------------------------
    # 4️⃣ Logistic Regression + GridSearchCV
    # ---------------------------------------------------
    print("⚙️ Starting GridSearch hyperparameter tuning...")

    param_grid = {
        "C": [0.1, 0.5, 1, 2, 5],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    }

    base_model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        n_jobs=-1,
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train_tfidf, y_train)

    best_model = grid.best_estimator_
    print(f"\n🏆 Best Parameters: {grid.best_params_}")
    print(f"📈 Best CV Score (weighted F1): {grid.best_score_:.3f}\n")

    # ---------------------------------------------------
    # 5️⃣ Evaluate on test data
    # ---------------------------------------------------
    print("📊 Evaluating final model...")
    y_pred = best_model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("=" * 60)
    print(f"🎯 Accuracy:   {acc:.3f}")
    print(f"📈 Precision:  {prec:.3f}")
    print(f"📉 Recall:     {rec:.3f}")
    print(f"💪 F1 Score:   {f1:.3f}")
    print("=" * 60)

    print("\nDetailed Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=3))

    # ---------------------------------------------------
    # 6️⃣ Show Sample Predictions
    # ---------------------------------------------------
    print("\n🔍 Sample Predictions:")
    sample_df = pd.DataFrame(
        {"Text": X_test.iloc[:8], "True": y_test.iloc[:8], "Pred": y_pred[:8]}
    )
    print(sample_df.to_string(index=False))

    # ---------------------------------------------------
    # 7️⃣ Save Model + Vectorizer
    # ---------------------------------------------------
    os.makedirs("models", exist_ok=True)
    save_model(vectorizer, VECTORIZER_PATH)
    save_model(best_model, MODEL_PATH)

    print(f"\n✅ Saved model and vectorizer to 'models/'")
    print(f"📦 Vectorizer: {VECTORIZER_PATH}")
    print(f"📦 Model:      {MODEL_PATH}")
    print("\n✨ Training complete!")


# ---------------------------------------------------
# Run Script
# ---------------------------------------------------
if __name__ == "__main__":
    train_baseline(force_clean=False, translate_fallback=True)
