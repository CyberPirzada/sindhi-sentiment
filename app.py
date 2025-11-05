# ---------------------------------------------------------
# 🗣️ Sindhi Sentiment Classifier App
# ---------------------------------------------------------
import os
import re
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from src.preprocess import normalize_sindhi
from src.utils import load_model, load_transformer


# ---------------------------------------------------------
# 1️⃣ PAGE CONFIGURATION
# ---------------------------------------------------------
os.environ["HF_HOME"] = "E:/Python_Bootcamp/project/hf_cache"
st.set_page_config(page_title="🗣️ Sindhi Sentiment Classifier", layout="centered")
st.title("🗣️ Sindhi Sentiment Classifier (لفظن جي احساس کي سمجهڻ)")

st.markdown(
    """
    This app detects **sentiment (Positive / Negative / Neutral)** from Sindhi text.  
    Only Sindhi script is supported. Urdu or Farsi will trigger a warning.
    """
)

# ---------------------------------------------------------
# 2️⃣ LANGUAGE DETECTION (Rule-based)
# ---------------------------------------------------------
def detect_language(text: str) -> str:
    """Rule-based detection for Sindhi, Urdu, Farsi, or English."""
    sindhi_chars = "ٽٿڊڍڙڦڻڳڪٻڱڇڄڃ"
    sindhi_pattern = re.compile(f"[{sindhi_chars}]")

    # Sindhi check
    if sindhi_pattern.search(text):
        return "sd"

    # Urdu check (Arabic script + Urdu-specific)
    urdu_pattern = re.compile(r"[اآبپتٹثجچحخدڈذرڑزسشصضطظعغفقکگلمنوهیے]")
    if urdu_pattern.search(text):
        return "ur"

    # Farsi check (Persian-specific letters)
    farsi_pattern = re.compile(r"[گچپژ]")
    if farsi_pattern.search(text):
        return "fa"

    # English check
    if re.search(r"[A-Za-z]", text):
        return "en"

    return "unknown"


# ---------------------------------------------------------
# 3️⃣ MODEL LOADING
# ---------------------------------------------------------
available_models = []
if os.path.exists("models/tfidf_vectorizer.joblib") and os.path.exists("models/logreg_model.joblib"):
    available_models.append("Baseline (TF-IDF + Logistic Regression)")
else:
    st.warning("⚠️ No local model found in 'models/'. Please train or place model files there.")

model_choice = st.radio("Select Model:", available_models)

# ---------------------------------------------------------
# 4️⃣ USER INPUT
# ---------------------------------------------------------
user_input = st.text_area(
    "✍️ Enter Sindhi text below:",
    height=150,
    placeholder="مثال: اڄ موسم تمام سٺو آهي",
)

# ---------------------------------------------------------
# 5️⃣ SENTIMENT ANALYSIS HANDLER
# ---------------------------------------------------------
if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first.")
        st.stop()

    # Detect language
    lang = detect_language(user_input)
    st.write(f"🌐 **Detected Language:** {lang.upper()}")

    # Validate Sindhi input
    if lang != "sd":
        if lang == "ur":
            st.error("⚠️ Urdu text detected. Please enter Sindhi text only.")
        elif lang == "fa":
            st.error("⚠️ Farsi text detected. Please enter Sindhi text only.")
        elif lang == "en":
            st.error("⚠️ English text detected. Please enter Sindhi text only.")
        else:
            st.error("❌ Unsupported text. Please use Sindhi script only.")
        st.stop()

    # Normalize Sindhi text
    clean_text = normalize_sindhi(user_input)
    st.info("✅ Sindhi text detected. Proceeding with sentiment analysis...")

    # -----------------------------------------------------
    # MODEL: TF-IDF + Logistic Regression
    # -----------------------------------------------------
    if "Baseline" in model_choice:
        try:
            vectorizer = load_model("models/tfidf_vectorizer.joblib")
            clf = load_model("models/logreg_model.joblib")

            X = vectorizer.transform([clean_text])
            pred = clf.predict(X)[0]
            probs = clf.predict_proba(X)[0]

            sentiment_labels = list(clf.classes_)
            sentiment_prob = dict(zip(sentiment_labels, probs))

            st.success(f"🎯 Predicted Sentiment: **{pred.upper()}**")
            st.bar_chart(sentiment_prob)

        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")

    # -----------------------------------------------------
    # (Optional) Transformer-based model section
    # -----------------------------------------------------
    else:
        try:
            model_dir = "models/sindhi_sentiment"
            model, tokenizer = load_transformer(model_dir)
            device = 0 if torch.cuda.is_available() else -1

            nlp = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device,
                return_all_scores=True
            )

            results = nlp(clean_text)[0]
            sorted_res = sorted(results, key=lambda x: x["score"], reverse=True)
            label = sorted_res[0]["label"]
            score = sorted_res[0]["score"]

            sentiment_map = {
                "LABEL_0": "Negative 🙁",
                "LABEL_1": "Neutral 😐",
                "LABEL_2": "Positive 🙂"
            }
            final_label = sentiment_map.get(label, label)
            st.success(f"Predicted Sentiment: **{final_label}** ({score:.2f})")
            st.bar_chart({r['label']: r['score'] for r in results})

        except Exception as e:
            st.error(f"❌ Transformer model failed: {e}")

# ---------------------------------------------------------
# 6️⃣ FOOTER
# ---------------------------------------------------------
st.caption('\nCreated by: ❤️ Akbar Pirzada')
st.caption('\nLinkedin  🟦 https://www.linkedin.com/in/akbar-pirzada/')
st.caption('\nGitHub	💻 https://github.com/CyberPirzada/')
st.caption('\nWhatsApp	🟢+92-3113870907')

#st.caption("Built with ❤️ using Streamlit • Scikit-Learn • Hugging Face Transformers")

