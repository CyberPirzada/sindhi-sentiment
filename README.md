 Sindhi Sentiment Classifier 
 — A Cultural AI for Sindhi Language Understanding

The Sindhi Sentiment Classifier is an AI-powered web application that detects emotions and sentiments in Sindhi text, celebrating the linguistic and cultural identity of Sindh.
Built with 💬 Machine Learning, NLP, and Streamlit, it helps analyze whether Sindhi sentences express positive, negative, or neutral emotions — all while embracing Ajrak-inspired aesthetics.

🌍 Features

✅ Supports Sindhi script input only (Urdu and Farsi detected as invalid)
✅ Cleans and normalizes Sindhi text using custom preprocessing
✅ Predicts sentiment via trained ML models (TF-IDF + Logistic Regression)
✅ Fast and lightweight — deployable anywhere

🧠 Tech Stack
Category	Technology
Frontend	Streamlit
Backend / ML	Scikit-learn, Joblib
Text Processing	Custom Sindhi Preprocessing
Languages Supported	Sindhi (Rejects Urdu/Farsi/English)
Deployment	Streamlit Cloud / Hugging Face Spaces / Localhost
📂 Project Structure
Sindhi-Sentiment-Classifier/
│
├── app.py                     # Main Streamlit app
├── requirements.txt           # Dependencies
├── models/                    # Trained model + vectorizer
│   ├── tfidf_vectorizer.joblib
│   └── logreg_model.joblib
├── src/                       # Source scripts
│   ├── preprocess.py          # Sindhi text preprocessing
│   ├── train_baseline.py      # Training script
│   └── utils.py               # Save/load utilities
└── README.md

⚙️ Installation

Clone the repository

git clone https://github.com/your-username/sindhi-sentiment-classifier.git
cd sindhi-sentiment-classifier


Create a virtual environment

python -m venv .venv
source .venv/bin/activate   # (Linux/macOS)
.venv\Scripts\activate      # (Windows)


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

🧩 Model Training (Optional)

If you want to retrain or fine-tune the model:

python -m src.train_baseline


The trained model and TF-IDF vectorizer will be saved under the models/ folder.

🖼️ App Preview

(Optional: Add a screenshot of your Streamlit app here)

![App Screenshot](assets/app_preview.png)

⚖️ License

This project is licensed under the MIT License — feel free to use, modify, and share with proper credit.

🌟 Acknowledgment

Special tribute to the Sindhi language, Ajrak culture, and Indus Civilization — the inspiration behind this project.
Developed with ❤️ to promote Sindhi linguistic AI and local language technology.

👨‍💻 Author

Akbar Pirzada
🔗 LinkedIn: https://www.linkedin.com/in/akbar-pirzada/

📧 akbar.pirzada@example.com

🌐 GitHub
