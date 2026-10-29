"""
Steps 1, 3, 4 & 5: Dataset Selection, TF-IDF Feature Extraction,
Model Training (Multinomial Naive Bayes, Logistic Regression, SVM),
and Evaluation Metrics (Precision, Recall, F1-Score, Accuracy).
"""

import io
import json
import os
import zipfile
import joblib
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from preprocessing import preprocess_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_PATH = os.path.join(DATA_DIR, "spam.csv")

UCI_SPAM_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)


def load_or_download_dataset() -> pd.DataFrame:
    """
    Step 1: Dataset Selection
    Downloads the standard SMS/Email Spam Collection dataset (5,572 messages)
    from the UCI Machine Learning Repository if not already saved locally.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATASET_PATH):
        print(f"[Step 1] Loading existing dataset from {DATASET_PATH}...")
        df = pd.read_csv(DATASET_PATH, encoding="utf-8")
        return df

    print("[Step 1] Downloading SMS/Email Spam Collection dataset from UCI repository...")
    try:
        response = requests.get(UCI_SPAM_ZIP_URL, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            with zf.open("SMSSpamCollection") as raw_file:
                df = pd.read_csv(
                    raw_file,
                    sep="\t",
                    header=None,
                    names=["label", "text"],
                    encoding="utf-8",
                )
        df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
        print(f"[Step 1] Saved {len(df):,} labeled messages to {DATASET_PATH}.")
        return df
    except Exception as exc:
        print(f"[Warning] Could not download dataset ({exc}). Creating built-in fallback dataset...")
        samples = [
            ("spam", "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Claim now!"),
            ("spam", "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May. Text FA to 87121 to receive entry"),
            ("spam", "Congratulations! Your email was selected to claim a $1000 Walmart gift card. Click http://claim-free.biz"),
            ("spam", "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"),
            ("spam", "Verify your bank account immediately! Unusual login activity detected. Click here to reset password."),
            ("spam", "Get cheap prescription medications online! 80% discount on Viagra, Cialis, free shipping worldwide!"),
            ("spam", "Earn $5,000 per week working from home! No experience required. Limited spots available, reply NOW!"),
            ("spam", "FINAL NOTICE: Your invoice #84920 is overdue. Pay immediately to avoid legal action and account suspension."),
            ("ham", "Hey, are we still meeting for lunch at 12:30 pm tomorrow near the office?"),
            ("ham", "Hi team, please find attached the Q3 financial report and project roadmap notes from Monday's meeting."),
            ("ham", "Can you review the pull request when you get a chance? I fixed the authentication bug."),
            ("ham", "Mom called earlier, she asked if you're coming home for dinner this weekend."),
            ("ham", "Thanks for sending over the updated slides. I'll present them during tomorrow's standup."),
            ("ham", "Don't forget to bring your laptop charger and notebook to the workshop at 3pm."),
            ("ham", "Just landed at the airport, I will take a cab and reach home in about 30 minutes."),
            ("ham", "Hi Sarah, let's reschedule our 1-on-1 call to Thursday afternoon if that works for you."),
        ] * 40
        df = pd.DataFrame(samples, columns=["label", "text"])
        df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
        return df


def train_and_evaluate() -> dict:
    """Run the full training and evaluation pipeline across Steps 1 to 5."""
    # Step 1: Load dataset
    df = load_or_download_dataset()
    df = df.dropna(subset=["label", "text"]).drop_duplicates()
    df["target"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    total_count = len(df)
    spam_count = int((df["target"] == 1).sum())
    ham_count = int((df["target"] == 0).sum())
    print(
        f"[Step 1] Dataset Summary: {total_count:,} unique messages "
        f"({ham_count:,} Ham [{ham_count/total_count:.1%}], "
        f"{spam_count:,} Spam [{spam_count/total_count:.1%}])"
    )

    # Step 2: Text Preprocessing (NLP)
    print("[Step 2] Running NLP text preprocessing (lowercasing, cleaning, stopwords, stemming)...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Stratified train/test split (80% train, 20% test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_text"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    # Step 3: Feature Extraction (TF-IDF Vectorization)
    print("[Step 3] Extracting numerical features with TfidfVectorizer (unigrams & bigrams)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # Step 4: Model Training
    print("[Step 4] Training classification models...")
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced", random_state=42),
        "Support Vector Machine (SVM)": CalibratedClassifierCV(
            SVC(kernel="linear", C=1.5, random_state=42), ensemble=False
        ),
    }

    trained_models = {}
    metrics_summary = {}

    # Step 5: Evaluation Metrics
    print("\n[Step 5] Model Evaluation Results on Test Set:")
    print("-" * 78)
    print(f"{'Model':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 78)

    best_model_name = None
    best_f1 = -1.0

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        trained_models[name] = model
        metrics_summary[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "report": classification_report(
                y_test, y_pred, target_names=["Ham (0)", "Spam (1)"], output_dict=True
            ),
        }

        print(f"{name:<32} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print("-" * 78)
    print(f"Best Performing Model (by F1-Score): {best_model_name} (F1 = {best_f1:.4f})\n")

    # Save artifacts for Step 6 (Deployment)
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifacts = {
        "vectorizer": vectorizer,
        "models": trained_models,
        "best_model_name": best_model_name,
        "metrics": metrics_summary,
        "dataset_stats": {
            "total_messages": total_count,
            "ham_count": ham_count,
            "spam_count": spam_count,
            "vocab_size": len(vectorizer.vocabulary_),
        },
    }

    artifacts_path = os.path.join(MODELS_DIR, "spam_artifacts.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    joblib.dump(artifacts, artifacts_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_name": best_model_name,
                "dataset_stats": artifacts["dataset_stats"],
                "metrics": metrics_summary,
            },
            f,
            indent=2,
        )

    print(f"[Saved] Model bundle written to {artifacts_path}")
    print(f"[Saved] Evaluation metrics written to {metrics_path}")
    return artifacts


if __name__ == "__main__":
    train_and_evaluate()
