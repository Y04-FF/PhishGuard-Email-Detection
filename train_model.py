from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emails.csv"
MODEL_PATH = ROOT / "phishing_model.joblib"
CHART_PATH = ROOT / "confusion_matrix.png"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
SUSPICIOUS_TERMS = [
    "verify", "urgent", "password", "suspended", "click here", "winner",
    "prize", "bank details", "confirm", "immediately", "account closure",
    "limited time", "payment failed", "credential", "claim your"
]

def url_features(texts):
    """Create numeric URL/word-pattern features for each email."""
    rows = []
    for raw in texts:
        text = str(raw)
        lower = text.lower()
        urls = URL_RE.findall(text)
        rows.append([
            len(urls),
            int(any(u.lower().startswith("http://") for u in urls)),
            int("@" in text),
            int(bool(re.search(r"\b(bit\.ly|tinyurl|t\.co|goo\.gl)\b", lower))),
            len(EMAIL_RE.findall(text)),
            sum(term in lower for term in SUSPICIOUS_TERMS),
            int("!" in text),
            int(bool(re.search(r"\b(verify|login|password|account|secure)\b", lower))),
        ])
    return pd.DataFrame(rows).to_numpy()

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["phishing", "safe"])].drop_duplicates(subset=["text"])
    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both 'phishing' and 'safe' labels.")

    X = df["text"]
    y = df["label"]
    # Stratify where possible; small custom datasets may not have enough examples.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=12000, sublinear_tf=True
        ))
    ])
    feature_union = FeatureUnion([
        ("text", text_pipe),
        ("url_features", Pipeline([
            ("extract", FunctionTransformer(url_features, validate=False))
        ]))
    ])
    model = Pipeline([
        ("features", feature_union),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 58)
    print("        PHISHING EMAIL DETECTION — EVALUATION")
    print("=" * 58)
    print(f"Total samples : {len(df)}")
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"Accuracy      : {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=["phishing", "safe"], zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=["phishing", "safe"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Phishing", "Safe"])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix — Phishing Email Detection")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=160)
    print(f"Confusion matrix saved to: {CHART_PATH}")

    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")
    print("\nTry the app with: streamlit run app.py")
    print("Note: demo dataset is small; do not interpret its score as real-world performance.")

if __name__ == "__main__":
    main()
