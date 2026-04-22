from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.classifier_preprocessing import build_classifier_text

DEFAULT_DATA_PATH = ROOT_DIR / "data" / "train" / "classify_train.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "classifier"
DEFAULT_VECTORIZER_PATH = DEFAULT_MODEL_DIR / "tfidf_vectorizer.pkl"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "svm_model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVM text classifier for P4")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="CSV path with text,label columns")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Output directory for model artifacts")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args()

def load_training_dataframe(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. "
            "Please prepare data/train/classify_train.csv. See data/train/README.md for format."
        )

    df = pd.read_csv(data_path, encoding="utf-8-sig")

    if "label" not in df.columns:
        raise ValueError("Training data must include 'label' column")

    if "text" not in df.columns:
        raise ValueError("Training data must include 'text' column")

    text_series = df["text"].fillna("").astype(str)
    if "title" in df.columns:
        title_series = df["title"].fillna("").astype(str)
        processed_texts = [build_classifier_text(title, text) for title, text in zip(title_series, text_series)]
    else:
        processed_texts = text_series.map(build_classifier_text)

    out = pd.DataFrame({
        "text": processed_texts,
        "label": df["label"].astype(str).str.strip(),
    })

    out = out[(out["text"] != "") & (out["label"] != "")].copy()

    if out.empty:
        raise ValueError("Training data is empty after cleaning")

    label_counts = out["label"].value_counts()
    if label_counts.shape[0] < 2:
        raise ValueError("At least two classes are required for classification training")
    if label_counts.min() < 2:
        raise ValueError("Each class should have at least 2 samples")

    return out


def choose_cv_folds(labels: pd.Series) -> int:
    min_count = int(labels.value_counts().min())
    return 3 if min_count >= 3 else 2


def train(args: argparse.Namespace) -> tuple[TfidfVectorizer, CalibratedClassifierCV, dict[str, float | str]]:
    data_path = Path(args.data_path)
    df = load_training_dataframe(data_path)
    n_classes = int(df["label"].nunique())

    if len(df) >= 12:
        test_count = max(n_classes, int(round(len(df) * args.test_size)))
        if test_count >= len(df):
            test_count = max(1, len(df) // 4)

        train_df, eval_df = train_test_split(
            df,
            test_size=test_count,
            stratify=df["label"],
            random_state=args.random_state,
        )

        if train_df["label"].value_counts().min() < 2:
            train_df = df
            eval_df = df
    else:
        train_df = df
        eval_df = df

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        token_pattern=r"(?u)\b\w+\b",
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    cv_folds = choose_cv_folds(y_train)
    base_svm = LinearSVC(C=args.svm_c, max_iter=args.max_iter)
    model = CalibratedClassifierCV(base_svm, method="sigmoid", cv=cv_folds)
    model.fit(X_train, y_train)

    X_eval = vectorizer.transform(eval_df["text"])
    y_eval = eval_df["label"]
    y_pred = model.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    report = classification_report(y_eval, y_pred, digits=4)

    metrics = {
        "samples": float(len(df)),
        "classes": float(df["label"].nunique()),
        "eval_accuracy": float(acc),
        "report": report,
    }
    return vectorizer, model, metrics


def save_artifacts(vectorizer: TfidfVectorizer, model: CalibratedClassifierCV, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")
    joblib.dump(model, model_dir / "svm_model.pkl")


def main() -> None:
    args = parse_args()
    vectorizer, model, metrics = train(args)

    model_dir = Path(args.model_dir)
    save_artifacts(vectorizer, model, model_dir)

    print("Training complete")
    print(f"Samples: {int(metrics['samples'])}")
    print(f"Classes: {int(metrics['classes'])}")
    print(f"Eval accuracy: {metrics['eval_accuracy']:.4f}")
    print("Classification report:")
    print(metrics["report"])
    print(f"Saved vectorizer: {model_dir / 'tfidf_vectorizer.pkl'}")
    print(f"Saved model: {model_dir / 'svm_model.pkl'}")


if __name__ == "__main__":
    main()
