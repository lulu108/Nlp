from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = ROOT_DIR.parent / "P3" / "data" / "thucnews_4class_200_clean.csv"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "train" / "classify_train.csv"

LABEL_ID_TO_NAME = {
    "0": "体育",
    "1": "科技",
    "2": "财经",
    "3": "教育",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare classifier training CSV for P4")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input CSV path from P3")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path for P4")
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def load_csv_auto(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as err:  # noqa: BLE001
            last_err = err
    raise ValueError(f"Failed to read CSV: {path}, last error: {last_err}")


def choose_label_series(df: pd.DataFrame) -> pd.Series:
    if "category" in df.columns:
        return df["category"].astype(str).map(normalize_text)

    if "label" not in df.columns:
        raise ValueError("Input CSV must include 'category' or 'label' column")

    raw = df["label"].astype(str).map(normalize_text)
    converted = raw.map(lambda x: LABEL_ID_TO_NAME.get(x, x))
    return converted


def build_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Input CSV must include 'text' column")

    label_series = choose_label_series(df)
    text_series = df["text"].astype(str).map(normalize_text)

    has_title = "title" in df.columns and df["title"].astype(str).map(normalize_text).ne("").any()

    if has_title:
        title_series = df["title"].astype(str).map(normalize_text)
        out = pd.DataFrame(
            {
                "title": title_series,
                "text": text_series,
                "label": label_series,
            }
        )
    else:
        out = pd.DataFrame(
            {
                "text": text_series,
                "label": label_series,
            }
        )

    out = out[(out["text"] != "") & (out["label"] != "")].copy()

    dedup_cols = ["text", "label"]
    if "title" in out.columns:
        dedup_cols = ["title", "text", "label"]
    out = out.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    return out


def imbalance_message(label_counts: pd.Series) -> str:
    if label_counts.empty:
        return "No label distribution available"

    min_count = int(label_counts.min())
    max_count = int(label_counts.max())
    ratio = max_count / min_count if min_count > 0 else float("inf")

    if ratio <= 1.2:
        level = "无明显类别不平衡"
    elif ratio <= 1.5:
        level = "存在轻微类别不平衡"
    else:
        level = "存在明显类别不平衡"

    return f"{level}（max/min={ratio:.2f}）"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = load_csv_auto(input_path)
    out_df = build_output_dataframe(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    label_counts = out_df["label"].value_counts().sort_index()

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows after cleaning: {len(out_df)}")
    print("Label counts:")
    for label, count in label_counts.items():
        print(f"  {label}: {int(count)}")
    print(imbalance_message(label_counts))


if __name__ == "__main__":
    main()
