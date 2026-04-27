from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "train" / "classify_train.csv"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "train" / "cluster_train_sports_edu_fin.csv"
TARGET_LABELS = ["体育", "教育", "财经"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a clustering subset from classify_train.csv")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="Input CSV path")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path")
    return parser.parse_args()


def load_csv_auto(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as err:  # noqa: BLE001
            last_err = err
    raise ValueError(f"failed to read CSV: {path}, last error: {last_err}")


def build_subset_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "text" not in dataframe.columns:
        raise ValueError("input CSV must include 'text' column")
    if "label" not in dataframe.columns:
        raise ValueError("input CSV must include 'label' column")

    columns = ["text", "label"]
    if "title" in dataframe.columns:
        columns = ["title", "text", "label"]

    subset = dataframe.loc[dataframe["label"].astype(str).isin(TARGET_LABELS), columns].copy()
    if subset.empty:
        raise ValueError("no rows matched the target labels")

    return subset.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    dataframe = load_csv_auto(input_path)
    subset = build_subset_dataframe(dataframe)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False, encoding="utf-8-sig")

    counts = subset["label"].astype(str).value_counts()
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(subset)}")
    print("Label counts:")
    for label in TARGET_LABELS:
        print(f"  {label}: {int(counts.get(label, 0))}")


if __name__ == "__main__":
    main()
