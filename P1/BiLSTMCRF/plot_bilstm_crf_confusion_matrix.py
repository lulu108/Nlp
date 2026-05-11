# -*- coding: utf-8 -*-
"""
Generate a BMES confusion matrix figure for BiLSTM-CRF.

Usage:
    python P1/BiLSTMCRF/plot_confusion_matrix.py

Input:
    P1/BiLSTMCRF/output/confusion_matrix.tsv

Output:
    P1/BiLSTMCRF/output/confusion_matrix.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    input_path: Path,
    output_path: Path,
    title: str = "BiLSTM-CRF BMES Confusion Matrix",
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find confusion matrix file: {input_path}")

    df = pd.read_csv(input_path, sep="\t", index_col=0)
    labels = list(df.columns)
    matrix = df.values.astype(int)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(title, fontsize=15, pad=15)

    row_sums = matrix.sum(axis=1, keepdims=True)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pct = matrix[i, j] / row_sums[i, 0] * 100 if row_sums[i, 0] else 0
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix figure to: {output_path}")


def main() -> None:
    input_path = Path("P1/BiLSTMCRF/output/confusion_matrix.tsv")
    output_path = Path("P1/BiLSTMCRF/output/confusion_matrix.png")
    plot_confusion_matrix(input_path, output_path)


if __name__ == "__main__":
    main()
