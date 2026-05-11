# -*- coding: utf-8 -*-
"""
Generate Macro F1 comparison chart for P1 sequence labeling experiments.

Recommended location:
    P1/scripts/plot_macro_f1_comparison.py

Output:
    P1/reports/macro_f1_comparison.png
"""

from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    models = ["HMM\n(BiLSTMCRF split)", "BiLSTM-CRF", "Java HMM"]
    macro_f1 = [0.7458, 0.8057, 0.6570]

    output_path = Path("P1/reports/macro_f1_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(models, macro_f1)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Macro F1 Comparison of HMM, BiLSTM-CRF and Java HMM", fontsize=14, pad=12)

    for bar, value in zip(bars, macro_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
