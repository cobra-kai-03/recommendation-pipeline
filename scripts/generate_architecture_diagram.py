"""Generate the pipeline architecture diagram as a PNG (reports/architecture_diagram.png).

Run: python scripts/generate_architecture_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "architecture_diagram.png"

STAGES = [
    ("Data\nIngestion", "MovieLens 100K\n/ synthetic\n-> data/raw"),
    ("Data\nValidation", "6 checks +\nerror quarantine\n-> data/errors"),
    ("Data\nPreparation", "clean + standardize\ntime-based split\n-> data/processed"),
    ("Data\nTransformation", "activity summary\nitem popularity\nuser-item matrix"),
    ("Feature\nManagement", "user/item/interaction\nfeatures + catalog\n-> data/features"),
    ("Model\nTraining", "Popularity +\nMatrixFactorization\n+ MLflow"),
    ("Model\nEvaluation", "P@K, R@K, NDCG\ncoverage, diversity\nRMSE/MAE"),
]

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2",
          "#E45756", "#72B7B2", "#EECA3B"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_xlim(0, len(STAGES) * 2.3)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_w, box_h, gap = 1.9, 2.4, 0.4
    y = 2.2
    centers = []
    for idx, ((title, detail), color) in enumerate(zip(STAGES, COLORS)):
        x = idx * (box_w + gap) + 0.3
        rect = mpatches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.5, edgecolor="#222", facecolor=color, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + box_h - 0.45, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(x + box_w / 2, y + box_h / 2 - 0.35, detail, ha="center", va="center",
                fontsize=8.2, color="white")
        centers.append((x + box_w, x, y + box_h / 2))

    # Arrows between consecutive stages.
    for i in range(len(STAGES) - 1):
        x_end = centers[i][0]
        x_start_next = centers[i + 1][1]
        ymid = centers[i][2]
        ax.annotate("", xy=(x_start_next, ymid), xytext=(x_end, ymid),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))

    # Orchestration band underneath.
    band_x = 0.3
    band_w = len(STAGES) * (box_w + gap) - gap
    band = mpatches.FancyBboxPatch(
        (band_x, 0.4), band_w, 1.0,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.3, edgecolor="#222", facecolor="#D3D3D3", alpha=0.7,
    )
    ax.add_patch(band)
    ax.text(band_x + band_w / 2, 0.9,
            "Orchestration — pure-Python runner (pipeline.py)  •  Apache Airflow DAG (dags/)  "
            "•  MLflow experiment tracking",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#222")

    ax.text(band_x, 5.5,
            "End-to-End Data Management Pipeline for a Recommendation System",
            fontsize=15, fontweight="bold", color="#222")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Architecture diagram written to {OUT}")


if __name__ == "__main__":
    main()
