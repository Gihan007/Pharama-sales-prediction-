from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document

from insert_chapter4_statistical_results import find_para, insert_after, picture_block, table_block


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new.docx"
TEMP = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new_with_4.4.docx"
RESULTS = ROOT / "src" / "evaluation_results" / "ensemble_results.json"
ASSET_DIR = ROOT / "docs" / "thesisi version" / "chapter4_inserted_assets"


def make_chart(results: dict) -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "figure_4_11_ensemble_mae_comparison.png"
    categories = list(results)
    weighted = [results[c]["weighted_average"]["MAE"] for c in categories]
    performance = [results[c]["performance_weighted"]["MAE"] for c in categories]
    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=180)
    ax.bar(x - width / 2, weighted, width, label="Weighted Average", color="#0f766e")
    ax.bar(x + width / 2, performance, width, label="Performance Weighted", color="#d97706")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_xlabel("Drug category")
    ax.set_xticks(x, categories)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    document = Document(DOC)
    marker = "Table 4.7: Category-Wise Performance of Ensemble Forecasting Methods"
    if any(marker in p.text for p in document.paragraphs):
        raise RuntimeError("Section 4.4 results have already been inserted")

    categories = list(results)
    category_rows = []
    for c in categories:
        w = results[c]["weighted_average"]
        p = results[c]["performance_weighted"]
        category_rows.append([
            c, f'{w["MAE"]:.4f}', f'{w["RMSE"]:.4f}', f'{w["MAPE"]:.4f}',
            f'{p["MAE"]:.4f}', f'{p["RMSE"]:.4f}', f'{p["MAPE"]:.4f}',
        ])

    average_rows = []
    for key, name in (("weighted_average", "Weighted Average"), ("performance_weighted", "Performance Weighted")):
        average_rows.append([
            name,
            f'{np.mean([results[c][key]["MAE"] for c in categories]):.4f}',
            f'{np.mean([results[c][key]["RMSE"] for c in categories]):.4f}',
            f'{np.mean([results[c][key]["MAPE"] for c in categories]):.4f}',
            f'{np.mean([results[c][key]["Time"] for c in categories]):.4f}',
        ])

    experiment_anchor = find_para(document, "The experiments showed that ensemble forecasting produced stable results")
    selected_anchor = find_para(document, "C7 Performance Weighted 29.4016 35.7646 100.7640")

    insert_after(experiment_anchor, table_block(
        document,
        ["Category", "Weighted MAE", "Weighted RMSE", "Weighted MAPE (%)", "Performance MAE", "Performance RMSE", "Performance MAPE (%)"],
        category_rows,
        marker,
    ))
    average_table = table_block(
        document,
        ["Ensemble Method", "Average MAE", "Average RMSE", "Average MAPE (%)", "Average Inference Time (s)"],
        average_rows,
        "Table 4.8: Average Performance Comparison of Ensemble Forecasting Methods",
    )
    insert_after(selected_anchor, average_table + picture_block(document, make_chart(results), "Figure 4.11: Category-Wise MAE Comparison of Ensemble Forecasting Methods"))
    document.save(TEMP)
    print(TEMP)


if __name__ == "__main__":
    main()
