from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document

from insert_chapter4_statistical_results import (
    caption,
    find_para,
    format_table,
    insert_after,
    picture_block,
    table_block,
)


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new.docx"
OUTPUT_DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new_with_4.2.docx"
METRICS = ROOT / "src" / "evaluation_results" / "model_metrics.json"
TIMES = ROOT / "src" / "evaluation_results" / "inference_times.json"
IMAGE_DIR = ROOT / "src" / "static" / "images"
OUT_DIR = ROOT / "docs" / "thesisi version" / "chapter4_inserted_assets"


def make_mae_chart(metrics: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "figure_4_6_machine_learning_mae_comparison.png"
    cats = list(metrics)
    xgb = [metrics[c]["xgboost"]["MAE"] for c in cats]
    lgb = [metrics[c]["lightgbm"]["MAE"] for c in cats]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=180)
    width = 0.38
    ax.bar(x - width / 2, xgb, width, label="XGBoost", color="#d97706")
    ax.bar(x + width / 2, lgb, width, label="LightGBM", color="#16a34a")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_xlabel("Drug category")
    ax.set_xticks(x, cats)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    times = json.loads(TIMES.read_text(encoding="utf-8"))
    document = Document(DOC)
    marker = "Table 4.3: Category-Wise Performance of XGBoost and LightGBM"
    if any(marker in p.text for p in document.paragraphs):
        raise RuntimeError("Section 4.2 results have already been inserted")

    cats = list(metrics)
    category_rows = []
    for c in cats:
        x, l = metrics[c]["xgboost"], metrics[c]["lightgbm"]
        category_rows.append([
            c, f'{x["MAE"]:.4f}', f'{x["RMSE"]:.4f}', f'{x["MAPE"]:.4f}',
            f'{l["MAE"]:.4f}', f'{l["RMSE"]:.4f}', f'{l["MAPE"]:.4f}',
        ])

    avg_rows = []
    for model, name in (("xgboost", "XGBoost"), ("lightgbm", "LightGBM")):
        avg_rows.append([
            name,
            f'{np.mean([metrics[c][model]["MAE"] for c in cats]):.4f}',
            f'{np.mean([metrics[c][model]["RMSE"] for c in cats]):.4f}',
            f'{np.mean([metrics[c][model]["MAPE"] for c in cats]):.4f}',
            f'{np.mean([times[c][model] for c in cats]):.4f}',
        ])

    setup_anchor = find_para(document, "The machine learning models were trained and saved as model artifacts.")
    xgb_anchor = find_para(document, "One of the main advantages of XGBoost in this project was its speed.")
    lgb_anchor = find_para(document, "LightGBM performed especially well because it could learn non-linear relationships")
    summary_anchor = find_para(document, "LightGBM 23.3728 28.9163 73.9233 0.1074 s")

    insert_after(setup_anchor, table_block(
        document,
        ["Category", "XGBoost MAE", "XGBoost RMSE", "XGBoost MAPE (%)", "LightGBM MAE", "LightGBM RMSE", "LightGBM MAPE (%)"],
        category_rows,
        marker,
    ))
    insert_after(xgb_anchor, picture_block(document, IMAGE_DIR / "C1_xgboost_forecast.png", "Figure 4.4: C1 Drug Sales Forecast Using the XGBoost Model"))
    insert_after(lgb_anchor, picture_block(document, IMAGE_DIR / "C1_lightgbm_forecast.png", "Figure 4.5: C1 Drug Sales Forecast Using the LightGBM Model"))

    average_table = table_block(
        document,
        ["Model", "Average MAE", "Average RMSE", "Average MAPE (%)", "Average Inference Time (s)"],
        avg_rows,
        "Table 4.4: Average Performance Comparison of Machine Learning Models",
    )
    chart = make_mae_chart(metrics)
    insert_after(summary_anchor, average_table + picture_block(document, chart, "Figure 4.6: Category-Wise MAE Comparison of XGBoost and LightGBM"))
    document.save(OUTPUT_DOC)
    print(OUTPUT_DOC)


if __name__ == "__main__":
    main()
