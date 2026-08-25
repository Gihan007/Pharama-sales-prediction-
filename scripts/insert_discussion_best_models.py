from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document

from insert_chapter4_statistical_results import find_para, insert_after, picture_block, table_block


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new.docx"
TEMP = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new_with_discussion.docx"
METRICS = ROOT / "src" / "evaluation_results" / "model_metrics.json"
ASSET_DIR = ROOT / "docs" / "thesisi version" / "chapter4_inserted_assets"


DISPLAY = {
    "sarimax": "SARIMAX", "prophet": "Prophet", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "lstm": "LSTM", "gru": "GRU", "transformer": "Transformer",
}
COLORS = {
    "GRU": "#16a34a", "Prophet": "#e377c2", "LightGBM": "#0f766e",
    "LSTM": "#2563eb", "XGBoost": "#d97706",
}


def chart(rows: list[list[str]]) -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "figure_5_1_category_best_models.png"
    categories = [r[0] for r in rows]
    models = [r[1] for r in rows]
    values = [float(r[2]) for r in rows]
    fig, ax = plt.subplots(figsize=(9.4, 5.0), dpi=180)
    bars = ax.bar(categories, values, color=[COLORS[m] for m in models])
    ax.set_xlabel("Drug category")
    ax.set_ylabel("Lowest category MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{m}\n{v:.2f}" for m, v in zip(models, values)], padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    document = Document(DOC)
    marker = "Table 5.1: Best Forecasting Model for Each Drug Category Based on MAE"
    if any(marker in p.text for p in document.paragraphs):
        raise RuntimeError("Discussion results have already been inserted")

    rows = []
    for category, models in metrics.items():
        winner = min(models, key=lambda key: models[key]["MAE"])
        result = models[winner]
        rows.append([
            category, DISPLAY[winner], f'{result["MAE"]:.4f}',
            f'{result["RMSE"]:.4f}', f'{result["MAPE"]:.4f}',
        ])

    anchor = find_para(document, "The results also show that no single model was best for every category.")
    elements = table_block(
        document,
        ["Category", "Best Model", "MAE", "RMSE", "MAPE (%)"],
        rows,
        marker,
    )
    elements += picture_block(document, chart(rows), "Figure 5.1: Category-Wise Best Forecasting Model Based on MAE")
    insert_after(anchor, elements)
    document.save(TEMP)
    print(TEMP)


if __name__ == "__main__":
    main()
