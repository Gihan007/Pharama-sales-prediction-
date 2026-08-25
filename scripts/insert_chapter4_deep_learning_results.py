from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document

from insert_chapter4_statistical_results import find_para, insert_after, picture_block, table_block


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new.docx"
TEMP = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new_with_4.3.docx"
METRICS = ROOT / "src" / "evaluation_results" / "model_metrics.json"
TIMES = ROOT / "src" / "evaluation_results" / "inference_times.json"
SUMMARY = ROOT / "src" / "evaluation_results" / "performance_summary.json"
IMAGE_DIR = ROOT / "src" / "static" / "images"
ASSET_DIR = ROOT / "docs" / "thesisi version" / "chapter4_inserted_assets"


def make_chart(metrics: dict) -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "figure_4_10_deep_learning_mae_comparison.png"
    categories = list(metrics)
    models = [("lstm", "LSTM", "#2563eb"), ("gru", "GRU", "#16a34a"), ("transformer", "Transformer", "#7c3aed")]
    x = np.arange(len(categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=180)
    for index, (key, label, colour) in enumerate(models):
        values = [metrics[c][key]["MAE"] for c in categories]
        ax.bar(x + (index - 1) * width, values, width, label=label, color=colour)
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
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    times = json.loads(TIMES.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    document = Document(DOC)
    marker = "Table 4.5: Category-Wise MAE of the Main Deep Learning Models"
    if any(marker in p.text for p in document.paragraphs):
        raise RuntimeError("Section 4.3 results have already been inserted")

    categories = list(metrics)
    category_rows = [
        [c, f'{metrics[c]["lstm"]["MAE"]:.4f}', f'{metrics[c]["gru"]["MAE"]:.4f}', f'{metrics[c]["transformer"]["MAE"]:.4f}']
        for c in categories
    ]
    average_rows = []
    for key, name in (("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")):
        average_rows.append([
            name,
            f'{summary[key]["final_mae"]:.4f}',
            f'{summary[key]["final_rmse"]:.4f}',
            f'{summary[key]["final_mape"]:.4f}',
            f'{summary[key]["final_time"]:.4f}',
        ])

    setup_anchor = find_para(document, "The deep learning models were evaluated using MAE, RMSE, MAPE, and inference time.")
    lstm_anchor = find_para(document, "The LSTM model produced strong forecasting results.")
    gru_anchor = find_para(document, "The GRU model achieved an average MAE")
    transformer_anchor = find_para(document, "The Transformer model achieved an average MAE")
    summary_anchor = find_para(document, "Transformer 28.8908 36.5221 53.6691 0.2426 s")

    insert_after(setup_anchor, table_block(
        document,
        ["Category", "LSTM MAE", "GRU MAE", "Transformer MAE"],
        category_rows,
        marker,
    ))
    insert_after(lstm_anchor, picture_block(document, IMAGE_DIR / "C1_lstm_forecast.png", "Figure 4.7: C1 Drug Sales Forecast Using the LSTM Model"))
    insert_after(gru_anchor, picture_block(document, IMAGE_DIR / "C1_gru_forecast.png", "Figure 4.8: C1 Drug Sales Forecast Using the GRU Model"))
    insert_after(transformer_anchor, picture_block(document, IMAGE_DIR / "C1_transformer_forecast.png", "Figure 4.9: C1 Drug Sales Forecast Using the Transformer Model"))

    average_table = table_block(
        document,
        ["Model", "Average MAE", "Average RMSE", "Average MAPE (%)", "Average Inference Time (s)"],
        average_rows,
        "Table 4.6: Average Performance Comparison of the Main Deep Learning Models",
    )
    insert_after(summary_anchor, average_table + picture_block(document, make_chart(metrics), "Figure 4.10: Category-Wise MAE Comparison of LSTM, GRU, and Transformer"))
    document.save(TEMP)
    print(TEMP)


if __name__ == "__main__":
    main()
