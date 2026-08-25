from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document

from insert_chapter4_statistical_results import find_para, insert_after, picture_block, table_block


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new.docx"
TEMP = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new_with_4.6.docx"
ASSET_DIR = ROOT / "docs" / "thesisi version" / "chapter4_inserted_assets"


def nas_figure() -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "figure_4_14_nas_architecture.png"
    dims = [256, 128, 256, 64]
    fig, ax = plt.subplots(figsize=(8.6, 4.5), dpi=180)
    bars = ax.bar(["Layer 1", "Layer 2", "Layer 3", "Layer 4"], dims, color=["#2563eb", "#0f766e", "#7c3aed", "#d97706"])
    ax.set_ylabel("Hidden units")
    ax.set_xlabel("Selected hidden layer")
    ax.set_ylim(0, 300)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def causal_figure() -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "figure_4_15_causal_driver_correlations.png"
    labels = ["week", "trend", "year", "sales_lag12", "sales_lag4", "sales_lag1"]
    values = [0.2255283, 0.2255283, 0.2255132, 0.2135744, 0.1202895, 0.0710116]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=180)
    bars = ax.barh(labels[::-1], values[::-1], color="#0f766e")
    ax.set_xlabel("Correlation with C1 sales")
    ax.set_xlim(0, 0.26)
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values[::-1]], padding=3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    document = Document(DOC)
    marker = "Table 4.11: Live API Validation of Advanced AI Modules"
    if any(marker in p.text for p in document.paragraphs):
        raise RuntimeError("Section 4.6 results have already been inserted")

    validation_rows = [
        ["Meta-learning status", "GET /api/meta-learning/status", "Successful", "Initialized; MAML not currently trained; C1-C8 available"],
        ["Neural Architecture Search", "POST /api/nas/search", "Successful", "C1; 1 generation; 8 architectures evaluated"],
        ["Federated learning", "POST /api/federated/train", "Successful", "C1; 2 clients; 1 FedAvg round"],
        ["Causal discovery", "POST /api/causal/discovery", "Successful", "C1; 5 maximum lags; 505 observations"],
    ]
    nas_rows = [
        ["Hidden layers", "4"],
        ["Hidden dimensions", "256, 128, 256, 64"],
        ["Dropout rates", "0.4, 0.3, 0.2, 0.2"],
        ["Activation", "tanh"],
        ["Learning rate", "0.001"],
        ["Batch size", "64"],
        ["Sequence length", "14"],
        ["Architectures evaluated", "8"],
        ["Validation loss", "0.0742"],
        ["Normalized-scale MAE", "0.2405"],
        ["Normalized-scale RMSE", "0.2860"],
    ]
    causal_rows = [
        ["week", "0.2255", "Positive", "Weak"],
        ["trend", "0.2255", "Positive", "Weak"],
        ["year", "0.2255", "Positive", "Weak"],
        ["sales_lag12", "0.2136", "Positive", "Weak"],
        ["sales_lag4", "0.1203", "Positive", "Weak"],
        ["sales_lag1", "0.0710", "Positive", "Weak"],
    ]

    validation_anchor = find_para(document, "The advanced AI modules were tested using backend endpoints.")
    nas_anchor = find_para(document, "In this project, the NAS module was tested using drug category data such as C1.")
    causal_anchor = find_para(document, "The causal inference experiment helps the system move beyond simple prediction.")

    insert_after(validation_anchor, table_block(
        document,
        ["Module", "Endpoint", "Result", "Verified output"],
        validation_rows,
        marker,
    ))
    insert_after(nas_anchor, table_block(
        document,
        ["NAS Output", "Verified C1 Result"],
        nas_rows,
        "Table 4.12: Best C1 Architecture Returned by the NAS Endpoint",
    ) + picture_block(document, nas_figure(), "Figure 4.14: Hidden-Layer Dimensions of the Best C1 NAS Architecture"))
    insert_after(causal_anchor, table_block(
        document,
        ["Candidate Driver", "Correlation", "Direction", "Reported Strength"],
        causal_rows,
        "Table 4.13: Leading C1 Candidate Drivers Returned by Causal Discovery",
    ) + picture_block(document, causal_figure(), "Figure 4.15: Correlations of Leading C1 Candidate Sales Drivers"))
    document.save(TEMP)
    print(TEMP)


if __name__ == "__main__":
    main()
