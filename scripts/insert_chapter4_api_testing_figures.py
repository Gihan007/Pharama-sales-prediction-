from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document

from insert_chapter4_statistical_results import find_para, insert_after, picture_block


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new.docx"
TEMP = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new_with_4.8.docx"
ASSET_DIR = ROOT / "docs" / "thesisi version" / "chapter4_inserted_assets"


def result_card(filename: str, title: str, fields: list[tuple[str, str]], status: str = "PASS") -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / filename
    fig, ax = plt.subplots(figsize=(9.2, 4.7), dpi=180)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.03, 0.06), 0.94, 0.88, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5))
    ax.text(0.07, 0.86, title, fontsize=17, fontweight="bold", color="#0f172a", va="center")
    ax.text(0.91, 0.86, status, fontsize=12, fontweight="bold", color="white", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#16a34a", edgecolor="none"))
    y = 0.70
    for label, value in fields:
        ax.text(0.08, y, label, fontsize=11, fontweight="bold", color="#475569", va="center")
        ax.text(0.36, y, value, fontsize=11, color="#0f172a", va="center")
        ax.plot([0.07, 0.93], [y - 0.065, y - 0.065], color="#e2e8f0", linewidth=1)
        y -= 0.12
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    document = Document(DOC)
    marker = "Figure 4.17: Live Historical Forecast Endpoint Test Result"
    if any(marker in p.text for p in document.paragraphs):
        raise RuntimeError("Section 4.8 figures have already been inserted")

    figures = [
        (
            "The forecast endpoint was tested for both historical and future prediction behavior.",
            result_card("figure_4_17_forecast_endpoint.png", "Forecast Endpoint Test", [
                ("Request", "POST /api/forecast"), ("Input", "C1, 2018-01-15, XGBoost"),
                ("HTTP status", "200"), ("Returned value", "28.33"),
                ("Reference date", "2018-01-14"), ("Mode", "Historical Data"),
            ]),
            marker,
        ),
        (
            "This test confirmed that model training can be triggered separately from prediction.",
            result_card("figure_4_18_training_artifact.png", "Training-Service Artifact Verification", [
                ("Training endpoint", "POST /training/run"), ("Model", "XGBoost, category C1"),
                ("Saved artifact", "artifacts/models/models_xgb/C1_xgb.pkl"),
                ("Artifact size", "299,637 bytes"), ("Artifact verified", "Present"),
            ]),
            "Figure 4.18: C1 XGBoost Training-Artifact Verification",
        ),
        (
            "During testing, the explainability endpoint returned successful responses.",
            result_card("figure_4_19_explainability_endpoint.png", "Explainability Endpoint Test", [
                ("Request", "POST /api/explainability"), ("Input", "C1, XGBoost"),
                ("HTTP status", "200"), ("Method returned", "fallback_lag_importance"),
                ("Highest feature", "sales_lag_4"), ("Importance", "0.2664"),
            ]),
            "Figure 4.19: Live Explainability Endpoint Test Result",
        ),
        (
            "Neural architecture search was tested using a small configuration.",
            result_card("figure_4_20_advanced_ai_endpoints.png", "Advanced-AI Endpoint Tests", [
                ("NAS", "Success; C1; 8 architectures"),
                ("Federated learning", "Success; 2 clients; 1 FedAvg round"),
                ("Causal discovery", "Success; C1; 505 observations"),
                ("Meta-learning status", "Initialized; MAML not currently trained"),
            ]),
            "Figure 4.20: Live Advanced-AI Endpoint Test Results",
        ),
        (
            "The tested services included the API gateway, forecast service, frontend service",
            result_card("figure_4_21_health_endpoints.png", "Live Health and Availability Checks", [
                ("API gateway /health", "HTTP 200; healthy"),
                ("Frontend /", "HTTP 200; rendered"),
                ("Meta-learning status", "HTTP 200"),
                ("Advanced-AI status", "HTTP 200"),
                ("Forecast workflow", "HTTP 200"),
                ("Explainability workflow", "HTTP 200"),
            ]),
            "Figure 4.21: Live Health and Availability Endpoint Results",
        ),
        (
            "Final status Successful",
            result_card("figure_4_22_pytest_summary.png", "Current Automated Smoke-Test Run", [
                ("Test suite", "tests/test_services_smoke.py"),
                ("Collected", "4 tests"), ("Passed", "4"), ("Failed", "0"),
                ("Result", "100% passed"),
            ]),
            "Figure 4.22: Current Automated API and Service Smoke-Test Summary",
        ),
    ]
    for anchor_text, image, caption in figures:
        insert_after(find_para(document, anchor_text), picture_block(document, image, caption))
    document.save(TEMP)
    print(TEMP)


if __name__ == "__main__":
    main()
