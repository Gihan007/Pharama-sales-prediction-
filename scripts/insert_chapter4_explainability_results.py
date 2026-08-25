from __future__ import annotations

from pathlib import Path

from docx import Document

from insert_chapter4_statistical_results import find_para, insert_after, picture_block, table_block


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new.docx"
TEMP = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new_with_4.5.docx"
IMAGE_DIR = ROOT / "services" / "frontend_service" / "app" / "static" / "images" / "shap"


def main() -> None:
    document = Document(DOC)
    marker = "Table 4.10: C1 XGBoost Fallback Lag-Importance Results"
    if any(marker in p.text for p in document.paragraphs):
        raise RuntimeError("Section 4.5 results have already been inserted")

    # Values returned by POST /api/explainability for C1 and XGBoost on 2026-07-12.
    rows = [
        ["1", "sales_lag_4", "0.2664", "26.64"],
        ["2", "sales_lag_5", "0.2088", "20.88"],
        ["3", "sales_lag_3", "0.1942", "19.42"],
        ["4", "sales_lag_2", "0.1853", "18.53"],
        ["5", "sales_lag_1", "0.1452", "14.52"],
    ]
    anchor = find_para(document, "The fallback method also generates visual outputs such as feature importance plots")
    elements = table_block(
        document,
        ["Rank", "Lag Feature", "Importance Score", "Normalized Importance (%)"],
        rows,
        marker,
    )
    elements += picture_block(
        document,
        IMAGE_DIR / "fallback_importance_C1_xgboost_20260712_041004.png",
        "Figure 4.12: C1 XGBoost Fallback Lag-Feature Importance",
    )
    elements += picture_block(
        document,
        IMAGE_DIR / "fallback_contribution_C1_xgboost_20260712_041004.png",
        "Figure 4.13: C1 XGBoost Fallback Lag-Contribution Analysis",
    )
    insert_after(anchor, elements)
    document.save(TEMP)
    print(TEMP)


if __name__ == "__main__":
    main()
