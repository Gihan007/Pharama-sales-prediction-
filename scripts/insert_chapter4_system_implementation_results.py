from __future__ import annotations

from pathlib import Path

from docx import Document

from insert_chapter4_statistical_results import find_para, insert_after, picture_block, table_block


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new.docx"
TEMP = ROOT / "docs" / "thesisi version" / "CSCI 43018- final unstructiured draft (1) - current state 2_new_with_4.7.docx"
SCREENSHOT = ROOT / "docs" / "thesisi version" / "chapter4_inserted_assets" / "figure_4_16_forecast_interface.png"


def main() -> None:
    document = Document(DOC)
    marker = "Table 4.14: Live System Endpoint Validation Results"
    if any(marker in p.text for p in document.paragraphs):
        raise RuntimeError("Section 4.7 results have already been inserted")

    rows = [
        ["API gateway health", "GET", "/health", "200", "Pass"],
        ["Frontend home page", "GET", "/", "200", "Pass"],
        ["Forecast workflow", "POST", "/api/forecast", "200", "Pass"],
        ["Explainability workflow", "POST", "/api/explainability", "200", "Pass"],
        ["Meta-learning status", "GET", "/api/meta-learning/status", "200", "Pass"],
        ["Advanced-AI status", "GET", "/api/advanced/status", "200", "Pass"],
    ]
    validation_anchor = find_para(document, "The web workflow was also tested with backend communication.")
    forecast_anchor = find_para(document, "The forecast page testing confirmed that the main prediction workflow works correctly.")
    insert_after(validation_anchor, table_block(
        document,
        ["Test", "Method", "Endpoint", "HTTP Status", "Result"],
        rows,
        marker,
    ))
    insert_after(forecast_anchor, picture_block(
        document,
        SCREENSHOT,
        "Figure 4.16: Live Drug Sales Forecasting Web Interface",
    ))
    document.save(TEMP)
    print(TEMP)


if __name__ == "__main__":
    main()
