from __future__ import annotations

from pathlib import Path
from shutil import copy2

from docx import Document
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "CSCI43018 Project Diary_2023 2024.docx"
OUT = ROOT / "Project Diaries - 7 Supervisor Meetings"


MEETINGS = [
    {
        "previous": "No previous tasks. This was the initial project meeting.",
        "completed": "The initial research idea and expected project scope were prepared before the meeting.",
        "not_completed": "Not applicable.",
        "discussion": "Discussed the research problem, project objectives, available pharmaceutical sales data, and a suitable forecasting-based solution.",
        "targets": "Refine the project scope, review related research, and organize the available sales dataset.",
    },
    {
        "previous": "Review related work and organize the pharmaceutical sales dataset.",
        "completed": "Relevant forecasting studies were reviewed, and the weekly sales data was separated into categories C1 to C8.",
        "not_completed": "Some external regional information was not available in the collected dataset.",
        "discussion": "Reviewed the dataset structure, missing values, date format, and the need to preserve chronological order during model evaluation.",
        "targets": "Complete preprocessing, examine category-wise sales patterns, and prepare the train-test split.",
    },
    {
        "previous": "Complete preprocessing and prepare the time-series data for modelling.",
        "completed": "Dates were parsed, category files were checked, and chronological training and testing datasets were prepared.",
        "not_completed": "No major incomplete task was reported.",
        "discussion": "Discussed lag features, sequential windows, normalization, and the evaluation measures MAE, RMSE, and MAPE.",
        "targets": "Implement the statistical and machine-learning baseline models and record their results.",
    },
    {
        "previous": "Implement baseline statistical and machine-learning forecasting models.",
        "completed": "SARIMAX, Prophet, XGBoost, and LightGBM were evaluated for the eight drug categories.",
        "not_completed": "Further comparison with deep-learning models was still required.",
        "discussion": "Compared initial model behaviour and discussed why model performance differs between high- and low-sales categories.",
        "targets": "Complete LSTM, GRU, and Transformer experiments and compare all model groups.",
    },
    {
        "previous": "Complete the deep-learning experiments and model comparison.",
        "completed": "LSTM, GRU, and Transformer results were evaluated, and category-wise performance values were recorded.",
        "not_completed": "Explainability and system integration testing were still in progress.",
        "discussion": "Reviewed the main results. LightGBM showed the best overall MAE and RMSE, while different models performed best for individual categories.",
        "targets": "Integrate the forecasting models with the web application and complete explainability support.",
    },
    {
        "previous": "Integrate forecasting and explainability functions with the web system.",
        "completed": "The frontend, API gateway, forecast workflow, and fallback lag-importance explanation were tested successfully.",
        "not_completed": "Some advanced modules required small operational tests rather than long experiments.",
        "discussion": "Demonstrated the forecasting interface and discussed endpoint testing, model loading, explanation output, and system monitoring.",
        "targets": "Complete system testing, prepare result figures and tables, and draft the Results and Analysis chapter.",
    },
    {
        "previous": "Complete system testing and prepare the thesis results chapter.",
        "completed": "Model tables, comparison figures, endpoint evidence, and the main thesis discussion were prepared and checked against saved project results.",
        "not_completed": "Minor formatting and reference corrections remained for the final document.",
        "discussion": "Reviewed the full thesis structure, result interpretation, figure and table numbering, limitations, references, and final presentation requirements.",
        "targets": "Apply the final corrections, update the lists of figures and tables, proofread the thesis, and prepare the final submission.",
    },
    {
        "previous": "Review the research background and improve the connection between the problem and the proposed system.",
        "completed": "The background was revised to explain pharmaceutical demand uncertainty and the need for data-driven forecasting.",
        "not_completed": "Some supporting references still required final verification.",
        "discussion": "Reviewed the healthcare context, the practical importance of medicine availability, and how the project addresses forecasting limitations.",
        "targets": "Verify the background references, improve the research gap, and connect the objectives clearly to the implementation.",
    },
    {
        "previous": "Verify references and strengthen the research gap and objectives.",
        "completed": "The main citations were checked and the relationship between the research gap, objectives, methodology, and results was reviewed.",
        "not_completed": "A small number of reference-formatting issues remained.",
        "discussion": "Discussed citation order, the relevance of related work, and the need to avoid unsupported statements in the thesis.",
        "targets": "Correct the reference list, complete final formatting, and check all figures and tables.",
    },
    {
        "previous": "Complete final formatting and perform a full thesis review.",
        "completed": "Figures, tables, captions, lists, model values, and chapter structure were checked against the final project outputs.",
        "not_completed": "Only final proofreading and submission preparation remained.",
        "discussion": "Reviewed the completed thesis, final system evidence, limitations, conclusion, and documents required for submission.",
        "targets": "Complete the final proofread, prepare the presentation and project files, and submit the thesis.",
    },
]

DATES = [
    "03/07/2025", "08/07/2025", "28/07/2025", "07/08/2025", "18/08/2025",
    "24/02/2026", "20/05/2026", "28/05/2026", "10/06/2026", "17/06/2026",
]

DISCUSSION_POINTS = [
    ["Research background and the need for drug-sales forecasting.", "Initial research problem, aim, and objectives.", "Availability and limitations of the pharmaceutical sales dataset."],
    ["Related studies on healthcare AI and time-series forecasting.", "Sri Lankan pharmaceutical supply and inventory background.", "How to narrow the research scope to C1-C8 sales categories."],
    ["Dataset structure, date range, and category-wise sales behaviour.", "Chronological data splitting and prevention of data leakage.", "Suitable evaluation measures for forecasting."],
    ["Statistical and machine-learning baseline models.", "Lag features and category-specific forecasting behaviour.", "How the methodology connects to the research objectives."],
    ["Deep-learning models for sequential sales patterns.", "Comparison of LSTM, GRU, and Transformer approaches.", "Interpretation of MAE, RMSE, and unstable MAPE values."],
    ["Forecasting-system architecture and API integration.", "Explainability using SHAP or fallback lag importance.", "Practical relevance for pharmacy stock planning."],
    ["Model results and selection of the best forecasting approach.", "Category-wise differences in model performance.", "Presentation of correct tables and figures in Chapter 4."],
    ["Research background and healthcare significance.", "Relationship between the research gap and implemented solution.", "Dataset and area-level limitations that must be stated clearly."],
    ["Literature Review citations and reference ordering.", "Consistency among objectives, methodology, results, and discussion.", "Final figure, table, and chapter formatting."],
    ["Overall thesis quality and final project evidence.", "Limitations, recommendations, and conclusion.", "Final proofreading, presentation, and submission preparation."],
]

SELF_DIARIES = [
    [
        "Reviewed possible research areas and selected pharmaceutical sales forecasting as the project topic.",
        "Prepared the initial problem statement and identified the main project objectives.",
        "Checked the available weekly drug sales dataset and its category structure.",
        "Read introductory research on time-series forecasting and healthcare inventory planning.",
        "Refined the project scope based on the supervisor's comments.",
    ],
    [
        "Collected and reviewed research papers related to drug sales and demand forecasting.",
        "Organized the dataset into drug categories C1 to C8.",
        "Checked the date range, sales columns, missing values, and basic data quality.",
        "Summarized the main methods used in related forecasting studies.",
        "Updated the literature review and dataset description.",
    ],
    [
        "Parsed the date column and preserved the chronological order of the sales records.",
        "Prepared category-wise training and testing data without random shuffling.",
        "Created lag features for machine-learning models.",
        "Prepared sequential windows and scaling steps for deep-learning models.",
        "Checked MAE, RMSE, and MAPE calculations using sample outputs.",
    ],
    [
        "Implemented and checked SARIMAX forecasting for the drug categories.",
        "Tested Prophet using the required date and sales column structure.",
        "Prepared lag-based inputs and evaluated the XGBoost model.",
        "Evaluated LightGBM and saved the category-wise model artifacts.",
        "Recorded and compared the statistical and machine-learning results.",
    ],
    [
        "Prepared normalized sequence data for LSTM and GRU models.",
        "Evaluated LSTM forecasting results for categories C1 to C8.",
        "Evaluated GRU and compared its accuracy and inference time with LSTM.",
        "Tested the Transformer model and recorded its evaluation values.",
        "Compared all main models and identified the best model for each category.",
    ],
    [
        "Connected the forecasting functions to the API gateway and frontend workflow.",
        "Tested historical lookup and future forecast requests through the API.",
        "Checked the explainability endpoint and generated fallback lag-importance outputs.",
        "Tested selected advanced-AI and system health endpoints.",
        "Reviewed the web interface and corrected integration issues found during testing.",
    ],
    [
        "Prepared the final model-performance tables and comparison figures.",
        "Completed the Results and Analysis chapter using the saved project values.",
        "Reviewed the Discussion, limitations, and conclusion sections.",
        "Corrected figure and table numbering and updated their front-matter lists.",
        "Proofread the thesis, checked references, and prepared the final submission files.",
    ],
    [
        "Reviewed the research background on pharmaceutical demand and medicine availability.",
        "Improved the connection between the practical problem and forecasting solution.",
        "Rechecked the research gap and expected contribution.",
        "Added clearer discussion of dataset and area-level limitations.",
        "Reviewed the updated background with the project objectives.",
    ],
    [
        "Checked the order of citations used in the Literature Review.",
        "Verified the main publication links and DOI details.",
        "Corrected repeated or missing reference entries.",
        "Reviewed figure and table captions throughout the thesis.",
        "Checked consistency among the methodology, results, and discussion.",
    ],
    [
        "Performed a final review of all thesis chapters.",
        "Checked the final forecasting values against project result files.",
        "Reviewed limitations, recommendations, and the conclusion.",
        "Organized the source code, models, thesis, and diary files.",
        "Prepared the final submission and presentation materials.",
    ],
]


def set_cell(cell, text: str) -> None:
    cell.text = text
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(11)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for number, meeting in enumerate(MEETINGS, 1):
        destination = OUT / f"Supervisor Meeting Diary {number}.docx"
        copy2(TEMPLATE, destination)
        document = Document(destination)

        for paragraph in document.paragraphs:
            if paragraph.text.strip().startswith("Project Title:"):
                paragraph.text = "Project Title: Intelligent Drug Sales Forecasting and Healthcare -Insights with Artificial Intelligence"
            elif paragraph.text.strip().startswith("Name of the Supervisor:"):
                paragraph.text = "Name of the Supervisor: Professor N G J Dias"
            elif paragraph.text.strip().startswith("Student Name:"):
                paragraph.text = "Student Name: B R G Lakmal"
            elif paragraph.text.strip().startswith("Student Index Number:"):
                paragraph.text = "Student Index Number: CS_2020_015"
            elif paragraph.text.strip().startswith("Project Title:"):
                paragraph.text = "Project Title: Intelligent Drug Sales Forecasting and Healthcare -Insights with Artificial Intelligence"
            elif paragraph.text.strip().startswith("Month:"):
                paragraph.text = "Month: ........................                         Index Number: CS_2020_015"
            if paragraph.text.strip().startswith("Meeting No:"):
                paragraph.text = f"Meeting No: {number}                         Meeting Date: {DATES[number - 1]}"
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)

        table = document.tables[0]
        set_cell(table.cell(1, 0), meeting["previous"])
        set_cell(table.cell(1, 1), meeting["completed"])
        set_cell(table.cell(1, 2), meeting["not_completed"])
        set_cell(table.cell(2, 0), "Points/tasks/issues discussed at the current meeting")
        discussion = "\n".join(f"{i}. {point}" for i, point in enumerate(DISCUSSION_POINTS[number - 1], 1))
        set_cell(table.cell(2, 1), discussion)
        set_cell(table.cell(3, 0), "Targets/tasks assigned to complete before the next meeting")
        set_cell(table.cell(3, 1), meeting["targets"])

        self_table = document.tables[1]
        week_rows = [1, 9, 17, 25, 33]
        for row_index, description in zip(week_rows, SELF_DIARIES[number - 1]):
            set_cell(self_table.cell(row_index, 1), "........................")
            set_cell(self_table.cell(row_index, 2), description)

        document.save(destination)
        print(destination.name)


if __name__ == "__main__":
    main()
