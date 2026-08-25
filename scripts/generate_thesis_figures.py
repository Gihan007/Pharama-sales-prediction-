from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from statistics import mean
from textwrap import wrap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "architecture" / "thesis_figures"
DATA_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "src" / "evaluation_results"


COLORS = {
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#334155",
    "border": "#cbd5e1",
    "bg": "#f8fafc",
    "white": "#ffffff",
    "blue": "#2563eb",
    "blue_l": "#dbeafe",
    "teal": "#0f766e",
    "teal_l": "#ccfbf1",
    "green": "#16a34a",
    "green_l": "#dcfce7",
    "amber": "#d97706",
    "amber_l": "#fef3c7",
    "orange": "#ea580c",
    "orange_l": "#ffedd5",
    "red": "#dc2626",
    "red_l": "#fee2e2",
    "violet": "#7c3aed",
    "violet_l": "#ede9fe",
    "pink": "#db2777",
    "pink_l": "#fce7f3",
    "gray_l": "#f1f5f9",
}


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def header(title: str, subtitle: str, w: int = 1600, h: int = 900) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,8 L11,4 z" fill="#334155"/></marker>',
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.13"/></filter>',
        "<style>",
        ".title{font:700 36px Arial,sans-serif;fill:#0f172a}.subtitle{font:400 18px Arial,sans-serif;fill:#475569}",
        ".h{font:700 20px Arial,sans-serif;fill:#0f172a}.b{font:400 16px Arial,sans-serif;fill:#334155}.s{font:400 14px Arial,sans-serif;fill:#475569}",
        ".metric{font:700 26px Arial,sans-serif;fill:#0f172a}.tiny{font:400 12px Arial,sans-serif;fill:#64748b}",
        ".card{filter:url(#shadow)}.arrow{stroke:#334155;stroke-width:3;fill:none;marker-end:url(#arrow)}.dash{stroke:#64748b;stroke-width:2.5;stroke-dasharray:8 7;fill:none;marker-end:url(#arrow)}",
        "</style>",
        "</defs>",
        f'<rect width="{w}" height="{h}" fill="{COLORS["bg"]}"/>',
        f'<text x="{w/2}" y="58" text-anchor="middle" class="title">{esc(title)}</text>',
        f'<text x="{w/2}" y="90" text-anchor="middle" class="subtitle">{esc(subtitle)}</text>',
    ]


def finish(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def text(x, y, content, cls="b", anchor="start") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{esc(content)}</text>'


def multiline(x, y, content, width=32, cls="b", anchor="start", line_h=22) -> list[str]:
    lines = []
    for i, line in enumerate(wrap(str(content), width=width) or [""]):
        lines.append(text(x, y + i * line_h, line, cls, anchor))
    return lines


def rect(x, y, w, h, fill, stroke, rx=16, cls="card") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2" class="{cls}"/>'


def line(x1, y1, x2, y2, dashed=False) -> str:
    cls = "dash" if dashed else "arrow"
    return f'<path class="{cls}" d="M{x1} {y1} L{x2} {y2}"/>'


def curved(x1, y1, cx1, cy1, cx2, cy2, x2, y2, dashed=False) -> str:
    cls = "dash" if dashed else "arrow"
    return f'<path class="{cls}" d="M{x1} {y1} C{cx1} {cy1} {cx2} {cy2} {x2} {y2}"/>'


def card(parts, x, y, w, h, title, body, fill, stroke, title_y=36, body_y=68, wrap_width=26):
    parts.append(rect(x, y, w, h, fill, stroke))
    parts.append(text(x + w / 2, y + title_y, title, "h", "middle"))
    parts.extend(multiline(x + w / 2, y + body_y, body, wrap_width, "b", "middle", 21))


def load_performance():
    return json.loads((EVAL_DIR / "performance_summary.json").read_text(encoding="utf-8"))


def load_c1_values():
    rows = []
    with (DATA_DIR / "C1.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["datum"], float(row["C1"])))
    return rows


def dataset_stats():
    rows = []
    for idx in range(1, 9):
        cat = f"C{idx}"
        with (DATA_DIR / f"{cat}.csv").open(newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        vals = [float(r[cat]) for r in data]
        rows.append(
            {
                "cat": cat,
                "records": len(data),
                "start": data[0]["datum"],
                "end": data[-1]["datum"],
                "mean": mean(vals),
                "zero": sum(1 for v in vals if v == 0),
            }
        )
    return rows


CATEGORIES = [
    ("C1", "M01AB", "Anti-inflammatory", "Diclofenac, Indomethacin"),
    ("C2", "M01AE", "Anti-inflammatory", "Ibuprofen, Naproxen"),
    ("C3", "N02BA", "Analgesics", "Aspirin"),
    ("C4", "N02BE", "Analgesics", "Metamizole"),
    ("C5", "N05B", "Anxiolytics", "Diazepam, Lorazepam"),
    ("C6", "N05C", "Hypnotics and sedatives", "Zolpidem, Zopiclone"),
    ("C7", "R03", "Obstructive airway drugs", "Salbutamol, Beclometasone"),
    ("C8", "R06", "Antihistamines", "Cetirizine, Loratadine"),
]


def fig01():
    p = header("Overall System Methodology", "End-to-end research workflow implemented for drug sales forecasting")
    steps = [
        ("1", "Data collection", "C1-C8 weekly sales records"),
        ("2", "Preprocessing", "date parsing and clean time-series format"),
        ("3", "Feature engineering", "lags, windows, temporal features"),
        ("4", "Model training", "SARIMAX, Prophet, XGBoost, LightGBM, LSTM, GRU, Transformer"),
        ("5", "Forecasting", "future sales prediction by category"),
        ("6", "Explainability", "SHAP or fallback lag importance"),
        ("7", "Evaluation", "MAE, RMSE, MAPE, inference time"),
        ("8", "Deployment", "web UI, APIs, Docker, Kubernetes"),
    ]
    cx, cy, r = 800, 470, 275
    p.append(f'<circle cx="{cx}" cy="{cy}" r="130" fill="{COLORS["white"]}" stroke="{COLORS["border"]}" stroke-width="2" class="card"/>')
    p.append(text(cx, cy - 12, "AI Drug Sales", "metric", "middle"))
    p.append(text(cx, cy + 22, "Forecasting Framework", "h", "middle"))
    for i, (num, title, body) in enumerate(steps):
        a = -math.pi / 2 + i * (2 * math.pi / len(steps))
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        fill_stroke = [(COLORS["blue_l"], COLORS["blue"]), (COLORS["teal_l"], COLORS["teal"]), (COLORS["green_l"], COLORS["green"]), (COLORS["amber_l"], COLORS["amber"]), (COLORS["orange_l"], COLORS["orange"]), (COLORS["pink_l"], COLORS["pink"]), (COLORS["violet_l"], COLORS["violet"]), (COLORS["red_l"], COLORS["red"])][i]
        p.append(rect(x - 120, y - 58, 240, 116, fill_stroke[0], fill_stroke[1], 18))
        p.append(f'<circle cx="{x-92}" cy="{y-32}" r="18" fill="{fill_stroke[1]}"/>')
        p.append(f'<text x="{x-92}" y="{y-26}" text-anchor="middle" style="font:700 15px Arial;fill:white">{num}</text>')
        p.append(text(x + 8, y - 27, title, "h", "middle"))
        p.extend(multiline(x, y + 2, body, 25, "s", "middle", 17))
        nx, ny = cx + 138 * math.cos(a), cy + 138 * math.sin(a)
        p.append(curved(x - (18 if x > cx else -18), y, (x + nx) / 2, y, (x + nx) / 2, ny, nx, ny, True))
    finish(p, OUT / "01_overall_system_methodology.svg")


def fig02():
    p = header("Proposed Drug Sales Prediction System Workflow", "User request to forecast output, visualization, and explanation")
    lanes = [("User", 150), ("Frontend", 360), ("API Gateway", 590), ("Forecast Service", 840), ("Model Store", 1085), ("Output", 1310)]
    for name, x in lanes:
        p.append(f'<line x1="{x}" y1="150" x2="{x}" y2="770" stroke="#cbd5e1" stroke-width="2" stroke-dasharray="8 8"/>')
        p.append(text(x, 130, name, "h", "middle"))
    events = [
        (150, 210, "Select C1-C8, date, model", COLORS["blue_l"], COLORS["blue"]),
        (360, 285, "Submit forecast form", COLORS["teal_l"], COLORS["teal"]),
        (590, 360, "POST /api/forecast", COLORS["amber_l"], COLORS["amber"]),
        (840, 435, "Historical lookup or future forecast", COLORS["green_l"], COLORS["green"]),
        (1085, 510, "Load model and scaler artifacts", COLORS["violet_l"], COLORS["violet"]),
        (840, 585, "Generate prediction and chart", COLORS["green_l"], COLORS["green"]),
        (1310, 660, "Sales value, plot, model used", COLORS["orange_l"], COLORS["orange"]),
    ]
    for x, y, label, fill, stroke in events:
        p.append(rect(x - 100, y - 36, 200, 72, fill, stroke, 14))
        p.extend(multiline(x, y - 8, label, 22, "b", "middle", 20))
    for (x1, y1, *_), (x2, y2, *__) in zip(events, events[1:]):
        p.append(line(x1 + 100, y1, x2 - 100, y2))
    p.append(curved(1310, 696, 1160, 780, 520, 780, 360, 322, True))
    p.append(text(780, 810, "Returned result is rendered in the web interface with forecast plot and explanation output", "s", "middle"))
    finish(p, OUT / "02_prediction_system_workflow.svg")


def fig03():
    p = header("Drug Category Classification Used in the Study", "C1-C8 pharmaceutical categories used by the forecasting system")
    fills = [COLORS["blue_l"], COLORS["teal_l"], COLORS["green_l"], COLORS["amber_l"], COLORS["orange_l"], COLORS["pink_l"], COLORS["violet_l"], COLORS["red_l"]]
    strokes = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["amber"], COLORS["orange"], COLORS["pink"], COLORS["violet"], COLORS["red"]]
    p.append(f'<rect x="680" y="150" width="240" height="90" rx="24" fill="{COLORS["white"]}" stroke="{COLORS["border"]}" stroke-width="2" class="card"/>')
    p.append(text(800, 185, "Drug Sales", "metric", "middle"))
    p.append(text(800, 215, "Categories", "h", "middle"))
    positions = [(150, 185), (465, 185), (985, 185), (1300, 185), (150, 560), (465, 560), (985, 560), (1300, 560)]
    for i, ((cat, code, group, examples), (x, y)) in enumerate(zip(CATEGORIES, positions)):
        p.append(curved(800, 240 if y < 400 else 240, 800, 380, x, 380, x, y - 75, True))
        p.append(rect(x - 135, y - 72, 270, 144, fills[i], strokes[i], 18))
        p.append(text(x - 105, y - 35, cat, "metric", "start"))
        p.append(text(x + 105, y - 35, code, "h", "end"))
        p.append(text(x, y - 2, group, "h", "middle"))
        p.extend(multiline(x, y + 28, examples, 26, "s", "middle", 18))
    finish(p, OUT / "03_drug_category_classification.svg")


def fig04():
    p = header("Data Preprocessing Pipeline", "Preparation of category-wise sales CSV files before model training")
    steps = [
        ("C1-C8 CSV files", "category-wise raw files", COLORS["blue_l"], COLORS["blue"]),
        ("Read CSV", "load selected category", COLORS["teal_l"], COLORS["teal"]),
        ("Parse datum", "convert date values", COLORS["green_l"], COLORS["green"]),
        ("Set time index", "preserve chronology", COLORS["amber_l"], COLORS["amber"]),
        ("Clean records", "check usable rows", COLORS["orange_l"], COLORS["orange"]),
        ("Select sales target", "category sales column", COLORS["pink_l"], COLORS["pink"]),
        ("Prepared dataset", "model-ready series", COLORS["violet_l"], COLORS["violet"]),
    ]
    x0, y = 70, 390
    for i, (title, body, fill, stroke) in enumerate(steps):
        x = x0 + i * 215
        p.append(rect(x, y, 170, 115, fill, stroke, 16))
        p.append(text(x + 85, y + 38, title, "h", "middle"))
        p.extend(multiline(x + 85, y + 68, body, 18, "s", "middle", 18))
        if i < len(steps) - 1:
            p.append(line(x + 170, y + 58, x + 215, y + 58))
    p.append(text(800, 580, "Implemented in data loading and forecasting utilities using the datum column as the time key", "s", "middle"))
    finish(p, OUT / "04_data_preprocessing_pipeline.svg")


def fig05():
    p = header("Time-Series Feature Engineering Process", "Transforming sales series into lag, sequence, and time-aware model inputs")
    card(p, 80, 205, 245, 120, "Raw Sales Series", "C1-C8 weekly date and sales values", COLORS["blue_l"], COLORS["blue"])
    card(p, 415, 205, 245, 120, "Target Setup", "predict the next sales value", COLORS["amber_l"], COLORS["amber"])
    card(p, 770, 140, 285, 115, "Lag Features", "lag_1 to lag_5 for XGBoost, LightGBM, SHAP", COLORS["pink_l"], COLORS["pink"])
    card(p, 770, 310, 285, 115, "Sequence Windows", "10-step or 30-step windows for neural models", COLORS["green_l"], COLORS["green"])
    card(p, 770, 480, 285, 115, "Time Features", "week, month, quarter, year, trend, seasonal flags", COLORS["violet_l"], COLORS["violet"])
    card(p, 1165, 310, 285, 125, "Model-Ready Inputs", "tabular lag matrix, tensor windows, causal features", COLORS["teal_l"], COLORS["teal"])
    p.append(line(325, 265, 415, 265))
    p.append(curved(660, 265, 710, 210, 730, 200, 770, 198))
    p.append(line(660, 265, 770, 368))
    p.append(curved(660, 265, 710, 430, 720, 520, 770, 538))
    p.append(curved(1055, 198, 1110, 220, 1135, 305, 1165, 350))
    p.append(line(1055, 368, 1165, 368))
    p.append(curved(1055, 538, 1110, 510, 1135, 440, 1165, 390))
    card(p, 305, 530, 300, 115, "Scaling When Required", "MinMaxScaler or StandardScaler before selected ML and DL models", COLORS["gray_l"], COLORS["border"])
    p.append(curved(455, 530, 530, 450, 660, 400, 770, 368, True))
    finish(p, OUT / "05_time_series_feature_engineering_process.svg")


def fig06():
    p = header("Time-Series Train-Test Splitting Method", "Chronological evaluation split used for each C1-C8 category")
    p.append(rect(80, 145, 1440, 110, COLORS["white"], COLORS["border"], 18))
    p.append(text(130, 188, "Dataset: 517 weekly observations per category", "h"))
    p.append(text(130, 220, "Full range: 2014-01-12 to 2023-12-03", "b"))
    p.append(text(1010, 188, "No random shuffle", "h"))
    p.append(text(1010, 220, "Past records predict later records", "b"))
    p.append(f'<rect x="120" y="365" width="1120" height="95" rx="16" fill="{COLORS["blue_l"]}" stroke="{COLORS["blue"]}" stroke-width="3"/>')
    p.append(f'<rect x="1240" y="365" width="220" height="95" rx="16" fill="{COLORS["orange_l"]}" stroke="{COLORS["orange"]}" stroke-width="3"/>')
    p.append(f'<line x1="1240" y1="340" x2="1240" y2="500" stroke="{COLORS["ink"]}" stroke-width="4" stroke-dasharray="10 8"/>')
    p.append(text(680, 405, "Training Period", "metric", "middle"))
    p.append(text(680, 438, "507 records: 2014-01-12 to 2023-09-24", "b", "middle"))
    p.append(text(1350, 405, "Test", "metric", "middle"))
    p.append(text(1350, 438, "10 weeks", "b", "middle"))
    p.append(text(120, 525, "2014-01-12", "s", "middle"))
    p.append(text(1240, 525, "2023-10-01", "s", "middle"))
    p.append(text(1460, 525, "2023-12-03", "s", "middle"))
    card(p, 190, 615, 330, 110, "Evaluation", "MAE, RMSE, MAPE, inference time", COLORS["green_l"], COLORS["green"])
    card(p, 635, 615, 330, 110, "Applied Separately", "same split logic for C1 through C8", COLORS["violet_l"], COLORS["violet"])
    card(p, 1080, 615, 330, 110, "Purpose", "avoid future data leakage", COLORS["red_l"], COLORS["red"])
    finish(p, OUT / "06_time_series_train_test_splitting_method.svg")


def fig07():
    p = header("Forecasting Model Development Workflow", "Comparative model families trained and evaluated in the project")
    card(p, 90, 380, 230, 120, "Prepared Dataset", "category-wise time-series data", COLORS["blue_l"], COLORS["blue"])
    groups = [
        (430, 165, "Statistical", "SARIMAX, Prophet", COLORS["amber_l"], COLORS["amber"]),
        (430, 340, "Machine Learning", "XGBoost, LightGBM with lag features", COLORS["green_l"], COLORS["green"]),
        (430, 515, "Deep Learning", "LSTM, GRU, Transformer, TFT, N-BEATS, Informer", COLORS["violet_l"], COLORS["violet"]),
        (790, 340, "Ensemble", "weighted and performance-weighted combination", COLORS["pink_l"], COLORS["pink"]),
        (1130, 340, "Evaluation", "MAE, RMSE, MAPE, inference time", COLORS["teal_l"], COLORS["teal"]),
    ]
    for x, y, title, body, fill, stroke in groups:
        card(p, x, y, 270, 125, title, body, fill, stroke)
    for _, y, *_ in groups[:3]:
        p.append(line(320, 440, 430, y + 62))
    p.append(line(700, 227, 790, 402))
    p.append(line(700, 402, 790, 402))
    p.append(line(700, 577, 790, 402))
    p.append(line(1060, 402, 1130, 402))
    p.append(text(800, 730, "The workflow compares model families instead of depending on a single forecasting method.", "s", "middle"))
    finish(p, OUT / "07_forecasting_model_development_workflow.svg")


def fig08():
    p = header("Model Training and Artifact Storage Flow", "Category-wise models and scalers saved for reuse during forecasting")
    card(p, 80, 240, 250, 120, "Category Data", "C1.csv to C8.csv from data/raw", COLORS["blue_l"], COLORS["blue"])
    card(p, 430, 240, 260, 120, "Training Service", "train selected model type for selected categories", COLORS["green_l"], COLORS["green"])
    p.append(line(330, 300, 430, 300))
    p.append(rect(820, 145, 640, 550, COLORS["white"], COLORS["border"], 18))
    p.append(text(1140, 190, "Artifact folders", "metric", "middle"))
    folders = [
        "models_xgb: C*_xgb.pkl",
        "models_lightgbm: C*_lightgbm.txt + scaler",
        "models_lstm: C*_lstm.pth + scaler",
        "models_gru: C*_gru.pth + scaler",
        "models_transformer: C*_transformer.pth + scaler",
        "models_prophet: C*_prophet.pkl",
        "models_tft / models_nbeats / models_informer",
    ]
    y = 245
    for i, item in enumerate(folders):
        fill = [COLORS["blue_l"], COLORS["green_l"], COLORS["violet_l"], COLORS["amber_l"], COLORS["teal_l"], COLORS["pink_l"], COLORS["orange_l"]][i]
        p.append(rect(890, y, 500, 48, fill, COLORS["border"], 8, ""))
        p.append(text(915, y + 30, item, "b"))
        y += 60
    p.append(line(690, 300, 820, 300))
    card(p, 430, 500, 260, 120, "Forecasting Service", "loads saved model and matching scaler", COLORS["orange_l"], COLORS["orange"])
    p.append(curved(1140, 695, 980, 760, 620, 710, 560, 620, True))
    finish(p, OUT / "08_model_training_artifact_storage_flow.svg")


def fig09():
    p = header("Forecasting Request-Response Flow", "How a selected category, date, and model becomes a returned forecast")
    actors = [("User", 170), ("Frontend", 390), ("API Gateway", 620), ("Forecast Service", 870), ("Model/Data Store", 1120), ("Frontend Result", 1370)]
    for name, x in actors:
        p.append(text(x, 140, name, "h", "middle"))
        p.append(f'<line x1="{x}" y1="160" x2="{x}" y2="750" stroke="#cbd5e1" stroke-width="2"/>')
    messages = [
        (170, 390, 215, "select category/date/model"),
        (390, 620, 285, "POST /api/forecast"),
        (620, 870, 355, "route request"),
        (870, 1120, 425, "load CSV/model/scaler"),
        (1120, 870, 495, "return artifacts/data"),
        (870, 620, 565, "forecast value + plot"),
        (620, 390, 635, "JSON response"),
        (390, 1370, 705, "render chart and value"),
    ]
    for x1, x2, y, msg in messages:
        p.append(line(x1 + (35 if x2 > x1 else -35), y, x2 - (35 if x2 > x1 else -35), y))
        p.append(text((x1 + x2) / 2, y - 12, msg, "s", "middle"))
    p.append(rect(810, 475, 120, 60, COLORS["green_l"], COLORS["green"], 8))
    p.append(text(870, 512, "predict", "h", "middle"))
    p.append(rect(810, 250, 120, 60, COLORS["amber_l"], COLORS["amber"], 8))
    p.append(text(870, 286, "validate", "h", "middle"))
    finish(p, OUT / "09_forecasting_request_response_flow.svg")


def fig10():
    p = header("Explainability Workflow for Forecasting Outputs", "SHAP and fallback feature importance used to interpret lag contributions")
    card(p, 90, 350, 250, 120, "Forecast Output", "category, model, predicted sales value", COLORS["blue_l"], COLORS["blue"])
    card(p, 450, 220, 280, 120, "SHAP Available?", "attempt model explainability for XGBoost/LightGBM", COLORS["amber_l"], COLORS["amber"])
    card(p, 860, 150, 280, 120, "SHAP Explanation", "feature contribution and summary plots", COLORS["green_l"], COLORS["green"])
    card(p, 860, 430, 280, 120, "Fallback Importance", "model feature importance or correlation-based lag importance", COLORS["pink_l"], COLORS["pink"])
    card(p, 1250, 290, 260, 140, "User Explanation", "lag contribution, important features, reasoning support", COLORS["violet_l"], COLORS["violet"])
    p.append(line(340, 410, 450, 280))
    p.append(curved(730, 280, 780, 210, 815, 210, 860, 210))
    p.append(text(805, 205, "Yes", "s", "middle"))
    p.append(curved(730, 280, 770, 420, 815, 485, 860, 490))
    p.append(text(805, 462, "No", "s", "middle"))
    p.append(curved(1140, 210, 1210, 225, 1220, 305, 1250, 330))
    p.append(curved(1140, 490, 1210, 470, 1220, 405, 1250, 370))
    finish(p, OUT / "10_explainability_workflow.svg")


def fig11():
    p = header("Advanced AI Module Workflow", "Research-oriented extensions included beyond baseline forecasting")
    modules = [
        (130, 190, "Meta-Learning", "transfer patterns across drug categories; few-shot adaptation", COLORS["blue_l"], COLORS["blue"]),
        (920, 190, "Neural Architecture Search", "search model configurations for drug prediction", COLORS["green_l"], COLORS["green"]),
        (130, 515, "Federated Learning", "simulated clients train without sharing raw sales data", COLORS["orange_l"], COLORS["orange"]),
        (920, 515, "Causal Inference", "lag, month, quarter, trend, seasonal indicators", COLORS["violet_l"], COLORS["violet"]),
    ]
    p.append(f'<circle cx="800" cy="450" r="130" fill="{COLORS["white"]}" stroke="{COLORS["border"]}" stroke-width="2" class="card"/>')
    p.append(text(800, 432, "Advanced AI", "metric", "middle"))
    p.append(text(800, 466, "Service", "h", "middle"))
    for x, y, title, body, fill, stroke in modules:
        card(p, x, y, 360, 150, title, body, fill, stroke, wrap_width=34)
        p.append(curved(x + 180, y + 75, 650 if x < 500 else 950, y + 75, 800, 450, 800, 450))
    finish(p, OUT / "11_advanced_ai_module_workflow.svg")


def fig12():
    p = header("Microservice-Based System Architecture", "Frontend, API gateway, forecasting, training, explainability, and advanced AI services")
    card(p, 90, 360, 240, 130, "Frontend Service", "dashboard, forecast, analytics, explainability pages", COLORS["blue_l"], COLORS["blue"])
    card(p, 455, 360, 240, 130, "API Gateway", "single entry point and route handling", COLORS["amber_l"], COLORS["amber"])
    services = [
        (850, 170, "Forecast Service", COLORS["green_l"], COLORS["green"]),
        (850, 330, "Training Service", COLORS["teal_l"], COLORS["teal"]),
        (850, 490, "Explainability Service", COLORS["pink_l"], COLORS["pink"]),
        (850, 650, "Advanced AI Service", COLORS["violet_l"], COLORS["violet"]),
    ]
    for x, y, title, fill, stroke in services:
        card(p, x, y, 260, 105, title, "REST endpoint and project logic", fill, stroke, wrap_width=28)
    card(p, 1240, 240, 260, 135, "Data Store", "data/raw C1-C8 CSV files", COLORS["gray_l"], COLORS["border"])
    card(p, 1240, 505, 260, 135, "Model Store", "artifacts/models folders", COLORS["orange_l"], COLORS["orange"])
    p.append(line(330, 425, 455, 425))
    for _, y, *_ in services:
        p.append(line(695, 425, 850, y + 52))
        p.append(line(1110, y + 52, 1240, 307 if y < 450 else 572))
    finish(p, OUT / "12_microservice_architecture.svg")


def fig13():
    p = header("API Gateway Communication Flow", "Frontend requests routed to backend services through FastAPI endpoints")
    card(p, 70, 365, 230, 125, "Web Frontend", "browser UI sends API calls", COLORS["blue_l"], COLORS["blue"])
    card(p, 475, 365, 250, 125, "API Gateway", "routes requests and serializes responses", COLORS["amber_l"], COLORS["amber"])
    endpoints = [
        (900, 115, "/api/forecast", "forecast service", COLORS["green_l"], COLORS["green"]),
        (900, 245, "/api/explainability", "SHAP/fallback output", COLORS["pink_l"], COLORS["pink"]),
        (900, 375, "/api/meta-learning/*", "MAML/few-shot/transfer", COLORS["blue_l"], COLORS["blue"]),
        (900, 505, "/api/nas/search", "architecture search", COLORS["violet_l"], COLORS["violet"]),
        (900, 635, "/health and /metrics", "service status", COLORS["gray_l"], COLORS["border"]),
    ]
    p.append(line(300, 427, 475, 427))
    for x, y, ep, desc, fill, stroke in endpoints:
        card(p, x, y, 350, 92, ep, desc, fill, stroke, title_y=34, body_y=62, wrap_width=34)
        p.append(line(725, 427, x, y + 46))
    card(p, 1310, 315, 210, 150, "JSON Response", "success, value, plot_url, results, status", COLORS["teal_l"], COLORS["teal"])
    for x, y, *_ in endpoints:
        p.append(line(x + 350, y + 46, 1310, 390))
    finish(p, OUT / "13_api_gateway_communication_flow.svg")


def fig14():
    p = header("Web Application Forecast Page Interface", "Browser-based workflow for selecting category, model, and prediction date")
    p.append(rect(150, 145, 1300, 650, COLORS["white"], COLORS["border"], 18))
    p.append(f'<rect x="150" y="145" width="1300" height="70" rx="18" fill="#0f172a"/>')
    p.append(f'<text x="190" y="188" style="font:700 20px Arial,sans-serif;fill:#ffffff">PharmaPredict AI</text>')
    for i, nav in enumerate(["Dashboard", "Forecast", "Explainability", "Advanced", "Analytics"]):
        p.append(f'<text x="{520+i*145}" y="188" style="font:400 15px Arial;fill:#e2e8f0">{esc(nav)}</text>')
    p.append(rect(205, 260, 380, 460, COLORS["gray_l"], COLORS["border"], 14, ""))
    p.append(text(245, 305, "Forecast Inputs", "metric"))
    for i, (lab, val) in enumerate([("Drug category", "C1 - M01AB"), ("Forecast model", "LightGBM"), ("Prediction date", "2025-01-01")]):
        y = 350 + i * 95
        p.append(text(245, y, lab, "s"))
        p.append(rect(245, y + 16, 285, 44, COLORS["white"], COLORS["border"], 8, ""))
        p.append(text(265, y + 45, val, "b"))
    p.append(rect(245, 640, 285, 48, COLORS["blue"], COLORS["blue"], 10, ""))
    p.append(f'<text x="387" y="671" text-anchor="middle" style="font:700 16px Arial;fill:white">Generate Forecast</text>')
    p.append(rect(660, 260, 315, 175, COLORS["green_l"], COLORS["green"], 14, ""))
    p.append(text(700, 305, "Forecast Result", "metric"))
    p.append(text(700, 350, "Predicted sales", "s"))
    p.append(text(700, 390, "52.84 units", "metric"))
    p.append(rect(1030, 260, 330, 460, COLORS["white"], COLORS["border"], 14, ""))
    p.append(text(1070, 305, "Forecast Chart", "metric"))
    pts = [(1070, 650), (1115, 610), (1160, 635), (1205, 575), (1250, 600), (1295, 515), (1340, 555)]
    p.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="{COLORS["blue"]}" stroke-width="4"/>')
    p.append(f'<circle cx="1340" cy="555" r="7" fill="{COLORS["red"]}"/>')
    p.append(rect(660, 485, 315, 235, COLORS["violet_l"], COLORS["violet"], 14, ""))
    p.append(text(700, 530, "Explanation", "metric"))
    p.extend(multiline(700, 570, "Top lag features explain recent sales influence on the prediction.", 30, "b", "start", 24))
    finish(p, OUT / "14_web_application_forecast_interface.svg")


def fig15():
    p = header("Forecast Result Visualization", "Example C1 historical sales line with future forecast point")
    rows = load_c1_values()
    vals = [v for _, v in rows[-80:]]
    min_v, max_v = min(vals), max(vals)
    x0, y0, w, h = 145, 170, 1220, 520
    p.append(rect(95, 130, 1360, 620, COLORS["white"], COLORS["border"], 18))
    for i in range(6):
        y = y0 + h - i * h / 5
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        label = min_v + i * (max_v - min_v) / 5
        p.append(text(x0 - 20, y + 5, f"{label:.0f}", "tiny", "end"))
    pts = []
    for i, v in enumerate(vals):
        x = x0 + i * (w - 95) / (len(vals) - 1)
        y = y0 + h - (v - min_v) / (max_v - min_v) * h
        pts.append((x, y))
    p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{COLORS["blue"]}" stroke-width="3"/>')
    last_x, last_y = pts[-1]
    pred_x, pred_y = x0 + w, y0 + h - (52.84 - min_v) / (max_v - min_v) * h
    p.append(line(last_x, last_y, pred_x, pred_y, True))
    p.append(f'<circle cx="{pred_x:.1f}" cy="{pred_y:.1f}" r="9" fill="{COLORS["red"]}"/>')
    p.append(text(pred_x, pred_y - 18, "Forecast point", "h", "middle"))
    p.append(text(750, 720, "Historical C1 sales pattern and example future prediction generated by the forecasting workflow", "s", "middle"))
    finish(p, OUT / "15_forecast_result_visualization.svg")


def fig16():
    p = header("Feature Importance Output for Lag-Based Forecasting", "Fallback lag contribution view generated from C1 time-series relationships")
    rows = load_c1_values()
    vals = [v for _, v in rows]
    importances = []
    for lag in range(1, 6):
        xs = vals[:-lag]
        ys = vals[lag:]
        mx, my = mean(xs), mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) or 1
        importances.append((f"sales_lag_{lag}", abs(num / den)))
    total = sum(v for _, v in importances) or 1
    importances = [(k, v / total) for k, v in importances]
    p.append(rect(120, 150, 650, 620, COLORS["white"], COLORS["border"], 18))
    p.append(text(445, 195, "Lag Feature Importance", "metric", "middle"))
    maxv = max(v for _, v in importances)
    for i, (name, val) in enumerate(importances):
        y = 260 + i * 82
        bw = 430 * val / maxv
        p.append(text(180, y + 28, name, "h"))
        p.append(rect(335, y, 430, 42, "#e2e8f0", "#e2e8f0", 8, ""))
        p.append(rect(335, y, bw, 42, COLORS["violet_l"], COLORS["violet"], 8, ""))
        p.append(text(335 + bw + 12, y + 28, f"{val*100:.1f}%", "b"))
    card(p, 880, 195, 510, 140, "Interpretation", "Recent and previous sales values are used to explain why the model predicted the next demand value.", COLORS["blue_l"], COLORS["blue"], wrap_width=46)
    card(p, 880, 395, 510, 140, "Project Behavior", "If SHAP is unavailable, the system falls back to model feature importance or correlation-based lag contribution.", COLORS["amber_l"], COLORS["amber"], wrap_width=46)
    card(p, 880, 595, 510, 140, "Decision Support", "Pharmacy users can see whether the forecast depends on recent demand or older repeating sales behavior.", COLORS["green_l"], COLORS["green"], wrap_width=46)
    finish(p, OUT / "16_feature_importance_output.svg")


def fig17():
    p = header("Overall Model Performance Comparison Chart", "Average MAE, RMSE, MAPE, and inference time across evaluated models")
    perf = load_performance()
    models = ["lightgbm", "lstm", "gru", "xgboost", "transformer", "sarimax", "prophet"]
    metrics = [("final_mae", "MAE", COLORS["blue"]), ("final_rmse", "RMSE", COLORS["green"]), ("final_mape", "MAPE", COLORS["orange"]), ("final_time", "Time", COLORS["violet"])]
    p.append(rect(80, 145, 1440, 645, COLORS["white"], COLORS["border"], 18))
    panels = [(120, 210), (835, 210), (120, 505), (835, 505)]
    for (metric_key, label, color), (px, py) in zip(metrics, panels):
        vals = [perf[m][metric_key] for m in models]
        maxv = max(vals)
        p.append(text(px, py - 25, label, "metric"))
        for i, (m, v) in enumerate(zip(models, vals)):
            x = px + i * 92
            bh = 190 * v / maxv
            p.append(f'<rect x="{x}" y="{py+205-bh:.1f}" width="52" height="{bh:.1f}" rx="6" fill="{color}" opacity="0.82"/>')
            p.append(text(x + 26, py + 228, m[:5], "tiny", "middle"))
            p.append(text(x + 26, py + 195 - bh, f"{v:.2f}", "tiny", "middle"))
    finish(p, OUT / "17_overall_model_performance_comparison.svg")


def fig18():
    p = header("Docker and Kubernetes Deployment Architecture", "Containerized multi-service deployment and monitoring readiness")
    card(p, 90, 245, 270, 130, "Source Code", "FastAPI services, frontend templates, model code", COLORS["blue_l"], COLORS["blue"])
    card(p, 90, 525, 270, 130, "Model and Data Files", "data/raw and artifacts/models", COLORS["gray_l"], COLORS["border"])
    card(p, 495, 190, 300, 170, "Docker Build", "Dockerfile packages application dependencies and runtime", COLORS["green_l"], COLORS["green"])
    card(p, 495, 465, 300, 170, "Docker Compose", "runs API gateway, frontend, forecast, training, explainability, advanced AI", COLORS["amber_l"], COLORS["amber"])
    card(p, 930, 190, 300, 170, "Kubernetes Manifests", "deployments, services, ingress, configmaps, shared namespace", COLORS["violet_l"], COLORS["violet"])
    card(p, 930, 465, 300, 170, "Service Monitoring", "health endpoints and Prometheus metrics", COLORS["pink_l"], COLORS["pink"])
    card(p, 1325, 330, 190, 170, "Ready for Future Cloud Deployment", "GKE-style service structure", COLORS["orange_l"], COLORS["orange"], wrap_width=20)
    p.append(line(360, 310, 495, 275))
    p.append(line(360, 590, 495, 550))
    p.append(line(795, 275, 930, 275))
    p.append(line(795, 550, 930, 550))
    p.append(curved(1230, 275, 1280, 290, 1300, 345, 1325, 380))
    p.append(curved(1230, 550, 1280, 530, 1300, 455, 1325, 430))
    finish(p, OUT / "18_docker_kubernetes_deployment_architecture.svg")


def make_index():
    items = [
        ("01_overall_system_methodology.svg", "Overall System Methodology"),
        ("02_prediction_system_workflow.svg", "Proposed Drug Sales Prediction System Workflow"),
        ("03_drug_category_classification.svg", "Drug Category Classification Used in the Study"),
        ("04_data_preprocessing_pipeline.svg", "Data Preprocessing Pipeline"),
        ("05_time_series_feature_engineering_process.svg", "Time-Series Feature Engineering Process"),
        ("06_time_series_train_test_splitting_method.svg", "Time-Series Train-Test Splitting Method"),
        ("07_forecasting_model_development_workflow.svg", "Forecasting Model Development Workflow"),
        ("08_model_training_artifact_storage_flow.svg", "Model Training and Artifact Storage Flow"),
        ("09_forecasting_request_response_flow.svg", "Forecasting Request-Response Flow"),
        ("10_explainability_workflow.svg", "Explainability Workflow for Forecasting Outputs"),
        ("11_advanced_ai_module_workflow.svg", "Advanced AI Module Workflow"),
        ("12_microservice_architecture.svg", "Microservice-Based System Architecture"),
        ("13_api_gateway_communication_flow.svg", "API Gateway Communication Flow"),
        ("14_web_application_forecast_interface.svg", "Web Application Dashboard or Forecast Page Interface"),
        ("15_forecast_result_visualization.svg", "Forecast Result Visualization"),
        ("16_feature_importance_output.svg", "Feature Importance or SHAP Explanation Output"),
        ("17_overall_model_performance_comparison.svg", "Overall Model Performance Comparison Chart"),
        ("18_docker_kubernetes_deployment_architecture.svg", "Docker and Kubernetes Deployment Architecture"),
    ]
    lines = ["# Thesis SVG Figure Set", "", "All figures are generated from project-specific details and saved as SVG files.", ""]
    for file, title in items:
        lines.append(f"- [{title}]({file})")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18]:
        fn()
    make_index()
    print(f"Generated SVG thesis figures in {OUT}")


if __name__ == "__main__":
    main()
