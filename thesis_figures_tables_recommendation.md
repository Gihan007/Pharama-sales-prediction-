# Recommended Figures and Tables for Current Thesis Draft

Reviewed file: `CSCI 43018- final unstructiured draft (1) - current state 2.pdf`

Page notation used below:

- **PDF page** = page number in the PDF viewer.
- **Thesis page** = printed page number shown inside the thesis.

## Important Formatting Notes

1. The current **List of Figures** lists many figures, but the body only clearly shows a few actual figure captions. Add the missing figures or remove them from the list.
2. The current document has a numbering issue: the preprocessing diagram appears as **FIGURE 1** on thesis page 23, but the List of Figures says **Figure 4: Data Preprocessing Pipeline**. Fix numbering after inserting all figures.
3. The **List of Tables** is currently not properly populated. The thesis needs tables because many result values are written as paragraphs.
4. Use consistent caption style:
   - `Figure 3.1: Data Preprocessing Pipeline`
   - `Table 4.1: Statistical Model Performance Comparison`
5. If your department requires simple numbering, use `Figure 1`, `Figure 2`, etc. If not, chapter-based numbering is cleaner.

## Recommended Figures

| No. | Insert at PDF page / thesis page | Recommended figure title | What to show | Priority |
|---|---:|---|---|---|
| 1 | PDF 27 / thesis 22 | Overall System Methodology | Data collection, preprocessing, feature engineering, model training, forecasting, explainability, evaluation, deployment | Must add |
| 2 | PDF 28-29 / thesis 23-24 | Proposed Drug Sales Prediction System Workflow | User input, frontend, API gateway, forecast service, model loading, prediction, visualization, explanation | Must add |
| 3 | PDF 19 or PDF 29 / thesis 14 or 24 | Drug Category Classification Used in the Study | C1-C8 mapped to pharmaceutical groups such as M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06 | Must add |
| 4 | PDF 28 / thesis 23 | Data Preprocessing Pipeline | C1-C8 CSV files, read CSV, parse `datum`, set time index, clean usable records, select sales target, prepared time-series dataset | Must add |
| 5 | PDF 32 / thesis 27 | Time-Series Feature Engineering Process | Raw sales series to lag features for XGBoost/LightGBM and sequence windows for LSTM/GRU/Transformer models | Must add |
| 6 | PDF 35 / thesis 30 | Time-Series Train-Test Splitting Method | Earlier records used for training and later records used for testing, without random shuffling | Must add |
| 7 | PDF 35-36 / thesis 30-31 | Forecasting Model Development Workflow | Statistical models, ML models, deep learning models, ensemble model, evaluation | Must add |
| 8 | PDF 40 / thesis 35 | Model Training and Artifact Storage Flow | Category data, model training, saved model files, saved scalers, artifact folders | Recommended |
| 9 | PDF 41 / thesis 36 | Forecasting Request-Response Flow | User category/date/model request, backend forecast, historical lookup or future forecast, returned chart and value | Recommended |
| 10 | PDF 41-42 / thesis 36-37 | Explainability Workflow for Forecasting Outputs | Prediction, SHAP/fallback feature importance, lag contribution, explanation output | Must add |
| 11 | PDF 42-43 / thesis 37-38 | Advanced AI Module Workflow | Meta-learning, NAS, federated learning, causal inference and their purpose | Recommended |
| 12 | PDF 43-44 / thesis 38-39 | Microservice-Based System Architecture | Frontend, API gateway, forecast service, training service, explainability service, advanced AI service, model/data storage | Must add |
| 13 | PDF 44-45 / thesis 39-40 | API Gateway Communication Flow | Frontend request routed to forecast/training/explainability/advanced services | Recommended |
| 14 | PDF 45-46 / thesis 40-41 | Web Application Dashboard or Forecast Page Interface | Screenshot of actual dashboard or forecast page | Must add |
| 15 | PDF 69-70 / thesis 64-65 | Forecast Result Visualization | Screenshot or generated forecast plot showing historical sales and predicted value | Must add |
| 16 | PDF 63-64 / thesis 58-59 | Feature Importance or SHAP Explanation Output | SHAP summary or fallback lag importance plot for one category/model | Recommended |
| 17 | PDF 75-76 / thesis 70-71 | Overall Model Performance Comparison Chart | Bar chart comparing MAE, RMSE, MAPE, and inference time across models | Must add |
| 18 | PDF 74-75 / thesis 69-70 | Docker and Kubernetes Deployment Architecture | Docker Compose/Kubernetes services and monitoring endpoints | Recommended |

## Recommended Tables

| No. | Insert at PDF page / thesis page | Recommended table title | Columns / content | Priority |
|---|---:|---|---|---|
| 1 | PDF 21-26 / thesis 16-21 | Summary of Related Work | Reference, method, dataset/domain, key contribution, limitation, relevance to this project | Must add |
| 2 | PDF 19 or PDF 29 / thesis 14 or 24 | Drug Category Mapping | Category code, drug group/code, description/example use | Must add |
| 3 | PDF 29-31 / thesis 24-26 | Dataset Summary for C1-C8 | Category, records, date range, missing sales, zero-sales weeks, mean, min, max | Must add |
| 4 | PDF 31 / thesis 26 | Data Preprocessing Steps | Step, project implementation, output | Recommended |
| 5 | PDF 32-34 / thesis 27-29 | Feature Engineering by Model Type | Model group, input transformation, scaler, sequence/lag length, target | Must add |
| 6 | PDF 35-40 / thesis 30-35 | Model Groups and Implemented Algorithms | Group, algorithms, purpose, input format, saved artifact type | Must add |
| 7 | PDF 40 / thesis 35 | Model Artifact Storage Structure | Model, folder, artifact file pattern, scaler file pattern | Recommended |
| 8 | PDF 45 / thesis 40 | API Endpoint Summary | Endpoint, input, service used, output | Must add |
| 9 | PDF 47-48 / thesis 42-43 | Evaluation Metrics Used | Metric, formula/meaning, interpretation, limitation | Must add |
| 10 | PDF 51 / thesis 46 | Statistical Model Performance Comparison | Model, Average MAE, Average RMSE, Average MAPE, Average inference time | Must add |
| 11 | PDF 54-55 / thesis 49-50 | Machine Learning Model Performance Comparison | Model, Average MAE, Average RMSE, Average MAPE, Average inference time | Must add |
| 12 | PDF 59 / thesis 54 | Deep Learning Model Performance Comparison | Model, Average MAE, Average RMSE, Average MAPE, Average inference time | Must add |
| 13 | PDF 61 / thesis 56 | Ensemble Forecasting Performance | Category, weighted average MAE/RMSE/MAPE, performance-weighted MAE/RMSE/MAPE, time | Recommended |
| 14 | PDF 65-68 / thesis 60-63 | Advanced AI Module Summary | Module, purpose, input, output, project role | Recommended |
| 15 | PDF 73 / thesis 68 | API Testing Summary | Testing area, total endpoints, passed, failed, result | Must add |
| 16 | PDF 74-75 / thesis 69-70 | Deployment and Monitoring Components | Component, purpose, project file/configuration, validation status | Recommended |
| 17 | PDF 75-76 / thesis 70-71 | Overall Model Ranking | Rank, model, MAE, RMSE, MAPE, inference time, best use case | Must add |
| 18 | PDF 78 / thesis 73 | Best Model by Drug Category | Category, best model by MAE, MAE, RMSE, comment | Must add |
| 19 | PDF 85-86 / thesis 80-81 | Limitations and Future Improvements | Limitation, effect on study, future improvement | Recommended |

## Tables With Exact Values Available From Project Files

### Dataset Summary Values

Use this table around PDF page 29-31 / thesis page 24-26.

| Category | Records | Date range | Missing sales | Zero-sales weeks | Mean | Min | Max |
|---|---:|---|---:|---:|---:|---:|---:|
| C1 | 517 | 2014-01-12 to 2023-12-03 | 0 | 0 | 37.4990 | 14.0000 | 65.3300 |
| C2 | 517 | 2014-01-12 to 2023-12-03 | 0 | 0 | 28.2651 | 7.7100 | 53.5710 |
| C3 | 517 | 2014-01-12 to 2023-12-03 | 0 | 0 | 30.3711 | 10.0000 | 60.1250 |
| C4 | 517 | 2014-01-12 to 2023-12-03 | 0 | 0 | 256.0434 | 86.2500 | 546.8990 |
| C5 | 517 | 2014-01-12 to 2023-12-03 | 0 | 0 | 72.4750 | 18.0000 | 154.0000 |
| C6 | 517 | 2014-01-12 to 2023-12-03 | 0 | 46 | 5.7648 | 0.0000 | 17.0000 |
| C7 | 517 | 2014-01-12 to 2023-12-03 | 0 | 0 | 49.7242 | 2.0000 | 131.0000 |
| C8 | 517 | 2014-01-12 to 2023-12-03 | 0 | 0 | 24.7425 | 3.0000 | 65.0000 |

### Overall Model Ranking Values

Use this table around PDF page 75-76 / thesis page 70-71.

| Rank by MAE | Model | MAE | RMSE | MAPE | Avg. time |
|---:|---|---:|---:|---:|---:|
| 1 | LightGBM | 23.3728 | 28.9163 | 73.9233 | 0.1074 s |
| 2 | LSTM | 24.0362 | 29.5830 | 62.6915 | 0.1260 s |
| 3 | GRU | 24.1104 | 29.7559 | 63.9476 | 0.1067 s |
| 4 | XGBoost | 25.4679 | 29.9704 | 86.4031 | 0.0343 s |
| 5 | Transformer | 28.8908 | 36.5221 | 53.6691 | 0.2426 s |
| 6 | SARIMAX | 30.4946 | 38.4108 | 56.0618 | 0.1498 s |
| 7 | Prophet | 32.5682 | 37.6763 | 88.0285 | 0.8725 s |

### Best Model by Category

Use this table around PDF page 78 / thesis page 73.

| Category | Best model by MAE | MAE | RMSE | Note |
|---|---|---:|---:|---|
| C1 | GRU | 9.8758 | 12.3715 | Best absolute error for C1 |
| C2 | Prophet | 13.1088 | 14.6696 | Slightly better MAE than LSTM/GRU |
| C3 | Prophet | 10.2570 | 12.2604 | Best MAE and RMSE for C3 |
| C4 | LightGBM | 72.6065 | 92.3468 | Best high-volume category result |
| C5 | LSTM | 24.8986 | 30.4647 | Very close to GRU |
| C6 | LSTM | 3.1634 | 4.0392 | Low-volume category with zero-sales weeks |
| C7 | XGBoost | 26.0462 | 33.0924 | Best MAE for C7 |
| C8 | LSTM | 13.8154 | 15.1094 | Best MAE and RMSE for C8 |

## What To Remove Or Fix

- Do not list figures that are not actually inserted in the body.
- Fix duplicate/mismatched numbering around `Figure 3` and the preprocessing figure.
- Convert inline result blocks like `Model Average MAE Average RMSE...` into proper tables.
- Add table captions above or below according to your department format, then update the List of Tables.
- Add figure captions consistently and update the List of Figures after final numbering.

## Best Minimal Set If You Are Short On Time

Add these first:

1. Figure: Data Preprocessing Pipeline, PDF 28 / thesis 23.
2. Figure: Overall System Methodology, PDF 27 / thesis 22.
3. Figure: Microservice Architecture, PDF 43-44 / thesis 38-39.
4. Figure: Forecast Result Visualization, PDF 69 / thesis 64.
5. Figure: Model Performance Comparison Chart, PDF 75-76 / thesis 70-71.
6. Table: Dataset Summary, PDF 29-31 / thesis 24-26.
7. Table: Model Groups and Algorithms, PDF 35-40 / thesis 30-35.
8. Table: Evaluation Metrics, PDF 47 / thesis 42.
9. Table: Overall Model Ranking, PDF 75-76 / thesis 70-71.
10. Table: Best Model by Category, PDF 78 / thesis 73.
