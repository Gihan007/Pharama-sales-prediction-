# Turnitin Similarity Cleanup Review

Report reviewed: `CSCI 43018- final unstructiured draft (1) - current state 1(1).pdf`

## Overall result

- Overall similarity: 11%
- Not cited or quoted: 10%
- Missing quotations: 0%
- Missing citation: 0%
- Cited and quoted: 0%
- Top source types: submitted student papers, internet sources, publications

This is not a very high similarity score, but the report shows many small matches across common thesis wording, model descriptions, methodology phrasing, results discussion, and conclusion text. The safest improvement is to rewrite generic sentences so they are more specific to this project, and to add citations where the text explains known methods such as SARIMAX, Prophet, XGBoost, LightGBM, LSTM, GRU, Transformer models, SHAP, and forecasting metrics.

## Highest priority pages

The strongest similarity clusters appear on these report pages:

- Page 19: Acknowledgement
- Page 17: Abstract
- Page 34: Structure of the thesis
- Pages 37, 41, 42, 46, 49, 50, 51: Literature review and methodology/model descriptions
- Pages 62, 69, 70: Results/model explanation sections
- Pages 92, 95, 96, 97, 98, 99, 100, 101: Discussion and conclusion
- Page 16: Declaration text, but this is institutional boilerplate and may not need rewriting if the university requires this wording

## Recommended action

1. Do not change required declaration wording unless your department allows it.
2. Rewrite the acknowledgement in a more personal style.
3. Rewrite repeated academic phrases such as "the main objective of this research", "this chapter presents", "machine learning and deep learning techniques", and "future research can improve this study".
4. Add IEEE citations in the methodology/literature review where algorithm definitions are described.
5. Keep exact model names, metrics, result values, and dataset details unchanged.
6. Re-check after editing the DOCX source, because this PDF is a Turnitin report and is not the editable thesis file.

## Suggested rewrites

### Declaration note

The declaration section is commonly matched because many institutions use standard academic wording. If this text is required by the university, keep it. If rewriting is allowed, make only minor wording changes and keep the legal meaning unchanged.

### Acknowledgement

Replace the current generic acknowledgement with a more personal version:

> I sincerely thank everyone who helped me complete this research project, Intelligent Drug Sales Forecasting and Healthcare Insights with Artificial Intelligence. I am especially grateful to my supervisor, Professor Niomal Dias, for the guidance, feedback, and encouragement given throughout the project. His support helped me strengthen both the technical implementation and the written thesis.
>
> I also appreciate the academic staff of the Faculty of Computing and Technology for the knowledge and learning environment they provided during my degree programme. My family deserves my deepest thanks for their patience, motivation, and support during the most demanding stages of this work. I am also thankful to my workplace supervisors, Mr. Sajeepan and Mr. Arun Balasundaram, for their encouragement and understanding while I balanced academic and professional responsibilities.
>
> Finally, I thank everyone who contributed directly or indirectly to the successful completion of this thesis. This project has been a valuable learning experience, and I am grateful for the support I received along the way.

### Abstract objective paragraph

Possible rewrite:

> This project develops a forecasting platform for estimating future pharmaceutical sales from historical category-wise sales records. Instead of relying on a single forecasting technique, the study evaluates a group of statistical, machine learning, and deep learning models, including XGBoost, LightGBM, SARIMAX, Prophet, LSTM, GRU, Transformer, Temporal Fusion Transformer, N-BEATS, and Informer. Additional components such as ensemble forecasting, meta-learning, neural architecture search, federated learning, causal inference, uncertainty estimation, and explainability were included to support model comparison and improve the interpretability of the forecasting results.

### Thesis structure section

Possible rewrite:

> The thesis is arranged into six chapters. Chapter 1 introduces the research background, problem statement, research gap, objectives, scope, and significance of the study. Chapter 2 reviews related work on pharmaceutical demand forecasting, time-series prediction, machine learning, deep learning, explainable AI, and advanced AI methods. Chapter 3 describes the research methodology, including the dataset, preprocessing steps, feature engineering process, model development, system architecture, API design, and deployment approach. Chapter 4 presents the experimental results and compares the forecasting behaviour of the implemented models. Chapter 5 discusses the findings in relation to the research objectives, practical value, limitations, and future improvements. Chapter 6 summarizes the research outcomes and provides the final conclusion.

### Methodology opening

Possible rewrite:

> The proposed framework follows a pipeline that begins with category-wise pharmaceutical sales data and ends with model evaluation and web-based prediction. Historical sales records from categories C1 to C8 are first cleaned and converted into time-series datasets. Lag values, rolling summaries, and date-related features are then generated so that both traditional models and supervised learning models can learn demand behaviour from previous sales patterns.

### Feature engineering section

Possible rewrite:

> Feature engineering was used to convert raw sales records into a format suitable for forecasting. Since medicine demand is often influenced by recent purchasing behaviour, previous sales values were used as lag features. Additional time-based variables were created to represent the order and timing of observations. For tree-based models such as XGBoost and LightGBM, these engineered features allow the model to estimate future sales from recent historical patterns. For sequential models such as LSTM, GRU, Transformer, TFT, N-BEATS, and Informer, the data was arranged into input windows so the models could learn temporal behaviour across consecutive observations.

### Model development section

Possible rewrite:

> Model development formed the central experimental stage of this research. Each drug category was treated as a separate forecasting problem, allowing the models to learn category-specific sales behaviour. Statistical methods were used to represent trend and seasonal patterns, while machine learning models used lag-based features to learn nonlinear relationships. Deep learning models were trained on sequential windows of past sales values. This design made it possible to compare simple, tree-based, recurrent, attention-based, and ensemble approaches under the same dataset structure.

### Machine learning models section

Possible rewrite:

> XGBoost and LightGBM were selected as the main tree-based machine learning models. These models are suitable for structured forecasting datasets because they can learn nonlinear relationships from engineered lag and calendar features. XGBoost builds boosted decision trees sequentially, while LightGBM uses an efficient gradient boosting framework designed for fast training on tabular data. In this study, both models were trained separately for each category so their performance could be compared across different drug sales patterns.

### Results chapter opening

Possible rewrite:

> This chapter reports the forecasting results obtained from the implemented drug sales prediction system. Since the dataset contains eight drug categories, each category was evaluated as an individual time-series prediction task. The analysis compares the behaviour of statistical, machine learning, deep learning, and ensemble models using MAE, RMSE, MAPE, and inference time. These metrics were selected to evaluate both prediction accuracy and practical response speed for the web-based forecasting system.

### Deep learning results explanation

Possible rewrite:

> The deep learning experiments examined whether sequential models could capture sales patterns that depend on earlier observations. The input data was arranged as windows of previous sales values, and each model attempted to estimate the following sales value. This setup is appropriate for pharmaceutical sales because demand in one period may be influenced by recent demand patterns, recurring purchasing behaviour, and category-specific trends. LSTM and GRU produced competitive results, showing that recurrent models can represent useful temporal dependencies in the sales data.

### Discussion opening

Possible rewrite:

> The findings show that historical weekly pharmaceutical sales can support useful demand forecasts when the data is prepared and evaluated category by category. The dataset contained records from 2014 to 2023, and the sales behaviour differed noticeably between categories. For example, C4 showed the highest average sales volume, while C6 had the lowest average sales volume and included weeks with zero sales. These differences explain why model performance varied across categories and why a single forecasting model was not equally suitable for every drug group.

### Conclusion opening

Possible rewrite:

> This research designed and evaluated a machine learning-based system for pharmaceutical drug sales forecasting. The study compared statistical, machine learning, deep learning, and ensemble approaches using weekly sales data from eight drug categories. The results show that LightGBM achieved the strongest overall performance based on MAE and RMSE, while LSTM and GRU also produced competitive forecasts. These findings indicate that both tree-based machine learning and recurrent deep learning methods are useful for pharmaceutical sales prediction when the data is properly structured.

### Limitations section

Possible rewrite:

> Several limitations should be considered when interpreting the results. First, the dataset was limited to eight drug categories, which restricts the generalizability of the findings. A larger dataset containing more products, pharmacies, regions, and brands would provide a stronger basis for future evaluation. Second, the current dataset did not include external demand drivers such as weather, holidays, disease outbreaks, promotions, price changes, prescription trends, or supplier delays. As a result, the models learned mainly from historical sales behaviour rather than from broader real-world factors that may influence medicine demand.

### Future work section

Possible rewrite:

> Future work should extend the dataset and improve the system's real-world applicability. Adding more drug categories, longer time periods, and data from multiple pharmacies or districts would allow the models to learn broader demand patterns. External variables such as holidays, seasonal illness trends, weather, promotions, pricing, and regional population information could also be integrated to improve forecast accuracy. Since the project aims to support area-based prediction in Sri Lanka, future versions should include explicit district, city, branch, or pharmacy-level sales records.

## Citation reminders

Add or verify IEEE citations near these technical explanations:

- Time-series forecasting and train-test splitting
- SARIMAX and Prophet descriptions
- XGBoost and LightGBM descriptions
- LSTM, GRU, Transformer, TFT, N-BEATS, and Informer descriptions
- SHAP/explainability
- MAE, RMSE, and MAPE definitions
- Ensemble forecasting and meta-learning, if discussed as research methods

## Important file note

The editable thesis DOCX was not found in the workspace. The available thesis-related PDF is the Turnitin report, so changes should be applied to the original DOCX file and then exported/re-uploaded for a fresh Turnitin check.
