from __future__ import annotations

from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "final_thesos .docx"
OUT = ROOT / "docs" / "thesisi version"


def main() -> None:
    document = Document(SOURCE)
    start = next(i for i, p in enumerate(document.paragraphs) if p.text.strip() == "Literature Review")
    end = next(i for i, p in enumerate(document.paragraphs[start + 1:], start + 1) if p.style.name == "Heading 1")
    paragraphs = [" ".join(document.paragraphs[i].text.split()) for i in range(start + 1, end) if document.paragraphs[i].text.strip()]

    # Citation-only corrections. The surrounding research logic is preserved.
    paragraphs[0] = paragraphs[0].replace(
        "For example, Abbas et al. proposed an explainable artificial intelligence-based model for ocular disease prediction, showing how AI can be used not only to make predictions but also to improve transparency in healthcare decision making. Similarly, Khan et al. , Nazir et al.,et al. , and Wijesinghe et al. applied deep learning techniques for medical image-based disease recognition.",
        "For example, Abbas et al. [1] proposed an explainable artificial intelligence-based model for ocular disease prediction, showing how AI can be used not only to make predictions but also to improve transparency in healthcare decision making. Similarly, Khan et al. [2], Nazir et al. [3], Gamage et al. [4], and Wijesinghe et al. [5] applied deep learning techniques for medical image-based disease recognition.",
    )
    paragraphs[5] = paragraphs[5].replace("Ekanayake et al. directly addressed", "Ekanayake et al. [6] directly addressed")

    # Placeholder [7] is intentionally used in exactly two locations requested by the user.
    paragraphs[9] = paragraphs[9].replace(
        "Transformer-based models have also become important in modern forecasting because they use attention mechanisms to learn relationships across time steps.",
        "Transformer-based models have also become important in modern forecasting because they use attention mechanisms to learn relationships across time steps [7].",
    )
    paragraphs[14] = paragraphs[14].replace(
        "Federated learning is also included as a future-ready design for pharmaceutical forecasting because real sales data may belong to different pharmacies, hospitals, distributors, or regional branches.",
        "Federated learning is also included as a future-ready design for pharmaceutical forecasting because real sales data may belong to different pharmacies, hospitals, distributors, or regional branches [7].",
    )

    paragraphs[10] = paragraphs[10].replace(
        "Lim and Zohdy discussed explainable artificial intelligence for business decision support using sales forecasting as a case study. Their work highlights the importance of transparency in forecasting systems, especially when predictions are used for operational decisions.",
        "Chen et al. [9] proposed an explainable artificial intelligence framework for sales forecasting by combining WaveNet, multiple regression, and SHAP. Their work highlights the importance of transparency in forecasting systems, especially when predictions are used for operational decisions.",
    )
    paragraphs[-1] = paragraphs[-1].replace(
        "Previous studies have demonstrated the effectiveness of deep learning in medical applications , the relevance of time-series forecasting for drug sales prediction , the usefulness of recurrent neural networks in forecasting , the importance of explainability in sales forecasting , and the strength of XGBoost-based models for drug sales prediction .",
        "Previous studies have demonstrated the effectiveness of deep learning in medical applications [1]–[5], the relevance of time-series forecasting for drug sales prediction [6], the usefulness of recurrent neural networks in forecasting [8], the importance of explainability in sales forecasting [9], and the strength of XGBoost-based models for drug sales prediction [10].",
    )

    literature_text = "LITERATURE REVIEW — CORRECTED CITATIONS\n\n" + "\n\n".join(paragraphs) + "\n"
    (OUT / "corrected_literature_review.txt").write_text(literature_text, encoding="utf-8")

    references = """REFERENCES — CORRECTED ORDER

[1] S. Abbas, A. Qaisar, M. S. Farooq, M. Saleem, M. Ahmad, and M. A. Khan, “Smart Vision Transparency: Efficient Ocular Disease Prediction Model Using Explainable Artificial Intelligence,” Sensors, vol. 24, no. 20, Art. no. 6618, 2024, doi: 10.3390/s24206618.

[2] M. S. Khan, N. Tafshir, K. N. Alam, A. R. Dhruba, M. M. Khan, A. A. Albraikan, and F. A. Almalki, “Deep Learning for Ocular Disease Recognition: An Inner-Class Balance,” Computational Intelligence and Neuroscience, vol. 2022, Art. no. 5007111, 2022, doi: 10.1155/2022/5007111. [Retracted: doi: 10.1155/2023/9838475.]

[3] T. Nazir et al., “Detection of Diabetic Eye Disease from Retinal Images Using a Deep Learning-Based CenterNet Model,” Sensors, vol. 21, no. 16, Art. no. 5283, 2021, doi: 10.3390/s21165283.

[4] T. Gamage et al., “Age-Related Retinal Disease Identification Using Image Processing and Deep Learning,” in Proc. 4th Int. Conf. Computing and Artificial Intelligence (CCAI), 2024. [Publication page: https://www.researchgate.net/publication/382751936_Age-Related_Retinal_Disease_Identification_Using_Image_Processing_and_Deep_Learning]

[5] K. H. Wijesinghe et al., “Identification of Diabetic Related Eye Diseases Using Deep Learning,” in Proc. 5th Int. Conf. Advancements in Computing (ICAC), 2023, pp. 786–791, doi: 10.1109/ICAC60630.2023.10417352.

[6] S. B. Ekanayake, G. Lakmal, A. Perera, M. Nasmeen, P. Vimanshani, and M. D. Chandrasena Atampalage, “Predicting Medical Drug Sales in a Specific Area for Categorical Drugs Using Time Series Forecasting,” in Proc. Int. Research Conf. Smart Computing and Systems Engineering (SCSE), Colombo, Sri Lanka, Apr. 2025, pp. 1–6, doi: 10.1109/SCSE65633.2025.11031056.

[7] [PLACEHOLDER — REFERENCE DETAILS TO BE ADDED LATER. This source is cited twice in the Literature Review and should support both Transformer-based forecasting and federated learning.]

[8] H. Hewamalage, C. Bergmeir, and K. Bandara, “Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions,” International Journal of Forecasting, vol. 37, no. 1, pp. 388–427, 2021, doi: 10.1016/j.ijforecast.2020.06.008.

[9] S. Chen, S. Ke, S. Han, S. Gupta, and U. Sivarajah, “Which Product Description Phrases Affect Sales Forecasting? An Explainable AI Framework by Integrating WaveNet Neural Network Models with Multiple Regression,” Decision Support Systems, vol. 176, Art. no. 114065, 2024, doi: 10.1016/j.dss.2023.114065.

[10] Y. Xiong, “Development of an AI-Driven Model for Drug Sales Prediction Using Enhanced Golden Eagle Optimization and XGBoost Algorithm,” Informatica, vol. 49, no. 17, pp. 37–50, 2025, doi: 10.31449/inf.v49i17.7491.

IMPORTANT
- Reference [7] is intentionally incomplete and must be replaced before final submission.
- Reference [2] has been retracted; replacing it with a non-retracted source is strongly recommended.
"""
    (OUT / "corrected_references.txt").write_text(references, encoding="utf-8")

    print("corrected_literature_review.txt")
    print("corrected_references.txt")


if __name__ == "__main__":
    main()
