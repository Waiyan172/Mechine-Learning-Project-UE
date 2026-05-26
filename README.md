# Predicting Mental Health Risk in Gamers Using Supervised Machine Learning

A comparative study of six supervised learning models for predicting mental-health risk among gamers using behavioural, lifestyle, and demographic features from a public Kaggle dataset of approximately ten million records and forty features.

**Author:** Wai Yan Tun
**Course:** Machine Learning — Assignment Submission II
**Instructor:** Prof. Raja Hashim Ali

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Research Questions](#research-questions)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [How to Reproduce](#how-to-reproduce)
- [Submission Materials](#submission-materials)

---

## Project Overview

Digital gaming is now one of the largest forms of entertainment worldwide, and excessive play has been linked to depression, anxiety, sleep loss, and social withdrawal. This project investigates how well off-the-shelf supervised learning methods can predict mental-health risk among gamers from behavioural and demographic features.

A stratified sample of 50,000 records is drawn from a ten-million-row Kaggle dataset and used to train and evaluate six classifiers: **Logistic Regression**, **Decision Tree**, **k-Nearest Neighbours**, **Random Forest**, **Support Vector Machine**, and **Extreme Gradient Boosting (XGBoost)**. The study is organised around seven research questions covering baseline performance, model comparison, preprocessing, feature importance, metric sensitivity, robustness, and practical usefulness.

---

## Repository Structure

```
.
├── Code/                                                # 7 Jupyter notebooks
│   ├── RQ1_baseline_performance.ipynb
│   ├── RQ2_model_comparison.ipynb
│   ├── RQ3_effect_of_preprocessing.ipynb
│   ├── RQ4_feature_importance.ipynb
│   ├── RQ5_sensitivity_to_metrics.ipynb
│   ├── RQ6_robustness.ipynb
│   └── RQ7_practical_usefulness.ipynb
│
├── Dataset/                                             # Dataset link and info
│   ├── DATASET REPORT.pdf
│
├── Figures/                                             # PDF figures, 300 dpi
│   ├── methodology_flow.pdf
│   ├── RQ1_figure.pdf
│   ├── RQ2_figure.pdf
│   ├── RQ3_figure.pdf
│   ├── RQ4_feature_importance.pdf
│   ├── RQ5_ranking_line_figure.pdf
│   ├── RQ6_figure.pdf
│   └── RQ7_figure.pdf
│
├── Tables/                                              # CSV tables for every RQ
│   ├── RQ1_table.csv
│   ├── RQ2_table.csv
│   ├── RQ3_table.csv
│   ├── RQ4_feature_importance_table.csv
│   ├── RQ5_ranking_table.csv
│   ├── RQ6_table.csv
│   └── RQ7_table.csv
│
├── Machine_learning_Assignment_Submission_II___Report.pdf
├── Submission Links.pdf
└── README.md
```

---

## Dataset

- **Name:** Gaming and Mental Health Dataset
- **Source:** Kaggle — user `sharmajicoder`
- **Link:** <https://www.kaggle.com/datasets/sharmajicoder/gaming-and-mental-health>
- **Size:** ~10,000,000 records, 40 features
- **Sample used in this project:** stratified random sample of 50,000 records (random seed = 42)

The raw CSV is too large to commit directly to GitHub, so a link to the original Kaggle source is kept in `Dataset/DATASET REPORT.pdf`.

---

## Research Questions

| RQ  | Question                                                                | Notebook                                 |
| --- | ----------------------------------------------------------------------- | ---------------------------------------- |
| RQ1 | How well do baseline models (LR, DT, k-NN) predict mental-health risk?  | `Code/RQ1_baseline_performance.ipynb`    |
| RQ2 | Which advanced model achieves the highest predictive performance?       | `Code/RQ2_model_comparison.ipynb`        |
| RQ3 | How do imputation, scaling, encoding, and SMOTE affect performance?     | `Code/RQ3_effect_of_preprocessing.ipynb` |
| RQ4 | Which features most strongly influence predicted risk?                  | `Code/RQ4_feature_importance.ipynb`      |
| RQ5 | How does model ranking change with different evaluation metrics?        | `Code/RQ5_sensitivity_to_metrics.ipynb`  |
| RQ6 | How robust is the model under cross-validation, noise, and missingness? | `Code/RQ6_robustness.ipynb`              |
| RQ7 | How suitable is the model for real-world deployment?                    | `Code/RQ7_practical_usefulness.ipynb`    |

---

## Methodology

The pipeline is organised into five sequential stages, illustrated in `Figures/methodology_flow.pdf`:

1. **Data acquisition** — load Kaggle CSV, draw stratified 50,000-record sample, 80/20 train/test split (seed = 42).
2. **Preprocessing** — median/mode imputation, one-hot encoding, z-score scaling, SMOTE on training set only.
3. **Model training** — fit the six classifiers on the same training partition with the same random seed.
4. **Evaluation** — compute Accuracy, Precision, Recall, F1-score, and AUC on the held-out test set.
5. **Robustness and interpretation** — 5-fold cross-validation, +10% label noise, 20% induced missingness, permutation feature importance, weighted decision matrix.

---

## Key Findings

- **Headline accuracy is high (≈ 0.90), but AUC is near random (≈ 0.51).** All six classifiers reach similar accuracy because they largely predict the majority class. AUC reveals that the binarised target is intrinsically hard to separate from the available features.
- **Preprocessing has limited effect.** Imputation, scaling, and encoding leave metrics unchanged, while SMOTE produces the expected trade-off — slightly lower accuracy (0.896 → 0.881) for slightly higher precision (0.803 → 0.816).
- **Most influential features (permutation importance):**
  1. mobile gaming ratio
  2. screen time total
  3. caffeine intake
  4. academic performance
  5. streaming hours
- **Robustness:** model performance stays stable under cross-validation, ten-percent label noise, and twenty-percent induced missingness, indicating that the limit is the signal in the data rather than the choice of classifier.
- **Practical recommendation (RQ7 weighted decision matrix):** Logistic Regression wins (weighted score 3.80) over Random Forest and XGBoost (3.00 each) because its strong interpretability, robustness, cost efficiency, and deployment readiness outweigh the small accuracy gain offered by XGBoost.

The headline numbers from each notebook are saved as CSVs in `Tables/` and reproduced in the report.

---

## How to Reproduce

### Requirements

- Python 3.10 or newer
- Jupyter Notebook (or run on Kaggle / Google Colab)
- Recommended environment: 16 GB RAM, a GPU is helpful but not required

### Python packages

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### Steps

1. Clone this repository.
   ```bash
   git clone https://github.com/Waiyan172/Mechine-Learning-Project-UE.git
   cd Mechine-Learning-Project-UE
   ```
2. Download the dataset from <https://www.kaggle.com/datasets/sharmajicoder/gaming-and-mental-health> and place the CSV in the `Dataset/` folder. The notebooks expect a file path that can be adjusted in the first code cell.
3. Open the notebooks in `Code/` in order (RQ1 → RQ7) and run each one end-to-end. Each notebook is self-contained and saves its tables and figures back into the `Tables/` and `Figures/` folders.
4. All random seeds are fixed to **42** at the level of NumPy, scikit-learn, and XGBoost, so re-running the notebooks should produce identical numbers.

---

## Submission Materials

- **Full report (PDF):** `Machine_learning_Assignment_Submission_II___Report.pdf`
- **Submission links (PDF):** `Submission Links.pdf` — contains the Overleaf link, the GitHub repository link, and the dataset link.
- **Figures:** all 7 RQ figures plus the methodology flow diagram, exported as 300 dpi PDFs in `Figures/`.
- **Tables:** raw CSV outputs from every notebook in `Tables/`, used to build the LaTeX tables in the report.
