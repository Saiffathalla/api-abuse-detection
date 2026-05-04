# REST API Abuse Detection Using Request Pattern Analysis

## Overview

This project implements a machine learning pipeline to detect and classify REST API abuse patterns using a cyber-attacks dataset. Inspired by the IEEE CloudCom 2021 paper *"API Security Threat Detection Using Machine Learning."*

**Dataset:** 10,000 records covering 10 attack types across multiple industries and locations.

**Tasks:**
- Multi-class attack type classification (10 classes)
- Binary API-specific attack detection
- Severity level prediction (Low / Medium / High / Critical)

---

## Project Structure

```
api_abuse_detection/
├── data/
│   └── cyber_attacks_dataset.csv        # Raw dataset
├── notebooks/
│   └── analysis.py                      # VS Code Jupyter notebook (# %% cells)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Dataset loading & inspection
│   ├── preprocessing.py                 # Encoding, scaling, train/test split
│   ├── feature_engineering.py           # Domain-specific feature creation
│   ├── models.py                        # Model definitions & training
│   └── evaluation.py                   # Metrics, plots, reports
├── models/
│   ├── best_model_rf.joblib             # Tuned Random Forest
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder_target.joblib
│   └── feature_names.joblib
├── outputs/
│   ├── figures/                         # All 12 visualization PNGs
│   └── reports/                         # Classification report TXTs
├── main.py                              # End-to-end pipeline script
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place `cyber_attacks_dataset.csv` into the `data/` folder.

---

## Run the Pipeline

```bash
python main.py
```

This will:
1. Load and inspect the dataset
2. Perform full EDA (12 visualizations saved to `outputs/figures/`)
3. Engineer domain-specific features
4. Preprocess and split data (80/20 stratified)
5. Train 5 models: Logistic Regression, Decision Tree, Random Forest, XGBoost, Gradient Boosting
6. Run GridSearchCV hyperparameter tuning on Random Forest
7. Evaluate all models (confusion matrices, ROC+AUC, classification reports)
8. Save best model and artefacts to `models/`

---

## Models Compared

| Model | CV F1 (Weighted) |
|---|---|
| Logistic Regression | baseline |
| Decision Tree | ~baseline |
| Random Forest (Tuned) | best |
| XGBoost | competitive |
| Gradient Boosting | competitive |

---

## Key Features Engineered

- **Temporal:** Year, Month, DayOfWeek, Quarter, IsWeekend
- **Damage:** Log_Damage, Damage_Per_System, High_Damage flag
- **Risk Scores:** Industry_Risk_Score, Location_Risk_Score
- **Frequency:** Industry_Attack_Frequency, Location_Attack_Frequency
- **Target:** Is_API_Attack binary label (SQL Injection, DDoS, Brute Force, XSS, Privilege Escalation, MitM)

---

## Output Figures

| File | Description |
|---|---|
| 01_data_distribution.png | Attack type, severity, industry, location distributions |
| 02_numerical_distributions.png | Log damage & affected systems histograms |
| 03_boxplots.png | Damage by severity, systems by attack type |
| 04_correlation_heatmap.png | Feature correlation matrix |
| 05_attack_severity_heatmap.png | Attack type × severity crosstab |
| 06_class_distribution.png | Class balance |
| 07_confusion_matrices.png | RF Tuned vs XGBoost |
| 08_roc_curves.png | Multi-class OvR ROC curves |
| 09_feature_importance.png | RF feature importances |
| 10_xgboost_training_history.png | XGBoost train vs eval log-loss |
| 11_model_comparison.png | Bar chart + radar comparison |
| 12_cross_validation.png | 5-fold CV score boxplots |

---

## Reference

> Syed, M. et al. "API Security Threat Detection Using Machine Learning." *IEEE CloudCom 2021.*
