# Heart Disease Risk Prediction 🫀

An end-to-end machine learning pipeline for **binary classification of cardiovascular disease risk** from clinical tabular data — covering EDA, feature engineering, multi-model comparison, hyperparameter tuning, and deployment as a live REST API.

**Live Demo →** [heart-disease-classification-cptr.onrender.com](https://heart-disease-classification-cptr.onrender.com)

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Model Comparison & Results](#model-comparison--results)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [API Usage](#api-usage)
- [Key Findings](#key-findings)

---

## Overview

Cardiovascular disease is a leading cause of mortality globally. This project builds a **production-ready predictive analytics pipeline** that takes patient clinical features and outputs a binary risk prediction (disease / no disease) with associated probability scores.

The goal is not just to train a model, but to demonstrate the full data science workflow:

1. Exploratory Data Analysis (EDA) with hypothesis validation
2. Feature engineering and selection
3. Class imbalance handling
4. Multi-algorithm benchmarking
5. Hyperparameter tuning
6. Model evaluation with clinical-grade metrics
7. Production deployment via REST API

---

## Dataset

**Source:** UCI Heart Disease Dataset (Cleveland)  
**Records:** 303 patients  
**Features:** 13 clinical attributes + 1 target variable

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age in years |
| `sex` | Categorical | 1 = Male, 0 = Female |
| `cp` | Categorical | Chest pain type (0–3) |
| `trestbps` | Numerical | Resting blood pressure (mm Hg) |
| `chol` | Numerical | Serum cholesterol (mg/dl) |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl (1 = True) |
| `restecg` | Categorical | Resting ECG results (0–2) |
| `thalach` | Numerical | Maximum heart rate achieved |
| `exang` | Categorical | Exercise-induced angina (1 = Yes) |
| `oldpeak` | Numerical | ST depression induced by exercise |
| `slope` | Categorical | Slope of peak exercise ST segment |
| `ca` | Numerical | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Categorical | Thalassemia type (0 = normal, 1 = fixed defect, 2 = reversable defect) |
| `target` | Binary | 1 = Disease present, 0 = No disease |

---

## Project Structure

```
heart-disease_classification/
│
├── data/
│   ├── raw/                    # Original UCI dataset
│   └── processed/              # Cleaned, encoded, scaled data
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis & visualisation
│   ├── 02_preprocessing.ipynb  # Feature engineering pipeline
│   └── 03_modelling.ipynb      # Model training, tuning & evaluation
│
├── src/
│   ├── preprocessing.py        # Feature engineering functions
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Evaluation metrics & comparison
│   └── predict.py              # Inference logic
│
├── models/
│   └── best_model.pkl          # Serialised production model
│
├── app/
│   ├── main.py                 # FastAPI REST API
│   └── schemas.py              # Pydantic input/output schemas
│
├── Dockerfile                  # Container definition
├── requirements.txt
└── README.md
```

---

## ML Pipeline

### 1. Exploratory Data Analysis
- Distribution plots for all 13 clinical features
- Correlation heatmap to identify multicollinearity
- Target class distribution check → found **mild class imbalance** (54% disease, 46% no disease)
- Key finding: `cp` (chest pain type), `thalach` (max heart rate), and `ca` (vessel count) showed highest correlation with the target

### 2. Data Preprocessing & Feature Engineering
- **Missing value handling:** Imputed 6 missing values in `ca` and `thal` using median strategy
- **Encoding:** One-hot encoding for multi-class categorical features (`cp`, `restecg`, `thal`, `slope`)
- **Scaling:** StandardScaler applied to all numerical features to normalise distributions
- **Class imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on the training fold to prevent model bias
- **Feature selection:** Removed low-importance features using variance threshold + Random Forest feature importances

### 3. Model Training & Comparison
Trained and evaluated **5 classification algorithms** under identical conditions:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 85.2% | 84.1% | 87.3% | 85.7% | 0.921 |
| Random Forest | 88.5% | 87.9% | 90.1% | 89.0% | 0.941 |
| **XGBoost** | **90.2%** | **89.6%** | **91.8%** | **90.7%** | **0.956** |
| SVM (RBF) | 86.9% | 85.4% | 88.6% | 87.0% | 0.934 |
| Decision Tree | 81.0% | 79.3% | 83.7% | 81.4% | 0.882 |

### 4. Hyperparameter Tuning
**GridSearchCV** with 5-fold stratified cross-validation on the XGBoost model:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
```

Best parameters: `n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8`

---

## Model Comparison & Results

**Selected Model:** XGBoost Classifier  
**Reason:** Highest ROC-AUC (0.956) and F1 score (90.7%) with balanced Precision-Recall

For medical diagnostic tasks, **Recall (sensitivity)** is prioritised over raw Accuracy — a false negative (missing a true case) carries higher clinical risk than a false positive.

> XGBoost achieved **91.8% Recall**, meaning it correctly identified 91.8% of all actual disease cases.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| ML Framework | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| API | FastAPI + Pydantic |
| Containerisation | Docker |
| Deployment | Render |

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Nithyaviswak/heart-disease_classification.git
cd heart-disease_classification

# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn app.main:app --reload --port 8000

# Or run with Docker
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

---

## API Usage

**POST** `/predict`

```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease Detected",
  "probability": 0.847,
  "risk_level": "High"
}
```

---

## Key Findings

- **Chest pain type (`cp`)** is the strongest single predictor — asymptomatic chest pain (type 0) was strongly associated with disease presence
- **Maximum heart rate (`thalach`)** showed an inverse relationship with disease risk — lower max heart rate correlated with higher disease probability
- **Number of vessels (`ca`)** — patients with 0 coloured vessels had markedly lower disease risk
- **Age alone** was a moderate predictor; combined with exercise-related features it became significantly stronger
- SMOTE had a marginal but measurable improvement (+1.2% Recall) on the minority class without overfitting

---

## Author

**Nithyananda Chari R**   
[LinkedIn](https://linkedin.com/in/nithyananda1311) · [GitHub](https://github.com/Nithyaviswak)****
