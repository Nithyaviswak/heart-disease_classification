"""
convert_to_onnx.py
==================
Trains an XGBoost Classifier on the Heart Disease dataset, then exports
the trained model to ONNX format for production-ready, portable inference.

Usage:
    python convert_to_onnx.py

Outputs:
    models/heart_disease_model.onnx   - ONNX model artifact
    models/scaler.pkl                 - Fitted StandardScaler
    models/feature_config.json        - Feature names, encoding maps, column order
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -- 1. Load Data -------------------------------------------------------------
print("=" * 60)
print("  Heart Disease - XGBoost -> ONNX Conversion Pipeline")
print("=" * 60)

DATA_PATH = "data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

# -- 2. Encode Categoricals --------------------------------------------------
categorical_cols = [
    "sex", "chest_pain_type", "fasting_blood_sugar", "rest_ecg",
    "exercise_induced_angina", "slope", "vessels_colored_by_flourosopy",
    "thalassemia",
]
numeric_cols = [
    "age", "resting_blood_pressure", "cholestoral", "Max_heart_rate", "oldpeak",
]

FEATURE_ORDER = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholestoral",
    "fasting_blood_sugar", "rest_ecg", "Max_heart_rate",
    "exercise_induced_angina", "oldpeak", "slope",
    "vessels_colored_by_flourosopy", "thalassemia",
]

encoders = {}
encoding_maps = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    encoding_maps[col] = {
        label: int(idx)
        for label, idx in zip(le.classes_, le.transform(le.classes_))
    }

print("[OK] Categorical columns label-encoded")
for col, mapping in encoding_maps.items():
    print(f"    {col}: {mapping}")

# -- 3. Scale Numerics --------------------------------------------------------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("[OK] Numeric columns standard-scaled")

# -- 4. Train / Test Split ----------------------------------------------------
X = df[FEATURE_ORDER]
y = df["target"]

print(f"[OK] Feature order ({len(FEATURE_ORDER)}): {FEATURE_ORDER}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -- 5. Train XGBoost --------------------------------------------------------
# Rename columns to f0..f12 so onnxmltools can parse the tree dump
feature_name_map = {name: f"f{i}" for i, name in enumerate(FEATURE_ORDER)}
X_train_r = X_train.rename(columns=feature_name_map)
X_test_r = X_test.rename(columns=feature_name_map)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

xgb_model.fit(X_train_r, y_train)

y_pred = xgb_model.predict(X_test_r)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n[OK] XGBoost trained  -  Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=["No Risk", "Risk"]))

# -- 6. Convert to ONNX ------------------------------------------------------
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

n_features = len(FEATURE_ORDER)
initial_type = [("float_input", FloatTensorType([None, n_features]))]

onnx_model = onnxmltools.convert_xgboost(
    xgb_model,
    initial_types=initial_type,
    target_opset=15,
)

ONNX_PATH = os.path.join(MODEL_DIR, "heart_disease_model.onnx")
onnxmltools.utils.save_model(onnx_model, ONNX_PATH)
print(f"[OK] ONNX model saved -> {ONNX_PATH}")

# -- 7. Validate ONNX against original model ---------------------------------
import onnxruntime as ort

sess = ort.InferenceSession(ONNX_PATH)
input_name = sess.get_inputs()[0].name
sample = X_test_r.iloc[:5].values.astype(np.float32)

onnx_preds = sess.run(None, {input_name: sample})
onnx_labels = onnx_preds[0]
xgb_labels = xgb_model.predict(X_test_r.iloc[:5])

print("\n-- ONNX Validation (first 5 test samples) --")
print(f"  XGBoost predictions: {xgb_labels.tolist()}")
print(f"  ONNX predictions:    {onnx_labels.tolist()}")
match = np.array_equal(xgb_labels, onnx_labels)
print(f"  Match: {'PASSED' if match else 'MISMATCH'}")

# -- 8. Save supporting artifacts ---------------------------------------------
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
pickle.dump(scaler, open(SCALER_PATH, "wb"))
print(f"\n[OK] Scaler saved -> {SCALER_PATH}")

ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
pickle.dump(encoders, open(ENCODERS_PATH, "wb"))
print(f"[OK] Encoders saved -> {ENCODERS_PATH}")

config = {
    "feature_order": FEATURE_ORDER,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "encoding_maps": encoding_maps,
}
CONFIG_PATH = os.path.join(MODEL_DIR, "feature_config.json")
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)
print(f"[OK] Feature config saved -> {CONFIG_PATH}")

print("\n" + "=" * 60)
print("  Pipeline complete. All artifacts saved to models/")
print("=" * 60)
