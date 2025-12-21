import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request, jsonify

# --- 1. MODEL INITIALIZATION AND TRAINING ---

try:
    data = pd.read_csv("data.csv")
except FileNotFoundError:
    print("FATAL ERROR: data.csv not found.")
    exit()

categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia']
numeric_cols = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']

df = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
MODEL_FEATURES = X.columns.tolist()

# --- 2. ADVICE LOGIC ---

def get_health_advice(prediction):
    if prediction == 1:
        return {
            "title": "Immediate Actions Required",
            "class": "text-red-700 bg-red-50 border-red-200",
            "items": [
                "Consult a cardiologist as soon as possible for a comprehensive evaluation.",
                "Follow any prescribed medication strictly and do not skip doses.",
                "Adopt a 'Heart-Healthy' diet: Reduce sodium, saturated fats, and processed sugars.",
                "Avoid strenuous physical activity until cleared by a medical professional.",
                "Monitor for symptoms like chest discomfort, unusual fatigue, or shortness of breath."
            ]
        }
    else:
        return {
            "title": "Maintenance & Prevention Tips",
            "class": "text-green-700 bg-green-50 border-green-200",
            "items": [
                "Maintain a regular exercise routine (at least 150 minutes of moderate activity per week).",
                "Keep a balanced diet rich in whole grains, lean proteins, and plenty of vegetables.",
                "Schedule annual check-ups to monitor blood pressure and cholesterol levels.",
                "Practice stress-management techniques like meditation or deep breathing exercises.",
                "Stay hydrated and ensure you get 7-9 hours of quality sleep daily."
            ]
        }

# --- 3. FLASK SETUP ---

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        input_data = {
            "age": float(form_data["age"]),
            "resting_blood_pressure": float(form_data["resting_blood_pressure"]),
            "cholestoral": float(form_data["cholestoral"]),
            "Max_heart_rate": float(form_data["Max_heart_rate"]),
            "oldpeak": float(form_data["oldpeak"]),
        }

        user_df_dict = {feature: 0 for feature in MODEL_FEATURES}
        for col in numeric_cols: user_df_dict[col] = input_data[col]

        # One-Hot Encoding mappings
        if form_data["sex"] == "Male": user_df_dict["sex_Male"] = 1
        if form_data["chest_pain_type"] != "Typical": user_df_dict[f"chest_pain_type_{form_data['chest_pain_type']}"] = 1
        if form_data["fasting_blood_sugar"] == "Yes": user_df_dict["fasting_blood_sugar_Yes"] = 1
        if form_data["rest_ecg"] != "Normal": user_df_dict[f"rest_ecg_{form_data['rest_ecg']}"] = 1
        if form_data["exercise_induced_angina"] == "Yes": user_df_dict["exercise_induced_angina_Yes"] = 1
        user_df_dict[f"slope_{form_data['slope']}"] = 1
        if form_data['vessels_colored_by_flourosopy'] != "0": user_df_dict[f"vessels_colored_by_flourosopy_{form_data['vessels_colored_by_flourosopy']}"] = 1
        if form_data["thalassemia"] != "Normal": user_df_dict[f"thalassemia_{form_data['thalassemia']}"] = 1

        user_df = pd.DataFrame([user_df_dict], columns=MODEL_FEATURES)
        user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

        prediction = int(model.predict(user_df)[0])
        prediction_proba = model.predict_proba(user_df)[0]
        
        advice = get_health_advice(prediction)
        result_text = "HIGH RISK ⚠️" if prediction == 1 else "LOW RISK ✅"
        
        return render_template('index.html', 
                               result=result_text, 
                               advice=advice,
                               prob_0=f"{prediction_proba[0]*100:.1f}%",
                               prob_1=f"{prediction_proba[1]*100:.1f}%",
                               form_data=form_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)