from flask import Flask, render_template, request
import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

categorical_cols = [
    'sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
    'exercise_induced_angina', 'slope', 'thalassemia',
    'vessels_colored_by_flourosopy'
]

numeric_cols = [
    'age', 'resting_blood_pressure', 'cholestoral',
    'Max_heart_rate', 'oldpeak'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = {col: request.form[col] for col in numeric_cols + categorical_cols}

    # Convert numeric values
    for col in numeric_cols:
        data[col] = float(data[col])

    df = pd.DataFrame([data])

    # Encode categorical
    for c in categorical_cols:
        df[c] = encoders[c].transform(df[c])

    # Scale numeric
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    result = "Likely to have heart disease" if pred == 1 else "Unlikely to have heart disease"

    return render_template("result.html", result=result, prob=round(prob * 100, 2), data=data)


if __name__ == "__main__":
    app.run(debug=True)
