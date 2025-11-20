from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# Identify categorical & numerical columns
cat_cols = list(encoders.keys())
num_cols = list(scaler.feature_names_in_)

# Categorical mapping (backup)
label_mapping = {
    "sex": {"Male": 1, "Female": 0},

    "chest_pain_type": {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3
    },

    "fasting_blood_sugar": {
        "Lower than 120 mg/ml": 0,
        "Greater than 120 mg/ml": 1
    },

    "rest_ecg": {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2
    },

    "exercise_induced_angina": {"No": 0, "Yes": 1},

    "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},

    "vessels_colored_by_flourosopy": {
        "Zero": 0, "One": 1, "Two": 2, "Three": 3
    },

    "thalassemia": {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversable Defect": 3
    }
}

# Column order
columns = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure",
    "cholestoral", "fasting_blood_sugar", "rest_ecg", "Max_heart_rate",
    "exercise_induced_angina", "oldpeak", "slope",
    "vessels_colored_by_flourosopy", "thalassemia"
]

@app.route("/")
def home():
    return render_template("index.html", columns=columns)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "age": int(request.form["age"]),
            "sex": request.form["sex"],
            "chest_pain_type": request.form["chest_pain_type"],
            "resting_blood_pressure": int(request.form["resting_blood_pressure"]),
            "cholestoral": int(request.form["cholestoral"]),
            "fasting_blood_sugar": request.form["fasting_blood_sugar"],
            "rest_ecg": request.form["rest_ecg"],
            "Max_heart_rate": int(request.form["Max_heart_rate"]),
            "exercise_induced_angina": request.form["exercise_induced_angina"],
            "oldpeak": float(request.form["oldpeak"]),
            "slope": request.form["slope"],
            "vessels_colored_by_flourosopy": request.form["vessels_colored_by_flourosopy"],
            "thalassemia": request.form["thalassemia"]
        }

        df = pd.DataFrame([data])

        # Apply encoders
        for col in cat_cols:
            df[col] = encoders[col].transform(df[col])

        # Scale numeric columns
        df[num_cols] = scaler.transform(df[num_cols])

        # Predict
        prediction = model.predict(df)[0]

        result_label = "Low Risk 🙂" if prediction == 0 else "High Risk ⚠️"

        return render_template("result.html", prediction=result_label)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
