import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

age = int(input("Age: "))
sex = input("Sex (Male/Female): ").strip().lower()
chest = input("Chest Pain Type (Typical, Atypical, Non-anginal, Asymptomatic): ").strip().title()
rbp = float(input("Resting Blood Pressure: "))
chol = float(input("Cholestoral: "))
fbs = input("Fasting Blood Sugar >120 mg/dl? (Yes/No): ").strip().lower()
restecg = input("Rest ECG (Normal, ST-T abnormality, LV hypertrophy): ").strip().title()
mhr = float(input("Max Heart Rate: "))
exa = input("Exercise Induced Angina (Yes/No): ").strip().lower()
oldpeak = float(input("Oldpeak: "))
slope = input("Slope (Upsloping, Flat, Downsloping): ").strip().title()
vessels = int(input("Vessels Colored by Fluoroscopy (0,1,2,3): "))
thal = input("Thalassemia (Normal, Fixed Defect, Reversable Defect): ").strip().title()

data = {
    'age': [age],
    'sex': [sex],
    'chest_pain_type': [chest],
    'resting_blood_pressure': [rbp],
    'cholestoral': [chol],
    'fasting_blood_sugar': [fbs],
    'rest_ecg': [restecg],
    'Max_heart_rate': [mhr],
    'exercise_induced_angina': [exa],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'vessels_colored_by_flourosopy': [vessels],
    'thalassemia': [thal]
}

df = pd.DataFrame(data)

for col, enc in encoders.items():
    df[col] = enc.transform(df[col])

numeric_cols = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak', 'vessels_colored_by_flourosopy']
df[numeric_cols] = scaler.transform(df[numeric_cols])

prediction = model.predict(df)[0]

print("\nHeart Disease Risk:", "YES" if prediction == 1 else "NO")
