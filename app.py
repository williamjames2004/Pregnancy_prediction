from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "pregnancy_model.pkl")

model = joblib.load(model_path)

# Expected column order
EXPECTED_COLUMNS = [
    "Age", "Ht", "Wt",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Hemoglobin",
    "Heart Disease",
    "Asthma",
    "Previous_complicated_status",
    "Previous_misscarraige",
    "Gestational Age",
    "Sleep",
    "Stress",
    "Water",
    "Junk",
    "Multiple_babies",
    "Activity",
    "Protein",
    "Thyroid"
]

# ----------------------------
# Home Route
# ----------------------------
@app.route("/")
def home():
    return "Pregnancy Risk Prediction API is running!"

# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Ensure correct column order
        df = df.reindex(columns=EXPECTED_COLUMNS)

        # Convert numeric columns safely
        numeric_cols = [
            "Age","Ht","Wt","Systolic Blood Pressure","Diastolic Blood Pressure",
            "Hemoglobin","Heart Disease","Asthma","Previous_complicated_status",
            "Previous_misscarraige","Gestational Age","Sleep","Stress","Water","Junk"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill missing values
        df.fillna(0, inplace=True)

        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        result = "Normal" if prediction == "Normal" else "Complex"

        return jsonify({
            "prediction": result,
            "risk_probability": round(probability * 100, 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e),
            "message": "Invalid input format"
        }), 400


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
