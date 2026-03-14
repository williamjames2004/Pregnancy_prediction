from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# ----------------------------
# Load Model
# ----------------------------
app = Flask(__name__)

# Safe model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "pregnancy_model.pkl")

model = joblib.load(model_path)

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

    data = request.get_json()

    # Convert to DataFrame
    input_data = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 0:
        result = "Normal"
    else:
        result = "Complex"

    return jsonify({
        "prediction": result,
        "risk_probability": round(probability * 100, 2)
    })

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
