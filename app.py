from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)
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

def generate_tips(data):

    tips = []

    sleep = int(data.get("Sleep", 0))
    stress = int(data.get("Stress", 0))
    water = int(data.get("Water", 0))
    junk = int(data.get("Junk", 0))
    hemo = int(data.get("Hemoglobin", 0))
    sys = int(data.get("Systolic Blood Pressure", 0))
    dia = int(data.get("Diastolic Blood Pressure", 0))
    gest = int(data.get("Gestational Age", 0))

    heart = int(data.get("Heart Disease", 0))
    asthma = int(data.get("Asthma", 0))

    activity = data.get("Activity", "")
    protein = data.get("Protein", "")

    # 1️⃣ Sleep
    if sleep < 6:
        tips.append("Sleep duration is low. Pregnant women should aim for 7–9 hours of sleep.")
    elif sleep <= 9:
        tips.append("Your sleep duration is healthy. Continue maintaining 7–9 hours daily.")
    else:
        tips.append("You are sleeping more than usual. Ensure balanced rest and daily activity.")

    # 2️⃣ Stress
    if stress > 3:
        tips.append("Stress level appears high. Relaxation techniques like meditation may help.")
    elif stress >= 1:
        tips.append("Stress level is moderate. Maintain relaxation routines and good rest.")
    else:
        tips.append("Stress level is very low. Maintaining emotional well-being is excellent.")

    # 3️⃣ Water
    if water < 3:
        tips.append("Water intake seems low. Aim for at least 2–3 liters daily.")
    elif water <= 5:
        tips.append("Water intake is good. Staying hydrated supports pregnancy health.")
    else:
        tips.append("High water intake detected. Ensure electrolyte balance as well.")

    # 4️⃣ Junk Food
    if junk > 2:
        tips.append("Frequent junk food intake detected. Try to reduce processed foods.")
    elif junk == 1 or junk == 2:
        tips.append("Occasional junk food intake is acceptable, but maintain a balanced diet.")
    else:
        tips.append("Great job avoiding junk food. Healthy nutrition supports fetal growth.")

    # 5️⃣ Hemoglobin
    if hemo < 11:
        tips.append("Hemoglobin level appears low. Include iron-rich foods like spinach and lentils.")
    elif hemo <= 14:
        tips.append("Hemoglobin level is healthy. Continue maintaining a balanced diet.")
    else:
        tips.append("Hemoglobin level is high. Ensure balanced iron intake and hydration.")

    # 6️⃣ Blood Pressure
    if sys > 140 or dia > 90:
        tips.append("Blood pressure appears elevated. Regular monitoring is recommended.")
    elif sys >= 100 and dia >= 60:
        tips.append("Blood pressure is within healthy range.")
    else:
        tips.append("Blood pressure appears low. Ensure adequate nutrition and hydration.")

    # 7️⃣ Activity Level
    if activity == "Low":
        tips.append("Low physical activity detected. Light walking can improve maternal health.")
    elif activity == "Moderate":
        tips.append("Moderate activity level is ideal for pregnancy. Keep maintaining it.")
    else:
        tips.append("High activity level detected. Ensure you avoid excessive strain.")

    # 8️⃣ Protein Intake
    if protein == "Low":
        tips.append("Protein intake appears low. Include eggs, beans, or dairy products.")
    elif protein == "Adequate":
        tips.append("Protein intake is adequate. This supports healthy fetal development.")
    else:
        tips.append("High protein intake detected. Maintain balanced nutrition.")

    # 9️⃣ Asthma
    if asthma == 1:
        tips.append("Asthma condition detected. Follow medical guidance and keep inhalers available.")
    else:
        tips.append("No asthma condition detected. Continue maintaining respiratory health.")

    # 🔟 Heart Disease
    if heart == 1:
        tips.append("Heart disease history detected. Regular cardiology checkups are recommended.")
    else:
        tips.append("No heart disease reported. Maintaining heart health is important.")

    # 1️⃣1️⃣ Gestational Age
    if gest < 12:
        tips.append("Early pregnancy stage. Proper nutrition and prenatal care are important.")
    elif gest <= 34:
        tips.append("Mid pregnancy stage. Continue regular checkups and balanced nutrition.")
    else:
        tips.append("Late pregnancy stage detected. Frequent monitoring and rest are essential.")

    # 4 General Tips
    general_tips_pool = [
        "Attend regular prenatal checkups with your healthcare provider.",
        "Maintain a balanced diet including fruits and vegetables.",
        "Take prenatal vitamins as prescribed by your doctor.",
        "Stay hydrated and drink enough water daily.",
        "Ensure you get enough rest and sleep.",
        "Practice light physical activity like walking if recommended.",
        "Avoid smoking and alcohol during pregnancy.",
        "Include iron-rich foods such as spinach and lentils.",
        "Consume calcium-rich foods like milk and yogurt.",
        "Manage stress through relaxation techniques.",
        "Maintain a healthy weight during pregnancy.",
        "Keep track of fetal movements regularly.",
        "Follow medical advice for medications and supplements.",
        "Avoid excessive caffeine intake.",
        "Maintain good hygiene and food safety practices.",
        "Eat small and frequent meals to avoid nausea.",
        "Include whole grains and fiber to support digestion.",
        "Wash fruits and vegetables properly before eating.",
        "Avoid raw or undercooked foods.",
        "Wear comfortable clothing and supportive footwear.",
        "Practice deep breathing or prenatal yoga if approved by your doctor.",
        "Sleep on your side during later stages of pregnancy.",
        "Avoid heavy lifting or strenuous activities.",
        "Monitor your weight gain according to medical guidance.",
        "Keep emergency contact numbers readily available.",
        "Prepare a birth plan and discuss it with your doctor.",
        "Limit processed foods and sugary drinks.",
        "Spend time outdoors for fresh air and sunlight.",
        "Ensure proper intake of folic acid during pregnancy.",
        "Stay positive and maintain emotional well-being.",
        "Avoid self-medication without consulting a doctor.",
        "Maintain good posture while sitting and standing.",
        "Take short breaks when working for long periods.",
        "Drink warm fluids if experiencing mild discomfort.",
        "Avoid exposure to harmful chemicals and smoke.",
        "Maintain a clean and comfortable sleeping environment.",
        "Discuss any unusual symptoms with your doctor immediately.",
        "Include healthy snacks like nuts and fruits.",
        "Keep yourself informed about pregnancy health.",
        "Prepare essentials for the baby's arrival in advance.",
        "Stay connected with family or support groups for emotional support.",
        "Follow safe travel practices during pregnancy.",
        "Keep track of important medical records and reports.",
        "Ensure regular dental checkups during pregnancy.",
        "Maintain gentle daily stretching if approved by a healthcare provider."
        ]
    
    general_tips = random.sample(general_tips_pool, 4)
    return tips, general_tips

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
        tips, general_tips = generate_tips(data)

        return jsonify({
            "prediction": result,
            "risk_probability": round(probability * 100, 2),
            "tips": tips,
            "general_tips": general_tips
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
