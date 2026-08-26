from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS
import random
import smtplib
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash
from routes import user_routes
from models import users_collection

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "pregnancy_risk_model.pkl")

model = joblib.load(model_path)

MODEL_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]    

def generate_tips(data):

    tips      = []
    age       = int(data.get("age", 0))
    bs        = int(data.get("bs", 0))
    bodytemp  = int(data.get("bodytemp", 0))
    heartrate = int(data.get("heartrate", 0))

    sleep     = int(data.get("sleep", 0))
    stress    = int(data.get("stress", 0))
    water     = int(data.get("water", 0))
    junk      = int(data.get("junk", 0))
    hemo      = int(data.get("homoglobin", 0))
    sys       = int(data.get("sbp", 0))
    dia       = int(data.get("dbp", 0))
    gest      = int(data.get("gestationalage", 0))

    heart     = int(data.get("heartdisease", 0))
    asthma    = int(data.get("asthma", 0))

    activity  = data.get("activity", "")
    protein   = data.get("protein", "")

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
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is empty"
            }), 400

        # -----------------------------------
        # Required fields for ML prediction
        # -----------------------------------

        required_fields = [
            "age",
            "sbp",
            "dbp",
            "bs",
            "bodytemp",
            "heartrate"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in data or data[field] in [None, ""]
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required prediction fields",
                "missing_fields": missing_fields
            }), 400

        # -----------------------------------
        # Prepare ONLY the 6 ML parameters
        # -----------------------------------

        model_input = {
            "Age": data["age"],
            "SystolicBP": data["sbp"],
            "DiastolicBP": data["dbp"],
            "BS": data["bs"],
            "BodyTemp": data["bodytemp"],
            "HeartRate": data["heartrate"]
        }

        df = pd.DataFrame([model_input])

        # Convert values to numeric
        for col in MODEL_COLUMNS:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # Check invalid values
        if df[MODEL_COLUMNS].isnull().any().any():
            return jsonify({
                "error": "Prediction parameters must be numeric"
            }), 400

        # -----------------------------------
        # ML Prediction
        # -----------------------------------

        prediction = model.predict(df)[0]

        probabilities = model.predict_proba(df)[0]

        classes = model.classes_

        prob_dict = {
            str(classes[i]): round(
                probabilities[i] * 100,
                2
            )
            for i in range(len(classes))
        }

        confidence = prob_dict[str(prediction)]

        # -----------------------------------
        # Generate health tips
        # using ALL Flutter data
        # -----------------------------------

        tips, general_tips = generate_tips(data)

        # -----------------------------------
        # Response
        # -----------------------------------

        return jsonify({
            "prediction": str(prediction),
            "confidence": confidence,
            "probabilities": prob_dict,
            "tips": tips,
            "general_tips": general_tips
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Invalid input format"
        }), 400
    
otp_store = {}
def send_email(to_email, otp):
    sender_email = "williamjames4219@gmail.com"
    sender_password = "fnev gzan hxuy txnw"

    subject = "Your OTP Code"
    body = f"Your OTP is {otp}. It will expire in 5 minutes."

    message = f"Subject: {subject}\n\n{body}"

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message)

@app.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.json
    email = data.get("email")

    if not email:
        return jsonify({"error": "Email required"}), 400

    otp = str(random.randint(100000, 999999))

    otp_store[email] = {
        "otp": otp,
        "expires": datetime.now() + timedelta(minutes=5)
    }

    send_email(email, otp)

    return jsonify({"message": "OTP sent successfully"})

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json
    email = data.get("email")
    user_otp = data.get("otp")

    record = otp_store.get(email)

    if not record:
        return jsonify({"error": "No OTP found"}), 400

    if datetime.now() > record["expires"]:
        return jsonify({"error": "OTP expired"}), 400

    if record["otp"] != user_otp:
        return jsonify({"error": "Invalid OTP"}), 400

    # Mark as verified
    record["verified"] = True

    return jsonify({"message": "OTP verified"})

users = {
    "test@gmail.com": {
        "password": "old_hash"
    }
}

@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.json
    email = data.get("email")
    new_password = data.get("password")
    confirm_password = data.get("confirm_password")

    if not email or not new_password or not confirm_password:
        return jsonify({"error": "All fields are required"}), 400

    if new_password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400

    # ✅ Check OTP verification
    record = otp_store.get(email)
    if not record or not record.get("verified"):
        return jsonify({"error": "OTP not verified"}), 400

    # 🔍 Find user in MongoDB
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # 🔐 Hash password
    hashed_password = generate_password_hash(new_password)

    # 🔄 Update password in DB
    users_collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_password}}
    )

    # 🧹 Clear OTP
    otp_store.pop(email, None)

    return jsonify({"message": "Password reset successful"})

app.register_blueprint(user_routes, url_prefix="/users")
# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
