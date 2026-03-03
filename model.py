import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ---------------------------
# 1️⃣ Load Dataset
# ---------------------------
df = pd.read_excel("sample_dataset.xlsx")

# ---------------------------
# 2️⃣ Separate Features & Target
# ---------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]  # Keep as string (Normal/Complex)

# ---------------------------
# 3️⃣ Define Column Groups
# ---------------------------

numeric_features = [
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
    "Junk"
]

binary_features = ["Multiple_babies"]

ordinal_features = ["Activity", "Protein"]

nominal_features = ["Thyroid"]

# ---------------------------
# 4️⃣ Define Encoders
# ---------------------------

ordinal_encoder = OrdinalEncoder(categories=[
    ["Low", "Moderate", "High"],      # Activity
    ["Low", "Adequate", "High"]       # Protein
])

binary_encoder = OrdinalEncoder(categories=[
    ["No", "Yes"]
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("bin", binary_encoder, binary_features),
        ("ord", ordinal_encoder, ordinal_features),
        ("nom", OneHotEncoder(drop="first"), nominal_features)
    ]
)

# ---------------------------
# 5️⃣ Create Pipeline
# ---------------------------

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
])

# ---------------------------
# 6️⃣ Train-Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 7️⃣ Train Model
# ---------------------------

pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, "pregnancy_model.pkl")

# ---------------------------
# 8️⃣ Evaluate Model
# ---------------------------

y_pred = pipeline.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="Complex"))
print("Recall   :", recall_score(y_test, y_pred, pos_label="Complex"))
print("F1 Score :", f1_score(y_test, y_pred, pos_label="Complex"))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))