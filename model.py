import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load new dataset
df = pd.read_csv("data.csv")

# Features & Target
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]   # low risk / mid risk / high risk

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = DecisionTreeClassifier(max_depth=4, random_state=42)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "pregnancy_risk_model.pkl")

print("Model retrained and saved successfully!")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance\n")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred, average='weighted'))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
