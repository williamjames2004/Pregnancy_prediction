import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def main():
    # Load dataset
    data = pd.read_csv("data.csv")

    X = data.drop("RiskLevel", axis=1)
    y = data["RiskLevel"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    # Example prediction
    new_data = pd.DataFrame(
        [[25, 130, 80, 15, 98, 46]],
        columns=X.columns
    )

    prediction = model.predict(new_data)
    risk_level = le.inverse_transform(prediction)[0]

    print("Predicted Risk Level:", risk_level)

    # Save model
    with open("risk_model.pkl", "wb") as file:
        pickle.dump(model, file)

    # Save label encoder
    with open("label_encoder.pkl", "wb") as file:
        pickle.dump(le, file)

    # Save feature names
    with open("feature_names.pkl", "wb") as file:
        pickle.dump(list(X.columns), file)

    print("Model, encoder, and feature names saved successfully!")


if __name__ == "__main__":
    main()
