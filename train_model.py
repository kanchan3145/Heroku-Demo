"""
train_model.py
Trains a simple classifier on the Iris dataset and saves it to disk
as model.joblib, ready to be loaded by app.py for serving predictions.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    # 1. Load data
    data = load_iris()
    X, y = data.data, data.target
    class_names = data.target_names.tolist()
    feature_names = data.feature_names

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=class_names))

    # 5. Save model + metadata together
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "class_names": class_names,
        },
        "model.joblib",
    )
    print("Saved model to model.joblib")

if __name__ == "__main__":
    main()
