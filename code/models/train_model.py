import os
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


def train_model():
    print("Loading digits dataset...")
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model (SVC)...")
    model = SVC(gamma=0.001, probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs('../../models', exist_ok=True)
    model_path = '../../models/digits_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model, acc


if __name__ == "__main__":
    train_model()
