import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import joblib
import os
import json
from datetime import datetime

def load_processed_data():
    print("Loading processed data...")
    
    train_path = '../../data/processed/train_data.csv'
    test_path = '../../data/processed/test_data.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed data files not found. Run data pipeline first.")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"Loaded training data: {train_data.shape}")
    print(f"Loaded test data: {test_data.shape}")
    
    return train_data, test_data

def prepare_features(train_data, test_data):
    print("Preparing features...")
    
    X_train = train_data.drop('digit', axis=1)
    y_train = train_data['digit']
    X_test = test_data.drop('digit', axis=1)
    y_test = test_data['digit']
    
    feature_names = list(X_train.columns)
    
    print(f"Features: {feature_names}")
    print(f"Target distribution in training set:")
    print(y_train.value_counts().sort_index())
    
    return X_train, X_test, y_train, y_test, feature_names

def train_model(X_train, y_train, params):
    print("Training model...")
    
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_test, y_test, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_macro': report['macro avg']['f1-score']
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    return metrics, y_pred, report

def log_model_with_mlflow(model, metrics, params, feature_names, X_test, y_test, y_pred):
    print("Logging to MLflow...")
    
    mlflow.set_tracking_uri("file:../../mlruns")
    
    with mlflow.start_run(run_name=f"digit_recognition_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="DigitRecognitionModel"
        )
        
        mlflow.log_text(
            json.dumps(feature_names, indent=2), 
            "feature_names.json"
        )
        
        model_info = {
            "model_type": "RandomForestClassifier",
            "features": feature_names,
            "target": "digit",
            "training_date": datetime.now().isoformat()
        }
        mlflow.log_text(
            json.dumps(model_info, indent=2), 
            "model_info.json"
        )
        
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(
            json.dumps(report, indent=2), 
            "classification_report.json"
        )
        
        print("Model logged to MLflow successfully!")

def save_model_locally(model, feature_names, metrics):
    print("Saving model locally...")
    
    os.makedirs('../../models', exist_ok=True)
    
    model_path = '../../models/digits_model.pkl'
    joblib.dump(model, model_path)

    feature_names_path = '../../models/feature_names.pkl'
    joblib.dump(feature_names, feature_names_path)

    metrics_path = '../../models/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Feature names saved to: {feature_names_path}")
    print(f"Metrics saved to: {metrics_path}")

def main():
    print("Starting model engineering pipeline...")
    
    train_data, test_data = load_processed_data()
    
    X_train, X_test, y_train, y_test, feature_names = prepare_features(train_data, test_data)
    
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    model = train_model(X_train, y_train, params)
    
    metrics, y_pred, report = evaluate_model(model, X_test, y_test)
    
    log_model_with_mlflow(model, metrics, params, feature_names, X_test, y_test, y_pred)
    
    save_model_locally(model, feature_names, metrics)
    
    print("Model engineering pipeline completed successfully!")
    return model, metrics

if __name__ == "__main__":
    main()
