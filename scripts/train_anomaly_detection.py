# scripts/train_anomaly_detection.py

import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def train_anomaly_detection(data, model_path):
    """
    Train an anomaly detection model.
    """
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data)
    joblib.dump(model, model_path)
    print(f"Anomaly detection model saved to {model_path}")

if __name__ == "__main__":
    anomaly_data_path = os.path.join("data", "processed", "anomaly_detection_data.csv")
    model_path = os.path.join("models", "anomaly_detection_model.joblib")
    
    # Load data
    data = load_data(anomaly_data_path)
    
    # Train anomaly detection model
    train_anomaly_detection(data, model_path)
