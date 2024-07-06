# scripts/generate_synthetic_data.py

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import joblib

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import Generator

def load_generator_model(model_path):
    """
    Load the trained generator model from file.
    """
    return tf.keras.models.load_model(model_path, custom_objects={'Generator': Generator})

def load_anomaly_detection_model(model_path):
    """
    Load the trained anomaly detection model from file.
    """
    return joblib.load(model_path)

def generate_synthetic_data(generator, latent_dim, num_samples):
    """
    Generate synthetic data using the trained generator model.
    """
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    synthetic_data = generator.predict(random_latent_vectors)
    return synthetic_data

def detect_anomalies(data, anomaly_detection_model):
    """
    Detect anomalies in the data using the trained anomaly detection model.
    """
    predictions = anomaly_detection_model.predict(data)
    return predictions

def save_synthetic_data(synthetic_data, anomalies, output_path):
    """
    Save the generated synthetic data to a CSV file.
    """
    df = pd.DataFrame(synthetic_data)
    df['anomaly'] = anomalies
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")

if __name__ == "__main__":
    # Configuration
    model_path = os.path.join("models", "trained_generator.keras")
    anomaly_model_path = os.path.join("models", "anomaly_detection_model.joblib")
    output_file = os.path.join("data", "synthetic", "synthetic_vital_signs.csv")
    latent_dim = 100
    num_samples = 1000  # Number of synthetic samples to generate
    
    # Load the trained generator model
    generator = load_generator_model(model_path)
    
    # Load the trained anomaly detection model
    anomaly_detection_model = load_anomaly_detection_model(anomaly_model_path)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(generator, latent_dim, num_samples)
    
    # Detect anomalies
    anomalies = detect_anomalies(synthetic_data, anomaly_detection_model)
    
    # Save synthetic data with anomaly labels
    save_synthetic_data(synthetic_data, anomalies, output_file)






