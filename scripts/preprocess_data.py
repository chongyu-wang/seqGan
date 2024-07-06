# scripts/preprocess_data.py

import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def normalize_data(df):
    """
    Normalize the data to the range [-1, 1].
    """
    return (df - df.min()) / (df.max() - df.min()) * 2 - 1

def fill_missing_values(df):
    """
    Fill missing values with the column median.
    """
    return df.fillna(df.median())

def preprocess_data(input_path, output_path, anomaly_output_path):
    """
    Load, normalize, and save the processed data.
    """
    # Load data
    df = load_data(input_path)
    
    # Select relevant columns for vital signs
    vital_signs_columns = ['age', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin']
    vital_signs_df = df[vital_signs_columns]
    
    # Fill missing values
    vital_signs_df_filled = fill_missing_values(vital_signs_df)
    
    # Normalize data
    df_normalized = normalize_data(vital_signs_df_filled)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data
    df_normalized.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    #Save anomaly detection data
    df_normalized.to_csv(anomaly_output_path, index=False)
    print(f"Anomaly detection data saved to {anomaly_output_path}")

if __name__ == "__main__":
    input_file = os.path.join("data", "pbc2_cleaned.csv")  #Ensure this file exists
    output_file = os.path.join("data", "processed", "patient_vital_signs_normalized.csv")
    anomaly_output_file = os.path.join("data", "processed", "anomaly_detection_data.csv")
    
    preprocess_data(input_file, output_file, anomaly_output_file)




