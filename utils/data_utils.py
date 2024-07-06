# utils/data_utils.py

import pandas as pd
import numpy as np

def load_csv(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def save_csv(data, file_path):
    """
    Save data to a CSV file.
    """
    data.to_csv(file_path, index=False)

def normalize_data(df):
    """
    Normalize the data to the range [-1, 1].
    """
    return (df - df.min()) / (df.max() - df.min()) * 2 - 1

