# tests/test_preprocessing.py

import unittest
import pandas as pd
import numpy as np
from scripts.preprocess_data import normalize_data

class TestPreprocessing(unittest.TestCase):

    def test_normalize_data(self):
        df = pd.DataFrame({
            'vital1': [0, 1, 2],
            'vital2': [1, 2, 3],
            'vital3': [2, 3, 4]
        })
        normalized_df = normalize_data(df)
        self.assertTrue(np.allclose(normalized_df.min(), -1))
        self.assertTrue(np.allclose(normalized_df.max(), 1))

if __name__ == "__main__":
    unittest.main()
