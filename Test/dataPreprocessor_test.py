# test_feature_engg.py
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source.DataPreprocessor import ThyroidDataPreprocessor

def test():
    cleaned_df = pd.read_csv("data/cleaned_thyroid.csv")  # your test CSV
    preprocessor = ThyroidDataPreprocessor(cleaned_df)
    train_df, test_df = preprocessor.split_and_resample()
    preprocessor.save_to_csv(train_df, test_df, 'train-V-1.csv', 'test-V-1.csv')

if __name__ == "__main__":
    test()