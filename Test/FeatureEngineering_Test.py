import os
import sys
import pandas as pd

# Add the parent folder to path so `source` module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.data_featureEngg import ThyroidFeatureEngineer  # Update this if file name is different

def test_thyroid_feature_engineer():
    # Define file paths
    input_path = "data/raw_thyroid.csv"
    output_path = "data/cleaned_thyroid.csv"

    # Load input CSV
    df = pd.read_csv(input_path)

    # Instantiate and clean data
    engineer = ThyroidFeatureEngineer(df)
    
    cleaned_df = engineer.clean_and_save(output_path)

    # Assert that output file was created
    assert os.path.exists(output_path), "[FAIL] Cleaned CSV was not saved."

    # Show top few rows
    print("[INFO] Cleaned Data Preview:")
    print(cleaned_df.head())

    # Basic test: check that 'class' column exists
    assert 'class' in cleaned_df.columns, "[FAIL] Target column was not mapped to 'class'."
    # Check for NaN values in critical columns
    critical_columns = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'class']
    for col in critical_columns:
        assert cleaned_df[col].notna().all(), f"[FAIL] NaN values found in column: {col}"   
    print("[PASS] All basic checks passed!")

if __name__ == "__main__":
    test_thyroid_feature_engineer()
