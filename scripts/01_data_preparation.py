"""
Data Preparation Script for AI Job Salary Analysis
-------------------------------------------------
This script loads the raw dataset, performs professional data cleaning,
handles missing values and duplicates, converts data types, and saves
the cleaned dataset for further analysis.

Author: Franciszek Tokarek
"""

import pandas as pd
import numpy as np
import os

# Define input and output file paths
RAW_DATA_PATH = 'data/ai_job_dataset.csv'
CLEAN_DATA_PATH = 'data/ai_job_dataset_clean.csv'

def load_data(path):
    """Load the dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded data shape: {df.shape}")
    return df

def basic_overview(df):
    """Print basic information about the dataset."""
    print("\n--- Basic Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Duplicates ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

def clean_data(df):
    """Perform data cleaning: remove duplicates, handle missing values, convert types."""
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Example: Fill missing numerical values with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"Filled missing values in '{col}' with median: {median}")

    # Example: Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            print(f"Filled missing values in '{col}' with mode: {mode}")

    # Convert columns to appropriate types (customize as needed)
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    # Example: Convert date columns if present
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"Converted '{col}' to datetime.")
            except Exception:
                pass

    return df

def save_data(df, path):
    """Save the cleaned dataset to a CSV file."""
    df.to_csv(path, index=False)
    print(f"\nCleaned data saved to: {path}")

def main():
    df = load_data(RAW_DATA_PATH)
    basic_overview(df)
    df_clean = clean_data(df)
    print("\n--- Cleaned Data Overview ---")
    print(df_clean.info())
    save_data(df_clean, CLEAN_DATA_PATH)

if __name__ == "__main__":
    main()