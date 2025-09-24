"""
Data Engineering Pipeline using DVC
Loads, cleans, and splits handwritten digit data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import dvc.api

def load_raw_data():
    print("Loading raw data...")
    from sklearn import datasets
    
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    feature_names = [f'pixel_{i}' for i in range(64)]
    df = pd.DataFrame(X, columns=feature_names)
    df['digit'] = y
    
    print(f"Loaded {len(df)} digit samples with {len(feature_names)} pixel features")
    print(f"Digit distribution: {df['digit'].value_counts().sort_index().to_dict()}")
    
    return df

def clean_data(df):
    print("Cleaning data...")

    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values, filling with median...")
        for column in df.columns[:-1]:
            if df[column].isnull().sum() > 0:
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
                print(f"Filled missing values in {column} with median: {median_value:.3f}")
    else:
        print("No missing values found in the dataset")

    print("Digit dataset is already clean - no outlier removal needed")
    
    return df

def split_data(df):
    print("Splitting data...")
    
    X = df.drop('digit', axis=1)
    y = df['digit']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Feature distribution in training set:")
    print(y_train.value_counts().sort_index())
    
    return train_data, test_data

def save_processed_data(train_data, test_data):
    print("Saving processed data...")

    os.makedirs('../../data/processed', exist_ok=True)

    train_path = '../../data/processed/train_data.csv'
    test_path = '../../data/processed/test_data.csv'
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    
    return train_path, test_path

def main():
    print("Starting data engineering pipeline...")
    
    raw_data = load_raw_data()
    
    os.makedirs('../../data/raw', exist_ok=True)
    raw_data.to_csv('../../data/raw/digit_data.csv', index=False)
    print("Raw data saved to: ../../data/raw/digit_data.csv")

    cleaned_data = clean_data(raw_data.copy())
    train_data, test_data = split_data(cleaned_data)
    train_path, test_path = save_processed_data(train_data, test_data)
    
    print("Data engineering pipeline completed successfully!")
    return train_path, test_path

if __name__ == "__main__":
    main()
