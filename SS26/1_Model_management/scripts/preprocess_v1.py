# scripts/preprocess_v1.py - Dataset Version 1: Basic preprocessing
"""
Dataset Version 1: Basic preprocessing strategy
- Drop rows with missing values in critical columns
- Label encode categorical variables
- Select core features only
- 70/30 stratified train/test split

This version prioritises simplicity and reproducibility.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_store import DatasetStore


def create_ds_v1(raw_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Create dataset version 1 with basic preprocessing.
    
    Preprocessing steps:
    1. Select core features: pclass, sex, age, sibsp, parch, fare, embarked
    2. Drop rows with missing values in 'age' or 'embarked'
    3. Label encode 'sex' and 'embarked' (ordinal encoding)
    4. Create 70/30 stratified train/test split on 'survived'
    
    Args:
        raw_df: Raw Titanic DataFrame from OpenML
        output_dir: Directory to save the versioned dataset
    
    Returns:
        dict: Metadata for this dataset version
    """
    df = raw_df.copy()
    
    # Select core features + target
    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target_col = 'survived'
    
    # Keep only relevant columns
    df = df[feature_cols + [target_col]].copy()
    
    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['age', 'embarked', 'fare', target_col])
    
    # Label encode categorical variables
    # Note: LabelEncoder assigns arbitrary ordinal values; for production,
    # consider OneHotEncoder or target encoding
    for col in ['sex', 'embarked']:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            # Handle potential NaNs (shouldn't exist after dropna, but be safe)
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
    
    # Prepare features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy().astype(int)
    
    # Stratified train/test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42, 
        stratify=y  # Preserve class distribution
    )
    
    # Save train/test splits
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train = pd.DataFrame(y_train, columns=[target_col])
    y_test = pd.DataFrame(y_test, columns=[target_col])
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    
    # Create dataset store and register this version
    store = DatasetStore(output_dir.parent)
    
    # For versioning, we save the full preprocessed dataset (before split)
    # The split info is tracked in metadata
    metadata = store.create_version(
        df=df,  # Full preprocessed dataset
        version_id="ds_v1",
        preprocessing_steps=[
            "select_core_features_7",
            "drop_na_age_embarked_fare_survived",
            "label_encode_sex_embarked",
            "convert_survived_to_int",
            "train_test_split_70_30_stratified_random42"
        ],
        split_info={
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_ratio": 0.3,
            "random_state": 42,
            "stratify": target_col,
            "train_path": "ds_v1/X_train.csv",
            "test_path": "ds_v1/X_test.csv"
        },
        source_version="titanic_openml_40945"
    )
    
    return metadata


if __name__ == "__main__":
    # Standalone execution for testing
    from utils.data_loader import load_titanic_openml
    
    print("Loading raw Titanic dataset from OpenML...")
    raw_df = load_titanic_openml()
    
    print("Creating dataset version 1...")
    base_dir = Path("data")
    metadata = create_ds_v1(raw_df, base_dir / "ds_v1")
    
    print("Dataset v1 Meta")
    print(f"  Version ID: {metadata['dataset_version_id']}")
    print(f"  Shape: {metadata['schema']['shape']}")
    print(f"  Features: {metadata['schema']['columns']}")
    print(f"  Preprocessing: {len(metadata['preprocessing_steps'])} steps")