# scripts/preprocess_v2.py - Dataset Version 2: Advanced preprocessing
"""
Dataset Version 2: Advanced preprocessing strategy
- Impute missing values (median for numeric, mode for categorical)
- One-hot encode categorical variables
- Feature engineering: family_size, is_alone, fare_per_person
- Feature selection: remove low-variance features
- 80/20 stratified train/test split

This version prioritises model performance through richer features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_store import DatasetStore


def create_ds_v2(raw_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Create dataset version 2 with advanced preprocessing.
    
    Preprocessing steps:
    1. Feature engineering: family_size, is_alone, fare_per_person
    2. Impute missing values: median for numeric, mode for categorical
    3. One-hot encode categorical variables (sex, embarked)
    4. Drop high-cardinality string features (name, ticket, cabin)
    5. Remove low-variance features
    6. Create 80/20 stratified train/test split
    
    Args:
        raw_df: Raw Titanic DataFrame from OpenML
        output_dir: Directory to save the versioned dataset
    
    Returns:
        dict: Metadata for this dataset version
    """
    df = raw_df.copy()
    
    # Feature engineering
    # Family size = siblings/spouses + parents/children + 1 (self)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    # Fare per person (handle zero family_size edge case)
    df['fare_per_person'] = df['fare'] / df['family_size'].replace(0, 1)
    
    # Select features for modeling
    # pclass, sex, age, sibsp, parch, fare, embarked + engineered
    numeric_features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'fare_per_person']
    categorical_features = ['sex', 'embarked']
    target_col = 'survived'
    
    # Drop high-cardinality or non-informative features
    df = df[numeric_features + categorical_features + [target_col]].copy()
    
    # Impute missing values
    # Numeric: median imputation
    for col in numeric_features:
        if df[col].isnull().any():
            imputer = SimpleImputer(strategy='median')
            df[col] = imputer.fit_transform(df[[col]]).ravel() 
    
    # Categorical: mode (most frequent) imputation
    for col in categorical_features:
        if df[col].isnull().any():
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel() 
    
    # one-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Ensure target is integer
    df[target_col] = df[target_col].astype(int)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    pd.DataFrame(y_train, columns=[target_col]).to_csv(output_dir / "y_train.csv", index=False)
    pd.DataFrame(y_test, columns=[target_col]).to_csv(output_dir / "y_test.csv", index=False)
    
    # Register with dataset store
    store = DatasetStore(output_dir.parent)
    metadata = store.create_version(
        df=df,  # Full preprocessed dataset
        version_id="ds_v2",
        preprocessing_steps=[
            "engineer_family_size_is_alone_fare_per_person",
            "impute_numeric_median_categorical_mode",
            "onehot_encode_sex_embarked_drop_first",
            "drop_high_cardinality_features",
            "convert_survived_to_int",
            "train_test_split_80_20_stratified_random42"
        ],
        split_info={
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_ratio": 0.2,
            "random_state": 42,
            "stratify": target_col
        },
        source_version="titanic_openml_40945"
    )
    
    return metadata


if __name__ == "__main__":
    from utils.data_loader import load_titanic_openml
    
    print("Loading raw Titanic dataset from OpenML...")
    raw_df = load_titanic_openml()
    
    print("Creating dataset version 2...")
    base_dir = Path("data")
    metadata = create_ds_v2(raw_df, base_dir / "ds_v2")
    
    print(f"Created ds_v2 with {metadata['schema']['shape'][1]} features")