# scripts/preprocess_v3.py - Dataset Version 3: Minimal preprocessing
"""
Dataset Version 3: Minimal preprocessing strategy
- Keep all available numeric features
- Simple mean imputation for missing values
- No encoding (use pandas categorical dtype)
- No feature engineering
- 60/40 split to have more training data

This version tests the impact of minimal preprocessing on model performance.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_store import DatasetStore


def create_ds_v3(raw_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Create dataset version 3 with minimal preprocessing.
    
    Preprocessing steps:
    1. Keep all numeric features from original dataset
    2. Mean imputation for all missing numeric values
    3. Convert categorical to pandas category dtype (no one-hot)
    4. No feature engineering or selection
    5. 60/40 stratified train/test split (more training data)
    
    Args:
        raw_df: Raw Titanic DataFrame from OpenML
        output_dir: Directory to save the versioned dataset
    
    Returns:
        dict: Metadata for this dataset version
    """
    df = raw_df.copy()
    
    # Select numeric features + target + key categoricals
    # Keep features that are numeric or can be easily encoded
    numeric_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    categorical_cols = ['sex', 'embarked']  # Keep as categories
    target_col = 'survived'
    
    df = df[numeric_cols + categorical_cols + [target_col]].copy()
    
    # Impute missing numeric values with mean
    imputer = SimpleImputer(strategy='mean')
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = imputer.fit_transform(df[[col]]).ravel() 
    
    # Convert categoricals to pandas category dtype
    # This preserves them as categorical without one-hot encoding
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Ensure target is integer
    df[target_col] = df[target_col].astype(int)
    
    # Prepare features and target
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Stratified train/test split (60/40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,  # More training data
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
        df=df,
        version_id="ds_v3",
        preprocessing_steps=[
            "select_numeric_and_key_categorical_features",
            "mean_imputation_for_numeric_missing",
            "convert_categorical_to_pandas_category_dtype",
            "no_feature_engineering_or_selection",
            "convert_survived_to_int",
            "train_test_split_60_40_stratified_random42"
        ],
        split_info={
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_ratio": 0.4,
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
    
    print("Creating dataset version 3...")
    base_dir = Path("data")
    metadata = create_ds_v3(raw_df, base_dir / "ds_v3")
    
    print(f"Created ds_v3 with {metadata['schema']['shape'][1]} features")