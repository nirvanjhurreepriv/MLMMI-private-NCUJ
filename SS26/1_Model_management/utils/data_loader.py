# utils/data_loader.py - Load Titanic dataset from OpenML
"""
Module to load the Titanic dataset from OpenML (ID: 40945).
The dataset is in ARFF format and contains passenger information
with the target variable 'survived' (0 = did not survive, 1 = survived).
"""

import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_titanic_openml(cache_dir: Path = None, as_frame: bool = True) -> pd.DataFrame:
    """
    Load the Titanic dataset from OpenML using scikit-learn's fetch_openml.
    
    Args:
        cache_dir: Directory to cache the downloaded dataset. If None, uses sklearn default.
        as_frame: If True, return as pandas DataFrame
    
    Returns:
        pd.DataFrame: The full Titanic dataset with all features and target.
    
    Note:
        OpenML dataset ID: 40945
        Format: ARFF (automatically parsed by fetch_openml)
    """
    # Use sklearn's built-in caching mechanism
    data_home = str(cache_dir) if cache_dir else None
    
    # Fetch dataset by ID
    titanic = fetch_openml(
        data_id=40945,           # Titanic dataset ID on OpenML 
        as_frame=as_frame,       # Return as pandas DataFrame
        cache=True,              # Cache locally to avoid re-downloading
        parser="auto",           # Auto-select best parser for ARFF
        data_home=data_home
    )
    
    # The fetch_openml returns a Bunch object; .frame contains the DataFrame
    df = titanic.frame.copy()
    
    # Ensure target column is named consistently
    if 'survived' not in df.columns and titanic.target_names:
        # Rename target if needed
        target_col = titanic.target_names[0] if isinstance(titanic.target_names, list) else titanic.target_names
        if target_col and target_col != 'survived':
            df = df.rename(columns={target_col: 'survived'})
    
    return df


def save_raw_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Save the raw dataset to disk for reproducibility.
    
    Args:
        df: The DataFrame to save
        output_path: Path where to save the CSV file
    
    Returns:
        Path: The actual path where the file was saved
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved raw dataset to: {output_path}")
    return output_path