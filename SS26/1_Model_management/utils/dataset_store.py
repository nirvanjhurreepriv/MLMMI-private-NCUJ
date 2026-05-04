# utils/dataset_store.py - Dataset versioning utilities
"""
Module for creating, storing, and tracking versioned datasets.
Each version includes preprocessing metadata for reproducibility.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


class DatasetStore:
    """
    Lightweight dataset versioning system.
    
    Tracks:
    - Dataset version ID
    - Source reference
    - Preprocessing steps applied
    - Split information
    - Feature schema
    """
    
    def __init__(self, base_data_dir: Path):
        """
        Initialise the dataset store.
        
        Args:
            base_data_dir: Base directory for all dataset versions (e.g., 'data/')
        """
        self.base_dir = Path(base_data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_version_hash(self, df: pd.DataFrame, steps: List[str]) -> str:
        """
        Compute a deterministic hash for dataset versioning.
        
        Args:
            df: The processed DataFrame
            steps: List of preprocessing step descriptions
        
        Returns:
            str: Short hash string for version identification
        """
        # Hash based on data shape, column names, and preprocessing steps
        content = f"{df.shape}{sorted(df.columns.tolist())}{steps}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def create_version(
        self,
        df: pd.DataFrame,
        version_id: str,
        preprocessing_steps: List[str],
        split_info: Dict[str, Any] = None,
        source_version: str = None,
        save_splits: bool = True
    ) -> Dict:
        """
        Create and save a new dataset version with full metadata.
        
        Args:
            df: The processed DataFrame (full dataset or train portion)
            version_id: Unique identifier for this version (e.g., 'ds_v1')
            preprocessing_steps: List of human-readable preprocessing descriptions
            split_info: Dict with train/test split details if applicable
            source_version: ID of source dataset version (for lineage)
            save_splits: If True, also save train/test splits as separate files
        
        Returns:
            Dict: Metadata dictionary for this dataset version
        """
        version_dir = self.base_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the main dataset
        dataset_path = version_dir / "dataset.csv"
        df.to_csv(dataset_path, index=False)
        
        # Compute hash for integrity tracking
        data_hash = self._compute_version_hash(df, preprocessing_steps)
        
        # Build metadata
        metadata = {
            "dataset_version_id": version_id,
            "source_dataset": source_version or "titanic_openml_40945",
            "preprocessing_steps": preprocessing_steps,
            "creation_time": datetime.now().isoformat(),
            "data_hash": data_hash,
            "schema": {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "shape": df.shape,
                "missing_counts": df.isnull().sum().to_dict()
            }
        }
        
        # Add split info
        if split_info:
            metadata["split_info"] = split_info
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Optionally save train/test splits
        if save_splits and "train_indices" in metadata.get("split_info", {}):
            # This would be extended to actually split and save
            pass
        
        print(f"Created dataset version '{version_id}' at {version_dir}")
        return metadata
    
    def load_version(self, version_id: str) -> tuple[pd.DataFrame, Dict]:
        """
        Load a dataset version and its metadata.
        
        Args:
            version_id: The version ID to load (here e.g., 'ds_v1')
        
        Returns:
            tuple: (DataFrame, metadata_dict)
        """
        version_dir = self.base_dir / version_id
        dataset_path = version_dir / "dataset.csv"
        metadata_path = version_dir / "metadata.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset version '{version_id}' not found at {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        return df, metadata
    
    def list_versions(self) -> List[Dict]:
        """List all available dataset versions with their metadata."""
        versions = []
        for version_dir in self.base_dir.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith("ds_v"):
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        versions.append(json.load(f))
        return versions