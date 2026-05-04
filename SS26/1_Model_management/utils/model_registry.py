# utils/model_registry.py - Model registration and metadata tracking
"""
Lightweight model registry system that tracks:
- Model artifacts (serialised models)
- Meta algorithm, hyperparameters, metrics, timing
- Lineage: derivation relationships between models
- Dataset version used for training

(Inspired by ModelDB paper principles on centralised metadata, reproducibility)
"""
import sys
import json
import joblib
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class ModelRegistry:
    """
    Lightweight model registry with comprehensive metadata tracking.
    
    Each registered model includes:
    - Serialised model artifact (.joblib)
    - Metadata JSON with algorithm, hyperparameters, metrics
    - Performance timing: training time, inference time
    - Lineage information for derivation tracking
    - Dataset version provenance
    """
    
    def __init__(self, registry_dir: Path, models_dir: Path):
        """
        Initialise the model registry.
        
        Args:
            registry_dir: Directory for metadata JSON files
            models_dir: Directory for serialized model artifacts
        """
        self.registry_dir = Path(registry_dir)
        self.models_dir = Path(models_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _measure_inference_time(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        n_runs: int = 10, 
        batch_size: int = 100
    ) -> float:
        """
        Measure average inference time per sample.
        
        Args:
            model: Trained scikit-learn model with predict() method
            X_test: Test features DataFrame
            n_runs: Number of measurement runs for averaging
            batch_size: Number of samples per inference batch
        
        Returns:
            float: Average inference time in seconds per sample
        """
        times = []
        X_values = X_test.values if hasattr(X_test, 'values') else X_test
        
        for _ in range(n_runs):
            # Sample a random batch
            n_samples = min(batch_size, len(X_values))
            idx = np.random.choice(len(X_values), n_samples, replace=False)
            batch = X_values[idx]
            
            # Measure prediction time
            start = time.perf_counter()
            _ = model.predict(batch)
            end = time.perf_counter()
            
            # Calculate per-sample time
            times.append((end - start) / n_samples)
        
        return float(np.mean(times))
    
    def register_model(
        self,
        model: Any,
        model_id: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        training_time_sec: float,
        inference_time_sec: float,
        dataset_version: str,
        lineage: Optional[Dict] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Register a trained model with full metadata.
        
        Args:
            model: Trained scikit-learn model object
            model_id: Unique identifier (e.g., 'model_001')
            algorithm: Algorithm name (e.g., 'LogisticRegression')
            hyperparameters: Dict of hyperparameter name -> value
            metrics: Dict of metric name -> value (must include 'accuracy')
            training_time_sec: Total training time in seconds
            inference_time_sec: Average inference time per sample (seconds)
            dataset_version: ID of dataset version used for training
        
        Returns:
            Dict: The complete registry entry that was saved
        """
        # Create model directory and save artifact
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Build comprehensive registry entry
        entry = {
            "model_id": model_id,
            "model_path": str(model_path.relative_to(self.models_dir.parent)),
            "algorithm": algorithm,
            "hyperparameters": hyperparameters,
            "metrics": metrics,  # Must include 'accuracy' per requirements
            "training_time_sec": training_time_sec,
            "inference_time_sec": inference_time_sec,
            "dataset_version": dataset_version,
            "creation_time": datetime.now().isoformat(),
            "lineage": lineage or {},
            "notes": notes,
            # Additional metadata for reproducibility
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "sklearn_version": joblib.__version__  # Approximate
        }
        
        # Save registry entry as JSON
        registry_path = self.registry_dir / f"{model_id}.json"
        with open(registry_path, "w") as f:
            json.dump(entry, f, indent=2)
        
        print(f"Registered model '{model_id}' (accuracy: {metrics.get('accuracy', 'N/A'):.4f})")
        return entry
    
    def list_models(self, filter_by: Dict = None) -> List[Dict]:
        """
        List all registered models, optionally filtered.
        
        Args:
            filter_by: Optional dict of field -> value to filter results
        
        Returns:
            List[Dict]: List of model registry entries
        """
        models = []
        for reg_file in self.registry_dir.glob("model_*.json"):
            with open(reg_file) as f:
                entry = json.load(f)
            
            # Apply filters if specified
            if filter_by:
                if all(entry.get(k) == v for k, v in filter_by.items()):
                    models.append(entry)
            else:
                models.append(entry)
        
        return sorted(models, key=lambda x: x.get("creation_time", ""))
    
    def get_model(self, model_id: str) -> tuple[Any, Dict]:
        """
        Load a registered model and its metadata.
        
        Args:
            model_id: The model ID to load
        
        Returns:
            tuple: (loaded_model_object, metadata_dict)
        """
        reg_path = self.registry_dir / f"{model_id}.json"
        if not reg_path.exists():
            raise FileNotFoundError(f"Model '{model_id}' not found in registry")
        
        with open(reg_path) as f:
            metadata = json.load(f)
        
        model_path = self.models_dir.parent / metadata["model_path"]
        model = joblib.load(model_path)
        
        return model, metadata
    
    def get_best_model(self, metric: str = "accuracy", dataset_version: str = None) -> tuple[Any, Dict]:
        """
        Get the best performing model by a given metric.
        
        Args:
            metric: Metric name to optimise (default: 'accuracy')
            dataset_version: Optional filter by dataset version
        
        Returns:
            tuple: (best_model_object, best_metadata_dict)
        """
        models = self.list_models()
        
        # Filter by dataset version if specified
        if dataset_version:
            models = [m for m in models if m.get("dataset_version") == dataset_version]
        
        if not models:
            raise ValueError("No models found in registry")
        
        # Find model with best metric value
        best = max(models, key=lambda m: m.get("metrics", {}).get(metric, -float("inf")))
        return self.get_model(best["model_id"])