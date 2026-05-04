# scripts/select_model.py - Budget constrained model selection
"""
Model selection under resource constraints.

Implements selection logic to pick the best model satisfying:
- Training time budget
- Inference time budget  
- Memory constraints (future extension)
- Custom scoring functions

"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional
import pandas as pd


def select_best_under_constraint(
    registry_dir: Path,
    constraint_fn: Callable[[Dict], bool],
    score_fn: Callable[[Dict], float],
    constraint_description: str,
    return_all_candidates: bool = False
) -> Dict:
    """
    Select the best model satisfying a budget constraint.
    
    Args:
        registry_dir: Path to registry directory with model_*.json files
        constraint_fn: Function that returns True if model meets constraint
        score_fn: Function to score models (higher = better)
        constraint_description: Human-readable description for logging
        return_all_candidates: If True, return all candidates + best; else just best
    
    Returns:
        Dict: Best model entry (or dict with best + candidates if return_all_candidates)
    """
    candidates = []
    
    # Load all registered models
    for reg_file in registry_dir.glob("model_*.json"):
        with open(reg_file) as f:
            model_entry = json.load(f)
        
        # Check if model satisfies constraint
        if constraint_fn(model_entry):
            score = score_fn(model_entry)
            candidates.append((model_entry, score))
    
    if not candidates:
        raise ValueError(
            f"No models satisfy the constraint: {constraint_description}\n"
            f"Available models: {[f.name for f in registry_dir.glob('model_*.json')]}"
        )
    
    # Select best by score
    best_entry, best_score = max(candidates, key=lambda x: x[1])
    
    # Log selection
    print(f"   Constraint: {constraint_description}")
    print(f"   Candidates: {len(candidates)} models satisfy constraint")
    print(f"   Selected: {best_entry['model_id']}")
    print(f"   Score: {best_score:.4f}")
    print(f"   Accuracy: {best_entry['metrics'].get('accuracy', 'N/A'):.4f}")
    print(f"   Training time: {best_entry['training_time_sec']:.2f}s")
    print(f"   Inference time: {best_entry['inference_time_sec']:.4f}s/sample")
    if best_entry.get('lineage', {}).get('description'):
        print(f"   Lineage: {best_entry['lineage']['description']}")
    
    if return_all_candidates:
        return {
            "best": best_entry,
            "best_score": best_score,
            "candidates": sorted(candidates, key=lambda x: -x[1]),
            "constraint_description": constraint_description
        }
    
    return best_entry


# Pre-defined constraint functions for common use cases

def training_time_constraint(max_seconds: float) -> Callable[[Dict], bool]:
    """Create a constraint: training_time_sec < max_seconds"""
    return lambda m: m.get("training_time_sec", float("inf")) < max_seconds

def inference_time_constraint(max_seconds: float) -> Callable[[Dict], bool]:
    """Create a constraint: inference_time_sec < max_seconds per sample"""
    return lambda m: m.get("inference_time_sec", float("inf")) < max_seconds

def accuracy_threshold_constraint(min_accuracy: float) -> Callable[[Dict], bool]:
    """Create a constraint: accuracy >= min_accuracy"""
    return lambda m: m.get("metrics", {}).get("accuracy", 0) >= min_accuracy

def dataset_version_constraint(version_id: str) -> Callable[[Dict], bool]:
    """Create a constraint: must use specific dataset version"""
    return lambda m: m.get("dataset_version") == version_id


# Pre-defined scoring functions

def accuracy_score_fn(model_entry: Dict) -> float:
    """Score by accuracy (higher = better)"""
    return model_entry.get("metrics", {}).get("accuracy", 0)

def f1_score_fn(model_entry: Dict) -> float:
    """Score by F1 if available, else accuracy"""
    metrics = model_entry.get("metrics", {})
    return metrics.get("f1_score", metrics.get("accuracy", 0))

def balanced_score_fn(model_entry: Dict, time_weight: float = 0.1) -> float:
    """
    Score balancing accuracy and speed: accuracy: time_weight * training_time
    Useful when both performance and efficiency matter.
    """
    acc = model_entry.get("metrics", {}).get("accuracy", 0)
    train_time = model_entry.get("training_time_sec", 0)
    return acc - time_weight * train_time


if __name__ == "__main__":
    # Example usage for testing
    registry_dir = Path("registry")
    
    if registry_dir.exists():
        print("Testing model selection...")
        
        # Example: Best accuracy with training time < 2 seconds
        best = select_best_under_constraint(
            registry_dir=registry_dir,
            constraint_fn=training_time_constraint(2.0),
            score_fn=accuracy_score_fn,
            constraint_description="maximize accuracy with training_time < 2.0 seconds"
        )
    else:
        print("Registry directory not found. Run model training first.")