"""
Data preparation script for CIFAR-10 transfer learning experiments.

Downloads CIFAR-10 via torchvision and materializes a reproducible subset
using fixed random seed. Persists indices for cross-run reproducibility.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

# Config
CIFAR10_ROOT = Path("data/raw")
SUBSET_PATH = Path("data/subset.json")
RANDOM_SEED = 42

# Subset sizes
TRAIN_SUBSET_SIZE = 10000
TEST_SUBSET_SIZE = 2000


def download_cifar10(root: Path) -> datasets.CIFAR10:
    """Download CIFAR-10 dataset using torchvision."""
    root.mkdir(parents=True, exist_ok=True)
    
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=None
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=None
    )
    return train_dataset, test_dataset


def create_reproducible_subset(
    train_dataset, test_dataset, seed: int = RANDOM_SEED
) -> dict:
    """
    Create reproducible train/test subsets using fixed random seed.
    
    Returns dict with indices for both splits.
    """
    rng = np.random.default_rng(seed)
    
    # Sample training indices
    train_indices = rng.choice(
        len(train_dataset), size=TRAIN_SUBSET_SIZE, replace=False
    ).tolist()
    
    # Sample test indices
    test_indices = rng.choice(
        len(test_dataset), size=TEST_SUBSET_SIZE, replace=False
    ).tolist()
    
    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "seed": seed,
        "train_subset_size": TRAIN_SUBSET_SIZE,
        "test_subset_size": TEST_SUBSET_SIZE,
    }


def save_subset(subset: dict, path: Path):
    """Persist subset indices to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved subset indices to {path}")


def load_subset(path: Path) -> dict:
    """Load previously saved subset indices."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    """Main entry point for data preparation."""
    print(f"Downloading CIFAR-10 to {CIFAR10_ROOT}...")
    train_dataset, test_dataset = download_cifar10(CIFAR10_ROOT)
    
    print(f"Creating reproducible subset (seed={RANDOM_SEED})...")
    subset = create_reproducible_subset(train_dataset, test_dataset)
    
    print(f"Subset stats:")
    print(f"Training samples: {len(subset['train_indices'])}")
    print(f"Test samples: {len(subset['test_indices'])}")
    
    save_subset(subset, SUBSET_PATH)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()