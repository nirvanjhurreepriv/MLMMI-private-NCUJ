#!/usr/bin/env python3
"""
Aggregate and compare results from registry JSON files.

Produces comparison table with accuracy, timing, and parameter counts
across all experimental runs.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate transfer learning experiments")
    parser.add_argument(
        "--registry_dir", type=Path, default=Path("registry"),
        help="Directory containing registry JSON files"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_comparison.csv"),
        help="Output CSV file path"
    )
    return parser.parse_args()


def load_registry_entries(registry_dir: Path) -> list[dict]:
    """Load all JSON entries from registry directory."""
    entries = []
    for json_file in registry_dir.glob("*.json"):
        with open(json_file, "r") as f:
            entries.append(json.load(f))
    return entries


def format_freeze_config(frozen_layers: list) -> str:
    """Format frozen layers list for display."""
    if not frozen_layers:
        return "none"
    return "+".join(frozen_layers)


def create_comparison_table(entries: list[dict]) -> pd.DataFrame:
    """Create formatted comparison DataFrame."""
    rows = []
    for entry in entries:
        row = {
            "run_id": entry["run_id"],
            "init": entry["init"],
            "frozen_blocks": format_freeze_config(entry.get("frozen_layers", [])),
            "test_accuracy": f"{entry['accuracy']:.3f}",
            "train_time_s": f"{entry['train_time_s']:.1f}",
            "inference_time_ms": f"{entry['inference_time_ms']:.2f}",
            "trainable_params": f"{entry['trainable_params']:,}",
            "total_params": f"{entry['total_params']:,}",
            "optimizations": ", ".join(entry.get("optimizations", [])) or "none",
            "parent_run": entry.get("parent_run_id", "-"),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add temporary helper columns for reliable sorting
    df["_init_order"] = df["init"].map({"scratch": 0, "pretrained": 1})
    df["_freeze_depth"] = df["frozen_blocks"].apply(
        lambda x: len(x.split("+")) if x != "none" else 0
    )
    
    # Sort logically: scratch first, then pretrained; by freeze depth
    df = df.sort_values(["_init_order", "_freeze_depth"], ascending=[True, True])
    
    # Clean up helper columns before returning
    return df.drop(columns=["_init_order", "_freeze_depth"])


def print_summary(df: pd.DataFrame):
    """Print formatted summary table."""
    print("\n" + "="*100)
    print("TRANSFER LEARNING EXPERIMENT COMPARISON")
    print("="*100)
    
    # Display key columns
    display_cols = [
        "run_id", "init", "frozen_blocks", "test_accuracy", 
        "train_time_s", "inference_time_ms", "trainable_params"
    ]
    
    print(df[display_cols].to_string(index=False))
    
    print("\n" + "-"*100)
    print("KEY OBSERVATIONS:")
    print("-"*100)
    
    # Scratch vs. pretrained comparison
    scratch = df[df["init"] == "scratch"].iloc[0] if len(df[df["init"]=="scratch"]) > 0 else None
    pretrained_full = df[(df["init"]=="pretrained") & (df["frozen_blocks"]=="none")].iloc[0] \
        if len(df[(df["init"]=="pretrained") & (df["frozen_blocks"]=="none")]) > 0 else None
    
    if scratch is not None and pretrained_full is not None:
        acc_gain = float(pretrained_full["test_accuracy"]) - float(scratch["test_accuracy"])
        print(f"Pretrained vs Scratch: +{acc_gain*100:.2f}pp accuracy")
        print(f"Training time ratio: {float(scratch['train_time_s'])/float(pretrained_full['train_time_s']):.2f}x")
    
    # Freezing impact
    frozen_runs = df[(df["init"]=="pretrained") & (df["frozen_blocks"]!="none")]
    if len(frozen_runs) > 0:
        best_frozen = frozen_runs.loc[frozen_runs["test_accuracy"].astype(float).idxmax()]
        print(f"Best frozen config: {best_frozen['run_id']} ({best_frozen['frozen_blocks']})")
        print(f"Accuracy: {best_frozen['test_accuracy']}, Trainable params: {best_frozen['trainable_params']}")
    
    print("="*100 + "\n")


def main():
    args = parse_args()
    
    print(f"Loading registry entries from {args.registry_dir}...")
    entries = load_registry_entries(args.registry_dir)
    
    if not entries:
        print("No registry entries found.")
        return
    
    print(f"Loaded {len(entries)} experiment entries")
    
    # Create comparison table
    df = create_comparison_table(entries)
    
    # Save (to CSV)
    df.to_csv(args.output, index=False)
    print(f"Comparison table saved: {args.output}")
    
    print_summary(df)
    
    # Generate visualisation hints
    print("Visualization suggestions:")
    print("Accuracy vs. Trainable Parameters (scatter)")
    print("Training Time vs. Frozen Blocks (bar chart)")
    print("Accuracy/Time efficiency ratio by configuration")


if __name__ == "__main__":
    main()