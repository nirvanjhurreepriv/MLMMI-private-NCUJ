import os
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

def main():
    print("=" * 60)
    print("Step 0: Train and Register 4 Model Heads")
    print("=" * 60)
    
    # Setup directories
    models_dir = Path("models")
    registry_dir = Path("registry")
    data_dir = Path("data")
    for d in [models_dir, registry_dir, data_dir]:
        d.mkdir(exist_ok=True)

    # Load dataset
    print("\n Loading 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = dataset.data
    labels = np.array(dataset.target)
    print(f"   Total samples: {len(texts)}")
    print(f"   Number of classes: {len(np.unique(labels))}")

    # Fixed train/test split for reproducibility
    print("\n Creating reproducible train/test split...")
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(texts))
    train_idx = indices[:int(0.8 * len(texts))]
    test_idx = indices[int(0.8 * len(texts)):]

    # Save split indices
    with open(data_dir / "subset.json", "w") as f:
        json.dump({
            "train_indices": train_idx.tolist(), 
            "test_indices": test_idx.tolist()
        }, f, indent=2)
    print(f"   Train: {len(train_idx)} samples")
    print(f"   Test: {len(test_idx)} samples")
    print(f"   Saved to data/subset.json")

    # Load encoder & compute embeddings
    print("\n Loading sentence transformer (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    print(" Computing train embeddings...")
    train_texts = [texts[i] for i in train_idx]
    train_embeddings = encoder.encode(
        train_texts, 
        batch_size=64, 
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(" Computing test embeddings...")
    test_texts = [texts[i] for i in test_idx]
    test_embeddings = encoder.encode(
        test_texts, 
        batch_size=64, 
        show_progress_bar=True,
        convert_to_numpy=True
    )

    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # Define and train heads
    print("\n Training classifier heads on pre-computed embeddings...")
    heads = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=42),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), 
            max_iter=200, 
            random_state=42, 
            early_stopping=True,
            validation_fraction=0.1
        ),
    }

    results = []
    for model_id, clf in heads.items():
        print(f"\n  Training {model_id}...")
        clf.fit(train_embeddings, train_labels)

        # Evaluate on held-out test set
        preds = clf.predict(test_embeddings)
        acc = accuracy_score(test_labels, preds)
        results.append((model_id, acc))

        # Save model
        model_path = models_dir / f"{model_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        # Save registry entry
        entry = {
            "model_id": model_id,
            "algorithm": model_id,
            "accuracy": round(acc, 4),
            "model_path": str(model_path),
            "embedding_dim": 384,
            "num_classes": 20
        }
        with open(registry_dir / f"{model_id}.json", "w") as f:
            json.dump(entry, f, indent=2)

        print(f"     Test accuracy: {acc:.4f}")
        print(f"     Saved model to {model_path}")
        print(f"     Saved registry to registry/{model_id}.json")

    # Print summary
    print("\n" + "=" * 60)
    print("Step 0 completed. Summary:")
    print("=" * 60)
    for model_id, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  {model_id:8s}: {acc:.4f}")
    print("\n All models trained and registered.")

if __name__ == "__main__":
    main()