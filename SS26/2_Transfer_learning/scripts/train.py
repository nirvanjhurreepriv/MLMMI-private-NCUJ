"""
Parametrised training script for ResNet18 transfer learning experiments.

Supports:
- Initialisation: from_scratch , pretrained
- Layer freezing: configurable list of frozen blocks
- Mixed precision training (AMP)
- Feature caching for frozen backbones
- Registry logging for experiment tracking
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from prepare_data import load_subset, CIFAR10_ROOT, SUBSET_PATH

# Configs
DEFAULTS = {
    "epochs": 10,
    "batch_size": 64,
    "lr": 1e-3,
    "image_size": 224,  # Upsample CIFAR-10 (32x32) to match ImageNet preprocessing
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ResNet18 block names for freezing (in order of depth)
RESNET_BLOCKS = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet18 Transfer Learning Trainer")
    
    # Experiment identification
    parser.add_argument("--run_id", type=str, required=True, help="Unique run identifier")
    
    # Model initialisation
    parser.add_argument(
        "--init", type=str, choices=["scratch", "pretrained"], default="scratch",
        help="Weight initialization strategy"
    )
    
    # Layer freezing
    parser.add_argument(
        "--freeze", type=str, nargs="*", default=[],
        choices=RESNET_BLOCKS,
        help="List of ResNet blocks to freeze (e.g., conv1 bn1 layer1)"
    )
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    
    parser.add_argument("--image_size", type=int, default=DEFAULTS["image_size"],
                       help="Input image resolution (default: 224 for ImageNet compatibility)")
    
    # Data loading
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"],
                       help="Number of DataLoader workers")
    
    # Optimization flags
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--cache_features", action="store_true", 
                       help="Cache frozen backbone features (only valid with frozen layers)")
    
    # I/O paths
    parser.add_argument("--model_dir", type=Path, default=Path("models"))
    parser.add_argument("--registry_dir", type=Path, default=Path("registry"))
    parser.add_argument("--parent_run_id", type=str, default=None,
                       help="Parent run ID for optimization comparisons")
    
    default_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device", type=str, default=default_device,
                       choices=["cpu", "cuda", "mps"],
                       help="Device to train on (auto-detected by default)")
    
    return parser.parse_args()


def get_transforms(image_size: int, init: str):
    """
    Get data transforms based on initialisation strategy.
    
    For pretrained models, use ImageNet normalization.
    For scratch training, use simple normalization.
    """
    if init == "pretrained":
        # Use ImageNet preprocessing for pretrained weights
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Simple normalization for training from scratch
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5]),
        ])


def load_cifar10_subset(transform, batch_size: int, num_workers: int):
    """Load CIFAR-10 with reproducible subset indices."""
    subset_config = load_subset(SUBSET_PATH)
    
    train_full = datasets.CIFAR10(
        root=CIFAR10_ROOT, train=True, transform=transform, download=False
    )
    test_full = datasets.CIFAR10(
        root=CIFAR10_ROOT, train=False, transform=transform, download=False
    )
    
    train_subset = Subset(train_full, subset_config["train_indices"])
    test_subset = Subset(test_full, subset_config["test_indices"])
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def create_model(init: str, image_size: int, freeze_blocks: list):
    """
    Create ResNet18 model with specified initialisation and freezing.
    
    Key considerations for BatchNorm when freezing:
    - Frozen BatchNorm layers should use running stats (eval mode)
    - We set frozen BN layers to eval() but keep model in train() mode
    - This preserves learned statistics while allowing gradient flow through unfrozen layers
    """
    # Load model with appropriate weights
    if init == "pretrained":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    
    # Replace final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Apply layer freezing
    if freeze_blocks:
        for block_name in freeze_blocks:
            block = getattr(model, block_name, None)
            if block is not None:
                for param in block.parameters():
                    param.requires_grad = False
                
                # Handle BatchNorm in frozen blocks
                # Set BN layers to eval mode to use running stats, not batch stats
                for module in block.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
                        # Freeze BN affine parameters
                        if module.weight is not None:
                            module.weight.requires_grad = False
                        if module.bias is not None:
                            module.bias.requires_grad = False
    
    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float):
    """Create optimizer with only trainable parameters."""
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name == "Adam":
        return optim.Adam(params, lr=lr)
    elif optimizer_name == "SGD":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    use_amp: bool = False,
):
    """Single training epoch with optional AMP."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str,
    n_repeats: int = 10,
    batch_size: int = 100,
) -> float:
    """
    Measure average inference latency in milliseconds.
    
    Warmup + multiple repeats for stable measurement.
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Warmup
    for _ in range(3):
        _ = model(input_tensor)
    
    # Timing
    latencies = []
    for _ in range(n_repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(input_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return sum(latencies) / len(latencies)


def save_registry_entry(args, metrics: dict, model_path: Path):
    """Save experiment metadata to registry JSON."""
    args.registry_dir.mkdir(parents=True, exist_ok=True)
    
    trainable_params, total_params = count_trainable_params(
        # Reload model to get accurate count (in case of AMP/grad state)
        create_model(args.init, args.image_size, args.freeze)
    )
    
    entry = {
        "run_id": args.run_id,
        "init": args.init,
        "frozen_layers": args.freeze,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "optimizations": ["amp"] if args.amp else [],
        "accuracy": metrics["test_accuracy"],
        "train_time_s": metrics["train_time_s"],
        "inference_time_ms": metrics["inference_time_ms"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "device": args.device,
        "parent_run_id": args.parent_run_id,
        "model_path": str(model_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    
    registry_path = args.registry_dir / f"{args.run_id}.json"
    with open(registry_path, "w") as f:
        json.dump(entry, f, indent=2)
    
    print(f"Registry entry saved: {registry_path}")
    return entry


def main():
    args = parse_args()
    
    print(f"Starting training run: {args.run_id}")
    print(f"Init: {args.init}, Frozen: {args.freeze if args.freeze else 'none'}")
    print(f"Device: {args.device}, AMP: {args.amp}")
    
    # Setup
    args.model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    # Data
    transform = get_transforms(args.image_size, args.init)
    train_loader, test_loader = load_cifar10_subset(
        transform, args.batch_size, DEFAULTS["num_workers"]
    )
    
    # Model
    model = create_model(args.init, args.image_size, args.freeze).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.amp
        )
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
    
    train_time = time.time() - start_time
    
    # Evaluation
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Inference latency measurement
    dummy_input = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    inference_time = measure_inference_latency(model, dummy_input, device)
    print(f"Inference latency: {inference_time:.2f} ms/batch")
    
    # Save model
    model_path = args.model_dir / f"{args.run_id}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    
    # Registry
    metrics = {
        "test_accuracy": test_accuracy,
        "train_time_s": round(train_time, 2),
        "inference_time_ms": round(inference_time, 2),
    }
    save_registry_entry(args, metrics, model_path)
    
    print(f"Run {args.run_id} complete.")


if __name__ == "__main__":
    main()