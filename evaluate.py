
import json
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from data.dataset import create_dataloaders
from models.baseline_cnn import BaselineCNN


CHECKPOINT_PATH = Path("checkpoints/baseline_best.pt")
RESULTS_PATH = Path("results/baseline_validation.json")
NUM_CLASSES = 120


def evaluate(model, loader, device):
    model.eval()

    total = 0
    top1_correct = 0
    top5_correct = 0

    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            # Highest-scoring class for each image.
            predictions = logits.argmax(dim=1)

            top1_correct += (
                predictions == labels
            ).sum().item()

            # Five highest-scoring classes for each image.
            top5_indices = logits.topk(
                k=5, dim=1
            ).indices

            top5_correct += (
                top5_indices == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_labels.extend(
                labels.cpu().tolist()
            )

    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    return {
        "samples": total,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_f1": float(macro_f1),
    }


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    _, val_loader, _ = create_dataloaders(
        batch_size=32
    )

    model = BaselineCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    metrics = evaluate(
        model,
        val_loader,
        device,
    )

    print("\n=== Baseline CNN Validation ===")
    print(f"Samples: {metrics['samples']}")
    print(
        f"Top-1 accuracy: "
        f"{metrics['top1_accuracy']:.2%}"
    )
    print(
        f"Top-5 accuracy: "
        f"{metrics['top5_accuracy']:.2%}"
    )
    print(
        f"Macro-F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model": "BaselineCNN",
        "dataset": "Stanford Dogs",
        "split": "validation",
        "checkpoint_epoch": checkpoint["epoch"],
        **metrics,
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
