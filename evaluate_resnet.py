
import json
from pathlib import Path

import torch

from data.dataset import create_dataloaders
from models.resnet50 import create_resnet50
from evaluate import evaluate


CHECKPOINT_PATH = Path("checkpoints/resnet50_best.pt")
RESULTS_PATH = Path("results/resnet50_validation.json")


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    _, val_loader, _ = create_dataloaders(batch_size=32)

    model = create_resnet50(num_classes=120).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, val_loader, device)

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.2%}")
    print(f"Top-5 accuracy: {metrics['top5_accuracy']:.2%}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    RESULTS_PATH.write_text(
        json.dumps({
            "model": "ResNet50",
            "split": "validation",
            "checkpoint_epoch": checkpoint["epoch"],
            **metrics,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
