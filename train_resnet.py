
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from data.dataset import create_dataloaders
from models.resnet50 import create_resnet50


EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

CHECKPOINT_PATH = Path("checkpoints/resnet50_best.pt")
HISTORY_PATH = Path("results/resnet50_history.csv")


def main():
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device, flush=True)

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = create_resnet50(num_classes=120).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    HISTORY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    best_val_accuracy = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()

        # Training
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS} - Training",
            unit="batch",
            leave=True,
        )

        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            train_loss_sum += loss.item() * batch_size
            train_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            train_total += batch_size

            train_progress.set_postfix(
                loss=f"{train_loss_sum / train_total:.4f}",
                acc=f"{train_correct / train_total:.2%}",
            )

        # Validation
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{EPOCHS} - Validation",
            unit="batch",
            leave=False,
        )

        with torch.inference_mode():
            for images, labels in val_progress:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)

                val_loss_sum += loss.item() * batch_size
                val_correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()
                val_total += batch_size

                val_progress.set_postfix(
                    loss=f"{val_loss_sum / val_total:.4f}",
                    acc=f"{val_correct / val_total:.2%}",
                )

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "elapsed_seconds": elapsed,
        }

        history.append(row)

        with HISTORY_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=row.keys()
            )
            writer.writeheader()
            writer.writerows(history)

        tqdm.write(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "model": "ResNet50",
                    "num_classes": 120,
                },
                CHECKPOINT_PATH,
            )

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
