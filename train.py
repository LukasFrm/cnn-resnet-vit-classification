
import time
import torch
from torch import nn
from torch.optim import Adam
from pathlib import Path

from data.dataset import create_dataloaders
from models.baseline_cnn import BaselineCNN


EPOCHS = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 32

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(42)

train_loader, val_loader, _ = create_dataloaders(
    batch_size=BATCH_SIZE
)

model = BaselineCNN(num_classes=120).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

Path("checkpoints").mkdir(exist_ok=True)

best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    start = time.perf_counter()


    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)


    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total

    elapsed = time.perf_counter() - start

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss / train_total:.4f} | "
        f"Train acc: {train_accuracy:.4f} | "
        f"Val loss: {val_loss / val_total:.4f} | "
        f"Val acc: {val_accuracy:.4f} | "
        f"Time: {elapsed:.1f}s",
        flush=True,
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
            },
            "checkpoints/baseline_best.pt",
        )

print(f"Best validation accuracy: {best_val_accuracy:.4f}")
