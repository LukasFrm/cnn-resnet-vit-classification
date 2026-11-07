
import torch
from data.dataset import create_dataloaders
from models.baseline_cnn import BaselineCNN

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=32
)

model = BaselineCNN(num_classes=120).to(device)

images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

# Forward pass
outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)

assert outputs.shape == (32, 120)

# Compute classification loss
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(outputs, labels)

print("Loss:", loss.item())

# Backpropagation
model.zero_grad()
loss.backward()

print("Backward pass successful")
