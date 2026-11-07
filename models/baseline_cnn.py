import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=120):
        super().__init__()

        self.features = nn.Sequential(
            # 3 × 224 × 224
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 32 × 112 × 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 64 × 56 × 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 128 × 28 × 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 256 × 14 × 14
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)

        # 256 × 1 × 1 -> 256
        x = torch.flatten(x, 1)

        # 256 -> 120 class scores
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    model = BaselineCNN()

    fake_batch = torch.randn(32, 3, 224, 224)

    output = model(fake_batch)

    print("Input shape:", fake_batch.shape)
    print("Output shape:", output.shape)