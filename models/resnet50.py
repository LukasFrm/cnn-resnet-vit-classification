
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(num_classes=120):
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    # Replace ImageNet's 1000-class output layer.
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model


if __name__ == "__main__":
    import torch

    model = create_resnet50()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print("Output shape:", output.shape)
