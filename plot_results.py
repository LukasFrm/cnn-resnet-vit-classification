
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Baseline CNN values from the recorded training output.
baseline = pd.DataFrame({
    "epoch": list(range(1, 16)),
    "train_loss": [
        4.7686, 4.6856, 4.6318, 4.5965, 4.5333,
        4.4716, 4.3921, 4.3286, 4.2552, 4.1903,
        4.1163, 4.0425, 3.9860, 3.9175, 3.8635,
    ],
    "train_accuracy": [
        0.0095, 0.0145, 0.0221, 0.0239, 0.0297,
        0.0387, 0.0454, 0.0518, 0.0632, 0.0676,
        0.0817, 0.0932, 0.0958, 0.1099, 0.1228,
    ],
    "val_loss": [
        4.6983, 4.6424, 4.5991, 4.5468, 4.4597,
        4.4383, 4.3255, 4.2474, 4.2062, 4.1322,
        4.1036, 4.0201, 3.9917, 3.8832, 3.8606,
    ],
    "val_accuracy": [
        0.0150, 0.0200, 0.0238, 0.0262, 0.0362,
        0.0421, 0.0537, 0.0700, 0.0654, 0.0779,
        0.0762, 0.0896, 0.0942, 0.1138, 0.1208,
    ],
})

# Load the actual ResNet-50 training history.
resnet = pd.read_csv(
    RESULTS_DIR / "resnet50_history.csv"
)

baseline.to_csv(
    RESULTS_DIR / "baseline_history.csv",
    index=False,
)


def plot_metric(metric, ylabel, filename, percentage=False):
    plt.figure(figsize=(8, 5))

    for name, data in [
        ("Baseline CNN", baseline),
        ("ResNet-50", resnet),
    ]:
        values = data[metric]

        if percentage:
            values = values * 100

        plt.plot(
            data["epoch"],
            values,
            marker="o",
            markersize=4,
            linewidth=2,
            label=name,
        )

    plt.title(ylabel + " by Training Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


plot_metric(
    "train_accuracy",
    "Training Accuracy (%)",
    "training_accuracy.png",
    percentage=True,
)

plot_metric(
    "val_accuracy",
    "Validation Accuracy (%)",
    "validation_accuracy.png",
    percentage=True,
)

plot_metric(
    "train_loss",
    "Training Cross-Entropy Loss",
    "training_loss.png",
)

plot_metric(
    "val_loss",
    "Validation Cross-Entropy Loss",
    "validation_loss.png",
)


# Final validation comparison.
comparison = pd.DataFrame({
    "Model": ["Baseline CNN", "ResNet-50"],
    "Initialization": ["Random", "ImageNet pretrained"],
    "Epochs": [15, 10],
    "Best checkpoint epoch": [15, 8],
    "Top-1 accuracy": [0.1208, 0.8296],
    "Top-5 accuracy": [0.3438, 0.9812],
    "Macro-F1": [0.1041, 0.8273],
})

comparison.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False,
)

print("Generated:")
for path in sorted(FIGURES_DIR.glob("*.png")):
    print(" ", path)

print(" ", RESULTS_DIR / "baseline_history.csv")
print(" ", RESULTS_DIR / "model_comparison.csv")
