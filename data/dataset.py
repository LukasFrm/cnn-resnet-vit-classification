from pathlib import Path
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import Dataset, Subset, DataLoader


DATASET_DIR = Path(__file__).parent / "stanford-dogs"
IMAGES_DIR = DATASET_DIR / "Images"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

DATASET_DIR = Path(__file__).parent / "stanford-dogs"


class StanfordDogsDataset(Dataset):
    def __init__(self, mat_file, transform=None):
            data = loadmat(mat_file)

            self.file_list = [
                  item[0][0]
                  for item in data["file_list"]
            ]   

            # Stanford labels: 1..120
            # PyTorch CrossEntropyLoss expects: 0..119

            self.labels = [
                  int(label[0]) - 1
                  for label in data["labels"]
            ]

            self.transform = transform

    def __len__(self):
          return len(self.file_list)

    def __getitem__(self, index):
          relative_path = self.file_list[index]
          label = self.labels[index]

          image_path = IMAGES_DIR / relative_path

          image = Image.open(image_path).convert("RGB")

          if self.transform is not None:
            image = self.transform(image)

          return image, label

if __name__ == "__main__":
    # Same 12,000 official training images, but different transforms.
    full_train_dataset = StanfordDogsDataset(
        DATASET_DIR / "train_list.mat",
        transform=train_transform,
    )

    full_val_dataset = StanfordDogsDataset(
        DATASET_DIR / "train_list.mat",
        transform=eval_transform,
    )

    test_dataset = StanfordDogsDataset(
        DATASET_DIR / "test_list.mat",
        transform=eval_transform,
    )

    indices = list(range(len(full_train_dataset)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=full_train_dataset.labels,
    )

    train_dataset = Subset(
        full_train_dataset,
        train_indices,
    )

    val_dataset = Subset(
        full_val_dataset,
        val_indices,
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    image, label = train_dataset[0]

    print("Train image shape:", image.shape)
    print("Train label:", label)


BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

images, labels = next(iter(train_loader))

print("Batch images:", images.shape)
print("Batch labels:", labels.shape)
print("First 10 labels:", labels[:10])

