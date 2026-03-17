import copy
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


DATA_ROOT = Path("filtered_simpsons/train")
OUTPUT_DIR = Path("resnet18_run")
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
VAL_SIZE = 0.2
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
STEP_SIZE = 5
GAMMA = 0.5
NUM_WORKERS = 32
SEED = 42


class SimpleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        image = self.transform(image)
        return image, label


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def make_output_dir():
    if OUTPUT_DIR.exists():
        for path in OUTPUT_DIR.iterdir():
            if path.is_file():
                path.unlink()
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_datasets():
    base_dataset = datasets.ImageFolder(DATA_ROOT)
    labels = [label for _, label in base_dataset.samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(base_dataset.samples)),
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_samples = [base_dataset.samples[i] for i in train_idx]
    val_samples = [base_dataset.samples[i] for i in val_idx]

    train_dataset = SimpleDataset(train_samples, train_transform)
    val_dataset = SimpleDataset(val_samples, val_transform)
    return base_dataset.classes, train_samples, val_samples, train_dataset, val_dataset


def build_loaders(train_dataset, val_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return train_loader, val_loader


def build_model(num_classes, device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model


def build_class_weights(train_samples, num_classes, device):
    counts = Counter(label for _, label in train_samples)
    total = sum(counts.values())
    weights = []
    for class_id in range(num_classes):
        count = counts[class_id]
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    metrics = get_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, y_true, y_pred


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.plot(epochs, history["val_f1_micro"], label="val_f1_micro")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics.png", dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    set_seed()
    make_output_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_names, train_samples, val_samples, train_dataset, val_dataset = build_datasets()
    train_loader, val_loader = build_loaders(train_dataset, val_dataset)

    model = build_model(len(class_names), device)
    class_weights = build_class_weights(train_samples, len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_f1 = -1.0
    best_model = None
    best_true = None
    best_pred = None
    history = {"train_loss": [], "val_loss": [], "val_f1_micro": []}

    print(f"train: {len(train_samples)}")
    print(f"val: {len(val_samples)}")
    print(f"classes: {len(class_names)}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1_micro"].append(val_metrics["f1_micro"])

        print(
            f"epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1_micro={val_metrics['f1_micro']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
            f"val_precision_macro={val_metrics['precision_macro']:.4f} | "
            f"val_recall_macro={val_metrics['recall_macro']:.4f}"
        )

        if val_metrics["f1_micro"] > best_f1:
            best_f1 = val_metrics["f1_micro"]
            best_model = copy.deepcopy(model.state_dict())
            best_true = y_true
            best_pred = y_pred

    torch.save(
        {
            "model_state_dict": best_model,
            "class_names": class_names,
            "best_val_f1_micro": best_f1,
        },
        OUTPUT_DIR / "best_model.pth",
    )

    plot_history(history)
    plot_confusion(best_true, best_pred, class_names)
    print("done")


if __name__ == "__main__":
    main()
