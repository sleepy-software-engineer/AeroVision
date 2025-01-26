import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from config.config import IMAGES_PATH
from functions.plots import plot_confusion_matrix


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    dataset_name: str,
) -> None:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    compute_metrics(all_labels, all_preds, dataset_name)


def compute_metrics(
    all_labels: np.ndarray, all_preds: np.ndarray, dataset_name: str
) -> None:
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "mcc": mcc,
        "precision": precision,
    }

    with open(IMAGES_PATH + "test/" + dataset_name + "/metrics.json", "w") as f:
        json.dump(metrics, f)

    plot_confusion_matrix(conf_matrix, dataset_name)
