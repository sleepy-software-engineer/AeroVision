from typing import List

import matplotlib.pyplot as plt
import numpy as np

from config.config import OUTPUT_PATH


def plot_losses(
    train_losses: List[float],
    dataset_name: str,
    model_name: str,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(OUTPUT_PATH + "train/" + dataset_name + "/" + model_name + "/loss.png")


def plot_confusion_matrix(
    conf_matrix: np.ndarray, dataset_name: str, model_name: str
) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.grid(False)
    plt.savefig(
        OUTPUT_PATH
        + "test/"
        + dataset_name
        + "/"
        + model_name
        + "/confusion_matrix.png"
    )
