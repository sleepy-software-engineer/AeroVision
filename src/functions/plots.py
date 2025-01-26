from typing import List

import matplotlib.pyplot as plt
import numpy as np

from config.config import IMAGES_PATH


def plot_losses(train_losses: List[float], val_losses: List[float]) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(IMAGES_PATH + "loss_plot.png")


def plot_accuracies(train_accs: List[float], val_accs: List[float]) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(IMAGES_PATH + "acc_plot.png")


def plot_confusion_matrix(conf_matrix: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.grid(False)
    plt.savefig("confusion_matrix.png")
    plt.show()
