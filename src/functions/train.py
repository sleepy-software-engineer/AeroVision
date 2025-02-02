from copy import deepcopy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from functions.plots import plot_losses
from logger.LoggerFactory import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 0,
    patience: int = 0,
    dataset_name: str = None,
    model_name: str = None,
) -> nn.Module:
    best_train_loss = float("inf")
    best_epoch = 0
    train_losses = []
    best_model_wts = deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)

        avg_train_loss = epoch_train_loss / len(train_loader)

        train_losses.append(avg_train_loss)

        logger.debug("--------------------------")
        logger.debug(f"Epoch {epoch + 1}/{num_epochs}")
        logger.debug(f"Train Loss: {avg_train_loss:.4f}")
        logger.debug("--------------------------")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch
            best_model_wts = deepcopy(model.state_dict())
            logger.info(
                f"New best model saved at epoch {epoch + 1} with train loss {avg_train_loss:.4f}"
            )

        if epoch - best_epoch >= patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)"
            )
            break

    plot_losses(train_losses, dataset_name, model_name)

    model.load_state_dict(best_model_wts)
    return model
