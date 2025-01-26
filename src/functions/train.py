import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config.config import MODELS_PATH
from functions.plots import plot_accuracies, plot_losses
from logger.LoggerFactory import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    patience: int = 30,
    train_set_size: int = None,
    val_set_size: int = None,
) -> None:
    best_val_acc = 0.0
    best_epoch = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_set_size
        val_loss /= len(val_loader)
        val_acc = val_correct / val_set_size

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.debug("--------------------------")
        logger.debug(f"Epoch {epoch + 1}/{num_epochs}")
        logger.debug(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%"
        )
        logger.debug(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        logger.debug("--------------------------")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODELS_PATH + "best_model.pth")
            logger.info(
                f"New best model saved at epoch {epoch + 1} with val acc {val_acc * 100:.2f}%"
            )

        if epoch - best_epoch >= patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)"
            )
            break

    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accs, val_accs)
