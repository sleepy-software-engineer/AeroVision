import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config.config import (
    BATCH_SIZE,
    CIFAR100,
    EPOCHS,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    PATIENCE,
    WEIGHT_DECAY,
)
from functions.test import test
from functions.train import train
from logger.LoggerFactory import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def cifar_100(model: nn.Module, model_name: str):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )

    test_set = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = train(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        EPOCHS,
        PATIENCE,
        CIFAR100,
        model_name,
    )

    test(model, test_loader, device, CIFAR100, model_name)
