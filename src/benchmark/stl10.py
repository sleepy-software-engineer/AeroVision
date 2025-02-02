import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config.config import (
    BATCH_SIZE,
    EPOCHS,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    PATIENCE,
    STL10,
    WEIGHT_DECAY,
)
from functions.test import test
from functions.train import train
from logger.LoggerFactory import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def stl_10(model: nn.Module, model_name: str):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ]
    )

    train_set = datasets.STL10(
        root="./data", split="train", download=True, transform=train_transform
    )

    test_set = datasets.STL10(
        root="./data", split="test", download=True, transform=test_transform
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
        STL10,
        model_name,
    )

    test(model, test_loader, device, STL10, model_name)
