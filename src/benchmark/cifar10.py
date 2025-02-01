import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from functions.dataset import TransformSubset
from functions.test import test
from functions.train import train
from logger.LoggerFactory import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def cifar_10(model: nn.Module, model_name: str):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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

    raw_train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None
    )

    train_size = int(0.8 * len(raw_train_set))
    val_size = len(raw_train_set) - train_size
    train_subset, val_subset = random_split(raw_train_set, [train_size, val_size])

    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=test_transform)

    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        100,
        25,
        len(train_subset),
        len(val_subset),
        "cifar10",
        model_name,
    )

    test(model, test_loader, device, "cifar10", model_name)
