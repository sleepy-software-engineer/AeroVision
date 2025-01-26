import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from functions.train import train
from logger.LoggerFactory import LoggerFactory
from model import TinyViT

logger = LoggerFactory.get_logger(__name__)


def cifar_10():
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

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = TinyViT(
        num_classes=10,
        embed_dim=64,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_heads=4,
        num_layers=6,
        mlp_dim=256,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        num_epochs=200,
        patience=25,
        train_set_size=len(train_set),
        val_set_size=len(test_set),
    )


def cifar_100():
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

    train_set = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = TinyViT(
        num_classes=100,
        embed_dim=64,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_heads=4,
        num_layers=6,
        mlp_dim=256,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        num_epochs=200,
        patience=25,
        train_set_size=len(train_set),
        val_set_size=len(test_set),
    )


if __name__ == "__main__":
    cifar_100()
