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


def stl_10():
    # Define transforms for STL-10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(96, padding=12),  # STL-10 images are 96x96
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
            ),  # STL-10 mean and std
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
            ),  # STL-10 mean and std
        ]
    )

    # Load STL-10 dataset
    train_set = datasets.STL10(
        root="./data", split="train", download=True, transform=train_transform
    )
    test_set = datasets.STL10(
        root="./data", split="test", download=True, transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Define the TinyViT model for STL-10
    model = TinyViT(
        num_classes=10,  # STL-10 has 10 classes
        embed_dim=64,
        img_size=96,  # STL-10 images are 96x96
        patch_size=8,  # Adjust patch size for 96x96 images
        in_chans=3,
        num_heads=4,
        num_layers=6,
        mlp_dim=256,
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
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
    stl_10()
