import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config.config import MODELS_PATH
from functions.dataset import TransformSubset
from functions.test import test
from functions.train import train
from logger.LoggerFactory import LoggerFactory
from model import TinyViT

logger = LoggerFactory.get_logger(__name__)


def stl_10():
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
    raw_train_set = datasets.STL10(
        root="./data", split="train", download=True, transform=None
    )

    train_size = int(0.8 * len(raw_train_set))
    val_size = len(raw_train_set) - train_size
    train_subset, val_subset = random_split(raw_train_set, [train_size, val_size])

    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=test_transform)

    test_set = datasets.STL10(
        root="./data", split="test", download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = TinyViT(
        num_classes=10,
        embed_dim=64,
        img_size=96,
        patch_size=4,
        in_chans=3,
        num_heads=4,
        num_layers=6,
        mlp_dim=256,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
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
        "stl10",
    )

    model.load_state_dict(torch.load(MODELS_PATH + "stl10/" + "model.pth"))
    test(model, test_loader, device, "stl10")
