import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10, pool_output_size=(8, 8)):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_output_size)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * pool_output_size[0] * pool_output_size[1], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
