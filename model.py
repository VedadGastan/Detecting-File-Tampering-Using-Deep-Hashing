import torch.nn as nn
from torchvision import models

class TripletHashNet(nn.Module):
    def __init__(self, hash_bits):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature = nn.Sequential(*list(base.children())[:-1])

        for p in list(self.feature.parameters())[:-20]:
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(base.fc.in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, hash_bits),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.feature(x).flatten(1)
        return self.head(x)
