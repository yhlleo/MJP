import torch
import torch.nn as nn
import torchvision
from torchvision import models

from munch import Munch

class ResNet50(nn.Module):
    def __init__(
        self, 
        num_classes
    ):
        super().__init__()
        model = models.resnet50(pretrained=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        layers = [v for v in model.children()]
        self.model = nn.Sequential(*layers[:-2])
        self.pool = layers[-2]
        self.fc   = layers[-1]

    def forward(self, x, **kwargs): 
        x = self.model(x) # [B, C, H, W]
        outs = Munch()

        x = self.fc(torch.flatten(self.pool(x), 1))
        outs.sup = x
        return outs
