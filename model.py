import typing

import torch
from torch import nn
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def LoadEmptyModel() -> models.ResNet:
    """Loads an empty resnet model and freezes the resnet backbone"""
    model = models.resnet50(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features

    # Use ResNet50 as a backbone and a 2 layer Neural network in front of it
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                             nn.ReLU(),
                             nn.Linear(128, 4)
                             )
    return model
