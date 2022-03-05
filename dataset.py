import os
import numpy as np
import cv2
import time

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

TRAIN_DATA_PATH = "/Users/ambuj/Desktop/GithubProjects/rock_paper_scissor/dataset/train"
TEST_DATA_PATH = "/Users/ambuj/Desktop/GithubProjects/rock_paper_scissor/dataset/test"

train_data = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
test_data = datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
print(train_data.class_to_idx)

trainloader = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=False, num_workers=0)
testloader = DataLoader(test_data, batch_size=4, shuffle=True, drop_last=False, num_workers=0)

model = models.densenet121(pretrained=True)
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 4)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

train_losses, test_losses = [], []
epochs = 7
steps = 0

for e in range(epochs):
    running_loss = 0

    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        train_losses.append(running_loss)

        if steps % 5 == 0:
            test_loss, accuracy = 0, 0

            with torch.no_grad():
                model.eval()

                for images, labels, in testloader:
                    images, labels = images.to(device), labels.to(device)

                    logits = model(images)

                    test_loss += criterion(logits, labels)

                    ps = torch.exp(logits)

                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch: {e + 1}/{epochs};"
                  f"Train_loss: {running_loss};"
                  f"Test_loss: {test_loss / len(testloader)};"
                  f"Accuracy: {accuracy / len(testloader)}")
            model.train()
            running_loss = 0


torch.save(model.state_dict(), '/Users/ambuj/Desktop/GithubProjects/rock_paper_scissor/model.pt')
