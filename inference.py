import argparse
from collections import OrderedDict

import torch
from torch import nn
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image


def ParseArgs():
    parser = argparse.ArgumentParser(description="Parser to make inference on the provided image")
    parser.add_argument('weights_path', type=str, help='Path to the model')
    parser.add_argument('image_path', type=str, help="Path to the image")
    args = parser.parse_args()
    return args


def CreateTransforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def LoadImage(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = CreateTransforms()(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image


def Main():
    args = ParseArgs()
    model = models.densenet121(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 3)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = LoadImage(args.image_path).to(device)
    output = model(image)
    index = output.data.numpy().argmax()
    print(index)


if __name__ == '__main__':
    Main()
