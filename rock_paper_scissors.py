from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image
import typing
from random import choice

import torch
from torch import nn
from torchvision import models, transforms
from torch.autograd import Variable

WEIGHTS_PATH = '/Users/ambuj/Desktop/GithubProjects/rock_paper_scissor/model_3.pt'


def GetUserMove(image_captured, model):
    """ Output class produced by the model. Returns it in string format """

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    pil_image = Image.fromarray(np.array(image_captured))
    image = transform(pil_image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    output = model(image)
    predicted_move = output.data.numpy().argmax()
    print(predicted_move)

    # predict the move made
    if predicted_move == 0:
        return "paper"
    elif predicted_move == 1:
        return "rock"
    elif predicted_move == 2:
        return "scissors"
    else:
        raise ValueError("Can only be 3 options. Rock, paper or scissors")


def LoadRPSModel(weights_path):
    """Loads the model and prepares it for inference"""
    model = models.densenet121(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 3)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def FindWinner(player_move: str, computer_move: str) -> str:
    """Takes player's move and computer's move into consideration and returns the winner"""

    if player_move == computer_move:
        return "It's a tie!!"

    if player_move == "rock":
        if computer_move == "paper":
            return "Computer wins!"
        elif computer_move == "scissors":
            return "User wins!"

    elif player_move == "scissors":
        if computer_move == "rock":
            return "Computer wins!"
        elif computer_move == "paper":
            return "User wins!"

    elif player_move == "paper":
        if computer_move == "scissors":
            return "Computer wins!"
        elif computer_move == "rock":
            return "User wins!"

    else:
        raise Exception("Unknown move found!")


def GetComputerMove():
    return choice(['rock', 'paper', 'scissors'])


def Main():

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Cannot open camera")

    start = False
    player_move = ''

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        # if frame is read correctly ret is True
        if not ret:
            continue

        # User's rectangle
        cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
        # Computer's rectangle
        cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)
        cv2.imshow("Rock Paper Scissors", frame)

        key = cv2.waitKey(10)

        # Start the game
        if key == ord('n'):
            # extract the region of image within the user rectangle
            roi = frame[100:500, 100:500]
            image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Get Different moves
            player_move = GetUserMove(image, LoadRPSModel(WEIGHTS_PATH))
            print(player_move)
            computer_move = GetComputerMove()
            winner = FindWinner(player_move, computer_move)

            # display the information
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Your Move: " + player_move,
                        (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Computer's Move: " + computer_move,
                        (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Winner: " + winner,
                        (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)


        if key == ord('x'):
            break

    # When everything related to camera is done
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Main()
