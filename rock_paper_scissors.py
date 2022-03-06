"""
Play a real-time game of rock paper scissors against the computer.
Place your choice on the user's box and see who won.
"""

import cv2
from PIL import Image
import typing
from random import choice

import torch
from torchvision import models, transforms
from torch.autograd import Variable

from model import LoadEmptyModel

WEIGHTS_PATH = '../resnet50_cv_cpu.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LABEL_TO_CLASS_MAPPING = {
    0: "none",
    1: "paper",
    2: "rock",
    3: "scissors"
}


def GetUserMove(image_path: str, model: models.resnet.ResNet):
    """ Output class produced by the model. Returns it in string format """
    loader = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    image = Image.open(image_path)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    output = model(image.to(device))
    predicted_move = torch.argmax(output).item()
    print(predicted_move)

    # predict the move made
    if predicted_move == 0:
        return "none"
    if predicted_move == 1:
        return "paper"
    elif predicted_move == 2:
        return "rock"
    elif predicted_move == 3:
        return "scissors"
    else:
        raise ValueError("Can only be 3 options. Rock, paper or scissors")


def LoadRPSModel(weights_path: str = WEIGHTS_PATH):
    """Loads the model and prepares it for inference"""
    model = LoadEmptyModel()
    model.eval()
    model.load_state_dict(torch.load(weights_path))
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


def GetComputerMove() -> str:
    """Returns a random choice between rock, paper, scissors as a computer's guess"""
    return choice(['rock', 'paper', 'scissors'])


def Main():
    model = LoadRPSModel(WEIGHTS_PATH)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Cannot open camera")

    prev_move = None

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        # if frame is read correctly ret is True
        if not ret:
            continue

        # User's rectangle
        cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)
        # Computer's rectangle
        cv2.rectangle(frame, (800, 100), (1100, 400), (255, 255, 255), 2)

        # extract the region of image within the user rectangle
        roi = frame[100:400, 100:400]

        image_path = "./user_input.jpg"
        cv2.imwrite(image_path, roi)
        # Get Different moves
        player_move = GetUserMove(image_path, model)
        print(player_move)

        # predict the winner (human vs computer)
        if prev_move != player_move:
            if player_move != "none":
                computer_move = GetComputerMove()
                winner = FindWinner(player_move, computer_move)
            else:
                computer_move = "none"
                winner = "Waiting for user!"

        prev_move = player_move

        # Display text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Your Move: " + player_move, (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + computer_move, (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Winner: " + winner, (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

        if computer_move != "none":
            icon = cv2.imread(
                "../dataset/stock/{}.png".format(computer_move))
            icon = cv2.resize(icon, (300, 300))
            frame[100:400, 800:1100] = icon

        cv2.imshow("Rock Paper Scissors", frame)

        key = cv2.waitKey(10)
        if key == ord('x'):
            break

    # When everything related to camera is done
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Main()
