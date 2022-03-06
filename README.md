Play a rock paper scissors game in real time with a computer.

## Requirements
* Python 3.7.11
* PyTorch 1.10.2
* OpenCV 3.4.2

Or you can simply use the environment.yml file attached to this repo

`conda env create -f environment.yml`

## Creating Dataset
Firstly, we have to create the dataset. For this, we use the webcam of the PC to generate images of the different classes. The script provided in collect_images.py helps the user to collect the images upon which the model will be trained later.

`python3 collect_images.py ./dataset/ --label rock --num_images 300`

Simply, follow these steps for paper, scissors and none class


## Training the model

With the dataset created, you can train the model using the notebook attached to this repo. The notebook also contain utility functions to visualize the output of the trained model as well.


## Play the game
After training the model and satisfied with the results. We can then move on to playing the game of rock paper scissors with the almighty computer.
Simply load the model you saved after training in the script. And lets move on to playing.

