"""
Script to collect images of the rock paper scissor hand gestures so as to create a dataset
"""
import argparse
import os
import sys
import logging

import cv2


def ParseArgs():
    parser = argparse.ArgumentParser(description="Parser to parse collecting images for rock paper scissors dataset")
    parser.add_argument('dataset_dir', type=str, help='Path to the dir where the dataset will be created')
    parser.add_argument('--label', type=str, choices=['rock', 'paper', 'scissors', 'none'], required=True, help="Class for which you want to assign the captured images")
    parser.add_argument('--num_images', type=int, required=True, help="Maximum number of images you want to capture")
    args = parser.parse_args()
    return args


def Main():
    args = ParseArgs()

    # Make individual directory for individual classes
    try:
        dir_to_label = os.path.join(args.dataset_dir, args.label)
        os.makedirs(dir_to_label)
    except FileExistsError:
        logging.info("The class already exists")
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Cannot open camera")

    start = False
    count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        
        # if frame is read correctly ret is True
        if not ret:
            continue

        cv2.rectangle(frame, (100, 100), (500, 500), (255, 0, 0), 2)
        cv2.putText(frame, "Captured {}".format(count), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Capturing images for dataset", frame)

        key = cv2.waitKey(10)
        
        # Start capturing images
        if key == ord('s'):
            start = True

        if key == ord('x'):
            break

        if start:
            bbox = frame[100:500, 100:500]
            cv2.imwrite(os.path.join(dir_to_label, '{}.jpg'.format(count+1)), bbox)
            count += 1
        
        # Terminating of while loop
        if count == args.num_images:
            break
    
    # When everything related to camera is done
    camera.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    Main()