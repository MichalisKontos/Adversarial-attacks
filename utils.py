import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Function to display images and a small graph
def display_info(original_image, corrupt_image, noise, steering_angle, corrupt_steering_angle, attack):
    fig = plt.figure()
    fig.suptitle(attack.upper())

    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, 4, 2)
    plt.imshow(corrupt_image)
    plt.title("Corrupt Image")

    plt.subplot(1, 4, 3)
    plt.imshow(noise)
    plt.title("Noise")

    plt.subplot(1, 4, 4)
    x = [0, 1]
    y = [steering_angle, corrupt_steering_angle]
    plt.plot(x, y, marker='o', markerfacecolor='blue', markersize=8)
    plt.xlabel("Correct Steering angle")
    plt.ylabel("Corrupted steering angle")
    plt.title("Steering angles")

    plt.tight_layout()
    plt.show()    

def crop(image):
    # Crop the image (removing the sky at the top and the car front at the bottom)
    return image[60:-25, :, :] # remove the sky and the car front

def resize(image):
    # Resize the image to the input shape used by the network model
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def preprocess(image):
    # Combine all preprocess functions into one
    image = crop(image)
    image = resize(image)
    return image