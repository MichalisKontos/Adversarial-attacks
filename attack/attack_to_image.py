# Input image of simulation (road)
# Depict the predicting steering angle without any attack
# Perform any of the attacks
# Show the new predicted steering angle along with difference from the original
#  and graphical difference

# TODO
# -> Read paper, adjust code according to it 
#bored in the house and im in the house bored

import os
import utils
import argparse
import pickle

# Deep Learning Libraries
import numpy as np
from keras.models import load_model
import cv2

# Used for the fgsm attack
import tensorflow as tf
import keras.backend as K

# Disable eager execution cause it was causing some errors
tf.compat.v1.disable_eager_execution()

# Attack function to introduce noise to the image
def attack(attack_type, image):

    # Random Noise
    if attack_type == "random":
         # Random Noise [-1, 1]
        noise = np.random.randint(2, size=(160, 320, 3)) - 1
        corrupt_image = np.array(image) + noise
        return corrupt_image,noise
            
    # One-Pixel Attack
    elif attack_type == "one_pixel":
	    # The attack increases each pixel by one
        noise = np.ones((160, 320, 3))
        corrupt_image = np.array(image) + noise
        noise = cv2.convertScaleAbs(noise)
        corrupt_image = cv2.convertScaleAbs(corrupt_image)
        return corrupt_image,noise

    # Fast gradient sign method attack
    # This attack attacks the vehicle to either side
    elif attack_type == "fgsm":
        # FGSM attack variables
        # ε is a multiplier to ensure pertubations are small
        # adversarial image = original image + ε * sign(∇x J(θ,x,y))
        # adversarial image = original image + ε * gradient of the loss w.r.t the input image
        epsilon = float(input('Enter the value of epsilon you want: '))
        # Get the loss and gradient of the loss wrt the inputs
        loss = K.mean(-model.output, axis=-1)
        grads = K.gradients(loss, model.input)
        # Get the sign of the gradient
        delta = K.sign(grads[0])
        # Calculate perturbation
        sess = tf.compat.v1.keras.backend.get_session()
        noise = epsilon * sess.run(delta, feed_dict={model.input:np.array([image])}) #(1, 160, 320, 3) shape
        noise = noise[0]
        corrupt_image = np.array(image) + noise
        corrupt_image = cv2.convertScaleAbs(corrupt_image)
        return corrupt_image, noise
    
    # Universal Adversarial Perturbation attack
    # This attack makes it hard for the car to turn in the direction specified
    elif attack_type == "uap":
        # Read the pickle files from the trained attacks
        if(os.path.isfile("unir_no_left.pickle")):
            with open('unir_no_left.pickle', 'rb') as f:
                unir_no_left = set_unir_no_left(pickle.load(f))
        if(os.path.isfile("unir_no_right.pickle")):
            with open('unir_no_right.pickle', 'rb') as f:
                unir_no_right = set_unir_no_right(pickle.load(f))
        att = str(input('From which side do you want to attack? '))
        if att == "left":
            noise = unir_no_left.reshape(160, 320, 3)
        elif att == "right":
            noise = unir_no_right.reshape(160, 320, 3)
        else:
            raise Exception("Sorry, invalid direction of UAP attack")
        corrupt_image = np.array(image) + noise
        corrupt_image = cv2.convertScaleAbs(corrupt_image)
        return corrupt_image, noise

    # If no argument is passed, return the original image (add zero noise)
    elif attack_type == "":
        noise = np.zeros((160, 320, 3))
        corrupt_image = np.array(image) + noise
        noise = cv2.convertScaleAbs(noise)
        corrupt_image = cv2.convertScaleAbs(corrupt_image)
        return corrupt_image,noise
    
    # If attack type is invalid
    else:
        raise Exception("Sorry, invalid attack type")    

def set_unir_no_left(unir_no_left):
    unir_no_left = unir_no_left
    return unir_no_left

def set_unir_no_right(unir_no_right):
    unir_no_right = unir_no_right
    return unir_no_right

# Function to compute the Mean Squared Error between two images
def MSE(img1, img2):
    squared_diff = (img1 - img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return err

# Function to calculate steering angle based on input image
# Makes prediction based on trained model (model.h5 file)
def calculate_steering_angle(image):
    # Without attack, image its just an array
    image = np.array([image]) 
    # predict the steering angle from the image
    steering_angle = float(model.predict(image, batch_size=1))
    return steering_angle
    
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Steering angle prediction and attacks')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image',
        nargs='?',
        default=None,
        help='Image file to perform the prediction and attack.'
    )
    parser.add_argument(
        'attack_type',
        type=str,
        nargs='?',
        default="",
        help='String, attack type e.g random, one_pixel, fgsm, uap.'
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)
    
    # Read input image & apply the preprocessing
    img = cv2.imread(args.image)
    img = utils.preprocess(img) 

    # Calculate steering angle
    steering_angle = calculate_steering_angle(img)
    
    # Perform attack on image
    attack_type = args.attack_type
    corrupt_image,noise = attack(attack_type, img)

    # Calculate the new steering angle on corrupted image
    corrupt_steering_angle = calculate_steering_angle(corrupt_image)

    # Calculate percentage difference between correct steering angle and corrupted steering angle
    percentage_diff = (np.abs(steering_angle - corrupt_steering_angle)*100)/np.abs(steering_angle)

    # Compute the Mean Square Error from original and corrupt images
    mse = MSE(img,corrupt_image)

    # Print values and display images
    print("Car should turn by",steering_angle)
    print("Corrupted steering angle is", corrupt_steering_angle)
    print("Percentage difference is", percentage_diff, "%")
    print("Mean Square Error:", mse)
    utils.display_info(img, corrupt_image, noise, steering_angle, corrupt_steering_angle, attack_type)