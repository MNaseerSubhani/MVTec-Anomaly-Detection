"""
Implementation of mvtec architecture inspired by:
https://github.com/cheapthrillandwine/Improving_Unsupervised_Defect_Segmentation/blob/master/Improving_AutoEncoder_Samples.ipynb
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
# Preprocessing parameters
RESCALE = 1.0 / 255
SHAPE = (512, 512)


# def Input_Process(img):
#   img = img.reshape(SHAPE)
#   img -= np.mean(img)
#   return img.reshape(*SHAPE, 1)
blur_value = 2
kernel = np.ones((blur_value,blur_value),np.float32)/(blur_value*blur_value)
kernel_1 = np.ones((30, 30), np.uint8)

def augmentation(img):
    img = img/255
    img = np.array(img*255, dtype='uint8')
    if(bool(random.getrandbits(1))):
        img = cv2.equalizeHist(img)

    if(bool(random.getrandbits(1))):
        img = cv2.flip(img, 1)
        
    if(bool(random.getrandbits(1))):
        img = cv2.flip(img, 0)
    
    img = img/255.00

    return img * 255

	
def Input_Process(img):
  img = img.reshape(SHAPE)
  #img = np.where(img == 0, random.randint(0,255)/255, img)
  img = augmentation(img)
  # n_map = np.where(gray_b > 162 ,1.0,0)
  # n_map = cv2.morphologyEx(n_map, cv2.MORPH_CLOSE, kernel_1)
  
  return img.reshape(*SHAPE, 1)
PREPROCESSING_FUNCTION = Input_Process
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3

    # define model
    input_img = keras.layers.Input(shape=(*SHAPE, channels))
    # Encode-----------------------------------------------------------
    x = keras.layers.Conv2D(16, (4, 4), strides=2, activation="relu", padding="same")(
        input_img
    )
    x = keras.layers.Conv2D(16, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(16, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(16, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(16, (3, 3), strides=1, activation="relu", padding="same")(x)
    encoded = keras.layers.Conv2D(1, (8, 8), strides=1, padding="same")(x)

    # Decode---------------------------------------------------------------------
    x = keras.layers.Conv2D(16, (3, 3), strides=1, activation="relu", padding="same")(
        encoded
    )
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(16, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(16, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((4, 4))(x)
    x = keras.layers.Conv2D(16, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(16, (8, 8), activation="relu", padding="same")(x)

    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(
        channels, (8, 8), activation="sigmoid", padding="same"
    )(x)

    model = keras.models.Model(input_img, decoded)

    return model
