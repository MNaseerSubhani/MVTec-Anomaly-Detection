"""
Model inspired by: https://github.com/natasasdj/anomalyDetection
"""

import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Normalization,
    LeakyReLU,
    Activation,
    concatenate,
    Flatten,
    Reshape,
    Add,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
SHAPE = (512, 512)

# Preprocessing parameters
RESCALE = 1.0 / 255
blur_value = 3
kernel = np.ones((blur_value,blur_value),np.float32)/(blur_value*blur_value)
kernel_1 = np.ones((5, 5), np.uint8)
def Input_Process(img):
  img = img.reshape(*SHAPE)
  gray_b = cv2.filter2D(img,-1,kernel)
  
  return gray_b.reshape(*SHAPE, 1)
  

PREPROCESSING_FUNCTION = None
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
    img_dim = (*SHAPE, channels)

    # input
    input_img = Input(shape=img_dim)
    # x = Normalization(mean=0.39732972826936414, variance=0.010313226193989894)(input_img)
    # encoder
    
    encoding_dim = 64  # 128
    x = Conv2D(8, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(
        input_img
    )
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    
    # ---------------------------------------------------------------------------------

    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    

    # added ---------------------------------------------------------------------------
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    #t = x
    
    # ---------------------------------------------------------------------------------

    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Dropout(0.5)(x)
    

    # added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Flatten()(x)
    x = Dense(encoding_dim, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    # encoded = x

    # decoder
    x = Reshape((4, 4, encoding_dim // 16))(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    

    ## added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    
    # ---------------------------------------------------------------------------------

    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    #x = Add()([x, t])
    

    ## added ---------------------------------------------------------------------------
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

 
    # ---------------------------------------------------------------------------------

    x = Conv2D(
        img_dim[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    decoded = x
    # model
    autoencoder = Model(input_img, decoded)
    return autoencoder
