import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
import cv2
import numpy as np


SHAPE = (512,512)
blur_value = 3
kernel = np.ones((blur_value,blur_value),np.float32)/(blur_value*blur_value)
kernel_1 = np.ones((30, 30), np.uint8)
def Input_Process(img):
  img = img.reshape(*SHAPE)
  img = cv2.filter2D(img,-1,kernel)
  # n_map = np.where(gray_b > 162 ,1.0,0)
  # n_map = cv2.morphologyEx(n_map, cv2.MORPH_CLOSE, kernel_1)
  
  return img.reshape(*SHAPE, 1)
  


class Preprocessor:
    def __init__(
        self, input_directory, rescale, shape, color_mode, preprocessing_function,
    ):
        self.input_directory = input_directory
        self.train_data_dir = os.path.join(input_directory, "train")
        self.test_data_dir = os.path.join(input_directory, "test")
        self.rescale = rescale
        self.shape = shape
        self.color_mode = color_mode
        self.preprocessing_function = preprocessing_function
        self.validation_split = config.VAL_SPLIT

        self.img_shp = 512
        self.cnt = int(self.img_shp/2)
        self.glb_mask = cv2.circle(np.zeros((self.img_shp, self.img_shp)), (self.cnt, self.cnt), self.cnt , (1,1,1), -1) 
        self.glb_mask = np.array(self.glb_mask, dtype = 'uint8')
        self.nb_val_images = None
        self.nb_test_images = None

    def get_train_generator(self, batch_size, shuffle=True):
        # This will do preprocessing and realtime data augmentation:
        train_datagen = ImageDataGenerator(
            # standarize input
            featurewise_center=False,
            featurewise_std_normalization=False,
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=config.ROT_ANGLE,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=config.W_SHIFT_RANGE,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=config.H_SHIFT_RANGE,
            # set mode for filling points outside the input boundaries
            fill_mode=config.FILL_MODE,
            # value used for fill_mode = "constant"
            cval=0.0,
            # randomly change brightness (darker < 1 < brighter)
            brightness_range=config.BRIGHTNESS_RANGE,
            # set rescaling factor (applied before any other transformation)
            rescale=self.rescale,
            # set function that will be applied on each input
            preprocessing_function=self.preprocessing_function,
            # image data format, either "channels_first" or "channels_last"
            data_format="channels_last",
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=self.validation_split,
        )

        # Generate training batches with datagen.flow_from_directory()
        train_generator = train_datagen.flow_from_directory(
            directory=self.train_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            subset="training",
            shuffle=True,
        )
        return train_generator
    
    

    def get_val_generator(self, batch_size, shuffle=True):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For validation dataset, only rescaling
        validation_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            validation_split=self.validation_split,
            preprocessing_function=self.preprocessing_function,
        )
        # Generate validation batches with datagen.flow_from_directory()
        validation_generator = validation_datagen.flow_from_directory(
            directory=self.train_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            subset="validation",
            shuffle=shuffle,
        )
        return validation_generator

    def get_test_generator(self, batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        test_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=self.preprocessing_function,
        )

        # Generate validation batches with datagen.flow_from_directory()
        test_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return test_generator
    
    def get_test_image(self, img_pth):
        img = cv2.imread(img_pth)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize_img = cv2.resize(gray, (self.shape[0], self.shape[1]) , interpolation = cv2.INTER_AREA)
        
        resize_img = cv2.bitwise_and(resize_img, resize_img, mask=self.glb_mask)
        #resize_img = Input_Process(resize_img)
        resize_img = resize_img.reshape(1,self.shape[0],self.shape[1],1 ) 

        return resize_img/255.0
      



    def get_finetuning_generator(self, batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        test_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=self.preprocessing_function,
        )

        # Generate validation batches with datagen.flow_from_directory()
        finetuning_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return finetuning_generator

    def get_total_number_test_images(self):
        total_number = 0
        sub_dir_names = os.listdir(self.test_data_dir)
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(self.test_data_dir, sub_dir_name)
            filenames = os.listdir(sub_dir_path)
            number = len(filenames)
            total_number = total_number + number
        return total_number


def get_preprocessing_function(architecture):
    if architecture in ["mvtecCAE", "baselineCAE", "baselineCAE_fast", "indexptionCAE", "resnetCAE", "skipCAE"]:
        preprocessing_function = None
    return preprocessing_function
