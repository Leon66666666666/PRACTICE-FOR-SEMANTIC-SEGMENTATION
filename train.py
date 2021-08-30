#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/4/24 18:20
# @Author : v
# @File : train.py
# @Software: PyCharm

import os, sys
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import cv2
import deeplab
import segnet
import segnet
CLASS_NUMBERS = 19  # classification numbers

WIDHT = 416
HEIGHT = 416
batch_size = 2  

"""data augmentation"""
data_aug=ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')

def random_flip(image):
    image=cv2.flip(image,1)
    return(image)

def shift(image):
    m=np.float32([[1,0,100],[0,1,0]])
    image=cv2.warpAffine(image,m,(image.shape[1],image.shape[0]))
    return image

def crop(image):
    h,w=image.shape[0],image.shape[1]
    scale=np.random.randint(h//2,h)/h
    crop_h=int(scale*h)
    crop_w=int(scale*w)
    h_begin=np.random.randint(0,h-crop_h)
    w_begin=np.random.randint(0,w-crop_w)
    image_crop=image[h_begin:h_begin+crop_h,w_begin:w_begin+crop_w,:]

    image_resize=cv2.resize(image_crop,(w,h))
    return image_resize




def addnoise(image):
    for i in range(1500):
        image[np.random.randint(0,image.shape[0]-1)][np.random.randint(0,image.shape[1]-1)][:]=255
    return image






def customied_loss(y_true, y_pred):
    """ loss function"""
    loss = binary_crossentropy(y_true, y_pred)
    return loss


def get_model():
    """load models """

    
    model = deeplab.main()
    # model.summary() cache_subdir='models_dir'

    # VGG OR resnet model
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    filename = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'  # initial weight
    checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    weights_path = get_file(filename, WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    # print(weights_path)

    # load the initial weight
    model.load_weights(weights_path, by_name=True)

    # complie
    model.compile(loss=customied_loss, optimizer=Adam(1e-3), metrics=['accuracy'])

    return model


def input_data():
    """
   extract the training data and validation data, return the name of the training data and validation data
    """

    
    with open('Datasets/dataset/trainData/train.txt', 'r') as f:
        lines = f.readlines()
    # print(lines)

    # shuffle
    np.random.seed(10101)  
    np.random.shuffle(lines)
    np.random.seed(None)

    # 70% for training and 30% for validation
    num_validation = int(len(lines) * 0.5)
    num_train = len(lines) - num_validation
    # print(num_validation)

    return lines, num_train, num_validation


def callbacksfunction():
    """ callback function"""

   
    logdir = os.path.join("callbacks")
    print(logdir)
    if not os.path.exists(logdir):  
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    callbacks = [
        #TensorBoard(logdir),
        ModelCheckpoint(output_model_file, save_best_only=True, save_freq='epoch'),
        ReduceLROnPlateau(factor=0.5, patience=3),
        EarlyStopping(min_delta=1e-3, patience=10)
    ]






    return callbacks, logdir


def generate_arrays_from_file(lines, batch_size):
    """
    generator
    """

    numbers = len(lines)  # measure the inout data

    read_line = 0  # first line
    while True:

        train_data = []  
        train_label = []  

        # for the images in one batch 

        for t in range(batch_size):

            # print(t)
            # shuffle after one epoch
            if read_line == 0:
                np.random.shuffle((lines))


            #  extrac the names of the training data and label in the txt
            train_data_name = lines[read_line].split(';')[0]

         
            img = Image.open('Datasets/dataset/trainData/trainpng' + '/' + train_data_name)

            # img.show()
            # print(img.size)
            img = img.resize((WIDHT, HEIGHT))  


            img_array = np.array(img)  
            img_array1 = shift(image=img_array)


            img_array = img_array / 255# normalization img_array.shape=（416，416，3）
            img_array1 = img_array1 / 255

            train_data.append(img_array)
            train_data.append(img_array1)
            
            train_label_name = lines[read_line].split(";")[1].replace('\n', '')

            # labels
            img = Image.open('Datasets/dataset/trainData/labelpng' + '/' + train_label_name)
            # img.show()
            # print(train_label_name)
            img = img.resize((int(WIDHT / 2), int(HEIGHT / 2)))  
            img_array = np.array(img)
            img_array2 = shift(image=img_array)

            labels = np.zeros((int(HEIGHT / 2), int(WIDHT / 2), CLASS_NUMBERS))
            labels1 = np.zeros((int(HEIGHT / 2), int(WIDHT / 2), CLASS_NUMBERS))
            # print('label shape: ', labels.shape)

            # one-hot coding
            for cn in range(CLASS_NUMBERS):  
                labels[:, :, cn] = (img_array[:, :, 0] == cn).astype(int)
                labels1[:, :, cn] = (img_array2[:, :, 0] == cn).astype(int)
            labels = np.reshape(labels, (-1, CLASS_NUMBERS))
            labels1 = np.reshape(labels1, (-1, CLASS_NUMBERS))# (208,208,2)=>(208*208,2)

            train_label.append(labels)
            train_label.append(labels1)

            read_line = (read_line + 1) % numbers



        yield (np.array(train_data), np.array(train_label))




def main():
    # load models
    model = get_model()
    
    # model.summary()
    lines, train_nums, val_nums = input_data()

  
    callbacks, logdir = callbacksfunction()


    generate_arrays_from_file(lines[:train_nums], batch_size=2)

    # start training
    model.fit_generator(generate_arrays_from_file(lines[:train_nums], batch_size=2),
                        steps_per_epoch=max(1, train_nums*2 // batch_size),
                        epochs=1, callbacks=callbacks,
                        validation_data=generate_arrays_from_file(lines[train_nums:], batch_size),
                        validation_steps=max(1, val_nums // batch_size),
                        initial_epoch=0)

    save_weight_path = os.path.join(logdir,'result.h5') # final weight

    model.save_weights(save_weight_path)




if __name__ == '__main__':
    main()
