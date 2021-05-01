#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/4/24 18:20
# @Author : v
# @File : train.py
# @Software: PyCharm

import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import get_file
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import models
#from deeplab import Deeplabv3

CLASS_NUMBERS = 32  # number of classification
HEIGHT = 416 
WIDHT = 416 
batch_size = 4  


def customied_loss(y_true, y_pred):
    """ loss function"""
    loss = categorical_crossentropy(y_true, y_pred)
    return loss


def get_model():
    """ load model """

   
    model = models.main()
    # model.summary() cache_subdir='models_dir'

    # VGG parameters
    #WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #filename = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'  # 下载后保存的文件名
    checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    #weights_path = get_file(filename, WEIGHTS_PATH_NO_TOP,cache_subdir='models')
    weights_path = (r"model/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5")
    #weights_path=(r"callbacks/last.h5")

    # print(weights_path)

    # load parameters
    model.load_weights(weights_path, by_name=True)


    model.compile(loss=customied_loss, optimizer=Adam(1e-5), metrics=['accuracy'])

    return model


def get_data():
    """
    extract the training dataset and validation dataset
   
    """

    
   
    with open('Datasets/dataset/trainData/train.txt', 'r') as f:
        lines = f.readlines()
    # print(lines)

    # disorder the image sequnence 
    np.random.seed(10101)  
    np.random.shuffle(lines)
    np.random.seed(None)

    # 30% for validation and 70% for training
    num_val = int(len(lines) * 0.3)
    num_train = len(lines) - num_val
    # print(num_val)

    return lines, num_train, num_val


def set_callbacks():
    """ callback function"""

    # 1. 有关回调函数的设置（callbacks)
    logdir = os.path.join("callbacks")
    print(logdir)
    if not os.path.exists(logdir):  
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    callbacks = [
        TensorBoard(logdir),
        ModelCheckpoint(output_model_file, save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3),
        EarlyStopping(min_delta=1e-3, patience=10)
    ]

    return callbacks, logdir


def generate_arrays_from_file(lines, batch_size):
    """
    generator
     
    """

    numbers = len(lines)  

    read_line = 0  
    while True:

        train_data = []  # training dataset
        train_label = []  # label

        # batch_size

        for t in range(batch_size):

            # print(t)
            # when the one epoch is done, disorder the images sequnence 
            if read_line == 0:
                np.random.shuffle((lines))

            

            # 1. extract  the names of triaining data
            train_x_name = lines[read_line].split(';')[0]

            # feed the images to the networks according the name
            img = Image.open('Datasets/dataset/trainData/trainpng' + '/' + train_x_name)
            # img.show()
            # print(img.size)
            img = img.resize((WIDHT, HEIGHT))  # resize
            img_array = np.array(img)  #  image to array


            img_array = img_array / 255  # normalization

            train_data.append(img_array)  # 

            # labels
            train_y_name = lines[read_line].split(";")[1].replace('\n', '')

            # 
            img = Image.open('Datasets/dataset/trainData/labelpng' + '/' + train_y_name)
            # img.show()
            # print(train_y_name)
            img = img.resize((int(WIDHT / 2), int(HEIGHT / 2)))  
            img_array = np.array(img)  
          

            # create the set for labels
            labels = np.zeros((int(HEIGHT / 2), int(WIDHT / 2), CLASS_NUMBERS))
            # print('label shape: ', labels.shape)

            # 下面将(208,208,3) => (208,208,2),不仅是通道数的变化，还有，
            # 原本背景和斑马线在一个通道里面，现在将斑马线和背景放在不同的通道里面。
            # 如，labels,第0通道放背景，是背景的位置，显示为1，其余位置显示为0
            # labels, 第1通道放斑马线，图上斑马线的位置，显示1，其余位置显示为0
            # 相当于合并的图层分层！！！！
            for cn in range(CLASS_NUMBERS):  # range(0,2) => 0,1
                # 标签数组中，斑马线的值为1，其他为0 （三个通道值相同！！！），所以下面选第0通道
                labels[:, :, cn] = (img_array[:] == cn).astype(int)
            labels = np.reshape(labels, (-1, CLASS_NUMBERS))  # (208,208,2)=>(208*208,2)
            train_label.append(labels)

            # 遍历所有数据，记录现在所处的行，读取完所有数据后，read_line=0,上面会打乱，重新开始
            read_line = (read_line + 1) % numbers

        yield (np.array(train_data), np.array(train_label))


def main():
    # main function
    model = get_model()
   
    # model.summary()

    # get the images and labels
    lines, train_nums, val_nums = get_data()

    # callback function
    callbacks, logdir = set_callbacks()

    # get the training data and labels
    generate_arrays_from_file(lines, batch_size=4)

    # training
    model.fit_generator(generate_arrays_from_file(lines[:train_nums], batch_size),
                        steps_per_epoch=max(1, train_nums // batch_size),
                        epochs=80, callbacks=callbacks,
                        validation_data=generate_arrays_from_file(lines[train_nums:], batch_size),
                        validation_steps=max(1, val_nums // batch_size),
                        initial_epoch=0)

    save_weight_path = os.path.join(logdir,'final.h5') # save the results

    model.save_weights(save_weight_path)


if __name__ == '__main__':
    main()
