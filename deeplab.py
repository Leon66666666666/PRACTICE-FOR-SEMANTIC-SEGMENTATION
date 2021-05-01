#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/4/23 21:49
# @Author : v
# @File : models.py
# @Software: PyCharm
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose

class_number = 32  

print('python version：' + sys.version)
print('tensorflow version：' + tf.__version__)


def encoder(input_height, input_width):
    

    # input
    img_input = Input(shape=(input_height, input_width, 3))

    
    # 416,416,3 -> 208,208,64,
    x = Conv2D(64, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(img_input)

    x = Conv2D(64, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = MaxPool2D((2, 2),
                  strides=(2, 2))(x)
    f1 = x  

    # 208,208,64 -> 104,104,128
    x = Conv2D(128, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = Conv2D(128, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = MaxPool2D((2, 2),
                  strides=(2, 2))(x)
    f2 = x  
    # 104,104,128 -> 52,52,256
    x = Conv2D(256, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = Conv2D(256, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)
    x = Conv2D(256, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = MaxPool2D((2, 2),
                  strides=(2, 2))(x)
    f3 = x  
    # 52,52,256 -> 26,26,512
    x = Conv2D(512, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = Conv2D(512, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = Conv2D(512, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = MaxPool2D((2, 2),
                  strides=(2, 2))(x)
    f4 = x  

    # 26,26,512 -> 13,13,512
    x = Conv2D(512, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = Conv2D(512, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = Conv2D(512, (3, 3),
               dilation_rate=2,
               activation='relu',
               padding='same')(x)

    x = MaxPool2D((2, 2),
                  strides=(2, 2))(x)
    f5 = x  

    return img_input, [f1, f2, f3, f4, f5]


def decoder(feature_map_list, class_number, input_height=512, input_width=256, encoder_level=3):
   
    feature_map = feature_map_list[encoder_level]

    # decoder（26,26,512） -> (208,208,64)

    # f4.shape=(26,26,512) -> 26,26,512
    x = ZeroPadding2D((1, 1))(feature_map)
    x = Conv2D(512, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # (26,26,512) -> (52,52,256)
    x=Conv2DTranspose(256,(3,3),strides=(2,2),activation='relu',padding='same')(x)
    #x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # (52,52,512) -> (104,104,128)
    #x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2,2),activation='relu', padding='same')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # (104,104,128) -> (208,208,64)
    #x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2,2),activation='relu', padding='same')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # (208,208,64) -> (208,208,2)
    x = Conv2D(class_number, (3, 3), padding='same')(x)
    # reshape: (208,208,2) -> (208*208,2)
    x = Reshape((int(input_height / 2) * int(input_width / 2), -1))(x)
    
    output = Softmax()(x)

    return output


def main(Height=416, Width=416):
    """ main function"""

   
    img_input, feature_map_list = encoder(input_height=Height, input_width=Width)

  
    output = decoder(feature_map_list, class_number=class_number, input_height=Height, input_width=Width,
                     encoder_level=3)

    model = Model(img_input, output)

    # model.summary()


    return model


if __name__ == '__main__':
    main(Height=416, Width=416)
