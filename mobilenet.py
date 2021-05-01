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

class_number = 32  # 分类数

print('python version：' + sys.version)
print('tensorflow version：' + tf.__version__)


def mobilenet_con(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='SAME', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def depthwise_conv(inputs,pointwise_conv_filters,strides=(1, 1)):
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='SAME',
                                        use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=pointwise_conv_filters, kernel_size=(1, 1),
                               padding='SAME', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)
    return out


def mobilenet_encoder(input_height, input_width):
    img_input = Input(shape=(input_height, input_width, 3))
    x = mobilenet_con(img_input, 64, strides=(2, 2))
    f1 = x
    # [208, 208, 64] => [104, 104, 64]
    x = depthwise_conv(x, 64)
    # [104, 104, 64] => [52, 52, 128]
    x = depthwise_conv(x, 128, strides=(2, 2))
    # [52, 52, 128] => [52, 52, 128]
    x = depthwise_conv(x, 128)
    f2 = x
    # [52, 52, 128] => [26, 26, 256]
    x = depthwise_conv(x, 256, strides=(2, 2))
    # [26, 26, 256] => [26, 26, 256]
    x = depthwise_conv(x, 256)
    f3 = x
    # [26, 26, 256] => [13, 13, 512]
    x = depthwise_conv(x, 512, strides=(2, 2))
    # [13, 13, 512] => [13, 13, 512]
    f4 = x
    x = depthwise_conv(x, 512)
    # [2, 2, 512] => [2, 2, 512]
    x = depthwise_conv(x, 512)
    f5 = x
    return img_input, [f1, f2, f3, f4, f5]





def decoder(feature_map_list, class_number, input_height=512, input_width=256, encoder_level=3):
    
    # get the feature map f4
    feature_map = feature_map_list[encoder_level]

    # decoder

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

    #  (52,52,512) -> (104,104,128)
    #x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2,2),activation='relu', padding='same')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    #104,104,128) -> (208,208,64)
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

    # first part encoder
    img_input, feature_map_list = mobilenet_encoder(input_height=Height, input_width=Width)

    # second decoder
    output = decoder(feature_map_list, class_number=class_number, input_height=Height, input_width=Width,
                     encoder_level=3)

    # build the model
    model = Model(img_input, output)

    #model.summary()


    return model


if __name__ == '__main__':
    main(Height=416, Width=416)
