#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2021/1/3
# @Author : v
# @File : predict.py
# @Software: PyCharm
import mobilenet
import numpy as np
import os, copy
from PIL import Image

import cv2

import random
import numpy as np
np.set_printoptions(threshold=np.inf)
HEIGHT = 416  # 图像的长
WIDHT = 416  # 图像的宽
CLASS_NUMBERS = 32  # 分类数
# class_colors = [[0, 255, 0], [255, 0, 0],[0, 0, 255],[111, 74, 0],[70, 70, 70],[128, 64, 128],[0, 0, 0],[102, 102, 156],[190, 153, 153],[150, 100, 100],[107, 142, 35],
#                  [152, 251, 152],[70, 130, 180],[220, 220, 0],[119, 11, 32],[215, 166, 66],[66, 88, 99],[154, 25, 244],[10, 155, 83],]


#def generateColorGradient(RGB1,RGB2,n):
    # dRGB=[float(x2-x1)/(n-1) for x1,x2 in zip(RGB1,RGB2)]
    # gradient=[tuple([int(x+k*dx) for x, dx in zip(RGB1,dRGB)]) for k in range(n)]
    # return gradient
#class_colors=generateColorGradient((0,0,0),(250,244,233),19)

A=np.random.randint(0,255,size=[32,1,3])
B=A.flatten()

# def list_of_groups(init_list,n):
#     list_of_groups=zip(*(iter(init_list),)*n)
#     end_list=[list(i) for i in list_of_groups] # i is a tuple
#     count = len(init_list)%n
#     end_list.append(init_list[-count:]) if count != 0 else end_list
#     return end_list
#
# class_colors=list_of_groups(B,3)
# print(class_colors)
class_colors = [[51, 137, 191], [176, 26, 245], [104, 166, 230], [53, 4, 201], [233, 126, 160], [134, 13, 225],
[119, 151, 244], [83, 224, 54], [215, 23, 111], [42, 133, 190], [190, 81, 58], [243, 206, 166], [207, 67, 143],
[180, 184, 183], [110, 204, 51], [207, 109, 48], [237, 168, 61], [200, 56, 32], [47, 53, 230], [96, 83, 218], [73, 44, 123],
[11, 75, 46], [177, 11, 233], [204, 220, 112], [250, 104, 45], [116, 21, 109], [20, 164, 30], [2, 71, 82], [113, 90, 229], [84, 189, 204],
[146, 146, 160], [125, 215, 33]]

#print(class_colors)

def get_model():
    """ 加载模型和参数"""
    # 获取模型
    model = mobilenet.main()

    # 加载参数
    # model.load_weights("callbacks/ep044-loss0.030-val_loss0.028.h5")
    model.load_weights('callbacks/mobileaugfinal2021322.h5')
    return model


def precess_img(img):
    """
     对图像进行预处理，使其满足输入数据的要求
     1， 改变大小，2. 数据归一化 3. 改变shape
    :param img: 要预处理的图像
    :return: 返回预处理后的数组
    """

    # 对测试图像进行预处理，以适应模型的输入

    test_img = img.resize((HEIGHT, WIDHT))  # 改变图像大小-> (416,416)
    test_img_array = np.array(test_img)  # 图像变成数组
    test_img_array = test_img_array / 255  # 归一化
    # print(test_img_array.shape)
    # (416,416,30 -> (1,416,416,3)
    test_img_array = test_img_array.reshape(-1, HEIGHT, WIDHT, 3)
    # print(test_img_array.shape)

    return test_img_array


def predicting(model):
    """ 预测"""
    # 获取测试图片
    test_data_path = "datasets/dataset/testData/img"  # 测图片的文件夹
    test_data = os.listdir(test_data_path)  # 读取测试图片
    print(test_data)

    for test_name in test_data:
        # 每个测试图片的具体路径
        test_img_full_path = os.path.join(test_data_path, test_name)
        # print(test_img_full_path)
        test_img = Image.open(test_img_full_path)  # 打开图片
        # test_img.show()
        old_test_img = copy.deepcopy(test_img)  # 复制图像 ， 备份
        test_img_array = np.array(test_img)  # 图片转成数组
        # print(test_img_array.shape)
        original_height = test_img_array.shape[0]  # 获得图像的长 600
        original_width = test_img_array.shape[1]  # 获得图像的宽 800

        test_img_array = precess_img(test_img)  # 图片预处理
        # (1,208*208,2) -> (208*208,2)
        predict_picture = model.predict(test_img_array)[0]  # 预测
        # (208 * 208, 2) -> (208,208,2)
        predict_picture = predict_picture.reshape((int(HEIGHT / 2), int(WIDHT / 2), CLASS_NUMBERS))
        # (208,208,2) -> (208,208)两个图层对应位置比较，保存最大的索引
        # 通道数为2，所以里面保存的是 0或 1
        predict_picture = predict_picture.argmax(axis=-1)  # 可以理解为合并的图层
        c = Image.fromarray(np.uint8(predict_picture))
        save_path = 'Datasets/dataset/testData'
        save_path = os.path.join(save_path, 'predict_mobileaug')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        c.save(save_path + '/' + test_name)
        # print(predict_picture.shape)

        # 下面要做的是对合并的图层，分层
        seg_img_array = np.zeros((int(HEIGHT / 2), int(WIDHT / 2), 3))  # (208,208,3) 值全为0

        colors = class_colors

        # 根据一张合并的图层（predict_picture) 将不同的类赋予不同的颜色
        for cn in range(CLASS_NUMBERS):
            # seg_img 0通道
            seg_img_array[:, :, 0] += ((predict_picture[:, :] == cn) * colors[cn][0]).astype('uint8')
            # seg_img 1 通道
            seg_img_array[:, :, 1] += ((predict_picture[:, :] == cn) * colors[cn][1]).astype('uint8')
            # segm_img 2 通道
            seg_img_array[:, :, 2] + ((predict_picture[:, :] == cn) * colors[cn][2]).astype('uint8')

        # 数组 转换成 图片
        seg_img = Image.fromarray(np.uint8(seg_img_array))  # (208,208,3)
        # seg_img.show()
        # print(seg_img.size)
        # 恢复图像大小 (208,208,3) ->(600,800) 方便和原图叠加
        seg_img = seg_img.resize((original_width, original_height))

        # 合并图像
        print(old_test_img.size)
        print(seg_img.size)
        image = Image.blend(old_test_img, seg_img, 3)

        # 保存图片
        save_path = 'Datasets/dataset/testData'
        save_path = os.path.join(save_path, 'imgout_mobileaug')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        image.save(save_path + '/' + test_name)
        # break


def main():
    """ 模型预测"""
    model = get_model()
    predicting(model)


if __name__ == '__main__':
    main()
