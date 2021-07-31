
import numpy as np
import torch.utils.data as Data
from PIL import Image
from collections import  Counter
import sys
import csv
import cv2
img_predict = Image.open('Datasets/dataset/testData/segnet_predict/aachen_000012_000019_leftImg8bit.png')
img_predict = img_predict.resize((2048, 1024))
img_array_predict = np.array(img_predict)
#img_predict = img_predict[:,:,0]
#print(img_predict.size)
C = img_array_predict

# print(min(C))
# print(max(C))
predict = C



img = Image.open('Datasets/dataset/trainData/labelpng/aachen_000012_000019_gtFine_labelTrainIds.png')
#img = img.resize((int(WIDHT / 2), int(HEIGHT / 2)))
img_array = np.array(img)
#print(img.size)
#img_array= img_array[:,:,0]
label = img_array
#print(B)

mask = (label >= 0) & (label < 19)
label_1 = 19*label[mask].astype('int') + predict[mask]
count = np.bincount(label_1,minlength=19*19)
confusion_matrix = count.reshape(19,19)


f_score=np.zeros((19,1))


if __name__ == '__main__':
    pa = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    recall = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    for i in range(0, 19):
        f_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])

    print(pa)
    print(precision)
    print(recall)
    print(f_score)
