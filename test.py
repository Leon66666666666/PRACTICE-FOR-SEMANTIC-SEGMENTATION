import numpy as np
from PIL import Image
import cv2
import sys
import models
import numpy as np
np.set_printoptions(threshold=np.inf)
import csv

CLASS_NUMBERS = 16
HEIGHT = 416
WIDHT = 416

#with open(r'data.txt') as f:
    #lines = f.readlines()
    #a = ''
    #for line in lines:
        #a+=line.strip()
    #c = a.split()
    #b = ''.join(c)

# g=[[1,2,3],
#    [1,7,3]]
# m=np.sum(g,axis=0)
# print(m)

img_predict = Image.open('Datasets/dataset/testData/predict_mobileaug/0001TP_007440.png')
img_predict = img_predict.resize((960, 720))
img_array_predict = np.array(img_predict)
#img_predict = img_predict[:,:,0]
print(img_array_predict.shape)
C = img_array_predict.flatten()
print(min(C))
print(max(C))

img = Image.open('Datasets/dataset/trainData/label/0001TP_007440_P.png')
#img = img.resize((int(WIDHT / 2), int(HEIGHT / 2)))

img_array = np.array(img)


print(img_array.shape)

#img_array= img_array[:,:,0]
B = img_array.flatten()
#print(B)

f_score=np.zeros((16,1))


class Evaluation(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # return the list which contains the accuracy of each prediction,like [0.9,0.8],represents the accuracies of the class 1 and 2

    def Recall(self):
        Re=np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=0)
        return Re

    def meanPixelAccuracy(self):
        classAcc = self.precision()
        meanAcc = np.nanmean(classAcc)  # determine the mean value, NaN indicates the denominator might zero
        return meanAcc  # np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # extrac the diagonal elements
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1 is the rows of confusion matrix； axis = 0 means the colume of confusion matrix
        IoU = intersection / union  # determine the IoU
        mIoU = np.nanmean(IoU)  
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # like the function in the FCN,score.py,fast_hist()
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        print(confusionMatrix)
        datas=confusionMatrix
        with open("./user_info.csv","w") as f:
            writer=csv.writer(f)
            for row in datas:
                writer.writerow(row)
        return confusionMatrix



    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    imgPredict = C  # predict
    imgLabel = B  # groundTruth
    metric = Evaluation(16)  # number of classification
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    precision = metric.precision()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    recall = metric.Recall()
    for i in range(0,16):
        f_score[i]= (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
    print(f_score)
    print(len(f_score))
    print('pa is : %f' % pa)
    print('cpa is :')  # list
    print(precision)
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % mIoU)
    print('recall is: ')
    print(recall)
