# calculate the color histogram

import cv2.cv2 as cv2
import numpy as np


def calcHistFeature(img, rectList, bins=16):
    channelNum = 3

    featureList = []

    for eachRect in rectList:
        imgRect = img[eachRect[1]:eachRect[1] + eachRect[3], eachRect[0]:eachRect[0] + eachRect[2]]

        histImgList = []

        for i in range(channelNum):
            histImg = cv2.calcHist([imgRect], [i], None,[bins], [0,bins])
            histImgList.append(histImg)
        histImgArr = np.asarray(histImgList).squeeze()
        histImgConcatenate = np.concatenate(histImgArr)

        cv2.normalize(histImgConcatenate, histImgConcatenate)

        featureList.append(histImgConcatenate)

    featureArr = np.asarray(featureList)

    return featureArr


def test():
    testImgPath = 'test.jpg'

    testImg = cv2.imread(testImgPath)
    histImgList = []

    channelNum = 3
    colorList = ('b','g','r')

    bins = 16

    for i in range(channelNum):
        histImg = cv2.calcHist([testImg], [i], None,[bins], [0,bins])
        histImgList.append(histImg)



if __name__ == '__main__':
    #calcHistFeature(cv2.imread('test.jpg'), [[0,0,100,100], [0,20,30,50]])
    test()
