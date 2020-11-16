# calculate the color histogram

import cv2.cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import correlation

def calcCropped(img, rect, cropped_ratio):
    w = rect[2]
    h = rect[3]

    # cropped_center = (w // 2, h // 2)
    cropped_center = ((rect[0] + rect[0] + w) // 2, (rect[1] + rect[1] + h) // 2)
    w_resize = int(w * cropped_ratio[0])
    h_resize = int(h * cropped_ratio[1])

    x1 = cropped_center[0] - w_resize // 2
    x2 = cropped_center[0] + w_resize // 2

    y1 = cropped_center[1] - h_resize // 2
    y2 = cropped_center[1] + h_resize // 2

    # imgRect = img[eachRect[1]:eachRect[3], eachRect[0]:eachRect[2]]
    # imgRect = img[eachRect[1]:eachRect[3], eachRect[0]:eachRect[2]]
    imgRect = img[y1:y2, x1:x2]

    return imgRect, x1, x2, y1, y2

def compareHistFeature(feature1, feature2, metricType = 'euclidean'):
    '''
    metric method:
    euclidean
    correlation

    '''
    if(metricType == 'euclidean'):
        dist = euclidean(feature1, feature2)
    elif(metricType == 'correlation'):
        dist = correlation(feature1, feature2)

    return dist

def compareHistFeature3D(feature1, feature2):
    return cv2.compareHist(feature1, feature2, cv2.HISTCMP_BHATTACHARYYA)


def calcHistFeatureNoRect(img, bins=16):
    channelNum = 3

    featureList = []


    histImgList = []

    for i in range(channelNum):
        histImg = cv2.calcHist([img], [i], None, [bins], [0, bins])
        histImgList.append(histImg)
    histImgArr = np.asarray(histImgList).squeeze()
    histImgConcatenate = np.concatenate(histImgArr)

    cv2.normalize(histImgConcatenate, histImgConcatenate)

    featureList.append(histImgConcatenate)

    featureArr = np.asarray(featureList)

    return featureArr

def calcHistFeature(img, rectList, bins=16, cropped_ratio=(0.8,0.8)):
    '''
        rectList: list of bbox tuple
        bbox: x1, y1, w, h
    '''

    channelNum = 3

    featureList = []

    if(len(rectList) == 0):
        return None

    for eachRect in rectList:
        img_rect, x1, x2, y1, y2 = calcCropped(img, eachRect, cropped_ratio)

        histImgList = []

        for i in range(channelNum):
            histImg = cv2.calcHist([img_rect], [i], None, [bins], [0, 256])
            histImgList.append(histImg)
        histImgArr = np.asarray(histImgList).squeeze()
        histImgConcatenate = np.concatenate(histImgArr)

        cv2.normalize(histImgConcatenate, histImgConcatenate)

        featureList.append(histImgConcatenate)

    featureArr = np.asarray(featureList)

    return featureArr

def calcHistogram3D(img, rectList, cropped_ratio=(0.8,0.8)):
    '''
        rectList: list of bbox tuple
        bbox: x1, y1, x2, y2
    '''
    featureList = []

    for eachRect in rectList:
        imgRect, x1, x2, y1, y2 = calcCropped(img, eachRect, cropped_ratio)

        histImgList = []

        histImg = cv2.calcHist([imgRect], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        cv2.normalize(histImg, histImg)

        histImgConcatenate = np.concatenate(histImg, axis=None)

        featureList.append(histImgConcatenate)

    featureArr = np.asarray(featureList)

    return featureArr

def calcHistogramHSV(img, rectList):
    '''
    rectList: list of bbox tuple
    bbox: x1, y1, x2, y2
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    featureList = []

    for eachRect in rectList:
        #imgRect = img[eachRect[1]:eachRect[1] + eachRect[3], eachRect[0]:eachRect[0] + eachRect[2]]
        imgRect = img[eachRect[1]:eachRect[3], eachRect[0]:eachRect[2]]

        #cv2.imshow('test', imgRect)
        #cv2.waitKey()

        hist_img = cv2.calcHist([imgRect], [0,1], None, [180,256], [0,180,0,256])
        cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        featureList.append(hist_img)

    featureArr = np.asarray(featureList)

    return featureArr

def compareHistFeatureHSV(feature1, feature2):
    return cv2.compareHist(feature1, feature2, cv2.HISTCMP_BHATTACHARYYA)




def test():
    testImgPath = 'tmp/343_1.png'
    testImg2Path = 'tmp/400_1.png'

    testImg = cv2.imread(testImgPath)
    testImg2 = cv2.imread(testImg2Path)

    img1_hsv = cv2.cvtColor(testImg, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(testImg2, cv2.COLOR_BGR2HSV)

    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([img2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)

    print(metric_val)




if __name__ == '__main__':
    #calcHistFeature(cv2.imread('test.jpg'), [[0,0,100,100], [0,20,30,50]])
    calcHistogram3Dcropped(cv2.imread('test.jpg'), [[0,0,100,100]])
    test()