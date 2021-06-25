# calculate the color histogram

import cv2

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import correlation

def calcCropped(img, rect, cropped_ratio=(1.0,1.0)):
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
def calcOrigin(img, rectList, resize_ratio=(64, 64), cropped_ratio=(1.0, 1.0)):
    featureList = []


    for eachRect in rectList:
        imgRect, x1, x2, y1, y2 = calcCropped(img, eachRect, cropped_ratio)

        imgRectResize = cv2.resize(imgRect, resize_ratio)
        imgRectConcatenate = np.concatenate(imgRectResize, axis=None)

        cv2.normalize(imgRectConcatenate, imgRectConcatenate)

        featureList.append(imgRectConcatenate)

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
    test_ImgA_1_Path = 'testSample/out/24_0.png'
    test_ImgA_2_Path = 'testSample/out/10_0.png'

    test_ImgB_1_Path = 'testSample/out/221_1.png'
    test_ImgB_2_Path = 'testSample/out/492_1.png'

    test_ImgA_1 = cv2.imread(test_ImgA_1_Path)
    test_ImgA_2 = cv2.imread(test_ImgA_2_Path)
    test_ImgB_1 = cv2.imread(test_ImgB_1_Path)
    test_ImgB_2 = cv2.imread(test_ImgB_2_Path)



    imgA_1_hsv = cv2.cvtColor(test_ImgA_1, cv2.COLOR_BGR2HSV)
    imgA_2_hsv = cv2.cvtColor(test_ImgA_2, cv2.COLOR_BGR2HSV)

    imgB_1_hsv = cv2.cvtColor(test_ImgB_1, cv2.COLOR_BGR2HSV)
    imgB_2_hsv = cv2.cvtColor(test_ImgB_2, cv2.COLOR_BGR2HSV)


    hist_imgA_1 = cv2.calcHist([imgA_1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_imgA_1, hist_imgA_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_imgA_2 = cv2.calcHist([imgA_2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist_imgA_2, hist_imgA_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_imgB_1 = cv2.calcHist([imgB_1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist_imgB_1, hist_imgB_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_imgB_2 = cv2.calcHist([imgB_2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist_imgB_2, hist_imgB_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    dist_intra_A = cv2.compareHist(hist_imgA_1, hist_imgA_2, cv2.HISTCMP_BHATTACHARYYA)
    dist_intra_B = cv2.compareHist(hist_imgB_1, hist_imgB_2, cv2.HISTCMP_BHATTACHARYYA)
    dist_inter_AB11 = cv2.compareHist(hist_imgA_1, hist_imgB_1, cv2.HISTCMP_BHATTACHARYYA)
    dist_inter_AB12 = cv2.compareHist(hist_imgA_1, hist_imgB_2, cv2.HISTCMP_BHATTACHARYYA)
    dist_inter_AB21 = cv2.compareHist(hist_imgA_2, hist_imgB_1, cv2.HISTCMP_BHATTACHARYYA)
    dist_inter_AB22 = cv2.compareHist(hist_imgA_2, hist_imgB_2, cv2.HISTCMP_BHATTACHARYYA)


    print('dist_intra_A: {}\ndist_intra_B: {}\ndist_inter_AB11: {}\ndist_inter_AB12:{}\ndist_inter_AB21: {}\ndist_inter_AB22: {}\n'.format(dist_intra_A, dist_intra_B, dist_inter_AB11, dist_inter_AB12, dist_inter_AB21, dist_inter_AB22))




if __name__ == '__main__':
    #calcHistFeature(cv2.imread('test.jpg'), [[0,0,100,100], [0,20,30,50]])
    #calcHistogram3Dcropped(cv2.imread('test.jpg'), [[0,0,100,100]])
    test()